"""Stage 4 training of the student chain.

Losses (all on fresh teacher samples; neuron l sees its true inputs for the local losses):
    val  sum_l mean (1/2 + A_1/(sqrt(12) A_0) - x_l)^2                       value loss, paper Eq. 7; rows i in {0, 1}
    nll  sum_l -mean log( max(N(x_l | inputs), eps) / Z(inputs) )            calibrated conditional likelihood; all rows
    e2e  sum_b sum_{i>=1} (c_L,i(b) - t_i(b))^2                               CONST_DENOM output moments vs target moments
         t_i(b) = mean f_i(h_L) of teacher samples under belief b, i.e. the squared L2 distance between the output
         polynomial and the target density projected on the basis (kernel-free); calibration-free forward pass.
a_0 = 1 is never trained. For `val`, d loss / d a_(i>=2, .) is identically zero, so the all-rows variant equals it.
"""

import math
import time

import numpy as np
import torch

from densprop.domain.chain import cd_layer
from densprop.domain.neuron import DT, Model, gauss_legendre, input_basis, legendre

SQ12 = math.sqrt(12.0)


class Params:
    """Trainable copies of the blocks of every neuron; entries outside `rows` stay at their initial values."""

    def __init__(self, models, rows="all"):
        self.meta = [(m.k, m.m, m.basis) for m in models]
        self.init = [{n: v.detach().clone() for n, v in m.blocks.items() if v is not None} for m in models]
        self.P = [{n: v.clone().requires_grad_(True) for n, v in d.items()} for d in self.init]
        self.tmask = []
        for d in self.init:
            tm = {}
            for n, v in d.items():
                rows_ok = torch.ones(v.shape[0], dtype=torch.bool, device=v.device)
                if rows == "val":
                    rows_ok[2:] = False
                if n == "c":
                    rows_ok[0] = False  # a_0 = 1
                tm[n] = rows_ok.reshape((-1,) + (1,) * (v.dim() - 1)).expand_as(v)
            self.tmask.append(tm)

    def tensors(self):
        return [t for d in self.P for t in d.values()]

    def models(self, upto=None):
        out = []
        for (k, m, basis), P, I, T in zip(self.meta[:upto], self.P, self.init, self.tmask):
            b = {n: torch.where(T[n], P[n], I[n]) for n in P}
            for n in ("W2", "W3"):
                b.setdefault(n, None)
            out.append(Model(k, m, basis, b))
        return out

    def snapshot(self):
        return [{n: v.detach().clone() for n, v in d.items()} for d in self.P]

    def load(self, snap):
        with torch.no_grad():
            for d, s in zip(self.P, snap):
                for n in d:
                    d[n].copy_(s[n])

    def distance(self):
        """L2 distance from the initial coefficients, all and the D-coefficients a_(0, j), j != 0:
        (relative all, absolute D, relative D or NaN if the initial D is flat)."""
        num = den = num0 = den0 = 0.0
        for P, I in zip(self.P, self.init):
            for n in P:
                d = (P[n] - I[n]).detach()
                num += float((d**2).sum())
                den += float((I[n] ** 2).sum())
                num0 += float((d[0] ** 2).sum())
                den0 += float((I[n][0] ** 2).sum()) - (1.0 if n == "c" else 0.0)
        return math.sqrt(num / den), math.sqrt(num0), math.sqrt(num0 / den0) if den0 > 1e-24 else float("nan")


class Data:
    """Basis tables of the neuron inputs for one split of teacher samples."""

    def __init__(self, samples, L, m, device):
        from densprop.domain.chain import neuron_inputs

        self.x, self.F, self.Fx = [], [], []
        for l in range(1, L + 1):
            x, Y = neuron_inputs(samples, l)
            x = torch.as_tensor(x, dtype=DT, device=device)
            self.x.append(x)
            self.Fx.append(legendre(x, m))
            self.F.append(input_basis(torch.as_tensor(np.asarray(Y), dtype=DT, device=device), m))


def loss_val(models, data, stats):
    tot = 0.0
    for l, mod in enumerate(models):
        A = mod.A(data.F[l])
        stats["cross"] = max(stats.get("cross", 0), int((A[:, 0].abs() < 1e-12).sum()))
        tot = tot + ((0.5 + A[:, 1] / (SQ12 * A[:, 0]) - data.x[l]) ** 2).mean()
    return tot


def make_loss_nll(eps, device):
    xg, wg = gauss_legendre(192, device)

    def loss(models, data, stats):
        tot = 0.0
        for l, mod in enumerate(models):
            A = mod.A(data.F[l])
            stats["cross"] = max(stats.get("cross", 0), int((A[:, 0].abs() < 1e-12).sum()))
            Z = torch.clamp(A @ legendre(xg, mod.m).T, min=eps) @ wg
            N = (data.Fx[l] * A).sum(1)
            tot = tot - torch.log(torch.clamp(N, min=eps) / Z).mean()
        return tot

    return loss


def cd_chain(models, Fy):
    A = models[0].A(Fy)
    c = A.mean(0)
    c = c / c[0]
    for mod in models[1:]:
        c = cd_layer(mod, c)
    return c


def make_loss_e2e(L, bel_F, targets):
    """bel_F: {belief: input basis tables of fixed layer-1 samples}; targets: {belief: (m+1,) target coefficients}."""

    def loss(models, data, stats):
        tot = 0.0
        for b, Fy in bel_F.items():
            c = cd_chain(models[:L], Fy)
            tot = tot + ((c[1:] - targets[b][1:]) ** 2).sum()
        return tot

    return loss


def train(params, loss_fn, splits, opt, ocfg, upto=None):
    """Optimise params on splits['train'] with early stopping on splits['val']; keeps the best validation state.

    Returns a record with losses before/after, the stopping status and the counted irregularities."""
    stats = {}
    t0 = time.time()
    ev = lambda split: float(loss_fn(params.models(upto), splits[split], stats))
    with torch.no_grad():
        before = {s: ev(s) for s in splits}
    best, best_snap, bad, n_nonfinite, status, it = before["val"], params.snapshot(), 0, 0, "max_steps", 0
    if opt == "adam":
        o = torch.optim.Adam(params.tensors(), lr=ocfg["lr"])
        steps = ocfg["steps"]
        for it in range(1, steps + 1):
            for g in o.param_groups:
                g["lr"] = ocfg["lr"] * 0.5 * (1 + math.cos(math.pi * (it - 1) / steps))
            loss = loss_fn(params.models(upto), splits["train"], stats)
            if not torch.isfinite(loss):
                n_nonfinite += 1
                status = "nonfinite"
                break
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()
            if it % ocfg["eval_every"] == 0:
                with torch.no_grad():
                    v = ev("val")
                if math.isfinite(v) and v < best:
                    best, best_snap, bad = v, params.snapshot(), 0
                else:
                    bad += 1
                    if bad >= ocfg["patience"]:
                        status = "early_stop"
                        break
    else:
        o = torch.optim.LBFGS(params.tensors(), lr=ocfg["lr"], max_iter=ocfg["max_iter"], history_size=ocfg["history"], line_search_fn="strong_wolfe")
        state = {"nf": 0}

        def closure():
            o.zero_grad(set_to_none=True)
            loss = loss_fn(params.models(upto), splits["train"], stats)
            if not torch.isfinite(loss):
                state["nf"] += 1
                return loss
            loss.backward()
            return loss

        for _ in range(ocfg["outer"]):
            try:
                o.step(closure)
            except RuntimeError as e:  # recorded, not patched
                status = f"failed: {str(e)[:80]}"
                break
            n_nonfinite = state["nf"]
            if any(not bool(torch.isfinite(t).all()) for t in params.tensors()):
                status = "nonfinite_params"
                break
            with torch.no_grad():
                v = ev("val")
            if math.isfinite(v) and v < best - 1e-12:
                best, best_snap, bad = v, params.snapshot(), 0
            else:
                bad += 1
                if bad >= ocfg["patience"]:
                    status = "early_stop"
                    break
    params.load(best_snap)
    with torch.no_grad():
        after = {s: ev(s) for s in splits}
    dist_all, dist_D_abs, dist_D = params.distance()
    return dict(
        status=status,
        iters=it,
        n_nonfinite=n_nonfinite,
        n_cross=stats.get("cross", 0),
        **{f"loss_{s}_before": v for s, v in before.items()},
        **{f"loss_{s}_after": v for s, v in after.items()},
        coef_dist=dist_all,
        coef_dist_D=dist_D,
        coef_dist_D_abs=dist_D_abs,
        train_s=time.time() - t0,
    )
