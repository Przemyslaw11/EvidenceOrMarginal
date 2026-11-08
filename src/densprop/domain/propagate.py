"""Monte Carlo propagation of a belief q(y) through one HCR neuron, and all metrics.

For samples y_s ~ q (weights w_s for importance-weighted beliefs), split into R contiguous replicate
blocks, the four output densities are
    CONST_DENOM  coefficients  sum_s w A_i / sum_s w A_0             (ratio of expectations, paper Eq. 10)
    EXACT_UNCAL  coefficients  sum_s w A_i/A_0 / sum_s w            (expectation of the ratio)
    EXACT_CAL    on the s-grid  mean_s max(N, eps) / int max(N, eps) dx   (192-node Gauss-Legendre normaliser)
    GOLD         on the s-grid  mean_s p_s(s | y_s)                  (analytic conditional of the copula)
Both coefficient forms use the same samples, so the covariance identity holds in-sample.
All TV, integrals, means and variances are evaluated on a trapezoid grid in normal-score s.
"""

import math

import numpy as np
import torch
from scipy.special import betaln

from densprop.domain.neuron import DT, chunk_for, gauss_legendre, input_basis, legendre

NAN = float("nan")


class Grid:
    """Trapezoid grid in s = Phi^{-1}(x) on [-s_max, s_max], plus the x Gauss-Legendre rule of the normaliser."""

    def __init__(self, s_nodes, s_max, x_gl, m_max, device):
        self.s = torch.linspace(-s_max, s_max, s_nodes, dtype=DT, device=device)
        self.h = h = float(self.s[1] - self.s[0])
        self.w = torch.full_like(self.s, h)
        self.w[0] = self.w[-1] = h / 2
        t = torch.linspace(0, 1, 17, dtype=DT, device=device)[:, None]  # cubic Lagrange weights on nodes -1, 0, 1, 2
        self._L = torch.cat([-t * (t - 1) * (t - 2) / 6, (t + 1) * (t - 1) * (t - 2) / 2, -(t + 1) * t * (t - 2) / 2, (t + 1) * t * (t - 1) / 6], 1)
        self.x = torch.special.ndtr(self.s)
        self.phi = torch.exp(-0.5 * self.s**2) / math.sqrt(2 * math.pi)
        self.F = legendre(self.x, m_max)
        self.xg, self.wg = gauss_legendre(x_gl, device)
        self.Fg = legendre(self.xg, m_max)

    def poly_s(self, coef):
        """Polynomial density sum_i coef_i f_i(x) as an s-density on the grid; coef (..., m+1)."""
        return (coef @ self.F[:, : coef.shape[-1]].T) * self.phi

    def tv(self, p, r):
        """0.5 int |p - r| ds. |.| of the 4-point cubic interpolant of p - r is integrated on each interval with
        16 sub-intervals, so crossings (kinks of |p - r|) cost O((h/16)^2) instead of O(h^2)."""
        d = p - r
        dp = torch.cat([d[..., :1], d, d[..., -1:]], -1)  # edge intervals: replicate the end values
        stack = torch.stack([dp[..., :-3], dp[..., 1:-2], dp[..., 2:-1], dp[..., 3:]], -1)  # (..., G-1, 4)
        v = (stack @ self._L.T).abs()  # (..., G-1, 17)
        return 0.5 * self.h * ((v[..., 1:] + v[..., :-1]).sum(-1) / 32).sum(-1)

    def moments(self, p):
        """Integral, mean and variance in x of (possibly signed, never renormalised) s-densities (..., G)."""
        I = (p * self.w).sum(-1)
        mean = (p * self.x * self.w).sum(-1)
        var = (p * (self.x - mean[..., None]) ** 2 * self.w).sum(-1)
        return I, mean, var


def tv_stats(grid, P, Q):
    """TV of the pooled estimates (last row) with replicate SE and the replicate-difference noise floor."""
    tv = grid.tv(P, Q)
    R = P.shape[0] - 1
    reps = tv[:-1]
    D = (P[:-1] - Q[:-1]) - (P[-1] - Q[-1])
    floor = grid.tv(D, torch.zeros_like(D)).mean() / math.sqrt(R - 1)
    return float(tv[-1]), float(reps.std() / math.sqrt(R)), float(floor)


# ---------------------------------------------------------------- beliefs
def beta_params(name, k, cfg_b):
    ps = cfg_b[name]
    return np.array([ps[i % len(ps)] for i in range(k)], float)  # (k, 2)


def _beta_terms(ab, device):
    """(a - 1, b - 1, -sum log B(a, b)) of a Beta product as tensors; ab: (k, 2)."""
    a, b = ab[:, 0], ab[:, 1]
    t = lambda v: torch.tensor(v, dtype=DT, device=device)
    return t(a - 1), t(b - 1), t(-betaln(a, b).sum())


def _beta_eval(Y, am, bm, c):
    Yc = torch.clamp(Y, 1e-300, 1.0)
    Zc = torch.clamp(1 - Y, 1e-300, 1.0)
    return (am * torch.log(Yc) + bm * torch.log(Zc)).sum(-1) + c


def beta_logq(ab):
    """log q of a Beta product (callable on (..., k), parameters kept as `logq.ab`) and its maximum."""
    a, b = ab[:, 0], ab[:, 1]
    mode = (a - 1) / (a + b - 2)
    on = {}  # parameters per device, copied once

    def logq(Y):
        if Y.device not in on:
            on[Y.device] = _beta_terms(ab, Y.device)
        return _beta_eval(Y, *on[Y.device])

    logq.ab = ab
    lmax = float(((a - 1) * np.log(mode) + (b - 1) * np.log(1 - mode)).sum() - betaln(a, b).sum())
    return logq, lmax


def sample_beliefs(rngs, M, k, cfg_b, names):
    """Belief samples that do not depend on the case (S1 reuses the S2 samples with weights D)."""
    out = {}
    for name in names:
        rng = rngs[name]
        if name == "S2":
            out[name] = dict(Y=rng.random((M, k)), X=rng.random(M))
        elif name.startswith("S3"):
            ab = beta_params(name, k, cfg_b)
            out[name] = dict(Y=rng.beta(ab[:, 0], ab[:, 1], size=(M, k)), ab=ab)
    return out


# ---------------------------------------------------------------- min D
def _starts(k, n, rng, extra):
    pts = []
    if k <= 10:
        corners = ((np.arange(2**k)[:, None] >> np.arange(k)) & 1).astype(float)
    else:
        corners = rng.integers(0, 2, (256, k)).astype(float)
    pts.append(corners)
    n_edge = k * 2 ** (k - 1)
    if n_edge <= 256:
        e = np.repeat(corners, k, 0)
        e[np.arange(len(e)), np.tile(np.arange(k), len(corners))] = 0.5
        edges = np.unique(e, axis=0)
    else:
        edges = rng.integers(0, 2, (256, k)).astype(float)
        edges[np.arange(256), rng.integers(0, k, 256)] = 0.5
    pts.append(edges)
    if extra is not None and len(extra):
        pts.append(extra)
    used = sum(len(p) for p in pts)
    pts.append(rng.random((max(n - used, 0), k)))
    return np.concatenate(pts)


def _restore(Y, logq, thr, lo, hi, iters=60):
    """Project infeasible rows (any leading shape) onto {log q >= thr} by Newton steps along grad log q (the set is
    convex for log-concave q, e.g. Beta products with parameters >= 1). Feasible rows are never moved."""
    for _ in range(iters):
        Yd = Y.detach().requires_grad_(True)
        g = logq(Yd) - thr
        bad = g < 0
        if not bool(bad.any()):
            break
        (gr,) = torch.autograd.grad(g.sum(), Yd)
        step = (-(g - 1e-9) / (gr * gr).sum(-1).clamp(min=1e-300))[..., None] * gr
        with torch.no_grad():
            Y[bad] = torch.clamp(Y[bad] + step[bad], lo, hi)


class DStack:
    """D(y) of G neurons (same k and basis, degrees <= m) evaluated together: row 0 of each model's blocks,
    zero-padded to degree m, so that one batched product serves all G models. Y: (G, n, k) -> (G, n)."""

    def __init__(self, models):
        k, m = models[0].k, max(md.m for md in models)
        assert all(md.k == k and md.basis == models[0].basis for md in models)
        self.k, self.m = k, m
        pa, pb = models[0].pa, models[0].pb

        def pad(t, dims):  # zero-pad the trailing degree axes `dims` of t to length m
            p = [0, 0] * t.dim()
            for d in dims:
                p[2 * (t.dim() - 1 - d) + 1] = m - t.shape[d]
            return torch.nn.functional.pad(t, p)

        b = [md.blocks for md in models]
        self.c0 = torch.stack([x["c"][0] for x in b])
        self.w1 = torch.stack([pad(x["W1"][0], (1,)).reshape(-1) for x in b])
        self.pa, self.pb = pa, pb
        self.w2 = torch.stack([pad(x["W2"][0], (1, 2)).reshape(-1) for x in b]) if b[0]["W2"] is not None else None
        self.w3 = torch.stack([pad(x["W3"][0], (1, 2, 4)).reshape(len(pa) * m * m, k * m) for x in b]) if b[0]["W3"] is not None else None

    def __call__(self, Y):
        G, n, _ = Y.shape
        Fy = input_basis(Y, self.m)
        Ff = Fy.reshape(G, n, -1)
        D = self.c0[:, None] + torch.bmm(Ff, self.w1[:, :, None])[..., 0]
        if self.w2 is not None:
            Pf = (Fy[:, :, self.pa, :, None] * Fy[:, :, self.pb, None, :]).reshape(G, n, -1)
            D = D + torch.bmm(Pf, self.w2[:, :, None])[..., 0]
            if self.w3 is not None:
                D = D + (torch.bmm(Pf, self.w3) * Ff).sum(-1)
        return D


def min_D_batch(models, rng, cfg_md, extras=None, device="cpu", supp=None, group=None):
    """Multi-start projected Adam on D(y) for several neurons at once: over [0,1]^k, or (supp = [(ab, thr)] per model)
    over supp q = {log q >= thr} of a Beta product q, by projection (`_restore`) after every step. One stacked
    problem per group of `group` models; Adam and the projection act row by row, so this equals separate runs.
    Starts are drawn from `rng` model by model.

    Returns per model (best feasible value, number of starts within 1e-6 of it, number of feasible end points);
    the value is an upper bound on the true minimum (NaN if no end point is feasible)."""
    extras = extras if extras is not None else [None] * len(models)
    starts = [_starts(md.k, cfg_md["starts"], rng, ex) for md, ex in zip(models, extras)]
    if group is None:  # keep the stacked pair-feature matrix of one group below ~3e8 elements (2.4 GB)
        P = models[0].k * (models[0].k - 1) // 2
        nf = max(P, 1) * max(md.m for md in models) ** 2
        group = max(1, int(3e8 // (max(len(s) for s in starts) * nf)))
    lo, hi = (0.0, 1.0) if supp is None else (1e-12, 1 - 1e-12)
    out = []
    for g0 in range(0, len(models), group):
        ms, ss = models[g0 : g0 + group], starts[g0 : g0 + group]
        n = max(len(s) for s in ss)
        # equal row counts per model: pad with copies of the first start, excluded from the result
        Y = torch.tensor(np.stack([np.concatenate([s, np.repeat(s[:1], n - len(s), 0)]) for s in ss]), dtype=DT, device=device)
        valid = torch.tensor(np.stack([np.arange(n) < len(s) for s in ss]), device=device)
        f = DStack(ms)
        logq = thr = None
        if supp is not None:
            terms = [_beta_terms(ab, device) for ab, _ in supp[g0 : g0 + group]]
            am, bm = (torch.stack([t[i] for t in terms])[:, None] for i in (0, 1))
            c = torch.stack([t[2] for t in terms])[:, None]
            thr = torch.tensor([th for _, th in supp[g0 : g0 + group]], dtype=DT, device=device)[:, None]
            logq = lambda Z: _beta_eval(Z, am, bm, c)
        with torch.no_grad():
            Y.clamp_(lo, hi)
        if logq is not None:
            _restore(Y, logq, thr, lo, hi)
        Y.requires_grad_(True)
        opt = torch.optim.Adam([Y], lr=cfg_md["lr"][0])
        steps = cfg_md["steps"]
        l0, l1 = cfg_md["lr"]
        for t in range(steps):
            for gr in opt.param_groups:
                gr["lr"] = l1 + 0.5 * (l0 - l1) * (1 + math.cos(math.pi * t / steps))
            loss = f(Y).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                Y.clamp_(lo, hi)
            if logq is not None:
                _restore(Y.data, logq, thr, lo, hi)
        with torch.no_grad():
            feas = valid if logq is None else valid & (logq(Y) >= thr)
            D = torch.where(feas, f(Y), math.inf)
            best = D.min(1).values
            natt = (D <= best[:, None] + 1e-6).sum(1)
        for b, a, v in zip(best.cpu(), natt.cpu(), feas.sum(1).cpu()):
            out.append((float(b), int(a), int(v)) if int(v) else (NAN, 0, 0))
    return out


def min_D(model, rng, cfg_md, extra=None, logq=None, thr=None, device="cpu"):
    """`min_D_batch` for one neuron: over [0,1]^k, or over supp q = {log q >= thr} for a Beta product q
    (`logq` from `beta_logq`)."""
    supp = None if logq is None else [(logq.ab, thr)]
    return min_D_batch([model], rng, cfg_md, [extra], device, supp)[0]


# ---------------------------------------------------------------- one case
class _Acc:
    """Per-replicate sums for one (belief, degree)."""

    def __init__(self, R, m, neps, G, Ng, dev):
        z = lambda *s: torch.zeros(*s, dtype=DT, device=dev)
        self.SW, self.SW2, self.SWD, self.SWD2 = z(R), z(R), z(R), z(R)
        self.SWA, self.SWQ, self.SWV = z(R, m + 1), z(R, m + 1), z(R)
        self.CAL, self.GOLD = z(R, neps, G), z(R, G)
        self.CALg = z(R, neps, Ng)
        # counters stay on the device during the MC pass (no host sync per chunk); `host()` converts them once
        self.n_bad, self.n_drop, self.n_Dneg, self.n_Nneg = (torch.zeros((), dtype=torch.long, device=dev) for _ in range(4))
        self.min_mc = torch.full((), math.inf, dtype=DT, device=dev)
        self.low_D = None
        self.low_Y = None

    def host(self):
        self.n_bad, self.n_drop, self.n_Dneg, self.n_Nneg = (int(v) for v in (self.n_bad, self.n_drop, self.n_Dneg, self.n_Nneg))
        self.min_mc = float(self.min_mc)


def evaluate_case(model6, cop, beliefs, grid, cfg, rng_md, degrees, device, dens=None):
    """All rows (degree x belief x eps) for one neuron (model6 of degree m_max) and one ground-truth copula.

    beliefs: {name: dict(Y=tensor (M, k), X=tensor (M,) for S2, logq, thr)}; 'S1' uses the S2 samples
    with weights D. If `dens` is a dict, pooled densities and coefficients are stored in it per (belief, m).
    A job with several cases runs `mc_pass` per case, one `solve_mind` for all and `case_rows` per case."""
    p = mc_pass(model6, cop, beliefs, grid, cfg, degrees, device)
    solve_mind([p], cfg, rng_md, device)
    return case_rows(p, grid, cfg, dens)


def mc_pass(model6, cop, beliefs, grid, cfg, degrees, device):
    """Monte Carlo pass of one case: per-replicate sums for every (belief, degree); see `evaluate_case`."""
    num = cfg["numerics"]
    R, eps_list, floor = num["R"], cfg["eps"], cfg["D_floor"]
    models = {m: model6.sliced(m) for m in degrees}
    G = grid.s.shape[0]
    stats = {}
    names = [n for n in cfg["order"] if n in beliefs or (n == "S1" and "S2" in beliefs)]
    for name in names:
        src = beliefs["S2" if name == "S1" else name]
        Y = src["Y"]
        M = Y.shape[0]
        block = M // R
        # largest power of two <= chunk_mc that keeps one pair-feature matrix below 4e8 elements
        chunk = min(2 ** int(math.log2(chunk_for(model6.k, model6.m, model6.basis, num["chunk_mc"], 4.0e8))), block)
        assert block * R == M and block % chunk == 0, "M must split into R blocks of whole chunks"
        logq, thr = src.get("logq"), src.get("thr")
        acc = {m: _Acc(R, m, len(eps_list), G, grid.xg.shape[0], device) for m in degrees}
        for r in range(R):
            for st in range(r * block, (r + 1) * block, chunk):
                Yc = Y[st : st + chunk]
                Fy = input_basis(Yc, model6.m)
                gold_c = cop.cond_s(grid.s, Yc)
                insupp = (logq(Yc) >= thr) if logq is not None else torch.ones(Yc.shape[0], dtype=torch.bool, device=device)
                for m in degrees:
                    a = acc[m]
                    A = models[m].A(Fy)
                    A0 = A[:, 0]
                    w = A0 if name == "S1" else torch.ones_like(A0)
                    a.SW[r] += w.sum()
                    a.SW2[r] += (w * w).sum()
                    a.SWD[r] += (w * A0).sum()
                    a.SWD2[r] += (w * A0 * A0).sum()
                    a.SWA[r] += w @ A
                    ok = A0 >= floor
                    wv = torch.where(ok, w, torch.zeros_like(w))
                    a.SWQ[r] += wv @ (A / torch.where(ok, A0, torch.ones_like(A0))[:, None])
                    a.SWV[r] += wv.sum()
                    a.n_bad += (~ok & insupp).sum()
                    a.n_drop += (~ok & ~insupp).sum()
                    a.min_mc = torch.minimum(a.min_mc, torch.where(insupp, A0, math.inf).min())
                    a.GOLD[r] += w @ gold_c
                    Ns = A @ grid.F[:, : m + 1].T
                    Ng = A @ grid.Fg[:, : m + 1].T
                    for e, eps in enumerate(eps_list):
                        Ngc = torch.clamp(Ng, min=eps)
                        Z = Ngc @ grid.wg
                        a.CAL[r, e] += (w / Z) @ torch.clamp(Ns, min=eps)
                        a.CALg[r, e] += (w / Z) @ Ngc
                    if name == "S2":
                        a.n_Dneg += (A0 < 0).sum()
                        Fx = legendre(src["X"][st : st + chunk], m)
                        a.n_Nneg += ((Fx * A).sum(1) < 0).sum()
                    if name != "S1":  # lowest-D samples in supp q seed the optimiser
                        Dm = torch.where(insupp, A0, torch.full_like(A0, math.inf))
                        K = min(num["mind"]["mc_seeds"], Dm.shape[0])
                        v, idx = torch.topk(Dm, K, largest=False)
                        cand_D = v if a.low_D is None else torch.cat([a.low_D, v])
                        cand_Y = Yc[idx] if a.low_Y is None else torch.cat([a.low_Y, Yc[idx]])
                        v2, i2 = torch.topk(cand_D, min(num["mind"]["mc_seeds"], cand_D.shape[0]), largest=False)
                        a.low_D, a.low_Y = v2, cand_Y[i2]
        for a in acc.values():
            a.host()
        stats[name] = acc

    return dict(
        models=models,
        names=names,
        stats=stats,
        degrees=degrees,
        mind={},
        M={n: int(beliefs["S2" if n == "S1" else n]["Y"].shape[0]) for n in names},
        supp={n: (beliefs[n]["logq"].ab, beliefs[n]["thr"]) for n in names if n.startswith("S3")},
    )


def solve_mind(passes, cfg, rng_md, device):
    """min D of every case in `passes` (from `mc_pass`): over the cube (S1, S2, S4, negativity) and over supp q for
    the Beta beliefs; all cube searches of all cases and degrees as one stacked problem, then all constrained ones."""
    md, floor = cfg["numerics"]["mind"], cfg["D_floor"]
    finite = lambda a: a.low_Y[torch.isfinite(a.low_D)]
    cube_models, extras = [], []
    for p in passes:
        for m in p["degrees"]:
            ex = [finite(p["stats"][n][m]) for n in p["names"] if n in ("S2", "S4")]
            cube_models.append(p["models"][m])
            extras.append(torch.cat(ex).cpu().numpy() if ex else None)
    cubes = iter(min_D_batch(cube_models, rng_md, md, extras, device))
    tasks = []
    for p in passes:
        stats, mind = p["stats"], p["mind"]
        for m in p["degrees"]:
            cube = next(cubes)
            mc_cube = min([stats[n][m].min_mc for n in p["names"] if n in ("S2", "S4")], default=math.inf)
            mind[(m, "cube")] = (min(cube[0], mc_cube), cube[0], mc_cube, cube[1], cube[2])
            for n in p["names"]:
                if n.startswith("S3"):
                    a = stats[n][m]
                    # the constrained search can only change the decision if the cube minimum is below the floor and
                    # no sample in supp(q) is; otherwise the MC value is reported (method 'mc')
                    if mind[(m, "cube")][0] >= floor or a.min_mc < floor:
                        mind[(m, n)] = (a.min_mc, NAN, a.min_mc, 0, 0, "mc")
                    else:
                        tasks.append((p, m, n))
    if tasks:
        res_all = min_D_batch(
            [p["models"][m] for p, m, _ in tasks],
            rng_md,
            md,
            [finite(p["stats"][n][m]).cpu().numpy() for p, m, n in tasks],
            device,
            [p["supp"][n] for p, _, n in tasks],
        )
        for (p, m, n), res in zip(tasks, res_all):
            a = p["stats"][n][m]
            best = min(res[0], a.min_mc) if not math.isnan(res[0]) else a.min_mc
            p["mind"][(m, n)] = (best, res[0], a.min_mc, res[1], res[2], "opt")


def case_rows(p, grid, cfg, dens=None):
    """Densities and metrics of one case from its `mc_pass` and `solve_mind` results; see `evaluate_case`."""
    R, eps_list, floor = cfg["numerics"]["R"], cfg["eps"], cfg["D_floor"]
    models, names, stats, degrees, mind = p["models"], p["names"], p["stats"], p["degrees"], p["mind"]
    rows = []
    # ---- densities and metrics
    for name in names:
        for m in degrees:
            a = stats[name][m]
            mc = mind[(m, "cube")]
            ms = mind[(m, name)] if name.startswith("S3") else mc
            q_valid = name != "S1" or mc[0] >= 0
            defined = bool(q_valid and ms[0] >= floor and a.n_bad == 0)
            SW = torch.cat([a.SW, a.SW.sum()[None]])
            SWA = torch.cat([a.SWA, a.SWA.sum(0)[None]])
            SWQ = torch.cat([a.SWQ, a.SWQ.sum(0)[None]])
            SWV = torch.cat([a.SWV, a.SWV.sum()[None]])
            cd_c = SWA / SWA[:, :1]
            eu_c = SWQ / SWV[:, None]
            CD, EU = grid.poly_s(cd_c), grid.poly_s(eu_c)
            CAL = torch.cat([a.CAL, a.CAL.sum(0)[None]]) / SW[:, None, None] * grid.phi  # x-density -> s-density
            GOLD = torch.cat([a.GOLD, a.GOLD.sum(0)[None]]) / SW[:, None]
            UNIF = grid.phi.expand_as(GOLD)
            mom = {nm: grid.moments(P) for nm, P in (("cd", CD), ("eu", EU), ("gold", GOLD))}
            Wt, W2t = float(SW[-1]), float(a.SW2.sum())
            ess = Wt * Wt / W2t if W2t > 0 else NAN
            EqD = float(a.SWD.sum() / SW[-1])
            sdD = math.sqrt(max(float(a.SWD2.sum() / SW[-1]) - EqD**2, 0.0))
            base = dict(
                m=m,
                belief=name,
                n_coef=models[m].n_coef,
                chi_D=models[m].chi_D(),
                minD=mc[0],
                minD_opt=mc[1],
                minD_mc=mc[2],
                minD_natt=mc[3],
                minD_nfeas=mc[4],
                minD_supp=ms[0],
                minD_supp_opt=ms[1],
                minD_supp_mc=ms[2],
                minD_supp_method=ms[5] if len(ms) > 5 else "cube",
                P_Dneg_u=stats["S2"][m].n_Dneg / p["M"]["S2"] if "S2" in stats else NAN,
                frac_Nneg=stats["S2"][m].n_Nneg / p["M"]["S2"] if "S2" in stats else NAN,
                q_valid=int(q_valid),
                defined=int(defined),
                n_bad=a.n_bad,
                n_drop=a.n_drop,
                M=p["M"][name],
                ess=ess,
                E_q_D=EqD,
                sd_q_D=sdD,
            )
            base["mc_limited"] = int(ess / base["M"] < cfg["mc_limited_ess"])
            if dens is not None:
                CALg = torch.cat([a.CALg, a.CALg.sum(0)[None]]) / SW[:, None, None]
                dens[(name, m)] = dict(
                    cd_c=cd_c[-1],
                    eu_c=eu_c[-1],
                    CD=CD[-1],
                    EU=EU[-1],
                    CAL=CAL[-1],
                    GOLD=GOLD[-1],
                    CALg=CALg[-1],
                    defined=defined,
                    q_valid=q_valid,
                    n_drop=a.n_drop,
                    all=dict(cd_c=cd_c, eu_c=eu_c, CAL=CAL, CALg=CALg, GOLD=GOLD),
                )  # rows: R replicates, pooled
            if not q_valid:
                for eps in eps_list:
                    rows.append({**base, "eps": eps})
                continue
            vals = {}

            def put(key, P, Q, ok=True):
                if ok:
                    v, se, fl = tv_stats(grid, P, Q)
                    vals.update({key: v, key + "_se": se, key + "_floor": fl, key + "_below": int(v < 3 * fl)})
                else:
                    vals.update({key: NAN, key + "_se": NAN, key + "_floor": NAN, key + "_below": NAN})

            put("d_denom", CD, EU, defined)
            put("d_model_uncal", EU, GOLD, defined)
            put("d_total", CD, GOLD)
            put("d_gold_unif", GOLD, UNIF)
            for key, (I, mu, var) in mom.items():
                ok = defined or key != "eu"
                vals[f"int_{key}"] = float(I[-1]) if ok else NAN
                vals[f"mean_{key}"] = float(mu[-1]) if ok else NAN
                vals[f"var_{key}"] = float(var[-1]) if ok else NAN
            vals["neg_mass_cd"] = float((torch.clamp(-CD[-1], min=0) * grid.w).sum())
            vals["neg_mass_eu"] = float((torch.clamp(-EU[-1], min=0) * grid.w).sum()) if defined else NAN
            for ref in ("eu", "gold"):
                ok = defined or ref != "eu"
                rv = (mom["cd"][2] - mom[ref][2]) / mom[ref][2]
                dm = (mom["cd"][1] - mom[ref][1]).abs()
                tag = "uncal" if ref == "eu" else "gold"
                vals[f"dmean_vs_{tag}"] = float(dm[-1]) if ok else NAN
                vals[f"smean_vs_{tag}"] = float(dm[-1] / torch.sqrt(mom[ref][2][-1])) if ok else NAN
                vals[f"relvar_vs_{tag}"] = float(rv[-1]) if ok else NAN
                vals[f"relvar_vs_{tag}_se"] = float(rv[:-1].std() / math.sqrt(R)) if ok else NAN
            for e, eps in enumerate(eps_list):
                Ce = CAL[:, e]
                row = {**base, **vals, "eps": eps}
                cv = {}
                for key, P, Q, ok in (("d_neg", EU, Ce, defined), ("d_model_cal", Ce, GOLD, True), ("d_dc", CD, Ce, True)):
                    if ok:
                        v, se, fl = tv_stats(grid, P, Q)
                        cv.update({key: v, key + "_se": se, key + "_floor": fl, key + "_below": int(v < 3 * fl)})
                    else:
                        cv.update({key: NAN, key + "_se": NAN, key + "_floor": NAN, key + "_below": NAN})
                I, mu, var = grid.moments(Ce[-1])
                cv.update(int_cal=float(I), mean_cal=float(mu), var_cal=float(var))
                rows.append({**row, **cv})
    return rows
