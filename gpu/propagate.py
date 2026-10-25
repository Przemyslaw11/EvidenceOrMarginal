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

from .core import DT, gauss_legendre, input_basis, legendre

NAN = float("nan")


class Grid:
    """Trapezoid grid in s = Phi^{-1}(x) on [-s_max, s_max], plus the x Gauss-Legendre rule of the normaliser."""

    def __init__(self, s_nodes, s_max, x_gl, m_max, device):
        self.s = torch.linspace(-s_max, s_max, s_nodes, dtype=DT, device=device)
        h = float(self.s[1] - self.s[0])
        self.w = torch.full_like(self.s, h)
        self.w[0] = self.w[-1] = h / 2
        self.x = torch.special.ndtr(self.s)
        self.phi = torch.exp(-0.5 * self.s ** 2) / math.sqrt(2 * math.pi)
        self.F = legendre(self.x, m_max)
        self.xg, self.wg = gauss_legendre(x_gl, device)
        self.Fg = legendre(self.xg, m_max)

    def poly_s(self, coef):
        """Polynomial density sum_i coef_i f_i(x) as an s-density on the grid; coef (..., m+1)."""
        return (coef @ self.F[:, : coef.shape[-1]].T) * self.phi

    def tv(self, p, r):
        return 0.5 * (torch.abs(p - r) * self.w).sum(-1)

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


def beta_logq(ab):
    a, b = ab[:, 0], ab[:, 1]
    mode = (a - 1) / (a + b - 2)
    const = torch.tensor(-betaln(a, b).sum())
    a_t, b_t = torch.tensor(a), torch.tensor(b)

    def logq(Y):
        Yc = torch.clamp(Y, 1e-300, 1.0)
        Zc = torch.clamp(1 - Y, 1e-300, 1.0)
        return ((a_t.to(Y.device) - 1) * torch.log(Yc) + (b_t.to(Y.device) - 1) * torch.log(Zc)).sum(-1) + const.to(Y.device)

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
        corners = ((np.arange(2 ** k)[:, None] >> np.arange(k)) & 1).astype(float)
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


def min_D(model, rng, cfg_md, extra=None, logq=None, thr=None, device="cpu"):
    """Multi-start projected Adam on D(y) over [0,1]^k (or over supp q = {log q >= thr}, penalty method).

    Returns (best feasible value, number of starts within 1e-6 of it, number of feasible end points).
    The value is an upper bound on the true minimum."""
    k = model.k
    Y0 = _starts(k, cfg_md["starts"], rng, extra)
    Y = torch.tensor(Y0, dtype=DT, device=device).requires_grad_(True)
    lo, hi = (1e-9, 1 - 1e-9) if logq is not None else (0.0, 1.0)
    with torch.no_grad():
        Y.clamp_(lo, hi)
    opt = torch.optim.Adam([Y], lr=cfg_md["lr"][0])
    steps = cfg_md["steps"]
    (l0, l1), (p0, p1) = cfg_md["lr"], cfg_md["penalty"]
    for t in range(steps):
        for g in opt.param_groups:
            g["lr"] = l1 + 0.5 * (l0 - l1) * (1 + math.cos(math.pi * t / steps))
        loss = model.D(Y).sum()
        if logq is not None:
            lam = p0 * (p1 / p0) ** (t / max(steps - 1, 1))
            loss = loss + lam * (torch.relu(thr - logq(Y)) ** 2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            Y.clamp_(lo, hi)
    with torch.no_grad():
        D = model.D(Y)
        feas = torch.ones_like(D, dtype=torch.bool) if logq is None else logq(Y) >= thr
        if not bool(feas.any()):
            return NAN, 0, 0
        best = D[feas].min()
        return float(best), int((D[feas] <= best + 1e-6).sum()), int(feas.sum())


# ---------------------------------------------------------------- one case
class _Acc:
    """Per-replicate sums for one (belief, degree)."""

    def __init__(self, R, m, neps, G, Ng, dev):
        z = lambda *s: torch.zeros(*s, dtype=DT, device=dev)
        self.SW, self.SW2, self.SWD, self.SWD2 = z(R), z(R), z(R), z(R)
        self.SWA, self.SWQ, self.SWV = z(R, m + 1), z(R, m + 1), z(R)
        self.CAL, self.GOLD = z(R, neps, G), z(R, G)
        self.CALg = z(R, neps, Ng)
        self.n_bad = self.n_drop = 0
        self.min_mc = math.inf
        self.low_D = None
        self.low_Y = None
        self.n_Dneg = self.n_Nneg = 0


def evaluate_case(model6, cop, beliefs, grid, cfg, rng_md, degrees, device, dens=None):
    """All rows (degree x belief x eps) for one neuron (model6 of degree m_max) and one ground-truth copula.

    beliefs: {name: dict(Y=tensor (M, k), X=tensor (M,) for S2, logq, thr)}; 'S1' uses the S2 samples
    with weights D. If `dens` is a dict, pooled densities and coefficients are stored in it per (belief, m)."""
    num = cfg["numerics"]
    R, eps_list, floor = num["R"], cfg["eps"], cfg["D_floor"]
    models = {m: model6.sliced(m) for m in degrees}
    G = grid.s.shape[0]
    rows = []
    stats = {}
    names = [n for n in cfg["order"] if n in beliefs or (n == "S1" and "S2" in beliefs)]
    for name in names:
        src = beliefs["S2" if name == "S1" else name]
        Y = src["Y"]
        M = Y.shape[0]
        block = M // R
        chunk = min(num["chunk_mc"], block)
        assert block * R == M and block % chunk == 0, "M must split into R blocks of whole chunks"
        logq, thr = src.get("logq"), src.get("thr")
        acc = {m: _Acc(R, m, len(eps_list), G, grid.xg.shape[0], device) for m in degrees}
        for r in range(R):
            for st in range(r * block, (r + 1) * block, chunk):
                Yc = Y[st: st + chunk]
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
                    a.n_bad += int((~ok & insupp).sum())
                    a.n_drop += int((~ok & ~insupp).sum())
                    if bool(insupp.any()):
                        a.min_mc = min(a.min_mc, float(A0[insupp].min()))
                    a.GOLD[r] += w @ gold_c
                    Ns = A @ grid.F[:, : m + 1].T
                    Ng = A @ grid.Fg[:, : m + 1].T
                    for e, eps in enumerate(eps_list):
                        Ngc = torch.clamp(Ng, min=eps)
                        Z = Ngc @ grid.wg
                        a.CAL[r, e] += (w / Z) @ torch.clamp(Ns, min=eps)
                        a.CALg[r, e] += (w / Z) @ Ngc
                    if name == "S2":
                        a.n_Dneg += int((A0 < 0).sum())
                        Fx = legendre(src["X"][st: st + chunk], m)
                        a.n_Nneg += int(((Fx * A).sum(1) < 0).sum())
                    if name != "S1":  # lowest-D samples in supp q seed the optimiser
                        Dm = torch.where(insupp, A0, torch.full_like(A0, math.inf))
                        K = min(num["mind"]["mc_seeds"], Dm.shape[0])
                        v, idx = torch.topk(Dm, K, largest=False)
                        cand_D = v if a.low_D is None else torch.cat([a.low_D, v])
                        cand_Y = Yc[idx] if a.low_Y is None else torch.cat([a.low_Y, Yc[idx]])
                        v2, i2 = torch.topk(cand_D, min(num["mind"]["mc_seeds"], cand_D.shape[0]), largest=False)
                        a.low_D, a.low_Y = v2, cand_Y[i2]
        stats[name] = acc

    # ---- min D: over the cube (S1, S2, S4, negativity) and over supp q for the Beta beliefs
    mind = {}
    for m in degrees:
        extra = [stats[n][m].low_Y[torch.isfinite(stats[n][m].low_D)] for n in names if n in ("S2", "S4")]
        extra = torch.cat(extra).cpu().numpy() if extra else None
        cube = min_D(models[m], rng_md, num["mind"], extra, device=device)
        mc_cube = min([stats[n][m].min_mc for n in names if n in ("S2", "S4")], default=math.inf)
        mind[(m, "cube")] = (min(cube[0], mc_cube), cube[0], mc_cube, cube[1], cube[2])
        for n in names:
            if n.startswith("S3"):
                a = stats[n][m]
                ex = a.low_Y[torch.isfinite(a.low_D)].cpu().numpy()
                res = min_D(models[m], rng_md, num["mind"], ex, beliefs[n]["logq"], beliefs[n]["thr"], device)
                best = min(res[0], a.min_mc) if not math.isnan(res[0]) else a.min_mc
                mind[(m, n)] = (best, res[0], a.min_mc, res[1], res[2])

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
            sdD = math.sqrt(max(float(a.SWD2.sum() / SW[-1]) - EqD ** 2, 0.0))
            base = dict(m=m, belief=name, n_coef=models[m].n_coef, chi_D=models[m].chi_D(),
                        minD=mc[0], minD_opt=mc[1], minD_mc=mc[2], minD_natt=mc[3], minD_nfeas=mc[4],
                        minD_supp=ms[0], minD_supp_opt=ms[1], minD_supp_mc=ms[2],
                        P_Dneg_u=stats["S2"][m].n_Dneg / beliefs["S2"]["Y"].shape[0] if "S2" in stats else NAN,
                        frac_Nneg=stats["S2"][m].n_Nneg / beliefs["S2"]["Y"].shape[0] if "S2" in stats else NAN,
                        q_valid=int(q_valid), defined=int(defined), n_bad=a.n_bad, n_drop=a.n_drop,
                        M=int(beliefs["S2" if name == "S1" else name]["Y"].shape[0]),
                        ess=ess, E_q_D=EqD, sd_q_D=sdD)
            base["mc_limited"] = int(ess / base["M"] < cfg["mc_limited_ess"])
            if dens is not None:
                dens[(name, m)] = dict(cd_c=cd_c[-1], eu_c=eu_c[-1], CD=CD[-1], EU=EU[-1], CAL=CAL[-1], GOLD=GOLD[-1],
                                       CALg=a.CALg.sum(0) / SW[-1], defined=defined, q_valid=q_valid)
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
