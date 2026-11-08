"""Numerical check of the paper's propositions against exact formulas and the production data (CPU, a few minutes).

    .venv/bin/python paper/check_theory.py      # writes paper/theory.json; exits non-zero if a check fails

Post hoc: nothing here is a study result or changes a verdict. Population moments of the equicorrelated Gaussian copula come
from the latent-factor form Z_l = sqrt(r) W + sqrt(1 - r) e_l, with Gauss-Hermite in W and e; the triple-basis output moments
use a three-dimensional Gauss-Hermite rule. Needs results/merged/population.parquet and results/summary/thresholds_*.csv.
"""

import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from numpy.polynomial import hermite_e as nph
from numpy.polynomial import legendre as npl
from scipy import stats
from scipy.optimize import brentq, linprog, minimize
from scipy.special import ndtr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gpu import copulas as C  # noqa: E402
from gpu.core import input_basis, population_model  # noqa: E402

KS = (2, 4, 8, 16, 32)
MS = (2, 4, 6)
fails, out = [], {}
ZZ, WW = nph.hermegauss(200)
WW = WW / WW.sum()
Z3, W3 = nph.hermegauss(70)
W3 = W3 / W3.sum()
UG = np.linspace(0, 1, 1001)
XG, WG = np.polynomial.legendre.leggauss(400)
XG, WG = (XG + 1) / 2, WG / 2


def check(name, ok):
    print(("ok   " if ok else "FAIL ") + name)
    fails.append(name) if not ok else None


def rs(rho):
    return 2 * math.sin(math.pi * rho / 6)


def fall(u, m):
    """f_0..f_m at u: shape u.shape + (m + 1,)."""
    return np.stack([math.sqrt(2 * n + 1) * npl.legval(2 * np.asarray(u) - 1, np.eye(m + 1)[n]) for n in range(m + 1)], -1)


def latent(r, m):
    """g_p(w) = E[f_p(Phi(sqrt(r) w + sqrt(1 - r) e))], p = 1..m, on the nodes ZZ: the moments of y given W = w."""
    return np.einsum("e,wep->wp", WW, fall(ndtr(math.sqrt(r) * ZZ[:, None] + math.sqrt(1 - r) * ZZ[None, :]), m)[..., 1:])


def amat(r, m):
    g = latent(r, m)
    return np.einsum("w,wp,wq->pq", WW, g, g)


def e23(xi):
    p1, p2, p3 = xi.sum(1), (xi**2).sum(1), (xi**3).sum(1)
    return 0.5 * (p1**2 - p2), (p1**3 - 3 * p1 * p2 + 2 * p3) / 6


def dpop(Y, r, m, basis):
    """D(y) of the population model through its latent form 1 + E_W[e_2(xi) (+ e_3(xi))], xi_l(w) = g(w) . f(y_l)."""
    e2, e3 = e23(latent(r, m) @ fall(Y, m)[:, 1:].T)
    return 1 + WW @ (e2 + (e3 if basis == "triple" else 0))


def tri(R3, m):
    """E[f_i(z_0) f_p(z_1) f_q(z_2)] for correlation matrix R3, degrees 0..m."""
    ev, V = np.linalg.eigh(R3)
    A = V * np.sqrt(np.maximum(ev, 0))
    G = np.stack([g.ravel() for g in np.meshgrid(Z3, Z3, Z3, indexing="ij")], 1)
    W = np.prod(np.meshgrid(W3, W3, W3, indexing="ij"), 0).ravel()
    T = [fall(ndtr(G @ A[i]), m) for i in range(3)]
    return np.einsum("n,ni,nj,nl->ijl", W, T[0], T[1], T[2], optimize=True)


def coupling(typ, k, rho):
    r = rs(rho)
    return r, (min(0.5, 0.8 * math.sqrt((1 + (k - 1) * r) / k)) if typ == "SUM" else min(0.5, 0.8 * math.sqrt((1 - r) / k)))


def bmat(c, m):
    """B[i, p] = E[f_i(x) f_p(y)] at Gaussian-copula correlation c, i, p = 0..m (bivariate rule in the same form)."""
    z = np.sqrt(1 - c * c) * ZZ[:, None] + c * ZZ[None, :]
    return np.einsum("a,b,abi,bp->ip", WW, WW, fall(ndtr(z), m), fall(ndtr(ZZ), m))


def t2_tv(typ, k, rho, m=4):
    """Pair: 0.5 ||E_u[G^2 R]||_1, closed form (k - 1) sum_l sum_i f_i B_ip(c_l) e_p; odd p vanish, so the sign of c_l drops."""
    r, c = coupling(typ, k, rho)
    A = amat(r, m)
    F = fall(XG, m)[:, 1:]
    e = (WG * np.einsum("np,pq,nq->n", F, A @ A, F)) @ F
    return 0.5 * WG @ np.abs(F @ (k * (k - 1) * (bmat(c, m)[1:, 1:] @ e)))


def t1_coef(typ, k, rho, m=4):
    """Triple: coefficients of T_1(x) = sum_{l<l'} sum_i f_i <A, C_i^{ll'}>, i = 1..m."""
    r, c = coupling(typ, k, rho)
    A = amat(r, m)

    def cc(s):
        T = tri(np.array([[1, c, s * c], [c, 1, r], [s * c, r, 1]]), m)
        return np.array([np.sum(A * T[i, 1:, 1:]) for i in range(1, m + 1)])

    if typ == "SUM":
        return math.comb(k, 2) * cc(1)
    h = k // 2
    return 2 * math.comb(h, 2) * cc(1) + h * h * cc(-1)


def t1_tv(typ, k, rho, m=4):
    return 0.5 * WG @ np.abs(fall(XG, m)[:, 1:] @ t1_coef(typ, k, rho, m))


def pop_rows(cols, **eq):
    f = [(a, "==", b) for a, b in eq.items()]
    return pd.read_parquet(os.path.join(ROOT, "results/merged/population.parquet"), columns=cols, filters=f)


# ---------------------------------------------------------------- 1. semantics, exact on a small grid
def semantics():
    rng = np.random.default_rng(1)
    n, nx = 40, 30
    J = rng.random((nx, n, n)) + 0.1
    J /= J.sum()  # a positive joint over (x, y1, y2)
    Dm = J.sum(0)
    q = rng.random((n, n))
    q /= q.sum()
    cd = (J * q).sum((1, 2)) / (Dm * q).sum()
    eu = (J / Dm * q).sum((1, 2))
    post = (J * q).sum((1, 2)) / (J * q).sum()  # Bayes with likelihood q(y) for a virtual observation
    z = q / Dm
    conv = (J * z).sum((1, 2)) / (Dm * z).sum()
    check("CD is the virtual-evidence posterior", np.abs(cd - post).max() < 1e-13)
    check("CD(q / D) = EU", np.abs(conv - eu).max() < 1e-13)
    til = q * Dm / (q * Dm).sum()
    check("EU(q D) = CD", np.abs((J / Dm * til).sum((1, 2)) - cd).max() < 1e-13)
    i = 7
    qm = np.zeros((n, n))
    qm[i] = 1 / n
    cdm = (J * qm).sum((1, 2)) / (Dm * qm).sum()
    cond = J[:, i, :].sum(1) / J[:, i, :].sum()
    eum = (J / Dm * qm).sum((1, 2))
    check("point on y1 and flat on y2: CD = p(x | y1)", np.abs(cdm - cond).max() < 1e-13)
    check("  ... and EU differs", np.abs(eum - cond).max() > 1e-5)
    # any positive g normalised after averaging equals the mixture of its conditionals under the Z-tilted belief
    g = np.exp(rng.standard_normal((nx, 50)))
    qq = rng.random(50)
    qq /= qq.sum()
    Zg = g.sum(0)
    check("normalising after averaging, general g", np.abs((g @ qq) / (Zg @ qq) - (g / Zg) @ (qq * Zg / (qq @ Zg))).max() < 1e-13)


# ---------------------------------------------------------------- 2. flat belief: exact expansion, then closed forms against data
def flat_expansion():
    rng = np.random.default_rng(2)
    Y = torch.tensor(rng.random((1 << 19, 2)))
    x, w = (torch.tensor(v) for v in (XG, WG))
    from gpu.core import legendre

    Fx = legendre(x, 4)
    res = {}
    for basis in ("pair", "triple"):
        R, _ = C.coupled_R("SUM", 2, 0.3)
        mod = population_model(R, 4, basis, nodes=64)
        A = mod.A(input_basis(Y, 4))
        D, G = A[:, 0], A[:, 0] - A[:, 0].mean()
        Rx = A[:, 1:] @ Fx[:, 1:].T
        h = (A / D[:, None]) @ Fx.T
        # in-sample identity with G centred in sample, so that it holds up to rounding
        lhs = (A @ Fx.T).mean(0) / D.mean() - h.mean(0)
        T1 = (G[:, None] * Rx).mean(0)
        rhs = (G[:, None] * (h - h.mean(0))).mean(0) / D.mean()
        res[basis] = dict(resid=float((lhs - rhs).abs().max()), t1=float(0.5 * (w * T1.abs()).sum()))
    check("flat-belief covariance form in sample", max(v["resid"] for v in res.values()) < 1e-12)
    out["expansion_sample"] = res
    # exact orthogonality with tensor quadrature: pair N_i is orthogonal to D - 1 under the flat belief
    u, wu = np.polynomial.legendre.leggauss(16)
    u, wu = (u + 1) / 2, wu / 2
    Yg = torch.tensor(np.stack([g.ravel() for g in np.meshgrid(u, u, indexing="ij")], 1))
    Wq = torch.tensor(np.outer(wu, wu).ravel())
    R, _ = C.coupled_R("SUM", 2, 0.3)
    Ap = population_model(R, 4, "pair", nodes=64).A(input_basis(Yg, 4))
    At = population_model(R, 4, "triple", nodes=64).A(input_basis(Yg, 4))
    op = float((Wq[:, None] * Ap[:, 1:] * (Ap[:, :1] - 1)).sum(0).abs().max())
    ot = float((Wq[:, None] * At[:, 1:] * (At[:, :1] - 1)).sum(0).abs().max())
    out["orthogonality"] = dict(pair=op, triple=ot)
    check("pair rows orthogonal to D - 1, flat belief", op < 1e-12)
    check("triple rows are not", ot > 1e-3)

    P = pop_rows(["k", "type", "basis", "rho_S", "d_denom", "d_denom_below", "chi_D", "defined"], m=4, eps=0.01, belief="S2")
    g = (
        P.groupby(["type", "basis", "k", "rho_S"])
        .agg(d=("d_denom", "median"), chi=("chi_D", "median"), und=("defined", lambda v: (v != 1).mean()), bl=("d_denom_below", "mean"))
        .reset_index()
    )
    g = g[(g.und < 0.5) & (g.bl < 0.5) & (g.chi > 0) & (g.d > 0)].copy()  # the point set of H1c
    g["pred"] = [t2_tv(r.type, r.k, r.rho_S) if r.basis == "pair" else t1_tv(r.type, r.k, r.rho_S) for r in g.itertuples()]
    g["ratio"] = g.d / g.pred
    cells = {}
    for (t, b), x in g.groupby(["type", "basis"]):
        lr = np.log(x.ratio)
        cells[f"{t}/{b}"] = dict(
            n=len(x),
            within2=float(((x.ratio >= 0.5) & (x.ratio <= 2)).mean()),
            lo=float(x.ratio.min()),
            hi=float(x.ratio.max()),
            med=float(x.ratio.median()),
            within125=float(((x.ratio >= 0.8) & (x.ratio <= 1.25)).mean()),
            sd_log=float(lr.std()),
        )
    small = g[g.chi <= 0.2]
    out["leading"] = dict(
        cells=cells,
        n=len(g),
        within2=float(((g.ratio >= 0.5) & (g.ratio <= 2)).mean()),
        lo=float(g.ratio.min()),
        hi=float(g.ratio.max()),
        small_n=len(small),
        small_lo=float(small.ratio.min()),
        small_hi=float(small.ratio.max()),
        points=g[["type", "basis", "k", "rho_S", "chi", "d", "pred"]].to_dict("records"),
    )
    check("leading-order formula within a factor 2 at every H1c point", out["leading"]["within2"] == 1)


def data_belief():
    """Triple: under the data belief CD - 1 = T_1 / (1 + chi^2), and GOLD is uniform, so d_total = 0.5 ||T_1||_1 / (1 + chi^2)."""
    P = pop_rows(["k", "type", "basis", "rho_S", "d_total", "chi_D", "E_q_D"], m=4, eps=0.01, belief="S4")
    g = P.groupby(["type", "basis", "k", "rho_S"]).agg(d=("d_total", "median"), chi=("chi_D", "median"), eqd=("E_q_D", "median")).reset_index()
    out["data_belief_eq_err"] = float((g.eqd - 1 - g.chi**2).abs().max() / (1 + g.chi**2).max())
    t = g[(g.basis == "triple") & (g.rho_S > 0) & (g.rho_S <= 0.3)].copy()
    t["pred"] = [t1_tv(r.type, r.k, r.rho_S) / (1 + r.chi**2) for r in t.itertuples()]
    t = t[t.d > 0.003]  # above the Monte Carlo level of d_total
    t["ratio"] = t.d / t.pred
    out["data_belief"] = dict(
        n=len(t), lo=float(t.ratio.min()), hi=float(t.ratio.max()), med=float(t.ratio.median()), pair_max=float(g[g.basis == "pair"].d.max())
    )
    check("data belief, triple: d_total matches T_1 / (1 + chi^2)", abs(math.log(out["data_belief"]["med"])) < 0.1)


# ---------------------------------------------------------------- 3. negativity
def relax(k, r, m, iters=200):
    """min of 1 + k^2/2 s'As - k/2 int phi dmu over probability measures on UG, by fully corrective Frank-Wolfe.
    Returns a certified lower value (objective minus duality gap), the atoms and their weights."""
    A = amat(r, m)
    F = fall(UG, m)[:, 1:]
    phi = np.einsum("np,pq,nq->n", F, A, F)
    act, w = [0, len(UG) - 1], np.array([0.5, 0.5])
    best = -np.inf
    for _ in range(iters):
        Fa, pa, n = F[act], phi[act], len(act)

        def obj(v):
            s = v @ Fa
            return 0.5 * k * k * s @ A @ s - 0.5 * k * (v @ pa)

        def grad(v):
            return k * k * (Fa @ (A @ (v @ Fa))) - 0.5 * k * pa

        res = minimize(
            obj,
            w,
            jac=grad,
            bounds=[(0, 1)] * n,
            method="SLSQP",
            options=dict(maxiter=300, ftol=1e-14),
            constraints=[dict(type="eq", fun=lambda v: v.sum() - 1, jac=lambda v: np.ones(n))],
        )
        w = np.clip(res.x, 0, None)
        w /= w.sum()
        keep = w > 1e-9
        act, w = [a for a, kp in zip(act, keep) if kp], w[keep]
        s = w @ F[act]
        g = k * k * (F @ (A @ s)) - 0.5 * k * phi
        val = 0.5 * k * k * s @ A @ s - 0.5 * k * (w @ phi[act])
        j = int(np.argmin(g))
        gap = w @ g[act] - g[j]
        best = max(best, val - gap)
        if gap < 1e-9:
            break
        if j not in act:
            act.append(j)
            w = np.r_[w * 0.95, 0.05]
    return 1 + best, UG[act], w


def local_min(k, r, m, atoms, wts, rng, starts=12):
    """Upper value of min D (pair) by coordinate descent over a fine grid, from a rounding of the relaxed measure and random points."""
    A = amat(r, m)
    F = fall(UG, m)[:, 1:]
    FA = F @ A
    phi = np.einsum("np,np->n", FA, F)
    best = np.inf
    for s in range(starts):
        if s == 0:
            cnt = np.floor(wts * k).astype(int)
            cnt[np.argsort(-(wts * k - cnt))[: k - cnt.sum()]] += 1
            idx = np.repeat([int(np.argmin(np.abs(UG - a))) for a in atoms], cnt)
        else:
            idx = rng.integers(0, len(UG), k)
        for _ in range(60):
            moved = False
            for l in rng.permutation(k):
                v = FA @ (F[idx].sum(0) - F[idx[l]])  # D as a function of y_l, up to a constant
                j = int(np.argmin(v))
                if j != idx[l] and v[j] < v[idx[l]] - 1e-14:
                    idx[l] = j
                    moved = True
            if not moved:
                break
        S = F[idx].sum(0)
        best = min(best, 1 + 0.5 * (S @ A @ S - phi[idx].sum()))
    return best


def bisect(fun, lo=1e-4, hi=0.9, it=24):
    if fun(hi) > 0:
        return float("nan")
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if fun(mid) > 0 else (lo, mid)
    return hi


def pair_onset():
    rng = np.random.default_rng(3)
    th = pd.read_csv(os.path.join(ROOT, "results/summary/thresholds_population.csv"))
    th = th[(th.belief == "S2") & (th.type == "SUM") & (th.basis == "pair")]
    # the quadratic form, against the repository's model
    R, _ = C.coupled_R("SUM", 8, 0.1)
    mod = population_model(R, 4, "pair", nodes=64)
    Y = rng.random((64, 8))
    r = rs(0.1)
    A = amat(r, 4)
    F = fall(Y, 4)[..., 1:]
    S = F.sum(1)
    q = 1 + 0.5 * (np.einsum("np,pq,nq->n", S, A, S) - np.einsum("nlp,pq,nlq->n", F, A, F))
    err = float(np.abs(q - mod.D(torch.tensor(Y)).numpy()).max())
    out["quadratic_form_err"] = err
    check("pair D is the quadratic form", err < 1e-6)
    check("A is positive semidefinite", min(np.linalg.eigvalsh(amat(rs(x), 4)).min() for x in (0.05, 0.3, 0.8)) > -1e-12)
    rows = []
    for m in MS:
        F = fall(UG, m)[:, 1:]
        for k in KS:
            crude = brentq(lambda x: 1 - k / 2 * np.einsum("np,pq,nq->n", F, amat(rs(x), m), F).max(), 1e-5, 0.95)
            lo = bisect(lambda x: relax(k, rs(x), m)[0])
            _, at, wt = relax(k, rs(lo), m)
            up = bisect(lambda x: local_min(k, rs(x), m, *relax(k, rs(x), m)[1:], rng), lo=lo * 0.98)
            on = float(th[(th.k == k) & (th.m == m)].neg_onset.iloc[0])
            rows.append(
                dict(
                    m=m,
                    k=k,
                    crude=crude,
                    lower=lo,
                    upper=up,
                    measured=on,
                    atoms=[float(a) for a in at],
                    weights=[float(w) for w in wt],
                    corner_neg=corner_onset(k, m, "pair")[0] is not None,
                )
            )
            print(
                f"pair m={m} k={k:2d} crude {crude:.4f} lower {lo:.4f} upper {up:.4f} measured {on:.4f} atoms {np.round(sorted(at), 3)}", flush=True
            )
    out["pair_onset"] = rows
    check("lower bound below the upper bound", all(r_["lower"] <= r_["upper"] * 1.0005 for r_ in rows))
    check("no pair corner goes negative", not any(r_["corner_neg"] for r_ in rows))
    b = [r_ for r_ in rows if r_["k"] >= 4]
    out["pair_onset_gap"] = dict(
        max_rel_upper_lower=max(r_["upper"] / r_["lower"] - 1 for r_ in b),
        max_rel_measured=max(abs(r_["measured"] / r_["lower"] - 1) for r_ in b),
        max_rel_measured_m4=max(abs(r_["measured"] / r_["lower"] - 1) for r_ in b if r_["m"] == 4),
    )


def psi_min(k, rho, m, nodes=801):
    """min of Psi_k over measures on a grid by Kelley's cutting planes: a linear program in the weights, with an epigraph variable for the quadratic."""
    A = amat(rs(rho), m)
    F = fall(np.linspace(0, 1, nodes), m)[:, 1:]
    phi = np.einsum("np,pq,nq->n", F, A, F)
    c = np.r_[-k / 2 * phi, 1.0]
    eq = np.r_[np.ones(nodes), 0][None]
    cuts = [np.zeros(m)]
    for _ in range(200):
        g = [k * k * (A @ u) for u in cuts]
        res = linprog(
            c,
            A_ub=np.array([np.r_[F @ gi, -1.0] for gi in g]),
            b_ub=[gi @ u / 2 for gi, u in zip(g, cuts)],
            A_eq=eq,
            b_eq=[1],
            bounds=[(0, None)] * nodes + [(None, None)],
            method="highs",
        )
        mu, tau = res.x[:-1], res.x[-1]
        u = mu @ F
        q = k * k / 2 * u @ A @ u
        if q - tau < 1e-7:
            break
        cuts.append(u)
    return 1 + q - k / 2 * mu @ phi


def pair_onset_large():
    """k rho* of the convex bound far beyond the study's range, m = 4. A numerical observation, not a limit theorem."""
    rows = []
    for k in (8, 32, 128, 512, 2048):
        x = brentq(lambda rho: psi_min(k, rho, 4), 0.3 / k, 3 / k, xtol=2e-3 / k)
        rows.append(dict(k=k, rho=x, k_rho=k * x))
        print(f"convex bound k={k} k rho* {k * x:.3f}", flush=True)
    out["pair_onset_large"] = rows
    fw = {r["k"]: r["lower"] for r in out["pair_onset"] if r["m"] == 4}
    check("cutting planes agree with Frank-Wolfe at k = 8, 32", all(abs(r["rho"] / fw[r["k"]] - 1) < 0.02 for r in rows if r["k"] in fw))
    check("k rho* of the convex bound settles", abs(rows[-1]["k_rho"] - rows[-2]["k_rho"]) < 0.005)


def corner_onset(k, m, basis):
    def val(x):
        v = [dpop(np.r_[np.ones(n), np.zeros(k - n)], rs(x), m, basis) for n in range(k // 2 + 1)]
        return min(v), int(np.argmin(v))

    g = np.geomspace(0.002, 0.89, 70)
    v = [val(x)[0] for x in g]
    hit = [i for i in range(len(g)) if v[i] < 0]
    if not hit:
        return None, None
    i = hit[0]
    c = brentq(lambda z: val(z)[0], g[i - 1] if i else 1e-4, g[i])
    return c, val(c * 1.001)[1]


def corner_formula():
    """Exact corner values through four moments of xi_+-(w) = g(w) . f(1 or 0), and the balanced pair corner 1 + (k (k - 1) a - k b) / 2."""
    worst, crit = 0.0, np.inf
    for m in MS:
        for x in (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.89):
            r = rs(x)
            g = latent(r, m)
            xp, xm = g @ fall(1.0, m)[1:], g @ fall(0.0, m)[1:]
            M2, C11, M3, C21 = WW @ xp**2, WW @ (xp * xm), WW @ xp**3, WW @ (xp**2 * xm)
            a, b = (M2 + C11) / 2, (M2 - C11) / 2
            kk = np.arange(2, 4098, 2)
            crit = min(crit, (1 + 0.5 * (kk * (kk - 1) * a - kk * b)).min())
            for k in (2, 3, 8, 16):
                for n in range(k + 1):
                    y = np.r_[np.ones(n), np.zeros(k - n)]
                    cb = math.comb
                    dp = 1 + (cb(n, 2) + cb(k - n, 2)) * M2 + n * (k - n) * C11
                    dt = dp + (cb(n, 3) + cb(k - n, 3)) * M3 + (cb(n, 2) * (k - n) + n * cb(k - n, 2)) * C21
                    worst = max(worst, abs(dp - dpop(y, r, m, "pair")), abs(dt - dpop(y, r, m, "triple")))
                    if 2 * n == k:  # the triple term at the balanced corner, through the even and odd parts of xi_+ in w
                        xe, xo = (xp + xm) / 2, (xp - xm) / 2
                        h = n
                        worst = max(worst, abs(dt - dp - h * (h - 1) * ((4 * h - 2) / 3 * (WW @ xe**3) - 2 * (WW @ (xe * xo**2)))))
    out["corner_formula"] = dict(err=float(worst), min_balanced_pair=float(crit))
    check("corner formula", worst < 1e-9)
    check("balanced pair corners stay positive for every even k up to 4096 on the grid", crit > 0)


def triple_onset():
    th = pd.read_csv(os.path.join(ROOT, "results/summary/thresholds_population.csv"))
    th = th[(th.belief == "S2") & (th.type == "SUM") & (th.basis == "triple")]
    P = pop_rows(["k", "basis", "m", "rho_S", "minD"], eps=0.01, belief="S2", type="SUM", basis="triple")
    rows = []
    for m in MS:
        for k in KS[1:]:
            c, n = corner_onset(k, m, "triple")
            rows.append(dict(m=m, k=k, corner=c, npos=n, measured=float(th[(th.k == k) & (th.m == m)].neg_onset.iloc[0])))
    out["triple_onset"] = rows
    # at equal rho, corners against the production minimum: min D <= corner value must hold, and is it nearly equal?
    g = P[(P.m == 4) & (P.rho_S > 0)].groupby(["k", "rho_S"]).minD.median().reset_index()
    g = g[(g.k >= 4) & (g.minD > -0.1) & (g.minD < 0.9)].copy()  # up to just past the onset; further on the minimiser leaves the corners
    g["corner"] = [
        min(dpop(np.r_[np.ones(n), np.zeros(int(r.k) - n)], rs(r.rho_S), 4, "triple") for n in range(int(r.k) // 2 + 1)) for r in g.itertuples()
    ]
    out["triple_corner_grid"] = dict(n=len(g), max_abs=float((g.corner - g.minD).abs().max()), min_diff=float((g.corner - g.minD).min()))
    check("m >= 4: corner onset within 7 % of the measured onset", all(abs(r["corner"] / r["measured"] - 1) < 0.07 for r in rows if r["m"] >= 4))
    check("m = 2: no triple corner goes negative", all(r["corner"] is None for r in rows if r["m"] == 2))


def belief_mass():
    """E_q[D] for product beliefs is 1 + E_W[e_2(beta) (+ e_3(beta))], beta_l(w) = g(w) . b_l."""
    AB = {"S3a": [(3, 3)], "S3b": [(4, 2)], "S3c": [(4, 2), (2, 4)], "S3d": [(6, 6)]}
    P = pop_rows(["k", "basis", "m", "belief", "rho_S", "E_q_D"], eps=0.01, type="SUM")
    P = P[P.belief.isin(AB) & (P.rho_S <= 0.85)]
    g = P.groupby(["k", "basis", "m", "belief", "rho_S"]).E_q_D.median().reset_index()
    F = fall(XG, 6)[:, 1:]

    def eqd(k, rho, m, basis, bel):
        bs = np.stack([(WG * stats.beta.pdf(XG, *AB[bel][i % len(AB[bel])])) @ F[:, :m] for i in range(k)])
        e2, e3 = e23(latent(rs(rho), m) @ bs.T)
        return 1 + WW @ (e2 + (e3 if basis == "triple" else 0))

    s = g.sample(800, random_state=0).copy()
    s["th"] = [eqd(r.k, r.rho_S, r.m, r.basis, r.belief) for r in s.itertuples()]
    rel = (s.th - s.E_q_D).abs() / np.maximum(1, s.E_q_D.abs())
    pr = g[g.basis == "pair"]
    out["eqd"] = dict(
        n=len(s),
        max_rel=float(rel.max()),
        med_abs=float((s.th - s.E_q_D).abs().median()),
        pair_same_min=float(pr[pr.belief != "S3c"].E_q_D.min()),
        pair_s3c_min=float(pr[pr.belief == "S3c"].E_q_D.min()),
        triple_min=float(g[g.basis == "triple"].E_q_D.min()),
    )
    check("E_q[D] formula against the population rows", out["eqd"]["max_rel"] < 0.01)
    check("pair, identical marginals: E_q[D] >= 1 up to Monte Carlo", out["eqd"]["pair_same_min"] > 1 - 1e-3)


# ---------------------------------------------------------------- 4. sampling noise
def noise_variance():
    """Var D_hat(y) = K(y) / n at independence, against simulated coefficient estimates, k = 3, n = 2000."""
    rng = np.random.default_rng(4)
    k, m, n, reps = 3, 2, 2000, 4000
    y0 = np.array([0.0, 1.0, 0.3])
    f0 = fall(y0, m)[:, 1:]
    idx = [(a, b) for a in range(k) for b in range(a + 1, k)]
    vals = np.empty(reps)
    for t in range(reps):
        F = fall(rng.random((n, k)), m)[..., 1:]
        v = sum(F[:, a].mean(0) @ f0[a] for a in range(k))
        v += sum((F[:, a] * (F[:, b] @ f0[b])[:, None]).mean(0) @ f0[a] for a, b in idx)
        vals[t] = v
    s = (f0**2).sum(1)
    K = s.sum() + sum(s[a] * s[b] for a, b in idx)
    out["noise_var"] = dict(ratio=float(vals.var() * n / K))
    check("noise variance formula", abs(out["noise_var"]["ratio"] - 1) < 0.08)


def surrogate(k, basis, m, reps, starts, rng):
    """Gaussian limit at independence: coefficients i.i.d. N(0, 1). Coordinate descent over the cube and over the corners only."""
    U = np.linspace(0, 1, 41)
    FU = fall(U, m)[:, 1:]
    T = m * (m + 2)
    res = []
    for _ in range(reps):
        a1 = rng.standard_normal((k, m))
        W = rng.standard_normal((k, k, m, m))
        i, j = np.triu_indices(k, 1)
        W2 = np.zeros((k, k, m, m))
        W2[i, j] = W[i, j]
        W2 = W2 + W2.transpose(1, 0, 3, 2)
        W3t = None
        if basis == "triple" and k >= 3:
            V = rng.standard_normal((k, k, k, m, m, m)).astype(np.float32)
            a, b, c = np.ogrid[:k, :k, :k]
            V *= ((a < b) & (b < c))[..., None, None, None]
            W3t = sum(V.transpose(*p, *(q + 3 for q in p)) for p in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)))
            del V

        def lin(l, F):
            c = a1[l] + np.einsum("jpq,jq->p", W2[l], F)
            if W3t is not None:
                c = c + 0.5 * np.einsum("abpqr,aq,br->p", W3t[l], F, F)
            return c

        def value(F):
            v = np.sum(a1 * F) + 0.5 * np.einsum("lp,ljpq,jq->", F, W2, F)
            if W3t is not None:
                v += np.einsum("abcpqr,ap,bq,cr->", W3t, F, F, F, optimize=True) / 6
            return v

        bc = bu = np.inf
        bd = 0.0
        for _s in range(starts):
            idx = rng.choice([0, len(U) - 1], k)
            for allowed in (np.array([0, len(U) - 1]), np.arange(len(U))):
                for _sw in range(30):
                    moved = False
                    for l in rng.permutation(k):
                        Fl = FU[idx].copy()
                        Fl[l] = 0
                        j2 = allowed[int(np.argmin(FU[allowed] @ lin(l, Fl)))]
                        if j2 != idx[l]:
                            idx[l], moved = j2, True
                    if not moved:
                        break
                v = value(FU[idx])
                if len(allowed) == 2:
                    bc = min(bc, v)
                elif v < bu:
                    bu, bd = v, float(np.mean((idx == 0) | (idx == len(U) - 1)))
        res.append((bc, min(bu, bc), bd))
    K = k * T + math.comb(k, 2) * T**2 + (math.comb(k, 3) * T**3 if basis == "triple" else 0)
    r = np.array(res)
    return dict(
        z_corner=float(-np.median(r[:, 0]) / math.sqrt(K)), z_cube=float(-np.median(r[:, 1]) / math.sqrt(K)), boundary=float(np.median(r[:, 2]))
    )


def noise_extremes():
    rng = np.random.default_rng(5)
    rows = []
    ph = {(r["basis"], r["k"]): r for r in json.load(open(os.path.join(ROOT, "paper/posthoc.json")))["noise"]}
    for basis, ks in (("pair", KS), ("triple", KS[1:4])):
        for k in ks:
            r = (
                surrogate(k, basis, 4, 20 if k <= 16 else 12, 20 if k <= 8 else 8, rng)
                if basis == "pair" or k <= 8
                else surrogate(k, basis, 4, 6, 3, rng)
            )
            r.update(basis=basis, k=k, union=math.sqrt(2 * k * math.log(2)), measured=ph[(basis, k)]["z"])
            rows.append(r)
            print(r, flush=True)
    out["noise_z"] = rows
    check("surrogate cube minimum at least as deep as corners", all(r["z_cube"] >= r["z_corner"] - 1e-9 for r in rows))
    check(
        "pair: measured z below the surrogate cube value at k = 32",
        ph[("pair", 32)]["z"] < next(r for r in rows if r["basis"] == "pair" and r["k"] == 32)["z_cube"],
    )


# ---------------------------------------------------------------- 5. the sharper TV bound
def tv_bounds():
    rng = np.random.default_rng(6)
    from gpu.core import legendre

    x, w = torch.tensor(XG), torch.tensor(WG)
    Fx = legendre(x, 4)
    ab = {"S2": (1, 1), "S3a": (3, 3), "S3b": (4, 2), "S3d": (6, 6)}
    rows = []
    for k, rhos in ((2, (0.1, 0.2, 0.3, 0.4)), (4, (0.05, 0.1, 0.15)), (8, (0.03, 0.06, 0.1))):
        for basis in ("pair", "triple"):
            for typ in ("SUM", "CONTRAST"):
                for rho in rhos:
                    R, _ = C.coupled_R(typ, k, rho)
                    mod = population_model(R, 4, basis, nodes=48)
                    for b, (p, q) in ab.items():
                        Y = torch.tensor(rng.beta(p, q, (1 << 17, k)))
                        A = mod.A(input_basis(Y, 4))
                        D = A[:, 0]
                        if float(D.min()) < 0.01:
                            continue
                        h = A / D[:, None]
                        hx = h @ Fx.T
                        cd, eu = (A.mean(0) / D.mean()) @ Fx.T, h.mean(0) @ Fx.T
                        tv = 0.5 * float((w * (cd - eu).abs()).sum())
                        cv = float(D.std() / D.mean().abs())
                        Sh = float((w * hx.std(0)).sum())
                        S2 = math.sqrt(float(h[:, 1:].var(0).sum()))
                        c1 = float((hx.abs() * w).sum(1).max())
                        e1 = float((D / D.mean() - 1).abs().mean())
                        rows.append(
                            dict(
                                k=k,
                                basis=basis,
                                type=typ,
                                rho=rho,
                                belief=b,
                                tv=tv,
                                sharp=0.5 * cv * Sh,
                                coef=0.5 * cv * S2,
                                old=0.5 * cv,
                                oldc=0.5 * c1 * cv,
                                mid=0.5 * c1 * e1,
                            )
                        )
    d = pd.DataFrame(rows)
    d = d[d.tv > 3e-4]  # above the Monte Carlo level at this sample size
    out["tv_bound"] = dict(
        n=len(d),
        max_sharp=float((d.tv / d.sharp).max()),
        med_sharp=float((d.tv / d.sharp).median()),
        med_coef=float((d.tv / d.coef).median()),
        max_coef=float((d.tv / d.coef).max()),
        med_old=float((d.tv / d.old).median()),
        med_oldc=float((d.tv / d.oldc).median()),
        med_gain=float((d.oldc / d.sharp).median()),
    )
    check("sharper bound holds on all cases", out["tv_bound"]["max_sharp"] <= 1 + 1e-9 and out["tv_bound"]["max_coef"] <= 1 + 1e-9)


def main():
    for f in (
        semantics,
        flat_expansion,
        data_belief,
        corner_formula,
        pair_onset,
        pair_onset_large,
        triple_onset,
        belief_mass,
        noise_variance,
        tv_bounds,
        noise_extremes,
    ):
        print("---", f.__name__, flush=True)
        f()
        json.dump(out, open(os.path.join(ROOT, "paper/theory.json"), "w"), indent=1, sort_keys=True)
    print("FAIL: " + "; ".join(fails) if fails else "PASS")
    sys.exit(bool(fails))


if __name__ == "__main__":
    main()
