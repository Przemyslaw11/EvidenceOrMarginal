"""Validation tests M1-M12 of the scaled-up study (M13: densprop.domain.convergence; M14: validation_chain).

    densprop validate pre  [--device cuda] [--only M4,M9]   # M1-M12, before production
    densprop validate post                                  # M13 from the merged convergence and gauss results
    densprop validate network [--device cuda]               # M14 (Stage 4)
Results are merged into the file given by --out, default results/small/test_results.json. tests/test_results.json is the
record of the GPU validation that the paper reports; a run writes there only when asked with --out tests/test_results.json.
Each test returns (pass, detail). A failing strict test is reported as failed. Tolerances were fixed before
production; the one later change, to the reference of M7, is described in the paper.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import mpmath
import numpy as np
import pandas as pd
import torch

from densprop.domain import copulas as C
from densprop.domain.jobs import rng_for, with_id
from densprop.domain.neuron import (
    DT,
    Estimator,
    Model,
    count_formula,
    dense3_from_model,
    gauss_legendre,
    input_basis,
    legendre,
    masks,
    model_from_dense3,
    order_of,
    population_model,
)
from densprop.domain.propagate import Grid, beta_logq, evaluate_case, min_D, sample_beliefs
from densprop.domain.sweep import run_job

ROOT = "."  # the repository: the three-input study's stored results and the working directory of M11
DEV = "cpu"
CFG: dict = {}


def configure(cfg: dict, device: str, root: str) -> None:
    """The study configuration, the device and the repository root of the tests; call before running them."""
    global CFG, DEV, ROOT  # the tests read these as module settings, as they did when they were a script
    CFG, DEV, ROOT = cfg, device, root


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def gauss_samples(R, n, seed):
    rng = np.random.default_rng(seed)
    cop = C.Gaussian(R, DEV)
    return cop, cop.uniforms(C.base_variates(rng, n, len(R), ("Z",)))


def estimate_model(R, n, basis, m=6, seed=0, chunk=200000):
    rng = np.random.default_rng(seed)
    cop = C.Gaussian(R, DEV)
    est = Estimator(len(R) - 1, m, basis, DEV)
    done = 0
    while done < n:
        c = min(chunk, n - done)
        est.add(cop.uniforms(C.base_variates(rng, c, len(R), ("Z",))))
        done += c
    return cop, est.model()


def beliefs_for(k, M, seed, names=("S2", "S3a", "S3b", "S3c", "S3d"), cop=None):
    rngs = {nm: np.random.default_rng([seed, i]) for i, nm in enumerate(CFG["order"])}
    raw = sample_beliefs(rngs, M, k, CFG["beliefs"], [n for n in names if n != "S4"])
    out = {}
    for nm, d in raw.items():
        b = dict(Y=torch.as_tensor(d["Y"], device=DEV))
        if "X" in d:
            b["X"] = torch.as_tensor(d["X"], device=DEV)
        if "ab" in d:
            lq, lmax = beta_logq(d["ab"])
            b.update(logq=lq, thr=lmax + math.log(CFG["supp_rel"]))
        out[nm] = b
    if "S4" in names:
        base = C.base_variates(rngs["S4"], M, k + 1, cop.kinds)
        out["S4"] = dict(Y=cop.uniforms(base)[:, 1:].contiguous())
    return out


def small_cfg(M, R=8, starts=2000, steps=400):
    cfg = json.loads(json.dumps(CFG))
    cfg["numerics"].update(M=M, R=R, chunk_mc=min(16384, M // R))
    cfg["numerics"]["mind"].update(starts=starts, steps=steps)
    return cfg


# ---------------------------------------------------------------- M1
def M1():
    x, w = gauss_legendre(64, DEV)
    F = legendre(x, 8)
    G = (F * w[:, None]).T @ F
    e1 = float((G - torch.eye(9, dtype=DT, device=DEV)).abs().max())
    u = x
    ex = [
        math.sqrt(3) * (2 * u - 1),
        math.sqrt(5) * (6 * u**2 - 6 * u + 1),
        math.sqrt(7) * (20 * u**3 - 30 * u**2 + 12 * u - 1),
        3 * (70 * u**4 - 140 * u**3 + 90 * u**2 - 20 * u + 1),
    ]
    e2 = max(float((F[:, i + 1] - ex[i]).abs().max()) for i in range(4))
    return e1 < 1e-12 and e2 < 1e-12, f"Gram max err {e1:.2e}; explicit f1..f4 max err {e2:.2e}"


# ---------------------------------------------------------------- M2
def M2():
    det = []
    ok = True
    for m in (2, 4, 6):
        n2 = Model(2, m, "full", _zero_blocks(2, m, "full")).n_coef
        n3 = Model(3, m, "full", _zero_blocks(3, m, "full")).n_coef
        ok &= n2 == (m + 1) ** 3 and n3 == (m + 1) ** 4 and n2 == Model(2, m, "triple", _zero_blocks(2, m, "triple")).n_coef
        det.append(f"full m={m}: k=2 {n2}, k=3 {n3}")
    for k, m in ((2, 2), (4, 3), (8, 4), (16, 2), (32, 2), (5, 6)):
        for basis in ("pair", "triple"):
            got = Model(k, m, basis, _zero_blocks(k, m, basis)).n_coef
            ok &= got == count_formula(k, m, basis)
    det.append("pair/triple counts match the formula for (k,m) in (2,2),(4,3),(8,4),(16,2),(32,2),(5,6)")
    import itertools

    k, m, cap = 3, 2, 3
    for basis in ("pair", "triple", "full"):
        o = order_of(basis, k)
        brute = sum(1 for j in itertools.product(range(m + 1), repeat=k + 1) if sum(v > 0 for v in j) <= o and sum(j) <= cap)
        got = Model(k, m, basis, _zero_blocks(k, m, basis), cap=cap).n_coef
        ok &= got == brute
    det.append("degree-cap masks match brute-force enumeration (k=3, m=2, cap=3)")
    R, _ = C.coupled_R("SUM", 2, 0.4)
    _, U = gauss_samples(R, 1000, 1)
    est = Estimator(2, 6, "triple", DEV)
    est.add(U)
    mod, _ = est.model()
    Un = _np(U)
    F = [_np(legendre(torch.as_tensor(Un[:, i]), 6)) for i in range(3)]
    dense = np.einsum("ni,nj,nl->ijl", *F) / len(Un)
    e = float(np.abs(_np(dense3_from_model(mod)) - dense).max())
    ok &= e < 1e-12
    det.append(f"block estimator vs dense d=3 estimator: max diff {e:.2e}")
    return bool(ok), "; ".join(det)


def _zero_blocks(k, m, basis):
    mk = masks(k, m, basis, None, DEV)
    return {n: (None if v is None else torch.zeros(v.shape, dtype=DT, device=DEV)) for n, v in mk.items()} | {
        "c": torch.zeros(m + 1, dtype=DT, device=DEV)
    }


# ---------------------------------------------------------------- M3
def M3():
    det, ok = [], True
    k, rho = 4, 0.3
    R, _ = C.coupled_R("SUM", k, rho)
    res = {}
    for n in (10**6, 10**7):
        _, (mod, se) = estimate_model(R, n, "pair", m=4, seed=3)
        res[n] = (mod, se)
    mod, se = res[10**6]
    b, s = mod.blocks, se
    ok &= abs(float(b["c"][0]) - 1) < 1e-15
    # pairwise a_(0,1,1) vs rho_S and vs the sample Spearman correlation of the same samples
    _, U = gauss_samples(R, 10**6, 3)
    from scipy.stats import rankdata

    Un = _np(U)
    rk = np.stack([rankdata(Un[:, i]) for i in range(1, k + 1)], 1)
    pa, pb = np.triu_indices(k, 1)
    z_sp = z_rho = 0.0
    for t, (a, bb) in enumerate(zip(pa, pb)):
        sp = np.corrcoef(rk[:, a], rk[:, bb])[0, 1]
        z_sp = max(z_sp, abs(float(b["W2"][0, t, 0, 0]) - sp) / float(s["W2"][0, t, 0, 0]))
        z_rho = max(z_rho, abs(float(b["W2"][0, t, 0, 0]) - rho) / float(s["W2"][0, t, 0, 0]))
    z_single = float((b["W1"][0].abs() / s["W1"][0]).max())
    z_x = float((b["c"][1:].abs() / s["c"][1:]).max())
    ok &= z_sp <= 5 and z_rho <= 5 and z_single <= 5 and z_x <= 5
    det.append(
        f"a_0=1; max |a_(0,1,1)-Spearman|/SE={z_sp:.2f}, vs rho_S {z_rho:.2f}; single-index max |a|/SE: "
        f"inputs {z_single:.2f}, output {z_x:.2f} (n=1e6, k=4, SUM, rho_S=0.3)"
    )
    ratio = float((res[10**6][1]["W2"] / res[10**7][1]["W2"]).median())
    ok &= abs(ratio / math.sqrt(10) - 1) < 0.05
    det.append(f"SE(1e6)/SE(1e7) median {ratio:.3f} (sqrt(10)={math.sqrt(10):.3f})")
    # population coefficients (Gauss-Hermite) vs MC estimates, and GH node convergence
    for basis in ("pair", "triple"):
        _, (mc, mse) = estimate_model(R, 10**6, basis, m=4, seed=5)
        gh = CFG["numerics"]["gh_nodes"]
        pop = population_model(R, 4, basis, gh, DEV)
        zmax = 0.0
        for nm in ("c", "W1", "W2", "W3"):
            if pop.blocks[nm] is None:
                continue
            mk = pop.mask[nm] & (mse[nm] > 0)
            z = ((mc.blocks[nm] - pop.blocks[nm]).abs() / torch.where(mk, mse[nm], torch.ones_like(mse[nm])))[mk]
            zmax = max(zmax, float(z.max()))
        ncoef = pop.n_coef
        # max of ~ncoef |z| scores: 5.5 covers ncoef up to ~1e7 at the 5% level under independence
        ok &= zmax <= 5.5
        det.append(f"{basis}: population vs MC max |z|={zmax:.2f} over {ncoef} coefficients")
    R9, _ = C.coupled_R("SUM", 4, 0.9)
    p1, p2 = population_model(R9, 6, "triple", gh, DEV), population_model(R9, 6, "triple", gh + 32, DEV)
    e_gh = max(float((p1.blocks[nm] - p2.blocks[nm]).abs().max()) for nm in ("W1", "W2", "W3"))
    ok &= e_gh < 1e-9
    det.append(f"Gauss-Hermite {gh} vs {gh + 32} nodes (m=6, rho_S=0.9, triple): max diff {e_gh:.1e}")
    return bool(ok), "; ".join(det)


# ---------------------------------------------------------------- copula checks (part of M7: GOLD correctness)
def COP():
    det, ok = [], True
    from scipy.stats import kendalltau

    k = 3
    s = torch.linspace(-8, 8, 4001, dtype=DT, device=DEV)
    for fam, typ in (("gauss", "SUM"), ("t3", "SUM"), ("t10", "CONTRAST"), ("clayton", "EXCH"), ("gumbel", "EXCH")):
        kk = 4 if typ == "CONTRAST" else k
        cop, info = C.make_copula(fam, typ, kk, 0.5, DEV)
        rng = np.random.default_rng(11)
        U = _np(cop.uniforms(C.base_variates(rng, 20000, kk + 1, cop.kinds)))
        tau_yy = kendalltau(U[:, 1], U[:, 2])[0]
        target = info["tau"]
        ok_tau = abs(tau_yy - target) < 0.02
        Uy = torch.as_tensor(U[:50, 1:], device=DEV)
        p = cop.cond_s(s, Uy)
        h = float(s[1] - s[0])
        integ = (p.sum(1) - 0.5 * (p[:, 0] + p[:, -1])) * h
        e_int = float((integ - 1).abs().max())
        ok &= ok_tau and e_int < 1e-6
        det.append(f"{fam}/{typ}: tau_yy {tau_yy:.3f} vs {target:.3f}; max|int c(x|y)dx-1| {e_int:.1e}")
    g = C.Gumbel(2.0, 3, DEV)
    rng = np.random.default_rng(5)
    th, w = rng.random(200000) * math.pi, rng.standard_exponential(200000)
    a = g.alpha
    V = (np.sin(a * th) / np.sin(th) ** (1 / a)) * (np.sin((1 - a) * th) / w) ** ((1 - a) / a)
    zs = [abs(np.exp(-t * V).mean() - math.exp(-(t**a))) / (np.exp(-t * V).std() / math.sqrt(len(V))) for t in (0.5, 1, 2)]
    ok &= max(zs) < 4
    det.append(f"positive-stable frailty Laplace transform max |z| {max(zs):.2f}")
    # analytic conditional vs finite-difference derivative of the closed-form conditional CDF (Clayton)
    cl = C.Clayton(1.5, 3, DEV)
    y = torch.tensor([[0.3, 0.7]], dtype=DT, device=DEV)
    T = 1 + (y ** (-1.5) - 1).sum()
    xs = torch.tensor([0.05, 0.3, 0.8], dtype=DT, device=DEV)
    cdf = lambda x: ((T + x ** (-1.5) - 1) / T) ** (-(1 / 1.5 + 2))
    hx = 1e-6
    fd = (cdf(xs + hx) - cdf(xs - hx)) / (2 * hx)
    sx = torch.special.ndtri(xs)
    an = cl.cond_s(sx, y)[0] / (torch.exp(-0.5 * sx**2) / math.sqrt(2 * math.pi))
    e = float(((an - fd) / fd).abs().max())
    ok &= e < 1e-6
    det.append(f"Clayton conditional vs d/dx conditional CDF rel err {e:.1e}")
    gm = C.Gumbel(1.8, 3, DEV)
    tt = ((-torch.log(y)) ** 1.8).sum()
    cdfg = lambda x: torch.exp(
        gm._log_g(2, torch.log(tt + (-torch.log(x)) ** 1.8)) - (tt + (-torch.log(x)) ** 1.8) ** gm.alpha - gm._log_g(2, torch.log(tt)) + tt**gm.alpha
    )
    fdg = (cdfg(xs + hx) - cdfg(xs - hx)) / (2 * hx)
    ang = gm.cond_s(sx, y)[0] / (torch.exp(-0.5 * sx**2) / math.sqrt(2 * math.pi))
    eg = float(((ang - fdg) / fdg).abs().max())
    ok &= eg < 1e-6
    det.append(f"Gumbel conditional vs d/dx conditional CDF rel err {eg:.1e}")
    from scipy.special import stdtr, stdtrit

    tc, _ = C.make_copula("t3", "SUM", 2, 0.5, DEV)
    ty = stdtrit(3, _np(y))[0]
    Q = ty @ np.linalg.solve(tc.R[1:, 1:], ty)
    mu = ty @ _np(tc.beta)
    sc = tc.sigma * math.sqrt((3 + Q) / 5)
    cdft = lambda x: stdtr(5, (stdtrit(3, x) - mu) / sc)
    xn = _np(xs)
    fdt = (cdft(xn + hx) - cdft(xn - hx)) / (2 * hx)
    ant = _np(tc.cond_s(sx, y)[0]) / (np.exp(-0.5 * _np(sx) ** 2) / math.sqrt(2 * math.pi))
    et = float(np.abs((ant - fdt) / fdt).max())
    ok &= et < 1e-6
    det.append(f"t3 conditional vs d/dx conditional CDF rel err {et:.1e}")
    # closed-form t CDF / quantile (on the device) vs 50-digit mpmath, independent of the scipy version. Doubles near 1
    # are 1.1e-16 apart, so 1 - F cannot carry a relative error: relative accuracy is checked on the lower tail, the
    # upper side in absolute terms, and the quantile through the tail probability it reproduces.
    el = eu = eq = 0.0
    xg = np.logspace(-6, 6, 300)
    ug = np.concatenate([np.logspace(-14, np.log10(0.5), 200), 1 - np.logspace(-14, np.log10(0.5), 200)])
    with mpmath.workdps(50):

        def t_tail(nu, x):  # P(T > |x|)
            a = mpmath.mpf(float(abs(x)))
            return mpmath.betainc(nu / 2, mpmath.mpf(1) / 2, 0, nu / (nu + a * a), regularized=True) / 2

        for nu in (3, 10):
            tail = [t_tail(nu, x) for x in xg]
            lower = _np(C.t_cdf(torch.as_tensor(-xg, dtype=DT, device=DEV), nu))
            upper = _np(C.t_cdf(torch.as_tensor(xg, dtype=DT, device=DEV), nu))
            el = max(el, *(float(abs(mpmath.mpf(float(f)) / t - 1)) for f, t in zip(lower, tail)))
            eu = max(eu, *(float(abs(mpmath.mpf(float(f)) - (1 - t))) for f, t in zip(upper, tail)))
            q = _np(C.t_ppf(torch.as_tensor(ug, dtype=DT, device=DEV), nu))
            ok &= bool((np.sign(q) * (ug - 0.5) >= 0).all())
            p = [min(mpmath.mpf(float(u)), 1 - mpmath.mpf(float(u))) for u in ug]
            eq = max(eq, *(float(abs(t_tail(nu, x) / b - 1)) for x, b in zip(q, p)))
    ok &= el < 1e-12 and eu < 1e-15 and eq < 1e-12
    det.append(
        f"t CDF (nu = 3, 10) vs mpmath: lower tail rel err {el:.1e}, upper side abs err {eu:.1e}; quantile: rel err of its tail probability {eq:.1e}"
    )
    return bool(ok), "; ".join(det)


# ---------------------------------------------------------------- M4
def M4(M):
    """k=2 MC engine fed with the d=3 coefficients vs the d=3 quadrature results."""
    from scipy.special import ndtr

    z = np.load(os.path.join(ROOT, "three_input_study", "results", "coeffs.npz"))
    old = pd.read_csv(os.path.join(ROOT, "three_input_study", "results", "results.csv"))
    cfg = small_cfg(M, R=16)
    grid = Grid(801, 8.0, 192, 6, DEV)
    names = ("S2", "S3a", "S3b", "S3c", "S3d")
    bel = beliefs_for(2, M, 21, names)
    metrics = ["d_denom", "d_neg", "d_model_uncal", "d_model_cal", "d_total", "d_dc"]
    worst, n_cmp, n_flag, viol, flagmis = 0.0, 0, 0, [], []
    ql_rows = []
    for typ in ("SUM", "CONTRAST"):
        for rho in (0.2, 0.3, 0.4, 0.6):
            a = z[f"{typ}_{rho}_a"]
            model = model_from_dense3(torch.as_tensor(a))
            model = Model(2, 6, "triple", {kk: (None if v is None else v.to(DEV)) for kk, v in model.blocks.items()})
            R, _ = C.coupled_R(typ, 2, rho)
            cop = C.Gaussian(R, DEV)
            rows = evaluate_case(model, cop, bel, grid, cfg, np.random.default_rng(1), [2, 4, 6], DEV)
            new = pd.DataFrame(rows)
            for _, r in new.iterrows():
                o = old[(old.type == typ) & np.isclose(old.rho_S, rho) & (old.m == r.m) & (old.scenario == r.belief) & np.isclose(old.eps, r.eps)]
                if len(o) != 1:
                    continue
                o = o.iloc[0]
                if int(o.defined) != int(r.defined):
                    flagmis.append(
                        f"{typ} {rho} m={r.m} {r.belief}: d3 defined={int(o.defined)} new={int(r.defined)} "
                        f"(d3 min_D={o.min_D:.4f}, new minD_supp={r.minD_supp:.4f})"
                    )
                for mt in metrics:
                    if pd.isna(o[mt]) or pd.isna(r[mt]):
                        continue
                    tol = max(3 * r[mt + "_se"], 1e-3)
                    d = abs(r[mt] - o[mt])
                    if o.quadrature_limited == 1 and mt in ("d_total", "d_model_uncal", "d_model_cal"):
                        n_flag += 1
                        ql_rows.append((typ, rho, int(r.m), r.belief, r.eps, mt, o[mt], r[mt], r[mt + "_se"]))
                        continue
                    n_cmp += 1
                    worst = max(worst, d / tol)
                    if d > tol:
                        viol.append(f"{typ} {rho} m={r.m} {r.belief} eps={r.eps} {mt}: d3 {o[mt]:.5f} new {r[mt]:.5f} (tol {tol:.1e})")
    # quadrature-limited GOLD rows: deterministic reference on a 3000-node Gauss-Legendre x-grid
    ref_viol, ref_det = [], []
    xs, wx = np.polynomial.legendre.leggauss(3000)
    xs, wx = (xs + 1) / 2, wx / 2
    sgrid = np.linspace(-8, 8, 401)
    tw = np.full(401, sgrid[1] - sgrid[0])
    tw[[0, -1]] /= 2
    phs = np.exp(-0.5 * sgrid**2) / math.sqrt(2 * math.pi)
    from scipy.stats import beta as bd

    for typ, rho, m, bel_n, _, mt, ov, nv, nse in ql_rows:
        if mt != "d_total":
            continue
        R, _ = C.coupled_R(typ, 2, rho)
        sig = R[0, 1:]
        bt = np.linalg.solve(R[1:, 1:], sig)
        s2 = 1 - sig @ bt
        yv = ndtr(sgrid)
        if bel_n == "S2":
            q = np.ones((401, 401))
        else:
            ab = np.array(CFG["beliefs"][bel_n] * 2)[:2]
            q = np.outer(bd.pdf(yv, *ab[0]), bd.pdf(yv, *ab[1 % len(ab)]))
        Wq = (q * np.outer(tw * phs, tw * phs)).ravel()
        mu = (bt[0] * sgrid[:, None] + bt[1] * sgrid[None, :]).ravel()
        sx = np.asarray(torch.special.ndtri(torch.tensor(xs)))
        gold = np.array([np.exp(-0.5 * (v - mu) ** 2 / s2 + 0.5 * v * v) @ Wq for v in sx]) / math.sqrt(s2)
        a = z[f"{typ}_{rho}_a"][: m + 1, : m + 1, : m + 1]
        Fy = np.asarray(legendre(torch.tensor(np.polynomial.legendre.leggauss(192)[0] / 2 + 0.5), m))
        wy = np.polynomial.legendre.leggauss(192)[1] / 2
        if bel_n == "S2":
            bq = np.zeros((m + 1, m + 1))
            bq[0, 0] = 1
        else:
            yg = np.polynomial.legendre.leggauss(192)[0] / 2 + 0.5
            qy = np.outer(bd.pdf(yg, *ab[0]), bd.pdf(yg, *ab[1 % len(ab)]))
            bq = Fy.T @ (qy * np.outer(wy, wy)) @ Fy
        num = np.einsum("ijk,jk->i", a, bq)
        cd = np.asarray(legendre(torch.tensor(xs), m)) @ (num / num[0])
        ref = 0.5 * wx @ np.abs(cd - gold)
        tol = max(3 * nse, 1e-3)
        ref_det.append(f"{typ} {rho} m={m} {bel_n}: d3 {ov:.4f}, new {nv:.4f}, 3000-node ref {ref:.4f}")
        if abs(nv - ref) > tol:
            ref_viol.append(ref_det[-1])
    ok = not viol and not ref_viol
    det = (
        f"{n_cmp} TV comparisons, worst |diff|/tol {worst:.2f}, {len(viol)} violations; {n_flag} GOLD-based values in "
        f"d3 quadrature-limited rows re-checked against a 3000-node x-grid reference: {len(ref_viol)} violations; "
        f"defined-flag mismatches: {len(flagmis)}"
    )
    return ok, det + (" | " + " | ".join(viol[:10]) if viol else "") + (" | REF " + " | ".join(ref_det[:8]) if ref_det else "") + (
        " | FLAGS " + " | ".join(flagmis[:6]) if flagmis else ""
    )


# ---------------------------------------------------------------- M5-M7
def _case(k, typ, rho, basis, M, names, n_est=200000, flat=False, pop=False):
    R, _ = C.coupled_R(typ, k, rho)
    cop, (model, _) = estimate_model(R, n_est, basis, seed=7)
    if pop:
        model = population_model(R, 6, basis, 48, DEV)
    if flat:
        model = model.with_flat_D()
    bel = beliefs_for(k, M, 9, names, cop)
    dens = {}
    rows = evaluate_case(
        model, cop, bel, Grid(801, 8.0, 192, 6, DEV), small_cfg(M, starts=500, steps=200), np.random.default_rng(2), [2, 4, 6], DEV, dens
    )
    return model, cop, bel, pd.DataFrame(rows), dens


def M5():
    worst = 0.0
    for k, basis, rho in ((2, "triple", 0.4), (4, "triple", 0.3), (8, "pair", 0.2)):
        *_, dens = _case(k, "SUM", rho, basis, 2**15, ("S1", "S2", "S3a", "S3b", "S3c", "S3d", "S4"), flat=True)
        for d in dens.values():
            worst = max(worst, float((d["cd_c"] - d["eu_c"]).abs().max()))
    return worst < 1e-10, f"flat-D control: max |CONST_DENOM - EXACT_UNCAL| coefficient diff {worst:.2e} over all beliefs, m, k in (2,4,8)"


def M6():
    worst, n = 0.0, 0
    for k, basis, typ, rho in ((2, "triple", "CONTRAST", 0.2), (4, "pair", "SUM", 0.2), (4, "triple", "CONTRAST", 0.1)):
        model, _, bel, _, dens = _case(k, typ, rho, basis, 2**15, ("S1", "S2", "S3a", "S3c", "S3d", "S4"))
        for (b, m), d in dens.items():
            if not d["defined"] or d["n_drop"]:
                continue
            Y = bel["S2" if b == "S1" else b]["Y"]
            A = model.sliced(m).A(input_basis(Y, m))
            w = A[:, 0] if b == "S1" else torch.ones(A.shape[0], dtype=DT, device=DEV)
            E = lambda v: (w[:, None] * v).sum(0) / w.sum() if v.dim() == 2 else (w * v).sum() / w.sum()
            ratio = A / A[:, :1]
            cov = E(ratio * A[:, :1]) - E(ratio) * E(A[:, 0])
            pred = cov / E(A[:, 0])
            worst = max(worst, float((d["cd_c"] - d["eu_c"] - pred).abs().max()))
            n += 1
    return worst < 1e-10, f"in-sample covariance identity: max residual {worst:.2e} over {n} defined (belief, m, case) combinations"


def M7():
    det, ok = [], True
    _, _, _, df, dens = _case(4, "CONTRAST", 0.2, "triple", 2**16, ("S1", "S2", "S3a", "S3b", "S3c", "S3d", "S4"))
    grid = Grid(801, 8.0, 192, 6, DEV)
    c0 = max(max(abs(float(d["cd_c"][0]) - 1), abs(float(d["eu_c"][0]) - 1) if d["defined"] else 0) for d in dens.values())
    si = max(float(np.nanmax(np.abs(df[c] - 1))) for c in ("int_cd", "int_eu"))
    calg = max(float((d["CALg"] * grid.wg).sum(-1).sub(1).abs().max()) for d in dens.values() if d["q_valid"])
    cals = float(np.nanmax(np.abs(df.int_cal - 1)))
    gi = float(np.nanmax(np.abs(df.int_gold - 1)))
    ok &= c0 < 1e-8 and si < 1e-8 and calg < 1e-8 and gi < 1e-6
    det.append(
        f"coefficient integrals |c0-1| {c0:.1e}; s-grid integrals CONST_DENOM/EXACT_UNCAL {si:.1e}; EXACT_CAL under "
        f"its 192-node normaliser rule {calg:.1e} (on the 801-node s-grid {cals:.1e}, reported per row as int_cal); "
        f"GOLD {gi:.1e}"
    )
    _, _, _, df0, _ = _case(4, "SUM", 0.0, "pair", 2**20, ("S4",), pop=True)
    tvu = float(df0.d_gold_unif.max())
    ok &= tvu <= 1e-3
    det.append(f"S4, population coefficients, rho_S=0: TV(GOLD, uniform) {tvu:.1e} (M=2^20)")
    cok, cdet = COP()
    ok &= cok
    det.append(cdet)
    return bool(ok), "; ".join(det)


# ---------------------------------------------------------------- M8
def M8():
    det, ok = [], True
    from scipy.stats import beta as bd

    grid = Grid(801, 8.0, 192, 6, DEV)
    xg, wg = gauss_legendre(400, DEV)
    worst = 0.0
    for (a1, b1), (a2, b2) in ((((2, 5), (3, 3))), (((4, 2), (2, 4))), (((2, 2), (3, 2)))):
        cof = lambda a, b: (legendre(xg, 6) * torch.as_tensor(bd.pdf(_np(xg), a, b), device=DEV)[:, None] * wg[:, None]).sum(0)
        c1, c2 = cof(a1, b1), cof(a2, b2)
        xs = np.linspace(0, 1, 200001)  # reference in x: |polynomial difference| on a very fine grid
        diff = _np(legendre(torch.as_tensor(xs, device=DEV), 6) @ (c1 - c2))
        exact = 0.5 * np.trapezoid(np.abs(diff), xs) if hasattr(np, "trapezoid") else 0.5 * np.trapz(np.abs(diff), xs)  # noqa: NPY201  numpy 1.26 on the cluster
        tvs = float(grid.tv(grid.poly_s(c1), grid.poly_s(c2)))
        worst = max(worst, abs(tvs - exact))
    ok &= worst < 1e-6
    det.append(f"TV in s vs x (polynomial Beta densities): max diff {worst:.1e}")
    g2 = Grid(1601, 8.0, 192, 6, DEV)
    wd = 0.0
    for typ in ("SUM", "CONTRAST"):
        for rho in (0.3, 0.6, 0.9):
            R, _ = C.coupled_R(typ, 2, rho)
            cop = C.Gaussian(R, DEV)
            bel = beliefs_for(2, 2**14, 4, ("S2", "S3c"))
            for b in bel.values():
                p1 = cop.cond_s(grid.s, b["Y"]).mean(0)
                p2 = cop.cond_s(g2.s, b["Y"]).mean(0)
                c = torch.zeros(7, dtype=DT, device=DEV)
                c[0] = 1
                c[2] = 0.3
                wd = max(wd, abs(float(grid.tv(grid.poly_s(c), p1)) - float(g2.tv(g2.poly_s(c), p2))))
    ok &= wd < 1e-6
    det.append(f"spiky GOLD (k=2, rho_S<=0.9): TV change 801->1601 s-nodes max {wd:.1e}")
    return bool(ok), "; ".join(det)


# ---------------------------------------------------------------- M9
def M9():
    z = np.load(os.path.join(ROOT, "three_input_study", "results", "coeffs.npz"))
    md = dict(CFG["numerics"]["mind"])
    worst, det = 0.0, []
    g = torch.linspace(0, 1, 401, dtype=DT, device=DEV)
    Yg = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    for typ in ("SUM", "CONTRAST"):
        for rho in (0.2, 0.5, 0.8):
            for m in (2, 4, 6):
                a = z[f"{typ}_{rho}_a"][: m + 1, : m + 1, : m + 1]
                mod = model_from_dense3(torch.as_tensor(a))
                mod = Model(2, m, "triple", {kk: (None if v is None else v.to(DEV)) for kk, v in mod.blocks.items()})
                gmin = float(mod.D(Yg).min())
                opt = min_D(mod, np.random.default_rng(0), md, device=DEV)[0]
                worst = max(worst, abs(opt - gmin))
    det.append(f"k=2 optimiser vs 401x401 endpoint-inclusive grid: max |diff| {worst:.1e}")
    ok = worst < 1e-3
    e1 = 0.0
    for typ in ("SUM", "CONTRAST"):
        for rho in (0.2, 0.5, 0.8):
            a = z[f"{typ}_{rho}_a"][:2, :2, :2]
            mod = model_from_dense3(torch.as_tensor(a))
            mod = Model(2, 1, "triple", {kk: (None if v is None else v.to(DEV)) for kk, v in mod.blocks.items()})
            opt = min_D(mod, np.random.default_rng(0), md, device=DEV)[0]
            e1 = max(e1, abs(opt - (1 - 3 * abs(a[0, 1, 1]))))
    for a1 in (0.0, 0.2, -0.4):
        b = {"c": torch.tensor([1.0, 0.0], device=DEV), "W1": torch.tensor([[[a1]], [[0.0]]], dtype=DT, device=DEV), "W2": None, "W3": None}
        mod = Model(1, 1, "pair", b)
        opt = min_D(mod, np.random.default_rng(0), md, device=DEV)[0]
        e1 = max(e1, abs(opt - (1 - math.sqrt(3) * abs(a1))))
    ok &= e1 < 0.01
    det.append(f"m=1 closed forms (k=2: 1-3|a_011|; k=1: 1-sqrt(3)|a_01|): max |diff| {e1:.1e}")
    # minimum over supp(q) for the Beta beliefs: not worse than a 2001x2001 grid restricted to supp(q) by > 1e-3
    g2 = torch.linspace(0, 1, 2001, dtype=DT, device=DEV)
    Y2 = torch.stack(torch.meshgrid(g2, g2, indexing="ij"), -1).reshape(-1, 2)
    worst_c, lower = -1.0, 0
    for typ in ("SUM", "CONTRAST"):
        for rho in (0.3, 0.4, 0.5):
            for m in (4, 6):
                a = z[f"{typ}_{rho}_a"][: m + 1, : m + 1, : m + 1]
                mod = model_from_dense3(torch.as_tensor(a))
                mod = Model(2, m, "triple", {kk: (None if v is None else v.to(DEV)) for kk, v in mod.blocks.items()})
                Dg = mod.D(Y2)
                for name in ("S3a", "S3b", "S3c", "S3d"):
                    ab = np.array((CFG["beliefs"][name] * 2)[:2], float)
                    lq, lmax = beta_logq(ab)
                    thr = lmax + math.log(CFG["supp_rel"])
                    gmin = float(Dg[lq(Y2) >= thr].min())
                    opt = min_D(mod, np.random.default_rng(0), md, None, lq, thr, DEV)[0]
                    worst_c = max(worst_c, opt - gmin)
                    lower += opt < gmin
    ok &= worst_c <= 1e-3
    det.append(f"min over supp(q), Beta beliefs (48 cases): optimiser - 2001^2 grid <= {worst_c:.1e} (optimiser lower than the grid in {lower})")
    return bool(ok), "; ".join(det)


# ---------------------------------------------------------------- M10
def M10():
    det, ok = [], True
    worst, red = 1.0, 0
    for k in (2, 4, 8, 16, 32):
        for typ in ("SUM", "CONTRAST", "EXCH"):
            for rho in CFG["rho_S"]:
                _, info = C.coupled_R(typ, k, rho)
                worst = min(worst, info["min_eig"])
                red += info["c_reductions"]
    ok &= worst > C.PD_TOL
    det.append(f"all sweep matrices PD (min eigenvalue {worst:.3e}), c reductions needed: {red}")
    import logging

    logs = []
    h = logging.Handler()
    h.emit = lambda r: logs.append(r.getMessage())
    C.log.addHandler(h)
    _, info = C.coupled_R("SUM", 4, 0.0, c_max=1.0, c_scale=2.0)
    C.log.removeHandler(h)
    ok &= info["c_reductions"] > 0 and info["min_eig"] > C.PD_TOL and len(logs) == info["c_reductions"]
    det.append(f"forced case: c {info['c_requested']:.3f} -> {info['c']:.3f} in {info['c_reductions']} logged steps")
    try:
        C.Gaussian(np.array([[1, 0.99, 0.99], [0.99, 1, -0.99], [0.99, -0.99, 1]]), DEV)
        ok = False
        det.append("non-PD matrix NOT rejected")
    except AssertionError:
        det.append("non-PD matrix rejected")
    rng = np.random.default_rng(0)
    for eta, d in ((1, 5), (4, 9)):
        Rs = [C.lkj(d, eta, rng) for _ in range(3000)]
        r = np.array([R[0, 1] for R in Rs])
        v = r.var()
        target = 1 / (2 * eta + d - 1)
        ok &= abs(v / target - 1) < 0.1 and min(C.min_eig(R) for R in Rs) > 0
        det.append(f"LKJ(eta={eta}, d={d}): Var(r) {v:.4f} vs {target:.4f}")
    return bool(ok), "; ".join(det)


# ---------------------------------------------------------------- M11
def _tiny_spec(k, basis, seed, cases):
    s = dict(
        exp="gauss",
        k=k,
        family="gauss",
        type="SUM",
        basis=basis,
        seed=seed,
        n_est=100000,
        variant="base",
        key=[1, k, seed],
        cases=[{"rho_S": r} for r in cases],
        override={"numerics": {"M": 8192, "R": 8, "chunk_mc": 1024, "mind": {"starts": 300, "steps": 100}}},
    )
    return with_id(s)


def M11():
    det, ok = [], True
    spec = _tiny_spec(4, "triple", 0, (0.1, 0.4))
    d1, _ = run_job(spec, CFG, DEV)
    d2, _ = run_job(spec, CFG, DEV)
    num = [c for c in d1.columns if d1[c].dtype.kind == "f" and "runtime" not in c]
    a, b = d1[num].to_numpy(), d2[num].to_numpy()
    rel = np.nanmax(np.abs(a - b) / np.maximum(np.abs(a), 1e-300)) if a.size else 0.0
    same_nan = bool((np.isnan(a) == np.isnan(b)).all())
    ok &= rel <= 1e-12 and same_nan
    det.append(f"same job twice on {DEV}: max relative diff {rel:.1e}, NaN pattern identical: {same_nan}")
    # kill-and-resume through the launcher vs a clean run
    specs = [_tiny_spec(2, "pair", s, (0.2, 0.5)) for s in range(3)]
    merged = []
    for mode in ("clean", "killed"):
        tmp = tempfile.mkdtemp(prefix=f"hcr_m11_{mode}_")
        env = dict(os.environ, HCR_RESULTS=tmp)
        jl = os.path.join(tmp, "jobs_gauss.jsonl")
        with open(jl, "w") as f:
            for s in specs:
                f.write(json.dumps(s) + "\n")
        cmd = [sys.executable, "-m", "densprop", "launch", "--worker", "--device", DEV, "--jobs", jl, "--stale", "0.05"]
        if mode == "killed":
            p = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            jd = os.path.join(tmp, "jobs", "gauss")
            t0 = time.time()
            while time.time() - t0 < 600 and not (os.path.isdir(jd) and len([f for f in os.listdir(jd) if f.endswith(".parquet")]) >= 1):
                time.sleep(0.5)
            time.sleep(1.0)
            p.kill()
            p.wait()
            subprocess.run(["pkill", "-f", tmp], check=False)  # the job subprocess of the killed worker
            n_claims = len(os.listdir(os.path.join(tmp, "claims"))) if os.path.isdir(os.path.join(tmp, "claims")) else 0
            time.sleep(4)  # older than --stale
            det.append(f"killed worker left {n_claims} claim(s)")
        subprocess.run(cmd, env=env, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([sys.executable, "-m", "densprop", "merge", "gauss"], env=env, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        merged.append(pd.read_parquet(os.path.join(tmp, "merged", "gauss.parquet")))
        shutil.rmtree(tmp, ignore_errors=True)
    num = [c for c in merged[0].columns if merged[0][c].dtype.kind == "f" and "runtime" not in c]
    a, b = merged[0][num].to_numpy(), merged[1][num].to_numpy()
    same = a.shape == b.shape and bool(np.array_equal(np.isnan(a), np.isnan(b))) and bool(np.nanmax(np.abs(a - b)) == 0 if a.size else True)
    ok &= same
    det.append(f"kill-and-resume merged results identical to a clean run: {same} ({len(merged[0])} rows)")
    return bool(ok), "; ".join(det)


# ---------------------------------------------------------------- M12
def M12():
    det, ok = [], True
    R, _ = C.coupled_R("SUM", 2, 0.3)
    R2, _ = C.coupled_R("SUM", 2, 0.31)
    a011, a110, se011, se110, diff = [], [], [], [], []
    for s in range(50):
        rng = rng_for(CFG, [1, 2, s], "est")
        base = C.base_variates(rng, 100000, 3, ("Z",))
        out = []
        for RR in (R, R2):
            est = Estimator(2, 2, "pair", DEV)
            est.add(C.Gaussian(RR, DEV).uniforms(base))
            out.append(est.model())
        (m1, s1), (m2, _) = out
        a011.append(float(m1.blocks["W2"][0, 0, 0, 0]))
        se011.append(float(s1["W2"][0, 0, 0, 0]))
        a110.append(float(m1.blocks["W1"][1, 0, 0]))
        se110.append(float(s1["W1"][1, 0, 0]))
        diff.append(float(m2.blocks["W2"][0, 0, 0, 0] - m1.blocks["W2"][0, 0, 0, 0]))
    from scipy.stats import chi2

    lo, hi = chi2.ppf(0.005, 49) / 49, chi2.ppf(0.995, 49) / 49
    for nm, v, se in (("a_(0,1,1)", a011, se011), ("a_(1,1,0)", a110, se110)):
        ratio = np.var(v, ddof=1) / np.mean(np.square(se))
        ok &= lo <= ratio <= hi
        det.append(f"{nm}: across-seed variance / mean SE^2 = {ratio:.3f} (99% band [{lo:.2f}, {hi:.2f}], 50 seeds)")
    crn = np.std(diff, ddof=1) / np.mean(se011)
    ok &= crn < 0.2
    det.append(f"common random numbers: sd of the rho_S 0.30->0.31 difference = {crn:.3f} x SE")
    return bool(ok), "; ".join(det)


TESTS_PRE = {"M1": M1, "M2": M2, "M3": M3, "M4": None, "M5": M5, "M6": M6, "M7": M7, "M8": M8, "M9": M9, "M10": M10, "M11": M11, "M12": M12}


def run_pre(only: list[str] | None, m4_samples: int | None, log=print) -> dict:
    """M1-M12 (or the selected ones): {name: {pass, detail, device, seconds}}; a crash is a failure with its message."""
    results = {}
    for name in only or list(TESTS_PRE):
        t0 = time.time()
        try:
            if name == "M4":
                ok, det = M4(m4_samples or (2**20 if DEV.startswith("cuda") else 2**18))
            else:
                ok, det = TESTS_PRE[name]()
        except Exception as e:  # a crash is a failure, recorded with its message
            import traceback

            ok, det = False, f"ERROR {type(e).__name__}: {e} | {traceback.format_exc()[-1500:]}"
        results[name] = {"pass": bool(ok), "detail": det, "device": DEV, "seconds": round(time.time() - t0, 1)}
        log(f"{name}: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s) {det}")
    return results
