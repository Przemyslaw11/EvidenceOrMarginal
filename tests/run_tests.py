"""Validation tests. `python tests/run_tests.py pre` (T1-T8) before the sweep; `post` (T9,T10) after."""
import sys, os, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + "/src")
import numpy as np, pandas as pd
from hcr import *
from sweep import run_sweep

cfg = load_cfg()
OUT = ROOT + "/tests/test_results.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def record(name, ok, detail):
    res[name] = {"pass": bool(ok), "detail": detail}
    print(f"{name}: {'PASS' if ok else 'FAIL'}  {detail}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


def pre():
    # T1
    u, w = gl(64); F = basis(u, 8); Gm = F.T @ (w[:, None] * F)
    e1 = np.abs(Gm - np.eye(9)).max()
    x = np.linspace(0.01, 0.99, 7)
    ex = [np.ones_like(x), np.sqrt(3) * (2 * x - 1), np.sqrt(5) * (6 * x**2 - 6 * x + 1),
          np.sqrt(7) * (20 * x**3 - 30 * x**2 + 12 * x - 1), 3 * (70 * x**4 - 140 * x**3 + 90 * x**2 - 20 * x + 1)]
    e2 = max(np.abs(basis(x, 4)[:, i] - ex[i]).max() for i in range(5))
    record("T1", e1 < 1e-12 and e2 < 1e-12, f"gram max err {e1:.2e}; explicit formulas max err {e2:.2e}")
    # estimation at reduced n for T2,T3,T8 and engine tests
    n = 400000
    types, rhos = cfg["types"], [0.0, 0.3, 0.4, 0.8]
    co = estimate_coeffs(cfg, types, rhos, n, cfg["seed"], chunk=100000)
    ok, worst = True, 0
    for (t, r), (a, se) in co.items():
        z = []
        ok &= abs(a[0, 0, 0] - 1) < 1e-12
        z.append(abs(a[0, 1, 1] - r) / se[0, 1, 1])
        for i in range(1, 7):
            for idx in [(i, 0, 0), (0, i, 0), (0, 0, i)]:
                z.append(abs(a[idx]) / se[idx])
        worst = max(worst, max(z))
    record("T2", ok and worst <= 5, f"a000=1; max |dev|/SE = {worst:.2f} (<=5) over a_011 vs rho_S and single-index coeffs, n={n}")
    # T3
    mx = 0
    for (t, r), (a, se) in co.items():
        a1 = a[:2, :2, :2]
        Ff = basis(np.linspace(0, 1, 401), 1)
        mind = model_D(Ff, a1[0]).min()
        mx = max(mx, abs(mind - (1 - 3 * abs(a[0, 1, 1]))))
    record("T3", mx < 0.01, f"max |minD - (1-3|a011|)| = {mx:.4f}")
    G = Grid(cfg["nodes"], cfg["mmax"])
    sp = scenario_specs(cfg)
    # T8
    worst1, worst2 = 0, 0
    for t in types:
        R = corr_matrix(t, 0.0, cfg); Ct = ctrue_grid(G, R)
        g = gold_for(G, R, Ct, {"S2": ("flat",)}, cfg)["S2"][0]
        worst1 = max(worst1, np.abs(g - 1).max()); worst2 = max(worst2, 0.5 * G.w @ np.abs(g - 1))
    record("T8", worst1 <= 1e-2 and worst2 <= 2e-3, f"max|GOLD-1|={worst1:.2e} (<=1e-2); TV(GOLD,uniform)={worst2:.2e} (<=2e-3)")
    # engine tests
    nogold = lambda s: {k: (np.ones(G.n), 0.0, "GL") for k in s}
    # T4 control: flat D
    a = co[("SUM", 0.4)][0][:5, :5, :5].copy()
    a[0] = 0; a[0, 0, 0] = 1
    R = corr_matrix("SUM", 0.4, cfg)
    rows, dens = evaluate(G, a, 4, R, sp, cfg, [0.1], nogold, want_dens=True)
    e4 = max(np.abs(dens[(k, 0.1)]["duda"] - dens[(k, 0.1)]["uncal"]).max() for k in sp)
    record("T4", e4 < 1e-10, f"flat-D control: max |DUDA-EXACT_UNCAL| = {e4:.2e} over all {len(sp)} scenarios")
    # T5, T6, T7 on a defined case
    a = co[("SUM", 0.3)][0][:5, :5, :5]
    R = corr_matrix("SUM", 0.3, cfg); Ct = ctrue_grid(G, R)
    gf = lambda s: gold_for(G, R, Ct, s, cfg)
    rows, dens = evaluate(G, a, 4, R, sp, cfg, [0.1], gf, want_dens=True)
    df = pd.DataFrame(rows)
    un = dens[("S1", 0.1)]["uncal"]
    marg = G.F[:, :5] @ a[:, 0, 0]
    e5 = np.abs(un - marg).max()
    record("T5", df.loc[df.scenario == "S1", "defined"].iloc[0] == 1 and e5 < 1e-8, f"S1: max|EXACT_UNCAL - sum a_i00 f_i| = {e5:.2e}")
    d6 = df[df.defined == 1]
    e6a = max(abs(d6.int_duda - 1).max(), abs(d6.int_uncal - 1).max())
    e6b = d6.gold_norm_err.max()
    record("T6", e6a < 1e-8 and e6b < 1e-3, f"max|int DUDA/UNCAL -1| = {e6a:.2e} (<=1e-8); max GOLD norm err = {e6b:.2e} (<=1e-3, fallback applied where needed)")
    # T7 independent covariance identity
    F = G.F[:, :5]; D = model_D(F, a[0]); N = model_N(G, a, 4)
    worst = 0
    for nm in ["S2", "S3a", "S3b", "S3c", "S1"]:
        q = q_on(("poly", a[0]) if nm == "S1" else sp[nm], G.u); Wq = q * G.W2
        ratio = N / D
        EqR = np.einsum("xab,ab->x", ratio, Wq); EqD = (D * Wq).sum()
        cov = np.einsum("xab,ab->x", ratio * D, Wq) - EqR * EqD
        lhs = dens[(nm, 0.1)]["duda"] - dens[(nm, 0.1)]["uncal"]
        worst = max(worst, np.abs(lhs - cov / EqD).max())
    record("T7", worst < 1e-10, f"max |(DUDA-EXACT_UNCAL) - Cov_q(N/D,D)/E_q[D]| = {worst:.2e}")


def post():
    r192 = pd.read_csv(ROOT + "/results/results.csv")
    r256 = pd.read_csv(ROOT + "/results/results_n256.csv")
    keys = ["type", "rho_S", "m", "scenario", "eps"]
    mg = r192.merge(r256, on=keys, suffixes=("", "_256"))
    tvc = ["d_denom", "d_neg", "d_model_uncal", "d_model_cal", "d_total", "d_dc"]
    def chg(d):
        vals = [np.nanmax(np.abs(d[c] - d[c + "_256"])) if d[c].notna().any() else 0.0 for c in tvc]
        return float(max(vals)), tvc[int(np.argmax(vals))]
    lo = mg[mg.rho_S <= 0.6 + 1e-9]; hi = mg[mg.rho_S > 0.65]
    (cl, wcl), (ch, wch) = chg(lo), chg(hi)
    (cl_ok, _), (ch_ok, _) = chg(lo[lo.quadrature_limited != 1]), chg(hi[hi.quadrature_limited != 1])
    und = int(((mg.defined != mg.defined_256)).sum())
    nql = int((lo.quadrature_limited == 1).sum())
    record("T9", cl < 1e-4,
           f"STRICT criterion (all rows, rho_S<=0.6): max |TV change| 192->256 = {cl:.2e} in {wcl} (<1e-4 required); "
           f"excluding rows flagged quadrature_limited (GOLD x-grid singular edges, {nql} such rows at rho_S<=0.6): {cl_ok:.2e}; "
           f"rho_S>=0.7 observed (reported): {ch:.2e} all rows, {ch_ok:.2e} excluding flagged; defined-flag changes: {und}. "
           f"Deviation: only GOLD-based TVs (d_total, d_model_*) of CONTRAST/S2 rows flagged quadrature_limited exceed 1e-4; see gold_check.csv")
    # T10 reduced re-run
    n = 200000
    out = []
    for _ in range(2):
        co = estimate_coeffs(cfg, cfg["types"], [0.2, 0.6], n, cfg["seed"], chunk=100000)
        df, _ = run_sweep(cfg, co, 96, rhos=[0.2, 0.6], ms=[4], log=lambda *a: None)
        out.append(df.drop(columns=["runtime_s"], errors="ignore"))
    same = out[0].equals(out[1])
    record("T10", same, f"same-seed reduced re-run (n={n}, rho in {{0.2,0.6}}, m=4, 96 nodes) identical excl. runtime: {same}; full results.csv not re-run (see ASSUMPTIONS 9)")


if __name__ == "__main__":
    {"pre": pre, "post": post}[sys.argv[1]]()
