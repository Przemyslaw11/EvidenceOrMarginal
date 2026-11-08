"""Stage 7: results/report.txt. Every number is computed here from results/*.csv and tests/test_results.json."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from hcr import ROOT, load_cfg

cfg = load_cfg()
d = pd.read_csv(ROOT + "/results/results.csv")
thr = pd.read_csv(ROOT + "/results/thresholds.csv")
gc = pd.read_csv(ROOT + "/results/gold_check.csv")
tay = pd.read_csv(ROOT + "/results/taylor.csv")
trn = pd.read_csv(ROOT + "/results/training.csv")
with open(ROOT + "/tests/test_results.json") as fh:
    tests = json.load(fh)
e01 = d[d.eps == 0.01]
E0 = d[d.eps == d.eps.min()]


def row(df, typ, m, sc, rho):
    g = df[(df.type == typ) & (df.m == m) & (df.scenario == sc) & np.isclose(df.rho_S, rho)]
    return g.iloc[0]


def f(x, n=3):
    return "n/d" if pd.isna(x) else f"{x:.{n}f}"


def p(x, n=1):
    return "n/d" if pd.isna(x) else f"{100 * x:.{n}f}%"


def th(x):
    return "—" if pd.isna(x) else f"{x:.2f}"


# ------------------------------------------------------------------ threshold tables
def thr_table(m_list):
    lines = ["| type | m | belief | d_denom≥1% | d_denom≥5% | var.err≥5% | var.err≥20% |", "|---|---|---|---|---|---|---|"]
    for _, r in thr[thr.m.isin(m_list)].sort_values(["type", "m", "scenario"]).iterrows():
        lines.append(f"| {r.type} | {r.m} | {r.scenario} | {th(r.denom_1pct)} | {th(r.denom_5pct)} | {th(r.relvar_5pct)} | {th(r.relvar_20pct)} |")
    return "\n".join(lines)


# ------------------------------------------------------------------ facts
S, C = "SUM", "CONTRAST"
a_S2 = {t: row(e01, t, 4, "S2", 0.2) for t in (S, C)}
a_S3d = {t: row(e01, t, 4, "S3d", 0.3) for t in (S, C)}
a_S1 = {t: row(e01, t, 4, "S1", 0.2) for t in (S, C)}
minD0 = {t: thr[(thr.type == t) & (thr.m == 4)].rho_minD_lt0.iloc[0] for t in (S, C)}
undefS = {t: d[(d.type == t) & (d.m == 4) & (d.scenario == "S2") & (d.defined == 0)].rho_S.min() for t in (S, C)}


# Q2 decomposition at m=4, eps=0.01
def decomp_lines(rho):
    out = []
    for t in (S, C):
        for sc in ("S2", "S3c"):
            r = row(e01, t, 4, sc, rho)
            if r.defined == 1:
                parts = {"d_denom": r.d_denom, "d_neg": r.d_neg, "d_model_uncal": r.d_model_uncal}
                dom = max(parts, key=parts.get)
                out.append(f"{t}/{sc}: d_denom {f(r.d_denom)}, d_neg {f(r.d_neg)}, d_model_uncal {f(r.d_model_uncal)} -> largest: {dom}")
            else:
                parts = {"d_model_cal": r.d_model_cal, "d_dc": r.d_dc}
                out.append(
                    f"{t}/{sc}: exact ratio undefined; d_model_cal {f(r.d_model_cal)}, d_dc {f(r.d_dc)}, d_total {f(r.d_total)}"
                    + (" (GOLD quadrature-limited)" if r.quadrature_limited == 1 else "")
                )
    return out


und_rows = ["| type/belief | rho_S | d_model_cal | d_dc | d_total |", "|---|---|---|---|---|"]
for rho in (0.5, 0.8):
    for t in (S, C):
        for sc in ("S2", "S3c"):
            r = row(e01, t, 4, sc, rho)
            und_rows.append(f"| {t}/{sc} | {rho} | {f(r.d_model_cal)} | {f(r.d_dc)} | {f(r.d_total)}{'*' if r.quadrature_limited == 1 else ''} |")
und_table = "\n".join(und_rows) + "\n(* GOLD quadrature-limited)"
# Q3
dd = d[(d.eps == 0.1) & (d.defined == 1) & (d.rho_S > 0)]
frac_var_gt = float((dd.relvar_vs_uncal.abs() > dd.dmean_vs_uncal).mean())
rat = {
    sc: (
        dd[(dd.m == 4) & (dd.scenario == sc)].relvar_vs_uncal.abs() / dd[(dd.m == 4) & (dd.scenario == sc)].dmean_vs_uncal.replace(0, np.nan)
    ).median()
    for sc in ["S1", "S2", "S3c"]
}
maxdm = {sc: dd[(dd.m == 4) & (dd.scenario == sc)].dmean_vs_uncal.max() for sc in ["S1", "S2", "S3c"]}
maxrv = {sc: dd[(dd.m == 4) & (dd.scenario == sc)].relvar_vs_uncal.abs().max() for sc in ["S1", "S2", "S3c"]}

# Q4
s3c = d[(d.type == S) & (d.m == 4) & (d.scenario == "S3c")]
eps_tab = s3c[s3c.rho_S.isin([0.7, 0.8, 0.9])].pivot_table(index="rho_S", columns="eps", values="d_model_cal")
dtot = s3c[s3c.eps == 0.1].set_index("rho_S").d_total
und = d[(d.defined == 0) & (d.q_valid == 1)]
stab = {}
for eps, g in und.groupby("eps"):
    ok = g.quadrature_limited != 1
    stab[eps] = (float((g.d_total < g.d_model_cal).mean()), float((g.d_total < g.d_model_cal)[ok].mean()), len(g), int(ok.sum()))
mdep = {t: [row(e01, t, m, "S2", 0.3).d_denom for m in (2, 4, 6)] for t in (S, C)}
width = {t: {sc: row(e01, t, 4, sc, 0.3).d_denom for sc in ["S1", "S2", "S3b", "S3c", "S3a", "S3d"]} for t in (S, C)}


# Taylor
def tay_row(t, sc, rho):
    g = tay[(tay.type == t) & (tay.scenario == sc) & np.isclose(tay.rho_S, rho)].set_index("K").d_taylor
    return g


ts = {rho: tay_row(S, "S2", rho) for rho in (0.1, 0.2, 0.3, 0.4)}
tc = {rho: tay_row(C, "S3c", rho) for rho in (0.1, 0.2, 0.4)}
series_ok_last = tay[(tay.type == S) & (tay.scenario == "S2") & (tay.series_ok == 1)].rho_S.max()

# training
tu = trn.drop_duplicates(["type", "rho_S"])
trn_rows = ["| type | rho_S | test MSE | min D | area D<0 | status (nit) |", "|---|---|---|---|---|---|"]
for t_ in ("SUM", "CONTRAST"):
    for r_ in cfg["train"]["rho_S"]:
        r = tu[(tu.type == t_) & np.isclose(tu.rho_S, r_)].iloc[0]
        trn_rows.append(
            f"| {t_} | {r_} | {r.mse_test_before:.3g}→{r.mse_test_after:.3g} | {r.min_D_before:.2f}→{r.min_D_after:.2f} | {p(r.fracDneg_before)}→{p(r.fracDneg_after)} | {int(r.opt_status)} ({int(r.nit)}) |"
        )
trn_table = "\n".join(trn_rows)


def trn_line(t, rho):
    r = tu[(tu.type == t) & np.isclose(tu.rho_S, rho)].iloc[0]
    return (
        f"{t[:3]} rho_S={rho}: test MSE {r.mse_test_before:.3g}->{r.mse_test_after:.3g}, min D {r.min_D_before:.2f}->{r.min_D_after:.2f}, "
        f"area D<0 {p(r.fracDneg_before)}->{p(r.fracDneg_after)}, optimiser status {int(r.opt_status)} (nit {int(r.nit)})"
    )


trn_dtot = trn[trn.scenario.isin(["S2", "S3c"])].assign(delta=lambda x: x.d_total_after - x.d_total_before)
dA = (tu[tu.rho_S >= 0.5].fracDneg_after - tu[tu.rho_S >= 0.5].fracDneg_before).abs().max()
trn_max_improve = -trn_dtot.delta.min()
trn_max_worse = trn_dtot.delta.max()
gck = gc[(gc.yz_nodes == 401) & (np.isclose(gc.rho_S, 0.6) | np.isclose(gc.rho_S, 0.9))]


def gcv(rho, nx, col):
    return gc[(gc.yz_nodes == 401) & np.isclose(gc.rho_S, rho) & (gc.x_nodes == nx)][col].iloc[0]


nql = int((E0.quadrature_limited == 1).sum())
nrows = len(d)
_z = np.load(ROOT + "/results/coeffs.npz")
se011 = float(np.median([_z[k][0, 1, 1] for k in _z.files if k.endswith("_se")]))
st = lambda k: "PASS" if tests[k]["pass"] else "FAIL (documented)"

# ------------------------------------------------------------------ report
md = f"""# Constant-denominator approximation in the HCR neuron: numerical check
*(all numbers generated by code from results/results.csv; seed {cfg["seed"]}, n={cfg["n_samples"]:.0e} samples per (type, rho_S))*

**Setup.** One HCR neuron, d=3 (output x, inputs y,z), Gaussian-copula ground truth, Legendre basis of degree m∈{{2,4,6}}, coefficients by direct moment estimation. Inputs' dependence is set by Spearman rho_S∈[0,0.9]; neuron types SUM (r_xy=r_xz=0.5) and CONTRAST (r_xy=+c, r_xz=−c). Incoming belief q: S1 (q=D), S2 flat, S3a–d Beta products (S3c = conflicting, S3d = narrow). Compared: DUDA (the paper's constant-denominator formula, Eq. 10), EXACT_UNCAL (E_q[N/D], defined only if D≥{cfg["D_floor"]} on supp q), EXACT_CAL (max(N,eps) + per-(y,z) normalisation) and GOLD (analytic copula conditional). Errors are total-variation (TV) distances; d_denom=TV(DUDA,EXACT_UNCAL), d_neg=TV(EXACT_UNCAL,EXACT_CAL), d_model=TV(·,GOLD), d_total=TV(DUDA,GOLD).
Tests: T1 {st("T1")}, T2 {st("T2")}, T3 {st("T3")}, T4 {st("T4")}, T5 {st("T5")}, T6 {st("T6")}, T7 {st("T7")}, T8 {st("T8")}, T9 {st("T9")} (see Deviations), T10 {st("T10")} (reduced re-run).

## Thresholds (m = 4; rho_S where the error first reaches the level; — = not reached before D<0 makes the exact ratio undefined, or up to 0.9)
{thr_table([4])}

For m=4, D becomes negative at rho_S≈{th(minD0[S])} (SUM) and {th(minD0[C])} (CONTRAST); the exact ratio is undefined from rho_S={th(undefS[S])} / {th(undefS[C])} on a 0.1 grid. All m and the min D<D_floor onset: results/thresholds.csv.

## Answers
**Q1 – when does the constant-denominator error become noticeable?** With a flat belief (S2, m=4), d_denom is {p(a_S2[S].d_denom)} (SUM) and {p(a_S2[C].d_denom)} (CONTRAST) at rho_S=0.2, and reaches 1% already at rho_S≈{th(thr[(thr.type == S) & (thr.m == 4) & (thr.scenario == "S2")].denom_1pct.iloc[0])} / {th(thr[(thr.type == C) & (thr.m == 4) & (thr.scenario == "S2")].denom_1pct.iloc[0])}. For the neuron's own density (S1) the signed relative variance error is {p(a_S1[S].relvar_vs_uncal)} (SUM) and {p(a_S1[C].relvar_vs_uncal)} (CONTRAST) at rho_S=0.2. For a narrow belief (S3d) it is small: d_denom={f(a_S3d[S].d_denom, 4)} (SUM) and {f(a_S3d[C].d_denom, 4)} (CONTRAST) at rho_S=0.3. The effect therefore depends strongly on how wide the incoming belief is (rho_S=0.3, m=4, d_denom, SUM / CONTRAST: S1 {f(width[S]["S1"])}/{f(width[C]["S1"])}, S2 {f(width[S]["S2"])}/{f(width[C]["S2"])}, S3b {f(width[S]["S3b"])}/{f(width[C]["S3b"])}, S3c {f(width[S]["S3c"])}/{f(width[C]["S3c"])}, S3a {f(width[S]["S3a"])}/{f(width[C]["S3a"])}, S3d {f(width[S]["S3d"])}/{f(width[C]["S3d"])}).

**Q2 – d_denom vs d_neg vs d_model.** At rho_S=0.2 (m=4, eps=0.01): {"; ".join(decomp_lines(0.2))}. Where D<0 (rho_S=0.5, 0.8) the exact ratio is undefined and only d_model_cal, d_dc and d_total remain:

{und_table}

The labels show which component dominates per case at rho_S=0.2. For CONTRAST/S2, d_total reaches {f(row(e01, C, 4, "S2", 0.8).d_total, 2)} at rho_S=0.8 (GOLD variance {f(row(e01, C, 4, "S2", 0.8).var_gold, 3)} vs DUDA {f(row(e01, C, 4, "S2", 0.8).var_duda, 3)}; GOLD there is quadrature-limited, see Deviations).

**Q3 – variance vs mean.** In {p(frac_var_gt, 0)} of the {len(dd)} defined (rho_S>0) configurations the relative variance error exceeds the absolute mean error. Median ratio |rel. var. err|/|Δmean| at m=4: S1 {rat["S1"]:.0f}, S2 {rat["S2"]:.0f}, S3c {rat["S3c"]:.0f}; maximum errors (m=4): S1 variance {p(maxrv["S1"])} vs mean {f(maxdm["S1"], 4)}; S2 {p(maxrv["S2"])} vs {f(maxdm["S2"], 4)}; S3c {p(maxrv["S3c"])} vs {f(maxdm["S3c"], 4)}. The mean is nearly preserved, the variance is not; with conflicting beliefs (S3c) the mean error is no longer negligible.

**Q4 – sensitivity.** *eps:* for SUM/S3c/m=4, d_model_cal at rho_S=0.7/0.8/0.9 is {f(eps_tab.loc[0.7, 0.1])}/{f(eps_tab.loc[0.8, 0.1])}/{f(eps_tab.loc[0.9, 0.1])} (eps=0.1), {f(eps_tab.loc[0.7, 0.01])}/{f(eps_tab.loc[0.8, 0.01])}/{f(eps_tab.loc[0.9, 0.01])} (0.01), {f(eps_tab.loc[0.7, 0.001])}/{f(eps_tab.loc[0.8, 0.001])}/{f(eps_tab.loc[0.9, 0.001])} (0.001), while d_total of DUDA is {f(dtot.loc[0.7])}/{f(dtot.loc[0.8])}/{f(dtot.loc[0.9])}: naive clipping can be farther from GOLD than the ratio of expectations here. *m:* at rho_S=0.3, S2, d_denom for m=2/4/6 is {"/".join(f(x) for x in mdep[S])} (SUM) and {"/".join(f(x) for x in mdep[C])} (CONTRAST): weak m-dependence, but the onset of D<0 moves earlier with m. *Stabiliser?* Over the {stab[0.01][2]} valid rows where EXACT_UNCAL is undefined (eps=0.01), DUDA is closer to GOLD than EXACT_CAL in {p(stab[0.01][0], 0)} (excluding GOLD-quadrature-limited rows: {p(stab[0.01][1], 0)} of {stab[0.01][3]}); eps=0.1: {p(stab[0.1][0], 0)}, eps=0.001: {p(stab[0.001][0], 0)}. DUDA stays defined and finite there, but it is not systematically more accurate than a calibrated conditional – it depends on belief and rho_S: SUM/S3c at rho_S=0.8: d_total {f(row(e01, S, 4, "S3c", 0.8).d_total)} vs d_model_cal {f(row(e01, S, 4, "S3c", 0.8).d_model_cal)} (DUDA closer); CONTRAST/S2 at rho_S=0.5: {f(row(e01, C, 4, "S2", 0.5).d_total)} vs {f(row(e01, C, 4, "S2", 0.5).d_model_cal)} (calibrated conditional closer).

## Optional stages (exploratory)
*Taylor-corrected denominator (m=4, SUM/S2, TV to EXACT_UNCAL, K=0/1/2):* rho_S=0.1: {"/".join(f(ts[0.1][k], 4) for k in (0, 1, 2))}; 0.2: {"/".join(f(ts[0.2][k], 4) for k in (0, 1, 2))}; 0.3: {"/".join(f(ts[0.3][k], 4) for k in (0, 1, 2))}; 0.4: {"/".join(f(ts[0.4][k], 4) for k in (0, 1, 2))}. It helps at small rho_S (K=2 best at 0.1–0.2), is marginal at 0.3 (K=1 {f(ts[0.3][1], 4)} vs {f(ts[0.3][0], 4)}, K=2 worse) and is worse from 0.4 (the series condition |1−D|<1 on supp q fails for SUM/S2 from rho_S=0.2 on, max|1−D| on supp q is {f(tay[(tay.type == S) & (tay.scenario == "S2") & np.isclose(tay.rho_S, 0.2) & (tay.K == 0)].max_abs_1mD.iloc[0], 2)} at rho_S=0.2; last rho_S where the condition holds: {th(series_ok_last)}). CONTRAST/S3c, rho_S=0.1/0.2/0.4: K=0 {"/".join(f(tc[r][0], 4) for r in (0.1, 0.2, 0.4))}, K=2 {"/".join(f(tc[r][2], 4) for r in (0.1, 0.2, 0.4))}.
*Does a value (MSE) loss reduce negativity / propagation error?* (a_1jk, a_0jk trained, L-BFGS ≤{cfg["train"]["maxiter"]} its, 2e5 samples; before→after):

{trn_table}

Largest change of d_total (S2,S3c) after training: improvement {f(trn_max_improve, 4)}, deterioration {f(trn_max_worse, 4)}. The test MSE drops sharply where the direct estimates have B≤0 on some samples (rho_S≥0.5), but the area with D<0 changes by at most {f(100 * dA, 2)} percentage points (rho_S≥0.5) and min D is not systematically raised; runs at rho_S≥0.5 stopped at the iteration limit (status 1, not converged). One optimiser, one run, only a_0jk and a_1jk trained – no conclusion.

## Limitations
Single neuron, d=3, Gaussian copula, low degrees (m≤6), no deep network and no end-to-end training; S2 (flat belief) is not the data distribution of (y,z), so DUDA(S2)=Σa_i00 f_i is the model's x-marginal while GOLD(S2) is not uniform even for a perfect model – part of d_total in S2 is this prior mismatch, not an HCR defect. EXACT_CAL uses per-(y,z) conditional normalisation (a harsher calibration than the paper's neglect of it); the paper's own remedy is correction by training, which is only probed in Stage 9. Sampling noise (n=2e6) enters through a_ijk (median SE of a_011 ≈ {se011:.1e}).

## Deviations / quadrature
{nql} of {len(E0)} rows (all CONTRAST: {", ".join(sorted(E0[E0.quadrature_limited == 1].scenario.unique()))}; rho_S≥{E0[E0.quadrature_limited == 1].rho_S.min():.1f}) are flagged `quadrature_limited`: the (y,z) integral is converged (401 vs 801 normal-score nodes identical) but GOLD(x) develops integrable spikes at x→0,1 that a fixed x-grid cannot resolve; ∫GOLD={f(gcv(0.6, 192, "int_gold"), 4)} at rho_S=0.6 and {f(gcv(0.9, 192, "int_gold"), 3)} at 0.9 on 192 x-nodes; TV(DUDA,GOLD) changes from {f(gcv(0.9, 192, "tv_duda_gold"))} (192) to {f(gcv(0.9, 1000, "tv_duda_gold"))} (1000 x-nodes) at rho_S=0.9, and {f(gcv(0.6, 192, "tv_duda_gold"), 4)}→{f(gcv(0.6, 1000, "tv_duda_gold"), 4)} at 0.6. T9: {tests["T9"]["detail"].split(" Deviation:")[0]}. T10 was verified on a reduced re-run only.

## Three talking points (measurements, not criticism)
1. With a flat belief, m=4 and rho_S=0.2, replacing E[N/D] by E[N]/E[D] changes the output density by {p(a_S2[S].d_denom)} (SUM) / {p(a_S2[C].d_denom)} (CONTRAST) in TV; for a narrow belief at rho_S=0.3 it is {f(a_S3d[S].d_denom, 4)} / {f(a_S3d[C].d_denom, 4)}.
2. The mean survives (|Δmean|≤{f(max(maxdm["S1"], maxdm["S2"]), 4)} for S1/S2) while the variance error reaches {p(max(maxrv["S1"], maxrv["S2"]))}; this is where the approximation is visible.
3. Dr Duda's point about negative densities is measurable: D<0 starts at rho_S≈{th(minD0[S])} (m=4), after which the exact ratio is undefined and only calibrated or ratio-of-expectation forms exist; whether training removes this was probed only once (Stage 9) with no systematic reduction of the negative area.
"""
with open(ROOT + "/results/report.txt", "w") as fh:
    fh.write(md)
print(f"report written ({len(md.split())} words)")
