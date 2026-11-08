"""Seed-median curves and thresholds with bootstrap CIs -> results/summary/curves_<exp>.csv.gz, thresholds_<exp>.csv

python -m analysis.thresholds [gauss population families lkj]
"""

import os
import sys

import numpy as np
import pandas as pd

from analysis.common import SUMMARY, boot_stat, crossing, load, onset_negative, seed_matrix

CELL = ["type", "family", "k", "basis", "m", "belief"]
TV = ["d_denom", "d_neg", "d_model_uncal", "d_model_cal", "d_total", "d_dc"]
COLS = (
    CELL
    + [
        "seed",
        "rho_S",
        "eps",
        "defined",
        "q_valid",
        "minD",
        "P_Dneg_u",
        "frac_Nneg",
        "chi_D",
        "sd_q_D",
        "relvar_vs_uncal",
        "relvar_vs_gold",
        "smean_vs_uncal",
        "smean_vs_gold",
        "dmean_vs_uncal",
        "dmean_vs_gold",
        "neg_mass_cd",
        "int_gold",
        "d_gold_unif",
        "rhoS_emp_yy",
        "tau",
        "theta",
        "minD_supp",
    ]
    + TV
    + [t + "_se" for t in TV]
    + [t + "_below" for t in TV]
    + [t + "_floor" for t in TV]
)


def curves(df):
    """Per cell, rho_S and eps: seed median and 2.5/97.5 % quantiles of every metric, share undefined, MC SE."""
    out = []
    keys = [c for c in CELL + ["eps", "rho_S"] if c in df]
    for key, g in df.groupby(keys, dropna=False):
        r = dict(zip(keys, key))
        r["n_seeds"] = g.seed.nunique()
        r["share_undefined"] = float((g.defined != 1).mean())
        for c in TV + [
            "minD",
            "P_Dneg_u",
            "frac_Nneg",
            "chi_D",
            "relvar_vs_uncal",
            "relvar_vs_gold",
            "smean_vs_uncal",
            "smean_vs_gold",
            "dmean_vs_uncal",
            "dmean_vs_gold",
            "neg_mass_cd",
            "sd_q_D",
            "rhoS_emp_yy",
        ]:
            if c not in g:
                continue
            v = g[c].to_numpy(float)
            ok = np.isfinite(v)
            r[c + "_n"] = int(ok.sum())
            if ok.any():
                r[c + "_med"], r[c + "_lo"], r[c + "_hi"] = np.median(v[ok]), *np.percentile(v[ok], [2.5, 97.5])
                r[c + "_sd"] = float(np.std(v[ok], ddof=1)) if ok.sum() > 1 else np.nan
            if c + "_se" in g:
                r[c + "_mcse"] = float(np.nanmedian(g[c + "_se"]))
                r[c + "_below_share"] = float(np.nanmean(g[c + "_below"])) if g[c + "_below"].notna().any() else np.nan
        out.append(r)
    return pd.DataFrame(out)


def thresholds(df):
    """Pre-registered thresholds per cell (eps = 0.01) with bootstrap CIs over seeds."""
    rows = []
    d = df[np.isclose(df.eps, 0.01)]
    keys = [c for c in CELL if c in d]
    for key, cell in d.groupby(keys, dropna=False):
        r = dict(zip(keys, key))
        g = cell.copy()
        g["abs_relvar"] = g.relvar_vs_uncal.abs()
        rhos, seeds, Vd = seed_matrix(g, "d_denom")
        r["n_seeds"] = len(seeds)
        for name, V, level in (
            ("denom_1pct", Vd, 0.01),
            ("denom_5pct", Vd, 0.05),
            ("relvar_5pct", seed_matrix(g, "abs_relvar")[2], 0.05),
            ("relvar_20pct", seed_matrix(g, "abs_relvar")[2], 0.20),
        ):
            p, lo, hi, _ = boot_stat(V, lambda c, lv=level: crossing(rhos, c, lv))
            per = np.array([crossing(rhos, V[i], level) for i in range(V.shape[0])])
            r.update(
                {
                    name: p,
                    name + "_lo": lo,
                    name + "_hi": hi,
                    name + "_seed_med": np.median(per),
                    name + "_seed_share_reached": float(np.isfinite(per).mean()),
                }
            )
        _, _, Vm = seed_matrix(g, "minD", defined_col=None)
        p, lo, hi, _ = boot_stat(Vm, lambda c: onset_negative(rhos, c))
        per = np.array([onset_negative(rhos, Vm[i]) for i in range(Vm.shape[0])])
        r.update(neg_onset=p, neg_onset_lo=lo, neg_onset_hi=hi, neg_onset_seed_med=np.median(per))
        und = g.groupby("rho_S").defined.apply(lambda v: float((v != 1).mean()))
        hit = und[und >= 0.5]
        r["undef_onset"] = float(hit.index.min()) if len(hit) else np.inf
        r["minD_at_0"] = float(np.median(Vm[:, 0]))
        rows.append(r)
    return pd.DataFrame(rows)


def main(exps):
    os.makedirs(SUMMARY, exist_ok=True)
    for exp in exps:
        df = load(exp, columns=None)
        if df is None:
            print(f"{exp}: no merged results")
            continue
        df = df[[c for c in COLS if c in df.columns]]
        if exp == "lkj":
            continue  # matrices are not on a rho grid: see analysis/hypotheses.py
        curves(df).to_csv(os.path.join(SUMMARY, f"curves_{exp}.csv.gz"), index=False)
        thresholds(df).to_csv(os.path.join(SUMMARY, f"thresholds_{exp}.csv"), index=False)
        print(f"{exp}: curves and thresholds written")


if __name__ == "__main__":
    main(sys.argv[1:] or ["gauss", "population", "families"])
