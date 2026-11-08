"""Stage 5: threshold table (+ decomposition summary)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from hcr import ROOT


def first_crossing(rho, v, thr, absval=True, below=False):
    """Smallest rho (linear interpolation) where |v|>=thr (or v<=thr if below). Only over defined (non-NaN) neighbours.
    Returns (value or nan, note)."""
    rho = np.asarray(rho, float)
    v = np.asarray(v, float)
    f = (lambda x: np.abs(x)) if absval else (lambda x: x)
    hit = (lambda x: x <= thr) if below else (lambda x: x >= thr)
    for i in range(len(rho)):
        if np.isnan(v[i]):
            return np.nan, f"not reached before undefined (from rho_S={rho[i]:.1f})"
        if hit(f(v[i])):
            if i == 0:
                return float(rho[0]), ""
            x0, x1, y0, y1 = rho[i - 1], rho[i], f(v[i - 1]), f(v[i])
            return float(x0 + (thr - y0) * (x1 - x0) / (y1 - y0)), ""
    return np.nan, "not reached up to rho_S=0.9"


def thresholds(df):
    d = df[df.eps == df.eps.min()].sort_values("rho_S")  # non-eps columns identical across eps
    rows = []
    for (typ, m, sc), g in d.groupby(["type", "m", "scenario"]):
        if g.q_valid.max() == 0:
            continue
        gg = g.sort_values("rho_S")
        r = dict(type=typ, m=m, scenario=sc)
        for name, col, thr in [
            ("denom_1pct", "d_denom", 0.01),
            ("denom_5pct", "d_denom", 0.05),
            ("relvar_5pct", "relvar_vs_uncal", 0.05),
            ("relvar_20pct", "relvar_vs_uncal", 0.20),
        ]:
            val, note = first_crossing(gg.rho_S, gg[col], thr)
            r[name], r[name + "_note"] = val, note
        # S1 rows beyond validity are NaN already; for undefined S1 rows the NaN rule applies
        rows.append(r)
    t = pd.DataFrame(rows)
    # min D crossings (scenario independent)
    md = d[d.scenario == "S2"].sort_values("rho_S")
    mrows = []
    for (typ, m), g in md.groupby(["type", "m"]):
        z, _ = first_crossing(g.rho_S, g.min_D, 0.0, absval=False, below=True)
        z2, _ = first_crossing(g.rho_S, g.min_D, 0.01, absval=False, below=True)
        mrows.append(dict(type=typ, m=m, rho_minD_lt0=z, rho_minD_lt_floor=z2))
    return t.merge(pd.DataFrame(mrows), on=["type", "m"])


if __name__ == "__main__":
    df = pd.read_csv(ROOT + "/results/results.csv")
    t = thresholds(df)
    t.to_csv(ROOT + "/results/thresholds.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(t[["type", "m", "scenario", "denom_1pct", "denom_5pct", "relvar_5pct", "relvar_20pct", "rho_minD_lt0"]].round(2).to_string(index=False))
