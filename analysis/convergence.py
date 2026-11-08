"""Test M13 / hypothesis H4b: convergence of every reported TV under doubling of M, n_est, s-nodes and x-nodes.

Each `convergence` job is paired with the `gauss` job of the same (k, type, basis, seed) (identical random streams).
A change is a violation if |TV_variant - TV_base| > max(3 * max(SE_base, SE_variant), 1e-3).
Writes results/summary/convergence.csv and convergence_violations.csv.
"""

import os

import numpy as np
import pandas as pd

from analysis.common import SUMMARY, load

TV = ["d_denom", "d_neg", "d_model_uncal", "d_model_cal", "d_total", "d_dc"]
KEY = ["k", "type", "basis", "seed", "rho_S", "m", "belief", "eps"]


def m13():
    cols = KEY + ["variant"] + TV + [t + "_se" for t in TV]
    conv, base = load("convergence", columns=cols), load("gauss", columns=[c for c in cols if c != "variant"])
    if conv is None or base is None:
        return False, "convergence or gauss results missing"
    base = base[base.seed.isin(conv.seed.unique()) & base.k.isin(conv.k.unique())]
    j = conv.merge(base, on=KEY, suffixes=("", "_base"))
    rows, viol = [], []
    for v, g in j.groupby("variant"):
        for t in TV:
            ok = g[t].notna() & g[t + "_base"].notna()
            if not ok.any():
                continue
            gg = g[ok]
            diff = (gg[t] - gg[t + "_base"]).abs()
            tol = np.maximum(3 * np.maximum(gg[t + "_se"], gg[t + "_se_base"]), 1e-3)
            bad = diff > tol
            rows.append(
                dict(
                    variant=v,
                    metric=t,
                    n=int(ok.sum()),
                    max_change=float(diff.max()),
                    median_change=float(diff.median()),
                    n_violations=int(bad.sum()),
                    worst_ratio=float((diff / tol).max()),
                )
            )
            if bad.any():
                b = gg[bad].assign(change=diff[bad], tol=tol[bad], metric=t)
                viol.append(
                    b[["variant", "metric"] + KEY + [t, t + "_base", "change", "tol"]].rename(columns={t: "value", t + "_base": "value_base"})
                )
    os.makedirs(SUMMARY, exist_ok=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(SUMMARY, "convergence.csv"), index=False)
    vt = pd.concat(viol, ignore_index=True) if viol else pd.DataFrame()
    vt.to_csv(os.path.join(SUMMARY, "convergence_violations.csv"), index=False)
    n_bad = int(tab.n_violations.sum())
    parts = [f"{r.variant}/{r.metric}: max change {r.max_change:.1e}, {r.n_violations}/{r.n} violations" for r in tab.itertuples() if r.n_violations]
    det = f"{int(tab.n.sum())} paired TV values, {n_bad} violations" + (": " + "; ".join(parts) if parts else "")
    if n_bad:
        by_k = vt.groupby(["variant", "k"]).size().to_dict()
        det += f" | violations by (variant, k): {by_k}"
    return n_bad == 0, det


if __name__ == "__main__":
    print(m13())
