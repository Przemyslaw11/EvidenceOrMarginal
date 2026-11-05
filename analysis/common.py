"""Shared analysis tools: loading merged results, seed-median curves, thresholds and seed bootstraps
(definitions: PREREGISTRATION.md, section 3)."""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("HCR_RESULTS", os.path.join(ROOT, "results"))
SUMMARY = os.path.join(RES, "summary")
B_BOOT = 2000
BOOT_SEED = 12345
INF = np.inf


def load(exp, columns=None, filters=None):
    path = os.path.join(RES, "merged", f"{exp}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path, columns=columns, filters=filters)


def seed_matrix(df, value, rho_col="rho_S", defined_col="defined"):
    """(seeds x rho) matrix of a metric for one cell; undefined -> +inf. Returns (rhos, seeds, V)."""
    d = df.copy()
    if defined_col is not None and defined_col in d:
        d.loc[d[defined_col] != 1, value] = INF
    d[value] = d[value].astype(float).fillna(INF)
    P = d.pivot_table(index="seed", columns=rho_col, values=value, aggfunc="first")
    return P.columns.to_numpy(float), P.index.to_numpy(), P.to_numpy(float)


def median_curve(V):
    """Seed median with +inf for undefined seeds; NaN where fewer than 50 % of the seeds are defined."""
    med = np.median(V, axis=-2)
    frac = np.mean(np.isfinite(V), axis=-2)
    return np.where(frac >= 0.5, med, np.nan)


def crossing(rhos, curve, level):
    """Smallest rho at which curve >= level along the defined run from rho = 0 (linear interpolation).
    +inf if not reached (undefined first or never reached)."""
    curve = np.asarray(curve, float)
    for i in range(len(rhos)):
        v = curve[i]
        if not np.isfinite(v):
            return INF
        if v >= level:
            if i == 0:
                return float(rhos[0])
            v0 = curve[i - 1]
            return float(rhos[i - 1] + (level - v0) / (v - v0) * (rhos[i] - rhos[i - 1]))
    return INF


def onset_negative(rhos, minD):
    """Smallest rho at which minD < 0 (linear interpolation); 0 if already negative at rho = 0."""
    minD = np.asarray(minD, float)
    for i in range(len(rhos)):
        if minD[i] < 0:
            if i == 0:
                return 0.0
            v0, v1 = minD[i - 1], minD[i]
            return float(rhos[i - 1] + v0 / (v0 - v1) * (rhos[i] - rhos[i - 1]))
    return INF


def boot_idx(n, B=B_BOOT, seed=BOOT_SEED):
    return np.random.default_rng(seed).integers(0, n, (B, n))


def boot_stat(V, fn, B=B_BOOT, seed=BOOT_SEED):
    """Statistic of the seed-median curve: point value and bootstrap percentile 95 % CI (resampling seeds)."""
    point = fn(median_curve(V))
    idx = boot_idx(V.shape[0], B, seed)
    vals = np.array([fn(median_curve(V[i])) for i in idx])
    lo, hi = np.percentile(vals, [2.5, 97.5], method="inverted_cdf")  # sample values: safe with +inf
    return point, lo, hi, vals


def fmt_rho(v):
    return "not reached" if not np.isfinite(v) else f"{v:.3f}"
