"""Seed-median curves, thresholds and seed bootstraps (definitions: PREREGISTRATION.md, section 3)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

B_BOOT = 2000
BOOT_SEED = 12345
INF = np.inf


def seed_matrix(
    df: pd.DataFrame, value: str, rho_col: str = "rho_S", defined_col: str | None = "defined"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(seeds x rho) matrix of a metric for one cell; undefined -> +inf. Returns (rhos, seeds, V)."""
    d = df.copy()
    if defined_col is not None and defined_col in d:
        d.loc[d[defined_col] != 1, value] = INF
    d[value] = d[value].astype(float).fillna(INF)
    P = d.pivot_table(index="seed", columns=rho_col, values=value, aggfunc="first")
    return P.columns.to_numpy(float), P.index.to_numpy(), P.to_numpy(float)


def median_curve(V: np.ndarray) -> np.ndarray:
    """Seed median with +inf for undefined seeds; NaN where fewer than 50 % of the seeds are defined."""
    med = np.median(V, axis=-2)
    frac = np.mean(np.isfinite(V), axis=-2)
    return np.where(frac >= 0.5, med, np.nan)


def crossing(rhos: np.ndarray, curve: np.ndarray, level: float) -> float:
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


def onset_negative(rhos: np.ndarray, minD: np.ndarray) -> float:
    """Smallest rho at which minD < 0 (linear interpolation); 0 if already negative at rho = 0."""
    minD = np.asarray(minD, float)
    for i in range(len(rhos)):
        if minD[i] < 0:
            if i == 0:
                return 0.0
            v0, v1 = minD[i - 1], minD[i]
            return float(rhos[i - 1] + v0 / (v0 - v1) * (rhos[i] - rhos[i - 1]))
    return INF


def boot_idx(n: int, B: int = B_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, (B, n))


def boot_stat(V: np.ndarray, fn: Callable[[np.ndarray], float], B: int = B_BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float, np.ndarray]:
    """Statistic of the seed-median curve: point value and bootstrap percentile 95 % CI (resampling seeds)."""
    point = fn(median_curve(V))
    idx = boot_idx(V.shape[0], B, seed)
    vals = np.array([fn(median_curve(V[i])) for i in idx])
    lo, hi = np.percentile(vals, [2.5, 97.5], method="inverted_cdf")  # sample values: safe with +inf
    return point, lo, hi, vals
