"""Seed statistics and the pre-registered aggregation rule."""

import numpy as np
import pandas as pd
import pytest

from densprop.domain.hypotheses import verdict
from densprop.domain.stats import B_BOOT, BOOT_SEED, boot_idx, boot_stat, crossing, median_curve, onset_negative, seed_matrix

RHOS = np.array([0.0, 0.1, 0.2, 0.3])


def test_crossing_interpolates_linearly():
    assert crossing(RHOS, [0.0, 0.02, 0.06, 0.1], 0.05) == pytest.approx(0.175)


def test_crossing_at_the_first_point_and_not_reached():
    assert crossing(RHOS, [0.2, 0.3, 0.4, 0.5], 0.05) == 0.0
    assert crossing(RHOS, [0.0, 0.01, 0.02, 0.03], 0.05) == np.inf


def test_crossing_stops_at_the_first_undefined_point():
    assert crossing(RHOS, [0.0, np.inf, 0.9, 0.9], 0.05) == np.inf
    assert crossing(RHOS, [0.0, np.nan, 0.9, 0.9], 0.05) == np.inf


def test_onset_negative():
    assert onset_negative(RHOS, [1.0, 0.5, -0.5, -1.0]) == pytest.approx(0.15)
    assert onset_negative(RHOS, [-1.0, -1.0, -1.0, -1.0]) == 0.0
    assert onset_negative(RHOS, [1.0, 1.0, 1.0, 1.0]) == np.inf


def test_median_curve_needs_half_of_the_seeds_defined():
    V = np.array([[1.0, np.inf], [2.0, np.inf], [3.0, 5.0]])
    out = median_curve(V)
    assert out[0] == 2.0
    assert np.isnan(out[1])


def test_seed_matrix_marks_undefined_as_inf():
    df = pd.DataFrame(dict(seed=[0, 0, 1, 1], rho_S=[0.0, 0.1, 0.0, 0.1], v=[1.0, 2.0, 3.0, 4.0], defined=[1, 0, 1, 1]))
    rhos, seeds, V = seed_matrix(df, "v")
    assert rhos.tolist() == [0.0, 0.1]
    assert seeds.tolist() == [0, 1]
    assert V.tolist() == [[1.0, np.inf], [3.0, 4.0]]


def test_bootstrap_resamples_are_fixed():
    idx = boot_idx(5)
    assert idx.shape == (B_BOOT, 5)
    assert (idx == np.random.default_rng(BOOT_SEED).integers(0, 5, (B_BOOT, 5))).all()
    V = np.arange(20.0).reshape(5, 4)
    p, lo, hi, vals = boot_stat(V, lambda c: float(c[0]))
    assert p == 8.0
    assert lo <= p <= hi
    assert len(vals) == B_BOOT


@pytest.mark.parametrize(
    ("sig", "n", "want"),
    [(2, 2, "inconclusive"), (2, 3, "supported"), (1, 3, "not supported"), (3, 6, "inconclusive"), (4, 6, "supported"), (2, 6, "not supported")],
)
def test_verdict_rule(sig, n, want):
    assert verdict(sig, n) == want
