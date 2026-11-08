"""Exploratory real-data pilot (CPU, tens of minutes): California housing through one untrained, directly estimated neuron.

    <venv>/bin/python paper/pilot.py        # writes paper/pilot.json

Not pre-registered, run after the study's results were seen, no tuning. Needs scikit-learn, which is deliberately not a project
requirement: run it from a separate virtual environment outside the repository (numpy, scipy, scikit-learn).
make_numbers.py reads only the JSON, so the paper rebuilds without scikit-learn.

Eight inputs and the median value are mapped to [0, 1] by the training EDF; 15,000 training and 2,500 test rows per seed. Scenarios:
  full     all inputs seen; the belief is a point
  missing  each test input lost with probability 1/2; the belief is a point on the rest and flat on the lost ones
  noisy    each input's normal score seen with noise tau = .7; the belief is the per-input posterior
p_CD is exact: a product belief enters only through its per-input moment vectors. p_EU uses 128 belief samples per row and drops those
with D < .01, as in the study; a row with no sample left returns the marginal. REG is the ridge regression message on the same features.
"""

import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy
import sklearn
from numpy.polynomial import legendre as Lg
from scipy.stats import norm, spearmanr
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS, N_TRAIN, N_TEST, TAU, FLOOR, SAMPLES, RIDGE = 10, 15000, 2500, 0.7, 0.01, 128, 1e-2
FAMILIES = (("pair", 2), ("pair", 3), ("triple", 2), ("triple", 3))
XQ, WQ = np.polynomial.legendre.leggauss(200)
XQ, WQ = (XQ + 1) / 2, WQ / 2


def f(u, m):
    t = 2 * np.asarray(u) - 1
    return np.stack([np.sqrt(2 * n + 1) * Lg.legval(t, [0] * n + [1]) for n in range(m + 1)], -1)


def edf(train, x):
    s = np.sort(train)
    return (np.searchsorted(s, x, side="left") + np.searchsorted(s, x, side="right")) / 2 / len(s) * (1 - 2e-3) + 1e-3


def phi_moments(F, order):
    """Features of a product belief from per-input moments F (n, k, m), index 0 left out."""
    n, k, _ = F.shape
    o = [np.ones((n, 1)), F.reshape(n, -1)]
    a, b = np.triu_indices(k, 1)
    o.append((F[:, a, :, None] * F[:, b, None, :]).reshape(n, -1))
    if order == 3:
        o.append(
            np.concatenate(
                [
                    (F[:, i, :, None, None] * F[:, j, None, :, None] * F[:, c, None, None, :]).reshape(n, -1)
                    for i, j, c in itertools.combinations(range(k), 3)
                ],
                1,
            )
        )
    return np.concatenate(o, 1)


def phi(Y, m, order):
    return phi_moments(f(Y, m)[..., 1:], order)


def fit(U, x, m, basis):
    """Direct moments W (rows i >= 1 restricted to the family) and the ridge message R on the same features."""
    order = 3 if basis == "triple" else 2
    P, Fx = phi(U, m, order), f(x, m)
    W = P.T @ Fx / len(x)
    k = U.shape[1]
    n1 = 1 + k * m
    n2 = n1 + k * (k - 1) // 2 * m * m
    W[(n1 if basis == "pair" else n2) :, 1:] = 0
    assert abs(W[0, 0] - 1) < 1e-12
    R = np.linalg.solve(P.T @ P / len(x) + RIDGE * np.eye(P.shape[1]), P.T @ Fx / len(x))
    return W, R, order


def score(c, x, m):
    """Predicted mean and log density, the density clipped at .01 and renormalised."""
    g = np.linspace(0, 1, 401)
    Z = np.trapezoid(np.maximum(c @ f(g, m).T, 0.01), g, axis=1)
    return 0.5 + c[:, 1] / np.sqrt(12), np.log(np.maximum((c * f(x, m)).sum(1), 0.01) / Z)


def one_seed(seed):
    Xa, ya = fetch_california_housing(return_X_y=True)
    res = []
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(ya))
    tr, te = p[:N_TRAIN], p[N_TRAIN : N_TRAIN + N_TEST]
    k = Xa.shape[1]
    U = np.stack([edf(Xa[tr, j], Xa[tr, j]) for j in range(k)], 1)
    Ut = np.stack([edf(Xa[tr, j], Xa[te, j]) for j in range(k)], 1)
    x, xt = edf(ya[tr], ya[tr]), edf(ya[tr], ya[te])
    C = spearmanr(U).statistic
    C = np.abs(C[np.triu_indices(k, 1)])
    n = len(te)
    mask = rng.random((n, k)) < 0.5
    S = norm.ppf(Ut)
    O = S + TAU * rng.standard_normal(S.shape)
    t2 = TAU**2
    v = t2 / (1 + t2)
    Sn = norm.ppf(U) + TAU * rng.standard_normal(U.shape)
    Xm = U.copy()
    Xm[rng.random(U.shape) < 0.5] = np.nan
    Xt = Ut.copy()
    Xt[mask] = np.nan
    boost = lambda A, B: HistGradientBoostingRegressor(max_iter=300, random_state=seed).fit(A, x).predict(B)
    for sc, pr in (("full", boost(U, Ut)), ("missing", boost(Xm, Xt)), ("noisy", boost(Sn, O))):
        res.append(
            dict(seed=seed, scenario=sc, form="boost", mse=float(np.mean((pr - xt) ** 2)), mse_clip=float(np.mean((np.clip(pr, 0, 1) - xt) ** 2)))
        )
        res.append(
            dict(seed=seed, scenario=sc, form="marginal", mse=float(np.mean((0.5 - xt) ** 2)), mse_clip=float(np.mean((0.5 - xt) ** 2)), logs=0.0)
        )
    for basis, m in FAMILIES:
        W, R, order = fit(U, x, m, basis)
        for sc in ("full", "missing", "noisy"):
            if sc == "full":
                Fb = f(Ut, m)[..., 1:]
            elif sc == "missing":
                Fb = f(Ut, m)[..., 1:] * (~mask)[..., None]
            else:  # posterior of u given the observed score, exact quadrature per input
                L = np.exp(-0.5 * ((O[..., None] - norm.ppf(XQ)) / TAU) ** 2) * WQ
                Fb = (L / L.sum(-1, keepdims=True)) @ f(XQ, m)[:, 1:]
            Pb = phi_moments(Fb, order)
            num = Pb @ W
            cd = num / np.where(np.abs(num[:, :1]) < 1e-12, 1e-12, num[:, :1])
            reg = Pb @ R
            reg = reg / reg[:, :1]
            eu, kept, low, mc = np.zeros((n, m + 1)), np.zeros(n), 0, np.zeros(n)
            samples = 1 if sc == "full" else SAMPLES
            for _ in range(samples):
                Y = (
                    Ut
                    if sc == "full"
                    else np.where(mask, rng.random((n, k)), Ut)
                    if sc == "missing"
                    else norm.cdf(O / (1 + t2) + np.sqrt(v) * rng.standard_normal((n, k)))
                )
                for i in range(0, n, 500):
                    A = phi(Y[i : i + 500], m, order) @ W
                    D = A[:, 0]
                    ok = D >= FLOOR
                    eu[i : i + 500] += np.where(ok[:, None], A / np.where(ok, D, 1)[:, None], 0)
                    kept[i : i + 500] += ok
                    low += (~ok).sum()
                    mc[i : i + 500] += D
            # the exact moment contraction must agree with the belief samples
            dev = mc / samples - num[:, 0]
            assert abs(dev.mean()) <= 6 * dev.std() / np.sqrt(n) + 1e-9, (sc, basis, m, dev.mean())
            e = eu / np.maximum(kept, 1)[:, None]
            e[:, 0] = 1
            for form, c in (("cd", cd), ("eu", e), ("reg", reg)):
                mean, ls = score(c, xt, m)
                res.append(
                    dict(
                        seed=seed,
                        scenario=sc,
                        form=form,
                        basis=basis,
                        m=m,
                        mse=float(np.mean((mean - xt) ** 2)),
                        mse_clip=float(np.mean((np.clip(mean, 0, 1) - xt) ** 2)),
                        logs=float(ls.mean()),
                        low_share=float(low / (n * samples)),
                        eqd_nonpos=float((num[:, 0] <= 0).mean()),
                        eqd_small=float((num[:, 0] < 0.1).mean()),
                    )
                )
        print(seed, basis, m, flush=True)
    return dict(rows=res, spearman_max=float(C.max()), spearman_median=float(np.median(C)))


def main():
    fetch_california_housing()  # one download before the workers start
    with ProcessPoolExecutor(4) as ex:
        out = list(ex.map(one_seed, range(SEEDS)))
    rows = [r for o in out for r in o["rows"]]
    with open(os.path.join(HERE, "pilot.json"), "w") as fh:
        json.dump(
            dict(
                seeds=SEEDS,
                n_train=N_TRAIN,
                n_test=N_TEST,
                tau=TAU,
                floor=FLOOR,
                samples=SAMPLES,
                inputs=8,
                ridge=RIDGE,
                spearman_max=out[0]["spearman_max"],
                spearman_median=out[0]["spearman_median"],
                versions=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
                rows=rows,
            ),
            fh,
            indent=0,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()
