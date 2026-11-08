"""A denominator-free route to the Jeffrey answer (CPU, about ten minutes): regress f_i(x) on features of y (post hoc).

    .venv/bin/python paper/regression.py       # writes paper/regression.json

Not pre-registered and not a study result. Gaussian teacher, type SUM, m = 4, coefficients estimated from n = 2^20 samples, target the
study's truth (the Jeffrey answer). Output moments are B^T E_q[features]: the run-time form of the moment message, with no denominator.
REG1 has additive features, REG2 adds the two-input products of the pair family. p_EU drops belief samples with D < .01, as in the study.
"""

import json
import os
import sys

import numpy as np
import torch
from scipy.special import ndtr, ndtri

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from densprop.domain import copulas as C  # noqa: E402
from densprop.domain.neuron import legendre  # noqa: E402

m, N, Q = 4, 1 << 20, 1 << 17
s = np.linspace(-8, 8, 1201)
hs = s[1] - s[0]
phi = np.exp(-0.5 * s * s) / np.sqrt(2 * np.pi)


def f(u):
    return legendre(torch.tensor(u), m).numpy()


FX = f(ndtr(s))


def feats(Y):
    F = f(Y)[..., 1:]
    n, k, _ = F.shape
    a, b = np.triu_indices(k, 1)
    return np.concatenate([np.ones((n, 1)), F.reshape(n, -1), (F[:, a, :, None] * F[:, b, None, :]).reshape(n, -1)], 1)


def run(k, rho, rng):
    R, _ = C.coupled_R("SUM", k, rho)
    L = np.linalg.cholesky(R)
    sig = R[0, 1:]
    beta = np.linalg.solve(R[1:, 1:], sig)
    sd = np.sqrt(1 - sig @ beta)
    q = feats(np.full((1, k), 0.5)).shape[1]
    Gram, B, done = np.zeros((q, q)), np.zeros((q, m + 1)), 0
    while done < N:
        U = ndtr(rng.standard_normal((1 << 16, k + 1)) @ L.T)
        P = feats(U[:, 1:])
        Gram += P.T @ P
        B += P.T @ f(U[:, 0])
        done += len(U)
    Gram /= done
    B /= done
    k1 = 1 + k * m
    W = np.zeros((q, m + 1))  # the pair neuron: its coefficients are mixed moments, row 0 from the Gram matrix's first column
    W[:, 0] = Gram[:, 0]
    W[:k1, 1:] = B[:k1, 1:]
    reg2 = np.linalg.solve(Gram + 1e-9 * np.eye(q), B)
    reg1 = np.linalg.solve(Gram[:k1, :k1] + 1e-9 * np.eye(k1), B[:k1])
    rows = []
    for name, ps in (("S2", None), ("S3a", [(3, 3)]), ("S3c", [(4, 2), (2, 4)]), ("S3d", [(6, 6)])):
        Y = rng.random((Q, k)) if ps is None else np.stack([rng.beta(*ps[i % len(ps)], Q) for i in range(k)], 1)
        P = feats(Y)
        A = P @ W
        D = A[:, 0]
        ok = D >= 0.01
        mu = ndtri(np.clip(Y, 1e-15, 1 - 1e-15)) @ beta
        gold = np.zeros_like(s)
        for i in range(0, Q, 1 << 13):
            gold += (np.exp(-0.5 * ((s[None] - mu[i : i + (1 << 13), None]) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))).sum(0)
        gold /= Q
        pm = P.mean(0)
        c2, c1 = pm @ reg2, pm[:k1] @ reg1
        tv = lambda c: float(0.5 * hs * np.abs((c @ FX.T) * phi - gold).sum())
        rows.append(
            dict(
                k=k,
                rho=rho,
                belief=name,
                low_share=float((~ok).mean()),
                cd=tv(A.mean(0) / D.mean()),
                eu=tv((A[ok] / D[ok, None]).mean(0)),
                reg1=tv(c1 / c1[0]),
                reg2=tv(c2 / c2[0]),
            )
        )
    return rows


def main():
    rng = np.random.default_rng(21)
    rows = []
    for k in (2, 4, 8, 16):
        for rho in (0.1, 0.3, 0.5, 0.7):
            rows += run(k, rho, rng)
            print(k, rho, flush=True)
    assert all(np.isfinite(r[c]) for r in rows for c in ("cd", "eu", "reg1", "reg2"))
    json.dump(dict(m=m, n=N, belief_samples=Q, rows=rows), open(os.path.join(ROOT, "paper", "regression.json"), "w"), indent=0, sort_keys=True)


if __name__ == "__main__":
    main()
