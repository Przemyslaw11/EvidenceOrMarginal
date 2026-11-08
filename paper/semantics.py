"""Illustration of the semantics theorem (CPU, about a minute): the better form changes with what the belief means.

    .venv/bin/python paper/semantics.py        # writes paper/semantics.json

Not a study result and not pre-registered. Gaussian teacher, type SUM, population coefficients of the repository's model, so the
only error besides the update itself is truncation. An observation o = s + tau e of the inputs' normal scores defines
  evidence  belief: per-input posterior, a product of N(o_l/(1+tau^2), tau^2/(1+tau^2)); truth p(x | o), closed form
  posterior belief: the joint posterior N(mu, V);                                         truth p(x | o), the same
  marginal  belief: the same product as for evidence, but the inputs really follow it;    truth E_q[p(x | y)], closed form (GOLD)
  missing   belief: a point on y_1 and flat on y_2;                                        truth p(x | y_1), closed form
k = 2 uses the full tensor and Gauss-Hermite quadrature; rho stays below the negativity onset so D > 0 everywhere and no floor is
involved. k = 4, 8 use the pair basis and Monte Carlo over the belief, also below the onset, to show truncation blurring the contrast.
flip() repeats the three problems beyond the pair family (post hoc): the full tensor at k = 3, 4 and the triple family at k = 3, 4, 8. It has its
own random stream, so the rows above do not depend on it, and it asserts min D > 0 at each setting.
"""

import itertools
import json
import math
import os
import sys

import numpy as np
import torch
from numpy.polynomial import hermite_e as nph
from scipy.special import ndtr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gpu import copulas as C  # noqa: E402
from gpu.core import input_basis, legendre, population_model  # noqa: E402

m = 4
s = np.linspace(-8, 8, 1601)
hs = s[1] - s[0]
phi = np.exp(-0.5 * s**2) / math.sqrt(2 * math.pi)
FX = legendre(torch.tensor(ndtr(s)), m).numpy()
Z, W = nph.hermegauss(48)
W = W / W.sum()
G2 = np.stack([g.ravel() for g in np.meshgrid(Z, Z, indexing="ij")], 1)
W2 = np.outer(W, W).ravel()


def nrm(mu, sd):
    return np.exp(-0.5 * ((s - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def tv(p, r):
    return 0.5 * hs * np.abs(p - r).sum()


def forms(mod, S, w):
    """Output s-densities of CD and EU for belief nodes S (normal scores) with weights w."""
    A = mod.A(input_basis(torch.tensor(np.clip(ndtr(S), 1e-15, 1 - 1e-15)), m)).numpy()
    D = A[:, 0]
    assert D.min() > 0
    return ((w @ A) / (w @ D)) @ FX.T * phi, (w @ (A / D[:, None])) @ FX.T * phi


def teacher(k, rho):
    R, _ = C.coupled_R("SUM", k, rho)
    Ryy, sxy = R[1:, 1:], R[0, 1:]
    beta = np.linalg.solve(Ryy, sxy)
    return R, Ryy, beta, 1 - sxy @ beta


def row(mod, Ryy, beta, sig2, tau, rng, nodes):
    k = len(beta)
    t2 = tau**2
    V = np.linalg.inv(np.linalg.inv(Ryy) + np.eye(k) / t2)
    o = np.linalg.cholesky(Ryy) @ rng.standard_normal(k) + tau * rng.standard_normal(k)
    mu = V @ o / t2
    v = t2 / (1 + t2)
    Sq, wq = nodes(o / (1 + t2), math.sqrt(v) * np.eye(k))
    Sj, wj = nodes(mu, np.linalg.cholesky(V))
    t_post = nrm(beta @ mu, math.sqrt(sig2 + beta @ V @ beta))
    t_marg = nrm(beta @ o / (1 + t2), math.sqrt(sig2 + v * beta @ beta))
    cd, eu = forms(mod, Sq, wq)
    cj, ej = forms(mod, Sj, wj)
    return dict(evidence=(tv(cd, t_post), tv(eu, t_post)), posterior=(tv(cj, t_post), tv(ej, t_post)), marginal=(tv(cd, t_marg), tv(eu, t_marg)))


def summary(rows):
    res = {}
    for sem in ("evidence", "posterior", "marginal"):
        a = np.array([r[sem] for r in rows])
        res[sem] = dict(cd=float(np.median(a[:, 0])), eu=float(np.median(a[:, 1])), cd_closer=float((a[:, 0] < a[:, 1]).mean()))
    return res


class Full:
    """Full tensor of degree m per variable, population coefficients by tensor Gauss-Hermite; A as in gpu.core.Model."""

    def __init__(self, R, nodes):
        d = len(R)
        z, w = nph.hermegauss(nodes)
        w = w / w.sum()
        L = np.linalg.cholesky(R)
        G = np.stack([g.ravel() for g in np.meshgrid(*[z] * d, indexing="ij")], 1)
        Wt = np.ones(1)
        for _ in range(d):
            Wt = np.multiply.outer(Wt, w).ravel()
        h, T, step = d // 2, 0, 1 << 18
        for i in range(0, len(G), step):
            F = [legendre(torch.tensor(ndtr(G[i : i + step] @ L[a])), m).numpy() for a in range(d)]
            T = T + (self.outer(F[:h]) * Wt[i : i + step, None]).T @ self.outer(F[h:])
        self.T = torch.tensor(T.reshape((m + 1,) * d).reshape(m + 1, -1))  # rows: f_i(x); columns: the input multi-index

    @staticmethod
    def outer(fs):
        o = fs[0]
        for g in fs[1:]:
            o = (o[:, :, None] * g[:, None, :]).reshape(len(o), -1)
        return o

    def A(self, Fy):
        ones = torch.ones(Fy.shape[0], 1, dtype=Fy.dtype)
        return self.outer([torch.cat([ones, Fy[:, a, :m]], 1) for a in range(Fy.shape[1])]) @ self.T.T


def min_d(mod, k, rng):
    """Upper bound on min D over the cube: coordinate descent from corners and random points (D is a polynomial in each coordinate)."""
    h = np.linspace(0, 1, 201)
    Y = np.concatenate([np.array(list(itertools.product((0.0, 1.0), repeat=k))), rng.random((64, k))])
    best = np.inf
    for _ in range(6):
        for a in range(k):
            T = np.repeat(Y[:, None, :], len(h), 1)
            T[:, :, a] = h
            D = mod.A(input_basis(torch.tensor(np.clip(T.reshape(-1, k), 1e-12, 1 - 1e-12)), m))[:, 0].numpy().reshape(len(Y), -1)
            Y[:, a] = h[D.argmin(1)]
            best = min(best, D.min())
    return float(best)


def flip():
    """The three problems in the models that can hold the conditional, below the onset of each setting."""
    rng = np.random.default_rng(20261106)
    rows = []
    for model, k, rho in (("full", 3, 0.15), ("full", 4, 0.15), ("triple", 3, 0.15), ("triple", 4, 0.09), ("triple", 8, 0.04)):
        R, Ryy, beta, sig2 = teacher(k, rho)
        mod = Full(R, {3: 30, 4: 20}[k]) if model == "full" else population_model(R, m, "triple", nodes=64)
        if model == "full":  # the tensor's one-input rows must match the repository's exact coefficients
            T = mod.T.reshape((m + 1,) * (k + 1)).numpy()
            W1 = population_model(R, m, "pair", nodes=64).blocks["W1"].numpy()
            assert abs(T.flat[0] - 1) < 1e-9 and np.abs(T[(slice(1, None), slice(1, None)) + (0,) * (k - 1)] - W1[1:, 0, :]).max() < 5e-3
        md = min_d(mod, k, rng)
        assert md > 0, (model, k, rho, md)
        for tau in (0.5, 1.0):
            mc = lambda mu, L: (mu + rng.standard_normal((1 << 15, k)) @ L.T, np.full(1 << 15, 2.0**-15))
            r = summary([row(mod, Ryy, beta, sig2, tau, rng, mc) for _ in range(60)])
            rows.append(dict(model=model, k=k, rho=rho, tau=tau, min_d=md, **r))
            print(model, k, rho, tau, round(md, 3), r, flush=True)
    return rows


def main():
    rng = np.random.default_rng(20261105)
    out = dict(k2=[], kbig=[], missing=[], example={})
    gh = lambda mu, L: (mu + G2 @ L.T, W2)
    for rho in (0.1, 0.2, 0.3, 0.4):
        R, Ryy, beta, sig2 = teacher(2, rho)
        mod = population_model(R, m, "triple", nodes=64)  # at k = 2 the triple family is the full tensor
        for tau in (0.5, 1.0):
            r = summary([row(mod, Ryy, beta, sig2, tau, rng, gh) for _ in range(200)])
            out["k2"].append(dict(rho=rho, tau=tau, **r))
            print(rho, tau, r, flush=True)
        # missing input: point on y_1, flat on y_2; truth p(x | y_1) with correlation c
        c = R[0, 1]
        u, wu = np.polynomial.legendre.leggauss(200)
        u, wu = (u + 1) / 2, wu / 2
        vals = []
        for s1 in np.linspace(-2, 2, 21):
            Y = np.stack([np.full_like(u, ndtr(s1)), u], 1)
            A = mod.A(input_basis(torch.tensor(Y), m)).numpy()
            cd = ((wu @ A) / (wu @ A[:, 0])) @ FX.T * phi
            eu = (wu @ (A / A[:, :1])) @ FX.T * phi
            t = nrm(c * s1, math.sqrt(1 - c * c))
            vals.append((tv(cd, t), tv(eu, t)))
        a = np.array(vals)
        out["missing"].append(dict(rho=rho, cd=float(np.median(a[:, 0])), eu=float(np.median(a[:, 1])), cd_closer=float((a[:, 0] < a[:, 1]).mean())))
    for k, rho in ((4, 0.1), (4, 0.15), (8, 0.05), (8, 0.1)):
        R, Ryy, beta, sig2 = teacher(k, rho)
        mod = population_model(R, m, "pair", nodes=64)
        for tau in (0.5, 1.0):
            mc = lambda mu, L: (mu + rng.standard_normal((1 << 15, k)) @ L.T, np.full(1 << 15, 2.0**-15))
            r = summary([row(mod, Ryy, beta, sig2, tau, rng, mc) for _ in range(60)])
            out["kbig"].append(dict(k=k, rho=rho, tau=tau, **r))
            print(k, rho, tau, r, flush=True)
    # the single case drawn in Figure 1
    rho, tau = 0.4, 0.5
    R, Ryy, beta, sig2 = teacher(2, rho)
    mod = population_model(R, m, "triple", nodes=64)
    g = np.linspace(0.005, 0.995, 100)
    Y = np.stack([a.ravel() for a in np.meshgrid(g, g, indexing="ij")], 1)
    D = mod.D(torch.tensor(Y)).numpy().reshape(100, 100)
    out["example"] = dict(rho=rho, tau=tau, g=g.tolist(), D=D.tolist())
    out["flip"] = flip()
    json.dump(out, open(os.path.join(ROOT, "paper/semantics.json"), "w"), indent=0, sort_keys=True)


if __name__ == "__main__":
    main()
