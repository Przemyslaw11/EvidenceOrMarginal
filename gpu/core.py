"""HCR neuron engine (torch, float64): basis, index families, block-stored models, estimation.

Variable 0 is the output x, variables 1..k are the inputs y_1..y_k. A model is stored by interaction
order instead of as a dense (m+1)^(k+1) tensor. Degree axes of input blocks hold degrees 1..m
(entry p means f_{p+1}); the x axis i holds degrees 0..m.

    c[i]                 = a_(i, 0, ..., 0)
    W1[i, a, p]          = a_(i, y_a: p+1)
    W2[i, P, p, q]       = a_(i, y_a: p+1, y_b: q+1)                for the pair P = (a, b), a < b
    W3[i3, P, p, q, c, r] = a_(i3, y_a: p+1, y_b: q+1, y_c: r+1)    for c > b (zero otherwise)

A family allows at most `order` nonzero indices over all k+1 variables: rows i = 0 allow `order`
nonzero inputs, rows i > 0 allow `order - 1`. Inputs interacting in groups larger than 3 are not
supported, so `full` is available for k <= 3 only.
"""
import math

import numpy as np
import torch

DT = torch.float64
ORDER = {"pair": 2, "triple": 3}


def legendre(u, m):
    """Orthonormal shifted Legendre basis f_n(u) = sqrt(2n+1) P_n(2u-1), n = 0..m; shape u.shape + (m+1,)."""
    t = 2 * u - 1
    P = [torch.ones_like(t), t]
    for n in range(1, m):
        P.append(((2 * n + 1) * t * P[n] - n * P[n - 1]) / (n + 1))
    norm = torch.sqrt(2 * torch.arange(m + 1, dtype=u.dtype, device=u.device) + 1)
    return torch.stack(P[: m + 1], -1) * norm


def input_basis(Y, m):
    """Basis tables of the inputs without f_0: (n, k, m)."""
    return legendre(Y, m)[..., 1:]


def gauss_legendre(n, device="cpu"):
    x, w = np.polynomial.legendre.leggauss(n)
    return torch.tensor((x + 1) / 2, device=device), torch.tensor(w / 2, device=device)


def pairs(k):
    a, b = np.triu_indices(k, 1)
    return a, b


def order_of(basis, k):
    if basis == "full":
        if k > 3:
            raise ValueError("full family only for k <= 3")
        return k + 1
    return ORDER[basis]


def masks(k, m, basis, cap=None, device="cpu"):
    """Boolean masks of the allowed entries of each block (None for an absent block)."""
    o = order_of(basis, k)
    deg_x = torch.arange(m + 1, device=device)
    deg_y = torch.arange(1, m + 1, device=device)
    ok = (lambda tot: tot <= cap) if cap is not None else (lambda tot: torch.ones_like(tot, dtype=torch.bool))
    out = {"c": ok(deg_x)}
    out["W1"] = ok(deg_x[:, None] + deg_y[None, :])[:, None, :].expand(m + 1, k, m).clone()  # o >= 2 always
    out["W2"] = out["W3"] = None
    P = len(pairs(k)[0])
    if o >= 2 and k >= 2:
        rows2 = (deg_x == 0) | (o >= 3)
        tot = deg_x[:, None, None] + deg_y[None, :, None] + deg_y[None, None, :]
        out["W2"] = (rows2[:, None, None] & ok(tot))[:, None].expand(m + 1, P, m, m).clone()
    if o >= 3 and k >= 3:
        n3 = m + 1 if o >= 4 else 1
        pa, pb = pairs(k)
        third = torch.arange(k, device=device)[None, :] > torch.as_tensor(pb, device=device)[:, None]  # (P, k)
        tot = (deg_x[:n3, None, None, None] + deg_y[None, :, None, None] + deg_y[None, None, :, None]
               + deg_y[None, None, None, :])  # (n3, m, m, m)
        out["W3"] = (ok(tot)[:, None, :, :, None, :] & third[None, :, None, None, :, None]).expand(n3, P, m, m, k, m).clone()
    return out


def count_formula(k, m, basis):
    """Number of coefficients (including a_0) of the pair/triple families without a degree cap."""
    o = order_of(basis, k)
    return sum(math.comb(k + 1, r) * m ** r for r in range(0, min(o, k + 1) + 1))


class Model:
    """HCR neuron with block-stored coefficients; see the module docstring."""

    def __init__(self, k, m, basis, blocks, cap=None):
        self.k, self.m, self.basis, self.cap = k, m, basis, cap
        dev = blocks["c"].device
        self.mask = masks(k, m, basis, cap, dev)
        self.pa, self.pb = (torch.as_tensor(v, device=dev) for v in pairs(k))
        self.blocks = {}
        for name, mk in self.mask.items():
            if mk is None:
                self.blocks[name] = None
            else:
                self.blocks[name] = torch.where(mk, blocks[name], torch.zeros((), dtype=DT, device=dev))
        self._mats()

    def _mats(self):
        b, m = self.blocks, self.m
        self.Wc = b["c"]
        self.W1m = b["W1"].reshape(m + 1, -1).T
        self.W2m = b["W2"].reshape(m + 1, -1).T if b["W2"] is not None else None
        if b["W3"] is not None:
            n3, P = b["W3"].shape[:2]
            self.W3m = b["W3"].reshape(n3, P * m * m, -1).permute(1, 0, 2).reshape(P * m * m, -1)
        else:
            self.W3m = None

    @property
    def n_coef(self):
        return int(sum(int(v.sum()) for v in self.mask.values() if v is not None))

    def A(self, Fy):
        """A_i(y) for i = 0..m from input basis tables Fy (n, k, >= m); returns (n, m+1). D(y) = A[:, 0]."""
        Fy = Fy[..., : self.m]
        n = Fy.shape[0]
        A = self.Wc + Fy.reshape(n, -1) @ self.W1m
        if self.W2m is not None:
            Pf = (Fy[:, self.pa, :, None] * Fy[:, self.pb, None, :]).reshape(n, -1)
            A = A + Pf @ self.W2m
            if self.W3m is not None:
                n3 = self.blocks["W3"].shape[0]
                T = (Pf @ self.W3m).reshape(n, n3, -1)
                A3 = (T * Fy.reshape(n, 1, -1)).sum(-1)
                A = A + torch.nn.functional.pad(A3, (0, self.m + 1 - n3))
        return A

    def D(self, Y):
        return self.A(input_basis(Y, self.m))[:, 0]

    def sliced(self, m):
        """Degree-m sub-model (indices 0..m on every variable)."""
        b = self.blocks
        nb = {"c": b["c"][: m + 1], "W1": b["W1"][: m + 1, :, :m]}
        nb["W2"] = b["W2"][: m + 1, :, :m, :m] if b["W2"] is not None else None
        if b["W3"] is not None:
            n3 = b["W3"].shape[0]
            nb["W3"] = b["W3"][: min(n3, m + 1), :, :m, :m, :, :m]
        return Model(self.k, m, self.basis, nb, self.cap)

    def with_flat_D(self):
        """Control model: a_(0, j) = 0 for every j != 0, so D(y) = 1."""
        nb = {k: (None if v is None else v.clone()) for k, v in self.blocks.items()}
        nb["W1"][0] = 0
        if nb["W2"] is not None:
            nb["W2"][0] = 0
        if nb["W3"] is not None:
            nb["W3"][0] = 0
        return Model(self.k, self.m, self.basis, nb, self.cap)

    def chi_D(self):
        """sqrt(int (D-1)^2 dy) = sqrt(sum of squared a_(0, j), j != 0): distance of D from flat."""
        b = self.blocks
        s = (b["W1"][0] ** 2).sum()
        for name in ("W2", "W3"):
            if b[name] is not None:
                s = s + (b[name][0] ** 2).sum()
        return float(torch.sqrt(s))

    def to_numpy(self):
        return {k: v.cpu().numpy() for k, v in self.blocks.items() if v is not None}


def model_from_dense3(a, basis="triple"):
    """Block model (k = 2) from a dense d = 3 tensor a[i, j, l] (x, y, z)."""
    a = torch.as_tensor(a, dtype=DT)
    m = a.shape[0] - 1
    W1 = torch.stack([a[:, 1:, 0], a[:, 0, 1:]], 1)
    W2 = a[:, 1:, 1:][:, None]
    return Model(2, m, basis, {"c": a[:, 0, 0], "W1": W1, "W2": W2, "W3": None})


def dense3_from_model(model):
    """Inverse of model_from_dense3 (k = 2)."""
    b, m = model.blocks, model.m
    a = torch.zeros((m + 1,) * 3, dtype=DT)
    a[:, 0, 0] = b["c"]
    a[:, 1:, 0] = b["W1"][:, 0]
    a[:, 0, 1:] = b["W1"][:, 1]
    if b["W2"] is not None:
        a[:, 1:, 1:] = b["W2"][:, 0]
    return a


# ---------------------------------------------------------------- estimation
class Kahan:
    """Compensated running sum of tensors (fp64 accumulation across chunks)."""

    def __init__(self, like):
        self.s = torch.zeros_like(like)
        self.c = torch.zeros_like(like)

    def add(self, x):
        y = x - self.c
        t = self.s + y
        self.c = (t - self.s) - y
        self.s = t


class Estimator:
    """Direct moment estimates a_j = mean prod f_{j_i}(u_i) for every block of one family, plus SE_j.

    Feed chunks of uniform samples U (n, k+1) with add(); call model() at the end."""

    def __init__(self, k, m, basis, device):
        self.k, self.m, self.basis, self.dev = k, m, basis, device
        mk = masks(k, m, basis, None, device)
        self.has2, self.has3 = mk["W2"] is not None, mk["W3"] is not None
        self.n3 = mk["W3"].shape[0] if self.has3 else 0
        self.pa, self.pb = (torch.as_tensor(v, device=device) for v in pairs(k))
        self.n = 0
        self.acc = None

    def _sums(self, Fx, Fy):
        n = Fx.shape[0]
        Fyf = Fy.reshape(n, -1)
        out = [Fx.sum(0), Fx.T @ Fyf]
        if self.has2:
            Pf = (Fy[:, self.pa, :, None] * Fy[:, self.pb, None, :]).reshape(n, -1)
            out.append(Fx.T @ Pf)
            if self.has3:
                out.append(torch.stack([(Pf * Fx[:, i: i + 1]).T @ Fyf for i in range(self.n3)]))
        return out

    def add(self, U):
        Fx = legendre(U[:, 0], self.m)
        Fy = input_basis(U[:, 1:], self.m)
        s1 = self._sums(Fx, Fy)
        s2 = self._sums(Fx * Fx, Fy * Fy)
        if self.acc is None:
            self.acc = [Kahan(x) for x in s1 + s2]
        for a, x in zip(self.acc, s1 + s2):
            a.add(x)
        self.n += U.shape[0]

    def model(self):
        """Returns (Model, SE blocks)."""
        n, m, k = self.n, self.m, self.k
        h = len(self.acc) // 2
        mean = [a.s / n for a in self.acc[:h]]
        sq = [a.s / n for a in self.acc[h:]]
        se = [torch.sqrt(torch.clamp(q - x * x, min=0) / n) for x, q in zip(mean, sq)]
        P = len(self.pa)

        def shape(lst):
            b = {"c": lst[0], "W1": lst[1].reshape(m + 1, k, m), "W2": None, "W3": None}
            if self.has2:
                b["W2"] = lst[2].reshape(m + 1, P, m, m)
                if self.has3:
                    b["W3"] = lst[3].reshape(self.n3, P, m, m, k, m)
            return b

        return Model(k, m, self.basis, shape(mean)), shape(se)


def chunk_for(k, m, basis, base, budget=2.0e8):
    """Samples per chunk so that one pair-feature matrix stays below `budget` elements."""
    P = k * (k - 1) // 2
    nf = P * m * m if order_of(basis, k) >= 2 and k >= 2 else k * m
    return int(max(1024, min(base, budget // max(nf, 1))))


# ---------------------------------------------------------------- population coefficients (Gaussian copula)
class GaussianMoments:
    """Exact E[prod f_{d_i}(Phi(z_i))] for z ~ N(0, R_sub), |sub| <= 3, by Gauss-Hermite quadrature (cached)."""

    def __init__(self, m, nodes=48):
        z, w = np.polynomial.hermite_e.hermegauss(nodes)
        self.z, self.w = z, w / w.sum()
        self.m = m
        self.cache = {}

    def table(self, Rs):
        Rs = np.asarray(Rs, float)
        key = tuple(np.round(Rs[np.triu_indices(len(Rs), 1)], 14))
        if key not in self.cache:
            from scipy.special import ndtr
            d = len(Rs)
            L = np.linalg.cholesky(Rs)
            grids = np.meshgrid(*([self.z] * d), indexing="ij")
            G = np.stack([g.ravel() for g in grids], 1)
            W = np.prod(np.meshgrid(*([self.w] * d), indexing="ij"), 0).ravel()
            X = G @ L.T
            F = [legendre(torch.tensor(ndtr(X[:, i])), self.m).numpy() for i in range(d)]
            if d == 2:
                T = np.einsum("n,ni,nj->ij", W, F[0], F[1])
            else:
                T = np.einsum("n,ni,nj,nl->ijl", W, F[0], F[1], F[2])
            self.cache[key] = T
        return self.cache[key]


def population_model(R, m, basis, nodes=48, device="cpu"):
    """Exact HCR coefficients of the Gaussian copula with correlation R (order x, y_1..y_k)."""
    R = np.asarray(R, float)
    k = R.shape[0] - 1
    if order_of(basis, k) > 3:
        raise ValueError("population coefficients implemented up to triple interactions")
    gm = GaussianMoments(m, nodes)
    pa, pb = pairs(k)
    P = len(pa)
    c = np.zeros(m + 1)
    c[0] = 1.0
    W1 = np.zeros((m + 1, k, m))
    for a in range(k):
        W1[:, a, :] = gm.table(R[np.ix_([0, a + 1], [0, a + 1])])[:, 1:]
    W1[0] = 0.0  # uniform marginals: E f_p(y_a) = 0 exactly
    blocks = {"c": c, "W1": W1, "W2": None, "W3": None}
    if k >= 2:
        W2 = np.zeros((m + 1, P, m, m))
        for t, (a, b) in enumerate(zip(pa, pb)):
            W2[0, t] = gm.table(R[np.ix_([a + 1, b + 1], [a + 1, b + 1])])[1:, 1:]
            if order_of(basis, k) >= 3:
                W2[1:, t] = gm.table(R[np.ix_([0, a + 1, b + 1], [0, a + 1, b + 1])])[1:, 1:, 1:]
        blocks["W2"] = W2
    if order_of(basis, k) >= 3 and k >= 3:
        W3 = np.zeros((1, P, m, m, k, m))
        for t, (a, b) in enumerate(zip(pa, pb)):
            for cc in range(b + 1, k):
                idx = [a + 1, b + 1, cc + 1]
                W3[0, t, :, :, cc, :] = gm.table(R[np.ix_(idx, idx)])[1:, 1:, 1:]
        blocks["W3"] = W3
    blocks = {kk: (None if v is None else torch.tensor(v, dtype=DT, device=device)) for kk, v in blocks.items()}
    return Model(k, m, basis, blocks)
