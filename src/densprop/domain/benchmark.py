"""The layers of the time and energy benchmark (Stage 7): HCR-CD, HCR-CAL, MLP and KAN with matched parameter counts.

The HCR layer stacks the matrices of W neurons (densprop.domain.neuron.Model), so it is the computation of the study's engine.
See densprop.app.energy for the measurement.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from densprop.domain.neuron import Model, gauss_legendre, input_basis, legendre, pairs

W_UNITS, M_DEG, S_SAMPLES, G_NODES, EPS = 64, 4, 64, 192, 0.01
KS, BATCHES, LAYERS = (2, 8, 32), (256, 16384), ("HCR-CD", "HCR-CAL", "MLP", "KAN")
REPEATS, REPEAT_S, WARM_S, IDLE_S = 30, 2.0, 3.0, 30.0
ELEM_BUDGET = 2**27  # elements of the largest intermediate tensor per chunk


# ---------------------------------------------------------------- layers
class HCRLayer:
    """W HCR neurons on the same k inputs (pair basis), stacked from neuron Model matrices."""

    def __init__(self, k, W, m, dtype, device, seed=0):
        g = torch.Generator().manual_seed(seed)
        P = len(pairs(k)[0])
        self.models = []
        for _ in range(W):
            blocks = {
                "c": torch.cat([torch.ones(1), torch.zeros(m)]).double(),
                "W1": 0.1 * torch.randn(m + 1, k, m, generator=g, dtype=torch.float64),
                "W2": 0.1 * torch.randn(m + 1, P, m, m, generator=g, dtype=torch.float64) / math.sqrt(max(P, 1)),
                "W3": None,
            }
            self.models.append(Model(k, m, "pair", blocks))
        self.k, self.W, self.m = k, W, m
        self.n_params = W * self.models[0].n_coef
        cat = lambda name: torch.cat([getattr(md, name) for md in self.models], -1).to(device, dtype)
        self.Wc, self.W1m = cat("Wc"), cat("W1m")
        self.W2m = cat("W2m") if k >= 2 else None
        self.pa, self.pb = (torch.as_tensor(v, device=device) for v in pairs(k))
        xg, wg = gauss_legendre(G_NODES, device)
        self.wg = wg.to(dtype)
        self.Fx = legendre(xg.to(dtype), m)  # (G, m+1)

    def A(self, F):
        """A_i for all W neurons from feature tables F (n, k, m): (n, W, m+1)."""
        n = F.shape[0]
        A = self.Wc + F.reshape(n, -1) @ self.W1m
        if self.W2m is not None:
            A = A + (F[:, self.pa, :, None] * F[:, self.pb, None, :]).reshape(n, -1) @ self.W2m
        return A.reshape(n, self.W, self.m + 1)

    def const_denom(self, b):
        A = self.A(b)
        return A / A[..., :1]

    def exact_cal(self, Y):
        """Y: (B, S, k) samples of B beliefs -> (B, W, m+1) coefficients of the calibrated output density."""
        B, S, k = Y.shape
        per = S * self.W * max(G_NODES, (len(self.pa) * self.m**2) // self.W + 1)
        bc = max(1, ELEM_BUDGET // per)
        out = []
        for st in range(0, B, bc):
            Yc = Y[st : st + bc].reshape(-1, k)
            N = torch.clamp(self.A(input_basis(Yc, self.m)) @ self.Fx.T, min=EPS)  # (n, W, G)
            p = N / (N @ self.wg)[..., None]
            p = p.reshape(-1, S, self.W, G_NODES).mean(1)
            out.append((p * self.wg) @ self.Fx)
        return torch.cat(out)


class KANLayer(torch.nn.Module):
    """B-spline KAN layer (efficient-kan formulation): SiLU base + cubic B-splines on a uniform grid over [-1, 1]."""

    def __init__(self, n_in, n_out, grid=5, order=3):
        super().__init__()
        h = 2.0 / grid
        self.register_buffer("t", torch.arange(-order, grid + order + 1, dtype=torch.float64) * h - 1.0)
        self.order = order
        self.base = torch.nn.Parameter(torch.randn(n_in, n_out) / math.sqrt(n_in))
        self.spline = torch.nn.Parameter(0.1 * torch.randn(n_in * (grid + order), n_out) / math.sqrt(n_in))

    def forward(self, x):
        t, xe = self.t, x[..., None]
        bases = ((xe >= t[:-1]) & (xe < t[1:])).to(x.dtype)
        for p in range(1, self.order + 1):
            bases = (xe - t[: -(p + 1)]) / (t[p:-1] - t[: -(p + 1)]) * bases[..., :-1] + (t[p + 1 :] - xe) / (t[p + 1 :] - t[1:-p]) * bases[..., 1:]
        return torch.nn.functional.silu(x) @ self.base + bases.reshape(x.shape[0], -1) @ self.spline


def mlp(k: int, H: int, W: int) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(k, H), torch.nn.GELU(), torch.nn.Linear(H, W))


def kan(k: int, H: int, W: int) -> torch.nn.Module:
    return torch.nn.Sequential(KANLayer(k, H), KANLayer(H, W))


def matched_width(kind: str, k: int, W: int, P: int) -> int:
    """Hidden width H of the 2-layer baseline whose parameter count is closest to P."""
    if kind == "MLP":
        return max(1, round((P - W) / (k + 1 + W)))
    return max(1, round(P / (9 * (k + W))))  # KAN, grid 5, order 3: (grid + order + 1) = 9 weights per edge


def n_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def build(layer: str, k: int, B: int, dtype: torch.dtype, device: str, seed: int = 0) -> tuple[Callable[[], object], int, str]:
    """Returns (forward callable, parameter count, description of the inputs)."""
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    hcr = HCRLayer(k, W_UNITS, M_DEG, dtype, device, seed)
    P = hcr.n_params
    if layer == "HCR-CD":
        b = (0.2 * torch.randn(B, k, M_DEG, generator=g, dtype=torch.float64)).to(device, dtype)
        return (lambda: hcr.const_denom(b)), P, f"{B} beliefs x {k} coefficient vectors"
    if layer == "HCR-CAL":
        Y = torch.rand(B, S_SAMPLES, k, generator=g, dtype=torch.float64).to(device, dtype)
        return (lambda: hcr.exact_cal(Y)), P, f"{B} beliefs x {S_SAMPLES} samples"
    H = matched_width(layer, k, W_UNITS, P)
    net = (mlp if layer == "MLP" else kan)(k, H, W_UNITS).to(device, dtype).eval()
    x = (2 * torch.rand(B, k, generator=g, dtype=torch.float64) - 1).to(device, dtype)

    def fwd():
        with torch.no_grad():
            return net(x)

    return fwd, n_params(net), f"{B} inputs, hidden width {H}"


# ---------------------------------------------------------------- self-check (CPU)
def check() -> None:
    torch.manual_seed(0)
    k, W, B = 3, 4, 5
    lay = HCRLayer(k, W, M_DEG, torch.float64, "cpu")
    b = 0.2 * torch.randn(B, k, M_DEG, dtype=torch.float64)
    cd = lay.const_denom(b)
    for j, md in enumerate(lay.models):
        A = md.A(b)
        assert torch.allclose(cd[:, j], A / A[:, :1], atol=1e-12, rtol=0), "layer differs from the neuron model"
    Y = torch.rand(B, S_SAMPLES, k, dtype=torch.float64)
    cal = lay.exact_cal(Y)
    assert torch.allclose(cal[..., 0], torch.ones(B, W, dtype=torch.float64), atol=1e-10), "EXACT_CAL not normalised"
    # with D > 0 and N >= eps everywhere, EXACT_CAL of a single-sample belief is the neuron's conditional density
    one = lay.exact_cal(Y[:, :1])
    A = lay.A(input_basis(Y[:, 0], M_DEG))
    Nmin = (A @ lay.Fx.T).min()
    if Nmin > EPS:
        assert torch.allclose(one, A / A[..., :1], atol=1e-10), "EXACT_CAL != N/D for a point belief"
    x = torch.rand(7, 2, dtype=torch.float64) * 2 - 1
    lk = KANLayer(2, 3).double()
    bases_sum = None
    with torch.no_grad():  # B-splines of order 3 form a partition of unity inside [-1, 1]
        t, xe = lk.t, x[..., None]
        bs = ((xe >= t[:-1]) & (xe < t[1:])).double()
        for p in range(1, 4):
            bs = (xe - t[: -(p + 1)]) / (t[p:-1] - t[: -(p + 1)]) * bs[..., :-1] + (t[p + 1 :] - xe) / (t[p + 1 :] - t[1:-p]) * bs[..., 1:]
        bases_sum = bs.sum(-1)
    assert torch.allclose(bases_sum, torch.ones_like(bases_sum), atol=1e-12), "B-spline basis not a partition of unity"
    for kk in KS:
        P = HCRLayer(kk, 1, M_DEG, torch.float64, "cpu").n_params * W_UNITS
        for kind in ("MLP", "KAN"):
            got = n_params((mlp if kind == "MLP" else kan)(kk, matched_width(kind, kk, W_UNITS, P), W_UNITS))
            # one hidden unit changes the count by (k + W) (MLP) or 9 (k + W) (KAN): at k = 2 the KAN is 6.5 % off
            assert abs(got - P) / P < 0.10, f"{kind} k={kk}: {got} vs {P} parameters"
