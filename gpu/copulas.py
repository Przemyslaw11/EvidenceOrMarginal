"""Ground-truth copulas over (x, y_1..y_k): correlation structures, samplers and analytic conditionals.

Samplers map *base variates* (numpy, generated once per chunk and shared by all cases of a job) to
uniforms, which gives common random numbers across rho_S, neuron types and bases. Conditionals are
returned in normal-score coordinates: p_s(s | y) = c(Phi(s) | y) phi(s).
"""
import logging
import math

import numpy as np
import torch
from scipy import special as sps

from .core import DT

log = logging.getLogger(__name__)
PD_TOL = 1e-6
SQ2PI = math.sqrt(2 * math.pi)


def r_from_rhoS(rho):
    return 2 * math.sin(math.pi * rho / 6)


def tau_from_rhoS(rho):
    """Kendall tau of a Gaussian pair with Spearman rho_S."""
    return 2 / math.pi * math.asin(r_from_rhoS(rho))


def min_eig(R):
    return float(np.linalg.eigvalsh(R).min())


def coupled_R(typ, k, rho, c_max=0.5, c_scale=0.8):
    """Correlation matrix (x, y_1..y_k) with equicorrelated inputs; returns (R, info).

    The output coupling c is reduced by factors of 0.95 (logged in info) until min eig > PD_TOL."""
    r = r_from_rhoS(rho)
    if typ == "SUM":
        c = min(c_max, c_scale * math.sqrt((1 + (k - 1) * r) / k))
        sign = np.ones(k)
    elif typ == "CONTRAST":
        if k % 2:
            raise ValueError("CONTRAST needs even k")
        c = min(c_max, c_scale * math.sqrt((1 - r) / k))
        sign = np.r_[np.ones(k // 2), -np.ones(k // 2)]
    elif typ == "EXCH":
        c, sign = r, np.ones(k)
    else:
        raise ValueError(typ)
    c0, n_red = c, 0
    while True:
        R = np.full((k + 1, k + 1), r)
        R[0, 1:] = R[1:, 0] = c * sign
        np.fill_diagonal(R, 1.0)
        ev = min_eig(R)
        if ev > PD_TOL:
            break
        c *= 0.95
        n_red += 1
        log.warning("coupled_R(%s, k=%d, rho=%.3f): min eig %.2e, reducing c to %.4f", typ, k, rho, ev, c)
    return R, dict(r_yy=r, c=c, c_requested=c0, c_reductions=n_red, min_eig=ev)


def lkj(d, eta, rng):
    """Random correlation matrix from the LKJ(eta) distribution (C-vine method, Lewandowski et al. 2009)."""
    b = eta + (d - 1) / 2
    P = np.zeros((d, d))
    S = np.eye(d)
    for k in range(d - 1):
        b -= 0.5
        for i in range(k + 1, d):
            P[k, i] = 2 * rng.beta(b, b) - 1
            p = P[k, i]
            for l in range(k - 1, -1, -1):
                p = p * math.sqrt((1 - P[l, i] ** 2) * (1 - P[l, k] ** 2)) + P[l, i] * P[l, k]
            S[k, i] = S[i, k] = p
    return S


# ---------------------------------------------------------------- base variates
KINDS_ALL = ("Z", "E", "u", "th", "w", "chi3", "chi10")


def base_variates(rng, n, d, kinds):
    """Shared random numbers for one chunk. Always drawn in the order of KINDS_ALL for the given kinds."""
    out = {}
    for kd in KINDS_ALL:
        if kd not in kinds:
            continue
        if kd == "Z":
            out[kd] = rng.standard_normal((n, d))
        elif kd == "E":
            out[kd] = rng.standard_exponential((n, d))
        elif kd == "u":
            out[kd] = rng.random(n)
        elif kd == "th":
            out[kd] = rng.random(n) * math.pi
        elif kd == "w":
            out[kd] = rng.standard_exponential(n)
        elif kd == "chi3":
            out[kd] = rng.chisquare(3, n)
        elif kd == "chi10":
            out[kd] = rng.chisquare(10, n)
    return out


def _phi(s):
    return torch.exp(-0.5 * s * s) / SQ2PI


def _t_logpdf(z, nu):
    return (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
            - (nu + 1) / 2 * torch.log1p(z * z / nu))


def _t_series(nu, extra=64):
    """Coefficients (a_j for odd nu, b_j for even nu) of the head (j < n) and tail (n <= j < n + extra); see `_t_lower`."""
    odd = nu % 2 == 1
    n = (nu - 1) // 2 if odd else nu // 2
    coef = [1.0]
    for jj in range(1, n + extra):
        coef.append(coef[-1] * (2 * jj / (2 * jj + 1) if odd else (2 * jj - 1) / (2 * jj)))
    return n, coef


def _t_lower(phi, nu):
    """P(T <= -|x|) for integer nu, with phi = atan2(sqrt(nu), |x|) in (0, pi/2]. With s2 = sin^2 phi, c = cos phi:
    odd nu: (phi - sin c sum_(j<n) a_j s2^j) / pi, n = (nu - 1) / 2, a_j = a_(j-1) 2j / (2j + 1);
    even nu: (1 - c sum_(j<n) b_j s2^j) / 2, n = nu / 2, b_j = b_(j-1) (2j - 1) / (2j).
    In the tails (s2 <= 1/2) the same values are the remainders sum_(j>=n) of the series phi / c = sin sum a_j s2^j
    and 1 / c = sum b_j s2^j (64 terms), which avoids the cancellation of the finite form."""
    n, coef = _t_series(nu)
    cf = torch.tensor(coef, dtype=phi.dtype, device=phi.device)
    sn, c = torch.sin(phi), torch.cos(phi)
    pw = (sn * sn)[..., None] ** torch.arange(len(coef), dtype=phi.dtype, device=phi.device)
    head, tail = pw[..., :n] @ cf[:n], pw[..., n:] @ cf[n:]
    if nu % 2:
        direct, series = (phi - sn * c * head) / math.pi, sn * c * tail / math.pi
    else:
        direct, series = 0.5 * (1 - c * head), 0.5 * c * tail
    return torch.where(sn * sn <= 0.5, series, direct)


def t_cdf(x, nu):
    """Student-t CDF for integer nu in closed form (torch, any device); both tails keep their relative accuracy."""
    lo = _t_lower(torch.atan2(torch.full_like(x, math.sqrt(nu)), x.abs()), nu)
    return torch.where(x > 0, 1 - lo, lo)


def t_ppf(u, nu, bisect=12, newton=5):
    """Inverse of `t_cdf` for the lower-tail probability p = min(u, 1 - u), in psi = log phi: bisection on
    [log 1e-18, log pi/2], then Newton steps on log P(psi) (nearly linear there: P ~ phi^nu), kept in the bracket."""
    p = torch.minimum(u, 1 - u)
    lp = torch.log(p)
    a = torch.full_like(u, math.log(1e-18))
    b = torch.full_like(u, math.log(math.pi / 2))
    for _ in range(bisect):
        m = 0.5 * (a + b)
        below = _t_lower(torch.exp(m), nu) < p
        a, b = torch.where(below, m, a), torch.where(below, b, m)
    psi = 0.5 * (a + b)
    rs = math.sqrt(nu)
    for _ in range(newton):
        phi = torch.exp(psi)
        P = _t_lower(phi, nu)
        sn = torch.sin(phi)
        x = rs * torch.cos(phi) / sn
        dP = torch.exp(_t_logpdf(x, nu)) * rs / (sn * sn) * phi  # dP/dpsi
        psi = torch.clamp(psi - (torch.log(P) - lp) * P / dP, a, b)
    phi = torch.exp(psi)
    x = rs * torch.cos(phi) / torch.sin(phi)  # |x| = sqrt(nu) cot(phi)
    return torch.where(u > 0.5, x, -x)


class Gaussian:
    kinds = ("Z",)

    def __init__(self, R, device):
        self.R = np.asarray(R, float)
        assert min_eig(self.R) > PD_TOL, "R not positive definite"
        self.d = len(self.R)
        self.L = torch.tensor(np.linalg.cholesky(self.R), dtype=DT, device=device)
        sig = self.R[0, 1:]
        beta = np.linalg.solve(self.R[1:, 1:], sig)
        self.beta = torch.tensor(beta, dtype=DT, device=device)
        self.sigma = math.sqrt(1 - sig @ beta)
        self.dev = device

    def uniforms(self, base):
        Z = torch.as_tensor(base["Z"], dtype=DT, device=self.dev)
        return torch.clamp(torch.special.ndtr(Z @ self.L.T), 1e-15, 1 - 1e-15)

    def cond_s(self, s, Uy):
        mu = torch.special.ndtri(Uy) @ self.beta
        return _phi((s[None, :] - mu[:, None]) / self.sigma) / self.sigma


class StudentT(Gaussian):
    def __init__(self, R, nu, device):
        super().__init__(R, device)
        self.nu = nu
        self.kinds = ("Z", f"chi{nu}")
        self.Syy_inv = torch.tensor(np.linalg.inv(self.R[1:, 1:]), dtype=DT, device=device)

    def uniforms(self, base):
        Z = torch.as_tensor(base["Z"], dtype=DT, device=self.dev)
        W = torch.as_tensor(base[f"chi{self.nu}"], dtype=DT, device=self.dev)
        T = (Z @ self.L.T) / torch.sqrt(W / self.nu)[:, None]
        return torch.clamp(t_cdf(T, self.nu), 1e-15, 1 - 1e-15)

    def cond_s(self, s, Uy):
        nu, k = self.nu, Uy.shape[1]
        ty = t_ppf(Uy, nu)
        tx = torch.as_tensor(sps.stdtrit(nu, sps.ndtr(s.cpu().numpy())), dtype=DT, device=self.dev)
        Q = ((ty @ self.Syy_inv) * ty).sum(1)
        mu = ty @ self.beta
        scale = self.sigma * torch.sqrt((nu + Q) / (nu + k))
        z = (tx[None, :] - mu[:, None]) / scale[:, None]
        logc = _t_logpdf(z, nu + k) - torch.log(scale)[:, None] - _t_logpdf(tx, nu)[None, :]
        return torch.exp(logc) * _phi(s)[None, :]


class Clayton:
    """Exchangeable (k+1)-dimensional Clayton copula, theta > 0 (Marshall-Olkin sampler)."""
    kinds = ("E", "u")

    def __init__(self, theta, d, device):
        self.theta, self.d, self.dev = theta, d, device

    def uniforms(self, base):
        V = sps.gammaincinv(1 / self.theta, base["u"])  # Gamma(1/theta) frailty by inversion: common random numbers
        U = (1 + base["E"] / V[:, None]) ** (-1 / self.theta)
        return torch.as_tensor(np.clip(U, 1e-15, 1 - 1e-15), dtype=DT, device=self.dev)

    def cond_s(self, s, Uy):
        th, k = self.theta, Uy.shape[1]
        T = 1 + (Uy ** (-th) - 1).sum(1)  # 1 + sum phi(y)
        lx = torch.log(torch.special.ndtr(s))
        lc = (math.log1p(k * th) - (th + 1) * lx[None, :]
              - (1 / th + k + 1) * torch.log(T[:, None] + torch.expm1(-th * lx)[None, :])
              + (1 / th + k) * torch.log(T)[:, None])
        return torch.exp(lc) * _phi(s)[None, :]


class Gumbel:
    """Exchangeable (k+1)-dimensional Gumbel copula, theta >= 1 (Marshall-Olkin, positive-stable frailty)."""
    kinds = ("E", "th", "w")

    def __init__(self, theta, d, device):
        self.theta, self.d, self.dev = theta, d, device
        self.alpha = 1 / theta
        self.g = self._g_terms(d)

    def _g_terms(self, nmax):
        """(-1)^n psi^(n)(t) = psi(t) * sum_j c[n][j] t^(j*alpha - n), psi(t) = exp(-t^alpha); all c >= 0."""
        a = self.alpha
        terms = [{0: 1.0}]
        for n in range(nmax):
            nxt = {}
            for j, c in terms[n].items():
                nxt[j + 1] = nxt.get(j + 1, 0.0) + a * c
                if n - j * a != 0:
                    nxt[j] = nxt.get(j, 0.0) + (n - j * a) * c
            terms.append(nxt)
        return terms

    def _log_g(self, n, lt):
        js = sorted(self.g[n])
        lc = torch.tensor([math.log(self.g[n][j]) for j in js], dtype=DT, device=lt.device)
        ex = torch.tensor([j * self.alpha - n for j in js], dtype=DT, device=lt.device)
        return torch.logsumexp(lc + ex * lt[..., None], -1)

    def uniforms(self, base):
        a, th, w = self.alpha, base["th"], base["w"]
        if a == 1.0:
            V = np.ones_like(w)
        else:  # Kanter's representation of the positive stable law with Laplace transform exp(-t^a)
            V = (np.sin(a * th) / np.sin(th) ** (1 / a)) * (np.sin((1 - a) * th) / w) ** ((1 - a) / a)
        U = np.exp(-((base["E"] / V[:, None]) ** a))
        return torch.as_tensor(np.clip(U, 1e-15, 1 - 1e-15), dtype=DT, device=self.dev)

    def cond_s(self, s, Uy):
        th, a, k = self.theta, self.alpha, Uy.shape[1]
        t = ((-torch.log(Uy)) ** th).sum(1)
        lx = torch.log(torch.special.ndtr(s))
        nlx = -lx
        phx = nlx ** th
        tt = t[:, None] + phx[None, :]
        lt_t = torch.log(tt)
        lc = (self._log_g(k + 1, lt_t) - tt ** a - self._log_g(k, torch.log(t))[:, None] + (t ** a)[:, None]
              + math.log(th) + (th - 1) * torch.log(nlx)[None, :] - lx[None, :])
        return torch.exp(lc) * _phi(s)[None, :]


class Independence(Gaussian):
    def __init__(self, d, device):
        super().__init__(np.eye(d), device)


def theta_from_tau(family, tau):
    return 2 * tau / (1 - tau) if family == "clayton" else 1 / (1 - tau)


def make_copula(family, typ, k, rho, device, cfg_c=None):
    """Copula for one case of the sweep (inputs at Spearman-equivalent rho_S, matched by Kendall tau)."""
    cfg_c = cfg_c or {}
    if family in ("clayton", "gumbel"):
        tau = tau_from_rhoS(rho)
        if tau == 0:
            return Independence(k + 1, device), dict(tau=0.0, theta=float("nan"))
        th = theta_from_tau(family, tau)
        cls = Clayton if family == "clayton" else Gumbel
        return cls(th, k + 1, device), dict(tau=tau, theta=th)
    R, info = coupled_R(typ, k, rho, cfg_c.get("c_max", 0.5), cfg_c.get("c_scale", 0.8))
    info["tau"] = tau_from_rhoS(rho)
    if family == "gauss":
        return Gaussian(R, device), info
    if family in ("t3", "t10"):
        return StudentT(R, int(family[1:]), device), info
    raise ValueError(family)
