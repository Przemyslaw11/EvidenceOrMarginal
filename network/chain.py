"""Stage 4: a chain of HCR neurons, its Gaussian teacher, and three ways of propagating a belief through it.

Teacher (normal scores s = Phi^{-1}(u)): s_y ~ N(0, R_yy) equicorrelated; s_h1 = beta . s_y + sigma_1 e_1 with the SUM
coupling of the single-neuron study; for l >= 2, s_hl = a s_h(l-1) + b s_wl + sqrt(1 - a^2 - b^2) e_l with exogenous
side inputs w_l ~ N(0, 1). The structure is Markov by construction, and w_l is independent of everything before it,
so the product belief rho_(l-1)(h) q_w(w) used by the paper's propagation is exact for this teacher.

Student: neuron 1 models (h_1; y_1..y_k) with the pair or triple basis; neuron l >= 2 models (h_l; h_(l-1), w_l) (k = 2).
Layer 1 is propagated by the Monte Carlo engine (gpu.propagate.evaluate_case); layers >= 2 deterministically on a
Gauss-Legendre grid in (h, w), with q_w uniform:
    CONST_DENOM  closed form  c_l,i = (c_i + sum_j W1[i,0,j-1] c_(l-1),j) / (1 + sum_j W1[0,0,j-1] c_(l-1),j)
    EXACT_UNCAL  e_l,i = int int A_i/A_0 rho_(l-1)(h) dh dw, defined only if rho_(l-1) >= 0 and D_l >= D_floor on its support
    EXACT_CAL    rho_l(x) = int int max(N, eps)/Z rho_(l-1)(h) dh dw with the calibrated rho_(l-1)
"""

import math

import numpy as np
import torch

from gpu import copulas as C
from gpu.core import DT, Estimator, gauss_legendre, input_basis, legendre
from gpu.propagate import evaluate_case, min_D, tv_stats

NAN = float("nan")


class Teacher:
    def __init__(self, k, rho, a_prev, b_side, cfg_c=None):
        cfg_c = cfg_c or {}
        self.k, self.rho = k, rho
        self.R1, self.info = C.coupled_R("SUM", k, rho, cfg_c.get("c_max", 0.5), cfg_c.get("c_scale", 0.8))
        sig = self.R1[0, 1:]
        self.beta = np.linalg.solve(self.R1[1:, 1:], sig)
        self.s1 = math.sqrt(1 - sig @ self.beta)
        self.a, self.b = a_prev, b_side
        e2 = 1 - a_prev**2 - b_side**2
        assert e2 > C.PD_TOL, "teacher noise variance must be positive"
        self.e = math.sqrt(e2)
        self.Lyy = np.linalg.cholesky(self.R1[1:, 1:])

    def sigma(self, L):
        """Conditional sd of s_hl given (y, w_2..w_l), l = 1..L."""
        out, v = [], self.s1**2
        for l in range(1, L + 1):
            if l > 1:
                v = self.a**2 * v + self.e**2
            out.append(math.sqrt(v))
        return out

    def sample(self, rng, n, L):
        """Uniforms: y (n, k), w (n, L) with w[:, 0] unused, h (n, L)."""
        Z = rng.standard_normal((n, self.k + 2 * L))
        sy = Z[:, : self.k] @ self.Lyy.T
        sw = np.zeros((n, L))
        sw[:, 1:] = Z[:, self.k + 1 : self.k + L]
        sh = np.zeros((n, L))
        sh[:, 0] = sy @ self.beta + self.s1 * Z[:, self.k]
        for l in range(1, L):
            sh[:, l] = self.a * sh[:, l - 1] + self.b * sw[:, l] + self.e * Z[:, self.k + L + l]
        from scipy.special import ndtr

        cl = lambda v: np.clip(ndtr(v), 1e-15, 1 - 1e-15)
        return dict(y=cl(sy), w=cl(sw), h=cl(sh), sy=sy, sw=sw, sh=sh)

    def h_given_y(self, rng, Y, L):
        """Teacher samples of h_L for given input uniforms Y (used as e2e targets under non-data beliefs)."""
        from scipy.special import ndtr, ndtri

        sy = ndtri(np.clip(Y, 1e-15, 1 - 1e-15))
        sh = sy @ self.beta + self.s1 * rng.standard_normal(len(Y))
        for _ in range(1, L):
            sh = self.a * sh + self.b * rng.standard_normal(len(Y)) + self.e * rng.standard_normal(len(Y))
        return np.clip(ndtr(sh), 1e-15, 1 - 1e-15)


def neuron_inputs(data, l):
    """(output uniforms, input uniforms) of neuron l (1-based) from teacher samples."""
    if l == 1:
        return data["h"][:, 0], data["y"]
    return data["h"][:, l - 1], np.stack([data["h"][:, l - 2], data["w"][:, l - 1]], 1)


def estimate_student(data, L, k, m, basis, device):
    models = []
    for l in range(1, L + 1):
        x, Y = neuron_inputs(data, l)
        est = Estimator(k if l == 1 else 2, m, basis if l == 1 else "triple", device)
        est.add(torch.as_tensor(np.column_stack([x, Y]), dtype=DT, device=device))
        models.append(est.model()[0])
    return models


# ---------------------------------------------------------------- deterministic layers >= 2
class LayerGrid:
    """Gauss-Legendre grid in (h, w) and x for the deterministic layers."""

    def __init__(self, n, m, device):
        self.u, self.wt = gauss_legendre(n, device)
        H, W = torch.meshgrid(self.u, self.u, indexing="ij")
        self.Y = torch.stack([H.reshape(-1), W.reshape(-1)], 1)
        self.Fh = legendre(self.u, m)  # (n, m+1)
        self.n = n


def cd_layer(model, c_prev):
    """CONST_DENOM through a (h_prev, w) neuron for the belief rho_prev(h) x uniform(w); c_prev (..., m+1)."""
    b = model.blocks
    num = b["c"] * c_prev[..., :1] + c_prev[..., 1:] @ b["W1"][:, 0, :].T
    return num / num[..., :1]


def kernels(model, lg, sgrid, eps_list, floor):
    """Kernels over h (integrated over w) for EXACT_UNCAL, EXACT_CAL and the D >= floor mask."""
    m, n = model.m, lg.n
    A = model.A(input_basis(lg.Y, m)).reshape(n, n, m + 1)  # (h, w, i)
    A0 = A[..., 0]
    ok = A0 >= floor
    Q = torch.where(ok[..., None], A / torch.where(ok, A0, torch.ones_like(A0))[..., None], torch.zeros_like(A))
    K_eu = torch.einsum("hwi,w->ih", Q, lg.wt)  # (m+1, h)
    Fx_g, Fx_s = sgrid.Fg[:, : m + 1], sgrid.F[:, : m + 1]
    Ng = torch.einsum("hwi,xi->xhw", A, Fx_g)
    Ns = torch.einsum("hwi,xi->xhw", A, Fx_s)
    K_cal_g, K_cal_s = [], []
    for eps in eps_list:
        Z = torch.einsum("xhw,x->hw", torch.clamp(Ng, min=eps), sgrid.wg)
        K_cal_g.append(torch.einsum("xhw,w->xh", torch.clamp(Ng, min=eps) / Z, lg.wt))
        K_cal_s.append(torch.einsum("xhw,w->xh", torch.clamp(Ns, min=eps) / Z, lg.wt) * sgrid.phi[:, None])
    return dict(K_eu=K_eu, okmask=ok, K_cal_g=torch.stack(K_cal_g), K_cal_s=torch.stack(K_cal_s))


def min_on_square(model, n=401, device="cpu"):
    g = torch.linspace(0, 1, n, dtype=DT, device=device)
    H, W = torch.meshgrid(g, g, indexing="ij")
    return model.D(torch.stack([H.reshape(-1), W.reshape(-1)], 1)).reshape(n, n), g


def poly_on(c, x):
    return c @ legendre(x, c.shape[-1] - 1).T


def gold_layers(teacher, Ys, Ws, sgrid, R):
    """GOLD_l on the s-grid for l = 1..L, rows: R replicates + pooled. Ys (M, k) belief samples of y, Ws (M, L)."""
    from scipy.special import ndtri

    L = Ws.shape[1]
    mu = torch.as_tensor(ndtri(Ys.cpu().numpy()) @ teacher.beta, dtype=DT, device=sgrid.s.device)
    sw = torch.special.ndtri(Ws)
    sig = teacher.sigma(L)
    M = mu.shape[0]
    out = []
    for l in range(L):
        if l > 0:
            mu = teacher.a * mu + teacher.b * sw[:, l]
        s = sig[l]
        G = torch.zeros(R + 1, sgrid.s.shape[0], dtype=DT, device=sgrid.s.device)
        blk = M // R
        for r in range(R):
            sl = slice(r * blk, (r + 1) * blk)
            G[r] = (torch.exp(-0.5 * ((sgrid.s[None, :] - mu[sl, None]) / s) ** 2) / (s * math.sqrt(2 * math.pi))).mean(0)
        G[R] = G[:R].mean(0)
        out.append(G)
    return out


def propagate_chain(models, teacher, beliefs, Ws, sgrid, lg, cfg, rng_md, device, eps_list=None):
    """Rows (layer x belief x eps) of the three propagation variants vs GOLD through the chain `models`.

    beliefs: {name: dict(Y, logq, thr, ...)} for the inputs of neuron 1 (S2 must be present for the negativity
    metrics of neuron 1); Ws (M, L) uniform side-input samples for GOLD."""
    eps_list = eps_list or cfg["eps"]
    models = [mod.detached() for mod in models]
    floor, R = cfg["D_floor"], cfg["numerics"]["R"]
    L, m = len(models), models[0].m
    dens1 = {}
    rows1 = evaluate_case(models[0], C.Gaussian(teacher.R1, device), beliefs, sgrid, cfg, rng_md, [m], device, dens1)
    neuron = [dict(l=1, minD=rows1[0]["minD"], P_Dneg_u=rows1[0]["P_Dneg_u"])]
    kern, sq = [None], [None]
    xs = torch.linspace(0, 1, 2001, dtype=DT, device=device)
    Uu = torch.as_tensor(np.random.default_rng(0).random((2**18, 2)), device=device)
    for l in range(1, L):
        kern.append(kernels(models[l], lg, sgrid, eps_list, floor))
        sq.append(min_on_square(models[l], device=device))
        md = min_D(models[l], rng_md, cfg["numerics"]["mind"], device=device)
        neuron.append(dict(l=l + 1, minD=min(md[0], float(sq[l][0].min())), P_Dneg_u=float((models[l].D(Uu) < 0).double().mean())))
    rows = []
    for name in beliefs:
        d1 = dens1[(name, m)]["all"]
        gold = gold_layers(teacher, beliefs[name]["Y"], Ws, sgrid, R)
        cd, eu = d1["cd_c"], d1["eu_c"]
        eu_ok = bool(dens1[(name, m)]["defined"])
        calg, cals = d1["CALg"], d1["CAL"]  # (R+1, neps, Ng), (R+1, neps, G)
        for l in range(L):
            if l > 0:
                cd = cd_layer(models[l], cd)
                if eu_ok:  # the incoming EXACT_UNCAL density must be a valid density, and D_l >= floor on its support
                    rho_x = poly_on(eu, xs)
                    if bool((rho_x < 0).any()):
                        eu_ok = False
                    else:
                        Dsq, g = sq[l]
                        rho_g = poly_on(eu, g)
                        supp = rho_g >= cfg["supp_rel"] * rho_g.max(-1, keepdim=True).values
                        rho_n = poly_on(eu, lg.u)
                        supp_n = rho_n >= cfg["supp_rel"] * rho_n.max(-1, keepdim=True).values
                        okD = kern[l]["okmask"]
                        bad = ((Dsq[None] < floor) & supp[:, :, None]).any() or ((~okD)[None] & supp_n[:, :, None]).any()
                        eu_ok = not bool(bad)
                        eu = (rho_n * lg.wt) @ kern[l]["K_eu"].T if eu_ok else eu
                rho_cal = calg  # x-density of the calibrated chain at the GL nodes, (R+1, neps, Ng)
                calg = torch.stack([(rho_cal[:, e] * lg.wt) @ kern[l]["K_cal_g"][e].T for e in range(len(eps_list))], 1)
                cals = torch.stack([(rho_cal[:, e] * lg.wt) @ kern[l]["K_cal_s"][e].T for e in range(len(eps_list))], 1)
            CD, GOLD = sgrid.poly_s(cd), gold[l]
            EU = sgrid.poly_s(eu) if eu_ok else None
            cdx = poly_on(cd[-1], xs)
            base = dict(
                layer=l + 1,
                belief=name,
                defined=int(eu_ok),
                neuron_minD=neuron[l]["minD"],
                neuron_P_Dneg_u=neuron[l]["P_Dneg_u"],
                min_cd=float(cdx.min()),
                neg_mass_cd=float((torch.clamp(-CD[-1], min=0) * sgrid.w).sum()),
                min_eu=float(poly_on(eu[-1], xs).min()) if eu_ok else NAN,
                neg_mass_eu=float((torch.clamp(-EU[-1], min=0) * sgrid.w).sum()) if eu_ok else NAN,
            )
            for key, P, Q in (("d_total", CD, GOLD), ("d_denom", CD, EU), ("d_model_uncal", EU, GOLD)):
                if Q is None or P is None:
                    base.update({key: NAN, key + "_se": NAN, key + "_floor": NAN})
                else:
                    v, se, fl = tv_stats(sgrid, P, Q)
                    base.update({key: v, key + "_se": se, key + "_floor": fl})
            I, mu_, var = sgrid.moments(torch.stack([CD[-1], GOLD[-1]]))
            base.update(mean_cd=float(mu_[0]), var_cd=float(var[0]), mean_gold=float(mu_[1]), var_gold=float(var[1]))
            for e, eps in enumerate(eps_list):
                CAL = cals[:, e]
                r = dict(base, eps=eps)
                for key, P, Q in (("d_neg", EU, CAL), ("d_model_cal", CAL, GOLD), ("d_dc", CD, CAL)):
                    if P is None:
                        r.update({key: NAN, key + "_se": NAN})
                    else:
                        v, se, _ = tv_stats(sgrid, P, Q)
                        r.update({key: v, key + "_se": se})
                rows.append(r)
    return rows, rows1
