"""Core library: basis, copula sampler, coefficient estimator, propagation engine, metrics.
Implemented from the task specification; float64 throughout."""
import os
import time
import numpy as np
import yaml
from scipy.special import eval_legendre, ndtr, ndtri
from scipy.stats import beta as beta_dist
from numpy.polynomial.legendre import leggauss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cfg(path=None):
    with open(path or ROOT + "/config.yaml") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ basis / quadrature
def basis(u, m):
    """Orthonormal shifted Legendre f_n(u)=sqrt(2n+1) P_n(2u-1), n=0..m. Returns (len(u), m+1)."""
    u = np.asarray(u, float)
    n = np.arange(m + 1)
    return np.sqrt(2 * n + 1)[None, :] * eval_legendre(n[None, :], (2 * u - 1)[:, None])


def gl(n):
    x, w = leggauss(n)
    return (x + 1) / 2, w / 2


class Grid:
    def __init__(self, n, mmax=6):
        self.n = n
        self.u, self.w = gl(n)
        self.F = basis(self.u, mmax)
        self.sx = ndtri(self.u)
        self.W2 = np.outer(self.w, self.w)


# ------------------------------------------------------------------ copula
def corr_matrix(typ, rhoS, cfg):
    ryz = 2 * np.sin(np.pi * rhoS / 6)
    if typ == "SUM":
        rxy = rxz = cfg["sum_r"]
    elif typ == "CONTRAST":
        c = min(cfg["contrast_c_max"], cfg["contrast_c_scale"] * np.sqrt((1 - ryz) / 2))
        rxy, rxz = c, -c
    else:
        raise ValueError(typ)
    R = np.array([[1, rxy, rxz], [rxy, 1, ryz], [rxz, ryz, 1.0]])
    assert np.linalg.eigvalsh(R).min() > 1e-6, "R not positive definite"
    return R


def estimate_coeffs(cfg, types, rhos, n, seed, chunk=None, mmax=6):
    """a_ijk = sample mean of f_i(x)f_j(y)f_k(z); common random numbers over all (type,rho). Returns {key:(a,se)}."""
    chunk = chunk or cfg["chunk"]
    rng = np.random.default_rng(seed)
    keys = [(t, r) for t in types for r in rhos]
    Ls = {k: np.linalg.cholesky(corr_matrix(k[0], k[1], cfg)) for k in keys}
    K = mmax + 1
    S1 = {k: np.zeros((K, K * K)) for k in keys}
    S2 = {k: np.zeros((K, K * K)) for k in keys}
    done = 0
    while done < n:
        c = min(chunk, n - done)
        Z = rng.standard_normal((c, 3))
        for k in keys:
            U = np.clip(ndtr(Z @ Ls[k].T), 1e-12, 1 - 1e-12)
            Fx, Fy, Fz = (basis(U[:, i], mmax) for i in range(3))
            S1[k] += Fx.T @ (Fy[:, :, None] * Fz[:, None, :]).reshape(c, -1)
            S2[k] += (Fx ** 2).T @ ((Fy ** 2)[:, :, None] * (Fz ** 2)[:, None, :]).reshape(c, -1)
        done += c
    out = {}
    for k in keys:
        a = (S1[k] / n).reshape(K, K, K)
        var = (S2[k] / n).reshape(K, K, K) - a ** 2
        out[k] = (a, np.sqrt(np.maximum(var, 0) / n))
    return out


def ctrue(R, sx, sy, sz):
    """Analytic conditional density of x given (y,z) for the Gaussian copula, normal-score arguments."""
    sig = R[0, 1:]
    beta = np.linalg.solve(R[1:, 1:], sig)
    s2 = 1 - sig @ beta
    mu = beta[0] * sy + beta[1] * sz
    return np.exp(-0.5 * (sx - mu) ** 2 / s2 + 0.5 * sx ** 2) / np.sqrt(s2)


def ctrue_grid(G, R):
    return ctrue(R, G.sx[:, None, None], G.sx[None, :, None], G.sx[None, None, :])


# ------------------------------------------------------------------ incoming beliefs
def q_on(spec, y):
    """q(y,z) on the outer grid y x y for a spec: ('flat',), ('beta',(a1,b1,a2,b2)), ('poly', a0 coefficient matrix)."""
    y = np.asarray(y, float)
    if spec[0] == "flat":
        return np.ones((len(y), len(y)))
    if spec[0] == "beta":
        a1, b1, a2, b2 = spec[1]
        return np.outer(beta_dist.pdf(y, a1, b1), beta_dist.pdf(y, a2, b2))
    if spec[0] == "poly":
        a0 = spec[1]
        F = basis(y, a0.shape[0] - 1)
        return F @ a0 @ F.T
    raise ValueError(spec)


def scenario_specs(cfg):
    sp = {"S1": ("own",), "S2": ("flat",)}
    for k, v in cfg["beta_params"].items():
        sp[k] = ("beta", tuple(v))
    return {k: sp[k] for k in cfg["scenarios"]}


# ------------------------------------------------------------------ model tensors
def model_N(G, a, m):
    F = G.F[:, : m + 1]
    T = np.tensordot(F, a, axes=(1, 0))        # (x,j,k)
    T = np.tensordot(T, F, axes=(1, 1))        # (x,k,a)
    return np.tensordot(T, F, axes=(1, 1))     # (x,a,b)


def model_D(F, a0):
    return F @ a0 @ F.T


def gold_for(G, R, Ct, specs, cfg):
    """GOLD(x)=int c_true(x|y,z) q dy dz. Plain GL; per-column normal-score fallback if |int GOLD -1| > tol.
    Returns {name: (vector, norm_err, method)}."""
    names = list(specs)
    if not names:
        return {}
    Qw = np.stack([(q_on(specs[n], G.u) * G.W2).ravel() for n in names], 1)
    Gv = Ct.reshape(G.n, -1) @ Qw
    out, fb = {}, []
    for i, nm in enumerate(names):
        err = abs(G.w @ Gv[:, i] - 1)
        out[nm] = (Gv[:, i].copy(), err, "GL")
        if err > cfg["gold_tol"]:
            fb.append(i)
    if fb:
        ns, smax = cfg["gold_fallback"]["nodes"], cfg["gold_fallback"]["smax"]
        s = np.linspace(-smax, smax, ns)
        tw = np.full(ns, s[1] - s[0]); tw[[0, -1]] *= 0.5
        y = ndtr(s)
        phi = np.exp(-0.5 * s ** 2) / np.sqrt(2 * np.pi)
        W = np.outer(tw * phi, tw * phi)
        Qs = np.stack([(q_on(specs[names[i]], y) * W).ravel() for i in fb], 1)
        sig = R[0, 1:]; beta = np.linalg.solve(R[1:, 1:], sig); s2 = 1 - sig @ beta
        syy, szz = np.meshgrid(s, s, indexing="ij")
        mu = (beta[0] * syy + beta[1] * szz).ravel()
        res = np.empty((G.n, len(fb)))
        for ix, sx in enumerate(G.sx):
            c = np.exp(-0.5 * (sx - mu) ** 2 / s2 + 0.5 * sx ** 2) / np.sqrt(s2)
            res[ix] = c @ Qs
        for j, i in enumerate(fb):
            out[names[i]] = (res[:, j].copy(), abs(G.w @ res[:, j] - 1), "normal-score")
    return out


# ------------------------------------------------------------------ metrics
def tv(G, p, r):
    return 0.5 * float(G.w @ np.abs(p - r))


def moments(G, p):
    I = float(G.w @ p)
    m1 = float(G.w @ (G.u * p))
    v = float(G.w @ ((G.u - m1) ** 2 * p))
    return I, m1, v


NAN = float("nan")


def evaluate(G, a, m, R, scen_specs, cfg, eps_list, gold_fn, want_dens=False):
    """All four propagated densities and metrics for one neuron (coeff tensor a of degree m) and all scenarios.
    scen_specs: {name: spec or ('own',)}; gold_fn(specs)->gold_for-like dict. Returns (rows, dens)."""
    t0 = time.time()
    F = G.F[:, : m + 1]
    D = model_D(F, a[0])
    ff = cfg["fine_grid"]
    Ff = basis(np.linspace(0, 1, ff), m)
    Dfine = model_D(Ff, a[0])
    minD, fracDneg = float(Dfine.min()), float((Dfine < 0).mean())
    N = model_N(G, a, m)
    fracJoint = float(np.einsum("x,ab,xab->", G.w, G.W2, (N < 0).astype(float)))
    floor, srel = cfg["D_floor"], cfg["supp_rel"]

    specs, qvalid = {}, {}
    for nm, sp in scen_specs.items():
        if sp[0] == "own":
            qvalid[nm] = minD >= 0
            if qvalid[nm]:
                specs[nm] = ("poly", a[0])
        else:
            qvalid[nm] = True
            specs[nm] = sp
    names = list(specs)
    qs = {nm: q_on(specs[nm], G.u) for nm in names}
    gold = gold_fn(specs)
    Qw = np.stack([(qs[nm] * G.W2).ravel() for nm in names], 1)
    okD = (D >= floor)
    defined = {nm: bool(D[qs[nm] >= srel * qs[nm].max()].min() >= floor) for nm in names}
    # EXACT_UNCAL (expectation of the ratio), integrated over cells with D>=floor; never regularised
    Cu = N / np.where(okD, D, 1.0)
    Cu *= okD
    UNC = Cu.reshape(G.n, -1) @ Qw
    del Cu
    # DUDA: constant-denominator formula, coefficient form
    duda, eqD = {}, {}
    for nm in names:
        Bq = G.F.T @ (qs[nm] * G.W2) @ G.F
        num = np.einsum("ijk,jk->i", a, Bq[: m + 1, : m + 1])
        duda[nm] = (F @ num) / num[0]
        eqD[nm] = float(num[0])
    # EXACT_CAL
    cal = {}
    for eps in eps_list:
        Nc = np.maximum(N, eps)
        Z = G.w @ Nc.reshape(G.n, -1)
        Nc /= Z.reshape(1, G.n, G.n)
        cal[eps] = Nc.reshape(G.n, -1) @ Qw
        del Nc
    del N
    rows, dens = [], {}
    rt = time.time() - t0
    for nm in scen_specs:
        base = dict(scenario=nm, m=m, min_D=minD, frac_D_neg=fracDneg, frac_joint_neg=fracJoint, q_valid=int(qvalid[nm]))
        if not qvalid[nm]:
            for eps in eps_list:
                rows.append({**base, "eps": eps, "defined": 0, "quadrature_limited": NAN})
            continue
        j = names.index(nm)
        du, un = duda[nm], UNC[:, j]
        gv, gerr, gmeth = gold[nm]
        d_ = defined[nm]
        un_n = un if d_ else np.full_like(un, NAN)
        mI_d, mm_d, mv_d = moments(G, du)
        mI_u, mm_u, mv_u = moments(G, un_n)
        mI_g, mm_g, mv_g = moments(G, gv)
        b = dict(base, defined=int(d_), quadrature_limited=int(gerr > cfg["gold_tol"]), gold_norm_err=float(gerr),
                 gold_method=gmeth, int_duda=mI_d, int_uncal=mI_u if d_ else NAN, int_gold=mI_g,
                 mean_duda=mm_d, var_duda=mv_d, mean_uncal=mm_u if d_ else NAN, var_uncal=mv_u if d_ else NAN,
                 mean_gold=mm_g, var_gold=mv_g, E_q_D=eqD[nm],
                 neg_mass_duda=float(-(G.w @ np.minimum(du, 0))),
                 d_denom=tv(G, du, un) if d_ else NAN,
                 d_total=tv(G, du, gv),
                 dmean_vs_uncal=abs(mm_d - mm_u) if d_ else NAN,
                 relvar_vs_uncal=(mv_d - mv_u) / mv_u if d_ else NAN,
                 dmean_vs_gold=abs(mm_d - mm_g), relvar_vs_gold=(mv_d - mv_g) / mv_g,
                 d_model_uncal=tv(G, un, gv) if d_ else NAN)
        for eps in eps_list:
            ca = cal[eps][:, j]
            rows.append(dict(b, eps=eps, d_neg=tv(G, un, ca) if d_ else NAN, d_model_cal=tv(G, ca, gv),
                             d_dc=tv(G, du, ca), int_cal=moments(G, ca)[0], runtime_s=rt))
            if want_dens:
                dens[(nm, eps)] = dict(x=G.u, gold=gv, uncal=un_n, cal=ca, duda=du)
    return rows, dens
