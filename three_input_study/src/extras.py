"""Optional stages: 8 Taylor-corrected denominator, 9 value-loss training (exploratory)."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from hcr import *
from scipy.optimize import minimize

SQ12 = np.sqrt(12.0)


# ----------------------------------------------------------------------------- stage 8
def stage8(cfg, co, log=print):
    G = Grid(cfg["nodes"], cfg["mmax"])
    specs = scenario_specs(cfg)
    rows = []
    for typ in cfg["types"]:
        for rho in cfg["rho_S"]:
            a6 = co[(typ, rho)][0]
            for m in cfg["taylor_m"]:
                a = a6[: m + 1, : m + 1, : m + 1]
                F = G.F[:, : m + 1]
                D = model_D(F, a[0])
                minD = float(model_D(basis(np.linspace(0, 1, cfg["fine_grid"]), m), a[0]).min())
                N = model_N(G, a, m)
                for nm, sp in specs.items():
                    if sp[0] == "own":
                        if minD < 0:
                            continue
                        sp = ("poly", a[0])
                    q = q_on(sp, G.u)
                    supp = q >= cfg["supp_rel"] * q.max()
                    okD = D >= cfg["D_floor"]
                    defined = bool(D[supp].min() >= cfg["D_floor"])
                    Wq = q * G.W2
                    Cu = np.where(okD, N / np.where(okD, D, 1.0), 0.0)
                    unc = Cu.reshape(G.n, -1) @ Wq.ravel() if defined else None
                    for K in [0] + list(cfg["taylor_K"]):
                        S = sum((1 - D) ** n for n in range(K + 1))
                        P = N.reshape(G.n, -1) @ (S * Wq).ravel() / float((D * S * Wq).sum())
                        rows.append(
                            dict(
                                type=typ,
                                rho_S=rho,
                                m=m,
                                scenario=nm,
                                K=K,
                                defined=int(defined),
                                series_ok=int(np.abs(1 - D[supp]).max() < 1),
                                max_abs_1mD=float(np.abs(1 - D[supp]).max()),
                                d_taylor=tv(G, P, unc) if defined else NAN,
                            )
                        )
            log(f"  taylor {typ} rho={rho}")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT + "/results/taylor.csv", index=False)
    return df


# ----------------------------------------------------------------------------- stage 9
def _feat(U, m):
    Fy, Fz = basis(U[:, 1], m), basis(U[:, 2], m)
    return (Fy[:, :, None] * Fz[:, None, :]).reshape(len(U), -1)


def _unpack(th, K2):
    return th[:K2], np.concatenate([[1.0], th[K2:]])


def _make_obj(P, x, counter):
    K2 = P.shape[1]
    n = len(x)

    def f(th):
        a1, a0 = _unpack(th, K2)
        A, B = P @ a1, P @ a0
        if (not np.all(np.isfinite(A))) or np.any(np.abs(B) < 1e-12):
            counter["bad"] += 1  # explicit record, no patching of B; huge loss steers the line search away
            return 1e300, np.zeros_like(th)
        yh = 0.5 + A / (SQ12 * B)
        r = yh - x
        g = 2 * r / n
        g1 = P.T @ (g / (SQ12 * B))
        g0 = P.T @ (-g * A / (SQ12 * B**2))
        return float(np.mean(r**2)), np.concatenate([g1, g0[1:]])

    return f


def stage9(cfg, co, log=print):
    tc = cfg["train"]
    m = tc["m"]
    G = Grid(cfg["nodes"], cfg["mmax"])
    specs = {k: v for k, v in scenario_specs(cfg).items() if k in ("S2", "S3b", "S3c")}
    rows, curves = [], []
    for typ in cfg["types"]:
        for rho in tc["rho_S"]:
            t0 = time.time()
            a6 = co[(typ, rho)][0]
            R = corr_matrix(typ, rho, cfg)
            L = np.linalg.cholesky(R)
            rng = np.random.default_rng(cfg["seed"] + 7919 + int(round(rho * 10)) + (0 if typ == "SUM" else 100))

            def draw(n):
                return np.clip(ndtr(rng.standard_normal((n, 3)) @ L.T), 1e-12, 1 - 1e-12)

            Utr, Ute = draw(tc["n_train"]), draw(tc["n_test"])
            Ptr, Pte = _feat(Utr, m), _feat(Ute, m)
            xtr, xte = Utr[:, 0], Ute[:, 0]
            a0e, a1e = a6[0, : m + 1, : m + 1], a6[1, : m + 1, : m + 1]
            K2 = (m + 1) ** 2
            th0 = np.concatenate([a1e.ravel(), a0e.ravel()[1:]])
            counter = {"bad": 0}
            obj = _make_obj(Ptr, xtr, counter)
            f0 = obj(th0)[0]
            res = minimize(obj, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=tc["maxiter"], maxfun=3 * tc["maxiter"]))
            th1 = res.x

            def mse(th, P, x):
                a1, a0 = _unpack(th, K2)
                B = P @ a0
                with np.errstate(all="ignore"):
                    yh = 0.5 + (P @ a1) / (SQ12 * B)
                return float(np.mean((yh - x) ** 2))

            def frac_Bneg(th):
                return float(np.mean((Ptr @ _unpack(th, K2)[1]) <= 0))

            # propagation evaluation before / after (other a_ijk, i>=2, stay at direct estimates)
            Ct = ctrue_grid(G, R)
            gf = lambda s: gold_for(G, R, Ct, s, cfg)
            out = {}
            for tag, th in (("before", th0), ("after", th1)):
                a = a6[: m + 1, : m + 1, : m + 1].copy()
                a1, a0 = _unpack(th, K2)
                a[1] = a1.reshape(m + 1, m + 1)
                a[0] = a0.reshape(m + 1, m + 1)
                r, _ = evaluate(G, a, m, R, specs, cfg, [0.01], gf)
                out[tag] = pd.DataFrame(r).set_index("scenario")
            base = dict(
                type=typ,
                rho_S=rho,
                m=m,
                n_train=tc["n_train"],
                opt_status=int(res.status),
                opt_message=str(res.message),
                nit=int(res.nit),
                nfev=int(res.nfev),
                bad_evals=counter["bad"],
                mse_train_before=f0,
                mse_train_after=float(res.fun),
                mse_test_before=mse(th0, Pte, xte),
                mse_test_after=mse(th1, Pte, xte),
                fracB_le0_train_before=frac_Bneg(th0),
                fracB_le0_train_after=frac_Bneg(th1),
                dist_a0_before=0.0,
                dist_a0_after=float(np.linalg.norm(_unpack(th1, K2)[1] - _unpack(th0, K2)[1])),
                dist_a1_after=float(np.linalg.norm(th1[:K2] - th0[:K2])),
                min_D_before=float(out["before"].min_D.iloc[0]),
                min_D_after=float(out["after"].min_D.iloc[0]),
                fracDneg_before=float(out["before"].frac_D_neg.iloc[0]),
                fracDneg_after=float(out["after"].frac_D_neg.iloc[0]),
            )
            for nm in specs:
                r = dict(base, scenario=nm)
                for tag in ("before", "after"):
                    for c in ("d_total", "d_denom", "d_dc", "relvar_vs_gold", "relvar_vs_uncal", "defined"):
                        r[f"{c}_{tag}"] = out[tag].loc[nm, c]
                rows.append(r)
            log(f"  train {typ} rho={rho}: mse {f0:.5f}->{res.fun:.5f} status={res.status} bad={counter['bad']} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT + "/results/training.csv", index=False)
    return df


if __name__ == "__main__":
    cfg = load_cfg()
    z = np.load(ROOT + "/results/coeffs.npz")
    co = {(t, r): (z[f"{t}_{r}_a"], z[f"{t}_{r}_se"]) for t in cfg["types"] for r in cfg["rho_S"]}
    which = sys.argv[1:] or ["8", "9"]
    if "8" in which:
        stage8(cfg, co)
    if "9" in which:
        stage9(cfg, co)
