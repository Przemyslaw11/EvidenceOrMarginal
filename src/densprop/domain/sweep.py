"""One fully specified job of the sweep: its cases, the coefficient estimation, the Monte Carlo passes and the result rows.

The random streams come from `jobs.rng_for`, so a job's numbers depend on its spec and the configuration only.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch

from densprop.domain import copulas as C
from densprop.domain.jobs import Spec, rng_for
from densprop.domain.neuron import Estimator, chunk_for, population_model
from densprop.domain.propagate import Grid, belief_tensors, case_rows, mc_pass, sample_beliefs, solve_mind


def build_cases(spec: Spec, cfg: dict, device: str) -> list[tuple[Any, dict]]:
    """[(copula, info dict)] for the cases of a job."""
    k, out = spec["k"], []
    for case in spec["cases"]:
        if "lkj_idx" in case:
            rng = rng_for(cfg, spec["key_lkj"], "lkj", case["lkj_idx"])
            n_rej = 0
            while True:
                R = C.lkj(k + 1, spec["eta"], rng)
                if C.min_eig(R) > C.PD_TOL:
                    break
                n_rej += 1
            off = np.abs(R[1:, 1:][np.triu_indices(k, 1)]) if k > 1 else np.array([np.nan])
            info = dict(
                lkj_idx=case["lkj_idx"],
                eta=spec["eta"],
                lkj_rejected=n_rej,
                min_eig=C.min_eig(R),
                mean_abs_ryy=float(off.mean()),
                mean_abs_rxy=float(np.abs(R[0, 1:]).mean()),
                rhoS_equiv_yy=float(np.mean(6 / math.pi * np.arcsin(off / 2))),
                R_json=json.dumps(np.round(R, 12).tolist()),
            )
            out.append((C.Gaussian(R, device), info))
        else:
            cop, info = C.make_copula(spec["family"], spec["type"], k, case["rho_S"], device, cfg["couplings"])
            info = dict(info)
            info["rho_S"] = case["rho_S"]
            out.append((cop, info))
    return out


def kinds_for(spec: Spec, cases: list) -> tuple[str, ...]:
    if spec["exp"] == "families":
        return C.KINDS_ALL  # identical draws for every family: common random numbers across the combos
    ks = set()
    for cop, _ in cases:
        ks.update(cop.kinds)
    return tuple(ks)


def estimate(spec: Spec, cfg: dict, cases: list, device: str, n_est: int) -> list:
    k, m6, basis = spec["k"], cfg["m_max"], spec["basis"]
    ests = [Estimator(k, m6, basis, device) for _ in cases]
    chunk = chunk_for(k, m6, basis, cfg["numerics"]["chunk_est"])
    rng = rng_for(cfg, spec["key"], "est")
    kinds = kinds_for(spec, cases)
    done = 0
    while done < n_est:
        n = min(chunk, n_est - done)
        base = C.base_variates(rng, n, k + 1, kinds)
        for (cop, _), est in zip(cases, ests):
            est.add(cop.uniforms(base))
        done += n
    return [e.model() for e in ests]


def run_job(spec: Spec, cfg: dict, device: str = "cpu", log: Callable[[str], None] | None = None) -> tuple[pd.DataFrame, float]:
    """The result rows of one job and the seconds spent estimating coefficients. Torch must be set up (float64)."""
    cfg = _merged(json.loads(json.dumps(cfg)), spec.get("override", {}))
    if spec["exp"] == "network":
        from densprop.domain.network_job import run_network_job

        df = run_network_job(spec, cfg, device, rng_for)
        return _tag(df, spec, int(cfg["network"]["n_est"])), 0.0
    num = cfg["numerics"]
    v = spec.get("variant", "base")
    n_est = int(spec["n_est"])
    if v == "M2":
        num["M"] *= 2
    elif v == "n2":
        n_est *= 2
    elif v == "s2":
        num["s_nodes"] = 2 * num["s_nodes"] - 1
    elif v == "x2":
        num["x_gl"] *= 2
    k = spec["k"]
    cases = build_cases(spec, cfg, device)
    t_est = time.time()
    if n_est > 0:
        models = estimate(spec, cfg, cases, device, n_est)
    else:  # population coefficients: exact moments of the Gaussian copula
        models = [(population_model(cop.R, cfg["m_max"], spec["basis"], num["gh_nodes"], device), None) for cop, _ in cases]
    t_est = time.time() - t_est
    grid = Grid(num["s_nodes"], num["s_max"], num["x_gl"], cfg["m_max"], device)
    names = cfg.get("beliefs_used", cfg["order"])
    M = num["M"]
    rngs = {nm: rng_for(cfg, spec["key"], "mc", i) for i, nm in enumerate(cfg["order"])}
    raw = sample_beliefs(rngs, M, k, cfg["beliefs"], [n for n in names if n != "S4"])
    beliefs = {nm: belief_tensors(d, cfg["supp_rel"], device) for nm, d in raw.items()}
    base_S4 = C.base_variates(rngs["S4"], M, k + 1, kinds_for(spec, cases)) if "S4" in names else None
    rng_md = rng_for(cfg, spec["key"], "mind")
    # MC pass per case, then the min-D searches of all cases as one stacked problem, then the metrics
    passes, t_case = [], []
    for (cop, _), (model, _) in zip(cases, models):
        t1 = time.time()
        if base_S4 is not None:
            beliefs["S4"] = dict(Y=cop.uniforms(base_S4)[:, 1:].contiguous())
        passes.append(mc_pass(model, cop, beliefs, grid, cfg, cfg["degrees"], device))
        t_case.append(time.time() - t1)
    beliefs.pop("S4", None)
    t1 = time.time()
    solve_mind(passes, cfg, rng_md, device)
    t_mind = (time.time() - t1) / len(cases)
    if log is not None:
        log(f"MC passes {sum(t_case):.0f}s, min D {t_mind * len(cases):.0f}s (estimation {t_est:.0f}s)")
    rows = []
    for ci, ((_, info), (model, se)) in enumerate(zip(cases, models)):
        t1 = time.time()
        r = case_rows(passes[ci], grid, cfg)
        extra = dict(
            case=ci,
            **info,
            rhoS_emp_yy=_emp_yy(model),
            rhoS_emp_xy=_emp_xy(model),
            se_a011=_se_yy(se),
            runtime_case_s=t_case[ci] + t_mind + time.time() - t1,
        )
        for x in r:
            x.update(extra)
        rows += r
    return _tag(pd.DataFrame(rows), spec, n_est), t_est


def _tag(df: pd.DataFrame, spec: Spec, n_est: int) -> pd.DataFrame:
    """The job's identifying columns on every row."""
    for key in ("exp", "k", "family", "type", "basis", "seed", "variant", "eta"):
        if key in spec:
            df[key] = spec[key]
    df["n_est"] = n_est
    df["job_id"] = spec["id"]
    return df


def _merged(cfg: dict, over: dict) -> dict:
    """Recursive update of the configuration with a job's `override` (used by tests and small runs)."""
    for key, v in over.items():
        cfg[key] = _merged(cfg.get(key, {}), v) if isinstance(v, dict) else v
    return cfg


def _emp_yy(model):
    W2 = model.blocks["W2"]
    return float(W2[0, :, 0, 0].mean()) if W2 is not None else math.nan


def _emp_xy(model):
    return float(model.blocks["W1"][1, :, 0].abs().mean())


def _se_yy(se):
    return float(se["W2"][0, :, 0, 0].median()) if se is not None and se["W2"] is not None else math.nan


def setup_torch(device: str) -> None:
    """Float64 and deterministic kernels; part of the computation, so every entry point that computes calls it first."""
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
