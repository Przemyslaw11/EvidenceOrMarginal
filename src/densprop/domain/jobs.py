"""Job specifications, their ids and their random streams.

Every job is fully specified; its id is a hash of the spec. Random streams are derived from
SeedSequence(root_seed, spawn_key=key + (stream,)), where key omits the neuron type and the basis, so
jobs that differ only in those share their random numbers (common random numbers). The gauss,
population and convergence experiments share keys, so their results are paired by seed.
This module needs no PyTorch, so that writing job lists does not either.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

Spec = dict[str, Any]  # a job specification; its JSON form defines the job id, so keys, values and types are fixed
STREAM = {"est": 0, "mc": 1, "mind": 2, "lkj": 3, "train": 4, "val": 5, "test": 6}
CODE = {"gauss": 1, "families": 2, "lkj": 3, "network": 4}


def rng_for(cfg: dict, key: list, stream: str, *extra: int) -> np.random.Generator:
    """The generator of one random stream of a job: common random numbers across neuron types and bases."""
    ss = np.random.SeedSequence(cfg["root_seed"], spawn_key=tuple(key) + (STREAM[stream],) + tuple(extra))
    return np.random.Generator(np.random.PCG64(ss))


def expected_minutes(spec: Spec, cost: dict) -> float:
    """GPU-minutes from the measured cost table (scheduler/cost_model.json, written by `densprop profile-cost`)."""
    cells = cost.get("cells")
    if not cells:
        return 15.0  # no measurement yet
    if spec["exp"] == "network":
        return 1.2 * next(iter(cost["network"].values())) / 60 if cost.get("network") else 15.0
    k, basis, n = spec["k"], spec["basis"], len(spec["cases"])
    fam = spec.get("family", "gauss")
    c = cells.get(f"{fam}_{k}_{basis}") or cells[f"gauss_{k}_{basis}"]
    if f"{fam}_{k}_{basis}" not in cells:  # family cost relative to the Gaussian, measured at k = 8 (triple)
        r = cells.get(f"{fam}_8_triple") or cells.get("t3_8_triple")
        g = cells["gauss_8_triple"]
        c = dict(est_s_per_case=c["est_s_per_case"] * r["est_s_per_case"] / g["est_s_per_case"], case_s=c["case_s"] * r["case_s"] / g["case_s"])
    v = spec.get("variant", "base")
    est = 0.0 if spec["exp"] == "population" else c["est_s_per_case"] * spec["n_est"] / cost["n_est_ref"]
    case = c["case_s"]
    if v == "n2":
        est *= 2
    elif v == "M2":
        case *= 2
    return 0.5 + n * (est + case) / 60  # 0.5 min: process start, belief sampling


def with_id(spec: Spec) -> Spec:
    """The spec with its id: the first 16 hex digits of the SHA-1 of its sorted-key JSON."""
    spec = dict(spec)
    spec["id"] = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    return spec


def types_for(k: int, types: list[str]) -> list[str]:
    return [t for t in types if not (t == "CONTRAST" and k % 2)]


def jobs(exp: str, cfg: dict) -> list[Spec]:
    e = cfg["experiments"].get(exp, {})
    rho_cases = [{"rho_S": r} for r in cfg["rho_S"]]
    out = []
    if exp in ("gauss", "population", "convergence"):
        variants = [v for v in e.get("variants", ["base"]) if not (exp == "convergence" and v == "base")]
        for k in e["k"]:
            for typ in types_for(k, e["types"]):
                for basis in e["basis"]:
                    for seed in range(e["seeds"]):
                        for v in variants:
                            out.append(
                                dict(
                                    exp=exp,
                                    k=k,
                                    family="gauss",
                                    type=typ,
                                    basis=basis,
                                    seed=seed,
                                    n_est=0 if exp == "population" else e["n_est"],
                                    variant=v,
                                    key=[CODE["gauss"], k, seed],
                                    cases=rho_cases,
                                )
                            )
    elif exp == "families":
        for k in e["k"]:
            for fam, typ in e["combos"]:
                if typ == "CONTRAST" and k % 2:
                    continue
                for basis in e["basis"]:
                    for seed in range(e["seeds"]):
                        out.append(
                            dict(
                                exp=exp,
                                k=k,
                                family=fam,
                                type=typ,
                                basis=basis,
                                seed=seed,
                                n_est=e["n_est"],
                                variant="base",
                                key=[CODE["families"], k, seed],
                                cases=rho_cases,
                            )
                        )
    elif exp == "lkj":
        nb = math.ceil(e["matrices"] / e["per_job"])
        for k in e["k"]:
            for eta in e["eta"]:
                for basis in e["basis"]:
                    for b in range(nb):
                        idx = range(b * e["per_job"], min((b + 1) * e["per_job"], e["matrices"]))
                        out.append(
                            dict(
                                exp=exp,
                                k=k,
                                family="gauss",
                                type="LKJ",
                                basis=basis,
                                seed=b,
                                eta=eta,
                                n_est=e["n_est"],
                                variant="base",
                                key=[CODE["lkj"], k, eta, b],
                                key_lkj=[CODE["lkj"], k, eta],
                                cases=[{"lkj_idx": i} for i in idx],
                            )
                        )
    elif exp == "network":
        nc = cfg["network"]
        for k in nc["k"]:
            for rho in nc["rho_S"]:
                for basis in nc["basis"]:
                    for seed in range(nc["seeds"]):
                        out.append(
                            dict(
                                exp=exp,
                                k=k,
                                family="gauss",
                                type="SUM",
                                basis=basis,
                                seed=seed,
                                rho_S=rho,
                                variant="base",
                                key=[CODE["network"], k, seed],
                            )
                        )
    else:
        raise ValueError(exp)
    return [with_id(s) for s in out]


def pilot_jobs(cfg: dict) -> list[Spec]:
    """Throughput pilot: seed index 1000 never occurs in production, results go to results/jobs/pilot."""
    out = []
    for exp in ("gauss", "population", "families", "lkj"):
        for s in jobs(exp, cfg):
            pick = (
                (exp in ("gauss", "population") and s["seed"] == 0 and s["type"] == "SUM" and s["k"] in (2, 8, 32))
                or (
                    exp == "families"
                    and s["seed"] == 0
                    and s["k"] == 8
                    and s["basis"] == "triple"
                    and s["family"] in ("t3", "clayton", "gumbel")
                    and s["type"] in ("SUM", "EXCH")
                )
                or (exp == "lkj" and s["seed"] == 0 and s["k"] == 8 and s["eta"] == 1)
            )
            if pick:
                p = {kk: v for kk, v in s.items() if kk != "id"}
                p["seed"] = 1000
                p["key"] = p["key"][:-1] + [1000]
                if exp == "lkj":
                    p["key_lkj"] = p["key_lkj"] + [1000]
                p["group"] = "pilot"
                out.append(with_id(p))
    return out


def job_list(exp: str, cfg: dict, cost: dict) -> list[Spec]:
    """The job list of an experiment ('pilot' for the throughput pilot) with expected GPU-minutes, in run order."""
    js = pilot_jobs(cfg) if exp == "pilot" else jobs(exp, cfg)
    for s in js:
        s["expected_min"] = round(expected_minutes(s, cost), 3)
    js.sort(key=lambda s: (s["seed"], -s["expected_min"]))  # seed by seed (PREREGISTRATION 4), longest first
    return js


def small_jobs(exp: str, cfg: dict) -> list[Spec]:
    """The reduced job list of the small reproduction: k in {2, 8}, pair basis, 3 seeds, n_est = 1e6, M = 2^18 (network: k = 2, one seed)."""
    net = exp == "network"
    out = []
    for s in jobs(exp, cfg):
        if s["k"] not in ((2,) if net else (2, 8)) or s["basis"] != "pair" or s["seed"] >= (1 if net else 3):
            continue
        if net and s["rho_S"] != 0.2:
            continue
        small = {kk: v for kk, v in s.items() if kk != "id"}
        if exp == "gauss":
            small["n_est"] = 10**6
        small["override"] = {"network": {"seeds": 1, "adam": {"steps": 500}, "lbfgs": {"outer": 10}}} if net else {"numerics": {"M": 2**18}}
        out.append(with_id(small))
    return out
