"""Measure job cost on this device, or refit it from production job metadata; job lists use it to order and time jobs.

Runs real jobs with two cases each, for every (k, basis) of the sweeps, a few copula families and one network job; records
the estimation time per case at n_est = 1e7 and the evaluation time per case. The seed index (2000) never occurs in
production; the outputs are discarded.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from densprop.domain.jobs import CODE, with_id
from densprop.domain.sweep import run_job

N_EST = 10_000_000


def from_results(cost: dict, metas: list[dict]) -> dict:
    """The cost table with the cells that production jobs have measured replaced (median over jobs)."""
    cost.setdefault("cells", {})
    groups, net = {}, []
    for m in metas:
        sp = m["spec"]
        if sp.get("group") == "pilot" or sp.get("variant", "base") != "base" or sp["exp"] == "population":
            continue
        if sp["exp"] == "network":
            net.append(m["wall_s"])
            continue
        n = len(sp["cases"])
        key = f"{sp.get('family', 'gauss')}_{sp['k']}_{sp['basis']}"
        groups.setdefault(key, []).append((m["est_s"] / n * cost["n_est_ref"] / sp["n_est"], (m["wall_s"] - m["est_s"] - 30) / n))
    for key, times in groups.items():
        v = np.array(times)
        cost["cells"][key] = dict(est_s_per_case=float(np.median(v[:, 0])), case_s=float(np.median(v[:, 1])), n_jobs=len(v), source="production")
    if net:
        cost["network"] = {"measured": float(np.median(net)), "n_jobs": len(net)}
    return cost


def profile(
    cfg: dict,
    device: str,
    device_name: str,
    ks: list[int],
    n_est: int,
    network: str,
    save: Callable[[dict], None],
    log: Callable[[str], None] = print,
) -> dict:
    """Time the cells; `network` is 'k,basis,rho_S' of the timed network job ('' to skip). Saves after every cell."""
    cost = dict(device=device_name, n_est_ref=n_est, cells={}, network={})

    def timed(spec):
        t = time.time()
        df, est_s = run_job(with_id(spec), cfg, device)
        n = len(spec["cases"])
        per_case = sorted(df.groupby("case")["runtime_case_s"].first())
        return dict(est_s_per_case=est_s / n, case_s=per_case[-1], case_s_all=per_case, wall_s=time.time() - t)

    warm = dict(exp="gauss", k=2, family="gauss", type="SUM", basis="pair", seed=2000, n_est=100000, variant="base")
    run_job(with_id(dict(warm, key=[CODE["gauss"], 2, 2000], cases=[{"rho_S": 0.2}])), cfg, device)
    cells = [(k, b, "gauss", "SUM") for k in ks for b in ("pair", "triple")]
    cells += [(8, "triple", "t3", "EXCH"), (8, "triple", "gumbel", "EXCH"), (8, "triple", "clayton", "EXCH")]
    for k, basis, fam, typ in cells:
        exp = "gauss" if fam == "gauss" else "families"
        spec = dict(
            exp=exp,
            k=k,
            family=fam,
            type=typ,
            basis=basis,
            seed=2000,
            n_est=n_est,
            variant="base",
            key=[CODE[exp], k, 2000],
            cases=[{"rho_S": 0.2}, {"rho_S": 0.5}],
        )
        cost["cells"][f"{fam}_{k}_{basis}"] = r = timed(spec)
        log(f"{fam} k={k} {basis}: {r}")
        save(cost)
    if network:
        k, basis, rho = network.split(",")
        spec = dict(
            exp="network",
            k=int(k),
            family="gauss",
            type="SUM",
            basis=basis,
            seed=2000,
            rho_S=float(rho),
            variant="base",
            key=[CODE["network"], int(k), 2000],
        )
        t = time.time()
        run_job(with_id(spec), cfg, device)
        cost["network"][f"{k}_{basis}"] = time.time() - t
        log(f"network k={k} {basis}: {time.time() - t:.0f}s")
        save(cost)
    return cost
