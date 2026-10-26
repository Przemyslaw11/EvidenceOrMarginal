"""Measure job cost on this device and write scheduler/cost_model.json.

    python scheduler/profile_cost.py --device cuda
    python scheduler/profile_cost.py --from-results     # refit the measured cells from production job metadata
Runs real jobs (`gpu.job.run_job`) with two cases each, for every (k, basis) of the sweeps, a few copula families
and one network job; records the estimation time per case at n_est = 1e7 and the evaluation time per case. The
seed index (2000) never occurs in production; the outputs are discarded. make_jobs.py reads the result.
"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gpu.job import load_cfg, run_job  # noqa: E402
from scheduler.make_jobs import CODE, with_id  # noqa: E402

OUT = os.path.join(ROOT, "scheduler", "cost_model.json")
N_EST = 10_000_000


def from_results():
    """Replace the cells of cost_model.json that production jobs have measured (median over jobs)."""
    import glob

    import numpy as np
    from gpu.job import read_meta
    cost = json.load(open(OUT))
    cost.setdefault("cells", {})
    groups, net = {}, []
    for f in glob.glob(os.path.join(ROOT, "results2", "jobs", "*", "*.parquet")):
        m = read_meta(f)
        sp = m["spec"]
        if sp.get("group") == "pilot" or sp.get("variant", "base") != "base" or sp["exp"] == "population":
            continue
        if sp["exp"] == "network":
            net.append(m["wall_s"])
            continue
        n = len(sp["cases"])
        key = f"{sp.get('family', 'gauss')}_{sp['k']}_{sp['basis']}"
        groups.setdefault(key, []).append((m["est_s"] / n * cost["n_est_ref"] / sp["n_est"],
                                           (m["wall_s"] - m["est_s"] - 30) / n))
    for key, v in groups.items():
        v = np.array(v)
        cost["cells"][key] = dict(est_s_per_case=float(np.median(v[:, 0])), case_s=float(np.median(v[:, 1])),
                                  n_jobs=len(v), source="production")
    if net:
        cost["network"] = {"measured": float(np.median(net)), "n_jobs": len(net)}
    with open(OUT, "w") as f:
        json.dump(cost, f, indent=1)
    print({k: (round(v["est_s_per_case"], 1), round(v["case_s"], 1)) for k, v in cost["cells"].items()})


def main():
    if "--from-results" in sys.argv:
        return from_results()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", default="2,4,8,16,32")
    ap.add_argument("--n-est", type=int, default=N_EST)
    ap.add_argument("--network", default="8,triple,0.4", help="k,basis,rho_S of the timed network job ('' to skip)")
    a = ap.parse_args()
    cfg = load_cfg()
    cost = dict(device=torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu", n_est_ref=a.n_est,
                cells={}, network={})

    def save():
        with open(OUT, "w") as f:
            json.dump(cost, f, indent=1)

    def timed(spec):
        t = time.time()
        df, meta = run_job(with_id(spec), cfg, a.device)
        n = len(spec["cases"])
        per_case = sorted(df.groupby("case")["runtime_case_s"].first())
        return dict(est_s_per_case=meta["est_s"] / n, case_s=per_case[-1], case_s_all=per_case,
                    wall_s=time.time() - t)

    run_job(with_id(dict(exp="gauss", k=2, family="gauss", type="SUM", basis="pair", seed=2000, n_est=100000,
                         variant="base", key=[CODE["gauss"], 2, 2000], cases=[{"rho_S": 0.2}])), cfg, a.device)  # warm-up
    cells = [(k, b, "gauss", "SUM") for k in map(int, a.k.split(",")) for b in ("pair", "triple")]
    cells += [(8, "triple", "t3", "EXCH"), (8, "triple", "gumbel", "EXCH"), (8, "triple", "clayton", "EXCH")]
    for k, basis, fam, typ in cells:
        exp = "gauss" if fam == "gauss" else "families"
        spec = dict(exp=exp, k=k, family=fam, type=typ, basis=basis, seed=2000, n_est=a.n_est, variant="base",
                    key=[CODE[exp], k, 2000], cases=[{"rho_S": 0.2}, {"rho_S": 0.5}])
        cost["cells"][f"{fam}_{k}_{basis}"] = r = timed(spec)
        print(f"{fam} k={k} {basis}: {r}", flush=True)
        save()
    if a.network:
        k, basis, rho = a.network.split(",")
        spec = dict(exp="network", k=int(k), family="gauss", type="SUM", basis=basis, seed=2000, rho_S=float(rho),
                    variant="base", key=[CODE["network"], int(k), 2000])
        t = time.time()
        run_job(with_id(spec), cfg, a.device)
        cost["network"][f"{k}_{basis}"] = time.time() - t
        print(f"network k={k} {basis}: {time.time() - t:.0f}s", flush=True)
        save()


if __name__ == "__main__":
    main()
