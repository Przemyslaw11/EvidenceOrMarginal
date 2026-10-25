"""Measure the per-component cost of a job on this device and write scheduler/cost_model.json.

    python scheduler/profile_cost.py --device cuda
Times (s) per unit of work for: estimation (per sample per feature), propagation (per sample per feature) and the
min-D optimiser (per start-step per feature), fitted over (k, basis) in the sweep; reported as GPU-minutes per unit.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gpu import copulas as C  # noqa: E402
from gpu.core import Estimator, chunk_for, input_basis, population_model  # noqa: E402
from gpu.job import load_cfg, setup_torch  # noqa: E402
from gpu.propagate import min_D  # noqa: E402
from scheduler.make_jobs import feat  # noqa: E402


def sync(dev):
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    setup_torch(dev)
    cfg = load_cfg()
    rows = []
    for k, basis in ((2, "pair"), (8, "pair"), (8, "triple"), (16, "triple"), (32, "pair"), (32, "triple")):
        R, _ = C.coupled_R("SUM", k, 0.3)
        cop = C.Gaussian(R, dev)
        rng = np.random.default_rng(0)
        chunk = chunk_for(k, 6, basis, cfg["numerics"]["chunk_est"])
        est = Estimator(k, 6, basis, dev)
        n = 4 * chunk
        base = [C.base_variates(rng, chunk, k + 1, ("Z",)) for _ in range(4)]
        sync(dev); t = time.time()
        for b in base:
            est.add(cop.uniforms(b))
        sync(dev); t_est = (time.time() - t) / n
        model = est.model()[0]
        Y = torch.rand(2 ** 17, k, dtype=torch.float64, device=dev)
        times = {}
        for m in (2, 4, 6):
            mm = model.sliced(m)
            sync(dev); t = time.time()
            for st in range(0, Y.shape[0], 16384):
                mm.A(input_basis(Y[st: st + 16384], 6))
            sync(dev); times[m] = (time.time() - t) / Y.shape[0]
        md = dict(cfg["numerics"]["mind"], starts=500, steps=50)
        sync(dev); t = time.time()
        min_D(model, np.random.default_rng(0), md, device=dev)
        sync(dev); t_md = (time.time() - t) / (500 * 50)
        f = feat(k, basis)
        rows.append(dict(k=k, basis=basis, feat=f, est_s_per_sample=t_est, A_s_per_sample=times,
                         mind_s_per_startstep_m6=t_md))
        print(rows[-1], flush=True)
    t = time.time()
    population_model(C.coupled_R("SUM", 8, 0.3)[0], 6, "triple", cfg["numerics"]["gh_nodes"], dev)
    print("population k=8 triple:", time.time() - t, "s", flush=True)
    fe = np.array([r["feat"] for r in rows], float)
    to_min = lambda v: float(np.median(np.array(v) / fe)) / 60
    cost = dict(est=to_min([r["est_s_per_sample"] for r in rows]) / 2,
                prop=to_min([np.mean(list(r["A_s_per_sample"].values())) for r in rows]),
                mind=to_min([r["mind_s_per_startstep_m6"] for r in rows]) / 3, case=0.05, measured=rows,
                device=torch.cuda.get_device_name(0) if dev.startswith("cuda") else "cpu")
    with open(os.path.join(ROOT, "scheduler", "cost_model.json"), "w") as f:
        json.dump(cost, f, indent=1)
    print(json.dumps({k: v for k, v in cost.items() if k != "measured"}))


if __name__ == "__main__":
    main()
