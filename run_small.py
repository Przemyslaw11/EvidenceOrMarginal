"""Small end-to-end reproduction of the scaled-up study on one GPU (or CPU), about 30 minutes on an A100.

    python run_small.py [--device cuda]
Runs the fast validation tests, a reduced sweep (k in {2, 8}, pair basis, 3 seeds, n_est = 1e6, M = 2^18), the
matching population run, one small chain/training job, then the analysis and figures, into results/small/ and
figures/small/. The full study runs on a cluster: see README (scheduler/make_jobs.py, scheduler/slurm_array.sh).
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from gpu.job import load_cfg
    from scheduler.make_jobs import jobs, with_id
    res = os.path.join(ROOT, "results", "small")
    os.makedirs(res, exist_ok=True)
    env = dict(os.environ, HCR_RESULTS=res, HCR_FIGURES=os.path.join(ROOT, "figures", "small"), PYTHONPATH=ROOT)
    py = sys.executable
    run = lambda *c: subprocess.run([py, *c], cwd=ROOT, env=env, check=True)
    run("tests/run_tests.py", "pre", "--device", a.device, "--only", "M1,M2,M5,M6,M7,M8,M10")
    cfg = load_cfg()
    ov = {"numerics": {"M": 2 ** 18}}
    for exp in ("gauss", "population", "network"):
        out = []
        for s in jobs(exp, cfg):
            if s["k"] not in ((2, 8) if exp != "network" else (2,)) or s["basis"] != "pair" or s["seed"] >= (3 if exp != "network" else 1):
                continue
            if exp == "network" and s["rho_S"] != 0.2:
                continue
            s = {kk: v for kk, v in s.items() if kk != "id"}
            if exp == "gauss":
                s["n_est"] = 10 ** 6
            s["override"] = ov if exp != "network" else {"network": {"seeds": 1, "adam": {"steps": 500}, "lbfgs": {"outer": 10}}}
            out.append(with_id(s))
        with open(os.path.join(res, f"jobs_{exp}.jsonl"), "w") as f:
            f.writelines(json.dumps(s) + "\n" for s in out)
    run("scheduler/launcher.py", "--worker", "--device", a.device, "--jobs",
        *[os.path.join(res, f"jobs_{e}.jsonl") for e in ("gauss", "population", "network")])
    run("scheduler/merge.py", "gauss", "population", "network")
    run("-m", "analysis.thresholds", "gauss", "population")
    run("-m", "analysis.figures")
    print("small reproduction done: results/small/, figures/small/")


if __name__ == "__main__":
    main()
