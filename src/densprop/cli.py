"""Command line: wires the use cases to files, the environment and the hardware.

    densprop make-jobs gauss population convergence families lkj network    # results/jobs_<exp>.jsonl
    densprop launch --jobs results/jobs_gauss.jsonl --gpus 0-3               # one worker per local GPU
    densprop launch --jobs results/jobs_gauss.jsonl --worker                 # one worker (a Slurm task)
    densprop merge [exp ...]                                                 # results/merged/
    densprop profile-cost --device cuda | --from-results                     # scheduler/cost_model.json
    densprop small [--device cuda]                                           # small reproduction, results/small/

`python -m densprop` is the same. Results go to $HCR_RESULTS (default results/).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
CONFIG = os.path.join(ROOT, "config.yaml")
COST = os.path.join(ROOT, "scheduler", "cost_model.json")


def results_dir() -> str:
    return os.environ.get("HCR_RESULTS", os.path.join(ROOT, "results"))


def make_jobs(a: argparse.Namespace) -> None:
    from densprop.adapters.config import load_config, load_cost
    from densprop.adapters.job_files import write_job_list
    from densprop.domain.jobs import job_list

    cfg, cost = load_config(CONFIG), load_cost(COST)
    for exp in a.exp or ["gauss", "population", "families", "lkj", "convergence"]:
        js = job_list(exp, cfg, cost)
        path = os.path.join(ROOT, "results", f"jobs_{exp}.jsonl")
        write_job_list(path, js)
        print(f"{exp}: {len(js)} jobs, expected {sum(s['expected_min'] for s in js) / 60:.1f} GPU-h -> {path}")


def setup(device: str) -> None:
    from densprop.domain.sweep import setup_torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS
    setup_torch(device)


def job(a: argparse.Namespace) -> None:
    """One job in this process; the worker runs each job like this."""
    from densprop.adapters.config import load_config
    from densprop.adapters.parquet_store import write_atomic
    from densprop.adapters.runtime import provenance
    from densprop.domain.sweep import run_job

    t0 = time.time()
    import torch

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    setup(device)
    spec = json.loads(a.spec)
    df, t_est = run_job(spec, load_config(a.config or CONFIG), device, log=lambda s: print(s, flush=True))
    write_atomic(df, provenance(spec, t0, t_est, device, ROOT), a.out)


def parse_gpus(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def launch(a: argparse.Namespace) -> None:
    from densprop.adapters.job_files import JobFiles, SubprocessRunner, parse_left
    from densprop.adapters.parquet_store import ParquetStore
    from densprop.app.worker import worker

    if a.gpus is not None:
        procs = []
        for g in parse_gpus(a.gpus):
            cmd = [sys.executable, "-m", "densprop", "launch", "--worker", "--device", a.device, "--jobs", *a.jobs]
            cmd += ["--stale", str(a.stale), "--timeout-factor", str(a.timeout_factor)]
            cmd += (["--max-jobs", str(a.max_jobs)] if a.max_jobs else []) + (["--time-left", a.time_left] if a.time_left else [])
            procs.append(subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))))
        sys.exit(max(p.wait() for p in procs))
    deadline = time.time() + 60 * (parse_left(a.time_left) - 2) if a.time_left else None
    res, slurm = results_dir(), os.environ.get("SLURM_JOB_ID", "")
    store = ParquetStore(res)
    tag = f"[{socket.gethostname()}:{os.environ.get('CUDA_VISIBLE_DEVICES', '-')}] "
    for jf in a.jobs:
        queue = JobFiles(jf, res, slurm)
        run = SubprocessRunner(queue, store.job_path, ROOT)
        worker(queue, store, run, a.device, a.stale, a.timeout_factor, a.max_jobs, deadline, log=lambda s: print(tag + s, flush=True))


def merge(a: argparse.Namespace) -> None:
    from densprop.adapters.job_files import JobFiles
    from densprop.adapters.parquet_store import ParquetStore
    from densprop.app.merge import merge as merge_exp

    res = results_dir()
    store = ParquetStore(res)
    for exp in a.exp or sorted(os.path.basename(p)[5:-6] for p in glob.glob(os.path.join(res, "jobs_*.jsonl"))):
        queue = JobFiles(os.path.join(res, f"jobs_{exp}.jsonl"), res)
        merge_exp(exp, queue.specs(), store, queue)


def profile_cost(a: argparse.Namespace) -> None:
    from densprop.adapters.config import load_config, load_cost, save_cost
    from densprop.app import profile_cost as pc

    if a.from_results:
        from densprop.adapters.parquet_store import ParquetStore

        cost = pc.from_results(load_cost(COST), ParquetStore(os.path.join(ROOT, "results")).all_metas())
        save_cost(COST, cost)
        print({k: (round(v["est_s_per_case"], 1), round(v["case_s"], 1)) for k, v in cost["cells"].items()})
        return
    import torch

    setup(a.device)
    name = torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu"
    ks = [int(k) for k in a.k.split(",")]
    pc.profile(load_config(CONFIG), a.device, name, ks, a.n_est, a.network, lambda c: save_cost(COST, c), log=lambda s: print(s, flush=True))


def small(a: argparse.Namespace) -> None:
    """Small end-to-end reproduction on one GPU (or CPU), about 30 minutes on an A100, into results/small/ and figures/small/.

    The fast validation tests, a reduced sweep and the matching population run, one small chain/training job, then the
    analysis and figures.
    """
    from densprop.adapters.config import load_config
    from densprop.adapters.job_files import write_job_list
    from densprop.domain.jobs import small_jobs

    res = os.path.join(ROOT, "results", "small")
    os.makedirs(res, exist_ok=True)
    env = dict(os.environ, HCR_RESULTS=res, HCR_FIGURES=os.path.join(ROOT, "figures", "small"))

    def run(*c: str) -> None:
        subprocess.run([sys.executable, *c], cwd=ROOT, env=env, check=True)

    run("tests/run_tests.py", "pre", "--device", a.device, "--only", "M1,M2,M5,M6,M7,M8,M10")
    cfg = load_config(CONFIG)
    exps = ("gauss", "population", "network")
    for exp in exps:
        write_job_list(os.path.join(res, f"jobs_{exp}.jsonl"), small_jobs(exp, cfg))
    run("-m", "densprop", "launch", "--worker", "--device", a.device, "--jobs", *[os.path.join(res, f"jobs_{e}.jsonl") for e in exps])
    run("-m", "densprop", "merge", *exps)
    run("-m", "analysis.thresholds", "gauss", "population")
    run("-m", "analysis.figures")
    print("small reproduction done: results/small/, figures/small/")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="densprop", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-jobs", help="write the job lists results/jobs_<exp>.jsonl ('pilot' for the throughput pilot)")
    p.add_argument("exp", nargs="*")
    p.set_defaults(fn=make_jobs)

    p = sub.add_parser("job", help="run one job and write its result file")
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    p.add_argument("--config", default=None)
    p.set_defaults(fn=job)

    p = sub.add_parser("launch", help="run the jobs of job lists: one worker, or one per local GPU")
    p.add_argument("--jobs", required=True, nargs="+")
    p.add_argument("--gpus", default=None, help="e.g. 0-3: one worker per GPU on this machine")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--stale", type=float, default=15.0, help="minutes without heartbeat before a claim is stale")
    p.add_argument("--timeout-factor", type=float, default=6.0)
    p.add_argument("--max-jobs", type=int, default=None)
    p.add_argument("--time-left", default=None, help="time left in the allocation (squeue %%L format)")
    p.set_defaults(fn=launch)

    p = sub.add_parser("merge", help="merge per-job results into results/merged/ (default: every job list)")
    p.add_argument("exp", nargs="*")
    p.set_defaults(fn=merge)

    p = sub.add_parser("profile-cost", help="measure job cost on this device (or refit it from production metadata)")
    p.add_argument("--from-results", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--k", default="2,4,8,16,32")
    p.add_argument("--n-est", type=int, default=10_000_000)
    p.add_argument("--network", default="8,triple,0.4", help="k,basis,rho_S of the timed network job ('' to skip)")
    p.set_defaults(fn=profile_cost)

    p = sub.add_parser("small", help="small end-to-end reproduction into results/small/")
    p.add_argument("--device", default="cuda")
    p.set_defaults(fn=small)

    a = ap.parse_args(argv)
    a.fn(a)
