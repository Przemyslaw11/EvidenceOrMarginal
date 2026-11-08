"""Idempotent job runner: one worker per GPU pulls jobs from a job list.

    python scheduler/launcher.py --jobs results/jobs_gauss.jsonl --gpus 0-3    # local multi-GPU
    python scheduler/launcher.py --jobs results/jobs_gauss.jsonl --worker      # one worker (Slurm task)

A job is done when results/jobs/<exp>/<id>.parquet exists (written atomically by gpu.job).
A worker claims a job by creating results/claims/<id>.claim with O_EXCL (atomic on NFS) and keeps
its mtime fresh while the job runs; a claim older than --stale minutes is taken over by renaming it
(only one renamer wins). Each attempt runs in a subprocess; failures are recorded in
results/failures/<id>.<n>.json and a job is retried at most twice (three attempts).
"""

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("HCR_RESULTS", os.path.join(ROOT, "results"))
MAX_ATTEMPTS = 3


def out_path(spec):
    return os.path.join(RES, "jobs", spec.get("group", spec["exp"]), spec["id"] + ".parquet")


def n_failures(jid):
    d = os.path.join(RES, "failures")
    return sum(1 for f in os.listdir(d) if f.startswith(jid + ".")) if os.path.isdir(d) else 0


def try_claim(jid, stale_min):
    os.makedirs(os.path.join(RES, "claims"), exist_ok=True)
    path = os.path.join(RES, "claims", jid + ".claim")
    info = json.dumps(dict(host=socket.gethostname(), pid=os.getpid(), slurm=os.environ.get("SLURM_JOB_ID", ""), time=time.time()))
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, info.encode())
            os.close(fd)
            return path
        except FileExistsError:
            try:
                age = (time.time() - os.path.getmtime(path)) / 60
            except FileNotFoundError:
                continue
            if age < stale_min:
                return None
            with contextlib.suppress(FileNotFoundError):  # take over a stale claim: rename is atomic, only one worker succeeds
                os.rename(path, f"{path}.stale.{socket.gethostname()}.{os.getpid()}")
    return None


def run_one(spec, claim, device, timeout_min):
    out = out_path(spec)
    cmd = [sys.executable, "-m", "gpu.job", "--spec", json.dumps(spec), "--out", out, "--device", device]
    t0 = time.time()
    os.makedirs(os.path.join(RES, "logs"), exist_ok=True)
    logf = os.path.join(RES, "logs", spec["id"] + ".log")
    with open(logf, "a") as lf:
        lf.write(f"--- {time.ctime()} {socket.gethostname()} slurm={os.environ.get('SLURM_JOB_ID', '')}\n")
        lf.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, text=True)
        killed = False
        while True:
            try:
                p.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                os.utime(claim)  # heartbeat
                if time.time() - t0 > timeout_min * 60:
                    p.kill()
                    p.wait()
                    killed = True
                    break
    lines = [open(logf).read()[-6000:] + (f"\nKILLED after {timeout_min} min" if killed else "")]
    ok = p.returncode == 0 and os.path.exists(out)
    if not ok:
        os.makedirs(os.path.join(RES, "failures"), exist_ok=True)
        rec = dict(
            id=spec["id"],
            returncode=p.returncode,
            host=socket.gethostname(),
            slurm=os.environ.get("SLURM_JOB_ID", ""),
            wall_s=time.time() - t0,
            log="".join(lines)[-6000:],
        )
        with open(os.path.join(RES, "failures", f"{spec['id']}.{n_failures(spec['id'])}.json"), "w") as f:
            json.dump(rec, f, indent=1)
    os.remove(claim)
    return ok, time.time() - t0


def parse_left(s):
    """Slurm time left ([D-]HH:MM:SS, MM:SS) in minutes."""
    d, _, hms = s.strip().rpartition("-")
    parts = [int(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(d or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]) / 60


def worker(jobs_file, device, stale_min, timeout_factor, max_jobs, deadline=None):
    specs = [json.loads(line) for line in open(jobs_file) if line.strip()]
    done = 0
    while max_jobs is None or done < max_jobs:
        progressed = False
        for spec in specs:
            if os.path.exists(out_path(spec)) or n_failures(spec["id"]) >= MAX_ATTEMPTS:
                continue
            # do not start a job that would not finish inside the allocation
            if deadline is not None and time.time() + 60 * 1.3 * spec.get("expected_min", 10) > deadline:
                continue
            claim = try_claim(spec["id"], stale_min)
            if claim is None:
                continue
            if os.path.exists(out_path(spec)):  # finished by another worker in the meantime
                os.remove(claim)
                continue
            timeout = max(30.0, timeout_factor * spec.get("expected_min", 10))
            ok, wall = run_one(spec, claim, device, timeout)
            print(
                f"[{socket.gethostname()}:{os.environ.get('CUDA_VISIBLE_DEVICES', '-')}] {spec['exp']} {spec['id']} "
                f"k={spec['k']} {spec.get('type')} {spec['basis']} seed={spec['seed']} {'OK' if ok else 'FAIL'} "
                f"{wall / 60:.1f} min",
                flush=True,
            )
            done += 1
            progressed = True
            break
        if not progressed:
            return


def parse_gpus(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, nargs="+")
    ap.add_argument("--gpus", default=None, help="e.g. 0-3: one worker per GPU on this machine")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stale", type=float, default=15.0, help="minutes without heartbeat before a claim is stale")
    ap.add_argument("--timeout-factor", type=float, default=6.0)
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--time-left", default=None, help="time left in the allocation (squeue %%L format)")
    a = ap.parse_args()
    if a.gpus is not None:
        procs = []
        for g in parse_gpus(a.gpus):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        "--worker",
                        "--device",
                        a.device,
                        "--jobs",
                        *a.jobs,
                        "--stale",
                        str(a.stale),
                        "--timeout-factor",
                        str(a.timeout_factor),
                    ]
                    + (["--max-jobs", str(a.max_jobs)] if a.max_jobs else [])
                    + (["--time-left", a.time_left] if a.time_left else []),
                    env=env,
                )
            )
        sys.exit(max(p.wait() for p in procs))
    deadline = time.time() + 60 * (parse_left(a.time_left) - 2) if a.time_left else None
    for jf in a.jobs:
        worker(jf, a.device, a.stale, a.timeout_factor, a.max_jobs, deadline)


if __name__ == "__main__":
    main()
