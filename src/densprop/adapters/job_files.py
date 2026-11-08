"""JobQueue on files: a JSONL job list, claim files and failure records under results/, and jobs run as subprocesses.

A worker claims a job by creating results/claims/<id>.claim with O_EXCL (atomic on NFS) and keeps its mtime fresh while the
job runs; a claim older than the stale limit is taken over by renaming it (only one renamer wins). Each attempt runs in a
subprocess with its log in results/logs/<id>.log; a failed attempt is recorded in results/failures/<id>.<n>.json.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import time

from densprop.domain.jobs import Spec


def read_job_list(path: str) -> list[Spec]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_job_list(path: str, specs: list[Spec]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for s in specs:
            f.write(json.dumps(s) + "\n")


def parse_left(s: str) -> float:
    """Slurm time left ([D-]HH:MM:SS, MM:SS) in minutes."""
    d, _, hms = s.strip().rpartition("-")
    parts = [int(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(d or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]) / 60


class JobFiles:
    def __init__(self, jobs_file: str, results: str, slurm: str = ""):
        self.jobs_file, self.results, self.slurm = jobs_file, str(results), slurm

    def specs(self) -> list[Spec]:
        return read_job_list(self.jobs_file)

    def n_failures(self, job_id: str) -> int:
        d = os.path.join(self.results, "failures")
        return sum(1 for f in os.listdir(d) if f.startswith(job_id + ".")) if os.path.isdir(d) else 0

    def record_failure(self, job_id: str, record: dict) -> None:
        os.makedirs(os.path.join(self.results, "failures"), exist_ok=True)
        with open(os.path.join(self.results, "failures", f"{job_id}.{self.n_failures(job_id)}.json"), "w") as f:
            json.dump(record, f, indent=1)

    def claim(self, job_id: str, stale_min: float) -> str | None:
        os.makedirs(os.path.join(self.results, "claims"), exist_ok=True)
        path = os.path.join(self.results, "claims", job_id + ".claim")
        info = json.dumps(dict(host=socket.gethostname(), pid=os.getpid(), slurm=self.slurm, time=time.time()))
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

    def release(self, claim: str) -> None:
        os.remove(claim)


class SubprocessRunner:
    """Runs `python -m densprop job` for one spec with a heartbeat on its claim and a timeout; records failures."""

    def __init__(self, queue: JobFiles, out_path, root: str):
        self.queue, self.out_path, self.root = queue, out_path, root

    def __call__(self, spec: Spec, claim: str, device: str, timeout_min: float) -> tuple[bool, float]:
        out = self.out_path(spec)
        cmd = [sys.executable, "-m", "densprop", "job", "--spec", json.dumps(spec), "--out", out, "--device", device]
        t0 = time.time()
        res = self.queue.results
        os.makedirs(os.path.join(res, "logs"), exist_ok=True)
        logf = os.path.join(res, "logs", spec["id"] + ".log")
        with open(logf, "a") as lf:
            lf.write(f"--- {time.ctime()} {socket.gethostname()} slurm={self.queue.slurm}\n")
            lf.flush()
            p = subprocess.Popen(cmd, cwd=self.root, stdout=lf, stderr=subprocess.STDOUT, text=True)
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
        with open(logf) as lf:
            lines = [lf.read()[-6000:] + (f"\nKILLED after {timeout_min} min" if killed else "")]
        ok = p.returncode == 0 and os.path.exists(out)
        if not ok:
            rec = dict(
                id=spec["id"],
                returncode=p.returncode,
                host=socket.gethostname(),
                slurm=self.queue.slurm,
                wall_s=time.time() - t0,
                log="".join(lines)[-6000:],
            )
            self.queue.record_failure(spec["id"], rec)
        self.queue.release(claim)
        return ok, time.time() - t0
