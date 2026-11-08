"""Idempotent job runner: a worker pulls jobs from a job list until none is left that it may run.

A job is done when its result exists. A job is claimed before it runs, so that workers never run it twice at the same time;
a job is retried at most twice (three attempts).
"""

from __future__ import annotations

import time
from collections.abc import Callable

from densprop.ports import JobQueue, JobRunner, ResultStore

MAX_ATTEMPTS = 3


def worker(
    queue: JobQueue,
    store: ResultStore,
    run: JobRunner,
    device: str,
    stale_min: float,
    timeout_factor: float,
    max_jobs: int | None,
    deadline: float | None = None,
    log: Callable[[str], None] = print,
) -> None:
    specs = queue.specs()
    done = 0
    while max_jobs is None or done < max_jobs:
        progressed = False
        for spec in specs:
            if store.done(spec) or queue.n_failures(spec["id"]) >= MAX_ATTEMPTS:
                continue
            # do not start a job that would not finish inside the allocation
            if deadline is not None and time.time() + 60 * 1.3 * spec.get("expected_min", 10) > deadline:
                continue
            claim = queue.claim(spec["id"], stale_min)
            if claim is None:
                continue
            if store.done(spec):  # finished by another worker in the meantime
                queue.release(claim)
                continue
            timeout = max(30.0, timeout_factor * spec.get("expected_min", 10))
            ok, wall = run(spec, claim, device, timeout)
            log(
                f"{spec['exp']} {spec['id']} k={spec['k']} {spec.get('type')} {spec['basis']} seed={spec['seed']} "
                f"{'OK' if ok else 'FAIL'} {wall / 60:.1f} min"
            )
            done += 1
            progressed = True
            break
        if not progressed:
            return
