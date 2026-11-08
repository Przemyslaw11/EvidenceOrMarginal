"""Merge the per-job results of an experiment and report its completeness.

The merged rows, one metadata row per job (wall time, GPU, versions, commit) and a status (expected / done / failed / missing
job ids) go to the result store.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from densprop.app.worker import MAX_ATTEMPTS
from densprop.domain.jobs import Spec
from densprop.ports import JobQueue, ResultStore

SORT = ("k", "type", "family", "basis", "seed", "case", "m", "belief", "eps")


def merge(exp: str, specs: list[Spec], store: ResultStore, queue: JobQueue, log: Callable[[str], None] = print) -> dict:
    ids = {s["id"] for s in specs}
    group = specs[0].get("group", exp) if specs else exp
    files = store.job_ids(group)
    done = set(files)
    stray = sorted(done - ids)  # results of jobs no longer in the list are not merged
    frames, metas = [], []
    for jid in files:
        if jid not in ids:
            continue
        df, meta = store.read_job(group, jid)
        frames.append(df)
        metas.append(
            dict(
                job_id=jid,
                wall_s=meta["wall_s"],
                est_s=meta["est_s"],
                gpu=meta["gpu"],
                host=meta["host"],
                git=meta["git"],
                torch=meta["torch"],
                numpy=meta["numpy"],
                nvidia_smi=meta.get("nvidia_smi", ""),
                expected_min=meta["spec"].get("expected_min"),
            )
        )
    df = None
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values([c for c in SORT if c in df])
    failed = sorted(i for i in ids - done if queue.n_failures(i) >= MAX_ATTEMPTS)
    missing = sorted(ids - done - set(failed))
    status = dict(
        exp=exp,
        expected=len(ids),
        done=len(done & ids),
        failed=failed,
        missing=missing,
        stray=stray,
        gpu_hours=sum(m["wall_s"] for m in metas) / 3600,
    )
    store.write_merged(exp, df, pd.DataFrame(metas) if frames else None, status)
    log(f"{exp}: {status['done']}/{status['expected']} done, {len(failed)} failed, {len(missing)} missing, {status['gpu_hours']:.2f} GPU-h")
    return status
