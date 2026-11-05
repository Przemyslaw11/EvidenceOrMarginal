"""Merge per-job result files into results/merged/<exp>.parquet and report completeness.

    python scheduler/merge.py [exp ...]
Writes results/merged/<exp>.parquet, <exp>_meta.csv (one row per job: wall time, GPU, versions, commit)
and <exp>_status.json (expected / done / failed / missing job ids).
"""
import glob
import json
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("HCR_RESULTS", os.path.join(ROOT, "results"))
sys.path.insert(0, ROOT)
from scheduler.launcher import MAX_ATTEMPTS, n_failures  # noqa: E402


def merge(exp):
    specs = [json.loads(line) for line in open(os.path.join(RES, f"jobs_{exp}.jsonl")) if line.strip()]
    ids = {s["id"] for s in specs}
    files = sorted(glob.glob(os.path.join(RES, "jobs", specs[0].get("group", exp) if specs else exp, "*.parquet")))
    done = {os.path.basename(f)[: -len(".parquet")] for f in files}
    stray = sorted(done - ids)  # results of jobs no longer in the list are not merged
    frames, metas = [], []
    for f in files:
        jid = os.path.basename(f)[: -len(".parquet")]
        if jid not in ids:
            continue
        frames.append(pd.read_parquet(f))
        meta = json.loads(pq.read_schema(f).metadata[b"hcr_meta"])
        metas.append(dict(job_id=jid, wall_s=meta["wall_s"], est_s=meta["est_s"], gpu=meta["gpu"], host=meta["host"],
                          git=meta["git"], torch=meta["torch"], numpy=meta["numpy"], nvidia_smi=meta.get("nvidia_smi", ""),
                          expected_min=meta["spec"].get("expected_min")))
    os.makedirs(os.path.join(RES, "merged"), exist_ok=True)
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values([c for c in ("k", "type", "family", "basis", "seed", "case", "m", "belief", "eps") if c in df])
        df.to_parquet(os.path.join(RES, "merged", f"{exp}.parquet"), index=False)
        pd.DataFrame(metas).to_csv(os.path.join(RES, "merged", f"{exp}_meta.csv"), index=False)
    failed = sorted(i for i in ids - done if n_failures(i) >= MAX_ATTEMPTS)
    missing = sorted(ids - done - set(failed))
    status = dict(exp=exp, expected=len(ids), done=len(done & ids), failed=failed, missing=missing, stray=stray,
                  gpu_hours=sum(m["wall_s"] for m in metas) / 3600)
    with open(os.path.join(RES, "merged", f"{exp}_status.json"), "w") as f:
        json.dump(status, f, indent=1)
    print(f"{exp}: {status['done']}/{status['expected']} done, {len(failed)} failed, {len(missing)} missing, "
          f"{status['gpu_hours']:.2f} GPU-h")
    return status


if __name__ == "__main__":
    for e in sys.argv[1:] or [os.path.basename(p)[5:-6] for p in glob.glob(os.path.join(RES, "jobs_*.jsonl"))]:
        merge(e)
