"""ResultStore on parquet files under results/.

jobs/<group>/<id>.parquet      one job's rows; its run metadata as JSON under the schema metadata key b"hcr_meta"
merged/<exp>.parquet           all rows of an experiment
merged/<exp>_meta.csv          one row per job: wall time, GPU, versions, commit
merged/<exp>_status.json       expected / done / failed / missing job ids
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from densprop.domain.jobs import Spec

META_KEY = b"hcr_meta"


def write_atomic(df: pd.DataFrame, meta: dict, out: str) -> None:
    """Write a job's rows with its metadata; the file appears only when complete (the worker counts it as done)."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = f"{out}.tmp.{os.getpid()}"
    t = pa.Table.from_pandas(df, preserve_index=False)
    t = t.replace_schema_metadata({**(t.schema.metadata or {}), META_KEY: json.dumps(meta, default=str).encode()})
    pq.write_table(t, tmp)
    os.replace(tmp, out)


def read_meta(path: str) -> dict:
    return json.loads(pq.read_schema(path).metadata[META_KEY])


class ParquetStore:
    def __init__(self, results: str):
        self.results = str(results)

    def job_path(self, spec: Spec) -> str:
        return os.path.join(self.results, "jobs", spec.get("group", spec["exp"]), spec["id"] + ".parquet")

    def done(self, spec: Spec) -> bool:
        return os.path.exists(self.job_path(spec))

    def job_ids(self, group: str) -> list[str]:
        d = os.path.join(self.results, "jobs", group)
        return sorted(f[: -len(".parquet")] for f in os.listdir(d) if f.endswith(".parquet")) if os.path.isdir(d) else []

    def read_job(self, group: str, job_id: str) -> tuple[pd.DataFrame, dict]:
        path = os.path.join(self.results, "jobs", group, job_id + ".parquet")
        return pd.read_parquet(path), read_meta(path)

    def all_metas(self) -> list[dict]:
        """The metadata of every per-job file of every group."""
        d = os.path.join(self.results, "jobs")
        groups = sorted(os.listdir(d)) if os.path.isdir(d) else []
        return [read_meta(os.path.join(d, g, j + ".parquet")) for g in groups for j in self.job_ids(g)]

    def load(self, exp: str, columns: list[str] | None = None, filters: list | None = None) -> pd.DataFrame | None:
        path = os.path.join(self.results, "merged", f"{exp}.parquet")
        if not os.path.exists(path):
            return None
        return pd.read_parquet(path, columns=columns, filters=filters)

    def write_merged(self, exp: str, df: pd.DataFrame | None, metas: pd.DataFrame | None, status: dict) -> None:
        d = os.path.join(self.results, "merged")
        os.makedirs(d, exist_ok=True)
        if df is not None:
            df.to_parquet(os.path.join(d, f"{exp}.parquet"), index=False)
            metas.to_csv(os.path.join(d, f"{exp}_meta.csv"), index=False)
        with open(os.path.join(d, f"{exp}_status.json"), "w") as f:
            json.dump(status, f, indent=1)
