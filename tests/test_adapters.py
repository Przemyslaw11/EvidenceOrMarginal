"""Contract tests of the file adapters, in temporary directories."""

import json
import os
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from densprop.adapters.config import load_cost, save_cost
from densprop.adapters.job_files import JobFiles, parse_left, read_job_list, write_job_list
from densprop.adapters.parquet_store import META_KEY, ParquetStore, read_meta, write_atomic
from densprop.adapters.records import FileRecords

SPEC = {"exp": "gauss", "id": "abc123", "k": 2}


def test_job_file_round_trip_with_metadata(tmp_path):
    store = ParquetStore(str(tmp_path))
    out = store.job_path(SPEC)
    assert out == os.path.join(str(tmp_path), "jobs", "gauss", "abc123.parquet")
    assert not store.done(SPEC)
    df = pd.DataFrame(dict(a=[1.0, 2.0], b=["x", "y"]))
    write_atomic(df, {"spec": SPEC, "wall_s": 1.5}, out)
    assert store.done(SPEC)
    assert store.job_ids("gauss") == ["abc123"]
    assert META_KEY == b"hcr_meta"
    assert json.loads(pq.read_schema(out).metadata[META_KEY]) == read_meta(out)
    got, meta = store.read_job("gauss", "abc123")
    pd.testing.assert_frame_equal(got, df)
    assert meta == {"spec": SPEC, "wall_s": 1.5}
    assert store.all_metas() == [meta]
    assert not [f for f in os.listdir(os.path.dirname(out)) if ".tmp." in f]


def test_pilot_jobs_go_to_their_group(tmp_path):
    assert ParquetStore(str(tmp_path)).job_path(dict(SPEC, group="pilot")).endswith(os.path.join("jobs", "pilot", "abc123.parquet"))


def test_merged_results_and_status(tmp_path):
    store = ParquetStore(str(tmp_path))
    assert store.load("gauss") is None
    store.write_merged("gauss", pd.DataFrame(dict(k=[2], v=[0.5])), pd.DataFrame(dict(job_id=["abc123"])), {"exp": "gauss", "done": 1})
    assert store.load("gauss", columns=["v"]).v.tolist() == [0.5]
    assert (tmp_path / "merged" / "gauss_status.json").read_text() == '{\n "exp": "gauss",\n "done": 1\n}'
    store.write_merged("lkj", None, None, {"exp": "lkj"})
    assert not (tmp_path / "merged" / "lkj.parquet").exists()
    assert (tmp_path / "merged" / "lkj_status.json").exists()


def test_job_list_round_trip(tmp_path):
    path = str(tmp_path / "jobs_gauss.jsonl")
    write_job_list(path, [SPEC, dict(SPEC, id="def456")])
    assert (tmp_path / "jobs_gauss.jsonl").read_text() == json.dumps(SPEC) + "\n" + json.dumps(dict(SPEC, id="def456")) + "\n"
    assert read_job_list(path)[1]["id"] == "def456"


def test_claims_are_exclusive_until_stale(tmp_path):
    a, b = JobFiles("", str(tmp_path), "1"), JobFiles("", str(tmp_path), "2")
    claim = a.claim("abc123", stale_min=15)
    assert claim
    assert json.loads(Path(claim).read_text())["slurm"] == "1"
    assert b.claim("abc123", stale_min=15) is None
    old = time.time() - 3600
    os.utime(claim, (old, old))
    taken = b.claim("abc123", stale_min=15)
    assert taken == claim
    assert json.loads(Path(taken).read_text())["slurm"] == "2"
    assert any(".stale." in f for f in os.listdir(tmp_path / "claims"))
    b.release(taken)
    assert not os.path.exists(taken)


def test_failures_are_counted_per_job(tmp_path):
    q = JobFiles("", str(tmp_path))
    assert q.n_failures("abc123") == 0
    q.record_failure("abc123", {"returncode": 1})
    q.record_failure("abc123", {"returncode": 2})
    q.record_failure("abc999", {"returncode": 3})
    assert q.n_failures("abc123") == 2
    assert json.loads((tmp_path / "failures" / "abc123.1.json").read_text())["returncode"] == 2


def test_slurm_time_left():
    assert parse_left("1-02:03:04") == (86400 + 2 * 3600 + 3 * 60 + 4) / 60
    assert parse_left("07:30") == 7.5
    assert parse_left("01:00:00") == 60


def test_records_round_trip(tmp_path):
    r = FileRecords(str(tmp_path))
    r.write_csv("sub/a.csv", pd.DataFrame(dict(x=[1, 2])))
    assert (tmp_path / "sub" / "a.csv").read_text() == "x\n1\n2\n"
    r.write_json("b.json", {"z": 1, "a": 2}, indent=1, sort_keys=True)
    assert r.read_json("b.json") == {"a": 2, "z": 1}
    assert (tmp_path / "b.json").read_text() == '{\n "a": 2,\n "z": 1\n}'
    r.write_text("c_1.txt", "hi\n")
    r.write_text("c_0.txt", "ho\n")
    assert r.names("c_*.txt") == ["c_0.txt", "c_1.txt"]
    assert sorted(r.names("c_*.txt", sort=False)) == ["c_0.txt", "c_1.txt"]
    assert r.exists("c_0.txt")
    assert not r.exists("d.txt")
    assert r.read_text("c_1.txt") == "hi\n"
    assert r.read_csv("sub/a.csv").x.tolist() == [1, 2]


def test_cost_table(tmp_path):
    path = str(tmp_path / "cost.json")
    assert load_cost(path) == {}
    save_cost(path, {"cells": {}})
    assert load_cost(path) == {"cells": {}}
