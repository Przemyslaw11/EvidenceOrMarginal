"""The worker and the merge against in-memory ports."""

import pandas as pd

from densprop.app.merge import merge
from densprop.app.worker import MAX_ATTEMPTS, worker

SPECS = [{"id": f"j{i}", "exp": "gauss", "k": 2, "type": "SUM", "basis": "pair", "seed": i, "expected_min": 1.0} for i in range(3)]


class MemoryQueue:
    def __init__(self, specs, failures=None):
        self._specs, self.failures, self.claims = specs, dict(failures or {}), set()

    def specs(self):
        return self._specs

    def claim(self, job_id, stale_min):
        if job_id in self.claims:
            return None
        self.claims.add(job_id)
        return job_id

    def release(self, claim):
        self.claims.discard(claim)

    def n_failures(self, job_id):
        return self.failures.get(job_id, 0)

    def record_failure(self, job_id, record):
        self.failures[job_id] = self.failures.get(job_id, 0) + 1


class MemoryStore:
    def __init__(self):
        self.rows, self.metas, self.merged = {}, {}, {}

    def job_path(self, spec):
        return spec["id"]

    def done(self, spec):
        return spec["id"] in self.rows

    def job_ids(self, group):
        return sorted(self.rows)

    def read_job(self, group, job_id):
        return self.rows[job_id], self.metas[job_id]

    def load(self, exp, columns=None, filters=None):
        return self.merged.get(exp, (None,))[0]

    def write_merged(self, exp, df, metas, status):
        self.merged[exp] = (df, metas, status)


def finished(store, spec, wall=60.0):
    store.rows[spec["id"]] = pd.DataFrame(dict(k=[spec["k"]], seed=[spec["seed"]], type=["SUM"], basis=["pair"], v=[float(spec["seed"])]))
    store.metas[spec["id"]] = dict(spec=spec, wall_s=wall, est_s=1.0, gpu="g", host="h", git="c", torch="t", numpy="n")


def test_worker_runs_each_job_once_and_skips_done_and_failed():
    store, queue, ran = MemoryStore(), MemoryQueue(SPECS, failures={"j2": MAX_ATTEMPTS}), []
    finished(store, SPECS[0])

    def run(spec, claim, device, timeout):
        ran.append((spec["id"], device, timeout))
        finished(store, spec)
        queue.release(claim)
        return True, 1.0

    worker(queue, store, run, "cpu", stale_min=15, timeout_factor=6, max_jobs=None, log=lambda s: None)
    assert ran == [("j1", "cpu", 30.0)]
    assert not queue.claims


def test_worker_retries_a_failing_job_until_the_limit():
    store, queue, ran = MemoryStore(), MemoryQueue(SPECS[:1]), []

    def run(spec, claim, device, timeout):
        ran.append(spec["id"])
        queue.record_failure(spec["id"], {})
        queue.release(claim)
        return False, 1.0

    worker(queue, store, run, "cpu", 15, 6, max_jobs=None, log=lambda s: None)
    assert ran == ["j0"] * MAX_ATTEMPTS


def test_worker_stops_after_max_jobs_and_before_the_deadline():
    store, queue, ran = MemoryStore(), MemoryQueue(SPECS), []

    def run(spec, claim, device, timeout):
        ran.append(spec["id"])
        finished(store, spec)
        queue.release(claim)
        return True, 1.0

    worker(queue, store, run, "cpu", 15, 6, max_jobs=2, log=lambda s: None)
    assert ran == ["j0", "j1"]
    worker(queue, store, run, "cpu", 15, 6, max_jobs=None, deadline=0.0, log=lambda s: None)
    assert ran == ["j0", "j1"]


def test_merge_reports_done_failed_missing_and_stray():
    store = MemoryStore()
    finished(store, SPECS[1], wall=7200.0)
    finished(store, SPECS[0], wall=3600.0)
    finished(store, dict(SPECS[0], id="old", seed=9))
    queue = MemoryQueue(SPECS, failures={"j2": MAX_ATTEMPTS})
    status = merge("gauss", SPECS, store, queue, log=lambda s: None)
    assert status == dict(exp="gauss", expected=3, done=2, failed=["j2"], missing=[], stray=["old"], gpu_hours=3.0)
    df, metas, _ = store.merged["gauss"]
    assert df.seed.tolist() == [0, 1]
    assert metas.job_id.tolist() == ["j0", "j1"]
    assert metas.expected_min.tolist() == [1.0, 1.0]
    assert (metas.nvidia_smi == "").all()


def test_merge_without_results_writes_only_the_status():
    store = MemoryStore()
    status = merge("gauss", SPECS, store, MemoryQueue(SPECS), log=lambda s: None)
    assert status["missing"] == ["j0", "j1", "j2"]
    assert store.merged["gauss"][0] is None
