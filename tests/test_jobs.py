"""Job specifications: ids, random streams and the stored job lists (invariant: nothing here may change)."""

import hashlib
import json

import numpy as np
import pytest

from densprop.domain.jobs import CODE, STREAM, job_list, jobs, rng_for, small_jobs, with_id

EXPERIMENTS = ["gauss", "population", "convergence", "families", "lkj", "network"]


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_job_lists_reproduce_the_stored_lists(exp, cfg, cost, root):
    stored = [json.loads(line) for line in (root / "results" / f"jobs_{exp}.jsonl").read_text().splitlines() if line.strip()]
    assert job_list(exp, cfg, cost) == stored


def test_job_list_lines_are_byte_identical(cfg, cost, root):
    text = "".join(json.dumps(s) + "\n" for s in job_list("gauss", cfg, cost))
    assert text == (root / "results" / "jobs_gauss.jsonl").read_text()


def test_id_is_the_sha1_of_the_sorted_spec():
    spec = with_id({"b": 1, "a": [2, 3]})
    assert spec["id"] == hashlib.sha1(json.dumps({"a": [2, 3], "b": 1}, sort_keys=True).encode()).hexdigest()[:16]
    assert with_id({"a": [2, 3], "b": 1})["id"] == spec["id"]


def test_random_streams_are_seed_sequence_children(cfg):
    key = [CODE["gauss"], 8, 3]
    for stream, code in STREAM.items():
        want = np.random.Generator(np.random.PCG64(np.random.SeedSequence(cfg["root_seed"], spawn_key=(*key, code))))
        assert rng_for(cfg, key, stream).random(4).tolist() == want.random(4).tolist()
    extra = np.random.Generator(np.random.PCG64(np.random.SeedSequence(cfg["root_seed"], spawn_key=(*key, STREAM["lkj"], 7))))
    assert rng_for(cfg, key, "lkj", 7).integers(1 << 30) == extra.integers(1 << 30)


def test_crn_key_omits_type_and_basis(cfg):
    by_seed = {}
    for s in jobs("gauss", cfg):
        by_seed.setdefault((s["k"], s["seed"]), set()).add(tuple(s["key"]))
    assert all(len(v) == 1 for v in by_seed.values())


def test_small_reproduction_jobs(cfg):
    g = small_jobs("gauss", cfg)
    assert {s["k"] for s in g} == {2, 8}
    assert {s["basis"] for s in g} == {"pair"}
    assert max(s["seed"] for s in g) == 2
    assert all(s["n_est"] == 10**6 and s["override"] == {"numerics": {"M": 2**18}} for s in g)
    net = small_jobs("network", cfg)
    assert len(net) == 1
    assert net[0]["k"] == 2
    assert net[0]["rho_S"] == 0.2
