"""The command line, end to end in temporary directories."""

import json
import platform

import pytest

GOLDEN_PLATFORM = ("Darwin", "arm64")  # the platform that wrote tests/golden/m_cpu.json


def test_help_lists_every_command(run_cli):
    out = run_cli("--help").stdout
    for cmd in (
        "make-jobs",
        "job",
        "launch",
        "merge",
        "profile-cost",
        "small",
        "thresholds",
        "hypotheses",
        "figures",
        "report",
        "energy",
        "validate",
    ):
        assert cmd in out


def test_energy_self_check(run_cli):
    assert "PASS" in run_cli("energy", "--check").stdout


def test_merge_of_an_empty_job_list(tmp_path, run_cli):
    (tmp_path / "jobs_gauss.jsonl").write_text(json.dumps({"exp": "gauss", "id": "abc"}) + "\n")
    run_cli("merge", "gauss", results=str(tmp_path))
    status = json.loads((tmp_path / "merged" / "gauss_status.json").read_text())
    assert status["missing"] == ["abc"]
    assert status["done"] == 0


@pytest.mark.slow
def test_fast_validation_tests_on_cpu(tmp_path, run_cli, root):
    """M1, M2, M5-M8, M10 as in CI; on the platform of the golden file the details must match it exactly."""
    out = tmp_path / "m.json"
    run_cli("validate", "pre", "--device", "cpu", "--only", "M1,M2,M5,M6,M7,M8,M10", "--out", str(out))
    got = json.loads(out.read_text())
    want = json.loads((root / "tests" / "golden" / "m_cpu.json").read_text())
    assert sorted(got) == sorted(want)
    assert all(got[k]["pass"] for k in got)
    if (platform.system(), platform.machine()) == GOLDEN_PLATFORM:
        assert {k: got[k]["detail"] for k in got} == {k: v["detail"] for k, v in want.items()}


@pytest.mark.slow
@pytest.mark.parquet
def test_population_thresholds_reproduce_the_summary(tmp_path, run_cli, root):
    """The population thresholds from the merged parquet file equal the stored summary byte for byte."""
    merged = root / "results" / "merged" / "population.parquet"
    if not merged.exists():
        pytest.skip("results/merged/population.parquet is not in the repository")
    (tmp_path / "merged").mkdir()
    (tmp_path / "merged" / "population.parquet").symlink_to(merged.resolve())
    run_cli("thresholds", "population", results=str(tmp_path))
    got = (tmp_path / "summary" / "thresholds_population.csv").read_bytes()
    assert got == (root / "results" / "summary" / "thresholds_population.csv").read_bytes()
