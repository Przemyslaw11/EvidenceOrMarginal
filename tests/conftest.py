import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def cfg() -> dict:
    from densprop.adapters.config import load_config

    return load_config(str(ROOT / "config.yaml"))


@pytest.fixture(scope="session")
def cost() -> dict:
    from densprop.adapters.config import load_cost

    return load_cost(str(ROOT / "scheduler" / "cost_model.json"))


@pytest.fixture
def run_cli(root):
    """Run `python -m densprop ...` in the repository with HCR_RESULTS set; returns the CompletedProcess."""
    import subprocess

    def run(*args: str, results: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        if results is not None:
            env["HCR_RESULTS"] = results
        return subprocess.run([sys.executable, "-m", "densprop", *args], cwd=root, env=env, capture_output=True, text=True, check=check)

    return run
