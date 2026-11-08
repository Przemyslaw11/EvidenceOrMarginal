"""Ports: the interfaces through which the use cases reach files, the scheduler and the hardware.

Only real I/O boundaries have a port. Tensors, arrays and data frames pass through unchanged; the configuration is the plain
dict of config.yaml.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import pandas as pd

from densprop.domain.jobs import Spec


class ResultStore(Protocol):
    """Per-job results with their run metadata, and the merged results of each experiment."""

    def job_path(self, spec: Spec) -> str:
        """Where the job's result file goes."""

    def done(self, spec: Spec) -> bool: ...

    def job_ids(self, group: str) -> list[str]:
        """Ids of the jobs of a group (an experiment or the pilot) that have a result, sorted."""

    def read_job(self, group: str, job_id: str) -> tuple[pd.DataFrame, dict]:
        """The rows of one job and its run metadata."""

    def load(self, exp: str, columns: list[str] | None = None, filters: list | None = None) -> pd.DataFrame | None:
        """The merged results of an experiment, or None if there are none."""

    def write_merged(self, exp: str, df: pd.DataFrame | None, metas: pd.DataFrame | None, status: dict) -> None: ...


class JobQueue(Protocol):
    """A job list with claims, so that workers never run a job twice at the same time, and a record of failed attempts."""

    def specs(self) -> list[Spec]: ...

    def claim(self, job_id: str, stale_min: float) -> str | None:
        """A claim token, or None if another worker holds a fresh claim."""

    def release(self, claim: str) -> None: ...

    def n_failures(self, job_id: str) -> int: ...

    def record_failure(self, job_id: str, record: dict) -> None: ...


# Runs one job outside the worker (spec, claim, device, timeout in minutes) and reports (success, wall seconds).
JobRunner = Callable[[Spec, str, str, float], tuple[bool, float]]


class RecordStore(Protocol):
    """Small result records (CSV, JSON, text) by name, relative to a directory: the summaries, the reports, the paper's inputs."""

    def exists(self, name: str) -> bool: ...

    def names(self, pattern: str, sort: bool = True) -> list[str]:
        """Names matching a glob pattern, sorted (or in directory order)."""

    def read_csv(self, name: str, **kw: Any) -> pd.DataFrame: ...

    def read_json(self, name: str) -> Any: ...

    def read_text(self, name: str) -> str: ...

    def write_csv(self, name: str, df: pd.DataFrame, **kw: Any) -> None:
        """Without the index unless asked; other to_csv options pass through."""

    def write_json(self, name: str, obj: Any, **kw: Any) -> None:
        """json.dump with the given formatting (indent, sort_keys, default): each record keeps its historical format."""

    def write_text(self, name: str, text: str) -> None: ...


class EnergyMeter(Protocol):
    """Power and energy of the GPU that runs this process."""

    def info(self) -> dict: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def energy(self, t0: float, t1: float) -> tuple[float, float]:
        """Energy (J) and mean power (W) over [t0, t1] (time.perf_counter seconds)."""

    def counter(self) -> float:
        """The device's energy counter (J)."""

    def clocks(self) -> tuple[int, int, int]:
        """SM clock, memory clock (MHz) and temperature (C)."""
