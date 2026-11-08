"""Where and with what a job ran: host, device, library versions, commit; stored with every job's result."""

from __future__ import annotations

import platform
import socket
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from densprop.domain.jobs import Spec


def git_commit(root: str) -> str:
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def nvidia_smi() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,power.limit", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        return ""


def provenance(spec: Spec, t0: float, t_est: float, device: str, root: str) -> dict:
    """The run metadata of a job that started at time.time() == t0 (the key order is that of the production files)."""
    cuda = device.startswith("cuda")
    return dict(
        spec=spec,
        wall_s=time.time() - t0,
        est_s=t_est,
        device=device,
        host=socket.gethostname(),
        gpu=torch.cuda.get_device_name(0) if cuda else platform.processor(),
        torch=torch.__version__,
        numpy=np.__version__,
        pandas=pd.__version__,
        git=git_commit(root),
        python=platform.python_version(),
        nvidia_smi=nvidia_smi() if cuda else "",
    )
