"""EnergyMeter on NVML: power sampled every 10 ms in a thread and integrated, the device energy counter, clocks."""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import torch


class Meter:
    """EnergyMeter on NVML for the CUDA device that runs this process; `visible` is CUDA_VISIBLE_DEVICES."""

    def __init__(self, torch_device: int = 0, visible: str = "0"):
        import pynvml

        self.nv = pynvml
        pynvml.nvmlInit()
        torch.zeros(1, device=f"cuda:{torch_device}")  # create the context so this process appears in NVML
        self.h, self.match = None, "pid"
        for i in range(pynvml.nvmlDeviceGetCount()):  # the device that runs this process (identical A100s otherwise)
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            if any(p.pid == os.getpid() for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h)):
                self.h = h
                break
        if self.h is None:
            self.match = "CUDA_VISIBLE_DEVICES"
            vis = visible.split(",")[torch_device]
            self.h = pynvml.nvmlDeviceGetHandleByIndex(int(vis)) if vis.isdigit() else pynvml.nvmlDeviceGetHandleByUUID(vis)
        name = pynvml.nvmlDeviceGetName(self.h)
        name = name.decode() if isinstance(name, bytes) else name
        assert name == torch.cuda.get_device_name(torch_device), f"NVML handle {name} is not the torch device"
        self.samples, self._run = [], False

    def info(self) -> dict:
        nv, h = self.nv, self.h
        procs = nv.nvmlDeviceGetComputeRunningProcesses(h)
        uuid = nv.nvmlDeviceGetUUID(h)
        bus = nv.nvmlDeviceGetPciInfo(h).busId
        return dict(
            name=torch.cuda.get_device_name(0),
            uuid=uuid.decode() if isinstance(uuid, bytes) else uuid,
            pci_bus=bus.decode() if isinstance(bus, bytes) else bus,
            nvml_match=self.match,
            driver=nv.nvmlSystemGetDriverVersion(),
            power_limit_W=nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1e3,
            max_sm_MHz=nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_SM),
            max_mem_MHz=nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_MEM),
            compute_procs=[int(p.pid) for p in procs],
            torch=torch.__version__,
        )

    def _loop(self):
        while self._run:
            self.samples.append((time.perf_counter(), self.nv.nvmlDeviceGetPowerUsage(self.h) / 1e3))
            time.sleep(0.01)

    def start(self) -> None:
        self._run = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def stop(self) -> None:
        self._run = False
        self.th.join()

    def energy(self, t0: float, t1: float) -> tuple[float, float]:
        """Trapezoid integral of the sampled power over [t0, t1] (J), linear interpolation at the ends."""
        s = np.array(self.samples)
        t, p = s[:, 0], s[:, 1]
        tt = np.concatenate([[t0], t[(t > t0) & (t < t1)], [t1]])
        pp = np.interp(tt, t, p)
        trap = getattr(np, "trapezoid", None) or np.trapz  # noqa: NPY201  numpy >= 2 renamed it; the cluster has 1.26
        return float(trap(pp, tt)), float(pp.mean())

    def counter(self) -> float:
        return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) / 1e3  # J

    def clocks(self) -> tuple[int, int, int]:
        nv, h = self.nv, self.h
        return (
            nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
            nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
            nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
        )
