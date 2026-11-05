"""Stage 7 diagnostic: why float64 MLP at k = 2, B = 16384 is faster than float32.

Per configuration: wall time per forward and CUDA kernel names and times from torch.profiler, so kernel selection and
GPU-bound vs dispatch-bound can be told apart. Not an energy measurement. Writes results/energy/probe_mlp_<jobid>.csv.
"""
import os
import time

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile

from gpu.energy import OUT, build

CALLS = 2000
CASES = [(dt, tf32, k, B) for k, B in ((2, 16384), (2, 4096), (2, 65536), (2, 256), (8, 16384))
         for dt, tf32 in ((torch.float32, False), (torch.float32, True), (torch.float64, False))]


def probe(dtype, tf32, k, B):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    fwd, _, desc = build("MLP", k, B, dtype, "cuda")
    for _ in range(200):
        fwd()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(CALLS):
        fwd()
    torch.cuda.synchronize()
    wall_ms = 1e3 * (time.perf_counter() - t0) / CALLS
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(CALLS):
            fwd()
        torch.cuda.synchronize()
    base = dict(dtype=str(dtype)[6:], tf32=tf32, k=k, B=B, inputs=desc, wall_ms=wall_ms)
    rows = [dict(base, kernel=e.key, calls_per_forward=e.count / CALLS, kernel_ms=e.self_device_time_total / 1e3 / CALLS)
            for e in prof.key_averages() if e.self_device_time_total > 0]
    gpu_ms = sum(r["kernel_ms"] for r in rows)
    print(f"\n{base['dtype']} tf32={tf32} k={k} B={B} ({desc}): wall {wall_ms:.4f} ms, kernels {gpu_ms:.4f} ms", flush=True)
    for r in sorted(rows, key=lambda r: -r["kernel_ms"]):
        print(f"  {r['kernel_ms']:.4f} ms x{r['calls_per_forward']:.0f}  {r['kernel'][:110]}", flush=True)
    return rows


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda, flush=True)
    rows = [r for c in CASES for r in probe(*c)]
    path = os.path.join(OUT, f"probe_mlp_{os.environ.get('SLURM_JOB_ID', 'local')}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print("wrote", path)
