"""Stage 7 (optional, descriptive): time and energy of one forward pass of an HCR layer vs MLP and KAN baselines.

    python -m densprop energy --check                             # CPU self-check of the layers
    python -m densprop energy --dtype float32 --replicate 0       # one measurement run on the GPU (Slurm job)
    python -m densprop energy --summarise                         # results/energy/summary.csv and report.txt
    python -m densprop energy --probe-mlp                         # kernel diagnostic of the MLP (not an energy measurement)

Layers (W = 64 units, k inputs, matched parameter count P = W * n_coef of an HCR neuron, pair basis, m = 4):
  HCR-CD   CONST_DENOM: B product beliefs given by their input coefficient vectors b (B, k, m);
           E_q[A_i] = A_i(b) because A is multilinear in distinct inputs; output A_i / A_0 (B, W, m+1).
  HCR-CAL  EXACT_CAL: each belief given by S = 64 samples; max(N, eps) on the 192-node Gauss-Legendre x-rule,
           normalised per sample, averaged over samples, projected back on f_0..f_m (B, W, m+1).
  MLP      k -> H -> W, GELU; H chosen so that the parameter count matches P.
  KAN      k -> H -> W, B-spline layers (grid 5, order 3, SiLU base), as in efficient-kan; H matches P.
The layers are in densprop.domain.benchmark; the HCR layer is the computation of the study's engine.
Energy: NVML power sampled every 10 ms and integrated over each repeat (primary), the device energy counter as a
check; idle power measured before the grid. Warm-up >= 3 s; 30 repeats of ~2 s each per configuration.
Results never enter the error tables of the statistical study.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import pandas as pd
import torch

from densprop.domain.benchmark import BATCHES, G_NODES, IDLE_S, KS, LAYERS, M_DEG, REPEAT_S, REPEATS, S_SAMPLES, W_UNITS, WARM_S, build
from densprop.ports import EnergyMeter, RecordStore


# ---------------------------------------------------------------- measurement
def measure(
    meter: EnergyMeter, out: RecordStore, dtype_name: str, replicate: int, run_id: str, host: str, commit: str, log: Callable[[str], None] = print
) -> None:
    """One measurement run on the GPU into run_<run_id>.csv (after every configuration) and run_<run_id>.json (at the end)."""
    dtype = getattr(torch, dtype_name)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False  # plain FP32 (and FP64) arithmetic
    meter.start()
    time.sleep(IDLE_S)
    t1 = time.perf_counter()
    _, idle_W = meter.energy(t1 - IDLE_S + 1.0, t1)
    meta = dict(
        meter.info(),
        run_id=run_id,
        dtype=dtype_name,
        replicate=replicate,
        host=host,
        commit=commit,
        idle_W=idle_W,
        W=W_UNITS,
        m=M_DEG,
        S=S_SAMPLES,
        G=G_NODES,
        repeats=REPEATS,
        repeat_s=REPEAT_S,
        warmup_s=WARM_S,
    )
    log(json.dumps(meta))
    rows = []
    for k in KS:
        for B in BATCHES:
            for layer in LAYERS:
                fwd, P, desc = build(layer, k, B, dtype, dev)
                tw, nw = time.perf_counter(), 0
                while time.perf_counter() - tw < WARM_S or nw < 5:
                    fwd()
                    torch.cuda.synchronize()
                    nw += 1
                per_call = (time.perf_counter() - tw) / nw
                n_calls = max(1, round(REPEAT_S / per_call))
                for r in range(REPEATS):
                    torch.cuda.synchronize()
                    e0, t0 = meter.counter(), time.perf_counter()
                    for _ in range(n_calls):
                        fwd()
                    torch.cuda.synchronize()
                    t1, e1 = time.perf_counter(), meter.counter()
                    E, Pw = meter.energy(t0, t1)
                    sm, mem, temp = meter.clocks()
                    rows.append(
                        dict(
                            run_id=run_id,
                            replicate=replicate,
                            dtype=dtype_name,
                            layer=layer,
                            k=k,
                            B=B,
                            params=P,
                            inputs=desc,
                            repeat=r,
                            n_calls=n_calls,
                            seconds=t1 - t0,
                            energy_J=E,
                            energy_counter_J=e1 - e0,
                            mean_power_W=Pw,
                            sm_MHz=sm,
                            mem_MHz=mem,
                            temp_C=temp,
                        )
                    )
                last = rows[-1]
                log(
                    f"{layer:8s} k={k:2d} B={B:5d} P={P:7d} {1e3 * last['seconds'] / n_calls:9.3f} ms/forward "
                    f"{last['energy_J'] / n_calls:9.4f} J/forward {last['mean_power_W']:6.1f} W"
                )
                del fwd
                torch.cuda.empty_cache()
                out.write_csv(f"run_{run_id}.csv", pd.DataFrame(rows))  # after every configuration: a cut slot keeps its data
    meter.stop()
    out.write_json(f"run_{run_id}.json", meta, indent=1)


# ---------------------------------------------------------------- summary
def summarise(out: RecordStore, log: Callable[[str], None] = print) -> None:
    """summary.csv and report.txt from the complete runs."""
    runs = out.names("run_*.csv")
    # directory order, as when the stored report was generated: it fixes the order of the idle powers in the report
    metas = {m["run_id"]: m for m in (out.read_json(n) for n in out.names("run_*.json", sort=False))}
    df = pd.concat([out.read_csv(n, dtype={"run_id": str}) for n in runs], ignore_index=True)
    df = df[df.run_id.isin(metas)]  # only complete runs (meta is written at the end)
    idle = df.run_id.map({k: v["idle_W"] for k, v in metas.items()})
    df["ms"] = 1e3 * df.seconds / df.n_calls
    df["J"] = df.energy_J / df.n_calls
    df["J_net"] = (df.energy_J - idle * df.seconds) / df.n_calls
    df["counter_ratio"] = df.energy_counter_J / df.energy_J
    key = ["dtype", "layer", "k", "B"]
    rep = df.groupby(key + ["run_id"]).agg(ms=("ms", "median"), J=("J", "median")).reset_index()
    spread = rep.groupby(key).agg(ms_lo=("ms", "min"), ms_hi=("ms", "max"), J_lo=("J", "min"), J_hi=("J", "max"), n_runs=("run_id", "nunique"))
    s = (
        df.groupby(key)
        .agg(
            params=("params", "first"),
            ms=("ms", "median"),
            J=("J", "median"),
            J_net=("J_net", "median"),
            power_W=("mean_power_W", "median"),
            sm_MHz=("sm_MHz", "median"),
            counter_ratio=("counter_ratio", "median"),
            n_repeats=("repeat", "size"),
        )
        .join(spread)
        .reset_index()
    )
    s["uJ_per_param"] = 1e6 * s.J / s.params
    s["uJ_per_sample"] = 1e6 * s.J / s.B
    out.write_csv("summary.csv", s)

    m0 = next(iter(metas.values()))
    # the command's name at the time of the stored report, kept so that the report is reproduced byte for byte
    L = [
        "# Stage 7: time and energy of one forward pass (HCR layer vs MLP and KAN)\n",
        f"*Generated by `python -m gpu.energy --summarise` from {len(metas)} runs "
        f"({', '.join(sorted(metas))}); separate from the statistical study.*\n",
        f"GPU {m0['name']}, driver {m0['driver']}, power limit {m0['power_limit_W']:.0f} W, max SM clock "
        f"{m0['max_sm_MHz']} MHz (clocks not locked; median SM clock per row below). Idle power "
        + ", ".join(f"{v['idle_W']:.0f} W" for v in metas.values())
        + f". W = {W_UNITS} units, HCR pair basis m = {M_DEG}, "
        f"EXACT_CAL with S = {S_SAMPLES} samples per belief and {G_NODES} x-nodes; MLP and KAN are k -> H -> W with H "
        f"chosen to match the HCR parameter count. Medians over {REPEATS} repeats x runs; range = min-max of the "
        f"run medians. Energy = NVML power sampled every 10 ms and integrated; `counter` = device energy counter / "
        f"sampled energy. TF32 off.\n",
    ]
    for dt in sorted(s.dtype.unique()):
        L += [
            f"## {dt}\n",
            "| layer | k | B | params | ms / forward (range) | J / forward (range) | J net of idle | "
            "µJ / param | µJ / input | power W | SM MHz | counter |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in s[s.dtype == dt].sort_values(["k", "B", "layer"]).itertuples():
            L.append(
                f"| {r.layer} | {r.k} | {r.B} | {r.params} | {r.ms:.3g} ({r.ms_lo:.3g}–{r.ms_hi:.3g}) | "
                f"{r.J:.3g} ({r.J_lo:.3g}–{r.J_hi:.3g}) | {r.J_net:.3g} | {r.uJ_per_param:.3g} | "
                f"{r.uJ_per_sample:.3g} | {r.power_W:.0f} | {r.sm_MHz:.0f} | {r.counter_ratio:.3f} |"
            )
        L.append("")

    cr, full = df.counter_ratio, df[df.B == max(BATCHES)]
    rel = rep.pivot_table(index=key[:1] + key[1:], columns="run_id", values="J")
    rel = pd.concat([x.dropna(axis=1).pipe(lambda t: t.div(t.mean(axis=1), axis=0)).mean() for _, x in rel.groupby(level="dtype")])
    L += [
        "## Checks\n",
        f"- Counter vs sampled energy over {len(df)} repeats: median {cr.median():.3f}, 1st-99th percentile "
        f"{cr.quantile(0.01):.3f}-{cr.quantile(0.99):.3f}, extremes {cr.min():.3f}-{cr.max():.3f}.",
        f"- Temperature {df.temp_C.min()}-{df.temp_C.max()} C. Repeats at >= 97.5 % of the power limit: "
        f"{(df.mean_power_W >= 0.975 * m0['power_limit_W']).sum()}.",
        f"- Configurations at B = {max(BATCHES)} with SM clock below maximum: "
        + "; ".join(
            f"{dt} {ly} k = {k} (lowest {v.sm_MHz.min():.0f} MHz, median power {v.mean_power_W.median():.0f} W)"
            for (dt, ly, k), v in full[full.sm_MHz < m0["max_sm_MHz"]].groupby(["dtype", "layer", "k"])
        )
        + ".",
        f"- Rows where idle power is over 90 % of the energy (J net of idle < 10 % of J; launch-bound, so J / forward "
        f"mostly measures idle power): {(s.J_net < 0.1 * s.J).sum()} of {len(s)}.",
        f"- Runs used {len({v['uuid'] for v in metas.values()})} physical GPUs. Mean energy of a run relative to the "
        "mean of its dtype's runs, over all configurations:",
        "",
    ]
    L += ["| run | dtype | GPU | host | idle W | relative energy |", "|---|---|---|---|---|---|"]
    for rid, v in sorted(metas.items(), key=lambda kv: kv[1]["uuid"]):
        L.append(f"| {rid} | {v['dtype']} | {v['uuid'][4:12]} | {v['host']} | {v['idle_W']:.1f} | {rel[rid]:.3f} |")
    L.append("")
    out.write_text("report.txt", "\n".join(L) + "\n")
    log(f"summary: {len(s)} rows from {len(metas)} runs -> results/energy/summary.csv, report.txt")


# ---------------------------------------------------------------- MLP kernel probe
PROBE_CALLS = 2000
PROBE_CASES = [
    (dt, tf32, k, B)
    for k, B in ((2, 16384), (2, 4096), (2, 65536), (2, 256), (8, 16384))
    for dt, tf32 in ((torch.float32, False), (torch.float32, True), (torch.float64, False))
]


def probe(dtype, tf32, k, B, log: Callable[[str], None] = print) -> list[dict]:
    """Wall time per forward and CUDA kernel names and times of the MLP: kernel selection vs dispatch-bound."""
    from torch.profiler import ProfilerActivity, profile

    torch.backends.cuda.matmul.allow_tf32 = tf32
    fwd, _, desc = build("MLP", k, B, dtype, "cuda")
    for _ in range(200):
        fwd()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(PROBE_CALLS):
        fwd()
    torch.cuda.synchronize()
    wall_ms = 1e3 * (time.perf_counter() - t0) / PROBE_CALLS
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(PROBE_CALLS):
            fwd()
        torch.cuda.synchronize()
    base = dict(dtype=str(dtype)[6:], tf32=tf32, k=k, B=B, inputs=desc, wall_ms=wall_ms)
    rows = [
        dict(base, kernel=e.key, calls_per_forward=e.count / PROBE_CALLS, kernel_ms=e.self_device_time_total / 1e3 / PROBE_CALLS)
        for e in prof.key_averages()
        if e.self_device_time_total > 0
    ]
    gpu_ms = sum(r["kernel_ms"] for r in rows)
    log(f"\n{base['dtype']} tf32={tf32} k={k} B={B} ({desc}): wall {wall_ms:.4f} ms, kernels {gpu_ms:.4f} ms")
    for r in sorted(rows, key=lambda r: -r["kernel_ms"]):
        log(f"  {r['kernel_ms']:.4f} ms x{r['calls_per_forward']:.0f}  {r['kernel'][:110]}")
    return rows


def probe_mlp(out: RecordStore, run_id: str, log: Callable[[str], None] = print) -> None:
    """Why float64 MLP at k = 2, B = 16384 is faster than float32: probe_mlp_<run_id>.csv."""
    log(f"{torch.cuda.get_device_name(0)} {torch.__version__} {torch.version.cuda}")
    rows = [r for c in PROBE_CASES for r in probe(*c, log=log)]
    out.write_csv(f"probe_mlp_{run_id}.csv", pd.DataFrame(rows))
    log(f"wrote probe_mlp_{run_id}.csv")
