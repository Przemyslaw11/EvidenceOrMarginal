"""Stage 7 (optional, descriptive): time and energy of one forward pass of an HCR layer vs MLP and KAN baselines.

    python -m gpu.energy --check                                  # CPU self-check of the layers
    python -m gpu.energy --dtype float32 --replicate 0            # one measurement run on the GPU (Slurm job)
    python -m gpu.energy --summarise                              # results2/energy/summary.csv and energy2.md

Layers (W = 64 units, k inputs, matched parameter count P = W * n_coef of an HCR neuron, pair basis, m = 4):
  HCR-CD   CONST_DENOM: B product beliefs given by their input coefficient vectors b (B, k, m);
           E_q[A_i] = A_i(b) because A is multilinear in distinct inputs; output A_i / A_0 (B, W, m+1).
  HCR-CAL  EXACT_CAL: each belief given by S = 64 samples; max(N, eps) on the 192-node Gauss-Legendre x-rule,
           normalised per sample, averaged over samples, projected back on f_0..f_m (B, W, m+1).
  MLP      k -> H -> W, GELU; H chosen so that the parameter count matches P.
  KAN      k -> H -> W, B-spline layers (grid 5, order 3, SiLU base), as in efficient-kan; H matches P.
The HCR layer stacks the matrices of W `gpu.core.Model` neurons, so it is the computation of the study's engine.
Energy: NVML power sampled every 10 ms and integrated over each repeat (primary), the device energy counter as a
check; idle power measured before the grid. Warm-up >= 3 s; 30 repeats of ~2 s each per configuration.
Results never enter the error tables of the statistical study.
"""
import argparse
import glob
import json
import math
import os
import socket
import subprocess
import threading
import time

import numpy as np
import torch

from .core import Model, gauss_legendre, input_basis, legendre, pairs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results2", "energy")
W_UNITS, M_DEG, S_SAMPLES, G_NODES, EPS = 64, 4, 64, 192, 0.01
KS, BATCHES, LAYERS = (2, 8, 32), (256, 16384), ("HCR-CD", "HCR-CAL", "MLP", "KAN")
REPEATS, REPEAT_S, WARM_S, IDLE_S = 30, 2.0, 3.0, 30.0
ELEM_BUDGET = 2 ** 27  # elements of the largest intermediate tensor per chunk


# ---------------------------------------------------------------- layers
class HCRLayer:
    """W HCR neurons on the same k inputs (pair basis), stacked from gpu.core.Model matrices."""

    def __init__(self, k, W, m, dtype, device, seed=0):
        g = torch.Generator().manual_seed(seed)
        P = len(pairs(k)[0])
        self.models = []
        for _ in range(W):
            blocks = {"c": torch.cat([torch.ones(1), torch.zeros(m)]).double(),
                      "W1": 0.1 * torch.randn(m + 1, k, m, generator=g, dtype=torch.float64),
                      "W2": 0.1 * torch.randn(m + 1, P, m, m, generator=g, dtype=torch.float64) / math.sqrt(max(P, 1)),
                      "W3": None}
            self.models.append(Model(k, m, "pair", blocks))
        self.k, self.W, self.m = k, W, m
        self.n_params = W * self.models[0].n_coef
        cat = lambda name: torch.cat([getattr(md, name) for md in self.models], -1).to(device, dtype)
        self.Wc, self.W1m = cat("Wc"), cat("W1m")
        self.W2m = cat("W2m") if k >= 2 else None
        self.pa, self.pb = (torch.as_tensor(v, device=device) for v in pairs(k))
        xg, wg = gauss_legendre(G_NODES, device)
        self.wg = wg.to(dtype)
        self.Fx = legendre(xg.to(dtype), m)  # (G, m+1)

    def A(self, F):
        """A_i for all W neurons from feature tables F (n, k, m): (n, W, m+1)."""
        n = F.shape[0]
        A = self.Wc + F.reshape(n, -1) @ self.W1m
        if self.W2m is not None:
            A = A + (F[:, self.pa, :, None] * F[:, self.pb, None, :]).reshape(n, -1) @ self.W2m
        return A.reshape(n, self.W, self.m + 1)

    def const_denom(self, b):
        A = self.A(b)
        return A / A[..., :1]

    def exact_cal(self, Y):
        """Y: (B, S, k) samples of B beliefs -> (B, W, m+1) coefficients of the calibrated output density."""
        B, S, k = Y.shape
        per = S * self.W * max(G_NODES, (len(self.pa) * self.m ** 2) // self.W + 1)
        bc = max(1, ELEM_BUDGET // per)
        out = []
        for st in range(0, B, bc):
            Yc = Y[st: st + bc].reshape(-1, k)
            N = torch.clamp(self.A(input_basis(Yc, self.m)) @ self.Fx.T, min=EPS)  # (n, W, G)
            p = N / (N @ self.wg)[..., None]
            p = p.reshape(-1, S, self.W, G_NODES).mean(1)
            out.append((p * self.wg) @ self.Fx)
        return torch.cat(out)


class KANLayer(torch.nn.Module):
    """B-spline KAN layer (efficient-kan formulation): SiLU base + cubic B-splines on a uniform grid over [-1, 1]."""

    def __init__(self, n_in, n_out, grid=5, order=3):
        super().__init__()
        h = 2.0 / grid
        self.register_buffer("t", torch.arange(-order, grid + order + 1, dtype=torch.float64) * h - 1.0)
        self.order = order
        self.base = torch.nn.Parameter(torch.randn(n_in, n_out) / math.sqrt(n_in))
        self.spline = torch.nn.Parameter(0.1 * torch.randn(n_in * (grid + order), n_out) / math.sqrt(n_in))

    def forward(self, x):
        t, xe = self.t, x[..., None]
        bases = ((xe >= t[:-1]) & (xe < t[1:])).to(x.dtype)
        for p in range(1, self.order + 1):
            bases = ((xe - t[: -(p + 1)]) / (t[p:-1] - t[: -(p + 1)]) * bases[..., :-1]
                     + (t[p + 1:] - xe) / (t[p + 1:] - t[1:-p]) * bases[..., 1:])
        return torch.nn.functional.silu(x) @ self.base + bases.reshape(x.shape[0], -1) @ self.spline


def mlp(k, H, W):
    return torch.nn.Sequential(torch.nn.Linear(k, H), torch.nn.GELU(), torch.nn.Linear(H, W))


def kan(k, H, W):
    return torch.nn.Sequential(KANLayer(k, H), KANLayer(H, W))


def matched_width(kind, k, W, P):
    """Hidden width H of the 2-layer baseline whose parameter count is closest to P."""
    if kind == "MLP":
        return max(1, round((P - W) / (k + 1 + W)))
    return max(1, round(P / (9 * (k + W))))  # KAN, grid 5, order 3: (grid + order + 1) = 9 weights per edge


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def build(layer, k, B, dtype, device, seed=0):
    """Returns (forward callable, parameter count, description of the inputs)."""
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    hcr = HCRLayer(k, W_UNITS, M_DEG, dtype, device, seed)
    P = hcr.n_params
    if layer == "HCR-CD":
        b = (0.2 * torch.randn(B, k, M_DEG, generator=g, dtype=torch.float64)).to(device, dtype)
        return (lambda: hcr.const_denom(b)), P, f"{B} beliefs x {k} coefficient vectors"
    if layer == "HCR-CAL":
        Y = torch.rand(B, S_SAMPLES, k, generator=g, dtype=torch.float64).to(device, dtype)
        return (lambda: hcr.exact_cal(Y)), P, f"{B} beliefs x {S_SAMPLES} samples"
    H = matched_width(layer, k, W_UNITS, P)
    net = (mlp if layer == "MLP" else kan)(k, H, W_UNITS).to(device, dtype).eval()
    x = (2 * torch.rand(B, k, generator=g, dtype=torch.float64) - 1).to(device, dtype)

    def fwd():
        with torch.no_grad():
            return net(x)
    return fwd, n_params(net), f"{B} inputs, hidden width {H}"


# ---------------------------------------------------------------- self-check (CPU)
def check():
    torch.manual_seed(0)
    k, W, B = 3, 4, 5
    lay = HCRLayer(k, W, M_DEG, torch.float64, "cpu")
    b = 0.2 * torch.randn(B, k, M_DEG, dtype=torch.float64)
    cd = lay.const_denom(b)
    for j, md in enumerate(lay.models):
        A = md.A(b)
        assert torch.allclose(cd[:, j], A / A[:, :1], atol=1e-12, rtol=0), "layer differs from gpu.core.Model"
    Y = torch.rand(B, S_SAMPLES, k, dtype=torch.float64)
    cal = lay.exact_cal(Y)
    assert torch.allclose(cal[..., 0], torch.ones(B, W, dtype=torch.float64), atol=1e-10), "EXACT_CAL not normalised"
    # with D > 0 and N >= eps everywhere, EXACT_CAL of a single-sample belief is the neuron's conditional density
    one = lay.exact_cal(Y[:, :1])
    A = lay.A(input_basis(Y[:, 0], M_DEG))
    Nmin = (A @ lay.Fx.T).min()
    if Nmin > EPS:
        assert torch.allclose(one, A / A[..., :1], atol=1e-10), "EXACT_CAL != N/D for a point belief"
    x = torch.rand(7, 2, dtype=torch.float64) * 2 - 1
    lk = KANLayer(2, 3).double()
    bases_sum = None
    with torch.no_grad():  # B-splines of order 3 form a partition of unity inside [-1, 1]
        t, xe = lk.t, x[..., None]
        bs = ((xe >= t[:-1]) & (xe < t[1:])).double()
        for p in range(1, 4):
            bs = (xe - t[: -(p + 1)]) / (t[p:-1] - t[: -(p + 1)]) * bs[..., :-1] + (t[p + 1:] - xe) / (t[p + 1:] - t[1:-p]) * bs[..., 1:]
        bases_sum = bs.sum(-1)
    assert torch.allclose(bases_sum, torch.ones_like(bases_sum), atol=1e-12), "B-spline basis not a partition of unity"
    for kk in KS:
        P = HCRLayer(kk, 1, M_DEG, torch.float64, "cpu").n_params * W_UNITS
        for kind in ("MLP", "KAN"):
            got = n_params((mlp if kind == "MLP" else kan)(kk, matched_width(kind, kk, W_UNITS, P), W_UNITS))
            # one hidden unit changes the count by (k + W) (MLP) or 9 (k + W) (KAN): at k = 2 the KAN is 6.5 % off
            assert abs(got - P) / P < 0.10, f"{kind} k={kk}: {got} vs {P} parameters"
    print("energy self-check PASS")


# ---------------------------------------------------------------- NVML meter
class Meter:
    def __init__(self, torch_device=0):
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
            vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[torch_device]
            self.h = pynvml.nvmlDeviceGetHandleByIndex(int(vis)) if vis.isdigit() else pynvml.nvmlDeviceGetHandleByUUID(vis)
        name = pynvml.nvmlDeviceGetName(self.h)
        name = name.decode() if isinstance(name, bytes) else name
        assert name == torch.cuda.get_device_name(torch_device), f"NVML handle {name} is not the torch device"
        self.samples, self._run = [], False

    def info(self):
        nv, h = self.nv, self.h
        procs = nv.nvmlDeviceGetComputeRunningProcesses(h)
        uuid = nv.nvmlDeviceGetUUID(h)
        bus = nv.nvmlDeviceGetPciInfo(h).busId
        return dict(name=torch.cuda.get_device_name(0), uuid=uuid.decode() if isinstance(uuid, bytes) else uuid,
                    pci_bus=bus.decode() if isinstance(bus, bytes) else bus, nvml_match=self.match,
                    driver=nv.nvmlSystemGetDriverVersion(), power_limit_W=nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1e3,
                    max_sm_MHz=nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_SM),
                    max_mem_MHz=nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_MEM),
                    compute_procs=[int(p.pid) for p in procs], torch=torch.__version__)

    def _loop(self):
        while self._run:
            self.samples.append((time.perf_counter(), self.nv.nvmlDeviceGetPowerUsage(self.h) / 1e3))
            time.sleep(0.01)

    def start(self):
        self._run = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def stop(self):
        self._run = False
        self.th.join()

    def energy(self, t0, t1):
        """Trapezoid integral of the sampled power over [t0, t1] (J), linear interpolation at the ends."""
        s = np.array(self.samples)
        t, p = s[:, 0], s[:, 1]
        tt = np.concatenate([[t0], t[(t > t0) & (t < t1)], [t1]])
        pp = np.interp(tt, t, p)
        trap = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2 renamed it
        return float(trap(pp, tt)), float(pp.mean())

    def counter(self):
        return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) / 1e3  # J

    def clocks(self):
        nv, h = self.nv, self.h
        return (nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM), nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
                nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))


# ---------------------------------------------------------------- measurement
def measure(dtype_name, replicate):
    dtype = getattr(torch, dtype_name)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False  # plain FP32 (and FP64) arithmetic
    meter = Meter()
    run_id = os.environ.get("SLURM_JOB_ID", f"local{int(time.time())}")
    os.makedirs(OUT, exist_ok=True)
    meter.start()
    time.sleep(IDLE_S)
    t1 = time.perf_counter()
    idle_J, idle_W = meter.energy(t1 - IDLE_S + 1.0, t1)
    meta = dict(meter.info(), run_id=run_id, dtype=dtype_name, replicate=replicate, host=socket.gethostname(),
                commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                idle_W=idle_W, W=W_UNITS, m=M_DEG, S=S_SAMPLES, G=G_NODES, repeats=REPEATS, repeat_s=REPEAT_S,
                warmup_s=WARM_S)
    print(json.dumps(meta), flush=True)
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
                    rows.append(dict(run_id=run_id, replicate=replicate, dtype=dtype_name, layer=layer, k=k, B=B,
                                     params=P, inputs=desc, repeat=r, n_calls=n_calls, seconds=t1 - t0,
                                     energy_J=E, energy_counter_J=e1 - e0, mean_power_W=Pw, sm_MHz=sm, mem_MHz=mem,
                                     temp_C=temp))
                last = rows[-1]
                print(f"{layer:8s} k={k:2d} B={B:5d} P={P:7d} {1e3 * last['seconds'] / n_calls:9.3f} ms/forward "
                      f"{last['energy_J'] / n_calls:9.4f} J/forward {last['mean_power_W']:6.1f} W", flush=True)
                del fwd
                torch.cuda.empty_cache()
                import pandas as pd  # written after every configuration so a cut slot keeps its data
                pd.DataFrame(rows).to_csv(os.path.join(OUT, f"run_{run_id}.csv"), index=False)
    meter.stop()
    json.dump(meta, open(os.path.join(OUT, f"run_{run_id}.json"), "w"), indent=1)


# ---------------------------------------------------------------- summary
def summarise():
    import pandas as pd
    runs = sorted(glob.glob(os.path.join(OUT, "run_*.csv")))
    metas = {json.load(open(p))["run_id"]: json.load(open(p)) for p in glob.glob(os.path.join(OUT, "run_*.json"))}
    df = pd.concat([pd.read_csv(p, dtype={"run_id": str}) for p in runs], ignore_index=True)
    df = df[df.run_id.isin(metas)]  # only complete runs (meta is written at the end)
    idle = df.run_id.map({k: v["idle_W"] for k, v in metas.items()})
    df["ms"] = 1e3 * df.seconds / df.n_calls
    df["J"] = df.energy_J / df.n_calls
    df["J_net"] = (df.energy_J - idle * df.seconds) / df.n_calls
    df["counter_ratio"] = df.energy_counter_J / df.energy_J
    key = ["dtype", "layer", "k", "B"]
    rep = df.groupby(key + ["run_id"]).agg(ms=("ms", "median"), J=("J", "median")).reset_index()
    spread = rep.groupby(key).agg(ms_lo=("ms", "min"), ms_hi=("ms", "max"), J_lo=("J", "min"), J_hi=("J", "max"),
                                  n_runs=("run_id", "nunique"))
    s = df.groupby(key).agg(params=("params", "first"), ms=("ms", "median"), J=("J", "median"),
                            J_net=("J_net", "median"), power_W=("mean_power_W", "median"),
                            sm_MHz=("sm_MHz", "median"), counter_ratio=("counter_ratio", "median"),
                            n_repeats=("repeat", "size")).join(spread).reset_index()
    s["uJ_per_param"] = 1e6 * s.J / s.params
    s["uJ_per_sample"] = 1e6 * s.J / s.B
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    m0 = next(iter(metas.values()))
    L = ["# Stage 7: time and energy of one forward pass (HCR layer vs MLP and KAN)\n",
         f"*Generated by `python -m gpu.energy --summarise` from {len(metas)} runs "
         f"({', '.join(sorted(metas))}); separate from the statistical study.*\n",
         f"GPU {m0['name']}, driver {m0['driver']}, power limit {m0['power_limit_W']:.0f} W, max SM clock "
         f"{m0['max_sm_MHz']} MHz (clocks not locked; median SM clock per row below). Idle power "
         + ", ".join(f"{v['idle_W']:.0f} W" for v in metas.values()) + f". W = {W_UNITS} units, HCR pair basis m = {M_DEG}, "
         f"EXACT_CAL with S = {S_SAMPLES} samples per belief and {G_NODES} x-nodes; MLP and KAN are k -> H -> W with H "
         f"chosen to match the HCR parameter count. Medians over {REPEATS} repeats x runs; range = min-max of the "
         f"run medians. Energy = NVML power sampled every 10 ms and integrated; `counter` = device energy counter / "
         f"sampled energy. TF32 off.\n"]
    for dt in sorted(s.dtype.unique()):
        L += [f"## {dt}\n", "| layer | k | B | params | ms / forward (range) | J / forward (range) | J net of idle | "
              "µJ / param | µJ / input | power W | SM MHz | counter |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in s[s.dtype == dt].sort_values(["k", "B", "layer"]).itertuples():
            L.append(f"| {r.layer} | {r.k} | {r.B} | {r.params} | {r.ms:.3g} ({r.ms_lo:.3g}–{r.ms_hi:.3g}) | "
                     f"{r.J:.3g} ({r.J_lo:.3g}–{r.J_hi:.3g}) | {r.J_net:.3g} | {r.uJ_per_param:.3g} | "
                     f"{r.uJ_per_sample:.3g} | {r.power_W:.0f} | {r.sm_MHz:.0f} | {r.counter_ratio:.3f} |")
        L.append("")
    open(os.path.join(ROOT, "energy2.md"), "w").write("\n".join(L) + "\n")
    print(f"summary: {len(s)} rows from {len(metas)} runs -> results2/energy/summary.csv, energy2.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--replicate", type=int, default=0)
    a = ap.parse_args()
    if a.check:
        check()
    elif a.summarise:
        summarise()
    else:
        measure(a.dtype, a.replicate)
