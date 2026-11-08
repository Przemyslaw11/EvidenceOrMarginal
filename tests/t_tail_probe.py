"""Diagnostic for the M7 t-CDF failure (job 3496797: tail rel err 1.0 on the GPU with SciPy 1.13.1). Run as job 3496826,
it gave the same metric on the cluster's CPU as on the GPU; a laptop CPU with SciPy 1.18.1 gave 9e-15. The reference
moves, not the t functions: near F = 1, 1 - F takes only a few discrete values.

Compares the lower tail itself (`_t_lower`, what t_cdf and t_ppf are built on) with scipy's stdtr(nu, -|x|), and repeats
M7's metric for each sign of x, so a real error in the tail can be told from rounding of F = 1 - lower near 1.
"""

import math

import numpy as np
import torch
from scipy.special import stdtr, stdtrit

from gpu import copulas as C

xg = np.concatenate([-np.logspace(-6, 6, 300), np.logspace(-6, 6, 300)])
ug = np.concatenate([np.logspace(-14, np.log10(0.5), 200), 1 - np.logspace(-14, np.log10(0.5), 200)])
pos = xg > 0
devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
out = {}
for nu in (1, 2, 3, 4, 5, 10):
    ax = np.abs(xg)
    true = stdtr(nu, -ax)
    for dev in devs:
        x = torch.as_tensor(xg, dtype=torch.float64, device=dev)
        phi = torch.atan2(torch.full_like(x, math.sqrt(nu)), x.abs())
        low = C._t_lower(phi, nu).cpu().numpy()
        out[nu, dev] = low
        rel = np.abs(low - true) / true
        F = C.t_cdf(x, nu).cpu().numpy()
        ref = stdtr(nu, xg)
        lo = np.minimum(ref, 1 - ref)
        ok = lo > 1e-300
        m = np.where(ok, np.abs(np.minimum(F, 1 - F) - lo) / np.where(ok, lo, 1), 0)
        i = int(m.argmax())
        q = C.t_ppf(torch.as_tensor(ug, dtype=torch.float64, device=dev), nu).cpu().numpy()
        qr = stdtrit(nu, ug)
        eq = np.abs(q - qr) / np.maximum(np.abs(qr), 1.0)
        print(
            f"nu={nu:2d} {dev:4s} lower-tail rel err {rel.max():.1e} (x={ax[rel.argmax()]:.3g}); "
            f"M7 metric x<0 {m[~pos].max():.1e}, x>0 {m[pos].max():.1e}; worst x={xg[i]:.4g} F={F[i]!r} ref={ref[i]!r} "
            f"true tail {true[i]:.3e}; max |F-ref| {np.abs(F - ref).max():.1e}; quantile rel err {eq.max():.1e} "
            f"(u={ug[eq.argmax()]:.3g})",
            flush=True,
        )
    if "cuda" in devs:
        d = np.abs(out[nu, "cpu"] - out[nu, "cuda"]) / true
        print(f"nu={nu:2d} cpu vs cuda lower tail rel diff {d.max():.1e}", flush=True)
