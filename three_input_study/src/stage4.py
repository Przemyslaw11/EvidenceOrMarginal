"""Stage 4: coefficient estimation (n=2e6) and sweeps at 192 (main) and 256 (convergence) nodes."""

import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from hcr import ROOT, estimate_coeffs, load_cfg
from sweep import run_sweep

cfg = load_cfg()
R = ROOT + "/results/"
t0 = time.time()
cpath = R + "coeffs.npz"
if not os.path.exists(cpath):
    co = estimate_coeffs(cfg, cfg["types"], cfg["rho_S"], cfg["n_samples"], cfg["seed"])
    np.savez(cpath, **{f"{t}_{r}_a": v[0] for (t, r), v in co.items()}, **{f"{t}_{r}_se": v[1] for (t, r), v in co.items()})
    print(f"coefficients estimated in {time.time() - t0:.0f}s", flush=True)
z = np.load(cpath)
co = {(t, r): (z[f"{t}_{r}_a"], z[f"{t}_{r}_se"]) for t in cfg["types"] for r in cfg["rho_S"]}
if "--only256" not in sys.argv:
    df, dens = run_sweep(cfg, co, cfg["nodes"], want_dens=True, log=lambda s: print(s, flush=True))
    df.to_csv(R + "results.csv", index=False)
    with open(R + "densities.pkl", "wb") as fh:
        pickle.dump(dens, fh)
    print(f"main sweep done {time.time() - t0:.0f}s, rows={len(df)}", flush=True)
if "--skip256" not in sys.argv:
    df2, _ = run_sweep(cfg, co, cfg["nodes_check"], log=lambda s: print(s, flush=True))
    df2.to_csv(R + "results_n256.csv", index=False)
    print(f"256 sweep done {time.time() - t0:.0f}s", flush=True)
