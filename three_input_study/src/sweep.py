"""Stage 4: full sweep -> long-format results table."""

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hcr import Grid, corr_matrix, ctrue_grid, evaluate, gold_for, scenario_specs


def run_sweep(cfg, coeffs, nodes, types=None, rhos=None, ms=None, want_dens=False, log=print):
    types = types or cfg["types"]
    rhos = rhos or cfg["rho_S"]
    ms = ms or cfg["degrees"]
    G = Grid(nodes, cfg["mmax"])
    specs = scenario_specs(cfg)
    rows, dens_all = [], {}
    for typ in types:
        for rho in rhos:
            t0 = time.time()
            a6 = coeffs[(typ, rho)][0]
            R = corr_matrix(typ, rho, cfg)
            Ct = ctrue_grid(G, R)
            cache = {}

            def gold_fn(sp):
                need = {k: v for k, v in sp.items() if k != "S1" and k not in cache}
                if need:
                    cache.update(gold_for(G, R, Ct, need, cfg))
                out = {k: cache[k] for k in sp if k != "S1"}
                if "S1" in sp:
                    out.update(gold_for(G, R, Ct, {"S1": sp["S1"]}, cfg))
                return out

            for m in ms:
                a = a6[: m + 1, : m + 1, : m + 1]
                wd = want_dens and typ == "CONTRAST" and m == 4
                r, d = evaluate(G, a, m, R, specs, cfg, cfg["eps"], gold_fn, want_dens=wd)
                for x in r:
                    x.update(type=typ, rho_S=rho, nodes=nodes)
                rows += r
                if wd:
                    dens_all.update({(rho,) + k: v for k, v in d.items()})
            log(f"  [{nodes}] {typ} rho={rho} done in {time.time() - t0:.1f}s")
    df = pd.DataFrame(rows)
    first = [
        "type",
        "rho_S",
        "m",
        "scenario",
        "eps",
        "defined",
        "quadrature_limited",
        "min_D",
        "frac_D_neg",
        "frac_joint_neg",
        "d_denom",
        "d_neg",
        "d_model_uncal",
        "d_model_cal",
        "d_total",
        "d_dc",
    ]
    df = df[first + [c for c in df.columns if c not in first]]
    return df, dens_all
