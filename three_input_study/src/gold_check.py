"""Diagnostic: sensitivity of GOLD-based TV to the x-grid and the (y,z) normal-score grid (CONTRAST, S2, m=4)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from hcr import ROOT, Grid, corr_matrix, load_cfg

cfg = load_cfg()
z = np.load(ROOT + "/results/coeffs.npz")
rows = []
for rho in [0.5, 0.6, 0.7, 0.8, 0.9]:
    R = corr_matrix("CONTRAST", rho, cfg)
    a = z[f"CONTRAST_{rho}_a"][:5, :5, :5]
    sig = R[0, 1:]
    beta = np.linalg.solve(R[1:, 1:], sig)
    s2 = 1 - sig @ beta
    for nx in [192, 400, 1000]:
        G = Grid(nx, 6)
        for ns in [401, 801]:
            s = np.linspace(-8, 8, ns)
            tw = np.full(ns, s[1] - s[0])
            tw[[0, -1]] *= 0.5
            phi = np.exp(-0.5 * s**2) / np.sqrt(2 * np.pi)
            W = np.outer(tw * phi, tw * phi).ravel()
            sy, sz = np.meshgrid(s, s, indexing="ij")
            mu = (beta[0] * sy + beta[1] * sz).ravel()
            g = np.array([(np.exp(-0.5 * (sx - mu) ** 2 / s2 + 0.5 * sx**2) / np.sqrt(s2)) @ W for sx in G.sx])
            du = G.F[:, :5] @ a[:, 0, 0]  # DUDA for flat q
            rows.append(dict(rho_S=rho, x_nodes=nx, yz_nodes=ns, int_gold=G.w @ g, tv_duda_gold=0.5 * G.w @ np.abs(du - g)))
pd.DataFrame(rows).to_csv(ROOT + "/results/gold_check.csv", index=False)
print(pd.DataFrame(rows).round(5).to_string(index=False))
