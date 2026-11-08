"""Numerical check of the gap proposition on one small case (CPU, seconds): identity, tilted belief, TV bound, flat and data beliefs.

.venv/bin/python paper/check_identity.py        # prints the residuals; exits non-zero if a check fails
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from densprop.domain import copulas as C
from densprop.domain.neuron import DT, input_basis, legendre, population_model

M, m, rho = 2**20, 4, 0.3
R, _ = C.coupled_R("SUM", 2, rho)
rng = np.random.default_rng(0)
xg, wg = (torch.tensor(v, dtype=DT) for v in np.polynomial.legendre.leggauss(400))
xg, wg = (xg + 1) / 2, wg / 2
Fx = legendre(xg, m)


def run(basis, Y):
    mod = population_model(R, m, basis, nodes=64)
    A = mod.A(input_basis(Y, m))
    D = A[:, 0]
    assert float(D.min()) > 0, "this case must have D > 0 on the belief"
    N, ratio = A @ Fx.T, (A / D[:, None]) @ Fx.T  # (M, x nodes)
    cd, eu = N.mean(0) / D.mean(), ratio.mean(0)
    cov = (ratio * D[:, None]).mean(0) - ratio.mean(0) * D.mean()
    tilt = ((D / D.sum())[:, None] * ratio).sum(0)
    tv = 0.5 * float((wg * (cd - eu).abs()).sum())
    l1 = float(((ratio.abs() * wg).sum(1)).max())
    return dict(
        mod=mod,
        cd=cd,
        D=D,
        identity=float((cd - eu - cov / D.mean()).abs().max()),
        tilted=float((cd - tilt).abs().max()),
        tv=tv,
        bound=l1 * 0.5 * float(D.std()) / float(D.mean()),
    )


fails = []
for basis in ("pair", "triple"):
    r = run(basis, torch.tensor(rng.random((M, 2))))
    print(
        f"{basis}, flat belief: identity residual {r['identity']:.1e}, tilted-belief residual {r['tilted']:.1e}, "
        f"TV {r['tv']:.4f} <= bound {r['bound']:.4f}; E_q[D] - 1 = {float(r['D'].mean()) - 1:+.1e}"
    )
    fails += [r["identity"] > 1e-12, r["tilted"] > 1e-12, r["tv"] > r["bound"]]
    c = r["mod"].blocks["c"]  # the x-marginal of the model is uniform here, so CONST_DENOM must be 1 up to the MC error
    gap = float((r["cd"] - 1).abs().max())
    print(f"  flat belief, CONST_DENOM - x-marginal: {gap:.1e} (MC level)")
    fails.append(gap > 0.02)
Z = rng.standard_normal((M, 2)) @ np.linalg.cholesky(R[1:, 1:]).T
from scipy.special import ndtr  # noqa: E402

for basis in ("pair", "triple"):
    r = run(basis, torch.tensor(ndtr(Z)))
    chi2 = r["mod"].chi_D() ** 2
    gap = float((r["cd"] - 1).abs().max())
    print(
        f"{basis}, data belief: identity residual {r['identity']:.1e}; E_q[D] {float(r['D'].mean()):.4f} vs 1 + chi_D^2 "
        f"{1 + chi2:.4f}; max |CONST_DENOM - 1| = {gap:.2e}"
    )
    fails += [r["identity"] > 1e-12, abs(float(r["D"].mean()) - 1 - chi2) > 5e-3]
    fails.append(gap > 0.02 if basis == "pair" else gap < 0.02)  # exact for pair; a real deviation for triple
print("FAIL" if any(fails) else "PASS")
sys.exit(int(any(fails)))
