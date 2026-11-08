"""M14: validation of the Stage 4 chain (called from tests/run_tests.py network)."""

import math

import numpy as np
import torch

from gpu import copulas as C
from gpu.core import DT, input_basis, legendre
from gpu.job import load_cfg
from gpu.propagate import Grid, beta_logq, evaluate_case, sample_beliefs
from network.chain import LayerGrid, Teacher, cd_layer, estimate_student, gold_layers, kernels, poly_on, propagate_chain
from network.train import Data, Params, loss_val, make_loss_e2e, make_loss_nll


def _setup(dev, k=4, rho=0.2, L=3, n_est=200000, M=2**15):
    cfg = load_cfg()
    cfg = dict(cfg, degrees=[4], numerics=dict(cfg["numerics"], M=M, R=8, chunk_mc=2048, mind=dict(cfg["numerics"]["mind"], starts=500, steps=100)))
    t = Teacher(k, rho, 0.6, 0.5)
    rng = np.random.default_rng(1)
    models = estimate_student(t.sample(rng, n_est, L), L, k, 4, "pair", dev)
    rngs = {nm: np.random.default_rng([2, i]) for i, nm in enumerate(cfg["order"])}
    raw = sample_beliefs(rngs, M, k, cfg["beliefs"], ["S2", "S3c"])
    bel = {}
    for nm, d in raw.items():
        b = dict(Y=torch.as_tensor(d["Y"], device=dev))
        if "X" in d:
            b["X"] = torch.as_tensor(d["X"], device=dev)
        if "ab" in d:
            lq, lmax = beta_logq(d["ab"])
            b.update(logq=lq, thr=lmax + math.log(cfg["supp_rel"]))
        bel[nm] = b
    Ws = torch.as_tensor(np.random.default_rng(3).random((M, L)), device=dev)
    return cfg, t, models, bel, Ws


def m14(dev):
    det, ok = [], True
    cfg, t, models, bel, Ws = _setup(dev)
    sg = Grid(801, 8.0, 192, 4, dev)
    lg = LayerGrid(192, 4, dev)
    # (a) chain of length 1 == single-neuron engine
    rc, r1 = propagate_chain(models[:1], t, bel, Ws[:, :1], sg, lg, cfg, np.random.default_rng(0), dev)
    single = evaluate_case(models[0], C.Gaussian(t.R1, dev), bel, sg, cfg, np.random.default_rng(0), [4], dev)
    diff = 0.0
    for r in rc:
        s = [x for x in single if x["belief"] == r["belief"] and x["eps"] == r["eps"]][0]
        for key in ("d_total", "d_model_cal", "d_dc"):
            diff = max(diff, abs(r[key] - s[key]))
    ok &= diff < 1e-12
    det.append(f"(a) chain L=1 vs single-neuron engine: max |TV diff| {diff:.1e}")
    # (b) deterministic layer >= 2 vs Monte Carlo with weights rho_1(h)
    dens = {}
    evaluate_case(models[0], C.Gaussian(t.R1, dev), {"S2": bel["S2"]}, sg, cfg, np.random.default_rng(0), [4], dev, dens)
    c1 = dens[("S2", 4)]["cd_c"]
    xs = torch.linspace(0, 1, 2001, dtype=DT, device=dev)
    assert float(poly_on(c1, xs).min()) > 0, "test needs a positive layer-1 density"
    mod2 = models[1]
    cd_k = cd_layer(mod2, c1)
    K = kernels(mod2, lg, sg, [0.01], cfg["D_floor"])
    eu_k = (poly_on(c1, lg.u) * lg.wt) @ K["K_eu"].T
    U = torch.as_tensor(np.random.default_rng(4).random((2**20, 2)), device=dev)
    w = poly_on(c1, U[:, 0])
    A = mod2.A(input_basis(U, 4))
    blocks = torch.arange(U.shape[0], device=dev) % 16
    cd_r = torch.stack([(w[blocks == b] @ A[blocks == b]) / (w[blocks == b] @ A[blocks == b, 0]) for b in range(16)])
    eu_r = torch.stack([(w[blocks == b] @ (A[blocks == b] / A[blocks == b, :1])) / w[blocks == b].sum() for b in range(16)])
    z_cd = float(((cd_r.mean(0) - cd_k) / (cd_r.std(0) / 4 + 1e-15)).abs()[1:].max())
    z_eu = float(((eu_r.mean(0) - eu_k) / (eu_r.std(0) / 4 + 1e-15)).abs()[1:].max())
    ok &= z_cd < 4.5 and z_eu < 4.5
    det.append(f"(b) layer-2 closed form / kernel vs MC (2^20 samples): max |z| CONST_DENOM {z_cd:.2f}, EXACT_UNCAL {z_eu:.2f}")
    # (c) autograd vs central finite differences, every loss
    data = Data(t.sample(np.random.default_rng(5), 20000, 3), 3, 4, dev)
    Fb = {"S2": input_basis(torch.as_tensor(np.random.default_rng(6).random((4096, 4)), device=dev), 4)}
    tg = {
        "S2": legendre(
            torch.as_tensor(t.h_given_y(np.random.default_rng(7), np.random.default_rng(6).random((4096, 4)), 3), dtype=DT, device=dev), 4
        ).mean(0)
    }
    worst = 0.0
    for name, lf, rows in (("val", loss_val, "val"), ("nll", make_loss_nll(0.01, dev), "all"), ("e2e", make_loss_e2e(3, Fb, tg), "all")):
        P = Params(models, rows=rows)
        g = torch.Generator(device="cpu").manual_seed(0)
        V = [torch.randn(p.shape, generator=g, dtype=DT).to(dev) for p in P.tensors()]
        loss = lf(P.models(), data, {})
        grads = torch.autograd.grad(loss, P.tensors(), allow_unused=True)
        ad = sum(float((gr * v).sum()) for gr, v in zip(grads, V) if gr is not None)
        h = 1e-8  # FD step: the nll has max(N, eps) kinks; h = 1e-6 straddles some of them
        with torch.no_grad():
            for p, v in zip(P.tensors(), V):
                p.add_(h * v)
            lp = float(lf(P.models(), data, {}))
            for p, v in zip(P.tensors(), V):
                p.sub_(2 * h * v)
            lm = float(lf(P.models(), data, {}))
        fd = (lp - lm) / (2 * h)
        rel = abs(fd - ad) / max(abs(ad), 1e-300)
        worst = max(worst, rel)
        det.append(f"(c) {name}: d/dv autograd {ad:.6e} vs FD {fd:.6e} (rel {rel:.1e})")
    ok &= worst < 1e-6
    # (d) value loss: gradient w.r.t. rows i >= 2 is identically zero (so the all-coefficients variant equals T-val)
    P = Params(models, rows="all")
    grads = torch.autograd.grad(loss_val(P.models(), data, {}), P.tensors(), allow_unused=True)
    g2 = max(float(gr[2:].abs().max()) for gr in grads if gr is not None and gr.shape[0] > 2)
    ok &= g2 == 0.0
    det.append(f"(d) value loss: max |grad| over rows i>=2 = {g2:.1e}")
    # (e) Markov property of the teacher: partial correlations in normal scores
    L = 4
    s = t.sample(np.random.default_rng(8), 10**6, L)
    worst_pc = 0.0
    for l in range(2, L):
        X = np.column_stack([s["sh"][:, l - 1], s["sw"][:, l]])
        res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
        r_h = res(s["sh"][:, l])
        for other in (s["sy"][:, 0], s["sh"][:, l - 2], s["sw"][:, l - 1]):
            pc = np.corrcoef(r_h, res(other))[0, 1]
            worst_pc = max(worst_pc, abs(pc) * math.sqrt(10**6))
    ok &= worst_pc < 4.5
    det.append(f"(e) teacher Markov property: max |partial corr| x sqrt(n) = {worst_pc:.2f}")
    # (f) the product belief is exact: layer-wise propagation of the true conditionals equals the direct GOLD
    Y = torch.as_tensor(np.random.default_rng(10).random((2**20, t.k)), device=dev)
    W4 = torch.as_tensor(np.random.default_rng(9).random((Y.shape[0], L)), device=dev)
    gold = gold_layers(t, Y, W4, sg, 8)
    p = gold[0][-1]
    v = math.sqrt(t.e**2 + t.b**2)
    Kt = torch.exp(-0.5 * ((sg.s[:, None] - t.a * sg.s[None, :]) / v) ** 2) / (v * math.sqrt(2 * math.pi))
    worst_tv = 0.0
    for l in range(1, L):
        p = Kt @ (p * sg.w)
        worst_tv = max(worst_tv, float(sg.tv(p, gold[l][-1])))
    ok &= worst_tv < 1e-3
    det.append(f"(f) layer-wise true conditionals vs direct GOLD: max TV {worst_tv:.1e} (L=4)")
    return bool(ok), "; ".join(det)
