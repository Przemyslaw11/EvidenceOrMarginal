"""Stage 4 job: one (k, rho_S, basis, seed) configuration of the HCR chain; called by gpu.job for exp == 'network'.

Runs: the directly estimated chain (L = 4, every layer reported, which covers L = 1..4 because propagation is causal),
then training from the direct estimates and from a flat-D control initialisation:
    local losses (val, nll) x (adam, lbfgs): trained once on all four neurons (local losses do not depend on L);
    end-to-end loss (e2e) x (adam, lbfgs) for every L in 1..4, trained on the beliefs S2, S3c, S4.
Every run is evaluated on all beliefs (training beliefs and the held-out beliefs S3a, S3b, S3d).
"""
import math
import time

import numpy as np
import pandas as pd
import torch

from gpu.core import DT, input_basis, legendre
from gpu.propagate import Grid, beta_logq, sample_beliefs
from network.chain import LayerGrid, Teacher, estimate_student, propagate_chain
from network.train import Data, Params, loss_val, make_loss_e2e, make_loss_nll, train


def run_network_job(spec, cfg, device, rng_for):
    nc = cfg["network"]
    k, rho, basis, m, Lmax = spec["k"], spec["rho_S"], spec["basis"], nc["m"], max(nc["L"])
    key = spec["key"]
    cfg = dict(cfg, degrees=[m], numerics=dict(cfg["numerics"], M=nc["M"]))
    teacher = Teacher(k, rho, nc["a_prev"], nc["b_side"], cfg["couplings"])
    rng = lambda s, *e: rng_for(cfg, key, s, *e)
    est = teacher.sample(rng("est"), nc["n_est"], Lmax)
    direct = estimate_student(est, Lmax, k, m, basis, device)
    sgrid = Grid(cfg["numerics"]["s_nodes"], cfg["numerics"]["s_max"], cfg["numerics"]["x_gl"], m, device)
    lg = LayerGrid(cfg["numerics"]["x_gl"], m, device)
    # beliefs on the inputs of neuron 1 (evaluation) and uniform side inputs for GOLD
    names = list(dict.fromkeys(nc["beliefs"] + nc["heldout_beliefs"]))
    M = nc["M"]
    rngs = {nm: rng("mc", i) for i, nm in enumerate(cfg["order"])}
    raw = sample_beliefs(rngs, M, k, cfg["beliefs"], [n for n in names if n != "S4"])
    beliefs = {}
    for nm in names:
        if nm == "S4":
            beliefs[nm] = dict(Y=torch.as_tensor(teacher.sample(rngs["S4"], M, 1)["y"], dtype=DT, device=device))
            continue
        d = raw[nm]
        b = dict(Y=torch.as_tensor(d["Y"], device=device))
        if "X" in d:
            b["X"] = torch.as_tensor(d["X"], device=device)
        if "ab" in d:
            lq, lmax = beta_logq(d["ab"])
            b.update(logq=lq, thr=lmax + math.log(cfg["supp_rel"]))
        beliefs[nm] = b
    Ws = torch.as_tensor(rng("mc", 99).random((M, Lmax)), dtype=DT, device=device)
    rng_md = rng("mind")

    rows = []

    def evaluate(models, tag):
        t = time.time()
        rc, r1 = propagate_chain(models, teacher, beliefs, Ws, sgrid, lg, cfg, rng_md, device)
        for r in rc:
            rows.append(dict(r, table="chain", **tag, eval_s=time.time() - t))
        for r in r1:
            rows.append(dict(r, table="neuron1", **tag))

    evaluate(direct, dict(run="direct", loss="none", opt="none", init="direct", L_train=0))
    # training data (fresh teacher samples) and e2e targets
    splits = {s: Data(teacher.sample(rng(f"{s}"), nc[f"n_{s}"], Lmax), Lmax, m, device) for s in ("train", "val", "test")}
    loss_nll = make_loss_nll(nc["eps"], device)
    e2e = {}
    for s in ("train", "val", "test"):
        bel_F, tgt = {}, {}
        for i, nm in enumerate(nc["beliefs"]):
            r = rng(f"{s}", 50 + i)
            if nm == "S4":
                Y = teacher.sample(r, 2 ** 16, 1)["y"]
            elif nm == "S2":
                Y = r.random((2 ** 16, k))
            else:
                ab = np.array([cfg["beliefs"][nm][j % len(cfg["beliefs"][nm])] for j in range(k)], float)
                Y = r.beta(ab[:, 0], ab[:, 1], size=(2 ** 16, k))
            bel_F[nm] = input_basis(torch.as_tensor(Y, dtype=DT, device=device), m)
            tgt[nm] = {}
            for L in nc["L"]:
                h = torch.as_tensor(teacher.h_given_y(r, Y, L), dtype=DT, device=device)
                tgt[nm][L] = legendre(h, m).mean(0)
        e2e[s] = (bel_F, tgt)
    inits = dict(direct=direct, flat=[mod.with_flat_D() for mod in direct])
    ocfg = dict(adam=nc["adam"], lbfgs=nc["lbfgs"])
    for init, base in inits.items():
        if init == "flat":
            evaluate(base, dict(run="flat_init", loss="none", opt="none", init="flat", L_train=0))
        for loss_name in ("val", "nll"):
            for opt in ("adam", "lbfgs"):
                P = Params(base, rows="val" if loss_name == "val" else "all")
                rec = train(P, loss_val if loss_name == "val" else loss_nll, splits, opt, ocfg[opt])
                evaluate(P.models(), dict(run=f"{loss_name}_{opt}_{init}", loss=loss_name, opt=opt, init=init, L_train=Lmax, **rec))
        for L in nc["L"]:
            for opt in ("adam", "lbfgs"):
                P = Params(base, rows="all")
                lf = {s: make_loss_e2e(L, e2e[s][0], {b: t[L] for b, t in e2e[s][1].items()}) for s in splits}
                rec = train(P, lambda mods, data, st: lf[data_split(data, splits)](mods, data, st), splits, opt, ocfg[opt])
                evaluate(P.models()[:L] + base[L:], dict(run=f"e2e{L}_{opt}_{init}", loss="e2e", opt=opt, init=init,
                                                         L_train=L, **rec))
    df = pd.DataFrame(rows)
    df["rho_S"] = rho
    return df


def data_split(data, splits):
    for s, d in splits.items():
        if d is data:
            return s
    raise KeyError("unknown split")
