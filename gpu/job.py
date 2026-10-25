"""Run one fully specified job of the sweep and write one parquet file (atomically) with metadata.

    python -m gpu.job --spec '<json>' --out results2/jobs/<exp>/<id>.parquet [--device cuda]
"""
import argparse
import json
import math
import os
import platform
import socket
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import yaml

from . import copulas as C
from .core import Estimator, chunk_for, population_model
from .propagate import Grid, beta_logq, evaluate_case, sample_beliefs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STREAM = {"est": 0, "mc": 1, "mind": 2, "lkj": 3}


def load_cfg(path=None):
    with open(path or os.path.join(ROOT, "config2.yaml")) as f:
        return yaml.safe_load(f)


def rng_for(cfg, key, stream, *extra):
    ss = np.random.SeedSequence(cfg["root_seed"], spawn_key=tuple(key) + (STREAM[stream],) + tuple(extra))
    return np.random.Generator(np.random.PCG64(ss))


def setup_torch(device):
    torch.set_default_dtype(torch.float64)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def build_cases(spec, cfg, device):
    """[(copula, info dict)] for the cases of a job."""
    k, out = spec["k"], []
    for case in spec["cases"]:
        if "lkj_idx" in case:
            rng = rng_for(cfg, spec["key_lkj"], "lkj", case["lkj_idx"])
            n_rej = 0
            while True:
                R = C.lkj(k + 1, spec["eta"], rng)
                if C.min_eig(R) > C.PD_TOL:
                    break
                n_rej += 1
            off = np.abs(R[1:, 1:][np.triu_indices(k, 1)]) if k > 1 else np.array([np.nan])
            info = dict(lkj_idx=case["lkj_idx"], eta=spec["eta"], lkj_rejected=n_rej, min_eig=C.min_eig(R),
                        mean_abs_ryy=float(off.mean()), mean_abs_rxy=float(np.abs(R[0, 1:]).mean()),
                        rhoS_equiv_yy=float(np.mean(6 / math.pi * np.arcsin(off / 2))),
                        R_json=json.dumps(np.round(R, 12).tolist()))
            out.append((C.Gaussian(R, device), info))
        else:
            cop, info = C.make_copula(spec["family"], spec["type"], k, case["rho_S"], device, cfg["couplings"])
            info = {kk: v for kk, v in info.items()}
            info["rho_S"] = case["rho_S"]
            out.append((cop, info))
    return out


def kinds_for(spec, cases):
    if spec["exp"] == "families":
        return C.KINDS_ALL  # identical draws for every family: common random numbers across the combos
    ks = set()
    for cop, _ in cases:
        ks.update(cop.kinds)
    return tuple(ks)


def estimate(spec, cfg, cases, device, n_est):
    k, m6, basis = spec["k"], cfg["m_max"], spec["basis"]
    ests = [Estimator(k, m6, basis, device) for _ in cases]
    chunk = chunk_for(k, m6, basis, cfg["numerics"]["chunk_est"])
    rng = rng_for(cfg, spec["key"], "est")
    kinds = kinds_for(spec, cases)
    done = 0
    while done < n_est:
        n = min(chunk, n_est - done)
        base = C.base_variates(rng, n, k + 1, kinds)
        for (cop, _), est in zip(cases, ests):
            est.add(cop.uniforms(base))
        done += n
    return [e.model() for e in ests]


def run_job(spec, cfg, device="cpu"):
    t0 = time.time()
    setup_torch(device)
    cfg = json.loads(json.dumps(cfg))
    num = cfg["numerics"]
    v = spec.get("variant", "base")
    n_est = int(spec["n_est"])
    if v == "M2":
        num["M"] *= 2
    elif v == "n2":
        n_est *= 2
    elif v == "s2":
        num["s_nodes"] = 2 * num["s_nodes"] - 1
    elif v == "x2":
        num["x_gl"] *= 2
    for key in ("M", "R", "degrees", "beliefs_used"):
        if key in spec.get("override", {}):
            (num if key in num else cfg)[key] = spec["override"][key]
    k = spec["k"]
    cases = build_cases(spec, cfg, device)
    t_est = time.time()
    if n_est > 0:
        models = estimate(spec, cfg, cases, device, n_est)
    else:  # population coefficients: exact moments of the Gaussian copula
        models = [(population_model(cop.R, cfg["m_max"], spec["basis"], num["gh_nodes"], device), None)
                  for cop, _ in cases]
    t_est = time.time() - t_est
    grid = Grid(num["s_nodes"], num["s_max"], num["x_gl"], cfg["m_max"], device)
    names = cfg.get("beliefs_used", cfg["order"])
    M = num["M"]
    rngs = {nm: rng_for(cfg, spec["key"], "mc", i) for i, nm in enumerate(cfg["order"])}
    raw = sample_beliefs(rngs, M, k, cfg["beliefs"], [n for n in names if n != "S4"])
    beliefs = {}
    for nm, d in raw.items():
        b = dict(Y=torch.as_tensor(d["Y"], device=device))
        if "X" in d:
            b["X"] = torch.as_tensor(d["X"], device=device)
        if "ab" in d:
            lq, lmax = beta_logq(d["ab"])
            b.update(logq=lq, thr=lmax + math.log(cfg["supp_rel"]))
        beliefs[nm] = b
    base_S4 = C.base_variates(rngs["S4"], M, k + 1, kinds_for(spec, cases)) if "S4" in names else None
    rng_md = rng_for(cfg, spec["key"], "mind")
    rows = []
    for ci, ((cop, info), (model, se)) in enumerate(zip(cases, models)):
        t1 = time.time()
        if base_S4 is not None:
            beliefs["S4"] = dict(Y=cop.uniforms(base_S4)[:, 1:].contiguous())
        r = evaluate_case(model, cop, beliefs, grid, cfg, rng_md, cfg["degrees"], device)
        extra = dict(case=ci, **info, rhoS_emp_yy=_emp_yy(model), rhoS_emp_xy=_emp_xy(model),
                     se_a011=_se_yy(se), runtime_case_s=time.time() - t1)
        for x in r:
            x.update(extra)
        rows += r
    df = pd.DataFrame(rows)
    for key in ("exp", "k", "family", "type", "basis", "seed", "variant", "eta"):
        if key in spec:
            df[key] = spec[key]
    df["n_est"] = n_est
    df["job_id"] = spec["id"]
    meta = dict(spec=spec, wall_s=time.time() - t0, est_s=t_est, device=device, host=socket.gethostname(),
                gpu=torch.cuda.get_device_name(0) if device.startswith("cuda") else platform.processor(),
                torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__, git=_git(),
                python=platform.python_version(), nvidia_smi=_smi() if device.startswith("cuda") else "")
    return df, meta


def _emp_yy(model):
    W2 = model.blocks["W2"]
    return float(W2[0, :, 0, 0].mean()) if W2 is not None else math.nan


def _emp_xy(model):
    return float(model.blocks["W1"][1, :, 0].abs().mean())


def _se_yy(se):
    return float(se["W2"][0, :, 0, 0].median()) if se is not None and se["W2"] is not None else math.nan


def _git():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def _smi():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,power.limit",
                               "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def write_atomic(df, meta, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = f"{out}.tmp.{os.getpid()}"
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.Table.from_pandas(df, preserve_index=False)
    t = t.replace_schema_metadata({**(t.schema.metadata or {}), b"hcr_meta": json.dumps(meta, default=str).encode()})
    pq.write_table(t, tmp)
    os.replace(tmp, out)


def read_meta(path):
    import pyarrow.parquet as pq
    return json.loads(pq.read_schema(path).metadata[b"hcr_meta"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    spec = json.loads(a.spec)
    df, meta = run_job(spec, load_cfg(a.config), a.device)
    write_atomic(df, meta, a.out)


if __name__ == "__main__":
    main()
