"""Write the job lists results2/jobs_<exp>.jsonl from config2.yaml.

Every job is fully specified; its id is a hash of the spec. Random streams are derived from
SeedSequence(root_seed, spawn_key=key + (stream,)), where key omits the neuron type and the basis, so
jobs that differ only in those share their random numbers (common random numbers). The gauss,
population and convergence experiments share keys, so their results are paired by seed.

    python scheduler/make_jobs.py [gauss population families lkj convergence network]
"""
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu.job import ROOT, load_cfg  # noqa: E402

CODE = {"gauss": 1, "families": 2, "lkj": 3, "network": 4}


def expected_minutes(spec, cfg, cost):
    """GPU-minutes from the measured cost table (scheduler/cost_model.json, written by profile_cost.py)."""
    cells = cost.get("cells")
    if not cells:
        return 15.0  # no measurement yet
    if spec["exp"] == "network":
        return 1.2 * next(iter(cost["network"].values())) / 60 if cost.get("network") else 15.0
    k, basis, n = spec["k"], spec["basis"], len(spec["cases"])
    fam = spec.get("family", "gauss")
    c = cells.get(f"{fam}_{k}_{basis}") or cells[f"gauss_{k}_{basis}"]
    if f"{fam}_{k}_{basis}" not in cells:  # family cost relative to the Gaussian, measured at k = 8 (triple)
        r = cells.get(f"{fam}_8_triple") or cells.get("t3_8_triple")
        g = cells["gauss_8_triple"]
        c = dict(est_s_per_case=c["est_s_per_case"] * r["est_s_per_case"] / g["est_s_per_case"],
                 case_s=c["case_s"] * r["case_s"] / g["case_s"])
    v = spec.get("variant", "base")
    est = 0.0 if spec["exp"] == "population" else c["est_s_per_case"] * spec["n_est"] / cost["n_est_ref"]
    case = c["case_s"]
    if v == "n2":
        est *= 2
    elif v == "M2":
        case *= 2
    return 0.5 + n * (est + case) / 60  # 0.5 min: process start, belief sampling


def with_id(spec):
    spec = dict(spec)
    spec["id"] = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    return spec


def types_for(k, types):
    return [t for t in types if not (t == "CONTRAST" and k % 2)]


def jobs(exp, cfg):
    e = cfg["experiments"].get(exp, {})
    rho_cases = [{"rho_S": r} for r in cfg["rho_S"]]
    out = []
    if exp in ("gauss", "population", "convergence"):
        variants = [v for v in e.get("variants", ["base"]) if not (exp == "convergence" and v == "base")]
        for k in e["k"]:
            for typ in types_for(k, e["types"]):
                for basis in e["basis"]:
                    for seed in range(e["seeds"]):
                        for v in variants:
                            out.append(dict(exp=exp, k=k, family="gauss", type=typ, basis=basis, seed=seed,
                                            n_est=0 if exp == "population" else e["n_est"], variant=v,
                                            key=[CODE["gauss"], k, seed], cases=rho_cases))
    elif exp == "families":
        for k in e["k"]:
            for fam, typ in e["combos"]:
                if typ == "CONTRAST" and k % 2:
                    continue
                for basis in e["basis"]:
                    for seed in range(e["seeds"]):
                        out.append(dict(exp=exp, k=k, family=fam, type=typ, basis=basis, seed=seed, n_est=e["n_est"],
                                        variant="base", key=[CODE["families"], k, seed], cases=rho_cases))
    elif exp == "lkj":
        nb = math.ceil(e["matrices"] / e["per_job"])
        for k in e["k"]:
            for eta in e["eta"]:
                for basis in e["basis"]:
                    for b in range(nb):
                        idx = range(b * e["per_job"], min((b + 1) * e["per_job"], e["matrices"]))
                        out.append(dict(exp=exp, k=k, family="gauss", type="LKJ", basis=basis, seed=b, eta=eta,
                                        n_est=e["n_est"], variant="base", key=[CODE["lkj"], k, eta, b],
                                        key_lkj=[CODE["lkj"], k, eta], cases=[{"lkj_idx": i} for i in idx]))
    elif exp == "network":
        nc = cfg["network"]
        for k in nc["k"]:
            for rho in nc["rho_S"]:
                for basis in nc["basis"]:
                    for seed in range(nc["seeds"]):
                        out.append(dict(exp=exp, k=k, family="gauss", type="SUM", basis=basis, seed=seed, rho_S=rho,
                                        variant="base", key=[CODE["network"], k, seed]))
    else:
        raise ValueError(exp)
    return [with_id(s) for s in out]


def pilot_jobs(cfg):
    """Throughput pilot: seed index 1000 never occurs in production, results go to results2/jobs/pilot."""
    out = []
    for exp in ("gauss", "population", "families", "lkj"):
        for s in jobs(exp, cfg):
            pick = (exp in ("gauss", "population") and s["seed"] == 0 and s["type"] == "SUM" and s["k"] in (2, 8, 32)
                    or exp == "families" and s["seed"] == 0 and s["k"] == 8 and s["basis"] == "triple"
                    and s["family"] in ("t3", "clayton", "gumbel") and s["type"] in ("SUM", "EXCH")
                    or exp == "lkj" and s["seed"] == 0 and s["k"] == 8 and s["eta"] == 1)
            if pick:
                s = {kk: v for kk, v in s.items() if kk != "id"}
                s["seed"] = 1000
                s["key"] = s["key"][:-1] + [1000]
                if exp == "lkj":
                    s["key_lkj"] = s["key_lkj"] + [1000]
                s["group"] = "pilot"
                out.append(with_id(s))
    return out


def load_cost():
    p = os.path.join(ROOT, "scheduler", "cost_model.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def main(argv):
    cfg = load_cfg()
    cost = load_cost()
    os.makedirs(os.path.join(ROOT, "results2"), exist_ok=True)
    for exp in argv or ["gauss", "population", "families", "lkj", "convergence"]:
        js = pilot_jobs(cfg) if exp == "pilot" else jobs(exp, cfg)
        for s in js:
            s["expected_min"] = round(expected_minutes(s, cfg, cost), 3)
        js.sort(key=lambda s: (s["seed"], -s["expected_min"]))  # seed by seed (PREREGISTRATION 4), longest first
        path = os.path.join(ROOT, "results2", f"jobs_{exp}.jsonl")
        with open(path, "w") as f:
            for s in js:
                f.write(json.dumps(s) + "\n")
        print(f"{exp}: {len(js)} jobs, expected {sum(s['expected_min'] for s in js) / 60:.1f} GPU-h -> {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
