"""The study's analysis: thresholds, convergence (M13 / H4b) and the pre-registered hypotheses, from the merged results into
the summaries (results/summary/)."""

from __future__ import annotations

from collections.abc import Callable

from densprop.domain import hypotheses as H
from densprop.domain.convergence import COLS as CONV_COLS
from densprop.domain.convergence import m13
from densprop.domain.thresholds import COLS, curves, thresholds
from densprop.ports import RecordStore, ResultStore


def run_thresholds(store: ResultStore, summary: RecordStore, exps: list[str], log: Callable[[str], None] = print) -> None:
    """curves_<exp>.csv.gz and thresholds_<exp>.csv: seed-median curves and thresholds with bootstrap CIs."""
    for exp in exps:
        df = store.load(exp, columns=None)
        if df is None:
            log(f"{exp}: no merged results")
            continue
        df = df[[c for c in COLS if c in df.columns]]
        if exp == "lkj":
            continue  # matrices are not on a rho grid: see the hypotheses (H3)
        summary.write_csv(f"curves_{exp}.csv.gz", curves(df))
        summary.write_csv(f"thresholds_{exp}.csv", thresholds(df))
        log(f"{exp}: curves and thresholds written")


def convergence(store: ResultStore, summary: RecordStore) -> tuple[bool, str]:
    """Test M13: convergence.csv and convergence_violations.csv; (pass, detail)."""
    conv = store.load("convergence", columns=CONV_COLS)
    base = store.load("gauss", columns=[c for c in CONV_COLS if c != "variant"])
    ok, det, tab, vt = m13(conv, base)
    if tab is not None:
        summary.write_csv("convergence.csv", tab)
        summary.write_csv("convergence_violations.csv", vt)
    return ok, det


def run_hypotheses(store: ResultStore, summary: RecordStore, log: Callable[[str], None] = print) -> dict:
    """hypotheses.json: a verdict per pre-registered hypothesis with its evaluable cells and evidence."""
    res = {}
    pop, gauss, fam, lkj = store.load("population"), store.load("gauss"), store.load("families"), store.load("lkj")
    cp, cg = "curves_population.csv.gz", "curves_gauss.csv.gz"
    curves_pop = summary.read_csv(cp) if summary.exists(cp) else None
    curves_g = summary.read_csv(cg) if summary.exists(cg) else None
    if pop is not None:
        res.update(H.h1a_h2a_h2b_pop(summary.read_csv("thresholds_population.csv")))
        res["H1c"] = H.h1c(curves_pop)
    if gauss is not None:
        res.update(H.h1b_h2b_gauss(gauss))
        res["H2c"] = H.h2c(curves_g)
        res["H4a"] = H.h4a(curves_g)
        if fam is not None:
            res.update(H.h3(gauss, fam, lkj, curves_g))
    if summary.exists("convergence.csv"):
        ok, det = convergence(store, summary)
        res["H4b"] = dict(verdict="supported" if ok else "not supported", detail=det)
    net = store.load("network")
    if net is not None:
        res.update(H.h5(net))
    summary.write_json("hypotheses.json", res, indent=1, default=float)
    for k in sorted(res):
        log(f"{k}: {res[k]['verdict']}")
    return res
