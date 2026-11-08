"""All numbers and data tables of the paper, generated from the study's result files.

    .venv/bin/python paper/make_numbers.py

Reads results/summary/, results/merged/*_status.json and *_meta.csv, results/energy/, results/run_record.json, three_input_study/results/report.txt,
tests/test_results.json, config.yaml, paper/posthoc.json, paper/theory.json, paper/semantics.json, paper/regression.json and paper/pilot.json. Writes generated/numbers.tex (the macro table behind
\\V{key}), generated/numbers.json (the same table, read by check_numbers.py) and generated/tab_*.tex. Nothing is typed by hand.
"""
import glob
import json
import math
import os
import re

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
S = os.path.join(ROOT, "results", "summary")
GEN = os.path.join(HERE, "generated")
KS = (2, 4, 8, 16, 32)
TYPES = ("SUM", "CONTRAST")
BASES = ("pair", "triple")
DASH = "---"

V = {}


def put(key, tex):
    assert key not in V, key
    V[key] = str(tex)


def f(v, d=3):
    return DASH if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def pc(v, d=0):
    return f"{100 * v:.{d}f}\\,\\%"


def sci(v, d=1):
    if v == 0:
        return "0"
    e = math.floor(math.log10(abs(v)))
    m = f"{v / 10 ** e:.{d}f}"
    if float(m) in (10.0, -10.0):
        m, e = f"{float(m) / 10:.{d}f}", e + 1
    return f"\\ensuremath{{{m}\\times 10^{{{e}}}}}"


def thou(n):
    return f"{int(n):,}".replace(",", "{,}")


def table(name, cols, rows, head, size="\\small", thin=0):
    """A booktabs tabular; rows may contain the string 'MID' for a midrule. head is a list of cells or raw header lines.
    thin: number of leading label columns in which a repeat of the row above is left blank (check_numbers.py fills them back)."""
    head = head if isinstance(head, str) else " & ".join(head)
    L = [size, f"\\begin{{tabular}}{{{cols}}}", "\\toprule", head + " \\\\", "\\midrule"]
    last = None
    for r in rows:
        if r == "MID":
            L.append("\\midrule")
            last = None
            continue
        c = [str(x) for x in r]
        shown = list(c)
        for i in range(thin):
            if last and c[:i + 1] == last[:i + 1]:
                shown[i] = ""
        last = c
        L.append(" & ".join(shown) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(GEN, f"tab_{name}.tex"), "w").write("\n".join(L) + "\n")


def mc(*cells):
    """Header cells over decimal-aligned columns."""
    return [f"\\multicolumn{{1}}{{c}}{{{c}}}" for c in cells]


def load():
    H = json.load(open(os.path.join(S, "hypotheses.json")))
    tg = pd.read_csv(os.path.join(S, "thresholds_gauss.csv"))
    tp = pd.read_csv(os.path.join(S, "thresholds_population.csv"))
    conv = pd.read_csv(os.path.join(S, "convergence.csv"))
    cg = pd.read_csv(os.path.join(S, "curves_gauss.csv.gz"))
    ph = json.load(open(os.path.join(HERE, "posthoc.json")))
    return H, tg, tp, conv, cg, ph


def compute_block():
    st = {os.path.basename(p)[:-12]: json.load(open(p)) for p in sorted(glob.glob(os.path.join(ROOT, "results/merged/*_status.json")))}
    put("jobs", thou(sum(s["done"] for s in st.values())))
    put("failed", sum(len(s["failed"]) for s in st.values()))
    put("gpuh.jobs", f"{sum(s['gpu_hours'] for s in st.values()):.1f}")
    rec = json.load(open(os.path.join(ROOT, "results", "run_record.json")))
    put("gpuh.billed", f"{rec['billed_gpu_hours'] + rec['final_validation_seconds'] / 3600:.2f}")
    meta = {e: pd.read_csv(os.path.join(ROOT, f"results/merged/{e}_meta.csv")) for e in st}
    order = ("6a98646", "8addd92", "9bac5d5", "b8e5341", "3dd5707")
    rows = []
    plain = {"gauss": "Gaussian sweep", "population": "population sweep", "convergence": "convergence study", "families": "copula families", "lkj": "LKJ matrices", "network": "chains"}
    for e in plain:
        c = meta[e].git.str[:7].value_counts()
        rows.append([plain[e], thou(st[e]["done"]), f"{st[e]['gpu_hours']:.1f}"] + [thou(c.get(o, 0)) if c.get(o, 0) else DASH for o in order])
    tot = pd.concat(meta.values()).git.str[:7].value_counts()
    rows += ["MID", ["total", V["jobs"], V["gpuh.jobs"]] + [thou(tot.get(o, 0)) for o in order]]
    assert set(tot.index) == set(order)
    table("batches", "lrr" + "r" * 5, rows, ["experiment", "runs", "GPU-h"] + [f"batch {i + 1}" for i in range(len(order))], "\\footnotesize")
    put("batches.n", len(tot))
    allm = pd.concat(meta.values())
    assert allm.torch.nunique() == 1 and allm.numpy.nunique() == 1 and allm.gpu.nunique() == 1
    put("sw.torch", allm.torch.iloc[0])
    put("sw.numpy", allm.numpy.iloc[0])
    put("sw.scipy", rec["scipy"])
    return st


def thresholds(H, tg, tp):
    g = tg[(tg.m == 4) & (tg.belief == "S2")].set_index(["type", "basis", "k"])
    p = tp[(tp.m == 4) & (tp.belief == "S2")].set_index(["type", "basis", "k"])
    for t in TYPES:
        for b in BASES:
            for k in KS:
                r, q = g.loc[(t, b, k)], p.loc[(t, b, k)]
                for key, v in (("d1", r.denom_1pct), ("d5", r.denom_5pct), ("on", r.neg_onset), ("pop", q.neg_onset), ("und", r.undef_onset)):
                    put(f"thr.{t}.{b}.{k}.{key}", f(v))

    def rows(types, ci):
        out = []
        for t in types:
            for b in BASES:
                for k in KS:
                    r, q = g.loc[(t, b, k)], p.loc[(t, b, k)]

                    def c(n):
                        if not np.isfinite(r[n]):
                            return DASH
                        return f"{f(r[n])} ({f(r[n + '_lo'])}--{f(r[n + '_hi'])})" if ci else f(r[n])
                    out.append(([t] if len(types) > 1 else []) + [b, k, c("denom_1pct"), c("denom_5pct"), c("neg_onset"), f(q.neg_onset), f(r.undef_onset)])
            if t != types[-1]:
                out.append("MID")
        return out
    h = ["basis", "$k$", "$d_{\\mathrm{denom}}\\geq 1\\,\\%$", "$d_{\\mathrm{denom}}\\geq 5\\,\\%$", "onset, $n=10^7$", "onset, pop.", "undefined from"]
    table("thr_main", "lrrrrrr", rows(("SUM",), False), h, "\\footnotesize", thin=1)
    table("thr_full", "llrlllrr", rows(TYPES, True), ["type"] + h, "\\scriptsize\\setlength{\\tabcolsep}{4pt}", thin=2)
    rr = []
    for t in TYPES:
        for b in BASES:
            for k in KS:
                r = g.loc[(t, b, k)]
                c = lambda n: DASH if not np.isfinite(r[n]) else f"{f(r[n])} ({f(r[n + '_lo'])}--{f(r[n + '_hi'])})"  # noqa: E731
                rr.append([t, b, k, c("relvar_5pct"), c("relvar_20pct"), f(r.minD_at_0).replace("-", "$-$")])
    table("thr_relvar", "llrllr", rr, ["type", "basis", "$k$", "$|\\mathrm{relvar}|\\geq 5\\,\\%$", "$|\\mathrm{relvar}|\\geq 20\\,\\%$", "$\\min D$ at $\\rho_S=0$"], "\\footnotesize", thin=2)
    # all beliefs, pair basis
    t4 = tg[(tg.m == 4) & (tg.basis == "pair")]
    rows_b = []
    for t in TYPES:
        for k in KS:
            c = [t, k]
            for bel in ("S2", "S3a", "S3b", "S3c", "S3d", "S4"):
                r = t4[(t4.type == t) & (t4.k == k) & (t4.belief == bel)].iloc[0]
                c.append(f"{f(r.denom_1pct)} / {f(r.undef_onset, 2 if r.undef_onset >= 0.1 or r.undef_onset == 0 else 3)}")
                put(f"bel.{t}.{k}.{bel}.d1", f(r.denom_1pct))
                put(f"bel.{t}.{k}.{bel}.und", f(r.undef_onset))
            rows_b.append(c)
        if t == "SUM":
            rows_b.append("MID")
    table("beliefs", "lrllllll", rows_b, ["type", "$k$", "S2", "S3a", "S3b", "S3c", "S3d", "S4"], "\\footnotesize", thin=1)
    # degree dependence of the onset (H2a)
    c2 = {(c["basis"], c["k"]): c for c in H["H2a"]["cells"] if c["type"] == "SUM"}
    r2 = [[k] + [f(c2[(b, k)][f"onset_m{m}"]) for b in BASES for m in (2, 4, 6)] for k in KS]
    table("onset_m", "r" + "r" * 6, r2, "& \\multicolumn{3}{c}{pair} & \\multicolumn{3}{c}{triple} \\\\ \\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n$k$ & $m=2$ & $m=4$ & $m=6$ & $m=2$ & $m=4$ & $m=6$", "\\footnotesize")


def hypotheses(H, conv, ph, cg):
    verdicts = {}
    for k, v in H.items():
        verdicts[k] = v["verdict"]
        put(f"{k}.verdict", v["verdict"])
    c = lambda h: len(H[h]["cells"])  # noqa: E731
    for h in ("H1a", "H1b", "H2a", "H2b_population", "H2b_gauss"):
        put(f"{h}.sig", H[h]["significant"])
        put(f"{h}.n", c(h))
    sp = {(x["type"], x["basis"]): x for x in H["H1b"]["cells"]}[("SUM", "pair")]
    put("H1b.sp.k2", f(sp["onset_k2"]))
    put("H1b.sp.k32", f(sp["onset_k32"]))
    put("H1b.sp.lo", f(sp["ci"][0]))
    put("H1b.sp.hi", f(sp["ci"][1]))
    w = {f"{x['type']}/{x['basis']}": x for x in H["H1c"]["cells"]}
    for key, x in w.items():
        put(f"H1c.{key}", f(x["within2"], 2))
    put("H2c.share", pc(H["H2c"]["share"]))
    put("H2c.n", H["H2c"]["n"])
    for h in ("H3a", "H3b"):
        put(f"{h}.in", H[h]["n_inside"])
        put(f"{h}.out", H[h]["n_outside"])
        put(f"{h}.n", H[h]["n"])
    h3c = {x["basis"]: x for x in H["H3c"]["cells"]}
    put("H3c.pair", pc(h3c["pair"]["within2"]))
    put("H3c.triple", pc(h3c["triple"]["within2"]))
    put("H3c.n", h3c["pair"]["n"])
    put("H4a.share", pc(H["H4a"]["share"]))
    put("H4a.n", H["H4a"]["n"])
    put("H4a.ratio", f(H["H4a"]["median_sd_over_mcse"], 1))
    a = cg[(cg.m == 4) & np.isclose(cg.eps, 0.01) & cg.belief.isin(["S2", "S3c", "S4"]) & (cg.rho_S > 0) & (cg.share_undefined < 0.5)
           & np.isfinite(cg.d_denom_sd)]  # post hoc: H4a over all headline cells, both bases and every k
    put("ph.H4a.all.share", pc((a.d_denom_sd > a.d_denom_mcse).mean()))
    put("ph.H4a.all.n", len(a))
    n2, oth = conv[conv.variant == "n2"], conv[conv.variant != "n2"]
    tot, viol = int(conv.n.sum()), int(conv.n_violations.sum())
    put("H4b.viol", thou(viol))
    put("H4b.total", thou(tot))
    put("H4b.share", pc(viol / tot, 2))
    put("H4b.n2", thou(n2.n.sum()))
    put("H4b.n2share", pc(viol / n2.n.sum(), 1))
    put("H4b.worst.n2", f(n2.worst_ratio.max(), 1))
    put("H4b.worst.other", f(oth.worst_ratio.max(), 2))
    put("H4b.other.viol", int(oth.n_violations.sum()))
    put("H4b.maxchange", f(n2.max_change.max(), 1))
    v = pd.read_csv(os.path.join(S, "convergence_violations.csv"))
    for k, n in v.groupby("k").size().items():
        put(f"H4b.k{k}", thou(n))
    assert set(v.variant) == {"n2"} and len(v) == viol
    assert viol == int(re.search(r"(\d+) violations", json.load(open(os.path.join(ROOT, "tests/test_results.json")))["M13"]["detail"]).group(1))
    for h in ("H5a", "H5b", "H5c", "H5d"):
        put(f"{h}.sig", H[h]["significant"])
        put(f"{h}.n", H[h]["n"])
    side = {"H5a": 1, "H5b": 0, "H5c": 0, "H5d": 0}
    opp = {h: sum(x["ci"][1] < 1 if h == "H5a" else x["ci"][0] > 0 for x in H[h]["cells"]) for h in side}
    for h, n in opp.items():
        put(f"{h}.opp", n)
    a = pd.DataFrame(H["H5a"]["cells"]).groupby("belief").ratio.median()
    for b, r in a.items():
        put(f"H5a.ratio.{b}", f(r, 2))

    # H2b is one hypothesis with two evaluations, both supported, so 16 rows are 15 hypotheses
    assert verdicts["H2b_gauss"] == verdicts["H2b_population"] == "supported"
    one = [v for k, v in verdicts.items() if k != "H2b_gauss"]
    put("hyp.total", len(one))
    put("hyp.rows", len(verdicts))
    for w_, key in (("supported", "sup"), ("inconclusive", "inc"), ("not supported", "not")):
        put(f"hyp.{key}", one.count(w_))
    assert len(one) == sum(one.count(w_) for w_ in ("supported", "inconclusive", "not supported"))
    vd = lambda k: f"\\emph{{{verdicts[k]}}}"  # noqa: E731
    wtxt = ", ".join(f"{k} {f(w[k]['within2'], 2)}" for k in ("SUM/pair", "CONTRAST/pair", "SUM/triple", "CONTRAST/triple"))
    rows = [
        ["H1a", "population onset falls with $k$", vd("H1a"), f"{V['H1a.sig']}/{V['H1a.n']} cells strictly decreasing"],
        ["H1b", "same at $n=10^7$", vd("H1b"), f"{V['H1b.sig']}/{V['H1b.n']}; SUM/pair {V['H1b.sp.k2']} ($k=2$) vs {V['H1b.sp.k32']} ($k=32$)"],
        ["H1c", "$d_{\\mathrm{denom}}$ one power law of $\\chi_D$", vd("H1c"), f"within factor 2: {wtxt}"],
        ["H2a", "higher $m$, earlier onset", vd("H2a"), f"{V['H2a.sig']}/{V['H2a.n']}"],
        ["H2b", "triple earlier than pair (population)", vd("H2b_population"), f"{V['H2b_population.sig']}/{V['H2b_population.n']}"],
        ["H2b", "triple earlier than pair ($n=10^7$)", vd("H2b_gauss"), f"{V['H2b_gauss.sig']}/{V['H2b_gauss.n']}"],
        ["H2c", "$d_{\\mathrm{denom}}$ depends weakly on $m$", vd("H2c"), f"{V['H2c.share']} of {V['H2c.n']} points; supported needs $\\geq 80\\,\\%$"],
        ["H3a", "onset within $\\pm 0.05$ of Gaussian", vd("H3a"), f"{V['H3a.in']}/{V['H3a.n']} CIs inside, {V['H3a.out']} entirely outside"],
        ["H3b", "$d_{\\mathrm{denom}}$ within factor 2 of Gaussian", vd("H3b"), f"{V['H3b.in']}/{V['H3b.n']} CIs inside, {V['H3b.out']} entirely outside"],
        ["H3c", "Gaussian power law predicts LKJ", vd("H3c"), f"pair {V['H3c.pair']}, triple {V['H3c.triple']} of {V['H3c.n']} matrices; $k=2$ only"],
        ["H4a", "seed spread exceeds MC error", vd("H4a"), f"{V['H4a.share']} of {V['H4a.n']} points; median ratio {V['H4a.ratio']}"],
        ["H4b", "doubling changes TV $<\\max(3\\,\\mathrm{SE},10^{-3})$", vd("H4b"), f"{V['H4b.viol']} violations, all under $n\\times 2$"],
        ["H5a", "$d_{\\mathrm{total}}$ grows from $L=1$ to 4", vd("H5a"), f"{V['H5a.sig']}/{V['H5a.n']}; {V['H5a.opp']} significantly below 1"],
        ["H5b", "training reduces $d_{\\mathrm{denom}}$", vd("H5b"), f"{V['H5b.sig']}/{V['H5b.n']}; {V['H5b.opp']} opposite"],
        ["H5c", "training reduces $P(D<0)$", vd("H5c"), f"{V['H5c.sig']}/{V['H5c.n']}; {V['H5c.opp']} opposite"],
        ["H5d", "end-to-end training reduces held-out $d_{\\mathrm{total}}$", vd("H5d"), f"{V['H5d.sig']}/{V['H5d.n']}; {V['H5d.opp']} opposite"],
    ]
    table("hyp", "@{}l>{\\raggedright\\arraybackslash}p{0.31\\linewidth}l>{\\raggedright\\arraybackslash}p{0.40\\linewidth}@{}", rows, ["", "claim", "verdict", "evidence"], "\\footnotesize")
    # H1c and H3c details
    table("h1c", "llrrrrl", [[x["type"], x["basis"], x["n"], f(x["beta"], 2), f(x["C"], 3), f(x["within2"], 2), x["verdict"]] for x in H["H1c"]["cells"]],
          ["type", "basis", "points", "$\\beta$", "$C$", "within $2\\times$", "cell verdict"], "\\footnotesize", thin=1)
    for x in H["H1c"]["cells"]:
        put(f"H1c.verdict.{x['type']}/{x['basis']}", x["verdict"])
    put("H1c.cells", len(H["H1c"]["cells"]))
    put("H1c.cells.sup", sum(x["verdict"] == "supported" for x in H["H1c"]["cells"]))
    # families
    c3 = pd.DataFrame(H["H3a"]["cells"])
    b3 = pd.DataFrame(H["H3b"]["cells"]).groupby("family").ratio.median()
    rows = []
    for fam, g in c3.groupby("family"):
        sg = lambda v: f"{v:+.3f}".replace("-", "$-$")  # noqa: E731
        rows.append([{"clayton": "Clayton", "gumbel": "Gumbel", "t10": "$t_{10}$", "t3": "$t_3$"}[fam], len(g), sg(g["diff"].min()), sg(g["diff"].max()), int(g.inside.sum()), int(g.outside.sum()), f(b3[fam], 2)])
        put(f"fam.{fam}.ratio", f(b3[fam], 2))
        put(f"fam.{fam}.min", f"{g['diff'].min():+.3f}".replace("-", "$-$"))
    table("families", "lrrrrrr", rows, ["family", "cells", "min diff.", "max diff.", "CI inside", "CI outside", "median ratio"], "\\footnotesize")
    rows = []
    for key, val in sorted(h3c["pair"]["by_k_eta"].items()):
        k_, e_ = key.split("_")
        rows.append([k_, e_, ph["lkj"][f"{k_}_{e_}_pair"]["defined"], ph["lkj"][f"{k_}_{e_}_pair"]["n"], pc(val), pc(h3c["triple"]["by_k_eta"][key])])
    table("lkj", "rrrrrr", rows, ["$k$", "$\\eta$", "defined", "drawn", "within $2\\times$, pair", "within $2\\times$, triple"], "\\footnotesize")
    put("lkj.defined.k4", sum(ph["lkj"][f"4_{e}_pair"]["defined"] for e in (1, 4)))
    put("lkj.defined.k8", sum(ph["lkj"][f"8_{e}_pair"]["defined"] for e in (1, 4)))
    # convergence table
    lab = {"M2": "$M\\times 2$", "n2": "$n\\times 2$", "s2": "$s$-nodes $\\times 2$", "x2": "$x$-nodes $\\times 2$"}
    rows = []
    for vn, g in conv.groupby("variant"):
        rows.append([lab[vn], thou(g.n.sum()), thou(g.n_violations.sum()), sci(g.median_change.median()), sci(g.max_change.max()), f(g.worst_ratio.max(), 2)])
    table("conv", "lrrrr" + "D{.}{.}{2}", rows, ["variant", "paired values", "violations", "median change", "max change"] + mc("worst ratio"), "\\footnotesize")


def decomposition(cg):
    c = cg[np.isclose(cg.eps, 0.01) & (cg.m == 4) & (cg.basis == "pair")]
    rows = []
    cols = ("d_denom", "d_neg", "d_model_uncal", "d_model_cal", "d_total")
    for bel in ("S2", "S3c", "S4"):
        for t in TYPES:
            for rho in (0.1, 0.2, 0.4):
                r = c[(c.belief == bel) & (c.type == t) & (c.k == 2) & np.isclose(c.rho_S, rho)].iloc[0]
                rows.append([bel, t, f(rho, 1)] + [f(r[x + "_med"], 4) for x in cols])
                for x in cols:
                    put(f"dec.{bel}.{t}.{rho}.{x}", f(r[x + "_med"], 4))
        if bel != "S4":
            rows.append("MID")
    table("decomp", "llrrrrrr", rows, ["belief", "type", "$\\rho_S$", "$d_{\\mathrm{denom}}$", "$d_{\\mathrm{neg}}$", "$d_{\\mathrm{model,uncal}}$", "$d_{\\mathrm{model,cal}}$", "$d_{\\mathrm{total}}$"], "\\footnotesize", thin=2)
    for t in TYPES:
        for rho in (0.2,):
            r = c[(c.belief == "S2") & (c.type == t) & (c.k == 2) & np.isclose(c.rho_S, rho)].iloc[0]
            put(f"relvar.S2.{t}.{rho}", pc(abs(r.relvar_vs_uncal_med), 1))
    s = cg[np.isclose(cg.eps, 0.01) & (cg.m == 4) & (cg.basis == "pair") & (cg.type == "SUM") & (cg.belief == "S2") & (cg.k == 2)]
    put("d.rho02.k2", f(s[np.isclose(s.rho_S, 0.2)].d_denom_med.iloc[0], 3))


def posthoc(ph, H):
    e = ph["eqd"]
    put("ph.eqd.nonpos", thou(e["n_nonpos"]))
    put("ph.eqd.rows", thou(e["n_rows"]))
    put("ph.eqd.nonpos4", thou(e["n_nonpos_m4"]))
    put("ph.eqd.rows4", thou(e["n_rows_m4"]))
    put("ph.eqd.mink", e["min_k"])
    put("ph.eqd.min", thou(round(abs(e["min_value"]))))
    put("ph.eqd.pop.nonpos", thou(e["pop_n_nonpos"]))
    put("ph.eqd.pop.rows", thou(e["pop_n_rows"]))
    put("ph.eqd.pop.mink", e["pop_min_k"])
    put("ph.eqd.pairmin", f(e["pair_min"], 2))
    assert e["bases"] == ["triple"] and e["pop_bases"] == ["triple"] and e["beliefs"] == ["S3a", "S3b", "S3c", "S3d"]
    t = ph["tv_above_one"]
    put("ph.tv1.n", thou(t["n"]))
    put("ph.tv1.share", pc(t["n"] / t["n_rows"], 1))
    put("ph.tv1.max", f(t["max_d_total"], 0))
    put("ph.tv1.triple", pc(t["share_triple"]))
    m = ph["s2_marginal"]
    put("ph.s2.gap", sci(m["median_gap"]))
    put("ph.s2.maxgap", f(m["max_gap"], 3))
    put("ph.s2.sdchi", f(m["sd_over_chi"], 2))
    s = ph["s4"]
    put("ph.s4.pair", sci(s["pair_median_d_total"]))
    put("ph.s4.pairmax", sci(s["pair_max_d_total"]))
    put("ph.s4.triple", f(s["triple_median_d_total"], 3))
    put("ph.s4.triplemax", f(s["triple_max_d_total"], 2))
    b = ph["tv_bound"]
    put("ph.bound.max", f(b["max_ratio"], 2))
    put("ph.bound.med", f(b["median_ratio"], 3))
    put("ph.bound.n", thou(b["n"]))
    c = ph["closer"]
    put("ph.closer.all", pc(c["all"]))
    put("ph.closer.undef", pc(c["undefined"]))
    put("ph.closer.undef.pair", pc(c["undefined_pair"]))
    put("ph.closer.def", pc(c["defined"]))
    for key in ("S4_pair", "S3b_pair", "S2_pair", "S2_triple", "S4_triple", "S3c_pair"):
        put(f"ph.closer.{key}", pc(c[key]))
    d = ph["denom_vs_neg"]
    put("ph.dvn.n", d["0.01"]["n"])
    for eps in ("0.1", "0.01", "0.001"):
        put(f"ph.dvn.{eps}", pc(d[eps]["share_neg_larger"]))
    for bel, v in d["by_belief"].items():
        put(f"ph.dvn.{bel}", pc(v))
        put(f"ph.dvn.ratio.{bel}", f(d["median_ratio"][bel], 1))
    mv = ph["mean_var"]
    put("ph.mv.n", mv["n"])
    put("ph.mv.share", pc(mv["share_var_larger"]))
    put("ph.mv.s2.var", pc(mv["max_relvar_s2"], 1))
    put("ph.mv.s2.mean", f(mv["max_smean_s2"], 4))
    put("ph.mv.s3c.var", f(mv["max_relvar_s3c_contrast"], 2))
    put("ph.mv.s3c.mean", f(mv["max_smean_s3c_contrast"], 2))
    put("ph.narrow.max", f(max(v["max_d_denom"] for v in ph["narrow"].values()), 4))
    put("ph.narrow.k32", f(ph["narrow"]["k32"]["max_d_denom"], 4))
    put("ph.narrow.k32.rho", f(ph["narrow"]["k32"]["max_rho"], 1))
    rows = []
    for r in ph["noise"]:
        if r["basis"] == "triple" and r["k"] == 2:
            continue
        rows.append([r["basis"], r["k"], thou(r["n_coef_D"]), f(r["sd_corner"], 3), f(r["drop"], 3), f(r["z"], 1), sci(r["chi0"], 2), sci(r["chi0_pred"], 2)])
    table("noise", "lrrrrrrr", rows, ["basis", "$k$", "$|B_D|$", "$\\sigma_{\\mathrm{c}}$", "$1-\\min D$", "ratio", "$\\chi_D$", "$\\sqrt{|B_D|/n}$"], "\\footnotesize", thin=1)
    zs = [r["z"] for r in ph["noise"]]
    put("ph.noise.zmin", f(min(zs), 1))
    put("ph.noise.zmax", f(max(zs), 1))
    put("ph.noise.chirel", pc(ph["noise_chi_maxrel"]))
    nz = {(r["basis"], r["k"]): r for r in ph["noise"]}
    put("ph.noise.pair32", f(nz[("pair", 32)]["sd_corner"], 2))
    put("ph.noise.triple16", f(nz[("triple", 16)]["sd_corner"], 2))
    put("ph.noise.triple32", f(nz[("triple", 32)]["sd_corner"], 1))
    put("ph.noise.drop.triple32", f(nz[("triple", 32)]["drop"], 1))
    put("ph.noise.drop.pair32", f(nz[("pair", 32)]["drop"], 2))
    put("ph.chi.min", f(ph["chi_lead"]["min"], 2))
    put("ph.chi.max", f(ph["chi_lead"]["max"], 2))
    ko = ph["k_times_onset"]
    put("ph.ko.pair.lo", f(min(ko[f"pair_{k}"] for k in KS), 2))
    put("ph.ko.pair.hi", f(max(ko[f"pair_{k}"] for k in KS), 2))
    put("ph.ko.triple.lo", f(min(ko[f"triple_{k}"] for k in KS[1:]), 2))
    put("ph.ko.triple.hi", f(max(ko[f"triple_{k}"] for k in KS[1:]), 2))
    fr = ph["families_rho0"]
    put("ph.t3.chi", f(fr["t3_2"]["chi"], 2))
    put("ph.t10.chi", f(fr["t10_2"]["chi"], 2))
    put("ph.g.chi", f(fr["gauss_2"]["chi"], 4))
    put("ph.t3.minD4", f(fr["t3_4"]["minD"], 2).replace("-", "$-$"))
    h = ph["h4b"]
    put("ph.h4b.rel", pc(h["median_rel_tv_except_denom"], 1))
    put("ph.h4b.triple", pc(h["share_triple"]))
    w = h["worst"]
    put("ph.h4b.worst.from", f(w["base"], 1))
    put("ph.h4b.worst.to", f(w["value"], 1))
    assert w["k"] == 32 and w["basis"] == "triple" and w["belief"] == "S3d" and w["m"] == 6
    # depth
    a6 = ph["depth_a6"]
    put("ph.a6", f(a6, 3))
    rows = []
    for bel in ("S2", "S3c", "S4"):
        for basis in BASES:
            g = ph["depth"][f"{bel}_{basis}"]
            sp = ph["h5a_split"][f"{bel}_{basis}"]
            rows.append([bel, basis] + [f(g[str(L)]["d_total"], 4) for L in (1, 2, 3, 4)] + [f(g["1"]["var_gold"], 4), f(g["4"]["var_gold"], 4), f(sp["ratio"], 2), f"{sp['sig']}/{sp['below']}/{sp['n']}"])
            put(f"ph.h5a.{bel}.{basis}.ratio", f(sp["ratio"], 2))
            put(f"ph.h5a.{bel}.{basis}.sig", sp["sig"])
            put(f"ph.h5a.{bel}.{basis}.n", sp["n"])
            for L in (1, 4):
                put(f"ph.depth.{bel}.{basis}.{L}", f(g[str(L)]["d_total"], 4))
                put(f"ph.depth.var.{bel}.{basis}.{L}", f(g[str(L)]["var_gold"], 4))
                put(f"ph.depth.varcd.{bel}.{basis}.{L}", f(g[str(L)]["var_cd"], 4))
    table("depth", "llrrrrrrrr", rows, "& & \\multicolumn{4}{c}{$d_{\\mathrm{total}}$ at layer} & \\multicolumn{2}{c}{$\\mathrm{Var}\\,p_{\\mathrm{G}}$} & \\multicolumn{2}{c}{H5a} \\\\ \\cmidrule(lr){3-6}\\cmidrule(lr){7-8}\\cmidrule(l){9-10}\nbelief & basis & $L=1$ & $L=2$ & $L=3$ & $L=4$ & $L=1$ & $L=4$ & ratio & above/below/all", "\\footnotesize", thin=1)
    s_ = ph["h5a_sig"]
    put("ph.h5a.sig.maxL4", f(s_["max_d_total_L4"], 4))
    put("ph.h5a.sig.floor", f(s_["max_over_floor_L4"], 2))
    put("ph.h5a.sig.rho0", s_["n_rho0"])
    put("ph.layer.minD", f(ph["layer_minD"]["2"], 2))
    put("ph.layer.minDlow", f(ph["layer_minD_low"], 2))
    # training
    tr = ph["training"]
    name = {"val": "value", "nll": "NLL", "e2e1": "end-to-end, $L=1$", "e2e2": "end-to-end, $L=2$", "e2e3": "end-to-end, $L=3$", "e2e4": "end-to-end, $L=4$"}
    rows = []
    for st in ("val", "nll", "e2e1", "e2e2", "e2e3", "e2e4"):
        for opt, on in (("adam", "Adam"), ("lbfgs", "L-BFGS")):
            r = tr[f"{st}_{opt}"]
            rows.append([name[st], on, pc(r["kept_initial"]), f(r["median_coef_dist"], 3), pc(r["median_rel_test"], 1).replace("-", "$-$"), pc(r["share_test_reduced"]),
                         f(r["median_dP_where_neg"], 3).replace("-", "$-$"), f(r["median_minD_change"], 2).replace("-", "$-$")])
            for key, val in (("kept", pc(r["kept_initial"])), ("test", pc(abs(r["median_rel_test"]), 1)), ("dP", f(abs(r["median_dP_where_neg"]), 3)),
                             ("dminD", f(abs(r["median_minD_change"]), 1))):
                put(f"ph.tr.{st}.{opt}.{key}", val)
    table("training", "llrrrrrr", rows, ["loss", "optimiser", "kept initial", "coef.\\ dist.", "test loss", "test loss down", "$\\Delta P(D<0)$", "$\\Delta\\min D$"], "\\footnotesize", thin=1)
    assert all(r["n_nonfinite"] == 0 and r["n_cross"] == 0 for r in tr.values())
    put("ph.tr.runs", thou(sum(r["n"] for r in tr.values())))
    by = ph["by_loss"]
    for h_ in ("H5b", "H5c"):
        for l, r in by[h_].items():
            put(f"ph.{h_}.{l}.sig", r["sig"])
            put(f"ph.{h_}.{l}.n", r["n"])
            put(f"ph.{h_}.{l}.opp", r["opp"])
    assert by["H5b_ks"] == [2]
    for g, d_ in by["H5d"].items():
        for l, r in d_.items():
            put(f"ph.H5d.{g}.{l}.sig", r["sig"])
            put(f"ph.H5d.{g}.{l}.opp", r["opp"])
            put(f"ph.H5d.{g}.{l}.n", r["n"])
    e1 = ph["e2e_beliefs"]
    for key in ("1_adam", "1_lbfgs"):
        put(f"ph.e2e.{key}.S2", f(abs(e1[key]["S2"]), 3))
        put(f"ph.e2e.{key}.S3c", f(abs(e1[key]["S3c"]), 3))
        put(f"ph.e2e.{key}.held", f(e1[key]["held"], 3))
    fa = ph["flat_after"]
    put("ph.flat.nll.P", pc(fa["nll_adam"]["P"], 1))
    put("ph.flat.val.P", pc(fa["val_adam"]["P"], 1))
    put("ph.flat.nll.chi", f(fa["nll_adam"]["chi"], 2))
    assert ph["flat_init"]["max_d_denom"] == 0 and ph["flat_init"]["max_P"] == 0
    assert set(ph["t_commits"]) == {"3dd5707", "b8e5341"}


def energy():
    e = pd.read_csv(os.path.join(ROOT, "results", "energy", "summary.csv"))
    sig = lambda v: f"{v:.3g}"  # noqa: E731
    assert e.n_runs.nunique() == 1 and e.n_repeats.nunique() == 1
    put("en.runs.dtype", e.n_runs.iloc[0])
    put("en.row.repeats", e.n_repeats.iloc[0])
    main = e[(e.k == 8) & (e.B == 16384)].set_index(["layer", "dtype"])
    rows = []
    for l in ("HCR-CD", "HCR-CAL", "MLP", "KAN"):
        a, b = main.loc[(l, "float32")], main.loc[(l, "float64")]
        rows.append([l, thou(a.params), sig(a.ms), sig(a.J), sig(b.ms), sig(b.J)])
        for dt, r in (("32", a), ("64", b)):
            put(f"en.{l}.{dt}.ms", sig(r.ms))
            put(f"en.{l}.{dt}.J", sig(r.J))
    table("energy_main", "lr" + "D{.}{.}{4}" * 4, rows, "& & \\multicolumn{2}{c}{float32} & \\multicolumn{2}{c}{float64} \\\\ \\cmidrule(lr){3-4}\\cmidrule(l){5-6}\nlayer & parameters & " + " & ".join(mc("ms", "J", "ms", "J")), "\\footnotesize")
    for dt in ("float32", "float64"):
        rows = []
        for r in e[e.dtype == dt].sort_values(["k", "B", "layer"], kind="stable").itertuples():
            rows.append([r.k, thou(r.B), r.layer, thou(r.params), sig(r.ms), sig(r.J), sig(r.J_net), f"{r.power_W:.0f}", f"{r.sm_MHz:.0f}", f"{r.counter_ratio:.3f}"])
        table(f"energy_{dt}", "rrlr" + "D{.}{.}{4}" + "D{.}{.}{5}" + "D{.}{.}{6}" + "rrr", rows, ["$k$", "batch", "layer", "params"] + mc("ms", "J", "J net") + ["W", "MHz", "counter"], "\\scriptsize", thin=2)
    t = open(os.path.join(ROOT, "results", "energy", "report.txt")).read()
    m = re.search(r"Counter vs sampled energy over (\d+) repeats: median ([\d.]+), 1st-99th percentile ([\d.]+)-([\d.]+)", t)
    put("sw.driver", re.search(r"driver ([\d.]+),", t).group(1))
    put("en.repeats", thou(m.group(1)))
    put("en.counter.med", m.group(2))
    put("en.counter.lo", m.group(3))
    put("en.counter.hi", m.group(4))
    m = re.search(r"Rows where idle power.*?: (\d+) of (\d+)", t)
    put("en.idle.n", m.group(1))
    put("en.idle.of", m.group(2))
    m = re.search(r"Repeats at >= 97.5 % of the power limit: (\d+)", t)
    put("en.limit.repeats", m.group(1))
    put("en.runs", len(re.search(r"from 6 runs \((.*?)\)", t).group(1).split(", ")))
    rel = [float(x) for x in re.findall(r"\| \d{7} \| float\d\d \| \w+ \| t\d+ \| [\d.]+ \| ([\d.]+) \|", t)]
    put("en.rel.hi1", f"{sorted(rel)[-2] * 100 - 100:.0f}")
    put("en.rel.hi2", f"{sorted(rel)[-1] * 100 - 100:.0f}")
    x = e.set_index(["dtype", "layer", "k", "B"])
    put("en.mlpfast.32", sig(x.loc[("float32", "MLP", 2, 16384)].ms))
    put("en.mlpfast.64", sig(x.loc[("float64", "MLP", 2, 16384)].ms))
    a = x.loc[("float64", "MLP", 32, 16384)]
    put("en.mlp.limit.W", f"{a.power_W:.0f}")
    for dt in ("float32", "float64"):
        r = x.loc[(dt, "HCR-CD", 8, 16384)].J / x.loc[(dt, "MLP", 8, 16384)].J
        put(f"en.cd.over.mlp.{dt[-2:]}", f(r, 1))
        r = x.loc[(dt, "HCR-CAL", 8, 16384)].J / x.loc[(dt, "HCR-CD", 8, 16384)].J
        put(f"en.cal.over.cd.{dt[-2:]}", f"{r:.0f}")
        r = x.loc[(dt, "KAN", 8, 16384)].J / x.loc[(dt, "HCR-CD", 8, 16384)].J
        put(f"en.kan.over.cd.{dt[-2:]}", f(r, 1))


def validation():
    T = json.load(open(os.path.join(ROOT, "tests", "test_results.json")))
    d = {k: v["detail"] for k, v in T.items()}
    what = {"M1": "orthonormality of the basis, explicit $f_1,\\dots,f_4$", "M2": "index families, coefficient counts, block vs dense estimator",
            "M3": "estimator vs population moments, SE scaling, Gauss--Hermite convergence", "M4": "reproduces the TV values of the $d=3$ study",
            "M5": "flat-$D$ control: $p_{\\mathrm{CD}}=p_{\\mathrm{EU}}$", "M6": "covariance identity in sample",
            "M7": "integrals of all densities; copula samplers and conditionals; Student-$t$ CDF and quantile against 50-digit mpmath",
            "M8": "TV in $s$ vs $x$; grid refinement with spiky $p_{\\mathrm{G}}$", "M9": "min-$D$ optimiser vs brute-force grids and closed forms",
            "M10": "positive definiteness of all correlation matrices; LKJ moments", "M11": "bitwise reproducibility; a run interrupted and resumed",
            "M12": "seed variance matches estimated SE; common random numbers", "M13": "convergence under doubling = hypothesis H4b, on the study's data",
            "M14": "chain: $L=1$ vs engine, closed form vs MC, autograd vs finite differences, teacher Markov property"}
    rows = [[k, what[k], "pass" if T[k]["pass"] else "\\textbf{fail}"] for k in sorted(T, key=lambda s: int(s[1:]))]
    assert all(T[k]["pass"] for k in T if k != "M13") and not T["M13"]["pass"] and len(T) == 14
    assert all(T[k].get("device") == "cuda" for k in T if k != "M13")
    table("validation", "@{}lp{0.72\\linewidth}l@{}", rows, ["test", "what it checks", "result"], "\\footnotesize")
    put("val.M6", re.search(r"max residual ([\d.e+-]+)", d["M6"]).group(1).replace("e-16", "\\times 10^{-16}"))
    put("val.M6.n", re.search(r"over (\d+) defined", d["M6"]).group(1))
    put("val.M9", re.search(r"max \|diff\| ([\d.e+-]+);", d["M9"]).group(1).replace("e-05", "\\times 10^{-5}"))
    put("val.M4.n", re.search(r"(\d+) TV comparisons", d["M4"]).group(1))
    put("val.M4.worst", re.search(r"worst \|diff\|/tol ([\d.]+)", d["M4"]).group(1))
    m = re.search(r"lower tail rel err ([\d.e+-]+), upper side abs err ([\d.e+-]+); quantile: rel err of its tail probability ([\d.e+-]+)", d["M7"])
    for key, v in zip(("low", "up", "q"), m.groups()):
        a, b = v.split("e-")
        put(f"val.M7.{key}", f"{a}\\times 10^{{-{int(b)}}}")


def d3():
    """Numbers of the earlier d = 3 study, read from three_input_study/results/report.txt (its own wording, one line each)."""
    t = open(os.path.join(ROOT, "three_input_study", "results", "report.txt")).read()
    m = re.search(r"d_denom is ([\d.]+)% \(SUM\) and ([\d.]+)% \(CONTRAST\) at rho_S=0.2", t)
    put("d3.flat.sum", m.group(1) + "\\,\\%")
    put("d3.flat.con", m.group(2) + "\\,\\%")
    m = re.search(r"d_denom=([\d.]+) \(SUM\) and ([\d.]+) \(CONTRAST\) at rho_S=0.3", t)
    put("d3.narrow.sum", m.group(1))
    put("d3.narrow.con", m.group(2))
    m = re.search(r"Over the (\d+) valid rows where EXACT_UNCAL is undefined \(eps=0.01\), \w+ is closer to GOLD than EXACT_CAL in (\d+)%", t)
    put("d3.closer.n", m.group(1))
    put("d3.closer", m.group(2) + "\\,\\%")
    m = re.search(r"K=0/1/2\):\* rho_S=0.1: ([\d.]+)/([\d.]+)/([\d.]+);.*?0.4: ([\d.]+)/([\d.]+)/([\d.]+)\.", t)
    for key, v in zip(("r1k0", "r1k1", "r1k2", "r4k0", "r4k1", "r4k2"), m.groups()):
        put(f"d3.taylor.{key}", v)


def design():
    c = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    g = c["numerics"]
    put("cfg.M", "2^{20}" if g["M"] == 2 ** 20 else g["M"])
    put("cfg.R", g["R"])
    put("cfg.snodes", g["s_nodes"])
    put("cfg.gl", g["x_gl"])
    put("cfg.starts", thou(g["mind"]["starts"]))
    put("cfg.floor", c["D_floor"])
    n = c["network"]
    put("cfg.a", n["a_prev"])
    put("cfg.b", n["b_side"])
    k, m = 32, 4
    for basis, o in (("pair", 2), ("triple", 3)):
        put(f"coef.{basis}", thou(sum(math.comb(k + 1, r) * m ** r for r in range(o + 1))))
    put("pairs32", math.comb(32, 2))


def theory(ph, H):
    """Numbers of the theory sections (paper/theory.json, paper/semantics.json) and of the post hoc tables built on them."""
    T = json.load(open(os.path.join(HERE, "theory.json")))
    Sm = json.load(open(os.path.join(HERE, "semantics.json")))
    BEL = ("S2", "S3a", "S3b", "S3c", "S3d", "S4")
    # semantics illustration
    for r in Sm["k2"]:
        for sem in ("evidence", "posterior", "marginal"):
            for form in ("cd", "eu"):
                put(f"sem.{sem}.{form}.{r['rho']}.{r['tau']}", f(r[sem][form]))
    rows = []
    for r in Sm["k2"]:
        rows.append([f(r["rho"], 1), f(r["tau"], 1)] + [f(r[s_][c]) for s_ in ("evidence", "posterior", "marginal") for c in ("cd", "eu")])
    SEM = "\\multicolumn{%d}{c}{evidence} & \\multicolumn{%d}{c}{posterior} & \\multicolumn{%d}{c}{marginal}"
    table("sem", "rr" + "rr" * 3, rows, "& & " + SEM % (2, 2, 2) + " \\\\ \\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(l){7-8}\n$\\rho_S$ & $\\sigma_{\\mathrm{obs}}$ & CD & EU & CD & EU & CD & EU", "\\footnotesize", thin=1)
    rows = [[r["k"], f(r["rho"], 2), f(r["tau"], 1)] + [x for s_ in ("evidence", "posterior", "marginal") for x in (f(r[s_]["cd"]), f(r[s_]["eu"]), pc(r[s_]["cd_closer"]))]
            for r in Sm["kbig"]]
    table("sem_big", "rrr" + "rrr" * 3, rows, "& & & " + SEM % (3, 3, 3) + " \\\\ \\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(l){10-12}\n$k$ & $\\rho_S$ & $\\sigma_{\\mathrm{obs}}$ & CD & EU & closer & CD & EU & closer & CD & EU & closer",
          "\\footnotesize\\setlength{\\tabcolsep}{4.5pt}", thin=2)
    big = Sm["kbig"]
    put("sem.big.lo", f(min(min(r[s_]["cd"], r[s_]["eu"]) for r in big for s_ in ("evidence", "marginal")), 2))
    put("sem.big.hi", f(max(max(r[s_]["cd"], r[s_]["eu"]) for r in big for s_ in ("evidence", "marginal")), 2))
    put("sem.big.marg.closer.lo", pc(min(r["marginal"]["cd_closer"] for r in big)))
    put("sem.big.marg.closer.hi", pc(max(r["marginal"]["cd_closer"] for r in big)))
    fl = Sm["flip"]
    SEMS = ("evidence", "posterior", "marginal")
    rows = [[r["model"], r["k"], f(r["rho"], 2), f(r["tau"], 1)] + [x for s_ in SEMS for x in (f(r[s_]["cd"]), f(r[s_]["eu"]), pc(r[s_]["cd_closer"]))] for r in fl]
    table("sem_flip", "lrrr" + "rrr" * 3, rows, "& & & & " + SEM % (3, 3, 3) + " \\\\ \\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(l){11-13}\nmodel & $k$ & $\\rho_S$ & $\\sigma_{\\mathrm{obs}}$ & CD & EU & closer & CD & EU & closer & CD & EU & closer",
          "\\footnotesize\\setlength{\\tabcolsep}{4pt}", thin=3)
    for r in fl:
        for s_ in SEMS:
            for c in ("cd", "eu"):
                put(f"sem.flip.{r['model']}.{r['k']}.{r['tau']}.{s_}.{c}", f(r[s_][c]))
    # the wording of the illustration rests on these orders: at sigma = 1 every model flips; at sigma = .5 the full tensor still does
    assert all(r["evidence"]["cd"] < r["evidence"]["eu"] and r["posterior"]["eu"] < r["posterior"]["cd"] and r["marginal"]["eu"] < r["marginal"]["cd"] for r in fl if r["tau"] == 1.0)
    assert all(r["evidence"]["cd"] < r["evidence"]["eu"] and r["posterior"]["eu"] < r["posterior"]["cd"] and r["marginal"]["eu"] < r["marginal"]["cd"] for r in fl if r["tau"] == 0.5 and r["model"] == "full")
    assert all(r["min_d"] > 0 for r in fl)
    put("sem.flip.mind.lo", f(min(r["min_d"] for r in fl), 2))
    put("sem.flip.mind.hi", f(max(r["min_d"] for r in fl), 2))
    weak = [r for r in fl if r["tau"] == 0.5 and r["model"] == "triple" and r["k"] >= 4]
    put("sem.flip.weak.hi", f(max(abs(r[s_]["cd"] - r[s_]["eu"]) for r in weak for s_ in ("posterior", "marginal")), 3))
    for r in Sm["missing"]:
        put(f"sem.miss.cd.{r['rho']}", f(r["cd"]))
        put(f"sem.miss.eu.{r['rho']}", f(r["eu"]))
    # gap at leading order
    L = T["leading"]
    put("lead.n", L["n"])
    put("lead.lo", f(L["lo"], 2))
    put("lead.hi", f(L["hi"], 2))
    put("lead.small.n", L["small_n"])
    put("lead.small.lo", f(L["small_lo"], 2))
    put("lead.small.hi", f(L["small_hi"], 2))
    put("orth.pair", sci(T["orthogonality"]["pair"], 0))
    cells = {f"{c['type']}/{c['basis']}": c for c in H["H1c"]["cells"]}
    rows = []
    for key in ("SUM/pair", "CONTRAST/pair", "SUM/triple", "CONTRAST/triple"):
        c, c0 = L["cells"][key], cells[key]
        rows.append([key.replace("/", ", "), c["n"], f(c0["beta"], 2), pc(c0["within2"]), pc(c["within2"]), f"{f(c['lo'], 2)}--{f(c['hi'], 2)}", f(c["med"], 2)])
    table("lead", "lrrrrrr", rows, ["type, basis", "points", "$\\beta$ (H1c)", "power law", "closed form", "ratio range", "median ratio"], "\\footnotesize")
    D = T["data_belief"]
    put("db.n", D["n"])
    put("db.lo", f(D["lo"], 2))
    put("db.hi", f(D["hi"], 2))
    # onsets
    P = [r for r in T["pair_onset"]]
    tr = {(r["m"], r["k"]): r for r in T["triple_onset"]}
    rows = []
    for m in (2, 4, 6):
        for k in KS:
            r = [x for x in P if x["m"] == m and x["k"] == k][0]
            t = tr.get((m, k))
            rows.append([m, k, f(r["crude"]), f(r["lower"]), f(min(r["upper"], r["measured"])), f(r["measured"]), f(k * r["measured"], 2),
                         (f(t["corner"]) if t and t["corner"] else DASH), (f(t["measured"]) if t else DASH)])
        if m != 6:
            rows.append("MID")
    table("onset_all", "rrrrrrrrr", rows, "& & \\multicolumn{5}{c}{pair} & \\multicolumn{2}{c}{triple} \\\\ \\cmidrule(lr){3-7}\\cmidrule(l){8-9}\n$m$ & $k$ & first & convex & search & measured & $k\\rho^*$ & corner & measured", "\\footnotesize", thin=1)
    rows = []
    for k in KS:
        r = [x for x in P if x["m"] == 4 and x["k"] == k][0]
        t = tr.get((4, k))
        rows.append([k, f(r["crude"]), f(r["lower"]), f(r["measured"]), (f(t["corner"]) if t and t["corner"] else DASH), (f(t["measured"]) if t else DASH)])
    table("onset", "rrrrrr", rows, "& \\multicolumn{3}{c}{pair basis} & \\multicolumn{2}{c}{triple basis} \\\\ \\cmidrule(lr){2-4}\\cmidrule(l){5-6}\n$k$ & first & convex & measured & corner & measured", "\\footnotesize")
    for r in P:
        if r["m"] == 4:
            put(f"on.pair.{r['k']}.lower", f(r["lower"]))
            put(f"on.pair.{r['k']}.meas", f(r["measured"]))
    for k in (KS[0], KS[-1]):  # the explicit sufficient condition of Proposition 8(a), m = 4: Theta = m (m + 2)
        put(f"on.arc.{k}", f(6 / math.pi * math.asin(1 / (k * 24)), 4))
    for r in T["pair_onset_large"]:
        put(f"on.conv.kr.{r['k']}", f(r["k_rho"], 3))
    g = T["pair_onset_gap"]
    put("on.gap", pc(g["max_rel_measured"], 1))
    put("on.gap4", pc(g["max_rel_measured_m4"], 1))
    a = [r for r in P if r["m"] == 4 and r["k"] == 32][0]
    put("on.atoms.k32", f(sorted(a["atoms"])[1], 2))
    put("on.atoms.k32b", f(sorted(a["atoms"])[2], 2))
    put("on.kr.pair.lo", f(min(r["k"] * r["measured"] for r in P if r["m"] == 4 and r["k"] >= 4), 2))
    put("on.kr.pair.hi", f(max(r["k"] * r["measured"] for r in P if r["m"] == 4 and r["k"] >= 4), 2))
    c4 = [r for r in T["triple_onset"] if r["m"] >= 4]
    put("cor.min", f(T["corner_formula"]["min_balanced_pair"], 2))
    put("cor.err", sci(T["corner_formula"]["err"], 0))
    put("on.tri.gap", pc(max(abs(r["corner"] / r["measured"] - 1) for r in c4), 1))
    put("on.tri.grid.n", T["triple_corner_grid"]["n"])
    put("on.tri.grid.err", sci(T["triple_corner_grid"]["max_abs"], 0))
    # E_q[D]
    E = T["eqd"]
    put("eqd.n", E["n"])
    put("eqd.rel", pc(E["max_rel"], 1))
    put("eqd.s3c", f(E["pair_s3c_min"], 2))
    put("eqd.triple", f(-E["triple_min"], 0))
    # noise extremes
    Z = T["noise_z"]
    rows = [[r["basis"], r["k"], f(r["measured"], 1), f(r["z_corner"], 1), f(r["z_cube"], 1), pc(r["boundary"]), f(r["union"], 1)] for r in Z]
    table("noise_z", "lrrrrrr", rows, ["basis", "$k$", "$z$ measured", "$z$ corners", "$z$ cube", "on the boundary", "$\\sqrt{2k\\ln 2}$"], "\\footnotesize", thin=1)
    z32 = [r for r in Z if r["basis"] == "pair" and r["k"] == 32][0]
    put("nz.pair32.cube", f(z32["z_cube"], 1))
    put("nz.pair32.corner", f(z32["z_corner"], 1))
    put("nz.pair32.meas", f(z32["measured"], 1))
    pr = [r for r in Z if r["basis"] == "pair" and r["k"] >= 4]
    put("nz.bound.lo", pc(min(r["boundary"] for r in pr)))
    put("nz.bound.hi", pc(max(r["boundary"] for r in pr)))
    put("nz.var", f(T["noise_var"]["ratio"], 2))
    Kc = lambda k, o: k * 24 + math.comb(k, 2) * 24 ** 2 + (math.comb(k, 3) * 24 ** 3 if o == 3 else 0)  # noqa: E731  m = 4, T = 24
    t16 = [r for r in Z if r["basis"] == "triple" and r["k"] == 16][0]
    put("nz.need.pair32", sci(z32["z_cube"] ** 2 * Kc(32, 2), 1))
    put("nz.need.triple16", sci(t16["z_cube"] ** 2 * Kc(16, 3), 1))
    put("nz.ratio.lo", f(min(r["z_cube"] / r["union"] for r in Z), 2))
    put("nz.ratio.hi", f(max(r["z_cube"] / r["union"] for r in Z), 2))
    B = T["tv_bound"]
    put("tvb.n", B["n"])
    put("tvb.sharp", f(B["med_sharp"], 2))
    put("tvb.sharpmax", f(B["max_sharp"], 2))
    put("tvb.old", f(B["med_oldc"], 3))
    put("tvb.gain", f(B["med_gain"], 1))
    # truncation, clipping
    Tr = ph["truncation"]
    rows = []
    for basis in BASES:
        for b in BEL:
            r = Tr[f"{b}_{basis}"]
            rows.append([basis, b, thou(r["n"]), f(r["denom"], 4), f(r["eu"], 4), f(r["cd"], 4), f(r["ratio"], 0 if r["ratio"] >= 100 else 1 if r["ratio"] >= 1 else 2),
                         pc(r["eu_closer"]), pc(r["ec_beats_eu"])])
            for key in ("denom", "eu", "cd"):
                put(f"tr.{b}.{basis}.{key}", f(r[key], 4))
            put(f"tr.{b}.{basis}.ratio", f(r["ratio"], 0 if r["ratio"] >= 10 else 1 if r["ratio"] >= 1 else 2))
            put(f"tr.{b}.{basis}.euc", pc(r["eu_closer"]))
            put(f"tr.{b}.{basis}.ec", pc(r["ec_beats_eu"]))
        if basis == "pair":
            rows.append("MID")
    table("trunc", "llrrrrrrr", rows, ["basis", "belief", "evaluations", "$d_{\\mathrm{denom}}$", "$\\mathrm{TV}(p_{\\mathrm{EU}},p_{\\mathrm{G}})$", "$\\mathrm{TV}(p_{\\mathrm{CD}},p_{\\mathrm{G}})$"]
          + mc("ratio") + ["EU closer", "EC beats EU"], "\\footnotesize", thin=1)
    put("clip.max", pc(max(r["ec_beats_eu"] for r in Tr.values())))
    put("clip.min", pc(min(r["ec_beats_eu"] for r in Tr.values())))
    for e in ("0.1", "0.01", "0.001"):
        put(f"clip.dneg.{e}", f(ph["clip_eps"][e], 3))
    put("clip.S4.eu", f(Tr["S4_pair"]["eu"], 4))
    put("clip.S4.ec", f(Tr["S4_pair"]["ec"], 3))
    rows = []
    for basis in BASES:
        for b in BEL:
            r = ph["past_onset"][f"{b}_{basis}"]
            rows.append([basis, b, thou(r["n"]), pc(r["share"]), f(r["d_total"], 3), f(r["d_dc"], 3), f(r["ec"], 3), pc(r["cd_closer"])])
        if basis == "pair":
            rows.append("MID")
    table("past", "llrrrrrr", rows, ["basis", "belief", "evaluations", "share undefined", "$d_{\\mathrm{total}}$", "$d_{\\mathrm{dc}}$", "$d_{\\mathrm{model,cal}}$", "CD closer than EC"], "\\footnotesize", thin=1)
    po = ph["past_onset"]
    put("past.pair.lo", f(min(po[f"{b}_pair"]["d_total"] for b in BEL if b != "S4"), 2))
    put("past.pair.hi", f(max(po[f"{b}_pair"]["d_total"] for b in BEL if b != "S4"), 2))
    put("past.dc.lo", f(min(po[f"{b}_pair"]["d_dc"] for b in BEL), 3))
    put("past.dc.hi", f(max(po[f"{b}_pair"]["d_dc"] for b in BEL), 3))
    # selection, floor, narrow belief
    Se = ph["selection"]
    put("sel.cv", f(Se["max_cv"], 2))
    put("sel.dmax", f(Se["max_d_denom"], 2))
    put("sel.q99", f(Se["q99_d_denom"], 3))
    put("sel.cv.all", f(Se["all_m"]["max_cv"], 3))
    put("sel.dmax.all", f(Se["all_m"]["max_d_denom"], 3))
    put("sel.k32.pair", pc(Se["s2_share_defined"]["pair_32"]))
    cut = ph["support_cut"]
    in32 = [v for k_, v in cut["inside"].items() if k_.endswith("_32")]
    put("cut.in.lo", f(min(in32), 2))
    put("cut.in.hi", f(max(in32), 2))
    put("cut.drop", sci(cut["max_dropped"], 0))
    put("cut.sign.2", pc(cut["sign_change"]["2"]))
    put("cut.sign.32", pc(cut["sign_change"]["32"]))
    put("sel.d5.s2", Se["d5"]["S2"]["reached"])
    put("sel.d5.s2n", Se["d5"]["S2"]["cells"])
    put("sel.d5.all", sum(v["reached"] for v in Se["d5"].values()))
    put("sel.d5.alln", sum(v["cells"] for v in Se["d5"].values()))
    put("floor.shift", f(ph["floor_max_shift"], 3))
    put("narrow.pair4", f(ph["narrow_by_basis"]["pair_4"], 4))
    put("narrow.triple4", f(ph["narrow_by_basis"]["triple_4"], 4))
    put("narrow.triple2", f(ph["narrow_by_basis"]["triple_2"], 4))
    # the e2e loss sums (c_i - t_i)^2 over rows 1..m and three beliefs, with targets from 2^16 samples: its noise level
    put("e2e.noise", sci(3 * 4 / 2 ** 16, 1))
    put("e2e.targets", 3 * 4)
    coefs = lambda k, o: sum(math.comb(k + 1, r) * 4 ** r for r in range(o + 1))  # noqa: E731
    put("e2e.coef.lo", coefs(2, 2))  # pair, k = 2, L = 1
    put("e2e.coef.hi", thou(coefs(8, 3) + 3 * coefs(2, 3)))  # triple, k = 8, then three triple neurons with k = 2
    for h in ("H1a", "H2a", "H2b_population"):  # D does not see the output coupling, so SUM and CONTRAST cells coincide for negativity
        put(f"{h}.distinct", int(V[f"{h}.n"]) // 2)
    put("e2e.L4", sci(ph["training"]["e2e4_adam"]["median_val_loss_before"], 1))


def regression():
    """The denominator-free message (paper/regression.py): post hoc."""
    R = pd.DataFrame(json.load(open(os.path.join(HERE, "regression.json")))["rows"])
    med = R.groupby("k")[["cd", "eu", "reg1", "reg2"]].median()
    table("reg", "rrrrr", [[k] + [f(v) for v in r] for k, r in zip(med.index, med.values)], ["$k$", "$p_{\\mathrm{CD}}$", "$p_{\\mathrm{EU}}$", "additive features", "pair features"], "\\footnotesize")
    put("reg.n", len(R))
    put("reg.share", pc((R.reg2 < R[["cd", "eu"]].min(axis=1)).mean(), 1))
    put("reg.logn", "2^{20}")
    assert (med.reg2 < med[["cd", "eu"]].min(axis=1)).all()
    assert ((med.reg1 / med.cd - 1).abs() < 0.15).all()  # "additive features do no better than p_CD"


def pilot():
    """The real-data pilot (paper/pilot.py): exploratory, read from its JSON so the build needs no scikit-learn."""
    J = json.load(open(os.path.join(HERE, "pilot.json")))
    d = pd.DataFrame(J["rows"])
    for key, v in (("seeds", J["seeds"]), ("train", thou(J["n_train"])), ("test", thou(J["n_test"])), ("inputs", J["inputs"]), ("tau", J["tau"]), ("samples", J["samples"]), ("floor", J["floor"])):
        put(f"pilot.{key}", v)
    put("pilot.rho.max", f(J["spearman_max"], 2))
    put("pilot.rho.med", f(J["spearman_median"], 2))
    put("pilot.sklearn", J["versions"]["sklearn"])
    fam = d[d.form.isin(["cd", "eu", "reg"])].groupby(["scenario", "form", "basis", "m"])[["mse_clip", "logs", "low_share", "eqd_nonpos"]].mean().reset_index()
    sc = (("full", "all inputs seen"), ("missing", "half missing"), ("noisy", "noisy"))
    span = lambda v, nd=3: f"{f(v.min(), nd)}--{f(v.max(), nd)}"  # noqa: E731
    rows, rows2 = [], []
    for s_, name in sc:
        x = fam[fam.scenario == s_]
        y = {c: x[x.form == c] for c in ("cd", "eu", "reg")}
        boost = d[(d.scenario == s_) & (d.form == "boost")].mse_clip.mean()
        marg = d[(d.scenario == s_) & (d.form == "marginal")].mse_clip.mean()
        rows.append([name] + [span(y[c].mse_clip) for c in ("cd", "eu", "reg")] + [f(boost), f(marg)] + [span(y[c].logs, 2) for c in ("cd", "eu", "reg")])
        for c in y:
            put(f"pilot.{s_}.{c}.lo", f(y[c].mse_clip.min()))
            put(f"pilot.{s_}.{c}.hi", f(y[c].mse_clip.max()))
            put(f"pilot.{s_}.{c}.logs.lo", f(y[c].logs.min(), 2))
            put(f"pilot.{s_}.{c}.logs.hi", f(y[c].logs.max(), 2))
        put(f"pilot.{s_}.boost", f(boost))
        w = d[(d.scenario == s_) & d.form.isin(["cd", "eu"])].pivot_table(index=["seed", "basis", "m"], columns="form", values="mse_clip")
        put(f"pilot.{s_}.win", int((w.cd < w.eu).sum()))
        put("pilot.win.n" if s_ == "full" else f"pilot.{s_}.win.n", len(w))
        c = d[(d.scenario == s_) & (d.form == "cd")]
        rows2.append([name, span(y["cd"].low_share * 100, 1) + "\\,\\%", span(y["cd"].eqd_nonpos * 100, 1) + "\\,\\%", f"{int((w.cd < w.eu).sum())}/{len(w)}", sci(c.mse.max(), 0)])
        put(f"pilot.{s_}.low.lo", pc(y["cd"].low_share.min(), 1))
        put(f"pilot.{s_}.low.hi", pc(y["cd"].low_share.max(), 1))
        put(f"pilot.{s_}.eqd.lo", pc(y["cd"].eqd_nonpos.min(), 1))
        put(f"pilot.{s_}.eqd.hi", pc(y["cd"].eqd_nonpos.max(), 1))
        # worded in the appendix: on average over seeds p_CD leads p_EU in each family exactly when the belief is evidence-type
        m = x.pivot_table(index=["basis", "m"], columns="form", values="mse_clip")
        assert (m.cd < m.eu).all() if s_ != "full" else (m.eu < m.cd).all()
        assert boost < min(y[c_].mse_clip.min() for c_ in y) and marg > max(y[c_].mse_clip.max() for c_ in y)
        # worded in the appendix: REG leads both forms on clipped-mean error except with half missing, on log density only with all seen
        best = lambda col: min(y["cd"][col].min(), y["eu"][col].min()) if col == "mse_clip" else max(y["cd"][col].max(), y["eu"][col].max())  # noqa: E731
        assert (y["reg"].mse_clip.max() < best("mse_clip")) == (s_ != "missing")
        assert (y["reg"].logs.min() > best("logs")) == (s_ == "full")
    M, L = "\\multicolumn{5}{c}{MSE of the clipped mean}", "\\multicolumn{3}{c}{mean clipped log density}"
    table("pilot", "l" * 9, rows, f"& {M} & {L} \\\\ \\cmidrule(lr){{2-6}}\\cmidrule(l){{7-9}}\nscenario & CD & EU & REG & boosting & marginal & CD & EU & REG", "\\scriptsize\\setlength{\\tabcolsep}{3.5pt}")
    table("pilot_fail", "lllrr", rows2, ["scenario", "below the floor", "rows with $\\mathrm{E}_q[D]\\leq0$", "CD ahead of EU", "largest raw MSE of CD"], "\\footnotesize")
    put("pilot.marg", f(d[d.form == "marginal"].mse_clip.mean()))
    put("pilot.sd", f(d.groupby(["scenario", "form", "basis", "m"], dropna=False).mse_clip.std().max()))
    put("pilot.rawmax", sci(d[d.form == "cd"].mse.max(), 0))


def main():
    os.makedirs(GEN, exist_ok=True)
    H, tg, tp, conv, cg, ph = load()
    compute_block()
    design()
    thresholds(H, tg, tp)
    hypotheses(H, conv, ph, cg)
    decomposition(cg)
    posthoc(ph, H)
    theory(ph, H)
    regression()
    pilot()
    energy()
    validation()
    d3()
    with open(os.path.join(GEN, "numbers.tex"), "w") as o:
        o.write("% generated; do not edit\n")
        for k, v in sorted(V.items()):
            o.write(f"\\expandafter\\def\\csname v@{k}\\endcsname{{{v}}}\n")
    json.dump(V, open(os.path.join(GEN, "numbers.json"), "w"), indent=0, sort_keys=True)
    print(len(V), "numbers,", len(glob.glob(os.path.join(GEN, "tab_*.tex"))), "tables")


if __name__ == "__main__":
    main()
