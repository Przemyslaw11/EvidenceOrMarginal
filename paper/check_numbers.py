"""Check every number of the paper against its source; exits non-zero on any failure.

    .venv/bin/python paper/check_numbers.py

1. Regenerates generated/ from the sources and requires identical bytes.
2. Re-applies the pre-registered decision rules to hypotheses.json and compares the hypothesis table, row by row.
3. Compares the generated tables and macros with the reports in results/, which other code wrote.
4. Recomputes post hoc counts from the merged parquet files, by a separate route, when those files exist.
5. Scans main.tex: every \\V key exists, and no decimal or percentage is typed outside the whitelist of design constants.
6. Confirms PREREGISTRATION.md is byte-identical to the frozen commit.
7. Scans main.tex and every generated file it includes for repository detail: the paper must not name files, commits or code.
8. Recomputes the headline numbers of the post hoc additions (flip, regression message, pilot, support cut, convex bound) by a separate route.
"""
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import make_numbers  # noqa: E402

GEN = os.path.join(HERE, "generated")
S = os.path.join(ROOT, "results", "summary")
bad = []
n_ok = 0


def check(cond, what):
    global n_ok
    if cond:
        n_ok += 1
    else:
        bad.append(what)


def regenerate():
    paths = sorted(glob.glob(os.path.join(GEN, "*")))
    before = {p: open(p, "rb").read() for p in paths}
    make_numbers.V.clear()
    make_numbers.main()
    for p in paths:
        check(open(p, "rb").read() == before[p], f"regenerated file differs: {os.path.basename(p)}")
    check(sorted(glob.glob(os.path.join(GEN, "*"))) == paths, "regeneration changed the file list")


def rule(sig, n):
    """Section 6 aggregation rule."""
    if n < 3:
        return "inconclusive"
    if sig >= 2 / 3 * n:
        return "supported"
    if n - sig >= 2 / 3 * n:
        return "not supported"
    return "inconclusive"


def hypotheses():
    H = json.load(open(os.path.join(S, "hypotheses.json")))
    want = {}
    for h in ("H1a", "H1b", "H2a", "H2b_population", "H2b_gauss", "H5a", "H5b", "H5c", "H5d"):
        c = H[h]["cells"]
        sig = sum(x["sig"] for x in c)
        check(sig == H[h]["significant"], f"{h}: significant count")
        want[h] = rule(sig, len(c))
    s = [x["within2"] for x in H["H1c"]["cells"]]
    cell = ["supported" if v >= 0.9 else "not supported" if v < 0.5 else "inconclusive" for v in s]
    check(cell == [x["verdict"] for x in H["H1c"]["cells"]], "H1c cell verdicts")
    # the pre-registration rules H1c per (type, basis); analysis/hypotheses.py at the freeze commit combines the cells by unanimity
    want["H1c"] = cell[0] if len(set(cell)) == 1 else "inconclusive"
    v = H["H2c"]["share"]
    want["H2c"] = "supported" if v >= 0.8 else "not supported" if v <= 0.5 else "inconclusive"
    for h in ("H3a", "H3b"):
        c = H[h]["cells"]
        check(sum(x["inside"] for x in c) == H[h]["n_inside"] and sum(x["outside"] for x in c) == H[h]["n_outside"], f"{h}: counts")
        want[h] = "not supported" if any(x["outside"] for x in c) else "supported" if all(x["inside"] for x in c) else "inconclusive"
    for x in H["H3a"]["cells"]:
        check(x["inside"] == (-0.05 <= x["ci"][0] and x["ci"][1] <= 0.05), f"H3a cell {x['family']} {x['k']}: inside")
    v = H["H3c"]["share"]
    want["H3c"] = "supported" if v >= 0.8 else "not supported" if v < 0.5 else "inconclusive"
    v = H["H4a"]["share"]
    want["H4a"] = "supported" if v >= 0.9 else "not supported" if v < 0.5 else "inconclusive"
    conv = pd.read_csv(os.path.join(S, "convergence.csv"))
    want["H4b"] = "not supported" if conv.n_violations.sum() > 0 else "supported"
    for h, w in want.items():
        check(w == H[h]["verdict"], f"{h}: rule gives {w}, file says {H[h]['verdict']}")
    # table rows, in file order, against the file
    t = open(os.path.join(GEN, "tab_hyp.tex")).read()
    rows = re.findall(r"^(H\d[a-d]) & (.*?) & \\emph\{(.*?)\} & (.*?) \\\\$", t, re.M)
    keys = ["H1a", "H1b", "H1c", "H2a", "H2b_population", "H2b_gauss", "H2c", "H3a", "H3b", "H3c", "H4a", "H4b", "H5a", "H5b", "H5c", "H5d"]
    check(len(rows) == len(keys) == len(H), "hypothesis table has every outcome")
    for r, k in zip(rows, keys):
        check(r[0] == k[:3] and r[2] == H[k]["verdict"], f"table row {r[0]}: verdict")
        if "significant" in H[k]:
            n = len(H[k]["cells"])
            check(r[3].startswith(f"{H[k]['significant']}/{n}"), f"table row {k}: count")
    check(set(w for _, _, w, _ in rows) <= {"supported", "not supported", "inconclusive"}, "verdict words")


def tex_rows(name):
    t = open(os.path.join(GEN, f"tab_{name}.tex")).read()
    rows = [[c.strip() for c in r.split(" & ")] for r in re.findall(r"^(.*?) \\\\$", t, re.M)][1:]
    for a, b in zip(rows, rows[1:]):  # the tables leave repeated labels blank
        for i, c in enumerate(b):
            if c:
                break
            b[i] = a[i]
    return rows


def reports():
    rep = open(os.path.join(ROOT, "results", "report.txt")).read()
    norm = lambda c: c.replace("—", "---").replace("–", "--").strip()  # noqa: E731
    md = {tuple(c[:3]): [norm(x) for x in c[3:]] for c in
          ([x.strip() for x in line.strip("|").split("|")] for line in rep.splitlines() if re.match(r"\| (SUM|CONTRAST) \| (pair|triple) \|", line))}
    rows = tex_rows("thr_full")
    last = None
    n = 0
    for r in rows:
        if r[0] in ("SUM", "CONTRAST"):
            n += 1
            check(md.get(tuple(r[:3])) == r[3:], f"thresholds differ from the report: {r[:3]}")
    check(n == len(md) == 20, "threshold rows")
    V = make_numbers.V
    for pat, key in ((r"\*\*(?:not )?supported\*\* \(4/4 cells", None),):
        check(re.search(pat, rep), "report H1a")
    for txt, key in (("21/48 CIs inside, 27 entirely outside", "H3a"), ("73/236 inside, 156 outside", "H3b"), ("11/72; 51 cells", "H5a"),
                     ("5/18 significant, 1 significant", "H5b"), ("24/78 significant, 19 significant", "H5c"), ("27/192 significant, 84 significant", "H5d")):
        a, b = (int(x) for x in re.findall(r"\d+", txt)[:2])
        if key in ("H3a", "H3b"):
            check((int(V[f"{key}.in"]), int(V[f"{key}.n"])) == (a, b), f"{key} vs report")
        else:
            check((int(V[f"{key}.sig"]), int(V[f"{key}.n"])) == (a, b), f"{key} vs report")
        check(txt in rep, f"report lacks {txt}")
    for fam in ("clayton", "gumbel", "t10", "t3"):
        m = re.search(rf"\| {fam} \| \d+ \| .*? \| \d+ \| \d+ \| ([\d.]+) \|", rep)
        check(m and m.group(1) == V[f"fam.{fam}.ratio"], f"family ratio {fam}")
    m = re.search(r"gives (\d+) of (\d+) \(k = 8 and 32", rep)
    check(int(m.group(1)) == int(V["H4b.viol"].replace("{,}", "")) and int(m.group(2)) == int(V["H4b.n2"].replace("{,}", "")), "H4b vs report")
    m = re.search(r"S2 ([\d.]+), S3c ([\d.]+), S4 ([\d.]+);", rep)
    check(m.groups() == (V["H5a.ratio.S2"], V["H5a.ratio.S3c"], V["H5a.ratio.S4"]), "H5a ratios vs report")
    # energy: every row of both full tables against the energy report
    en = open(os.path.join(ROOT, "results", "energy", "report.txt")).read()
    secs = re.split(r"^## (float\d\d)$", en, flags=re.M)
    for dt, body in zip(secs[1::2], secs[2::2]):
        md = {}
        for line in body.splitlines():
            c = [x.strip() for x in line.strip("|").split("|")]
            if len(c) >= 12 and c[1].isdigit():
                md[(c[0], c[1], c[2])] = (c[4].split()[0], c[5].split()[0])
        rows = tex_rows(f"energy_{dt}")
        check(len(rows) == len(md) == 24, f"energy rows {dt}")
        for r in rows:
            key = (r[2], r[0], r[1].replace("{,}", ""))
            a, b = md[key]
            check(abs(float(a) / float(r[4]) - 1) < 6e-3 and abs(float(b) / float(r[5]) - 1) < 6e-3, f"energy {dt} {key}")


def parquet():
    p = os.path.join(ROOT, "results", "merged", "gauss.parquet")
    if not os.path.exists(p):
        print("parquet files absent: post hoc recomputation skipped")
        return
    V = make_numbers.V
    n = lambda k: int(V[k].replace("{,}", ""))  # noqa: E731
    G = pd.read_parquet(p, columns=["basis", "k", "m", "E_q_D", "d_total"], filters=[("eps", "==", 0.01)])
    check(len(G) == n("ph.eqd.rows"), "rows")
    neg = G[G.E_q_D <= 0]
    check(len(neg) == n("ph.eqd.nonpos"), "E_q[D] <= 0 count")
    check((neg.m == 4).sum() == n("ph.eqd.nonpos4"), "E_q[D] <= 0 count, m = 4")
    check(set(neg.basis) == {"triple"} and neg.k.min() == int(V["ph.eqd.mink"]), "E_q[D] <= 0 cells")
    check((G.d_total > 1).sum() == n("ph.tv1.n"), "d_total > 1 count")
    check(f"{G.d_total.max():.0f}" == V["ph.tv1.max"], "max d_total")
    Pp = pd.read_parquet(os.path.join(ROOT, "results", "merged", "population.parquet"), columns=["E_q_D"], filters=[("eps", "==", 0.01)])
    check((Pp.E_q_D <= 0).sum() == n("ph.eqd.pop.nonpos") and len(Pp) == n("ph.eqd.pop.rows"), "population E_q[D]")
    # headline of the truncation table, straight from the rows
    D = pd.read_parquet(p, columns=["basis", "belief", "d_denom", "d_model_uncal", "d_model_cal"],
                        filters=[("eps", "==", 0.01), ("m", "==", 4), ("defined", "==", 1), ("rho_S", ">", 0)])
    for b, basis, nd in (("S3d", "pair", 0), ("S4", "pair", 1), ("S4", "triple", 2), ("S2", "triple", 2)):
        x = D[(D.belief == b) & (D.basis == basis)]
        check(f"{(x.d_model_uncal / x.d_denom).median():.{nd}f}" == V[f"tr.{b}.{basis}.ratio"], f"truncation ratio {b} {basis}")
    w = D[D.belief != "S1"].groupby(["belief", "basis"]).apply(lambda x: (x.d_model_cal < x.d_model_uncal).mean(), include_groups=False)
    check(f"{100 * w.max():.0f}" in V["clip.max"], "clipping beats the reference, largest share")
    st = [json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "results/merged/*_status.json"))]
    check(sum(s["done"] for s in st) == n("jobs") and not any(s["failed"] for s in st), "jobs")
    v = pd.read_csv(os.path.join(S, "convergence_violations.csv"))
    check(len(v) == n("H4b.viol") and (v.k == 8).sum() == n("H4b.k8") and (v.k == 32).sum() == n("H4b.k32"), "H4b violations")


def additions():
    V = make_numbers.V
    Sm = json.load(open(os.path.join(HERE, "semantics.json")))
    fl = Sm["flip"]
    check(len(fl) == 10 and {(r["model"], r["k"]) for r in fl} == {("full", 3), ("full", 4), ("triple", 3), ("triple", 4), ("triple", 8)}, "flip settings")
    check(all(r["min_d"] > 0 for r in fl), "flip rows are below the onset")
    for r in fl:
        if r["tau"] == 1.0:
            check(r["evidence"]["cd"] < r["evidence"]["eu"] and r["posterior"]["cd"] > r["posterior"]["eu"] and r["marginal"]["cd"] > r["marginal"]["eu"], f"flip at sigma 1: {r['model']} {r['k']}")
            check(V[f"sem.flip.{r['model']}.{r['k']}.1.0.evidence.cd"] == f"{r['evidence']['cd']:.3f}", "flip macro")
    R = json.load(open(os.path.join(HERE, "regression.json")))["rows"]
    check(len(R) == 64 and int(V["reg.n"]) == 64, "regression settings")
    best = sum(r["reg2"] < min(r["cd"], r["eu"]) for r in R)
    check(V["reg.share"] == f"{100 * best / len(R):.1f}\\,\\%", "regression share")
    rows = {int(x[0]): x[1:] for x in tex_rows("reg")}
    for k in (2, 4, 8, 16):
        for i, c in enumerate(("cd", "eu", "reg1", "reg2")):
            v = sorted(r[c] for r in R if r["k"] == k)
            check(f"{(v[7] + v[8]) / 2:.3f}" == rows[k][i], f"regression median k = {k} {c}")
    J = json.load(open(os.path.join(HERE, "pilot.json")))
    P = J["rows"]
    check(J["seeds"] == 10 and len({r["seed"] for r in P}) == 10, "pilot seeds")
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    for sc in ("full", "missing", "noisy"):
        for form in ("cd", "eu", "reg"):
            fam = [mean([r["mse_clip"] for r in P if r["scenario"] == sc and r["form"] == form and (r["basis"], r["m"]) == b]) for b in (("pair", 2), ("pair", 3), ("triple", 2), ("triple", 3))]
            check(V[f"pilot.{sc}.{form}.lo"] == f"{min(fam):.3f}" and V[f"pilot.{sc}.{form}.hi"] == f"{max(fam):.3f}", f"pilot {sc} {form}")
        check(V[f"pilot.{sc}.boost"] == f"{mean([r['mse_clip'] for r in P if r['scenario'] == sc and r['form'] == 'boost']):.3f}", f"pilot {sc} boosting")
        cd = {(r["seed"], r["basis"], r["m"]): r["mse_clip"] for r in P if r["scenario"] == sc and r["form"] == "cd"}
        eu = {(r["seed"], r["basis"], r["m"]): r["mse_clip"] for r in P if r["scenario"] == sc and r["form"] == "eu"}
        check(int(V[f"pilot.{sc}.win"]) == sum(cd[x] < eu[x] for x in cd), f"pilot {sc} wins")
    for k, want in ((2, 6 / math.pi * math.asin(1 / 48)), (32, 6 / math.pi * math.asin(1 / 768))):
        check(V[f"on.arc.{k}"] == f"{want:.4f}", "explicit condition of the first bound")
    Tj = json.load(open(os.path.join(HERE, "theory.json")))["pair_onset_large"]
    check([r["k"] for r in Tj] == [8, 32, 128, 512, 2048] and all(a["k_rho"] < b["k_rho"] for a, b in zip(Tj, Tj[1:])), "convex bound rises")
    check(all(V[f"on.conv.kr.{r['k']}"] == f"{r['k_rho']:.3f}" for r in Tj), "convex bound macros")
    p = os.path.join(ROOT, "results", "merged", "gauss.parquet")
    if os.path.exists(p):
        G = pd.read_parquet(p, columns=["k", "basis", "belief", "m", "rho_S", "defined", "d_denom", "d_total", "d_model_cal", "E_q_D", "sd_q_D", "minD", "n_drop", "M"], filters=[("eps", "==", 0.01)])
        G = G[G.belief != "S1"]
        a = G[(G.rho_S > 0) & (G.defined == 1)]
        check(f"{(a.sd_q_D / a.E_q_D.abs()).max():.3f}" == V["sel.cv.all"] and f"{a.d_denom.max():.3f}" == V["sel.dmax.all"], "selection over all m")
        u = G[(G.m == 4) & (G.defined != 1) & (G.basis == "pair")]
        check(f"{100 * (u.d_total < u.d_model_cal).mean():.0f}\\,\\%" == V["ph.closer.undef.pair"], "past the onset, pair alone")
        b = a[(a.m == 4) & a.belief.str.startswith("S3")]
        c = b[(b.belief == "S3d") & (b.k == 32)]
        check(f"{100 * (c.minD < 0).mean():.0f}\\,\\%" == V["cut.sign.32"], "sign change among defined evaluations")
        check((b.n_drop / b.M).max() < 4e-4, "dropped samples")


ALLOWED = {  # design constants and mathematics, never results
    "0", "0.01", "0.02", "0.03", "0.05", "0.075", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "0.001", "0.003", "0.005", "0.0001",
    "1", "1.5", "2", "4", "5", "10", "1%", "5%", "50%", "80%", "90%", "95%", "10%", "97.5", "97.5%", "2.5", "2.5%", "1.0", "4.0",
}


def scan():
    t = open(os.path.join(HERE, "main.tex")).read()
    t = t[t.index("\\begin{document}"):]
    keys = re.findall(r"\\V\{([^}]*)\}", t)
    for k in keys:
        check(k in make_numbers.V, f"unknown key {k}")
    print(len(keys), "numbers used in main.tex,", len(set(keys)), "distinct")
    x = re.sub(r"%.*$", "", t.replace("\\%", "PCT"), flags=re.M)
    x = re.sub(r"\\(V|ref|eqref|label|cite[tp]?|input|includegraphics|texttt|section|subsection)(\[[^\]]*\])*\{[^}]*\}", " ", x)
    x = x.replace("\\,", "").replace("PCT", "%")
    for m in re.finditer(r"(?<![\w.{^_-])\d+\.\d+%?|(?<![\w.{^_-])\d+%", x):
        check(m.group(0) in ALLOWED, f"typed number {m.group(0)}: ...{x[max(0, m.start() - 40):m.end() + 20]!r}")


REPO = re.compile(r"\.md\b|\.py\b|\.csv\b|commit|\bgit\b|branch|repositor|README|version-controlled|(?-i:PREREGISTRATION)|\bresults/|\bgpu/|\bnetwork/|SeedSequence"
                  r"|\\texttt\{[^}]*\b[0-9a-f]{7,40}\b|\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{7,40}\b|\bjob\b 3\d{6}", re.I)


def repo_detail():
    """A reader meets a study, not a code base: file names, hashes and version-control words stay out of everything the PDF is built from."""
    t = open(os.path.join(HERE, "main.tex")).read()
    n = sorted(set(re.findall(r"\\input\{(generated/[^}]+)\}", t)))
    check("generated/numbers.tex" in n and any("tab_hyp" in x for x in n), "included files")
    for name, x in [("main.tex", t)] + [(f, open(os.path.join(HERE, f if f.endswith(".tex") else f + ".tex")).read()) for f in n]:
        for m in REPO.finditer(x):
            bad.append(f"repository detail in {name}: ...{x[max(0, m.start() - 30):m.end() + 20]!r}")
        check(True, name)


FROZEN_SHA256 = "55b88cae43d32e71e71900b98af8d38e9e526a0fcea2903ec5a6f14940fc3024"  # PREREGISTRATION.md at 8abbf12


def frozen():
    b = open(os.path.join(ROOT, "PREREGISTRATION.md"), "rb").read()
    check(hashlib.sha256(b).hexdigest() == FROZEN_SHA256, "PREREGISTRATION.md changed")
    try:  # a full clone can also compare with the freeze commit itself; an archive export has no history
        a = subprocess.run(["git", "show", "8abbf12:PREREGISTRATION.md"], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        print("freeze commit 8abbf12 not available: git comparison skipped")
        return
    check(a == b, "PREREGISTRATION.md differs from commit 8abbf12")


def main():
    for f in (regenerate, hypotheses, reports, parquet, additions, scan, repo_detail, frozen):
        f()
    print(f"{n_ok} checks passed, {len(bad)} failed")
    for b in bad:
        print("FAIL:", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
