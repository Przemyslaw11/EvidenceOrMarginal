"""Post hoc (not pre-registered) descriptive statistics of the paper, from the merged production results.

    .venv/bin/python paper/posthoc.py        # writes paper/posthoc.json; needs results/merged/*.parquet

Nothing here changes a pre-registered verdict. The parquet files are large and not in git, so the output is committed
and paper/make_numbers.py reads it; paper/check_numbers.py recomputes a sample of it when the parquet files exist.
"""

import glob
import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED = os.path.join(ROOT, "results", "merged")
SUMMARY = os.path.join(ROOT, "results", "summary")
A_PREV, N_EST = 0.6, 1e7  # config.yaml: network.a_prev, experiments.gauss.n_est


def pq(name, cols):
    return pd.read_parquet(os.path.join(MERGED, name + ".parquet"), columns=cols)


def primary(d):
    return d[np.isclose(d.eps, 0.01)]


def single_neuron(out):
    cols = [
        "k",
        "type",
        "basis",
        "seed",
        "rho_S",
        "m",
        "belief",
        "eps",
        "E_q_D",
        "sd_q_D",
        "defined",
        "d_denom",
        "d_total",
        "d_gold_unif",
        "d_model_cal",
    ]
    G = primary(pq("gauss", cols))
    P = primary(pq("population", ["k", "basis", "belief", "m", "rho_S", "eps", "E_q_D", "d_total"]))
    bad = G[G.E_q_D <= 0]
    out["eqd"] = dict(
        n_rows=len(G),
        n_nonpos=len(bad),
        bases=sorted(bad.basis.unique()),
        min_k=int(bad.k.min()),
        beliefs=sorted(bad.belief.unique()),
        min_value=float(G.E_q_D.min()),
        n_nonpos_m4=int((G[G.m == 4].E_q_D <= 0).sum()),
        n_rows_m4=int((G.m == 4).sum()),
        pop_n_nonpos=int((P.E_q_D <= 0).sum()),
        pop_n_rows=len(P),
        pop_min=float(P.E_q_D.min()),
        pop_min_k=int(P[P.E_q_D <= 0].k.min()),
        pop_bases=sorted(P[P.E_q_D <= 0].basis.unique()),
        pair_min=float(G[G.basis == "pair"].E_q_D.min()),
        s2_min=float(G[G.belief == "S2"].E_q_D.min()),
        s2_max=float(G[G.belief == "S2"].E_q_D.max()),
    )
    big = G[G.d_total > 1]
    out["tv_above_one"] = dict(
        n=len(big),
        n_rows=len(G),
        max_d_total=float(G.d_total.max()),
        pop_n=int((P.d_total > 1).sum()),
        pop_rows=len(P),
        share_triple=float((big.basis == "triple").mean()),
        median_abs_eqd=float(big.E_q_D.abs().median()),
    )
    s2 = G[(G.belief == "S2") & (G.m == 4)]
    gap = (s2.d_total - s2.d_gold_unif).abs()
    out["s2_marginal"] = dict(
        median_gap=float(gap.median()),
        max_gap=float(gap.max()),
        sd_over_chi=float(
            (
                pd.read_parquet(
                    os.path.join(MERGED, "gauss.parquet"),
                    columns=["belief", "m", "eps", "chi_D", "sd_q_D"],
                    filters=[("belief", "==", "S2"), ("m", "==", 4), ("eps", "==", 0.01)],
                ).eval("sd_q_D / chi_D")
            ).median()
        ),
    )
    s4 = G[(G.belief == "S4") & (G.m == 4)]
    out["s4"] = dict(
        pair_median_d_total=float(s4[s4.basis == "pair"].d_total.median()),
        pair_max_d_total=float(s4[s4.basis == "pair"].d_total.max()),
        triple_median_d_total=float(s4[(s4.basis == "triple") & (s4.k >= 4) & (s4.rho_S > 0)].d_total.median()),
        triple_max_d_total=float(s4[s4.basis == "triple"].d_total.max()),
    )
    # TV bound of the tilted-belief view: d_denom <= (sup_y ||N/D||_1) * sd_q(D) / (2 |E_q D|)
    d = G[(G.defined == 1) & G.d_denom.notna() & (G.belief != "S1")]
    r = d.d_denom / (0.5 * d.sd_q_D / d.E_q_D.abs())
    out["tv_bound"] = dict(n=int(r.notna().sum()), max_ratio=float(r.max()), median_ratio=float(r.median()))
    # is the constant-denominator output or the clipped conditional closer to GOLD? (seed level, m = 4, eps = 0.01)
    x = G[(G.m == 4) & (G.belief != "S1")].assign(closer=lambda t: t.d_total < t.d_model_cal)
    out["closer"] = {f"{b}_{basis}": float(v) for (b, basis), v in x.groupby(["belief", "basis"]).closer.mean().items()}
    out["closer"]["all"] = float(x.closer.mean())
    out["closer"]["undefined"] = float(x[x.defined != 1].closer.mean())
    out["closer"]["undefined_pair"] = float(x[(x.defined != 1) & (x.basis == "pair")].closer.mean())
    out["closer"]["defined"] = float(x[x.defined == 1].closer.mean())


def curves_stats(out):
    c = pd.read_csv(os.path.join(SUMMARY, "curves_gauss.csv.gz"))
    ok = (c.share_undefined < 0.5) & (c.m == 4) & (c.rho_S > 0) & (c.belief != "S1")
    # which of the two approximations is larger where both exist, by calibration floor
    res = {}
    for eps in (0.1, 0.01, 0.001):
        g = c[
            ok & np.isclose(c.eps, eps) & c.d_denom_med.notna() & c.d_neg_med.notna() & ((c.d_denom_below_share < 0.5) | (c.d_neg_below_share < 0.5))
        ]
        res[str(eps)] = dict(n=len(g), share_neg_larger=float((g.d_neg_med > g.d_denom_med).mean()))
        if eps == 0.01:
            res["by_belief"] = {b: float((x.d_neg_med > x.d_denom_med).mean()) for b, x in g.groupby("belief")}
            res["by_belief_n"] = {b: len(x) for b, x in g.groupby("belief")}
            res["median_ratio"] = {b: float((x.d_neg_med / x.d_denom_med).median()) for b, x in g.groupby("belief")}
    out["denom_vs_neg"] = res
    # mean versus variance error, the points of the mean-variance figure
    g = c[ok & np.isclose(c.eps, 0.01) & (c.basis == "pair") & c.belief.isin(["S2", "S3c", "S4"])]
    g = g[g.relvar_vs_uncal_med.notna() & g.smean_vs_uncal_med.notna()]
    out["mean_var"] = dict(
        n=len(g),
        share_var_larger=float((g.relvar_vs_uncal_med.abs() > g.smean_vs_uncal_med).mean()),
        max_smean_s2=float(g[g.belief == "S2"].smean_vs_uncal_med.max()),
        max_relvar_s2=float(g[g.belief == "S2"].relvar_vs_uncal_med.abs().max()),
        max_smean_s3c_contrast=float(g[(g.belief == "S3c") & (g.type == "CONTRAST")].smean_vs_uncal_med.max()),
        max_relvar_s3c_contrast=float(g[(g.belief == "S3c") & (g.type == "CONTRAST")].relvar_vs_uncal_med.abs().max()),
    )
    # narrow belief: largest seed-median d_denom anywhere it is defined
    g = c[(c.share_undefined < 0.5) & (c.m == 4) & np.isclose(c.eps, 0.01) & (c.belief == "S3d") & (c.basis == "pair")]
    out["narrow"] = {f"k{int(k)}": dict(max_d_denom=float(x.d_denom_med.max()), max_rho=float(x.rho_S.max())) for k, x in g.groupby("k")}
    # noise scale: Var D_hat(y) = (K(y) - 1) / n at independence; corner value of the kernel for degree m = 4
    T = 4 * (4 + 2)
    t = pd.read_csv(os.path.join(SUMMARY, "thresholds_gauss.csv"))
    rows = []
    for basis in ("pair", "triple"):
        for k in (2, 4, 8, 16, 32):
            K1 = k * T + math.comb(k, 2) * T**2 + (math.comb(k, 3) * T**3 if basis == "triple" else 0)
            sd = math.sqrt(K1 / N_EST)
            r = t[(t.basis == basis) & (t.k == k) & (t.m == 4) & (t.belief == "S2") & (t.type == "SUM")].iloc[0]
            x = c[
                (c.basis == basis) & (c.k == k) & (c.m == 4) & (c.type == "SUM") & (c.belief == "S2") & (c.rho_S == 0) & np.isclose(c.eps, 0.01)
            ].iloc[0]
            P = k * 4 + math.comb(k, 2) * 16 + (math.comb(k, 3) * 64 if basis == "triple" else 0)
            rows.append(
                dict(
                    basis=basis,
                    k=k,
                    sd_corner=sd,
                    drop=1 - float(r.minD_at_0),
                    z=(1 - float(r.minD_at_0)) / sd,
                    chi0=float(x.chi_D_med),
                    chi0_pred=math.sqrt(P / N_EST),
                    n_coef_D=P,
                )
            )
    out["noise"] = rows
    out["noise_chi_maxrel"] = max(abs(r["chi0"] / r["chi0_pred"] - 1) for r in rows)
    # chi_D against its leading term rho_S sqrt(k (k - 1) / 2), SUM, pair
    g = c[(c.m == 4) & (c.type == "SUM") & (c.belief == "S2") & (c.basis == "pair") & np.isclose(c.eps, 0.01) & (c.rho_S > 0.04) & (c.rho_S <= 0.2)]
    q = g.chi_D_med / (g.rho_S * np.sqrt(g.k * (g.k - 1) / 2))
    out["chi_lead"] = dict(min=float(q.min()), max=float(q.max()))
    tp = pd.read_csv(os.path.join(SUMMARY, "thresholds_population.csv"))
    tp = tp[(tp.m == 4) & (tp.belief == "S2") & (tp.type == "SUM")]
    out["k_times_onset"] = {f"{r.basis}_{int(r.k)}": float(r.k * r.neg_onset) for r in tp.itertuples()}
    # t copulas are dependent at zero correlation: chi_D at rho_S = 0
    f = pd.read_csv(os.path.join(SUMMARY, "curves_families.csv.gz"))
    g = f[(f.rho_S == 0) & (f.m == 4) & (f.belief == "S2") & (f.type == "EXCH") & (f.basis == "pair") & np.isclose(f.eps, 0.01)]
    out["families_rho0"] = {f"{r.family}_{int(r.k)}": dict(chi=float(r.chi_D_med), minD=float(r.minD_med)) for r in g.itertuples()}


def reframe(out):
    """Which error is largest, what clipping does, what the `defined` filter selects, and the outcomes past the onset."""
    cols = [
        "k",
        "type",
        "basis",
        "belief",
        "rho_S",
        "seed",
        "m",
        "eps",
        "defined",
        "d_denom",
        "d_model_uncal",
        "d_model_cal",
        "d_total",
        "d_dc",
        "d_neg",
        "E_q_D",
        "sd_q_D",
        "minD",
        "n_drop",
        "M",
    ]
    A = pq("gauss", cols)
    A = A[A.belief != "S1"]
    G = primary(A)
    g4 = G[(G.m == 4) & (G.rho_S > 0)]
    d = g4[g4.defined == 1]
    tr = {}
    for (b, basis), x in d.groupby(["belief", "basis"]):
        ratio = x.d_model_uncal / x.d_denom
        tr[f"{b}_{basis}"] = dict(
            n=len(x),
            denom=float(x.d_denom.median()),
            model=float(x.d_model_uncal.median()),
            ratio=float(ratio.median()),
            model_bigger=float((x.d_model_uncal > x.d_denom).mean()),
            cd=float(x.d_total.median()),
            eu=float(x.d_model_uncal.median()),
            ec=float(x.d_model_cal.median()),
            eu_closer=float((x.d_model_uncal < x.d_total).mean()),
            ec_beats_eu=float((x.d_model_cal < x.d_model_uncal).mean()),
        )
    out["truncation"] = tr
    # clipping: how much the floor eps matters, on defined rows, pair, flat belief
    e = A[(A.m == 4) & (A.rho_S > 0) & (A.defined == 1) & (A.belief == "S2") & (A.basis == "pair")]
    out["clip_eps"] = {str(k_): float(x.d_neg.median()) for k_, x in e.groupby("eps")}
    # past the onset: only d_total and d_dc exist
    u = g4[g4.defined != 1]
    out["past_onset"] = {
        f"{b}_{basis}": dict(
            n=len(x),
            share=float(len(x) / len(g4[(g4.belief == b) & (g4.basis == basis)])),
            d_total=float(x.d_total.median()),
            d_dc=float(x.d_dc.median()),
            ec=float(x.d_model_cal.median()),
            cd_closer=float((x.d_total < x.d_model_cal).mean()),
        )
        for (b, basis), x in u.groupby(["belief", "basis"])
    }
    # the `defined` filter: how large can the gap be among the survivors, and how much of the grid survives
    cv = d.sd_q_D / d.E_q_D.abs()
    s2 = G[(G.m == 4) & (G.belief == "S2")]
    out["selection"] = dict(
        max_cv=float(cv.max()),
        max_d_denom=float(d.d_denom.max()),
        q99_d_denom=float(d.d_denom.quantile(0.99)),
        s2_share_defined={f"{b}_{int(k_)}": float((x.defined == 1).mean()) for (b, k_), x in s2.groupby(["basis", "k"])},
    )
    allm = G[(G.rho_S > 0) & (G.defined == 1)]
    out["selection"]["all_m"] = dict(max_cv=float((allm.sd_q_D / allm.E_q_D.abs()).max()), max_d_denom=float(allm.d_denom.max()))
    # `defined` under a Beta belief is relative to the protocol's support cut, not to the open cube
    b = d[d.belief.str.startswith("S3")]
    rng = np.random.default_rng(20261105)
    inside = {}
    for name, ab in (("S3a", [(3, 3)]), ("S3b", [(4, 2)]), ("S3c", [(4, 2), (2, 4)]), ("S3d", [(6, 6)])):
        for k_ in (16, 32):
            pars = [ab[i % len(ab)] for i in range(k_)]
            Y = np.stack([rng.beta(a_, b_, 1 << 20) for a_, b_ in pars], 1)
            lq = sum(stats.beta.logpdf(Y[:, i], *pars[i]) for i in range(k_))
            lmax = sum(stats.beta.logpdf((a_ - 1) / (a_ + b_ - 2), a_, b_) for a_, b_ in pars)
            inside[f"{name}_{k_}"] = float((lq >= lmax + math.log(1e-6)).mean())
    out["support_cut"] = dict(
        inside=inside,
        max_dropped=float((b.n_drop / b.M).max()),
        sign_change={f"{int(k_)}": float((x.minD < 0).mean()) for k_, x in b[b.belief == "S3d"].groupby("k")},
    )
    t = pd.read_csv(os.path.join(SUMMARY, "thresholds_gauss.csv"))
    t = t[(t.m == 4) & (t.belief != "S1")]
    out["selection"]["d5"] = {
        b: dict(cells=len(x), reached=int(np.isfinite(x.denom_5pct).sum()), reached1=int(np.isfinite(x.denom_1pct).sum()))
        for b, x in t.groupby("belief")
    }
    # the floor: share of flat-belief rows with min D above it; D_floor itself is frozen, this only shows how little it decides
    out["floor"] = {
        str(fl): {f"{b}_{int(k_)}": float((x.minD >= fl).mean()) for (b, k_), x in s2.groupby(["basis", "k"])} for fl in (0.0, 0.01, 0.05, 0.1)
    }
    out["floor_max_shift"] = max(abs(out["floor"]["0.0"][c] - out["floor"]["0.05"][c]) for c in out["floor"]["0.0"])
    # narrow belief, seed medians, by basis and degree
    c = pd.read_csv(os.path.join(SUMMARY, "curves_gauss.csv.gz"))
    n = c[(c.belief == "S3d") & np.isclose(c.eps, 0.01) & (c.share_undefined < 0.5)]
    out["narrow_by_basis"] = {f"{b}_{int(m)}": float(x.d_denom_med.max()) for (b, m), x in n.groupby(["basis", "m"])}
    out["narrow_rows_pair_m4"] = float(d[(d.belief == "S3d") & (d.basis == "pair")].d_denom.max())


def lkj(out):
    L = primary(pq("lkj", ["k", "basis", "belief", "m", "eps", "eta", "defined"]))
    L = L[(L.m == 4) & (L.belief == "S2")]
    out["lkj"] = {f"{int(k)}_{int(e)}_{b}": dict(n=len(x), defined=int(x.defined.sum())) for (k, e, b), x in L.groupby(["k", "eta", "basis"])}


def convergence(out):
    v = pd.read_csv(os.path.join(SUMMARY, "convergence_violations.csv"))
    rel = v.change / v.value_base.abs()
    w = v.loc[v.change.idxmax()]
    out["h4b"] = dict(
        median_rel_by_metric={m: float(x) for m, x in rel.groupby(v.metric).median().items()},
        median_rel_tv_except_denom=float(rel[v.metric != "d_denom"].median()),
        share_triple=float((v.basis == "triple").mean()),
        share_d_denom=float((v.metric == "d_denom").mean()),
        worst=dict(
            metric=w.metric,
            k=int(w.k),
            basis=w.basis,
            belief=w.belief,
            m=int(w.m),
            rho=float(w.rho_S),
            value=float(w.value),
            base=float(w.value_base),
            change=float(w.change),
        ),
    )


def network(out):
    H = json.load(open(os.path.join(SUMMARY, "hypotheses.json")))
    N = primary(pd.read_parquet(os.path.join(MERGED, "network.parquet")))
    ch, n1 = N[N.table == "chain"], N[N.table == "neuron1"]
    d = ch[ch.run == "direct"]
    # depth
    dd = d[d.belief.isin(["S2", "S3c", "S4"])].assign(dv=lambda t: (t.var_gold - 1 / 12).abs())
    dep = {}
    for (b, basis), g in dd[dd.rho_S > 0].groupby(["belief", "basis"]):
        dep[f"{b}_{basis}"] = {
            str(int(L)): dict(
                d_total=float(x.d_total.median()),
                dv=float(x.dv.median()),
                var_cd=float(x.var_cd.median()),
                var_gold=float(x.var_gold.median()),
                floor=float(x.d_total_floor.median()),
            )
            for L, x in g.groupby("layer")
        }
    out["depth"] = dep
    out["depth_a6"] = A_PREV**6
    out["layer_minD"] = {str(int(L)): float(x.neuron_minD.median()) for L, x in d.groupby("layer")}
    out["layer_minD_low"] = float(d[d.layer >= 2].neuron_minD.min())
    a = pd.DataFrame(H["H5a"]["cells"]).assign(below=lambda t: t.ci.apply(lambda v: v[1] < 1))
    cnt = a.groupby(["belief", "basis"])[["sig"]].sum().sig
    bel = a.groupby(["belief", "basis"]).below.sum()
    med = a.groupby(["belief", "basis"]).ratio.median()
    out["h5a_split"] = {
        f"{b}_{basis}": dict(
            sig=int(cnt[(b, basis)]), below=int(bel[(b, basis)]), ratio=float(med[(b, basis)]), n=int(((a.belief == b) & (a.basis == basis)).sum())
        )
        for b, basis in cnt.index
    }
    # the 11 significant cells: size of d_total relative to the Monte Carlo noise floor
    keys = ["k", "rho_S", "basis", "belief"]
    m1 = d[d.layer == 1].groupby(keys)[["d_total", "d_total_floor"]].median()
    m4 = d[d.layer == 4].groupby(keys)[["d_total", "d_total_floor"]].median()
    sig = a[a.sig].set_index(keys)
    out["h5a_sig"] = dict(
        n=len(sig),
        max_d_total_L1=float(m1.loc[sig.index].d_total.max()),
        max_d_total_L4=float(m4.loc[sig.index].d_total.max()),
        max_over_floor_L4=float((m4.loc[sig.index].d_total / m4.loc[sig.index].d_total_floor).max()),
        n_rho0=int((a[a.sig].rho_S == 0).sum()),
    )
    # training records (one row per trained model: S2 rows of the neuron-1 table)
    r = n1[(n1.belief == "S2") & ~n1.run.isin(["direct", "flat_init"])].copy()
    r["stem"] = r.run.str.replace("_direct", "").str.replace("_flat", "")
    r["ini"] = np.where(r.run.str.endswith("flat"), "flat", "direct")
    r["rel_test"] = (r.loss_test_after - r.loss_test_before) / r.loss_test_before.abs()
    tr = {}
    base = n1[(n1.belief == "S2") & (n1.run == "direct")].set_index(["k", "rho_S", "basis", "seed"])
    for stem, g in r[r.ini == "direct"].groupby("stem"):
        a2 = g.set_index(["k", "rho_S", "basis", "seed"])
        dP = (a2.P_Dneg_u - base.P_Dneg_u).dropna()
        act = base.P_Dneg_u > 0
        tr[stem] = dict(
            n=len(g),
            kept_initial=float((g.coef_dist == 0).mean()),
            median_rel_test=float(g.rel_test.median()),
            share_test_reduced=float((g.rel_test < 0).mean()),
            median_coef_dist=float(g.coef_dist.median()),
            median_dP_where_neg=float(dP[act.reindex(dP.index)].median()),
            median_minD_change=float((a2.minD - base.minD).dropna().median()),
            median_val_loss_before=float(g.loss_val_before.median()),
            n_nonfinite=int(g.n_nonfinite.sum()),
            n_cross=int(g.n_cross.sum()),
        )
    out["training"] = tr
    fl = {}
    for stem, g in r[r.ini == "flat"].groupby("stem"):
        fl[stem] = dict(P=float(g.P_Dneg_u.median()), chi=float(g.chi_D.median()), minD=float(g.minD.median()))
    out["flat_after"] = fl
    fi = n1[(n1.belief == "S2") & (n1.run == "flat_init")]
    out["flat_init"] = dict(max_d_denom=float(fi.d_denom.max()), max_P=float(fi.P_Dneg_u.max()))
    # verdict cells by loss (no decision changes; the pre-registered verdict is over all cells)
    by = {}
    for h in ("H5b", "H5c"):
        c = pd.DataFrame(H[h]["cells"]).assign(opp=lambda t: t.ci.apply(lambda v: v[0] > 0))
        c["loss"] = c.run.str.split("_").str[0].str.replace("e2e4", "e2e")
        by[h] = {l: dict(n=len(x), sig=int(x.sig.sum()), opp=int(x.opp.sum())) for l, x in c.groupby("loss")}
        by[h + "_ks"] = sorted(int(v) for v in c.k.unique())
    c = pd.DataFrame(H["H5d"]["cells"]).assign(opp=lambda t: t.ci.apply(lambda v: v[0] > 0))
    by["H5d"] = {g: {str(l): dict(n=len(x), sig=int(x.sig.sum()), opp=int(x.opp.sum())) for l, x in c.groupby(g)} for g in ("L", "k", "basis")}
    out["by_loss"] = by
    # end-to-end training: change of d_total at layer L on training and held-out beliefs
    idx = ["k", "rho_S", "basis", "seed", "belief"]
    e2e = {}
    for L in (1, 2, 3, 4):
        b = d[d.layer == L].set_index(idx).d_total
        for opt in ("adam", "lbfgs"):
            a3 = ch[(ch.run == f"e2e{L}_{opt}_direct") & (ch.layer == L)].set_index(idx).d_total
            x = (a3 - b).dropna().reset_index()
            e2e[f"{L}_{opt}"] = dict(
                train=float(x[x.belief.isin(["S2", "S3c", "S4"])].d_total.median()),
                held=float(x[x.belief.isin(["S3a", "S3b", "S3d"])].d_total.median()),
                S2=float(x[x.belief == "S2"].d_total.median()),
                S3c=float(x[x.belief == "S3c"].d_total.median()),
                S4=float(x[x.belief == "S4"].d_total.median()),
            )
    out["e2e_beliefs"] = e2e


def commits(out):
    rows = {}
    for f in sorted(glob.glob(os.path.join(MERGED, "*_meta.csv"))):
        d = pd.read_csv(f)
        rows[os.path.basename(f)[:-9]] = {c[:7]: int(n) for c, n in d.git.value_counts().items()}
    out["commits"] = rows
    fam = pq("families", ["job_id", "family"]).drop_duplicates()
    m = pd.read_csv(os.path.join(MERGED, "families_meta.csv"))[["job_id", "git"]]
    j = fam.merge(m, on="job_id")
    out["t_commits"] = sorted(c[:7] for c in j[j.family.isin(["t3", "t10"])].git.unique())


def main():
    out = {}
    for f in (single_neuron, curves_stats, reframe, lkj, convergence, network, commits):
        f(out)
        print("done", f.__name__)
    json.dump(out, open(os.path.join(ROOT, "paper", "posthoc.json"), "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
