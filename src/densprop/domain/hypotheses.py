"""The pre-registered hypotheses (PREREGISTRATION.md, section 6), evaluated into results/summary/hypotheses.json.

Each hypothesis yields a verdict (supported / not supported / inconclusive), the evaluable cells and the evidence.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from densprop.domain.stats import B_BOOT, BOOT_SEED, boot_idx, median_curve, onset_negative, seed_matrix

LOG2 = math.log(2.0)


def verdict(sig: int, n: int) -> str:
    """Aggregation rule: supported if >= 2/3 significant, not supported if >= 2/3 not significant, else inconclusive."""
    if n < 3:
        return "inconclusive"
    if sig >= 2 / 3 * n:
        return "supported"
    if n - sig >= 2 / 3 * n:
        return "not supported"
    return "inconclusive"


def share_verdict(share: float, hi: float, lo: float) -> str:
    return "supported" if share >= hi else ("not supported" if share < lo else "inconclusive")


def boot_onset(g, B=B_BOOT, idx=None):
    rhos, seeds, V = seed_matrix(g, "minD", defined_col=None)
    idx = boot_idx(V.shape[0], B) if idx is None else idx
    return onset_negative(rhos, median_curve(V)), np.array([onset_negative(rhos, median_curve(V[i])) for i in idx]), seeds


def ci(v: np.ndarray) -> tuple[float, float]:
    return tuple(np.percentile(v, [2.5, 97.5], method="inverted_cdf"))


def prim(df: pd.DataFrame, m: int = 4, eps: float = 0.01) -> pd.DataFrame:
    return df[(df.m == m) & np.isclose(df.eps, eps)]


# ---------------------------------------------------------------- Q1, Q2
def h1a_h2a_h2b_pop(th: pd.DataFrame) -> dict:
    """From the population thresholds (thresholds_population.csv)."""
    th = th[th.belief == "S2"]  # minD does not depend on the belief
    out = {}
    cells = []
    for (typ, basis), cell in th[th.m == 4].groupby(["type", "basis"]):
        g = cell.sort_values("k")
        v = g.neg_onset.to_numpy()
        cells.append(dict(type=typ, basis=basis, k=g.k.tolist(), onset=v.tolist(), sig=bool(np.all(np.diff(v) < 0))))
    s = sum(c["sig"] for c in cells)
    out["H1a"] = dict(verdict=verdict(s, len(cells)), significant=s, cells=cells)
    cells = []
    for (typ, k, basis), g in th.groupby(["type", "k", "basis"]):
        o = {int(r.m): r.neg_onset for r in g.itertuples()}
        if not all(m in o for m in (2, 4, 6)):
            continue
        cells.append(dict(type=typ, k=int(k), basis=basis, onset_m2=o[2], onset_m4=o[4], onset_m6=o[6], sig=bool(o[6] < o[4] < o[2])))
    s = sum(c["sig"] for c in cells)
    out["H2a"] = dict(verdict=verdict(s, len(cells)), significant=s, cells=cells)
    cells = []
    t4 = th[(th.m == 4) & (th.k >= 4)]
    for (typ, k), g in t4.groupby(["type", "k"]):
        o = {r.basis: r.neg_onset for r in g.itertuples()}
        if "pair" in o and "triple" in o:
            cells.append(dict(type=typ, k=int(k), pair=o["pair"], triple=o["triple"], sig=bool(o["triple"] < o["pair"])))
    s = sum(c["sig"] for c in cells)
    out["H2b_population"] = dict(verdict=verdict(s, len(cells)), significant=s, cells=cells)
    return out


def h1b_h2b_gauss(g: pd.DataFrame) -> dict:
    out, cells, cells2 = {}, [], []
    d = prim(g)
    d = d[d.belief == "S2"]
    for (typ, basis), gg in d.groupby(["type", "basis"]):
        ks = sorted(gg.k.unique())
        if 2 not in ks or 32 not in ks:
            continue
        p2, b2, _ = boot_onset(gg[gg.k == 2])
        p32, b32, _ = boot_onset(gg[gg.k == 32], idx=boot_idx(gg[gg.k == 32].seed.nunique(), seed=BOOT_SEED + 1))
        diff = b32 - b2
        lo, hi = ci(diff)
        cells.append(dict(type=typ, basis=basis, onset_k2=p2, onset_k32=p32, diff=p32 - p2, ci=[lo, hi], sig=bool(hi < 0)))
    s = sum(c["sig"] for c in cells)
    out["H1b"] = dict(verdict=verdict(s, len(cells)), significant=s, cells=cells)
    for (typ, k), gg in d[d.k >= 4].groupby(["type", "k"]):
        if set(gg.basis) != {"pair", "triple"}:
            continue
        seeds = sorted(set(gg[gg.basis == "pair"].seed) & set(gg[gg.basis == "triple"].seed))
        idx = boot_idx(len(seeds))  # paired: pair and triple share random numbers within a seed
        res = {}
        for basis in ("pair", "triple"):
            rhos, _, V = seed_matrix(gg[(gg.basis == basis) & gg.seed.isin(seeds)], "minD", defined_col=None)
            res[basis] = (onset_negative(rhos, median_curve(V)), np.array([onset_negative(rhos, median_curve(V[i])) for i in idx]))
        diff = res["triple"][1] - res["pair"][1]
        lo, hi = ci(diff)
        cells2.append(dict(type=typ, k=int(k), pair=res["pair"][0], triple=res["triple"][0], ci=[lo, hi], sig=bool(hi < 0)))
    s = sum(c["sig"] for c in cells2)
    out["H2b_gauss"] = dict(verdict=verdict(s, len(cells2)), significant=s, cells=cells2)
    return out


def powerlaw(curves, typ, basis):
    c = curves[(curves.type == typ) & (curves.basis == basis) & (curves.m == 4) & (curves.belief == "S2") & np.isclose(curves.eps, 0.01)]
    c = c[np.isfinite(c.d_denom_med) & (c.d_denom_med > 0) & (c.chi_D_med > 0) & (c.share_undefined < 0.5) & (c.d_denom_below_share.fillna(1) < 0.5)]
    if len(c) < 3:
        return None
    x, y = np.log(c.chi_D_med.to_numpy()), np.log(c.d_denom_med.to_numpy())
    beta, logC = np.polyfit(x, y, 1)
    res = y - (logC + beta * x)
    return dict(
        beta=float(beta),
        C=float(math.exp(logC)),
        n=len(c),
        within2=float(np.mean(np.abs(res) <= LOG2)),
        k=c.k.tolist(),
        rho=c.rho_S.tolist(),
        chi=c.chi_D_med.tolist(),
        d=c.d_denom_med.tolist(),
    )


def h1c(curves_pop: pd.DataFrame) -> dict:
    cells = []
    for (typ, basis), _ in curves_pop.groupby(["type", "basis"]):
        f = powerlaw(curves_pop, typ, basis)
        if f is not None:
            cells.append(
                dict(
                    type=typ,
                    basis=basis,
                    **{k: v for k, v in f.items() if k in ("beta", "C", "n", "within2")},
                    verdict=share_verdict(f["within2"], 0.9, 0.5),
                )
            )
    vs = [c["verdict"] for c in cells]
    v = (
        "supported"
        if vs and all(x == "supported" for x in vs)
        else ("not supported" if vs and all(x == "not supported" for x in vs) else "inconclusive")
    )
    return dict(verdict=v, cells=cells)


def h2c(curves_g: pd.DataFrame) -> dict:
    c = curves_g[np.isclose(curves_g.eps, 0.01) & curves_g.belief.isin(["S2", "S3c", "S4"]) & (curves_g.rho_S > 0)]
    keys = ["type", "k", "basis", "belief", "rho_S"]
    n = ok = 0
    for _, g in c.groupby(keys):
        if set(g.m) != {2, 4, 6}:
            continue
        if not (np.isfinite(g.d_denom_med).all() and (g.share_undefined < 0.5).all() and (g.d_denom_below_share < 0.5).all()):
            continue
        v = g.d_denom_med.to_numpy()
        n += 1
        ok += v.max() / v.min() <= 1.5
    share = ok / n if n else float("nan")
    return dict(verdict=share_verdict(share, 0.8, 0.5) if n else "inconclusive", share=share, n=n)


# ---------------------------------------------------------------- Q3
def h3(gauss: pd.DataFrame, fam: pd.DataFrame, lkj: pd.DataFrame | None, curves_g: pd.DataFrame) -> dict:
    out = {}
    cells_a, cells_b = [], []
    G = prim(gauss)
    F = prim(fam)
    for (famname, typ, k, basis), g in F[F.family != "gauss"].groupby(["family", "type", "k", "basis"]):
        if typ == "EXCH":
            ref = F[(F.family == "gauss") & (F.type == "EXCH") & (F.k == k) & (F.basis == basis)]
            paired = True
        else:
            ref = G[(G.type == typ) & (G.k == k) & (G.basis == basis)]
            ref = ref[ref.seed < g.seed.nunique()]  # first seeds of the gauss sweep (independent streams)
            paired = False
        if ref.empty:
            continue
        gs = g[g.belief == "S2"]
        rs = ref[ref.belief == "S2"]
        idx_g = boot_idx(gs.seed.nunique())
        idx_r = idx_g if paired else boot_idx(rs.seed.nunique(), seed=BOOT_SEED + 7)
        pg, bg, _ = boot_onset(gs, idx=idx_g)
        pr, br, _ = boot_onset(rs, idx=idx_r)
        lo, hi = ci(bg - br)
        inside, outside = lo >= -0.05 and hi <= 0.05, hi < -0.05 or lo > 0.05
        cells_a.append(
            dict(
                family=famname,
                type=typ,
                k=int(k),
                basis=basis,
                onset=pg,
                onset_gauss=pr,
                diff=pg - pr,
                ci=[lo, hi],
                inside=bool(inside),
                outside=bool(outside),
                paired=paired,
            )
        )
        rg, _, Vg = seed_matrix(gs, "d_denom")
        rr, _, Vr = seed_matrix(rs, "d_denom")
        bel_g = gs.groupby("rho_S").d_denom_below.mean()
        bel_r = rs.groupby("rho_S").d_denom_below.mean()
        for j, rho in enumerate(rg):
            if rho <= 0 or rho not in rr:
                continue
            jr = list(rr).index(rho)
            if bel_g.get(rho, 1) >= 0.5 or bel_r.get(rho, 1) >= 0.5:
                continue
            mg, mr = median_curve(Vg[:, j : j + 1])[0], median_curve(Vr[:, jr : jr + 1])[0]
            if not (np.isfinite(mg) and np.isfinite(mr)) or mr <= 0:
                continue
            bgv = np.array([median_curve(Vg[i, j : j + 1])[0] for i in idx_g])
            brv = np.array([median_curve(Vr[i, jr : jr + 1])[0] for i in idx_r])
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = bgv / brv
            ratio = np.where(np.isnan(ratio), np.inf, ratio)
            lo, hi = ci(ratio)
            cells_b.append(
                dict(
                    family=famname,
                    type=typ,
                    k=int(k),
                    basis=basis,
                    rho_S=float(rho),
                    ratio=mg / mr,
                    ci=[lo, hi],
                    inside=bool(lo >= 0.5 and hi <= 2),
                    outside=bool(hi < 0.5 or lo > 2),
                )
            )
    for name, cells in (("H3a", cells_a), ("H3b", cells_b)):
        if not cells:
            out[name] = dict(verdict="inconclusive", cells=[])
            continue
        v = "supported" if all(c["inside"] for c in cells) else ("not supported" if any(c["outside"] for c in cells) else "inconclusive")
        out[name] = dict(verdict=v, n=len(cells), n_inside=sum(c["inside"] for c in cells), n_outside=sum(c["outside"] for c in cells), cells=cells)
    cells = []
    if lkj is not None:
        L = prim(lkj)
        L = L[(L.belief == "S2") & (L.defined == 1) & (L.d_denom_below != 1)]
        for basis, g in L.groupby("basis"):
            f = powerlaw(curves_g, "SUM", basis)
            if f is None or g.empty:
                continue
            pred = f["C"] * g.chi_D.to_numpy() ** f["beta"]
            w = np.abs(np.log(g.d_denom.to_numpy() / pred)) <= LOG2
            cells.append(
                dict(
                    basis=basis,
                    n=len(g),
                    within2=float(w.mean()),
                    beta=f["beta"],
                    C=f["C"],
                    by_k_eta={
                        f"{int(k)}_{int(e)}": float(np.mean(w[(g.k == k).to_numpy() & (g.eta == e).to_numpy()]))
                        for k, e in g[["k", "eta"]].drop_duplicates().itertuples(index=False)
                    },
                )
            )
    share = np.mean([c["within2"] for c in cells]) if cells else float("nan")
    out["H3c"] = dict(verdict=share_verdict(share, 0.8, 0.5) if cells else "inconclusive", share=share, cells=cells)
    return out


# ---------------------------------------------------------------- Q4
def h4a(curves_g: pd.DataFrame) -> dict:
    c = curves_g[
        (curves_g.m == 4)
        & np.isclose(curves_g.eps, 0.01)
        & (curves_g.basis == "pair")
        & curves_g.k.isin([2, 8, 32])
        & curves_g.belief.isin(["S2", "S3c", "S4"])
        & (curves_g.rho_S > 0)
        & (curves_g.share_undefined < 0.5)
        & np.isfinite(curves_g.d_denom_sd)
    ]
    share = float((c.d_denom_sd > c.d_denom_mcse).mean()) if len(c) else float("nan")
    return dict(
        verdict=share_verdict(share, 0.9, 0.5) if len(c) else "inconclusive",
        share=share,
        n=len(c),
        median_sd_over_mcse=float((c.d_denom_sd / c.d_denom_mcse).median()) if len(c) else float("nan"),
    )


# ---------------------------------------------------------------- Q5
def paired_change(before: pd.DataFrame, after: pd.DataFrame, value: str, keys: list[str], sign: int = -1) -> list[dict]:
    """Per cell: paired (after - before) over seeds; significant if the bootstrap CI of the median lies on `sign` side of 0."""
    cells = []
    b = before.set_index(keys + ["seed"])[value]
    a = after.set_index(keys + ["seed"])[value]
    j = pd.concat([b.rename("b"), a.rename("a")], axis=1, join="inner").dropna()
    for key, g in j.groupby(level=list(range(len(keys)))):
        n_seeds_total = before.set_index(keys).loc[key].seed.nunique() if len(keys) else before.seed.nunique()
        if len(g) < 0.5 * n_seeds_total or len(g) < 3:
            continue
        ch = (g.a - g.b).to_numpy()
        idx = boot_idx(len(ch))
        med = np.median(ch[idx], axis=1)
        lo, hi = ci(med)
        sig = hi < 0 if sign < 0 else lo > 0
        cells.append(
            dict(
                zip(keys, [k.item() if hasattr(k, "item") else k for k in (key if isinstance(key, tuple) else (key,))]),
                n=len(ch),
                median_change=float(np.median(ch)),
                ci=[float(lo), float(hi)],
                sig=bool(sig),
                before=float(np.median(g.b)),
                after=float(np.median(g.a)),
            )
        )
    return cells


def h5(net: pd.DataFrame) -> dict:
    out = {}
    ch = net[(net.table == "chain") & np.isclose(net.eps, 0.01)]
    d = ch[(ch.run == "direct") & ch.belief.isin(["S2", "S3c", "S4"])]
    l1 = d[d.layer == 1].set_index(["k", "rho_S", "basis", "belief", "seed"]).d_total
    l4 = d[d.layer == 4].set_index(["k", "rho_S", "basis", "belief", "seed"]).d_total
    r = (l4 / l1).dropna().rename("ratio").reset_index()
    cells = []
    for key, g in r.groupby(["k", "rho_S", "basis", "belief"]):
        v = g.ratio.to_numpy()
        med = np.median(v[boot_idx(len(v))], axis=1)
        lo, hi = ci(med)
        cells.append(
            dict(
                k=int(key[0]),
                rho_S=float(key[1]),
                basis=key[2],
                belief=key[3],
                ratio=float(np.median(v)),
                ci=[float(lo), float(hi)],
                sig=bool(lo > 1),
            )
        )
    s = sum(c["sig"] for c in cells)
    out["H5a"] = dict(verdict=verdict(s, len(cells)), significant=s, n=len(cells), cells=cells)
    n1 = net[(net.table == "neuron1") & np.isclose(net.eps, 0.01) & (net.belief == "S2")].copy()
    n1.loc[n1.defined != 1, "d_denom"] = np.nan
    before = n1[n1.run == "direct"]
    for hname, value in (("H5b", "d_denom"), ("H5c", "P_Dneg_u")):
        cells = []
        for run_prefix in ("val_adam", "val_lbfgs", "nll_adam", "nll_lbfgs", "e2e4_adam", "e2e4_lbfgs"):
            after = n1[n1.run == f"{run_prefix}_direct"]
            if after.empty:
                continue
            bb = before[before.rho_S > 0] if hname == "H5b" else before
            cs = paired_change(bb, after, value, ["k", "rho_S", "basis"])
            for c in cs:
                if hname == "H5c" and not c["before"] > 0:
                    continue
                cells.append(dict(c, run=run_prefix))
        s = sum(c["sig"] for c in cells)
        out[hname] = dict(verdict=verdict(s, len(cells)), significant=s, n=len(cells), cells=cells)
    held = ch[ch.belief.isin(["S3a", "S3b", "S3d"])]
    cells = []
    for L in (1, 2, 3, 4):
        hb = held[(held.run == "direct") & (held.layer == L)].groupby(["k", "rho_S", "basis", "seed"]).d_total.mean().reset_index()
        for opt in ("adam", "lbfgs"):
            ha = held[(held.run == f"e2e{L}_{opt}_direct") & (held.layer == L)].groupby(["k", "rho_S", "basis", "seed"]).d_total.mean().reset_index()
            if ha.empty:
                continue
            for c in paired_change(hb, ha, "d_total", ["k", "rho_S", "basis"]):
                cells.append(dict(c, L=L, opt=opt))
    s = sum(c["sig"] for c in cells)
    out["H5d"] = dict(verdict=verdict(s, len(cells)), significant=s, n=len(cells), cells=cells)
    return out
