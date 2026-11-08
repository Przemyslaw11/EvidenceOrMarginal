"""Figures G1-G9 and summary_onepager.png from results/summary/*.csv and the merged network results.

    python -m analysis.figures
Okabe-Ito colours in a fixed order per entity (validated: CVD separation and normal-vision floor pass; orange and
reddish purple are below 3:1 contrast, so every series also has its own marker and a legend). Undefined points are
gaps; points below 3x their MC noise floor are hollow markers; bands are 2.5-97.5 % seed intervals.

    HCR_PAPER=1 HCR_FIGURES=paper/figures python -m analysis.figures
renders the same data for the paper at print size (text width, 1:1): no figure titles (the captions carry them), the paper's
symbols and type, panel letters, hatched bars for greyscale, PDF only. Nothing plotted changes.
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.common import ROOT, SUMMARY, load

OUT = os.environ.get("HCR_FIGURES", os.path.join(ROOT, "figures"))
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]
MK = ["o", "s", "^", "D", "v", "P", "X"]
KCOL = {2: 0, 4: 1, 8: 2, 16: 3, 32: 4}
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "savefig.dpi": 200,
    }
)
PAPER = os.environ.get("HCR_PAPER") == "1"
SYMBOLS = (
    ("CONST_DENOM", r"$p_{\mathrm{CD}}$"),
    ("EXACT_UNCAL", r"$p_{\mathrm{EU}}$"),
    ("EXACT_CAL", r"$p_{\mathrm{EC}}$"),
    ("GOLD", r"$p_{\mathrm{G}}$"),
    ("d_denom", r"$d_{\mathrm{denom}}$"),
    ("d_neg", r"$d_{\mathrm{neg}}$"),
    ("d_model_cal", r"$d_{\mathrm{model,cal}}$"),
    ("d_total", r"$d_{\mathrm{total}}$"),
    ("ρ_S", r"$\rho_S$"),
    ("n_est=1e7", r"$n=10^7$"),
    ("[0,1]^k", r"$[0,1]^k$"),
    ("E_q[N/D]", r"$E_q[N/D]$"),
    ("lbfgs", "L-BFGS"),
    ("adam", "Adam"),
    ("min D", r"$\min D$"),
    ("P(D < 0)", "P(D<0)"),
    ("P(D<0)", r"$P(D<0)$"),
    ("D<0 onset", r"$D<0$ onset"),
    ("flat-D", r"flat-$D$"),
    ("≥ 1 %", r"$\geq 1\,\%$"),
    ("95 %", r"$95\,\%$"),
    ("Kendall τ", r"Kendall's $\tau$"),
    ("number of inputs k", r"number of inputs $k$"),
    ("layer l", r"layer $l$"),
    ("y uniform", r"$y$ uniform"),
    ("(TV)", ""),
    ("gauss", "Gaussian"),
    ("clayton", "Clayton"),
    ("gumbel", "Gumbel"),
    ("t10", r"$t_{10}$"),
    ("t3", r"$t_3$"),
)
PAGE = 6.3  # text width in inches: paper figures are drawn at this size and included at scale 1
CAP = 3
HATCH = [None] * 4
BAR = [0, 1, 2, 3]
if PAPER:
    CAP = 1.5
    BAR = [3, 1, 2, 0]  # d_denom keeps the colour it has in G2
    HATCH = [None, "////", "....", "xxxx"]
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1,
            "lines.markersize": 3.2,
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.4,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.3,
            "ytick.minor.width": 0.3,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 3,
            "axes.labelpad": 2,
            "legend.frameon": False,
            "hatch.linewidth": 0.4,
            "pdf.fonttype": 42,
            "figure.subplot.left": 0.065,
            "figure.subplot.right": 0.995,
            "figure.subplot.bottom": 0.13,
            "figure.subplot.top": 0.92,
            "figure.subplot.wspace": 0.16,
        }
    )
    _subplots = plt.subplots

    def _scaled(*a, figsize, **kw):
        w, h = figsize
        s = PAGE / w if w >= 10 else 0.5  # the one square figure is half width
        return _subplots(*a, figsize=(w * s, h * s * (0.85 if h > 0.5 * w and w >= 10 else 1)), **kw)

    plt.subplots = _scaled


def panel(ax, tag, title=""):
    """Panel label and title, the same in every figure."""
    ax.set_title(f"({tag}) {title}".strip(), loc="left")


def run_label(r):
    """'e2e4_adam' -> 'end-to-end\nL=4\nadam'."""
    loss, opt = r.split("_")
    if PAPER:
        return {"val": "value", "nll": "NLL", "e2e4": "end-to-\nend,\n$L=4$"}[loss] + "\n" + {"adam": "Adam", "lbfgs": "L-BFGS"}[opt]
    return {"val": "val. loss", "nll": "NLL", "e2e4": "end-to-\nend, L=4"}[loss] + "\n" + opt


def symbols(txt):
    if "$" in txt:  # already typeset
        return txt
    for a, b in SYMBOLS:
        txt = txt.replace(a, b)
    txt = re.sub(r"\$\\rho_S\$=([\d.]+)", r"$\\rho_S=\1$", txt)
    return re.sub(r"(?<![\w$\\{=])([kLm])=(\d+)", r"$\1=\2$", txt)


def typeset(fig):
    """Paper rendering of a finished figure: presentation only."""
    if fig._suptitle is not None:
        fig._suptitle.remove()
        fig._suptitle = None
    axs = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    for i, ax in enumerate(axs):
        t = ax.get_title("left")
        t = "" if t.startswith("G6:") else t
        if len(axs) > 1 and not t.startswith("("):
            panel(ax, "abcdefgh"[i], t)
        else:
            ax.set_title(t, loc="left")
    for t in fig.findobj(matplotlib.text.Text):
        t.set_text(symbols(t.get_text()))
        t.set_fontsize(min(t.get_fontsize(), 8))
    if len(fig.axes) >= 4 and not fig.get_constrained_layout():
        fig.subplots_adjust(hspace=0.42)


def fit(fig):
    """Narrow the axes until the figure, labels included, fits the text width: fonts stay at print size."""
    for _ in range(4):
        fig.canvas.draw()
        excess = fig.get_tightbbox(fig.canvas.get_renderer()).width - (PAGE - 0.03)  # 0.02 is savefig's padding
        if excess <= 0 or fig.get_constrained_layout():
            return
        fig.subplots_adjust(left=fig.subplotpars.left + excess / fig.get_figwidth())


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    if PAPER:
        typeset(fig)
        fit(fig)
        fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.01, metadata={"CreationDate": None})  # no date: reproducible
        plt.close(fig)
        return
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def csv(name):
    p = os.path.join(SUMMARY, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def sel(c, **kw):
    for k, v in kw.items():
        c = c[np.isclose(c[k], v)] if isinstance(v, float) else c[c[k] == v]
    return c


def curve(ax, c, metric, color, marker, label, logy=True, band=True, ls="-"):
    """Seed-median curve with seed band; gaps where >= 50 % undefined; hollow markers where below noise."""
    c = c.sort_values("rho_S")
    x = c.rho_S.to_numpy()
    y = c[metric + "_med"].to_numpy(float) if metric + "_med" in c else np.full(len(c), np.nan)
    if "share_undefined" in c and metric in ("d_denom", "d_neg", "d_model_uncal", "relvar_vs_uncal"):
        y = np.where(c.share_undefined.to_numpy() >= 0.5, np.nan, y)
    if logy:
        y = np.where(y > 0, y, np.nan)
    ax.plot(x, y, color=color, marker=None, ls=ls)
    ax.plot([], [], color=color, marker=marker, ls=ls, label=label)  # legend entry shows the marker too
    below = c[metric + "_below_share"].to_numpy(float) >= 0.5 if metric + "_below_share" in c else np.zeros(len(c), bool)
    ok = np.isfinite(y)
    ax.plot(x[ok & ~below], y[ok & ~below], ls="none", marker=marker, color=color)
    ax.plot(x[ok & below], y[ok & below], ls="none", marker=marker, mfc="white", color=color)
    if band and metric + "_lo" in c:
        lo, hi = c[metric + "_lo"].to_numpy(float), c[metric + "_hi"].to_numpy(float)
        m = ok & np.isfinite(lo) & np.isfinite(hi)
        if logy:
            m &= lo > 0
        ax.fill_between(x[m], lo[m], hi[m], color=color, alpha=0.15, lw=0)
    if logy:
        ax.set_yscale("log")


def g1(cg):
    """d_denom vs rho_S per k; beliefs S2, S3c, S4 x types; pair basis, m = 4."""
    if cg is None:
        return
    types = [t for t in ("SUM", "CONTRAST") if t in set(cg.type)]
    fig, axs = plt.subplots(len(types), 3, figsize=(15, 4.3 * len(types)), squeeze=False, sharex=True)
    for i, typ in enumerate(types):
        for j, b in enumerate(("S2", "S3c", "S4")):
            ax = axs[i, j]
            for k in sorted(set(cg.k)):
                c = sel(cg, type=typ, belief=b, basis="pair", m=4, eps=0.01, k=k)
                if len(c):
                    curve(ax, c, "d_denom", OI[KCOL.get(k, 5)], MK[KCOL.get(k, 5)], f"k={k}")
            ax.axhline(0.01, color="0.4", ls=":", lw=1)
            ax.axhline(0.05, color="0.4", ls="--", lw=1)
            ax.set_title(f"{typ}, belief {b}")
            if i == len(types) - 1:
                ax.set_xlabel("input Spearman ρ_S")
            if j == 0:
                ax.set_ylabel("d_denom (TV)")
    if PAPER:
        fig.subplots_adjust(wspace=0.24)
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07))
    else:
        axs[0, 0].legend(title="inputs", loc="lower right")
    fig.suptitle("G1: constant-denominator error vs input dependence (pair basis, m=4, n_est=1e7; gaps = EXACT_UNCAL undefined)")
    save(fig, "G1_d_denom_vs_rho")


def g2(tg, tp):
    """Thresholds vs k with CIs: negativity onset (finite n and population) and d_denom >= 1 % (S2)."""
    if tg is None:
        return
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, typ in zip(axs, ("SUM", "CONTRAST")):
        n = 0
        for src, t, ls in (("n_est=1e7", tg, "-"), ("population", tp, "--")):
            if t is None:
                continue
            for bi, basis in enumerate(("pair", "triple")):
                c = sel(t, type=typ, basis=basis, m=4, belief="S2").sort_values("k")
                if c.empty:
                    continue
                col, mk = OI[bi * 2], MK[bi * 2 + (src == "population")]
                y = c.neg_onset.replace(np.inf, np.nan)
                lo = (y - c.neg_onset_lo).clip(lower=0) if "neg_onset_lo" in c else 0
                hi = (c.neg_onset_hi.replace(np.inf, np.nan) - y).clip(lower=0) if "neg_onset_hi" in c else 0
                ax.errorbar(c.k, y, yerr=[lo, hi], color=col, ls=ls, marker=mk, capsize=CAP, label=f"D<0 onset, {basis}, {src}")
                n += 1
        c = sel(tg, type=typ, basis="pair", m=4, belief="S2").sort_values("k")
        y = c.denom_1pct.replace(np.inf, np.nan)
        ax.errorbar(
            c.k,
            y,
            yerr=[(y - c.denom_1pct_lo).clip(lower=0), (c.denom_1pct_hi.replace(np.inf, np.nan) - y).clip(lower=0)],
            color=OI[3],
            marker=MK[3],
            capsize=CAP,
            label="d_denom ≥ 1 % (S2, pair)",
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"])
        ax.set_xlabel("number of inputs k")
        ax.set_title(typ)
    axs[0].set_ylabel("threshold ρ_S (95 % bootstrap CI)")
    if PAPER:
        fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.22))
    else:
        axs[0].legend(fontsize=12)
    fig.suptitle("G2: thresholds vs number of inputs (m=4; missing = not reached before EXACT_UNCAL is undefined)")
    save(fig, "G2_thresholds_vs_k")


def g3(cg):
    if cg is None:
        return
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for j, basis in enumerate(("pair", "triple")):
        for k in sorted(set(cg.k)):
            c = sel(cg, type="SUM", belief="S2", basis=basis, m=4, eps=0.01, k=k)
            if c.empty:
                continue
            col, mk = OI[KCOL.get(k, 5)], MK[KCOL.get(k, 5)]
            curve(axs[0, j], c, "minD", col, mk, f"k={k}", logy=False)
            curve(axs[1, j], c, "P_Dneg_u", col, mk, f"k={k}", logy=True, band=False)
        axs[0, j].axhline(0, color="0.3", lw=1)
        axs[0, j].set_yscale("symlog", linthresh=0.1)
        axs[0, j].set_title(f"SUM, {basis} basis")
        axs[1, j].set_xlabel("input Spearman ρ_S")
    axs[0, 0].set_ylabel("min D over [0,1]^k\n(upper bound)")
    axs[1, 0].set_ylabel("P(D < 0), y uniform")
    if PAPER:
        fig.subplots_adjust(left=0.085)
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    else:
        axs[0, 0].legend(title="inputs")
    fig.suptitle("G3: negativity of the modelled input density D (m=4, n_est=1e7)")
    save(fig, "G3_negativity")


def g4(cg):
    if cg is None:
        return
    fig, axs = plt.subplots(2, 2, figsize=(13, 7.5), layout="constrained")
    for i, typ in enumerate(("SUM", "CONTRAST")):
        for j, basis in enumerate(("pair", "triple")):
            c = sel(cg, type=typ, belief="S2", basis=basis, m=4, eps=0.01)
            P = c.pivot_table(index="k", columns="rho_S", values="share_undefined")
            ax = axs[i, j]
            if PAPER:  # vector cells, same array and cell centres as the image below
                im = ax.pcolormesh(
                    np.arange(P.shape[1] + 1) - 0.5,
                    np.arange(P.shape[0] + 1) - 0.5,
                    P.to_numpy(),
                    cmap="Blues",
                    vmin=0,
                    vmax=1,
                    edgecolors="face",
                    linewidth=0.2,
                )
                ax.grid(False)
            else:
                im = ax.imshow(P.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1, origin="lower")
            ax.set_yticks(range(len(P.index)), [str(int(v)) for v in P.index])
            ax.set_xticks(range(0, len(P.columns), 2), [f"{v:g}" for v in P.columns[::2]], rotation=45)
            ax.set_title(f"{typ}, {basis}")
            ax.set_ylabel("$k$" if PAPER else "k")
            ax.set_xlabel("ρ_S")
    cb = fig.colorbar(im, ax=axs, label="share of seeds with EXACT_UNCAL undefined (S2)")
    if PAPER:
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor("face")
    fig.suptitle("G4: where the exact ratio E_q[N/D] is undefined (m=4)")
    save(fig, "G4_undefined_share")


def g5(tf, cf):
    if tf is None:
        return
    fams = [("gauss", "EXCH"), ("t10", "EXCH"), ("t3", "EXCH"), ("clayton", "EXCH"), ("gumbel", "EXCH"), ("t3", "SUM"), ("t10", "SUM")]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    for bi, basis in enumerate(("pair", "triple")):
        for fi, (fam, typ) in enumerate(fams):
            c = sel(tf, family=fam, type=typ, basis=basis, m=4, belief="S2").sort_values("k")
            if c.empty:
                continue
            y = c.neg_onset.replace(np.inf, np.nan)
            axs[bi].errorbar(
                c.k * (1 + 0.03 * (fi - 3)),
                y,
                yerr=[(y - c.neg_onset_lo).clip(lower=0), (c.neg_onset_hi.replace(np.inf, np.nan) - y).clip(lower=0)],
                color=OI[fi % 7],
                marker=MK[fi % 7],
                capsize=CAP,
                label=f"{fam} {typ}",
            )
        axs[bi].set_xscale("log", base=2)
        axs[bi].set_xticks([2, 4, 8], ["2", "4", "8"])
        axs[bi].set_title(f"{basis} basis")
        axs[bi].set_xlabel("number of inputs k")
    axs[0].set_ylabel("D<0 onset (ρ_S-equivalent,\nmatched Kendall τ)")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4 if PAPER else 7, bbox_to_anchor=(0.5, -0.26 if PAPER else -0.1))
    fig.suptitle("G5: negativity onset by copula family (m=4, n_est=1e7, 95 % CI)")
    save(fig, "G5_copula_families")


def g6(cg):
    if cg is None:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for bi, b in enumerate(("S2", "S3c", "S4")):
        c = sel(cg, m=4, eps=0.01, basis="pair", belief=b)
        c = c[(c.share_undefined < 0.5) & (c.rho_S > 0)]
        ax.scatter(c.smean_vs_uncal_med, c.relvar_vs_uncal_med.abs(), s=9 if PAPER else 30, color=OI[bi], marker=MK[bi], label=b, alpha=0.8)
    lim = [1e-6, 1]
    ax.plot(lim, lim, color="0.4", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|Δ mean| / sd of EXACT_UNCAL (standardised)")
    ax.set_ylabel("|relative variance error| vs EXACT_UNCAL")
    ax.legend(title="belief")
    ax.set_title("G6: variance error vs mean error (all k, types; dimensionless)")
    save(fig, "G6_mean_vs_variance")


def g7(cg):
    if cg is None:
        return
    mets = ["d_denom", "d_neg", "d_model_cal", "d_total"]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for i, typ in enumerate(("SUM", "CONTRAST")):
        for j, rho in enumerate((0.2, 0.5, 0.8)):
            ax = axs[i, j]
            ks = [k for k in (2, 8, 32) if k in set(cg.k)]
            for mi, mt in enumerate(mets):
                v = []
                for k in ks:
                    c = sel(cg, type=typ, belief="S2", basis="pair", m=4, eps=0.01, k=k, rho_S=rho)
                    val = c[mt + "_med"].iloc[0] if len(c) else np.nan
                    if len(c) and mt in ("d_denom", "d_neg") and c.share_undefined.iloc[0] >= 0.5:
                        val = np.nan
                    v.append(val)
                xs = np.arange(len(ks)) + (mi - 1.5) * 0.2
                ax.bar(xs, np.nan_to_num(v, nan=0), width=0.18, color=OI[BAR[mi]], label=mt, hatch=HATCH[mi], edgecolor="white", lw=0)
                for x, val in zip(xs, v):
                    if not np.isfinite(val):
                        ax.text(x, 1.3e-4, "n/d", ha="center", va="bottom", fontsize=12, rotation=90, color="0.2")
            ax.set_xticks(range(len(ks)), [f"$k={k}$" if PAPER else f"k={k}" for k in ks])
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 2)
            ax.set_title(f"{typ}, ρ_S={rho}")
    axs[0, 0].set_ylabel("TV (seed median)")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("G7: error decomposition (S2, pair, m=4, eps=0.01; n/d = EXACT_UNCAL undefined)")
    save(fig, "G7_decomposition")


def net_frames():
    net = load("network")
    if net is None:
        return None
    return net[np.isclose(net.eps, 0.01)]


def g8(net):
    if net is None:
        return
    ch = net[(net.table == "chain") & (net.run == "direct")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, b in zip(axs, ("S2", "S3c", "S4")):
        for ki, k in enumerate(sorted(set(ch.k))):
            for ri, rho in enumerate(sorted(set(ch.rho_S))):
                c = ch[(ch.k == k) & np.isclose(ch.rho_S, rho) & (ch.belief == b) & (ch.basis == "pair")]
                g = c.groupby("layer").d_total.median()
                if g.empty:
                    continue
                ax.plot(
                    g.index,
                    g.values,
                    color=OI[ki],
                    marker=MK[ri],
                    ls=("-", "--", ":")[ki],
                    alpha=0.5 + 0.5 * ri / 3,
                    label=f"k={k}, ρ_S={rho}" if b == "S2" else None,
                )
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("layer l")
        ax.set_title(f"belief {b}")
    axs[0].set_ylabel("d_total = TV(CONST_DENOM, GOLD), seed median")
    fig.legend(
        *axs[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.3 if PAPER else -0.2),
        **(dict(handlelength=1.6, handletextpad=0.4, columnspacing=0.8) if PAPER else {}),
    )
    fig.suptitle("G8: depth compounding in the HCR chain (direct estimates, pair basis)")
    save(fig, "G8_depth")


def g9(net):
    if net is None:
        return
    n1 = net[(net.table == "neuron1") & (net.belief == "S2")]
    runs = ["val_adam", "val_lbfgs", "nll_adam", "nll_lbfgs", "e2e4_adam", "e2e4_lbfgs"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric, lab) in zip(axs, (("d_dc", "change of TV(CONST_DENOM, EXACT_CAL),\nneuron 1, S2"), ("P_Dneg_u", "change of P(D<0), neuron 1"))):
        for ii, init in enumerate(("direct", "flat")):
            base = n1[n1.run == ("direct" if init == "direct" else "flat_init")].set_index(["k", "rho_S", "basis", "seed"])[metric]
            med, lo, hi = [], [], []
            for r in runs:
                a = n1[n1.run == f"{r}_{init}"].set_index(["k", "rho_S", "basis", "seed"])[metric]
                d = (a - base).dropna()
                med.append(d.median() if len(d) else np.nan)
                lo.append(d.quantile(0.1) if len(d) else np.nan)
                hi.append(d.quantile(0.9) if len(d) else np.nan)
            x = np.arange(len(runs)) + (ii - 0.5) * 0.3
            med, lo, hi = map(np.array, (med, lo, hi))
            ax.errorbar(
                x,
                med,
                yerr=[med - lo, hi - med],
                ls="none",
                marker=MK[ii],
                color=OI[ii],
                capsize=CAP,
                label="from direct estimates" if init == "direct" else "flat-D control",
            )
        ax.axhline(0, color="0.3", lw=1)
        ax.set_xticks(range(len(runs)), [run_label(r) for r in runs])
        ax.set_ylabel(lab)
    if PAPER:
        fig.subplots_adjust(wspace=0.3)
        axs[0].legend(loc="lower left", bbox_to_anchor=(0.1, 1), ncol=2, borderaxespad=0, columnspacing=1.0)  # above the axes: clear of the bars
    else:
        axs[0].legend()
    fig.suptitle("G9: effect of training (median and 10-90 % over all k, ρ_S, bases, seeds)")
    save(fig, "G9_training")


def onepager(cg, tg, tp, net):
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), dpi=100)
    ax = axs[0, 0]
    if cg is not None:
        for k in (2, 8, 32):
            for typ in ("SUM", "CONTRAST"):
                c = sel(cg, type=typ, belief="S2", basis="pair", m=4, eps=0.01, k=k)
                if len(c):
                    curve(ax, c, "d_denom", OI[KCOL[k]], MK[KCOL[k]], f"k={k} {typ}", band=False, ls="-" if typ == "SUM" else "--")
        ax.axhline(0.01, color="0.4", ls=":", lw=1)
        ax.legend(fontsize=12, ncol=2)
    ax.set_title("(a) constant-denominator error, flat belief")
    ax.set_xlabel("input Spearman ρ_S")
    ax.set_ylabel("TV(E[N]/E[D], E[N/D])")
    ax = axs[0, 1]
    if tg is not None:
        for bi, basis in enumerate(("pair", "triple")):
            for typ in ("SUM", "CONTRAST"):
                c = sel(tg, type=typ, basis=basis, m=4, belief="S2").sort_values("k")
                ax.plot(
                    c.k,
                    c.denom_1pct.replace(np.inf, np.nan),
                    color=OI[bi * 2],
                    marker=MK[bi * 2],
                    ls="-" if typ == "SUM" else "--",
                    label=f"{basis} {typ}",
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"])
        ax.legend(fontsize=12)
    ax.set_title("(b) ρ_S where the error reaches 1 % (S2)")
    ax.set_xlabel("number of inputs k")
    ax.set_ylabel("threshold ρ_S")
    ax = axs[1, 0]
    for src, t, ls in (("n_est=1e7", tg, "-"), ("exact coefficients", tp, "--")):
        if t is None:
            continue
        for ti, typ in enumerate(("SUM", "CONTRAST")):
            c = sel(t, type=typ, basis="pair", m=4, belief="S2").sort_values("k")
            ax.plot(c.k, c.neg_onset.replace(np.inf, np.nan), color=OI[ti * 3], marker=MK[ti * 3], ls=ls, label=f"{typ}, {src}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"])
    ax.legend(fontsize=12)
    ax.set_title("(c) ρ_S at which D first becomes negative (pair, m=4)")
    ax.set_xlabel("number of inputs k")
    ax.set_ylabel("negativity onset ρ_S")
    ax = axs[1, 1]
    if net is not None:
        n1 = net[(net.table == "neuron1") & (net.belief == "S2")]
        base = n1[n1.run == "direct"].set_index(["k", "rho_S", "basis", "seed"])
        runs = ["val_adam", "nll_adam", "e2e4_adam", "val_lbfgs", "nll_lbfgs", "e2e4_lbfgs"]
        for mi, (metric, lab) in enumerate((("d_dc", "Δ TV(CONST_DENOM, EXACT_CAL)"), ("P_Dneg_u", "Δ P(D<0)"))):
            vals = []
            for r in runs:
                a = n1[n1.run == f"{r}_direct"].set_index(["k", "rho_S", "basis", "seed"])[metric]
                vals.append((a - base[metric]).dropna().median())
            x = np.arange(len(runs)) + (mi - 0.5) * 0.38
            ax.bar(x, vals, width=0.36, color=OI[mi], label=lab)
            ax.plot(x, vals, ls="none", marker="o", mfc="none", mec=OI[mi], ms=8)  # keeps zero-height bars visible
        ax.set_xticks(range(len(runs)), [run_label(r) for r in runs], fontsize=12)
        ax.axhline(0, color="0.3", lw=1)
        ax.legend(fontsize=12)
    ax.set_title("(d) does training help? median change, neuron 1")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "summary_onepager.png"), dpi=100)
    fig.savefig(os.path.join(OUT, "summary_onepager.pdf"))
    plt.close(fig)


def main():
    cg, tg, tp = csv("curves_gauss.csv.gz"), csv("thresholds_gauss.csv"), csv("thresholds_population.csv")
    tf, cf = csv("thresholds_families.csv"), csv("curves_families.csv.gz")
    net = net_frames()
    for f, args in (
        (g1, (cg,)),
        (g2, (tg, tp)),
        (g3, (cg,)),
        (g4, (cg,)),
        (g5, (tf, cf)),
        (g6, (cg,)),
        (g7, (cg,)),
        (g8, (net,)),
        (g9, (net,)),
        (onepager, (cg, tg, tp, net)),
    ):
        if PAPER and f is onepager:
            continue
        if net is None and f in (g8, g9):
            print("figure", f.__name__, "skipped: no network data")
            continue
        f(*args)
        print("figure", f.__name__, "done")


if __name__ == "__main__":
    main()
