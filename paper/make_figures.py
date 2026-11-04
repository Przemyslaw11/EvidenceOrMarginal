"""The three figures of the paper that are not study figures G1-G9: the idea, the gap's scaling, and where the reference exists.

    .venv/bin/python paper/make_figures.py      # after paper/semantics.py and paper/check_theory.py; writes paper/figures/F*.pdf

Same colours, markers and sizes as the paper mode of analysis2/figures.py. Reads paper/semantics.json, paper/theory.json and
results2/summary/thresholds_*.csv, so nothing plotted here is typed by hand.
"""
import json
import math
import os
import sys

os.environ["HCR_PAPER"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HCR_FIGURES2"] = os.path.join(ROOT, "paper", "figures")
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.special import ndtr, ndtri  # noqa: E402

from analysis2 import figures as F  # noqa: E402

plt = F.plt
CD, EU, GREY = F.OI[0], F.OI[1], "#555555"
KS = (2, 4, 8, 16, 32)


def f1(sem):
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.3), gridspec_kw=dict(width_ratios=[1, 1.15, 1.15], wspace=0.36))
    ex = sem["example"]
    g, D = np.array(ex["g"]), np.array(ex["D"])
    a = axs[0]
    s = ndtri(g)
    mu = np.array([1.2, -1.2]) / (1 + ex["tau"] ** 2)  # conflicting observations of the two inputs
    v = ex["tau"] ** 2 / (1 + ex["tau"] ** 2)
    dens = lambda z, m: np.exp(-0.5 * (z - m) ** 2 / v + 0.5 * z ** 2)  # noqa: E731  density in y of a normal score belief
    q = np.outer(dens(s, mu[0]), dens(s, mu[1]))
    t = q * D
    a.contourf(g, g, D.T, levels=np.linspace(0, 3.2, 9), cmap="Greys", alpha=0.45)
    for z, c, ls in ((q, CD, "-"), (t, EU, "--")):
        a.contour(g, g, z.T, levels=[z.max() * x for x in (0.2, 0.6)], colors=c, linestyles=ls, linewidths=1.3)
    a.set_xlabel("$y_1$")
    a.set_ylabel("$y_2$")
    a.set_aspect("equal")
    a.grid(False)
    a.legend(handles=[Line2D([], [], color=CD, ls="-", label="$q$"), Line2D([], [], color=EU, ls="--", label=r"$\tilde q\propto qD$")], loc="upper right", frameon=True)
    a.text(0.03, 0.04, "shading: $D(y)$", transform=a.transAxes)
    a.text(-0.02, 1.03, "(a)", transform=a.transAxes, fontweight="bold")
    rows = [r for r in sem["k2"] if r["tau"] == 1.0]
    rho = [r["rho"] for r in rows]
    for ax, key, tag, title in ((axs[1], "evidence", "(b)", "truth is $p(x\\mid o)$ (evidence)"), (axs[2], "marginal", "(c)", "truth is $\\mathrm{E}_q[p(x\\mid y)]$ (marginal)")):
        ax.plot(rho, [r[key]["cd"] for r in rows], color=CD, marker="o", label=r"$p_{\mathrm{CD}}$")
        ax.plot(rho, [r[key]["eu"] for r in rows], color=EU, marker="s", ls="--", label=r"$p_{\mathrm{EU}}$")
        ax.set_xlabel(r"input correlation $\rho_S$")
        ax.set_ylim(0, 0.032)
        ax.set_xticks(rho)
        ax.text(-0.02, 1.03, f"{tag} {title}", transform=ax.transAxes, fontweight="bold")
    axs[1].set_ylabel("TV to the truth (median)")
    axs[1].legend(loc="upper left", frameon=False)
    axs[2].tick_params(labelleft=False)
    F.save(fig, "F1_semantics")


def f2(th):
    d = pd.DataFrame(th["leading"]["points"])
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True, gridspec_kw=dict(wspace=0.08))
    for k in KS:
        for basis, mk in (("pair", "o"), ("triple", "^")):
            for typ, fill in (("SUM", True), ("CONTRAST", False)):
                x = d[(d.k == k) & (d.basis == basis) & (d.type == typ)]
                c = F.OI[F.KCOL[k]]
                kw = dict(s=15, marker=mk, edgecolors=c, facecolors=c if fill else "none", linewidths=0.8)
                axs[0].scatter(x.chi, x.d, **kw)
                axs[1].scatter(x.pred, x.d, **kw)
    lo, hi = 1e-5, 0.2
    z = np.array([lo, hi])
    axs[1].fill_between(z, z / 2, z * 2, color="#000000", alpha=0.08, lw=0)
    axs[1].plot(z, z, color=GREY, lw=0.9)
    axs[1].text(4e-5, 2.2e-4, "factor 2", rotation=38, color=GREY)
    for a in axs:
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_ylim(lo, hi)
    axs[1].set_xlim(lo, hi)
    axs[0].set_xlabel(r"$\chi_D$ (the frozen H1c variable)")
    axs[1].set_xlabel(r"closed form, no fitted constant")
    axs[0].set_ylabel(r"measured $d_{\mathrm{denom}}$")
    axs[0].text(-0.02, 1.03, "(a)", transform=axs[0].transAxes, fontweight="bold")
    axs[1].text(-0.02, 1.03, "(b)", transform=axs[1].transAxes, fontweight="bold")
    h = [Line2D([], [], ls="", marker="o", color=F.OI[F.KCOL[k]], label=f"$k={k}$") for k in KS]
    h += [Line2D([], [], ls="", marker="o", color=GREY, label="pair"), Line2D([], [], ls="", marker="^", color=GREY, label="triple"),
          Line2D([], [], ls="", marker="o", color=GREY, mfc="none", label="CONTRAST (hollow)")]
    fig.legend(handles=h, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.16), columnspacing=1.0, handletextpad=0.2)
    F.save(fig, "F2_scaling")


def f3(th):
    tg = pd.read_csv(os.path.join(ROOT, "results2/summary/thresholds_gauss.csv"))
    tp = pd.read_csv(os.path.join(ROOT, "results2/summary/thresholds_population.csv"))
    sel = lambda t, b: t[(t.type == "SUM") & (t.m == 4) & (t.belief == "S2") & (t.basis == b)].sort_values("k")  # noqa: E731
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True, gridspec_kw=dict(wspace=0.08))
    floor = 2.2e-3
    for ax, b in zip(axs, ("pair", "triple")):
        p, g = sel(tp, b), sel(tg, b)
        ax.fill_between(p.k, floor, p.neg_onset, color=CD, alpha=0.13, lw=0)
        ax.plot(p.k, p.neg_onset, color=CD, marker="o", label="population")
        pos = g[g.neg_onset > 0]
        ax.plot(pos.k, pos.neg_onset, color=EU, marker="s", ls="--", label=r"$n=10^7$")
        zero = g[g.neg_onset <= 0]
        if len(zero):
            ax.scatter(zero.k, [floor * 1.25] * len(zero), marker="v", color=EU, s=22, zorder=3)
            ax.text(zero.k.iloc[0] * 0.98, floor * 1.75, r"$\min D<0$ at $\rho_S=0$", ha="center", fontsize=7.5)
        if b == "pair":
            r = pd.DataFrame(th["pair_onset"])
            r = r[r.m == 4].sort_values("k")
            ax.plot(r.k, r.crude, color=GREY, ls=":", marker="x", label="closed bound")
            ax.plot(r.k, r.lower, color=F.OI[2], ls="-.", marker="^", label="convex bound")
        else:
            r = pd.DataFrame(th["triple_onset"])
            r = r[(r.m == 4) & r.corner.notna()].sort_values("k")
            ax.plot(r.k, r.corner, color=F.OI[2], ls="-.", marker="^", label="balanced corner")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.minorticks_off()
        ax.set_ylim(floor, 0.8)
        ax.set_xlabel("number of inputs $k$")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, handlelength=1.8, handletextpad=0.4, frameon=False, columnspacing=0.8)
        ax.text(0.03, 0.05 if b == "pair" else 0.32, r"$D>0$: $p_{\mathrm{EU}}$ exists", transform=ax.transAxes)
        ax.text(-0.02, 1.03, f"({'ab'[b == 'triple']}) {b} basis", transform=ax.transAxes, fontweight="bold")
    axs[0].set_yticks([0.01, 0.1, 0.5], ["0.01", "0.1", "0.5"])
    axs[0].set_ylabel(r"input correlation $\rho_S$")
    F.save(fig, "F3_existence")


def main():
    sem = json.load(open(os.path.join(ROOT, "paper/semantics.json")))
    th = json.load(open(os.path.join(ROOT, "paper/theory.json")))
    f1(sem)
    f2(th)
    f3(th)


if __name__ == "__main__":
    main()
