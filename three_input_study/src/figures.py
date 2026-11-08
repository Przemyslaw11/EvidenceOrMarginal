"""Stage 6: figures F1-F7 and summary_onepager."""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hcr import ROOT, load_cfg

# Okabe-Ito colour-blind-safe palette
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", verm="#D55E00", sky="#56B4E9", purple="#CC79A7", black="#000000", yellow="#F0E442")
MCOL = {2: C["sky"], 4: C["blue"], 6: C["verm"]}
MMK = {2: "o", 4: "s", 6: "^"}
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)
FIG = ROOT + "/figures/"
cfg = load_cfg()
df = pd.read_csv(ROOT + "/results/results.csv")
E0 = df[df.eps == df.eps.min()]  # non-eps columns identical across eps rows
E1 = df[df.eps == 0.01]


def save(fig, name, png_dpi=200):
    fig.savefig(FIG + name + ".png", dpi=png_dpi, bbox_inches="tight")
    fig.savefig(FIG + name + ".pdf", bbox_inches="tight")
    plt.close(fig)


def series(d, typ, m, sc, col):
    g = d[(d.type == typ) & (d.m == m) & (d.scenario == sc)].sort_values("rho_S")
    return g.rho_S.values, g[col].values, g


def line_with_gaps(ax, x, y, ymark, **kw):
    """Plot y(x) with gaps at NaN and a hollow marker at ymark for each undefined x."""
    ax.plot(x, y, **kw)
    bad = np.isnan(y)
    if bad.any():
        ax.plot(x[bad], np.full(bad.sum(), ymark), linestyle="none", marker=kw.get("marker", "o"), mfc="none", mec=kw.get("color", "k"), ms=7)


def f1():
    scs = ["S2", "S3a", "S3b", "S3c"]
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.5), sharey=True)
    for i, typ in enumerate(["SUM", "CONTRAST"]):
        for j, sc in enumerate(scs):
            ax = axs[i, j]
            for m in [2, 4, 6]:
                x, y, _ = series(E0, typ, m, sc, "d_denom")
                line_with_gaps(ax, x, y, 2e-5, color=MCOL[m], marker=MMK[m], label=f"m={m}")
            ax.axhline(0.01, color="gray", ls=":")
            ax.axhline(0.05, color="gray", ls="--")
            ax.set_yscale("log")
            ax.set_ylim(1e-5, 1)
            ax.set_title(f"{typ}, {sc}")
            if i == 1:
                ax.set_xlabel(r"$\rho_S$ (inputs y,z)")
            if j == 0:
                ax.set_ylabel(r"$d_{denom}$ = TV(DUDA, EXACT_UNCAL)")
    axs[0, 0].legend(loc="upper left")
    fig.suptitle(
        "F1  Constant-denominator error. Dotted/dashed: 1% / 5%. Hollow markers at the bottom: EXACT_UNCAL undefined (D<0.01 on supp q)", y=1.0
    )
    fig.tight_layout()
    save(fig, "F1_d_denom")


def f2():
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    for i, typ in enumerate(["SUM", "CONTRAST"]):
        for j, (col, ttl) in enumerate(
            [("min_D", "min of D(y,z)"), ("frac_D_neg", "area fraction with D<0"), ("frac_joint_neg", "volume fraction with N(x,y,z)<0")]
        ):
            ax = axs[i, j]
            for m in [2, 4, 6]:
                x, y, _ = series(E0, typ, m, "S2", col)
                ax.plot(x, y, color=MCOL[m], marker=MMK[m], label=f"m={m}")
            if col == "min_D":
                ax.axhline(0, color="k", lw=1)
                ax.axhline(cfg["D_floor"], color="gray", ls=":")
            ax.set_title(f"{typ}: {ttl}")
            if i == 1:
                ax.set_xlabel(r"$\rho_S$")
    axs[0, 0].legend()
    fig.suptitle("F2  Negativity of the modelled input density D and of the joint model N (N<0 already occurs at rho_S=0)")
    fig.tight_layout()
    save(fig, "F2_negativity")


def f3():
    scs = ["S1", "S2", "S3c"]
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), sharey=True)
    for i, typ in enumerate(["SUM", "CONTRAST"]):
        for j, sc in enumerate(scs):
            ax = axs[i, j]
            x, y1, _ = series(E0, typ, 4, sc, "dmean_vs_uncal")
            _, y2, _ = series(E0, typ, 4, sc, "relvar_vs_uncal")
            line_with_gaps(ax, x, np.abs(y1), 1.5e-6, color=C["blue"], marker="o", label="|Δ mean|")
            line_with_gaps(ax, x, np.abs(y2), 1.5e-6, color=C["verm"], marker="s", label="|rel. variance error|")
            ax.set_yscale("log")
            ax.set_ylim(1e-6, 2)
            ax.set_title(f"{typ}, {sc}, m=4")
            if i == 1:
                ax.set_xlabel(r"$\rho_S$")
            if j == 0:
                ax.set_ylabel("DUDA vs EXACT_UNCAL")
    axs[0, 0].legend()
    fig.suptitle("F3  Mean error vs relative variance error (hollow: EXACT_UNCAL undefined or q invalid)")
    fig.tight_layout()
    save(fig, "F3_mean_vs_variance")


def f4():
    rhos = [0.2, 0.5, 0.8]
    comps = [
        ("d_denom", "d_denom", C["blue"]),
        ("d_neg", "d_neg (eps=0.01)", C["orange"]),
        ("d_model_uncal", "d_model UNCAL", C["green"]),
        ("d_model_cal", "d_model CAL (eps=0.01)", C["purple"]),
        ("d_total", "d_total", C["black"]),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharey=True)
    for i, typ in enumerate(["SUM", "CONTRAST"]):
        for j, sc in enumerate(["S2", "S3c"]):
            ax = axs[i, j]
            for r_i, rho in enumerate(rhos):
                row = E1[(E1.type == typ) & (E1.m == 4) & (E1.scenario == sc) & (np.isclose(E1.rho_S, rho))].iloc[0]
                for k, (col, lab, colr) in enumerate(comps):
                    xpos = r_i * 6.5 + k
                    v = row[col]
                    if np.isnan(v):
                        ax.text(xpos, 4e-4, "n/d", rotation=90, ha="center", va="bottom", fontsize=10)
                    else:
                        ax.bar(
                            xpos,
                            v,
                            color=colr,
                            label=lab if r_i == 0 else None,
                            hatch="//" if (col.startswith("d_model") or col == "d_total") and row.quadrature_limited == 1 else None,
                        )
            ax.set_xticks([2 + 6.5 * k for k in range(3)])
            ax.set_xticklabels([rf"$\rho_S$={r}" for r in rhos])
            ax.set_yscale("log")
            ax.set_ylim(3e-4, 1)
            ax.set_title(f"{typ}, {sc}, m=4")
            ax.grid(axis="x", alpha=0)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.04), fontsize=11)
    fig.suptitle("F4  Error decomposition (TV; not additive). n/d: EXACT_UNCAL undefined. Hatched: GOLD quadrature-limited")
    fig.tight_layout()
    save(fig, "F4_decomposition")


def draw_dens(ax, d, ylim=(-0.6, 3.2)):
    x = d["x"]
    ax.plot(x, d["gold"], color=C["black"], lw=2.2, label="GOLD")
    if np.isfinite(d["uncal"]).any():
        ax.plot(x, d["uncal"], color=C["green"], lw=1.8, ls="--", label="EXACT_UNCAL")
    ax.plot(x, d["cal"], color=C["orange"], lw=1.8, ls="-.", label="EXACT_CAL (eps=0.01)")
    ax.plot(x, d["duda"], color=C["blue"], lw=2, label="DUDA")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylim(*ylim)


def f5(dens):
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), sharey=True)
    for i, sc in enumerate(["S2", "S3c"]):
        for j, rho in enumerate([0.0, 0.4, 0.8]):
            key = next(k for k in dens if k[1] == sc and np.isclose(k[0], rho) and k[2] == 0.01)
            ax = axs[i, j]
            draw_dens(ax, dens[key])
            ax.set_title(rf"CONTRAST, {sc}, $\rho_S$={rho}")
            if i == 1:
                ax.set_xlabel("x")
    axs[0, 0].legend(loc="upper center", fontsize=10)
    fig.suptitle("F5  Propagated output densities (m=4). y-axis clipped to [-0.6, 3.2]; GOLD spikes near x=0,1 are cut for S2 at high rho_S")
    fig.tight_layout()
    save(fig, "F5_densities")


def f6():
    t = pd.read_csv(ROOT + "/results/taylor.csv")
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    cols = {0: C["black"], 1: C["blue"], 2: C["verm"]}
    for i, typ in enumerate(["SUM", "CONTRAST"]):
        for j, sc in enumerate(["S2", "S3c"]):
            ax = axs[i, j]
            for K in [0, 1, 2]:
                g = t[(t.type == typ) & (t.scenario == sc) & (t.K == K)].sort_values("rho_S")
                line_with_gaps(
                    ax, g.rho_S.values, g.d_taylor.values, 2e-5, color=cols[K], marker="o", label="K=0 (Duda)" if K == 0 else f"Taylor K={K}"
                )
            ax.set_yscale("log")
            ax.set_ylim(1e-5, 1)
            ax.set_title(f"{typ}, {sc}, m=4")
            if i == 1:
                ax.set_xlabel(r"$\rho_S$")
            if j == 0:
                ax.set_ylabel("TV(P_K, EXACT_UNCAL)")
    axs[0, 0].legend()
    fig.suptitle("F6  Optional: Taylor-corrected denominator vs constant denominator (hollow: EXACT_UNCAL undefined)")
    fig.tight_layout()
    save(fig, "F6_taylor")


def f7():
    t = pd.read_csv(ROOT + "/results/training.csv")
    u = t.drop_duplicates(["type", "rho_S"])
    lab = [f"{r.type[:3]} {r.rho_S}" for r in u.itertuples()]
    xx = np.arange(len(u))
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    ax = axs[0]
    ax.bar(xx - 0.2, u.mse_test_before, 0.4, color=C["sky"], label="before")
    ax.bar(xx + 0.2, u.mse_test_after, 0.4, color=C["blue"], label="after")
    ax.set_yscale("log")
    ax.set_title("test MSE of E[x|y,z] (Eq. 7)")
    ax.set_xticks(xx)
    ax.set_xticklabels(lab, rotation=45)
    ax.legend()
    ax = axs[1]
    ax.bar(xx - 0.2, u.min_D_before, 0.4, color=C["sky"], label="min D before")
    ax.bar(xx + 0.2, u.min_D_after, 0.4, color=C["blue"], label="min D after")
    ax.axhline(0, color="k", lw=1)
    ax.set_title("min D(y,z) of the model")
    ax.set_xticks(xx)
    ax.set_xticklabels(lab, rotation=45)
    ax.legend()
    ax = axs[2]
    for sc, colr, mk in [("S2", C["blue"], "o"), ("S3c", C["verm"], "s")]:
        g = t[t.scenario == sc].reset_index(drop=True)
        ax.plot(xx, g.d_total_before, color=colr, marker=mk, mfc="none", ls="none", ms=10, label=f"{sc} before")
        ax.plot(xx, g.d_total_after, color=colr, marker=mk, ls="none", ms=8, label=f"{sc} after")
    ax.set_title("d_total = TV(DUDA, GOLD)")
    ax.set_xticks(xx)
    ax.set_xticklabels(lab, rotation=45)
    ax.legend(fontsize=10)
    fig.suptitle("F7  Optional, exploratory: does a value (MSE) loss change negativity and propagation error? (m=4)")
    fig.tight_layout()
    save(fig, "F7_training")


def onepager(dens):
    fs = 16
    plt.rcParams.update(
        {"font.size": fs, "axes.titlesize": fs, "axes.labelsize": fs, "xtick.labelsize": fs - 2, "ytick.labelsize": fs - 2, "legend.fontsize": fs - 3}
    )
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    tcol = {"SUM": C["blue"], "CONTRAST": C["verm"]}
    ax = axs[0, 0]
    for typ in ["SUM", "CONTRAST"]:
        x, y, _ = series(E0, typ, 4, "S2", "d_denom")
        line_with_gaps(ax, x, y, 6e-5, color=tcol[typ], marker="o", lw=2.5, label=typ)
    ax.axhline(0.05, color="gray", ls="--")
    ax.set_yscale("log")
    ax.set_ylim(3e-5, 0.5)
    ax.set_xlabel(r"$\rho_S$ between the two inputs")
    ax.set_ylabel("TV(DUDA, exact ratio)")
    ax.set_title("(a) Constant-denominator error")
    ax.legend(loc="lower right")
    ax.text(0.02, 0.97, "m=4, flat input belief\nhollow: exact ratio undefined", transform=ax.transAxes, va="top", fontsize=fs - 4)
    ax = axs[0, 1]
    for m in [2, 4, 6]:
        x, y, _ = series(E0, "SUM", m, "S2", "min_D")
        ax.plot(x, y, color=MCOL[m], marker=MMK[m], lw=2.5, label=f"m={m}")
    ax.axhline(0, color="k", lw=1.5)
    ax.axhline(cfg["D_floor"], color="gray", ls=":")
    ax.set_xlabel(r"$\rho_S$")
    ax.set_ylabel("min of modelled input density D")
    ax.set_title("(b) Negativity onset")
    ax.legend(loc="upper right")
    ax = axs[1, 0]
    for typ in ["SUM", "CONTRAST"]:
        x, y1, _ = series(E0, typ, 4, "S1", "dmean_vs_uncal")
        x, y2, _ = series(E0, typ, 4, "S1", "relvar_vs_uncal")
        line_with_gaps(ax, x, np.abs(y1), 3e-6, color=tcol[typ], marker="o", lw=2.5, ls="--", label=f"{typ}: |Δ mean|")
        line_with_gaps(ax, x, np.abs(y2), 3e-6, color=tcol[typ], marker="s", lw=2.5, label=f"{typ}: |rel. var. err|")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel(r"$\rho_S$")
    ax.set_ylabel("DUDA vs exact ratio")
    ax.set_title("(c) Variance vs mean error")
    ax.legend(loc="center left", fontsize=fs - 5)
    ax.text(0.98, 0.03, "m=4, belief = neuron's own D", transform=ax.transAxes, ha="right", fontsize=fs - 4)
    ax = axs[1, 1]
    key = next(k for k in dens if k[1] == "S2" and np.isclose(k[0], 0.4) and k[2] == 0.01)
    draw_dens(ax, dens[key], ylim=(0, 2.2))
    ax.legend(loc="upper center", fontsize=fs - 4)
    ax.set_xlabel("x (output)")
    ax.set_title(r"(d) Example, CONTRAST, $\rho_S$=0.4")
    fig.suptitle("HCR neuron, density propagation through a Gaussian-copula test neuron (numerical check)", fontsize=fs + 2, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG + "summary_onepager.png", dpi=100)
    fig.savefig(FIG + "summary_onepager.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with open(ROOT + "/results/densities.pkl", "rb") as fh:
        dens = pickle.load(fh)
    f1()
    f2()
    f3()
    f4()
    f5(dens)
    if os.path.exists(ROOT + "/results/taylor.csv"):
        f6()
    if os.path.exists(ROOT + "/results/training.csv"):
        f7()
    onepager(dens)
    print("figures done")
