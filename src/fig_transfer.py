#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure T3 — Le classement se transfère, le seuil non (cutoff 25 %, population active).

(a) AUC par module : (A) aléatoire et (B) LOMO, variables absolues ; (B) avec
    variables relatives au rang dans la présentation.
(b) Part d'étudiants signalés (p >= 0,5) : (A), (B) et taux à risque observé.
(c) Dérive de signalement (B − A, même famille de variables) : modèle brut,
    recalibré sur 100 étudiants étiquetés du module, variables relatives.

Lit results/transfer_by_module_c25_active[_relatives].csv et
results/recalibration_k_c25_active.csv ; écrit figures/transfer.pdf et .png.
Usage : python src/fig_transfer.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

RES, FIG = Path("results"), Path("figures")
META = "_c25_active"
K = 100
BLEU, ORANGE, AQUA, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#6b6a66"
TEXTE, TEXTE2, GRILLE = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": TEXTE2, "axes.labelcolor": TEXTE, "xtick.color": TEXTE2,
    "ytick.color": TEXTE2, "text.color": TEXTE, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42,
})


def main():
    a = pd.read_csv(RES / f"transfer_by_module{META}.csv", index_col=0)
    r = pd.read_csv(RES / f"transfer_by_module{META}_relatives.csv", index_col=0)
    rk = pd.read_csv(RES / f"recalibration_k{META}.csv", header=[0, 1], index_col=[0, 1, 2])
    sig = rk[("signalement", "mean")].xs(K, level="k").unstack("methode")
    derive_recal = sig["lomo_recalibre"] - sig["aleatoire"]

    mods = sorted(a.index)[::-1]  # AAA en haut
    y = np.arange(len(mods))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.9), sharey=True, constrained_layout=True)

    # (a) Classement
    ax = axes[0]
    for i, m in enumerate(mods):
        ax.plot([a.loc[m, "auc_aleatoire"], a.loc[m, "auc_lomo"]], [i, i], color=GRILLE, lw=2, zorder=1)
    ax.scatter(a.loc[mods, "auc_aleatoire"], y, s=28, color=BLEU, edgecolor="white", lw=0.8, zorder=3)
    ax.scatter(a.loc[mods, "auc_lomo"], y, s=26, marker="s", color=ORANGE, edgecolor="white", lw=0.8, zorder=3)
    ax.scatter(r.loc[mods, "auc_lomo"], y, s=30, marker="D", facecolor="none", edgecolor=GRIS, lw=1.2, zorder=4)
    ax.set_title("(a) Classement : AUC", loc="left")
    ax.set_xlim(0.70, 0.87)

    # (b) Seuil : part signalée
    ax = axes[1]
    for i, m in enumerate(mods):
        ax.plot([a.loc[m, "signal_aleatoire"], a.loc[m, "signal_lomo"]], [i, i], color=GRILLE, lw=2, zorder=1)
    ax.scatter(a.loc[mods, "taux_risque"], y, s=70, marker="|", color=TEXTE, lw=1.6, zorder=2)
    ax.scatter(a.loc[mods, "signal_aleatoire"], y, s=28, color=BLEU, edgecolor="white", lw=0.8, zorder=3)
    ax.scatter(a.loc[mods, "signal_lomo"], y, s=26, marker="s", color=ORANGE, edgecolor="white", lw=0.8, zorder=3)
    ax.set_title("(b) Seuil : part d'étudiants signalés", loc="left")
    ax.set_xlim(0.2, 1.0)

    # (c) Dérive de signalement après remèdes
    ax = axes[2]
    h = 0.26
    series = [(a["derive_signalement"], ORANGE, "LOMO brut"),
              (derive_recal, AQUA, f"Recalibré ({K} étudiants étiquetés)"),
              (r["derive_signalement"], GRIS, "Variables relatives (sans étiquette)")]
    for j, (s, c, _) in enumerate(series):
        ax.barh(y + (1 - j) * h, s.loc[mods], height=h - 0.04, color=c, zorder=2)
    ax.axvline(0, color=TEXTE2, lw=0.8, zorder=3)
    g = list(mods).index("GGG")
    ax.annotate(f"+{a.loc['GGG', 'derive_signalement']:.2f}", (a.loc["GGG", "derive_signalement"], g + h),
                xytext=(3, 0), textcoords="offset points", va="center", fontsize=7, color=TEXTE2)
    ax.set_title("(c) Dérive de signalement (B − A)", loc="left")
    ax.set_xlim(-0.15, 0.62)

    for ax in axes:
        ax.set_yticks(y, mods)
        ax.grid(axis="x", color=GRILLE, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)

    poignees = [
        Line2D([], [], ls="", marker="o", color=BLEU, mec="white", ms=6, label="(A) Aléatoire, module vu"),
        Line2D([], [], ls="", marker="s", color=ORANGE, mec="white", ms=6, label="(B) LOMO, module non vu"),
        Line2D([], [], ls="", marker="D", mfc="none", mec=GRIS, mew=1.2, ms=5.5,
               label="(B) LOMO, variables relatives"),
        Line2D([], [], ls="", marker="|", color=TEXTE, ms=8, mew=1.6, label="Taux à risque observé"),
        plt.Rectangle((0, 0), 1, 1, color=AQUA, label=f"(B) recalibré sur {K} étudiants"),
        plt.Rectangle((0, 0), 1, 1, color=GRIS, label="(B) variables relatives"),
    ]
    fig.legend(handles=poignees, loc="outside lower center", ncol=3, frameon=False, fontsize=7)

    FIG.mkdir(exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"transfer.{ext}", dpi=200)
    print(f"Écrit : {FIG / 'transfer.pdf'}, {FIG / 'transfer.png'}")


if __name__ == "__main__":
    main()
