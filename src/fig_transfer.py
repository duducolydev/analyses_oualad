#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure T3 — Le classement se transfère, le seuil non (cutoff 25 %, population active).

(a) AUC par module : (A) aléatoire et (B) LOMO, variables absolues ; (B) avec
    variables relatives au rang dans la présentation.
(b) Part d'étudiants signalés (p >= 0,5) : (A), (B) et taux à risque observé.
(c) Remèdes, évalués sur les MÊMES étudiants (ceux non tirés pour Platt) et
    contre la MÊME référence, le modèle (A) à variables absolues : part
    signalée − celle de (A), pour le modèle brut, Platt sur 100 étudiants
    étiquetés (moyenne ± écart-type sur 50 tirages) et les variables relatives.

Lit results/transfer_by_module_c25_active[_relatives].csv et
results/remedes_k_c25.csv (14_remedies.py) ; écrit figures/[en/]transfer.pdf et .png.
Usage : python src/fig_transfer.py [--lang en]
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from figlang import GRILLE, TEXTE, TEXTE2, enregistrer, langue, style

RES = Path("results")
META = "_c25_active"
K = 100
BLEU, ORANGE, AQUA, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#6b6a66"
TXT = {
    "fr": {"a": "(a) Classement : AUC", "b": "(b) Seuil : part d'étudiants signalés",
           "c": "(c) Remèdes : signalés − (A) absolu",
           "A": "(A) Aléatoire, module vu", "B": "(B) LOMO, module non vu",
           "Brel": "(B) LOMO, variables relatives", "obs": "Taux à risque observé",
           "brut": "(B) brut", "recal": f"(B) Platt, {K} étudiants étiquetés", "rel": "(B) variables relatives"},
    "en": {"a": "(a) Ranking: AUC", "b": "(b) Threshold: share flagged",
           "c": "(c) Remedies: flagged − (A) absolute",
           "A": "(A) Random split, module seen", "B": "(B) LOMO, module unseen",
           "Brel": "(B) LOMO, relative features", "obs": "Observed at-risk rate",
           "brut": "(B) raw", "recal": f"(B) Platt, {K} labelled students", "rel": "(B) relative features"},
}


def main():
    lang, dossier = langue(__doc__)
    t = TXT[lang]
    style()
    a = pd.read_csv(RES / f"transfer_by_module{META}.csv", index_col=0)
    r = pd.read_csv(RES / f"transfer_by_module{META}_relatives.csv", index_col=0)
    rk = pd.read_csv(RES / "remedes_k_c25.csv", header=[0, 1], index_col=[0, 1, 2]).xs(K, level="k")
    derive = rk[("derive_ref", "mean")].unstack("regle")
    et = rk[("derive_ref", "std")].unstack("regle")

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
    ax.set_title(t["a"], loc="left")
    ax.set_xlim(0.70, 0.87)

    # (b) Seuil : part signalée
    ax = axes[1]
    for i, m in enumerate(mods):
        ax.plot([a.loc[m, "signal_aleatoire"], a.loc[m, "signal_lomo"]], [i, i], color=GRILLE, lw=2, zorder=1)
    ax.scatter(a.loc[mods, "taux_risque"], y, s=70, marker="|", color=TEXTE, lw=1.6, zorder=2)
    ax.scatter(a.loc[mods, "signal_aleatoire"], y, s=28, color=BLEU, edgecolor="white", lw=0.8, zorder=3)
    ax.scatter(a.loc[mods, "signal_lomo"], y, s=26, marker="s", color=ORANGE, edgecolor="white", lw=0.8, zorder=3)
    ax.set_title(t["b"], loc="left")
    ax.set_xlim(0.2, 1.0)

    # (c) Dérive de signalement après remèdes
    ax = axes[2]
    h = 0.26
    for j, (regle, c) in enumerate([("brut", ORANGE), ("platt", AQUA), ("relatives", GRIS)]):
        ax.barh(y + (1 - j) * h, derive.loc[mods, regle], height=h - 0.04, color=c, zorder=2,
                xerr=et.loc[mods, regle] if regle == "platt" else None,
                error_kw={"elinewidth": 0.8, "ecolor": TEXTE2, "capsize": 0})
    ax.axvline(0, color=TEXTE2, lw=0.8, zorder=3)
    g = list(mods).index("GGG")
    ax.annotate(f"+{derive.loc['GGG', 'brut']:.2f}", (derive.loc["GGG", "brut"], g + h),
                xytext=(3, 0), textcoords="offset points", va="center", fontsize=7, color=TEXTE2)
    ax.set_title(t["c"], loc="left")
    ax.set_xlim(-0.25, 0.62)

    for ax in axes:
        ax.set_yticks(y, mods)
        ax.grid(axis="x", color=GRILLE, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)

    poignees = [
        Line2D([], [], ls="", marker="o", color=BLEU, mec="white", ms=6, label=t["A"]),
        Line2D([], [], ls="", marker="s", color=ORANGE, mec="white", ms=6, label=t["B"]),
        Line2D([], [], ls="", marker="D", mfc="none", mec=GRIS, mew=1.2, ms=5.5, label=t["Brel"]),
        Line2D([], [], ls="", marker="|", color=TEXTE, ms=8, mew=1.6, label=t["obs"]),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label=t["brut"]),
        plt.Rectangle((0, 0), 1, 1, color=AQUA, label=t["recal"]),
        plt.Rectangle((0, 0), 1, 1, color=GRIS, label=t["rel"]),
    ]
    fig.legend(handles=poignees, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    enregistrer(fig, dossier, "transfer")


if __name__ == "__main__":
    main()
