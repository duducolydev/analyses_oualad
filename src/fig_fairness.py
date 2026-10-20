#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure T6 — Équité : écarts de taux d'erreur entre groupes (cutoff 25 %, population active).

(a, b) Écart de TPR et de FPR entre groupes contrastés (F − M, IMD Q1 − Q5,
       handicap O − N), IC 95 % bootstrap groupé par étudiant ; (A) aléatoire
       variables absolues, (B) LOMO absolues, (B) LOMO relatives.
(c)    Écart de FPR femmes − hommes calculé à l'intérieur de chaque module, en
       (B) LOMO : l'écart global des variables absolues est un effet de
       composition (femmes concentrées dans BBB et GGG, modules qui dérivent).

Lit results/fairness_gaps_c25_active.csv et les prédictions ; écrit figures/fairness.pdf et .png.
Usage : python src/fig_fairness.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

RES, FIG = Path("results"), Path("figures")
CLE = ["code_module", "code_presentation", "id_student"]
BLEU, ORANGE, GRIS = "#2a78d6", "#eb6834", "#6b6a66"
TEXTE, TEXTE2, GRILLE = "#0b0b0b", "#52514e", "#e4e3df"
CONFIGS = [("absolues", "aleatoire", "(A) Aléatoire, var. absolues", BLEU, "o", True),
           ("absolues", "lomo", "(B) LOMO, var. absolues", ORANGE, "s", True),
           ("relatives", "lomo", "(B) LOMO, var. relatives", GRIS, "D", False)]
ATTRIBUTS = [("genre", "Femmes − hommes"), ("imd", "IMD Q1 − Q5"), ("handicap", "Handicap O − N")]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": TEXTE2, "axes.labelcolor": TEXTE, "xtick.color": TEXTE2,
    "ytick.color": TEXTE2, "text.color": TEXTE, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42,
})


def marque(ax, x, y, c, m, plein, **kw):
    ax.scatter(x, y, s=28, marker=m, color=c if plein else "none",
               edgecolor="white" if plein else c, lw=0.8 if plein else 1.2, zorder=3, **kw)


def ecart_fpr_par_module():
    info = pd.read_parquet("data/cache/studentInfo.parquet")[CLE + ["gender"]]
    out = {}
    for v, tag in [("absolues", "rf"), ("relatives", "rf_rel")]:
        p = pd.read_parquet(RES / "preds" / f"preds_{tag}_active_c25.parquet").merge(info, on=CLE)
        neg = p[p["at_risk"] == 0].assign(f=lambda d: d["p_lomo"] >= 0.5)
        t = neg.groupby(["code_module", "gender"])["f"].mean().unstack()
        out[v] = t["F"] - t["M"]
        out[v + "_global"] = neg[neg.gender == "F"]["f"].mean() - neg[neg.gender == "M"]["f"].mean()
    return out


def main():
    gaps = pd.read_csv(RES / "fairness_gaps_c25_active.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 1.05]})

    pas = 0.22
    for ax, met, titre in [(axes[0], "TPR", "(a) Écart de rappel (TPR)"),
                           (axes[1], "FPR", "(b) Écart de faux positifs (FPR)")]:
        for i, (att, _) in enumerate(ATTRIBUTS):
            y0 = len(ATTRIBUTS) - 1 - i
            for j, (v, proto, _, c, m, plein) in enumerate(CONFIGS):
                r = gaps[(gaps.variables == v) & (gaps.protocole == proto)
                         & (gaps.attribut == att) & (gaps.metrique == met)].iloc[0]
                yy = y0 + (1 - j) * pas
                ax.plot([r.ic_bas, r.ic_haut], [yy, yy], color=c, lw=1.4, zorder=2)
                marque(ax, r.ecart, yy, c, m, plein)
        ax.axvline(0, color=TEXTE2, lw=0.8)
        ax.set_yticks(range(len(ATTRIBUTS)), [lib for _, lib in ATTRIBUTS][::-1])
        ax.set_title(titre, loc="left")
        ax.set_xlabel("Écart entre groupes (IC 95 %)")
        ax.set_xlim(-0.12, 0.24)
    axes[1].set_yticklabels([])

    ax = axes[2]
    e = ecart_fpr_par_module()
    mods = sorted(e["absolues"].index)[::-1]
    y = np.arange(len(mods))
    for i, mod in enumerate(mods):
        ax.plot([e["absolues"][mod], e["relatives"][mod]], [i, i], color=GRILLE, lw=2, zorder=1)
    marque(ax, e["absolues"][mods], y, ORANGE, "s", True)
    marque(ax, e["relatives"][mods], y, GRIS, "D", False)
    yg = len(mods) + 0.3
    marque(ax, e["absolues_global"], yg, ORANGE, "s", True)
    marque(ax, e["relatives_global"], yg, GRIS, "D", False)
    ax.plot([e["absolues_global"], e["relatives_global"]], [yg, yg], color=GRILLE, lw=2, zorder=1)
    ax.annotate(f"+{e['absolues_global']:.2f}", (e["absolues_global"], yg), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=7, color=TEXTE2)
    ax.axhline(len(mods) - 0.35, color=GRILLE, lw=0.8)
    ax.set_yticks(list(y) + [yg], mods + ["Tous"])
    ax.axvline(0, color=TEXTE2, lw=0.8)
    ax.set_title("(c) FPR F − H par module", loc="left")
    ax.set_xlabel("Écart de FPR, protocole (B) LOMO")
    ax.set_xlim(-0.12, 0.24)

    for ax in axes:
        ax.grid(axis="x", color=GRILLE, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)

    poignees = [Line2D([], [], ls="-", lw=1.4, marker=m, color=c, mfc=c if plein else "none",
                       mec="white" if plein else c, mew=0.8 if plein else 1.2, ms=5.5, label=lib)
                for _, _, lib, c, m, plein in CONFIGS]
    fig.legend(handles=poignees, loc="outside lower center", ncol=3, frameon=False, fontsize=7)

    FIG.mkdir(exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"fairness.{ext}", dpi=200)
    print(f"Écrit : {FIG / 'fairness.pdf'}, {FIG / 'fairness.png'}")


if __name__ == "__main__":
    main()
