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

Lit results/fairness_gaps_c25_active.csv et les prédictions ; écrit figures/[en/]fairness.pdf et .png.
Usage : python src/fig_fairness.py [--lang en]
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from figlang import GRILLE, TEXTE2, enregistrer, langue, style

RES = Path("results")
CLE = ["code_module", "code_presentation", "id_student"]
BLEU, ORANGE, GRIS = "#2a78d6", "#eb6834", "#6b6a66"
CONFIGS = [("absolues", "aleatoire", "A", BLEU, "o", True),
           ("absolues", "lomo", "Babs", ORANGE, "s", True),
           ("relatives", "lomo", "Brel", GRIS, "D", False)]
ATTRIBUTS = ["genre", "imd", "handicap"]
TXT = {
    "fr": {"att": {"genre": "Femmes − hommes", "imd": "IMD Q1 − Q5", "handicap": "Handicap O − N"},
           "a": "(a) Écart de rappel (TPR)", "b": "(b) Écart de faux positifs (FPR)",
           "c": "(c) FPR F − H par module", "x": "Écart entre groupes (IC 95 %)",
           "cx": "Écart de FPR, protocole (B) LOMO", "tous": "Tous",
           "A": "(A) Aléatoire, var. absolues", "Babs": "(B) LOMO, var. absolues",
           "Brel": "(B) LOMO, var. relatives"},
    "en": {"att": {"genre": "Female − male", "imd": "IMD Q1 − Q5", "handicap": "Disability yes − no"},
           "a": "(a) Recall (TPR) gap", "b": "(b) False-positive rate gap",
           "c": "(c) FPR F − M per module", "x": "Between-group gap (95% CI)",
           "cx": "FPR gap, protocol (B) LOMO", "tous": "All",
           "A": "(A) Random, absolute features", "Babs": "(B) LOMO, absolute features",
           "Brel": "(B) LOMO, relative features"},
}


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
    lang, dossier = langue(__doc__)
    t = TXT[lang]
    style()
    gaps = pd.read_csv(RES / "fairness_gaps_c25_active.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 1.05]})

    pas = 0.22
    for ax, met, cle in [(axes[0], "TPR", "a"), (axes[1], "FPR", "b")]:
        for i, att in enumerate(ATTRIBUTS):
            y0 = len(ATTRIBUTS) - 1 - i
            for j, (v, proto, _, c, m, plein) in enumerate(CONFIGS):
                r = gaps[(gaps.variables == v) & (gaps.protocole == proto)
                         & (gaps.attribut == att) & (gaps.metrique == met)].iloc[0]
                yy = y0 + (1 - j) * pas
                ax.plot([r.ic_bas, r.ic_haut], [yy, yy], color=c, lw=1.4, zorder=2)
                marque(ax, r.ecart, yy, c, m, plein)
        ax.axvline(0, color=TEXTE2, lw=0.8)
        ax.set_yticks(range(len(ATTRIBUTS)), [t["att"][a] for a in ATTRIBUTS][::-1])
        ax.set_title(t[cle], loc="left")
        ax.set_xlabel(t["x"])
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
    ax.set_yticks(list(y) + [yg], mods + [t["tous"]])
    ax.axvline(0, color=TEXTE2, lw=0.8)
    ax.set_title(t["c"], loc="left")
    ax.set_xlabel(t["cx"])
    ax.set_xlim(-0.12, 0.24)

    for ax in axes:
        ax.grid(axis="x", color=GRILLE, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)

    poignees = [Line2D([], [], ls="-", lw=1.4, marker=m, color=c, mfc=c if plein else "none",
                       mec="white" if plein else c, mew=0.8 if plein else 1.2, ms=5.5, label=t[k])
                for _, _, k, c, m, plein in CONFIGS]
    fig.legend(handles=poignees, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    enregistrer(fig, dossier, "fairness")


if __name__ == "__main__":
    main()
