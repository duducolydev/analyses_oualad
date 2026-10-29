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


def ecart_fpr_par_module(gaps: pd.DataFrame) -> pd.DataFrame:
    """Écart de FPR femmes − hommes en (B) LOMO, par module et global, avec IC 95 %
    (07_fairness.py, mêmes tirages bootstrap)."""
    intra = pd.read_csv(RES / "fairness_within_module_c25_active.csv")
    intra = intra[(intra.protocole == "lomo") & (intra.attribut == "genre") & (intra.metrique == "FPR")]
    glob = gaps[(gaps.protocole == "lomo") & (gaps.attribut == "genre") & (gaps.metrique == "FPR")]
    glob = glob.assign(module="__tous__")
    return pd.concat([intra, glob])[["variables", "module", "ecart", "ic_bas", "ic_haut"]]


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
    e = ecart_fpr_par_module(gaps).set_index(["variables", "module"])
    mods = sorted(m for m in e.index.get_level_values("module").unique() if m != "__tous__")[::-1]
    lignes = mods + ["__tous__"]
    y = {m: i if m != "__tous__" else len(mods) + 0.4 for i, m in enumerate(lignes)}
    for v, dy, c, mk, plein in [("absolues", 0.17, ORANGE, "s", True), ("relatives", -0.17, GRIS, "D", False)]:
        for m in lignes:
            r = e.loc[(v, m)]
            ax.plot([r.ic_bas, r.ic_haut], [y[m] + dy] * 2, color=c, lw=1.2, zorder=2)
            marque(ax, r.ecart, y[m] + dy, c, mk, plein)
    ax.axhline(len(mods) - 0.3, color=GRILLE, lw=0.8)
    ax.set_yticks([y[m] for m in lignes], mods + [t["tous"]])
    ax.axvline(0, color=TEXTE2, lw=0.8)
    ax.set_title(t["c"], loc="left")
    ax.set_xlabel(t["cx"])
    ax.set_xlim(-0.17, 0.24)

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
