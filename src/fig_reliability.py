#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure (relecture R2, M4) — Diagrammes de fiabilité au cutoff 25 % : part
observée d'étudiants à risque selon la probabilité prédite (10 classes de même
largeur ; classes de moins de 30 étudiants omises), tous modules confondus et
GGG. (A) et (B) à variables absolues, (B) à variables relatives. Avec des
poids de classe équilibrés, les courbes sont sous la diagonale même en (A) :
c'est l'écart entre (B) et (A) qui mesure l'effet du transfert.

Lit results/reliability_c25.csv (17_ap_reliability.py) ; écrit figures/[en/]reliability.pdf et .png.
Usage : python src/fig_reliability.py [--lang en]
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from figlang import GRILLE, TEXTE2, enregistrer, langue, style

RES = Path("results")
N_MIN = 30
SERIES = [("absolues", "aleatoire", "#2a78d6", "o", True, "A"),
          ("absolues", "lomo", "#eb6834", "s", True, "Babs"),
          ("relatives", "lomo", "#6b6a66", "D", False, "Brel")]
TXT = {
    "fr": {"tous": "(a) Tous les modules", "GGG": "(b) Module GGG", "x": "Probabilité prédite",
           "y": "Part observée à risque", "A": "(A) Aléatoire, var. absolues",
           "Babs": "(B) LOMO, var. absolues", "Brel": "(B) LOMO, var. relatives", "diag": "Calibration parfaite"},
    "en": {"tous": "(a) All modules", "GGG": "(b) Module GGG", "x": "Predicted probability",
           "y": "Observed at-risk share", "A": "(A) Random, absolute features",
           "Babs": "(B) LOMO, absolute features", "Brel": "(B) LOMO, relative features", "diag": "Perfect calibration"},
}


def main():
    lang, dossier = langue(__doc__)
    t = TXT[lang]
    style()
    d = pd.read_csv(RES / "reliability_c25.csv")
    d = d[d.n >= N_MIN]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), constrained_layout=True, sharey=True)
    for ax, portee in zip(axes, ["tous", "GGG"]):
        ax.plot([0, 1], [0, 1], color=TEXTE2, lw=0.8, ls=":", zorder=1)
        for v, proto, c, m, plein, cle in SERIES:
            s = d[(d.portee == portee) & (d.variables == v) & (d.protocole == proto)].sort_values("p_moyen")
            ax.plot(s.p_moyen, s.observe, "-", color=c, lw=1.6, zorder=2)
            ax.scatter(s.p_moyen, s.observe, s=22, marker=m, color=c if plein else "none",
                       edgecolor="white" if plein else c, lw=0.8 if plein else 1.2, zorder=3)
        ax.set_title(t[portee], loc="left")
        ax.set_xlabel(t["x"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(color=GRILLE, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(t["y"])
    poignees = [Line2D([], [], color=c, marker=m, mfc=c if plein else "none", mec="white" if plein else c,
                       lw=1.6, ms=5, label=t[cle]) for _, _, c, m, plein, cle in SERIES]
    poignees.append(Line2D([], [], color=TEXTE2, lw=0.8, ls=":", label=t["diag"]))
    fig.legend(handles=poignees, loc="outside lower center", ncol=4, frameon=False, fontsize=7)
    enregistrer(fig, dossier, "reliability")


if __name__ == "__main__":
    main()
