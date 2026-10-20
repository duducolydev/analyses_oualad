#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure T2 — Performance en fonction de la fenêtre d'observation (population active).

Trait plein : modèle principal, variables relatives (rang dans la présentation).
Tirets      : référence, variables absolues (usage courant dans la littérature).
Les deux protocoles sont agrégés de la MÊME façon : moyenne des métriques
calculées à l'intérieur de chaque module (en (A), sur les prédictions hors-pli).
Bande : étendue min–max des modules en LOMO (variables relatives).

Lit results/early_curve_by_module[_relatives].csv ; écrit figures/early_curve.pdf et .png.
Usage : python src/fig_early_curve.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

RES, FIG = Path("results"), Path("figures")
POP = "active"
COULEUR = {"aleatoire": "#2a78d6", "lomo": "#eb6834"}
MARQUEUR = {"aleatoire": "o", "lomo": "s"}
LIBELLE = {"aleatoire": "(A) Aléatoire groupé par étudiant",
           "lomo": "(B) Leave-one-module-out"}
TEXTE, TEXTE2, GRILLE = "#0b0b0b", "#52514e", "#e4e3df"
METRIQUES = ["AUC", "Rappel", "Précision"]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "axes.edgecolor": TEXTE2, "axes.labelcolor": TEXTE, "xtick.color": TEXTE2,
    "ytick.color": TEXTE2, "text.color": TEXTE, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42,
})


def moyennes(suffixe):
    mod = pd.read_csv(RES / f"early_curve_by_module{suffixe}.csv")
    mod = mod[mod.population == POP]
    return mod, mod.groupby(["protocole", "cutoff"])[METRIQUES].mean()


def etiquettes_fin(ax, valeurs):
    """Valeur en bout de courbe ; écarte verticalement les étiquettes proches."""
    ordre = sorted(valeurs, key=lambda v: v[1])
    for (x, y), dy in zip(ordre, [-4, 4]):
        ax.annotate(f"{y:.3f}", (x, y), xytext=(5, dy), textcoords="offset points",
                    va="center", fontsize=7, color=TEXTE2)


def main():
    mod_rel, rel = moyennes("_relatives")
    _, abso = moyennes("")
    bande = mod_rel[mod_rel.protocole == "lomo"].groupby("cutoff")["AUC"].agg(["min", "max"])
    cutoffs = sorted(mod_rel.cutoff.unique())

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6), constrained_layout=True)
    for ax, met in zip(axes, METRIQUES):
        if met == "AUC":
            ax.fill_between(bande.index * 100, bande["min"], bande["max"],
                            color=COULEUR["lomo"], alpha=0.12, linewidth=0)
        fins = []
        for proto in ["aleatoire", "lomo"]:
            d = abso.loc[proto, met]
            ax.plot(d.index * 100, d.values, "--", lw=1.1, color=COULEUR[proto], alpha=0.8)
            d = rel.loc[proto, met]
            ax.plot(d.index * 100, d.values, "-", lw=2, color=COULEUR[proto],
                    marker=MARQUEUR[proto], ms=4.5, mec="white", mew=0.8)
            fins.append((d.index[-1] * 100, d.values[-1]))
        etiquettes_fin(ax, fins)
        ax.set_title(met, loc="left", color=TEXTE)
        ax.set_xlabel("Fenêtre observée (% de la session)")
        ax.set_xticks([c * 100 for c in cutoffs])
        ax.set_xlim(6, 60)
        ax.grid(axis="y", color=GRILLE, lw=0.6)
        ax.set_axisbelow(True)

    poignees = [Line2D([], [], color=COULEUR[p], marker=MARQUEUR[p], lw=2, ms=4.5,
                       mec="white", label=LIBELLE[p]) for p in ["aleatoire", "lomo"]]
    poignees += [Line2D([], [], color=TEXTE2, lw=2, label="Variables relatives (modèle principal)"),
                 Line2D([], [], color=TEXTE2, lw=1.1, ls="--", label="Variables absolues (référence)"),
                 plt.Rectangle((0, 0), 1, 1, color=COULEUR["lomo"], alpha=0.12,
                               label="(B) min–max des modules")]
    fig.legend(handles=poignees, loc="outside lower center", ncol=3, frameon=False, fontsize=7)

    FIG.mkdir(exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"early_curve.{ext}", dpi=200)
    print(f"Écrit : {FIG / 'early_curve.pdf'}, {FIG / 'early_curve.png'}")


if __name__ == "__main__":
    main()
