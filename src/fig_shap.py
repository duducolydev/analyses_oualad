#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure T4 — Mêmes variables, mauvais seuils (cutoff 25 %, population active).

(a) Concordance des importances SHAP entre le modèle transféré (LOMO) et un
    modèle local (entraîné sur le module), par module : Spearman sur les 18
    variables, variables absolues vs relatives.
(b, c) Décomposition de la dérive : SHAP signé moyen du modèle transféré sur
    les étudiants du module, par famille de variables (écart de risque prédit
    à la population d'entraînement) ; trait noir = total.

Lit results/shap_stability_c25_active.csv et results/shap_rows_c25_active_*.parquet ;
écrit figures/[en/]shap.pdf et .png ; écrit results/shap_drift_decomposition_c25_active.csv.
Usage : python src/fig_shap.py [--lang en]
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
FAMILLES = {
    "Notes précoces": ["mean_early_score", "n_submissions"],
    "Engagement VLE": ["total_clicks", "active_days", "distinct_sites",
                       "clicks_per_active_day", "first_activity", "last_activity"],
}
COULEUR_FAM = {"Notes précoces": "#2a78d6", "Engagement VLE": "#1baf7a",
               "Démographie, inscription": "#b9b8b2"}
ORANGE, GRIS = "#eb6834", "#6b6a66"
TXT = {
    "fr": {"a": "(a) Concordance transféré / local", "ax": "Spearman des importances SHAP",
           "b": "(b) Dérive, variables absolues", "c": "(c) Dérive, variables relatives",
           "bx": "SHAP signé moyen par famille",
           "abs": "(a) Variables absolues", "rel": "(a) Variables relatives",
           "fam": {"Notes précoces": "(b, c) Notes précoces", "Engagement VLE": "(b, c) Engagement VLE",
                   "Démographie, inscription": "(b, c) Démographie, inscription"},
           "total": "(b, c) Total : écart à la population d'entraînement"},
    "en": {"a": "(a) Transferred vs local agreement", "ax": "Spearman of SHAP importances",
           "b": "(b) Drift, absolute features", "c": "(c) Drift, relative features",
           "bx": "Mean signed SHAP per family",
           "abs": "(a) Absolute features", "rel": "(a) Relative features",
           "fam": {"Notes précoces": "(b, c) Early scores", "Engagement VLE": "(b, c) VLE engagement",
                   "Démographie, inscription": "(b, c) Demographics, registration"},
           "total": "(b, c) Total: shift from the training population"},
}


def decomposition(variante: str) -> pd.DataFrame:
    s = pd.read_parquet(RES / f"shap_rows{META}_{variante}.parquet")
    feats = [c for c in s.columns if c not in ["ligne", "code_module", "modele"]]
    t = s[s["modele"] == "transfere"].groupby("code_module")[feats].mean()
    d = pd.DataFrame({f: t[cols].sum(axis=1) for f, cols in FAMILLES.items()})
    autres = [c for c in feats if not any(c in cols for cols in FAMILLES.values())]
    d["Démographie, inscription"] = t[autres].sum(axis=1)
    d["total"] = t.sum(axis=1)
    return d


def barres_divergentes(ax, d, mods):
    for i, m in enumerate(mods):
        pos, neg = 0.0, 0.0
        for fam, c in COULEUR_FAM.items():
            v = d.loc[m, fam]
            gauche = pos if v >= 0 else neg + v
            ax.barh(i, abs(v), left=gauche, height=0.6, color=c, edgecolor="white", lw=1, zorder=2)
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        ax.plot([d.loc[m, "total"]] * 2, [i - 0.38, i + 0.38], color=TEXTE, lw=1.6, zorder=3)
    ax.axvline(0, color=TEXTE2, lw=0.8, zorder=1)
    ax.set_xlim(-0.2, 0.36)


def main():
    lang, dossier = langue(__doc__)
    t = TXT[lang]
    style()
    stab = pd.read_csv(RES / f"shap_stability{META}.csv")
    conc = stab.pivot(index="module", columns="variables", values="spearman_transfere_local")
    dec = {v: decomposition(v) for v in ["absolues", "relatives"]}
    pd.concat(dec, names=["variables"]).to_csv(RES / f"shap_drift_decomposition{META}.csv")

    mods = sorted(conc.index)[::-1]
    y = np.arange(len(mods))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.9), sharey=True, constrained_layout=True)

    ax = axes[0]
    for i, m in enumerate(mods):
        ax.plot(conc.loc[m, ["absolues", "relatives"]], [i, i], color=GRILLE, lw=2, zorder=1)
    ax.scatter(conc.loc[mods, "absolues"], y, s=26, marker="s", color=ORANGE, edgecolor="white", lw=0.8, zorder=3)
    ax.scatter(conc.loc[mods, "relatives"], y, s=30, marker="D", facecolor="none", edgecolor=GRIS, lw=1.2, zorder=3)
    ax.set_title(t["a"], loc="left")
    ax.set_xlabel(t["ax"])
    ax.set_xlim(0.3, 1.0)

    for ax, v, cle in [(axes[1], "absolues", "b"), (axes[2], "relatives", "c")]:
        barres_divergentes(ax, dec[v], mods)
        ax.set_title(t[cle], loc="left")
        ax.set_xlabel(t["bx"])

    for ax in axes:
        ax.set_yticks(y, mods)
        ax.grid(axis="x", color=GRILLE, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)

    poignees = [Line2D([], [], ls="", marker="s", color=ORANGE, mec="white", ms=6, label=t["abs"]),
                Line2D([], [], ls="", marker="D", mfc="none", mec=GRIS, mew=1.2, ms=5.5, label=t["rel"])]
    poignees += [plt.Rectangle((0, 0), 1, 1, color=c, label=t["fam"][f]) for f, c in COULEUR_FAM.items()]
    poignees.append(Line2D([], [], color=TEXTE, lw=1.6, label=t["total"]))
    fig.legend(handles=poignees, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    enregistrer(fig, dossier, "shap")


if __name__ == "__main__":
    main()
