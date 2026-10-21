#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure T5 — Information temporelle et famille de modèle (population active).

(a, b) Gain d'AUC par rapport à la forêt agrégée (ligne zéro), en (A)
       aléatoire et (B) LOMO, selon la fenêtre observée, variables relatives :
       gradient boosting agrégé, gradient boosting avec séquence hebdomadaire, GRU.
(c)    Dérive moyenne |part signalée (B) − (A)| au cutoff 25 %, variables
       absolues vs relatives, pour chaque modèle.

Lit results/sequence_by_module.csv, results/early_curve_by_module*.csv et les
prédictions de la forêt ; écrit figures/[en/]sequence.pdf et .png.
Usage : python src/fig_sequence.py [--lang en]
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
ORDRE = ["rf_agg", "hgb_agg", "hgb_seq", "gru"]
COULEUR = {"rf_agg": "#6b6a66", "hgb_agg": "#2a78d6", "hgb_seq": "#1baf7a", "gru": "#eda100"}
MARQUEUR = {"rf_agg": "", "hgb_agg": "o", "hgb_seq": "s", "gru": "^"}
ORANGE, GRIS = "#eb6834", "#6b6a66"
TXT = {
    "fr": {"noms": {"rf_agg": "Forêt, agrégées (référence)", "hgb_agg": "Boosting, agrégées",
                    "hgb_seq": "Boosting, + séquence", "gru": "GRU, séquence + statiques"},
           "court": {"rf_agg": "Forêt", "hgb_agg": "Boosting", "hgb_seq": "Boosting + séq.", "gru": "GRU"},
           "a": "(a) Gain d'AUC, (A) module vu", "b": "(b) Gain d'AUC, (B) non vu",
           "c": "(c) Dérive du seuil à 25 %", "x": "Fenêtre observée (% session)",
           "y": "AUC − AUC de la forêt",
           "cx": "|signalés (B) − (A)| moyen", "abs": "Variables absolues", "rel": "Variables relatives"},
    "en": {"noms": {"rf_agg": "Forest, aggregated (reference)", "hgb_agg": "Boosting, aggregated",
                    "hgb_seq": "Boosting, + sequence", "gru": "GRU, sequence + static"},
           "court": {"rf_agg": "Forest", "hgb_agg": "Boosting", "hgb_seq": "Boosting + seq.", "gru": "GRU"},
           "a": "(a) AUC gain, (A) module seen", "b": "(b) AUC gain, (B) unseen",
           "c": "(c) Threshold drift at 25%", "x": "Observed window (% of course)",
           "y": "AUC − forest AUC",
           "cx": "Mean |flagged (B) − (A)|", "abs": "Absolute features", "rel": "Relative features"},
}


def donnees() -> pd.DataFrame:
    """Une ligne par (cutoff, variables, modèle, protocole, module) : AUC et part signalée."""
    seq = pd.read_csv(RES / "sequence_by_module.csv")[
        ["cutoff", "variables", "modele", "protocole", "module", "AUC", "signalement"]]
    rf = []
    for var, suf, tag in [("absolues", "", "rf"), ("relatives", "_relatives", "rf_rel")]:
        d = pd.read_csv(RES / f"early_curve_by_module{suf}.csv")
        d = d[d.population == "active"]
        for cut in sorted(d.cutoff.unique()):
            p = pd.read_parquet(RES / "preds" / f"preds_{tag}_active_c{round(cut * 100):02d}.parquet")
            for proto, col in [("aleatoire", "p_aleatoire"), ("lomo", "p_lomo")]:
                sig = (p[col] >= 0.5).groupby(p.code_module).mean()
                auc = d[(d.cutoff == cut) & (d.protocole == proto)].set_index("module")["AUC"]
                rf += [{"cutoff": cut, "variables": var, "modele": "rf_agg", "protocole": proto,
                        "module": m, "AUC": auc[m], "signalement": sig[m]} for m in auc.index]
    return pd.concat([seq, pd.DataFrame(rf)], ignore_index=True)


def derive_moyenne(d: pd.DataFrame) -> pd.Series:
    s = d.pivot_table(index=["variables", "modele", "cutoff", "module"], columns="protocole", values="signalement")
    return (s["lomo"] - s["aleatoire"]).abs().groupby(level=["variables", "modele", "cutoff"]).mean()


def main():
    lang, dossier = langue(__doc__)
    t = TXT[lang]
    style()
    d = donnees()
    modeles = [m for m in ORDRE if m in set(d.modele)]
    auc = d[d.variables == "relatives"].groupby(["protocole", "modele", "cutoff"])["AUC"].mean()

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.7), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for ax, proto, cle in [(axes[0], "aleatoire", "a"), (axes[1], "lomo", "b")]:
        ref = auc.loc[(proto, "rf_agg")]
        ax.axhline(0, color=COULEUR["rf_agg"], lw=1.2, ls="--", zorder=1)
        for m in [m for m in modeles if m != "rf_agg"]:
            s = auc.loc[(proto, m)] - ref
            ax.plot(s.index * 100, s.values, "-", lw=1.8, color=COULEUR[m], marker=MARQUEUR[m], ms=4.5,
                    mec="white", mew=0.6, zorder=2)
        ax.set_title(t[cle], loc="left")
        ax.set_xlabel(t["x"])
        ax.set_xticks(sorted(ref.index * 100), [f"{c * 100:.0f}" for c in sorted(ref.index)], fontsize=7)
        ax.set_xlim(6, 54)
        ax.set_ylim(-0.012, 0.016)
        ax.grid(axis="y", color=GRILLE, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(t["y"])

    ax = axes[2]
    dm = derive_moyenne(d).xs(0.25, level="cutoff")
    y = np.arange(len(modeles))[::-1]
    for yi, m in zip(y, modeles):
        a, r = dm.get(("absolues", m), np.nan), dm.get(("relatives", m), np.nan)
        ax.plot([a, r], [yi, yi], color=GRILLE, lw=2, zorder=1)
        ax.scatter(a, yi, s=28, marker="s", color=ORANGE, edgecolor="white", lw=0.8, zorder=3)
        ax.scatter(r, yi, s=30, marker="D", facecolor="none", edgecolor=GRIS, lw=1.2, zorder=3)
    ax.set_yticks(y, [t["court"][m] for m in modeles])
    ax.tick_params(axis="y", length=0)
    ax.set_title(t["c"], loc="left")
    ax.set_xlabel(t["cx"])
    ax.set_xlim(0, None)
    ax.grid(axis="x", color=GRILLE, lw=0.6)
    ax.set_axisbelow(True)

    poignees = [Line2D([], [], color=COULEUR[m], marker=MARQUEUR[m], ls="--" if m == "rf_agg" else "-",
                       lw=1.6, ms=5, mec="white" if m != "rf_agg" else None, label=t["noms"][m]) for m in modeles]
    poignees += [Line2D([], [], ls="", marker="s", color=ORANGE, mec="white", ms=6, label="(c) " + t["abs"]),
                 Line2D([], [], ls="", marker="D", mfc="none", mec=GRIS, mew=1.2, ms=5.5, label="(c) " + t["rel"])]
    fig.legend(handles=poignees, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    enregistrer(fig, dossier, "sequence")


if __name__ == "__main__":
    main()
