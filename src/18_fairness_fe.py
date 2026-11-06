#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relectures R1/R2 — Équité : effet de composition formalisé et intersectionnalité.

1. Effet de composition (R2, mineure 7). Sur les étudiants NON à risque (faux
   positifs possibles), modèle de probabilité linéaire du signalement :
     signalé ~ femme                        (écart global, comme la Fig. 4)
     signalé ~ femme + effets fixes module   (écart à module donné)
   Si l'écart global disparaît avec les effets fixes, il vient de la
   composition des modules. IC 95 % par bootstrap groupé par étudiant (1000).
   Configurations : (A) absolues, (B) absolues, (B) relatives.
2. Intersectionnalité (R1). Taux de faux positifs de (B) pour genre × handicap
   et genre × quintile d'IMD (Q1, Q5), avec les effectifs de négatifs et des IC.

Sorties : results/fairness_fe_c25.csv, results/fairness_intersection_c25.csv
Usage : python src/18_fairness_fe.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("results")
TARGET = "at_risk"
CLE = ["code_module", "code_presentation", "id_student"]
N_BOOT = 1000
QUINTILE = {"0-10%": "Q1", "10-20": "Q1", "10-20%": "Q1", "80-90%": "Q5", "90-100%": "Q5"}
CONFIGS = [("absolues", "aleatoire", "rf", "p_aleatoire"), ("absolues", "lomo", "rf", "p_lomo"),
           ("relatives", "lomo", "rf_rel", "p_lomo")]


def coef_femme(f, femme, modules, w):
    """Coefficient de « femme » par moindres carrés pondérés, sans puis avec
    effets fixes module (centrage intra-module pondéré)."""
    def mco(x, y):
        xc, yc = x - np.average(x, weights=w), y - np.average(y, weights=w)
        return (w * xc * yc).sum() / (w * xc * xc).sum()

    brut = mco(femme, f)
    xi, yi = femme.astype(float).copy(), f.astype(float).copy()
    for m in np.unique(modules):
        k = modules == m
        if w[k].sum() > 0:
            xi[k] -= np.average(femme[k], weights=w[k])
            yi[k] -= np.average(f[k], weights=w[k])
    fe = (w * xi * yi).sum() / (w * xi * xi).sum()
    return brut, fe


def main():
    info = pd.read_parquet("data/cache/studentInfo.parquet")[CLE + ["gender", "imd_band", "disability"]]
    rng = np.random.default_rng(0)
    lignes, inter = [], []
    base = pd.read_parquet(RES / "preds" / "preds_rf_active_c25.parquet").merge(info, on=CLE, how="left")
    neg = (base[TARGET] == 0).values
    etu, idx = np.unique(base.loc[neg, "id_student"].values, return_inverse=True)
    W = [np.bincount(rng.integers(0, len(etu), len(etu)), minlength=len(etu))[idx].astype(float)
         for _ in range(N_BOOT)]
    femme = (base.loc[neg, "gender"] == "F").values.astype(float)
    modules = base.loc[neg, "code_module"].values
    for v, proto, tag, col in CONFIGS:
        p = pd.read_parquet(RES / "preds" / f"preds_{tag}_active_c25.parquet")
        assert (p[CLE].values == base[CLE].values).all()
        f = (p[col].values >= 0.5)[neg].astype(float)
        b0, fe0 = coef_femme(f, femme, modules, np.ones_like(f))
        bs = np.array([coef_femme(f, femme, modules, w) for w in W])
        lignes.append({"variables": v, "protocole": proto,
                       "ecart_global": b0, "ecart_global_bas": np.percentile(bs[:, 0], 2.5),
                       "ecart_global_haut": np.percentile(bs[:, 0], 97.5),
                       "ecart_effets_fixes": fe0, "ecart_ef_bas": np.percentile(bs[:, 1], 2.5),
                       "ecart_ef_haut": np.percentile(bs[:, 1], 97.5)})

        if proto == "lomo":
            d = base.loc[neg, ["gender", "disability", "imd_band"]].assign(f=f)
            d["imd_q"] = d.imd_band.map(QUINTILE)
            for nom, groupes in [("genre x handicap", ["gender", "disability"]), ("genre x IMD", ["gender", "imd_q"])]:
                for cle_g, g in d.dropna(subset=groupes).groupby(groupes):
                    k = g.index.values
                    pos = np.searchsorted(base.index[neg], k)
                    bs_g = [np.average(f[pos], weights=w[pos]) if w[pos].sum() > 0 else np.nan for w in W]
                    inter.append({"variables": v, "croisement": nom, "groupe": " × ".join(map(str, cle_g)),
                                  "n_negatifs": len(k), "FPR": g.f.mean(),
                                  "ic_bas": np.nanpercentile(bs_g, 2.5), "ic_haut": np.nanpercentile(bs_g, 97.5)})
    fe = pd.DataFrame(lignes)
    fe.to_csv(RES / "fairness_fe_c25.csv", index=False)
    it = pd.DataFrame(inter)
    it.to_csv(RES / "fairness_intersection_c25.csv", index=False)
    pd.set_option("display.width", 200)
    print("Écart de taux de faux positifs femmes − hommes (probabilité linéaire, IC 95 %) :")
    print(fe.round(3).to_string(index=False))
    print("\nIntersectionnalité, (B) :")
    print(it.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
