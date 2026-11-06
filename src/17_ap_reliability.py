#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relectures R1/R2 — Average Precision et diagrammes de fiabilité, à partir des
prédictions enregistrées (aucun réentraînement).

1. Average Precision (aire sous la courbe précision-rappel) par module, en (A)
   et (B), pour toutes les prédictions disponibles au cutoff 25 % : forêt
   (variables absolues / relatives, cibles « Withdrawn ou Fail » et
   « Withdrawn seul »), boosting et GRU. Utile surtout quand la classe est rare
   (Withdrawn seul : 13,7 %).
2. Fiabilité : 10 classes de probabilité prédite de même largeur ; part
   observée d'étudiants à risque dans chaque classe, pour (A) et (B), variables
   absolues et relatives ; tous modules confondus et GGG.

Sorties : results/average_precision_c25.csv, results/reliability_c25.csv
Usage : python src/17_ap_reliability.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

RES = Path("results")
TARGET = "at_risk"
FICHIERS = {  # nom -> (fichier de prédictions, cible, variables, modèle)
    "rf": ("preds_rf_active_c25", "withdrawn_fail", "absolues", "forêt"),
    "rf_rel": ("preds_rf_rel_active_c25", "withdrawn_fail", "relatives", "forêt"),
    "rf_wd": ("preds_rf_wd_active_c25", "withdrawn", "absolues", "forêt"),
    "rf_rel_wd": ("preds_rf_rel_wd_active_c25", "withdrawn", "relatives", "forêt"),
    "hgb": ("preds_hgb_agg_active_c25", "withdrawn_fail", "absolues", "boosting"),
    "hgb_rel": ("preds_hgb_agg_rel_active_c25", "withdrawn_fail", "relatives", "boosting"),
    "gru": ("preds_gru_active_c25", "withdrawn_fail", "absolues", "gru"),
    "gru_rel": ("preds_gru_rel_active_c25", "withdrawn_fail", "relatives", "gru"),
}
BORNES = np.linspace(0, 1, 11)


def main():
    ap_lignes, fiab = [], []
    for nom, (f, cible, var, modele) in FICHIERS.items():
        chemin = RES / "preds" / f"{f}.parquet"
        if not chemin.exists():
            print(f"absent : {chemin}")
            continue
        p = pd.read_parquet(chemin)
        for mod, g in p.groupby("code_module"):
            y = g[TARGET].values
            ap_lignes.append({"source": nom, "cible": cible, "variables": var, "modele": modele, "module": mod,
                              "prevalence": y.mean(),
                              "ap_A": average_precision_score(y, g.p_aleatoire),
                              "ap_B": average_precision_score(y, g.p_lomo)})
        if nom in ("rf", "rf_rel"):
            for portee, d in [("tous", p), ("GGG", p[p.code_module == "GGG"])]:
                for proto, col in [("aleatoire", "p_aleatoire"), ("lomo", "p_lomo")]:
                    classe = np.clip(np.digitize(d[col], BORNES[1:-1]), 0, 9)
                    t = d.assign(classe=classe).groupby("classe").agg(
                        p_moyen=(col, "mean"), observe=(TARGET, "mean"), n=(TARGET, "size")).reset_index()
                    fiab.append(t.assign(variables=var, protocole=proto, portee=portee))
    ap = pd.DataFrame(ap_lignes)
    ap.to_csv(RES / "average_precision_c25.csv", index=False)
    pd.concat(fiab).to_csv(RES / "reliability_c25.csv", index=False)
    pd.set_option("display.width", 200)
    print(ap.groupby(["cible", "modele", "variables"])[["prevalence", "ap_A", "ap_B"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
