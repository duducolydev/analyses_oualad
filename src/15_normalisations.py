#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Révision (étape 3, revue 2 ; étape 5, relectures R1/R2) — Normalisations par
cohorte et familles de modèles.

Variantes des variables d'engagement et de notes (oulad_features.normaliser) :
  absolues  valeurs brutes (référence de la littérature)
  rang      rang centile dans la présentation (modèle principal, D7)
  zscore    centrage-réduction dans la présentation, après écrêtage 1–99 %
  robuste   (x − médiane) / écart interquartile dans la présentation
Modèles (--model) :
  rf      forêt pondérée équilibrée (principal), seuil 0,5
  logit   régression logistique pondérée équilibrée (R1 : l'effet tient-il à la
          représentation des variables plutôt qu'aux arbres ?)
  rf_cal  forêt NON pondérée, probabilités calibrées par isotonie (R2 : le
          choix pondéré + seuil 0,5 est-il en cause ?)
Protocoles (A) et (B) LOMO strict, population active.

Par module : AUC et Average Precision (A) et (B), part signalée au seuil 0,5,
risque prédit moyen et score de Brier (A) et (B), risque observé, précision et
F1 de (B) ; précision et rappel de (B) à capacité fixe (les 30 % de scores les
plus élevés de chaque cohorte, CAPACITE ; valeur d'exemple d'une capacité
d'accompagnement).

Sortie : results/normalisations[_<modèle>].csv (sans suffixe pour rf)
Usage : python src/15_normalisations.py --data data/OULAD [--model logit]
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from oulad_features import charger, construire
from transfer_eval import eval_aleatoire, eval_contexte_exclu, scores

warnings.filterwarnings("ignore")
ID_COLS = ["id_student", "code_module", "code_presentation"]
TARGET = "at_risk"
CAPACITE = 0.30
VARIANTES = {"absolues": (False, "rang"), "rang": (True, "rang"),
             "zscore": (True, "zscore"), "robuste": (True, "robuste")}


def capacite(df, p, m, q=CAPACITE):
    """Précision et rappel quand on signale les q % de scores les plus élevés de chaque cohorte."""
    d = df.loc[m, ["code_presentation", TARGET]].assign(p=p[m])
    rang = d.groupby("code_presentation")["p"].rank(pct=True, ascending=False)
    f, y = (rang <= q).values, d[TARGET].values.astype(bool)
    return (y & f).sum() / max(f.sum(), 1), (y & f).sum() / max(y.sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--cutoffs", type=float, nargs="+", default=[0.10, 0.25, 0.50])
    ap.add_argument("--model", choices=["rf", "logit", "rf_cal"], default="rf")
    ap.add_argument("--variantes", nargs="+", choices=list(VARIANTES), default=list(VARIANTES))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    sortie = args.out / ("normalisations.csv" if args.model == "rf" else f"normalisations_{args.model}.csv")

    tables = charger(args.data, args.cache)
    lignes = []
    for cut in args.cutoffs:
        ref_A = None
        for nom in args.variantes:
            rel, norm = VARIANTES[nom]
            t = time.time()
            df = construire(tables, cut, "active", rel, normalisation=norm)
            feats = [c for c in df.columns if c not in ID_COLS + [TARGET]]
            _, oof = eval_aleatoire(df, feats, TARGET, "id_student", args.model, return_oof=True)
            _, pb = eval_contexte_exclu(df, feats, TARGET, "code_module", args.model, return_pred=True,
                                        exclure_communs="id_student")
            if nom == "absolues":
                ref_A = oof >= 0.5
            for mod in sorted(df.code_module.unique()):
                m = df.code_module.values == mod
                y = df.loc[m, TARGET].values
                a, b = scores(y, oof[m]), scores(y, pb[m])
                pc, rc = capacite(df, pb, m)
                lignes.append({"modele": args.model, "cutoff": cut, "variante": nom, "module": mod, "risque": y.mean(),
                               "auc_A": a["AUC"], "auc_B": b["AUC"],
                               "ap_A": average_precision_score(y, oof[m]), "ap_B": average_precision_score(y, pb[m]),
                               "signal_A": float((oof[m] >= 0.5).mean()), "signal_B": float((pb[m] >= 0.5).mean()),
                               "signal_A_absolues": float(ref_A[m].mean()) if ref_A is not None else np.nan,
                               "p_moyen_A": float(oof[m].mean()), "p_moyen_B": float(pb[m].mean()),
                               "brier_A": float(((oof[m] - y) ** 2).mean()), "brier_B": float(((pb[m] - y) ** 2).mean()),
                               "precision_B": b["Précision"], "F1_B": b["F1"],
                               "precision_cap_B": pc, "rappel_cap_B": rc})
            d = pd.DataFrame([l for l in lignes if l["cutoff"] == cut and l["variante"] == nom])
            print(f"[{args.model} c={cut:.2f} {nom:8s}] AUC A {d.auc_A.mean():.3f}  B {d.auc_B.mean():.3f}  "
                  f"|dérive| {(d.signal_B - d.signal_A).abs().mean():.3f}  "
                  f"|p moyen B − risque| {(d.p_moyen_B - d.risque).abs().mean():.3f}  ({time.time() - t:.0f}s)",
                  flush=True)
            pd.DataFrame(lignes).to_csv(sortie, index=False)

    d = pd.DataFrame(lignes)
    d["cout"] = d.auc_A - d.auc_B
    d["derive_meme"] = (d.signal_B - d.signal_A).abs()
    d["ecart_risque"] = (d.signal_B - d.risque).abs()
    d["calib_A"] = (d.p_moyen_A - d.risque).abs()
    d["calib_B"] = (d.p_moyen_B - d.risque).abs()
    pd.set_option("display.width", 220)
    print("\nMoyennes sur les modules :")
    print(d.groupby(["cutoff", "variante"])[["auc_A", "auc_B", "cout", "ap_B", "derive_meme", "ecart_risque",
                                             "calib_A", "calib_B", "brier_B", "precision_B", "F1_B",
                                             "precision_cap_B"]].mean().round(3).to_string())
    print(f"Écrit : {sortie}")


if __name__ == "__main__":
    main()
