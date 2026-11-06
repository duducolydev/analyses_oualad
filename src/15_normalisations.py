#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Révision (étape 3, revue 2) — Faut-il des RANGS, ou toute normalisation par
cohorte suffit-elle ?

Variantes des variables d'engagement et de notes (oulad_features.normaliser) :
  absolues  valeurs brutes (référence de la littérature)
  rang      rang centile dans la présentation (modèle principal, D7)
  zscore    centrage-réduction dans la présentation, après écrêtage 1–99 %
  robuste   (x − médiane) / écart interquartile dans la présentation
Forêt, protocoles (A) et (B) LOMO strict, population active.

Par module : AUC (A) et (B), part signalée (A) et (B), risque observé,
précision et F1 de (B). Dérive « même variante » = (B) − (A) de la variante ;
dérive « commune » = (B) − (A) à variables absolues.

Sortie : results/normalisations.csv
Usage : python src/15_normalisations.py --data data/OULAD
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from oulad_features import charger, construire
from transfer_eval import eval_aleatoire, eval_contexte_exclu, scores

warnings.filterwarnings("ignore")
ID_COLS = ["id_student", "code_module", "code_presentation"]
TARGET = "at_risk"
VARIANTES = {"absolues": (False, "rang"), "rang": (True, "rang"),
             "zscore": (True, "zscore"), "robuste": (True, "robuste")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--cutoffs", type=float, nargs="+", default=[0.10, 0.25, 0.50])
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    tables = charger(args.data, args.cache)
    lignes = []
    for cut in args.cutoffs:
        ref_A = None
        for nom, (rel, norm) in VARIANTES.items():
            t = time.time()
            df = construire(tables, cut, "active", rel, normalisation=norm)
            feats = [c for c in df.columns if c not in ID_COLS + [TARGET]]
            _, oof = eval_aleatoire(df, feats, TARGET, "id_student", "rf", return_oof=True)
            _, pb = eval_contexte_exclu(df, feats, TARGET, "code_module", "rf", return_pred=True,
                                        exclure_communs="id_student")
            if nom == "absolues":
                ref_A = oof >= 0.5
            for mod in sorted(df.code_module.unique()):
                m = df.code_module.values == mod
                y = df.loc[m, TARGET].values
                a, b = scores(y, oof[m]), scores(y, pb[m])
                lignes.append({"cutoff": cut, "variante": nom, "module": mod, "risque": y.mean(),
                               "auc_A": a["AUC"], "auc_B": b["AUC"],
                               "signal_A": float((oof[m] >= 0.5).mean()), "signal_B": float((pb[m] >= 0.5).mean()),
                               "signal_A_absolues": float(ref_A[m].mean()),
                               "precision_B": b["Précision"], "F1_B": b["F1"]})
            d = pd.DataFrame([l for l in lignes if l["cutoff"] == cut and l["variante"] == nom])
            print(f"[c={cut:.2f} {nom:8s}] AUC A {d.auc_A.mean():.3f}  B {d.auc_B.mean():.3f}  "
                  f"|dérive| {(d.signal_B - d.signal_A).abs().mean():.3f}  ({time.time() - t:.0f}s)", flush=True)
            pd.DataFrame(lignes).to_csv(args.out / "normalisations.csv", index=False)

    d = pd.DataFrame(lignes)
    d["cout"] = d.auc_A - d.auc_B
    d["derive_meme"] = (d.signal_B - d.signal_A).abs()
    d["derive_commune"] = (d.signal_B - d.signal_A_absolues).abs()
    d["ecart_risque"] = (d.signal_B - d.risque).abs()
    pd.set_option("display.width", 200)
    print("\nMoyennes sur les modules :")
    print(d.groupby(["cutoff", "variante"])[["auc_A", "auc_B", "cout", "derive_meme", "derive_commune",
                                             "ecart_risque", "precision_B", "F1_B"]].mean().round(3).to_string())
    b = d.pivot_table(index=["cutoff", "module"], columns="variante", values="auc_B")
    print("\nModules où l'AUC (B) dépasse celle des variables absolues :")
    print(pd.DataFrame({v: (b[v] > b["absolues"]).groupby("cutoff").sum() for v in ["rang", "zscore", "robuste"]}))
    print("\nAUC (B) par module, cutoff 25 % :\n", b.xs(0.25, level="cutoff").round(3).to_string())


if __name__ == "__main__":
    main()
