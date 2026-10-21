#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5 — Le temporel améliore-t-il la prédiction précoce, et se transfère-t-il ?

Modèles comparés à la forêt agrégée (référence, 03_early_curve.py) :
  hgb_agg  gradient boosting, mêmes variables agrégées        -> effet de la famille de modèle
  hgb_seq  gradient boosting, agrégées + séquence + tendances -> apport de l'information temporelle
  gru      GRU sur la séquence + MLP sur les variables statiques (agrégées, tendances, démographie)
Chaque modèle est évalué en (A) aléatoire groupé par étudiant et (B) LOMO,
avec variables absolues et relatives, aux 5 cutoffs (population active).

Sorties : results/sequence_by_module.csv (une ligne par cutoff, variables,
modèle, protocole, module ; métriques et part signalée) ;
results/preds/preds_<modèle>[_rel]_active_c<cutoff>.parquet

Usage : python src/09_sequence_models.py --data data/OULAD
        python src/09_sequence_models.py --data data/OULAD --cutoffs 0.25 --modeles hgb_seq
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from oulad_features import charger, construire
from oulad_sequences import sequences
from transfer_eval import eval_aleatoire, eval_contexte_exclu, scores

warnings.filterwarnings("ignore")
ID_COLS = ["id_student", "code_module", "code_presentation"]
TARGET = "at_risk"
MODELES = {"hgb_agg": ("hgb", False), "hgb_seq": ("hgb", True), "gru": ("gru", True)}


def par_module(df, p, protocole):
    lignes = []
    for mod in sorted(df["code_module"].unique()):
        m = df["code_module"].values == mod
        y = df.loc[m, TARGET].values
        lignes.append({"protocole": protocole, "module": mod, "n": int(m.sum()), "taux_risque": y.mean(),
                       **scores(y, p[m]), "signalement": float((p[m] >= 0.5).mean())})
    return lignes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--cutoffs", type=float, nargs="+", default=[0.10, 0.20, 0.25, 0.33, 0.50])
    ap.add_argument("--variables", nargs="+", choices=["absolues", "relatives"], default=["relatives", "absolues"])
    ap.add_argument("--modeles", nargs="+", choices=list(MODELES), default=list(MODELES))
    ap.add_argument("--population", default="active")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    (args.out / "preds").mkdir(parents=True, exist_ok=True)

    tables = charger(args.data, args.cache)
    sortie = args.out / "sequence_by_module.csv"
    lignes = []
    for var in args.variables:
        rel = var == "relatives"
        for cut in args.cutoffs:
            agr = construire(tables, cut, args.population, rel)
            seq = sequences(tables, agr[ID_COLS], cut, rel)
            complet = pd.concat([agr, seq], axis=1)
            for nom in args.modeles:
                kind, avec_seq = MODELES[nom]
                df = complet if avec_seq else agr
                feats = [c for c in df.columns if c not in ID_COLS + [TARGET]]
                t = time.time()
                _, oof = eval_aleatoire(df, feats, TARGET, "id_student", kind, return_oof=True)
                _, pred_b = eval_contexte_exclu(df, feats, TARGET, "code_module", kind, return_pred=True)
                df[ID_COLS + [TARGET]].assign(p_aleatoire=oof, p_lomo=pred_b).to_parquet(
                    args.out / "preds" / f"preds_{nom}{'_rel' if rel else ''}_{args.population}_c{round(cut*100):02d}.parquet",
                    index=False)
                meta = {"cutoff": cut, "variables": var, "modele": nom}
                res = [{**meta, **l} for l in par_module(df, oof, "aleatoire") + par_module(df, pred_b, "lomo")]
                lignes += res
                r = pd.DataFrame(res).groupby("protocole")[["AUC", "signalement"]].mean()
                s = pd.DataFrame(res).pivot(index="module", columns="protocole", values="signalement")
                derive = float((s["lomo"] - s["aleatoire"]).abs().mean())
                print(f"[{var:9s} c={cut:.2f} {nom:7s}] AUC A {r.loc['aleatoire', 'AUC']:.3f}  "
                      f"B {r.loc['lomo', 'AUC']:.3f}  |dérive| {derive:.3f}  ({time.time() - t:.0f}s)", flush=True)
            # écriture incrémentale : un arrêt en cours de route ne perd pas les configurations finies
            pd.DataFrame(lignes).to_csv(sortie, index=False)
    print(f"Écrit : {sortie}")


if __name__ == "__main__":
    main()
