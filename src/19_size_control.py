#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relecture R1 (3.3) — Le coût de transfert vient-il de l'absence du module cible
ou d'une différence de taille d'entraînement entre (A) et (B) ?

Pour chaque module m, ses étudiants sont répartis en 5 plis groupés par
étudiant. Pour chaque pli de test (étudiants de m) :
  B         modèle LOMO strict : six autres modules, sans aucun étudiant de m
  A_taille  modèle entraîné sur TOUS les modules (m compris), hors étudiants du
            pli de test, sous-échantillonné à exactement la taille d'entraînement
            de B pour ce module
Les deux modèles sont évalués sur les mêmes étudiants de test. L'écart
AUC(A_taille) − AUC(B) est le coût de transfert à taille égale.

Sortie : results/size_control_c25.csv
Usage : python src/19_size_control.py --data data/OULAD
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from oulad_features import charger, construire
from transfer_eval import SEED, construire_modele

warnings.filterwarnings("ignore")
ID_COLS = ["id_student", "code_module", "code_presentation"]
TARGET = "at_risk"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--cutoff", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    tables = charger(args.data, args.cache)
    lignes = []
    for var in ["absolues", "relatives"]:
        df = construire(tables, args.cutoff, "active", var == "relatives")
        feats = [c for c in df.columns if c not in ID_COLS + [TARGET]]
        rng = np.random.default_rng(SEED)
        for mod in sorted(df.code_module.unique()):
            dans_m = (df.code_module == mod).values
            etu_m = set(df.loc[dans_m, "id_student"])
            tr_B = ~dans_m & ~df.id_student.isin(etu_m).values
            n_B = int(tr_B.sum())
            mB = construire_modele(df.loc[tr_B, feats], "rf", SEED).fit(df.loc[tr_B, feats], df.loc[tr_B, TARGET].values)
            idx_m = np.where(dans_m)[0]
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
            for pli, (_, te_rel) in enumerate(sgkf.split(idx_m, df[TARGET].values[idx_m],
                                                         df.id_student.values[idx_m])):
                te = idx_m[te_rel]
                etu_te = set(df.id_student.values[te])
                candidats = np.where(~df.id_student.isin(etu_te).values)[0]
                tr_A = rng.choice(candidats, size=min(n_B, len(candidats)), replace=False)
                mA = construire_modele(df[feats].iloc[tr_A], "rf", SEED).fit(df[feats].iloc[tr_A],
                                                                              df[TARGET].values[tr_A])
                y = df[TARGET].values[te]
                X = df[feats].iloc[te]
                lignes.append({"variables": var, "module": mod, "pli": pli, "n_entrainement": n_B,
                               "part_module_dans_A": float(dans_m[tr_A].mean()),
                               "auc_A_taille": roc_auc_score(y, mA.predict_proba(X)[:, 1]),
                               "auc_B": roc_auc_score(y, mB.predict_proba(X)[:, 1])})
            d = pd.DataFrame([l for l in lignes if l["variables"] == var and l["module"] == mod])
            print(f"[{var:9s} {mod}] n = {n_B}  AUC A_taille {d.auc_A_taille.mean():.3f}  "
                  f"B {d.auc_B.mean():.3f}  coût {d.auc_A_taille.mean() - d.auc_B.mean():+.3f}", flush=True)
            pd.DataFrame(lignes).to_csv(args.out / "size_control_c25.csv", index=False)

    d = pd.DataFrame(lignes)
    m = d.groupby(["variables", "module"])[["auc_A_taille", "auc_B"]].mean()
    m["cout"] = m.auc_A_taille - m.auc_B
    print("\nCoût de transfert à taille d'entraînement égale (moyenne des modules) :")
    print(m.groupby(level="variables").mean().round(3).to_string())


if __name__ == "__main__":
    main()
