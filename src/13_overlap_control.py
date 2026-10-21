#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Révision (étape 2) — Contrôle du test de chevauchement (10_overlap_check.py).

Retirer de l'entraînement les étudiants présents dans le module testé réduit
aussi la taille d'entraînement. Pour séparer les deux effets, pour chaque
module à fort chevauchement on compare, sur plusieurs graines, l'AUC (B) :
  standard   LOMO habituel
  communs    sans les étudiants présents dans le module testé
  hasard     sans le MÊME nombre d'inscriptions tirées au hasard parmi les autres

Sortie : results/overlap_control_c25.csv
Usage : python src/13_overlap_control.py --data data/OULAD
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from oulad_features import charger, construire
from transfer_eval import construire_modele

warnings.filterwarnings("ignore")
ID_COLS = ["id_student", "code_module", "code_presentation"]
TARGET = "at_risk"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--modules", nargs="+", default=["CCC", "DDD", "EEE"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    tables = charger(args.data, args.cache)
    lignes = []
    for var in ["absolues", "relatives"]:
        df = construire(tables, 0.25, "active", var == "relatives")
        feats = [c for c in df.columns if c not in ID_COLS + [TARGET]]
        for mod in args.modules:
            te = df[df.code_module == mod]
            tr = df[df.code_module != mod]
            communs = tr.id_student.isin(set(te.id_student))
            for s in args.seeds:
                hasard = tr[~communs].sample(n=int(communs.sum()), random_state=s).index
                for cond, d in [("standard", tr), ("communs", tr[~communs]), ("hasard", tr.drop(index=hasard))]:
                    m = construire_modele(d[feats], "rf", s).fit(d[feats], d[TARGET].values)
                    p = m.predict_proba(te[feats])[:, 1]
                    lignes.append({"variables": var, "module": mod, "graine": s, "condition": cond,
                                   "n_retires": len(tr) - len(d), "auc_B": roc_auc_score(te[TARGET], p),
                                   "signalement": float((p >= 0.5).mean())})
                print(f"[{var:9s} {mod} graine {s}] " + "  ".join(
                    f"{l['condition']} {l['auc_B']:.3f}" for l in lignes[-3:]), flush=True)
                pd.DataFrame(lignes).to_csv(args.out / "overlap_control_c25.csv", index=False)

    d = pd.DataFrame(lignes)
    r = d.pivot_table(index=["variables", "module", "graine"], columns="condition", values="auc_B")
    r["effet_communs"] = r.communs - r.standard
    r["effet_hasard"] = r.hasard - r.standard
    print(r.groupby(["variables", "module"])[["effet_communs", "effet_hasard"]].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
