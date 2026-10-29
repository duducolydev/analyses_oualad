#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustesse (revue 2, point 1 ; D12, D14) — Effet des étudiants inscrits dans
plusieurs modules.

Depuis D14, le protocole (B) principal est le LOMO STRICT : un étudiant du
module testé n'est jamais en entraînement (comme en (A), groupé par étudiant).
Ce script calcule l'alternative, le LOMO STANDARD (étudiants communs gardés en
entraînement), et la compare au LOMO strict (prédictions de 03_early_curve.py)
sur les mêmes étudiants : AUC, part signalée, dérive.

Sorties : results/overlap_check.csv (une ligne par cutoff, variables, module) ;
          results/preds/preds_rf[_rel]_standard_active_c<cutoff>.parquet
Usage : python src/10_overlap_check.py --data data/OULAD
"""
import argparse
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from oulad_features import charger, construire
from transfer_eval import eval_contexte_exclu

warnings.filterwarnings("ignore")
ID_COLS = ["id_student", "code_module", "code_presentation"]
TARGET = "at_risk"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--cutoffs", type=float, nargs="+", default=[0.10, 0.20, 0.25, 0.33, 0.50])
    ap.add_argument("--variables", nargs="+", choices=["absolues", "relatives"], default=["absolues", "relatives"])
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    tables = charger(args.data, args.cache)
    lignes = []
    for var in args.variables:
        rel = var == "relatives"
        tag = "rf_rel" if rel else "rf"
        for cut in args.cutoffs:
            t = time.time()
            c = f"c{round(cut * 100):02d}"
            df = construire(tables, cut, "active", rel)
            feats = [x for x in df.columns if x not in ID_COLS + [TARGET]]
            _, p_std = eval_contexte_exclu(df, feats, TARGET, "code_module", "rf", return_pred=True)
            ref = pd.read_parquet(args.out / "preds" / f"preds_{tag}_active_{c}.parquet")
            assert (ref[ID_COLS].values == df[ID_COLS].values).all(), "prédictions désalignées"
            ref = ref.assign(p_lomo_standard=p_std)
            ref.to_parquet(args.out / "preds" / f"preds_{tag}_standard_active_{c}.parquet", index=False)
            for mod, g in ref.groupby("code_module"):
                autres = set(df.loc[df.code_module != mod, "id_student"])
                y = g[TARGET].values
                lignes.append({
                    "cutoff": cut, "variables": var, "module": mod, "n": len(g),
                    "part_chevauchement": g["id_student"].isin(autres).mean(),
                    "auc_A": roc_auc_score(y, g.p_aleatoire),
                    "auc_B_strict": roc_auc_score(y, g.p_lomo),
                    "auc_B_standard": roc_auc_score(y, g.p_lomo_standard),
                    "signal_A": (g.p_aleatoire >= 0.5).mean(),
                    "signal_B_strict": (g.p_lomo >= 0.5).mean(),
                    "signal_B_standard": (g.p_lomo_standard >= 0.5).mean()})
            d = pd.DataFrame([l for l in lignes if l["cutoff"] == cut and l["variables"] == var])
            print(f"[{var:9s} c={cut:.2f}] AUC B strict {d.auc_B_strict.mean():.3f}  standard "
                  f"{d.auc_B_standard.mean():.3f} | |dérive| strict "
                  f"{(d.signal_B_strict - d.signal_A).abs().mean():.3f}  standard "
                  f"{(d.signal_B_standard - d.signal_A).abs().mean():.3f}  ({time.time() - t:.0f}s)", flush=True)
            pd.DataFrame(lignes).to_csv(args.out / "overlap_check.csv", index=False)
    print(f"Écrit : {args.out / 'overlap_check.csv'}")


if __name__ == "__main__":
    main()
