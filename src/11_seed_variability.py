#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Révision (étape 2) — Variabilité due à la graine.

La graine fixe la forêt (tirages bootstrap, variables candidates) et, en (A),
la répartition en plis. On relance (A) et (B) avec plusieurs graines pour
savoir quels écarts par module dépassent ce bruit (ex. : variables relatives
vs absolues pour CCC et EEE).

Sortie : results/seed_variability_c<cutoff>.csv (une ligne par graine,
variables, protocole, module : AUC, rappel, précision, part signalée).
Usage : python src/11_seed_variability.py --data data/OULAD --seeds 0 1 2 3 4 5 6 7 8 9
"""
import argparse
import time
import warnings
from pathlib import Path

import pandas as pd

from oulad_features import charger, construire
from transfer_eval import eval_aleatoire, eval_contexte_exclu, scores

warnings.filterwarnings("ignore")
ID_COLS = ["id_student", "code_module", "code_presentation"]
TARGET = "at_risk"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--cutoff", type=float, default=0.25)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    tables = charger(args.data, args.cache)
    sortie = args.out / f"seed_variability_c{round(args.cutoff * 100):02d}.csv"
    lignes = []
    for var in ["absolues", "relatives"]:
        df = construire(tables, args.cutoff, "active", var == "relatives")
        feats = [c for c in df.columns if c not in ID_COLS + [TARGET]]
        for s in args.seeds:
            t = time.time()
            _, oof = eval_aleatoire(df, feats, TARGET, "id_student", "rf", seed=s, return_oof=True)
            _, pb = eval_contexte_exclu(df, feats, TARGET, "code_module", "rf", seed=s, return_pred=True,
                                         exclure_communs="id_student")  # LOMO strict (D14)
            for proto, p in [("aleatoire", oof), ("lomo", pb)]:
                for mod in sorted(df.code_module.unique()):
                    m = df.code_module.values == mod
                    y = df.loc[m, TARGET].values
                    lignes.append({"variables": var, "graine": s, "protocole": proto, "module": mod,
                                   **scores(y, p[m]), "signalement": float((p[m] >= 0.5).mean())})
            d = pd.DataFrame(lignes)
            d = d[(d.variables == var) & (d.graine == s)].groupby("protocole")["AUC"].mean()
            print(f"[{var:9s} graine {s}] AUC A {d['aleatoire']:.3f}  B {d['lomo']:.3f}  ({time.time() - t:.0f}s)",
                  flush=True)
            pd.DataFrame(lignes).to_csv(sortie, index=False)
    print(f"Écrit : {sortie}")


if __name__ == "__main__":
    main()
