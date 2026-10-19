#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2 — Courbe « précocité vs performance ».

Pour chaque cutoff et chaque population (active / all), construit les variables
puis évalue (A) aléatoire groupé par étudiant et (B) leave-one-module-out.

Sorties :
  results/early_curve.csv            une ligne par (cutoff, population, protocole)
  results/early_curve_by_module.csv  une ligne par (cutoff, population, protocole, module)
  results/versions.txt               versions des librairies

Usage :
    python src/03_early_curve.py --data data/OULAD
    python src/03_early_curve.py --data data/OULAD --cutoffs 0.25 --populations active
"""
import argparse
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from oulad_features import POPULATIONS, charger, construire
from transfer_eval import SEED, eval_aleatoire, eval_contexte_exclu, scores

warnings.filterwarnings("ignore")
ID_COLS = ["id_student", "code_module", "code_presentation"]
TARGET = "at_risk"
METRIQUES = ["AUC", "Rappel", "Précision", "F1"]


def une_config(df, kind):
    feats = [c for c in df.columns if c not in ID_COLS + [TARGET]]
    a_plis, oof = eval_aleatoire(df, feats, TARGET, "id_student", kind, return_oof=True)
    b = eval_contexte_exclu(df, feats, TARGET, "code_module", kind)
    b[METRIQUES] = b[METRIQUES].astype(float)

    lignes = [
        {"protocole": "aleatoire", "n": len(df), "taux_risque": df[TARGET].mean(),
         **a_plis.mean().to_dict(), "AUC_sd": a_plis["AUC"].std(), "AUC_min": a_plis["AUC"].min()},
        {"protocole": "lomo", "n": len(df), "taux_risque": df[TARGET].mean(),
         **b[METRIQUES].mean().to_dict(), "AUC_sd": b["AUC"].std(), "AUC_min": b["AUC"].min()},
    ]
    par_module = []
    for mod, g in df.groupby("code_module"):
        m = df["code_module"].values == mod
        par_module.append({"protocole": "aleatoire", "module": mod, "n": len(g),
                           "taux_risque": g[TARGET].mean(), **scores(g[TARGET].values, oof[m])})
    for mod, r in b.iterrows():
        par_module.append({"protocole": "lomo", "module": mod, "n": int(r["n"]),
                           "taux_risque": r["taux_risque"], **r[METRIQUES].to_dict()})
    return lignes, par_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--cutoffs", type=float, nargs="+", default=[0.10, 0.20, 0.25, 0.33, 0.50])
    ap.add_argument("--populations", nargs="+", choices=POPULATIONS, default=list(POPULATIONS))
    ap.add_argument("--model", choices=["rf", "logit"], default="rf")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    (args.out / "versions.txt").write_text(
        f"python {platform.python_version()}\nnumpy {np.__version__}\n"
        f"pandas {pd.__version__}\nscikit-learn {sklearn.__version__}\nseed {SEED}\n",
        encoding="utf-8")

    tables = charger(args.data, args.cache)
    agr, mod = [], []
    for pop in args.populations:
        for cut in args.cutoffs:
            t = time.time()
            df = construire(tables, cut, pop)
            lignes, par_module = une_config(df, args.model)
            meta = {"cutoff": cut, "population": pop, "modele": args.model}
            agr += [{**meta, **l} for l in lignes]
            mod += [{**meta, **l} for l in par_module]
            a, b = lignes
            print(f"[{pop:6s} cutoff {cut:.2f}] n={len(df)} risque={df[TARGET].mean():.1%} | "
                  f"AUC aléa {a['AUC']:.3f}  LOMO {b['AUC']:.3f}  écart {a['AUC']-b['AUC']:+.3f} | "
                  f"rappel aléa {a['Rappel']:.3f}  LOMO {b['Rappel']:.3f}  ({time.time()-t:.0f}s)",
                  flush=True)

    pd.DataFrame(agr).to_csv(args.out / "early_curve.csv", index=False)
    pd.DataFrame(mod).to_csv(args.out / "early_curve_by_module.csv", index=False)
    print(f"Écrit : {args.out / 'early_curve.csv'}, {args.out / 'early_curve_by_module.csv'}")


if __name__ == "__main__":
    main()
