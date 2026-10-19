#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Étape 1 — Construction des variables (prédiction PRÉCOCE).

Idée directrice de l'article (créneau visé) : prédire le risque de décrochage
en n'utilisant QUE les données disponibles au début du cours (par défaut les
premiers 25 % de la session), puis évaluer la GÉNÉRALISATION À UN MODULE NON
VU (voir étape 2). Le calcul des variables est dans `oulad_features.py`.

Usage :
    python src/01_build_features.py --data data/synthetique --out results/features.parquet
    python src/01_build_features.py --data data/OULAD --out results/features.parquet --cutoff 0.25
    python src/01_build_features.py --data data/OULAD --population all   # sans filtre
"""
import argparse
from pathlib import Path

from oulad_features import POPULATIONS, charger, construire


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("features.parquet"))
    ap.add_argument("--cutoff", type=float, default=0.25,
                    help="Fraction de la session utilisée (fenêtre précoce).")
    ap.add_argument("--population", choices=POPULATIONS, default="active",
                    help="active = exclut les étudiants déjà partis au cutoff.")
    ap.add_argument("--cache", type=Path, default=None,
                    help="Dossier de cache parquet des CSV (accélère les relectures).")
    args = ap.parse_args()

    df = construire(charger(args.data, args.cache), args.cutoff, args.population)
    try:
        df.to_parquet(args.out, index=False)
    except Exception:
        args.out = args.out.with_suffix(".csv")
        df.to_csv(args.out, index=False)
    taux = df["at_risk"].mean()
    print(f"Table : {len(df)} lignes, {df.shape[1]-1} variables + cible.")
    print(f"Taux d'étudiants à risque : {taux:.1%}")
    print(f"Modules : {sorted(df['code_module'].unique())}")
    print(f"Écrit dans {args.out}")


if __name__ == "__main__":
    main()
