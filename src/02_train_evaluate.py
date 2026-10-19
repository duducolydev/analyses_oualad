#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Étape 2 — Entraînement et évaluation, avec DEUX protocoles comparés :

  (A) Découpage ALÉATOIRE au niveau étudiant (le protocole habituel,
      optimiste : les mêmes modules sont dans train et test).
  (B) LEAVE-ONE-MODULE-OUT : on entraîne sur tous les modules sauf un et
      on teste sur le module laissé de côté. C'est un proxy honnête de la
      généralisation à un NOUVEAU contexte (ex. une formation UN-CHK non vue).

L'écart de performance entre (A) et (B) est le cœur de l'article : il mesure
ce qui ne se transfère pas. On rapporte AUC, rappel (sensibilité au décrochage,
prioritaire) et F1, plus l'importance des variables. Les protocoles eux-mêmes
sont dans `transfer_eval.py` (génériques, réutilisables sur UN-CHK).

Usage :
    python src/02_train_evaluate.py --features results/features.parquet
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance

from transfer_eval import SEED, construire_modele, eval_aleatoire, eval_contexte_exclu

warnings.filterwarnings("ignore")
ID_COLS = ["id_student", "code_module", "code_presentation"]
TARGET = "at_risk"


def charger(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def variables(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS + [TARGET]]


def importance(df, kind):
    X = df[variables(df)]
    y = df[TARGET].values
    m = construire_modele(X, kind).fit(X, y)
    r = permutation_importance(m, X, y, n_repeats=5, random_state=SEED,
                               scoring="roc_auc", n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
    return imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--model", choices=["rf", "logit"], default="rf")
    args = ap.parse_args()

    df = charger(args.features)
    feats = variables(df)
    print(f"Données : {len(df)} lignes, cible à {df[TARGET].mean():.1%}.\n")

    print("=== (A) Découpage aléatoire groupé par étudiant ===")
    a = eval_aleatoire(df, feats, TARGET, "id_student", args.model).mean()
    print(a.round(3).to_string(), "\n")

    print("=== (B) Leave-one-module-out (généralisation à un module non vu) ===")
    b = eval_contexte_exclu(df, feats, TARGET, "code_module", args.model)
    b = b[["AUC", "Rappel", "Précision", "F1"]].astype(float)
    print(b.round(3).to_string())
    print("\nMoyenne LOMO :")
    print(b.mean().round(3).to_string())

    ecart = a["AUC"] - b["AUC"].mean()
    print(f"\n>> Écart d'AUC (aléatoire - LOMO) : {ecart:+.3f}")
    print("   C'est la perte de performance quand le module (contexte) change.")
    print("   Un écart important justifie l'apport de l'article sur la transférabilité.\n")

    print("=== Importance des variables (permutation, AUC) ===")
    print(importance(df, args.model).round(4).head(12).to_string())


if __name__ == "__main__":
    main()
