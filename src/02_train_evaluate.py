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
prioritaire) et F1, plus l'importance des variables.

Modèle : Random Forest et Régression logistique (références solides sur OULAD).
SHAP est utilisé s'il est installé ; sinon on se rabat sur l'importance par
permutation.

Usage :
    python 02_train_evaluate.py --features features.parquet
"""
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
ID_COLS = ["id_student", "code_module", "code_presentation"]
TARGET = "at_risk"


def charger(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def construire_modele(X: pd.DataFrame, kind: str) -> Pipeline:
    cat = X.select_dtypes(include=["object"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    if kind == "rf":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     n_jobs=-1, random_state=0)
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("pre", pre), ("clf", clf)])


def scores(y, p, seuil=0.5):
    yhat = (p >= seuil).astype(int)
    return {"AUC": roc_auc_score(y, p),
            "Rappel": recall_score(y, yhat),
            "Précision": precision_score(y, yhat, zero_division=0),
            "F1": f1_score(y, yhat)}


def eval_aleatoire(df, kind):
    """(A) 5-fold stratifié, groupé par étudiant (pas de fuite d'un étudiant)."""
    X = df.drop(columns=ID_COLS + [TARGET])
    y = df[TARGET].values
    groups = df["id_student"].values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    res = []
    for tr, te in sgkf.split(X, y, groups):
        m = construire_modele(X, kind).fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        res.append(scores(y[te], p))
    return pd.DataFrame(res).mean()


def eval_lomo(df, kind):
    """(B) Leave-one-module-out."""
    res = {}
    for mod in sorted(df["code_module"].unique()):
        tr = df[df["code_module"] != mod]
        te = df[df["code_module"] == mod]
        Xtr = tr.drop(columns=ID_COLS + [TARGET])
        Xte = te.drop(columns=ID_COLS + [TARGET])
        m = construire_modele(Xtr, kind).fit(Xtr, tr[TARGET].values)
        p = m.predict_proba(Xte)[:, 1]
        res[mod] = scores(te[TARGET].values, p)
    return pd.DataFrame(res).T


def importance(df, kind):
    X = df.drop(columns=ID_COLS + [TARGET])
    y = df[TARGET].values
    m = construire_modele(X, kind).fit(X, y)
    r = permutation_importance(m, X, y, n_repeats=5, random_state=0,
                               scoring="roc_auc", n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
    return imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--model", choices=["rf", "logit"], default="rf")
    args = ap.parse_args()

    df = charger(args.features)
    print(f"Données : {len(df)} lignes, cible à {df[TARGET].mean():.1%}.\n")

    print("=== (A) Découpage aléatoire groupé par étudiant ===")
    a = eval_aleatoire(df, args.model)
    print(a.round(3).to_string(), "\n")

    print("=== (B) Leave-one-module-out (généralisation à un module non vu) ===")
    b = eval_lomo(df, args.model)
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
