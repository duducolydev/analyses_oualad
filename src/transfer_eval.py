# -*- coding: utf-8 -*-
"""
Évaluation générique, indépendante d'OULAD : deux protocoles comparés.

  (A) eval_aleatoire : k-fold stratifié, groupé par individu (pas de fuite
      d'un même étudiant entre train et test). Les contextes sont mélangés.
  (B) eval_contexte_exclu : leave-one-group-out sur une colonne de CONTEXTE
      (module OULAD aujourd'hui ; formation ou pays UN-CHK demain).

Les deux fonctions prennent un DataFrame, la liste des variables, la cible et
les colonnes de groupe ; elles renvoient une ligne de métriques par pli/contexte.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 0


def construire_modele(X: pd.DataFrame, kind: str = "rf", seed: int = SEED) -> Pipeline:
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    if kind == "rf":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     n_jobs=-1, random_state=seed)
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("pre", pre), ("clf", clf)])


def scores(y, p, seuil=0.5) -> dict:
    yhat = (p >= seuil).astype(int)
    return {"AUC": roc_auc_score(y, p),
            "Rappel": recall_score(y, yhat),
            "Précision": precision_score(y, yhat, zero_division=0),
            "F1": f1_score(y, yhat)}


def eval_aleatoire(df, features, target, group_col, kind="rf",
                   n_splits=5, seed=SEED, return_oof=False):
    """(A) Une ligne de métriques par pli ; avec return_oof, renvoie aussi les
    probabilités hors-pli (pour ventiler les scores par contexte)."""
    X, y = df[features], df[target].values
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    res, oof = [], np.full(len(df), np.nan)
    for tr, te in sgkf.split(X, y, df[group_col].values):
        m = construire_modele(X, kind, seed).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        res.append(scores(y[te], oof[te]))
    return (pd.DataFrame(res), oof) if return_oof else pd.DataFrame(res)


def eval_contexte_exclu(df, features, target, context_col, kind="rf",
                        seed=SEED) -> pd.DataFrame:
    """(B) Une ligne de métriques par contexte laissé de côté (index = contexte)."""
    res = {}
    for ctx in sorted(df[context_col].unique()):
        tr, te = df[df[context_col] != ctx], df[df[context_col] == ctx]
        m = construire_modele(tr[features], kind, seed).fit(tr[features], tr[target].values)
        res[ctx] = scores(te[target].values, m.predict_proba(te[features])[:, 1])
        res[ctx].update(n=len(te), taux_risque=te[target].mean())
    return pd.DataFrame(res).T
