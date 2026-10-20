#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4 — Interprétabilité comparée : les variables importantes changent-elles
d'un module à l'autre ?

Pour chaque module m, importance SHAP globale (moyenne des |SHAP|, agrégée par
variable d'origine, one-hot recombinés) de deux modèles, sur des étudiants de m :
  - transfere : modèle LOMO (entraîné sans m) — ce sur quoi s'appuie le modèle
                qui arrive dans un contexte non vu ;
  - local     : modèle entraîné sur m seul, en validation croisée à 2 plis
                groupés par étudiant (on explique le pli non vu) — ce qui
                prédit réellement dans m.

Coût : TreeSHAP exact sur des arbres non élagués est lent (~1,7 s/ligne pour
300 arbres). La valeur SHAP d'une forêt étant la moyenne de celles de ses
arbres, on l'estime sans biais sur un sous-échantillon aléatoire d'arbres
(--n-trees, 100 par défaut : ρ = 0,985 avec l'exact sur les importances).

Sorties : results/shap_by_module_c<cutoff>_<population>_<variables>.csv
          results/shap_rows_c<cutoff>_<population>_<variables>.parquet (valeurs par ligne)
Usage : python src/05_shap_by_module.py --data data/OULAD --variables relatives
"""
import argparse
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

from oulad_features import charger, construire
from transfer_eval import SEED, construire_modele

warnings.filterwarnings("ignore")
ID_COLS = ["id_student", "code_module", "code_presentation"]
TARGET = "at_risk"


def shap_par_variable(modele, X: pd.DataFrame, n_arbres: int, seed: int) -> pd.DataFrame:
    """Valeurs SHAP (classe à risque) par ligne, agrégées par variable d'origine."""
    pre, rf = modele.named_steps["pre"], modele.named_steps["clf"]
    Xt = pre.transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    sous = copy.copy(rf)
    idx = np.random.default_rng(seed).choice(len(rf.estimators_), n_arbres, replace=False)
    sous.estimators_ = [rf.estimators_[i] for i in idx]
    sous.n_estimators = n_arbres
    sv = shap.TreeExplainer(sous).shap_values(Xt, check_additivity=False)[:, :, 1]
    noms = pre.get_feature_names_out()
    cat = pre.transformers_[1][2]
    origine = []
    for n in noms:
        bloc, nom = n.split("__", 1)
        origine.append(nom if bloc == "num" else next(c for c in cat if nom.startswith(c + "_")))
    return pd.DataFrame(sv, columns=origine, index=X.index).T.groupby(level=0).sum().T


def un_module(df, feats, mod, n_lignes, n_arbres, seed):
    te = df[df["code_module"] == mod]
    rng = np.random.default_rng(seed)
    res = []

    # Modèle transféré (LOMO)
    tr = df[df["code_module"] != mod]
    m = construire_modele(tr[feats], seed=seed).fit(tr[feats], tr[TARGET].values)
    ech = te.iloc[rng.choice(len(te), min(n_lignes, len(te)), replace=False)]
    res.append(shap_par_variable(m, ech[feats], n_arbres, seed).assign(modele="transfere"))

    # Modèle local, validation croisée à 2 plis groupés par étudiant
    sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    for itr, ite in sgkf.split(te[feats], te[TARGET].values, te["id_student"].values):
        a, b = te.iloc[itr], te.iloc[ite]
        m = construire_modele(a[feats], seed=seed).fit(a[feats], a[TARGET].values)
        ech = b.iloc[rng.choice(len(b), min(n_lignes // 2, len(b)), replace=False)]
        res.append(shap_par_variable(m, ech[feats], n_arbres, seed).assign(modele="local"))

    out = pd.concat(res)
    out.insert(0, "code_module", mod)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--cutoff", type=float, default=0.25)
    ap.add_argument("--population", default="active")
    ap.add_argument("--variables", choices=["absolues", "relatives"], default="relatives")
    ap.add_argument("--n-lignes", type=int, default=400)
    ap.add_argument("--n-trees", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=3,
                    help="Modules traités en parallèle (~2 Go de RAM chacun).")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    df = construire(charger(args.data, args.cache), args.cutoff, args.population,
                    args.variables == "relatives")
    feats = [c for c in df.columns if c not in ID_COLS + [TARGET]]
    lignes = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(un_module)(df, feats, m, args.n_lignes, args.n_trees, SEED)
        for m in sorted(df["code_module"].unique()))
    lignes = pd.concat(lignes)

    meta = f"_c{round(args.cutoff*100):02d}_{args.population}_{args.variables}"
    lignes.reset_index(names="ligne").to_parquet(args.out / f"shap_rows{meta}.parquet", index=False)

    imp = (lignes.groupby(["code_module", "modele"])[feats]
           .agg(lambda s: s.abs().mean()).stack().rename("mean_abs_shap").reset_index()
           .rename(columns={"level_2": "variable"}))
    imp["part"] = imp["mean_abs_shap"] / imp.groupby(["code_module", "modele"])["mean_abs_shap"].transform("sum")
    imp["rang"] = imp.groupby(["code_module", "modele"])["mean_abs_shap"].rank(ascending=False)
    imp.to_csv(args.out / f"shap_by_module{meta}.csv", index=False)
    print(f"Écrit : {args.out}/shap_by_module{meta}.csv, shap_rows{meta}.parquet")


if __name__ == "__main__":
    main()
