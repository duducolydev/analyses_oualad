#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 — Hétérogénéité du transfert : classement vs calibration.

À partir des prédictions de 03_early_curve.py (aucun réentraînement du modèle
principal), pour un cutoff et une population :

1. Par contexte (module, puis présentation) :
     difficulté intrinsèque  = AUC en (A) aléatoire, sur les lignes du contexte
     coût de transfert       = AUC(A) − AUC(B LOMO), mêmes lignes
     dérive de calibration   = taux de signalement (p >= 0,5) en (B) − en (A)
2. Descripteurs de contexte (oulad_profiles.py) + dissimilarité estimée
   (classifieur contexte vs reste), corrélations de Spearman avec les trois axes.
3. Recalibration : le seuil se rétablit-il avec un petit échantillon étiqueté
   du contexte cible ?
     (i)  k étudiants tirés au hasard (k = 25, 50, 100, 200 ; 50 tirages)
     (ii) la première présentation du module (cohorte pilote), évaluée sur
          les présentations suivantes

Sorties dans results/ : transfer_by_module.csv, transfer_by_presentation.csv,
transfer_correlations.csv, recalibration_k.csv, recalibration_temporelle.csv

Usage : python src/04_transfer_analysis.py --data data/OULAD --cutoff 0.25
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from oulad_features import CIBLES, CLE, charger, construire
from oulad_profiles import profils
from transfer_eval import dissimilarite_contexte, recalibrer, scores

warnings.filterwarnings("ignore")
ID_COLS = ["id_student", "code_module", "code_presentation"]
TARGET = "at_risk"
AXES = ["auc_aleatoire", "cout_transfert_auc", "derive_signalement"]
KS = [25, 50, 100, 200]
N_TIRAGES = 50


def issues(g: pd.DataFrame) -> dict:
    """Les trois axes + détail des métriques pour un groupe de lignes."""
    y = g[TARGET].values
    a, b = scores(y, g["p_aleatoire"].values), scores(y, g["p_lomo"].values)
    return {"n": len(g), "taux_risque": y.mean(),
            "auc_aleatoire": a["AUC"], "auc_lomo": b["AUC"],
            "cout_transfert_auc": a["AUC"] - b["AUC"],
            "signal_aleatoire": (g["p_aleatoire"] >= 0.5).mean(),
            "signal_lomo": (g["p_lomo"] >= 0.5).mean(),
            "derive_signalement": (g["p_lomo"] >= 0.5).mean() - (g["p_aleatoire"] >= 0.5).mean(),
            "rappel_aleatoire": a["Rappel"], "rappel_lomo": b["Rappel"],
            "precision_aleatoire": a["Précision"], "precision_lomo": b["Précision"],
            "f1_aleatoire": a["F1"], "f1_lomo": b["F1"]}


def ecart_structure(prof: pd.DataFrame, cols, module_de) -> pd.Series:
    """Distance (variables standardisées) entre chaque contexte et la moyenne
    des contextes des AUTRES modules — hypothèse du brief (T3)."""
    z = (prof[cols] - prof[cols].mean()) / prof[cols].std(ddof=0)
    mods = pd.Series(module_de, index=prof.index)
    return pd.Series([float(np.linalg.norm(z.loc[i] - z[mods != mods[i]].mean()))
                      for i in prof.index], index=prof.index, name="ecart_structure_eval")


def correlations(t: pd.DataFrame, descripteurs, niveau: str) -> pd.DataFrame:
    lignes = []
    for d in descripteurs:
        for ax in AXES:
            rho, p = spearmanr(t[d], t[ax])
            lignes.append({"niveau": niveau, "n_contextes": len(t), "descripteur": d,
                           "axe": ax, "spearman_rho": rho, "p_valeur": p})
    return pd.DataFrame(lignes)


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    """q-valeurs de Benjamini-Hochberg (taux de fausses découvertes)."""
    ordre = p.sort_values()
    q = (ordre * len(p) / np.arange(1, len(p) + 1))[::-1].cummin()[::-1].clip(upper=1)
    return q.reindex(p.index)


def metriques_seuil(y, p) -> dict:
    s = scores(y, p)
    return {"rappel": s["Rappel"], "precision": s["Précision"], "f1": s["F1"],
            "signalement": (p >= 0.5).mean()}


def recalibration_k(preds: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    lignes = []
    for mod, g in preds.groupby("code_module"):
        y, pa, pb = g[TARGET].values, g["p_aleatoire"].values, g["p_lomo"].values
        for k in KS:
            for r in range(N_TIRAGES):
                cal = rng.choice(len(g), size=k, replace=False)
                if y[cal].min() == y[cal].max():
                    continue  # échantillon d'une seule classe : recalibration impossible
                reste = np.setdiff1d(np.arange(len(g)), cal)
                prc = recalibrer(pb[cal], y[cal], pb[reste])
                for meth, p in [("aleatoire", pa[reste]), ("lomo_brut", pb[reste]),
                                ("lomo_recalibre", prc)]:
                    lignes.append({"module": mod, "k": k, "tirage": r, "methode": meth,
                                   **metriques_seuil(y[reste], p)})
    return (pd.DataFrame(lignes).groupby(["module", "k", "methode"])
            .agg(["mean", "std"]).drop(columns="tirage"))


def recalibration_temporelle(preds: pd.DataFrame) -> pd.DataFrame:
    lignes = []
    for mod, g in preds.groupby("code_module"):
        pres = sorted(g["code_presentation"].unique())  # 2013B < 2013J < 2014B < 2014J
        cal, test = g[g["code_presentation"] == pres[0]], g[g["code_presentation"] != pres[0]]
        y = test[TARGET].values
        prc = recalibrer(cal["p_lomo"].values, cal[TARGET].values, test["p_lomo"].values)
        for meth, p in [("aleatoire", test["p_aleatoire"].values),
                        ("lomo_brut", test["p_lomo"].values), ("lomo_recalibre", prc)]:
            lignes.append({"module": mod, "presentation_calibration": pres[0],
                           "n_calibration": len(cal), "n_test": len(test), "methode": meth,
                           **metriques_seuil(y, p)})
    return pd.DataFrame(lignes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--cutoff", type=float, default=0.25)
    ap.add_argument("--population", default="active")
    ap.add_argument("--model", default="rf")
    ap.add_argument("--variables", choices=["absolues", "relatives"], default="absolues")
    ap.add_argument("--cible", choices=list(CIBLES), default="withdrawn_fail")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    relatives = args.variables == "relatives"
    wd = args.cible == "withdrawn"
    tag = args.model + ("_rel" if relatives else "") + ("_wd" if wd else "")

    tables = charger(args.data, args.cache)
    # descripteurs : valeurs absolues
    feats = construire(tables, args.cutoff, args.population, cible=args.cible)
    feats_modele = construire(tables, args.cutoff, args.population, relatives, args.cible)
    fichier = args.out / "preds" / f"preds_{tag}_{args.population}_c{round(args.cutoff*100):02d}.parquet"
    preds = pd.read_parquet(fichier)
    assert len(preds) == len(feats) and (preds[ID_COLS].values == feats[ID_COLS].values).all(), \
        "prédictions désalignées : relancer 03_early_curve.py"

    # 1-2. Module : axes, descripteurs, dissimilarité, corrélations
    variables = [c for c in feats.columns if c not in ID_COLS + [TARGET]]
    t_mod = pd.DataFrame({m: issues(g) for m, g in preds.groupby("code_module")}).T
    prof_mod = profils(tables, feats, args.cutoff, "code_module")
    prof_mod = prof_mod.join(dissimilarite_contexte(feats_modele, variables, "code_module", "id_student"))
    cols_eval = ["n_eval_avant_cutoff", "premiere_echeance", "part_examen"]
    prof_mod = prof_mod.join(ecart_structure(prof_mod, cols_eval, prof_mod.index))
    t_mod = t_mod.join(prof_mod.drop(columns=["n", "taux_risque"])).astype(float)

    # Présentation : mêmes axes (le modèle LOMO reste celui du module exclu)
    t_pres = pd.DataFrame({k: issues(g) for k, g in preds.groupby(CLE)}).T
    t_pres.index.names = CLE
    prof_pres = profils(tables, feats, args.cutoff, "presentation")
    prof_pres = prof_pres.join(ecart_structure(prof_pres, cols_eval,
                                               prof_pres.index.get_level_values("code_module")))
    t_pres = t_pres.join(prof_pres.drop(columns=["n", "taux_risque"])).astype(float)

    descr = [c for c in prof_mod.columns if c != "n"]
    corr = pd.concat([correlations(t_mod, descr, "module"),
                      correlations(t_pres, [d for d in descr if d in t_pres.columns], "presentation")])
    corr["q_valeur_BH"] = corr.groupby("niveau")["p_valeur"].transform(benjamini_hochberg)

    # 3. Recalibration
    rk = recalibration_k(preds)
    rt = recalibration_temporelle(preds)

    meta = (f"_c{round(args.cutoff*100):02d}_{args.population}"
            + ("_relatives" if relatives else "") + ("_withdrawn" if wd else ""))
    t_mod.to_csv(args.out / f"transfer_by_module{meta}.csv", index_label="module")
    t_pres.to_csv(args.out / f"transfer_by_presentation{meta}.csv")
    corr.to_csv(args.out / f"transfer_correlations{meta}.csv", index=False)
    rk.to_csv(args.out / f"recalibration_k{meta}.csv")
    rt.to_csv(args.out / f"recalibration_temporelle{meta}.csv", index=False)

    pd.set_option("display.width", 200)
    print(t_mod[["n", "taux_risque"] + AXES + ["signal_aleatoire", "signal_lomo",
                                              "precision_aleatoire", "precision_lomo"]].round(3))
    print("\nDescripteurs par module :\n", prof_mod.round(3))
    fortes = corr[corr["p_valeur"] < 0.05].sort_values(["niveau", "axe", "p_valeur"])
    print("\nCorrélations de Spearman p < 0,05 (q = Benjamini-Hochberg par niveau) :\n",
          fortes.round(3).to_string(index=False))
    print("\nRecalibration temporelle :\n", rt.round(3).to_string(index=False))
    print("\nRecalibration k (moyenne sur les modules) :\n",
          rk.xs("mean", axis=1, level=1).groupby(["k", "methode"]).mean().round(3))
    print(f"\nÉcrit dans {args.out}/ (suffixe {meta})")


if __name__ == "__main__":
    main()
