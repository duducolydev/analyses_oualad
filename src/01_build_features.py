#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Étape 1 — Construction des variables (prédiction PRÉCOCE).

Idée directrice de l'article (créneau visé) : prédire le risque de décrochage
en n'utilisant QUE les données disponibles au début du cours (par défaut les
premiers 25 % de la session), puis évaluer la GÉNÉRALISATION À UN MODULE NON
VU (voir étape 2). La plupart des travaux sur OULAD utilisent un découpage
aléatoire qui mélange les modules : c'est optimiste et peu transférable.

Ce script lit les 7 CSV OULAD et produit une table de modélisation :
une ligne par (étudiant, module, présentation), avec :
  - variables démographiques (studentInfo)
  - variables d'engagement précoce (studentVle tronqué à la fenêtre)
  - variables d'évaluation précoce (studentAssessment avant la fenêtre)
  - la cible : at_risk = 1 si final_result in {Withdrawn, Fail}, sinon 0

Usage :
    python 01_build_features.py --data ./data_synthetique --out features.parquet
    python 01_build_features.py --data ./OULAD --out features.parquet --cutoff 0.25
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def charger(data: Path):
    lire = lambda n: pd.read_csv(data / n)
    return (lire("studentInfo.csv"), lire("studentRegistration.csv"),
            lire("studentVle.csv"), lire("studentAssessment.csv"),
            lire("assessments.csv"), lire("courses.csv"))


def construire(data: Path, cutoff: float) -> pd.DataFrame:
    info, reg, vle, sa, assess, courses = charger(data)

    # Longueur de chaque présentation -> date limite de la fenêtre précoce
    courses = courses.copy()
    courses["cutoff_day"] = (courses["module_presentation_length"] * cutoff).round().astype(int)
    cle = ["code_module", "code_presentation"]

    # --- Engagement VLE dans la fenêtre [début, cutoff_day] ---
    vle = vle.merge(courses[cle + ["cutoff_day"]], on=cle, how="left")
    vle_f = vle[vle["date"] <= vle["cutoff_day"]]
    agg_vle = (vle_f.groupby(cle + ["id_student"])
               .agg(total_clicks=("sum_click", "sum"),
                    active_days=("date", "nunique"),
                    distinct_sites=("id_site", "nunique"),
                    first_activity=("date", "min"),
                    last_activity=("date", "max"))
               .reset_index())
    # clics par semaine moyenne sur la fenêtre
    agg_vle["clicks_per_active_day"] = (agg_vle["total_clicks"]
                                        / agg_vle["active_days"].clip(lower=1))

    # --- Évaluations soumises avant la fenêtre ---
    assess_f = assess.merge(courses[cle + ["cutoff_day"]], on=cle, how="left")
    assess_f = assess_f[assess_f["date"].fillna(1e9) <= assess_f["cutoff_day"]]
    sa_f = sa.merge(assess_f[["id_assessment"] + cle], on="id_assessment", how="inner")
    agg_sa = (sa_f.groupby(cle + ["id_student"])
              .agg(n_submissions=("id_assessment", "nunique"),
                   mean_early_score=("score", "mean"))
              .reset_index())

    # --- Inscription (date relative, retrait) ---
    reg2 = reg.copy()
    reg2["late_registration"] = reg2["date_registration"].fillna(0).clip(upper=0).abs()

    # --- Assemblage ---
    df = info.merge(reg2[cle + ["id_student", "date_registration",
                               "late_registration"]],
                    on=cle + ["id_student"], how="left")
    df = df.merge(agg_vle, on=cle + ["id_student"], how="left")
    df = df.merge(agg_sa, on=cle + ["id_student"], how="left")

    # Valeurs manquantes d'engagement = aucune activité
    for c in ["total_clicks", "active_days", "distinct_sites",
              "clicks_per_active_day", "n_submissions"]:
        df[c] = df[c].fillna(0)
    df["mean_early_score"] = df["mean_early_score"].fillna(-1)  # -1 = non soumis
    df["first_activity"] = df["first_activity"].fillna(999)
    df["last_activity"] = df["last_activity"].fillna(-999)

    # --- Cible ---
    df["at_risk"] = df["final_result"].isin(["Withdrawn", "Fail"]).astype(int)

    # On NE garde PAS final_result ni les scores postérieurs comme prédicteurs
    cols_drop = ["final_result"]
    df = df.drop(columns=[c for c in cols_drop if c in df.columns])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("features.parquet"))
    ap.add_argument("--cutoff", type=float, default=0.25,
                    help="Fraction de la session utilisée (fenêtre précoce).")
    args = ap.parse_args()

    df = construire(args.data, args.cutoff)
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
