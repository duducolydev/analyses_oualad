# -*- coding: utf-8 -*-
"""
Construction des variables OULAD pour la prédiction PRÉCOCE.

Une ligne par (étudiant, module, présentation), calculée uniquement à partir
des données disponibles au jour `cutoff_day` = round(cutoff × durée) :
  - variables démographiques (studentInfo)
  - engagement VLE jusqu'au cutoff (studentVle, date <= cutoff_day)
  - évaluations dont l'échéance ET la remise sont <= cutoff_day
  - cible (paramètre `cible`) :
      "withdrawn_fail" (défaut) : at_risk = 1 si final_result in {Withdrawn, Fail}
      "withdrawn"               : at_risk = 1 si final_result == Withdrawn
                                  (abandon postérieur au cutoff en population active)

Variables relatives (paramètre `relatives`) : les variables d'engagement et de
notes sont remplacées par leur rang centile au sein de la présentation (parmi
les étudiants de la population retenue, au même cutoff). Le volume d'activité
« normal » dépend de la conception du module ; le rang, lui, est comparable
d'un contexte à l'autre. Calculable au cutoff, sans aucune étiquette.

Population (paramètre `population`) :
  - "active" (défaut) : étudiants inscrits au cutoff et pas encore désinscrits.
    Ceux qui sont déjà partis ne relèvent plus de la prévention, et leur
    proportion varie fortement d'un module à l'autre (biais de composition).
  - "all" : toute la table studentInfo (protocole des scripts initiaux, gardé
    comme point de comparaison).
"""
from pathlib import Path
import pandas as pd

CLE = ["code_module", "code_presentation"]
TABLES = ["studentInfo", "studentRegistration", "studentVle",
          "studentAssessment", "assessments", "courses", "vle"]
POPULATIONS = ("active", "all")
CIBLES = {"withdrawn_fail": ["Withdrawn", "Fail"], "withdrawn": ["Withdrawn"]}
VARIABLES_CONTEXTUELLES = ["total_clicks", "active_days", "distinct_sites",
                           "first_activity", "last_activity", "clicks_per_active_day",
                           "n_submissions", "mean_early_score"]


def charger(data: Path, cache: Path | None = None) -> dict[str, pd.DataFrame]:
    """Lit les CSV OULAD ; si `cache` est fourni, les met en parquet au 1er appel."""
    data = Path(data)
    tables = {}
    for nom in TABLES:
        csv = data / f"{nom}.csv"
        if cache is None:
            tables[nom] = pd.read_csv(csv)
            continue
        pq = Path(cache) / f"{nom}.parquet"
        if not pq.exists():
            pq.parent.mkdir(parents=True, exist_ok=True)
            pd.read_csv(csv).to_parquet(pq, index=False)
        tables[nom] = pd.read_parquet(pq)
    return tables


def construire(tables: dict[str, pd.DataFrame], cutoff: float,
               population: str = "active", relatives: bool = False,
               cible: str = "withdrawn_fail") -> pd.DataFrame:
    if population not in POPULATIONS:
        raise ValueError(f"population doit valoir {POPULATIONS}")
    if cible not in CIBLES:
        raise ValueError(f"cible doit valoir {tuple(CIBLES)}")
    info, reg, vle = tables["studentInfo"], tables["studentRegistration"], tables["studentVle"]
    sa, assess, courses = tables["studentAssessment"], tables["assessments"], tables["courses"]

    # Longueur de chaque présentation -> date limite de la fenêtre précoce
    courses = courses[CLE + ["module_presentation_length"]].copy()
    courses["cutoff_day"] = (courses["module_presentation_length"] * cutoff).round().astype(int)

    # --- Engagement VLE dans la fenêtre [début, cutoff_day] ---
    vle = vle.merge(courses[CLE + ["cutoff_day"]], on=CLE, how="left")
    vle_f = vle[vle["date"] <= vle["cutoff_day"]]
    agg_vle = (vle_f.groupby(CLE + ["id_student"])
               .agg(total_clicks=("sum_click", "sum"),
                    active_days=("date", "nunique"),
                    distinct_sites=("id_site", "nunique"),
                    first_activity=("date", "min"),
                    last_activity=("date", "max"))
               .reset_index())
    agg_vle["clicks_per_active_day"] = (agg_vle["total_clicks"]
                                        / agg_vle["active_days"].clip(lower=1))

    # --- Évaluations : échéance avant le cutoff ET remise avant le cutoff ---
    assess_f = assess.merge(courses[CLE + ["cutoff_day"]], on=CLE, how="left")
    assess_f = assess_f[assess_f["date"].fillna(1e9) <= assess_f["cutoff_day"]]
    sa_f = sa.merge(assess_f[["id_assessment", "cutoff_day"] + CLE],
                    on="id_assessment", how="inner")
    sa_f = sa_f[sa_f["date_submitted"] <= sa_f["cutoff_day"]]
    agg_sa = (sa_f.groupby(CLE + ["id_student"])
              .agg(n_submissions=("id_assessment", "nunique"),
                   mean_early_score=("score", "mean"))
              .reset_index())

    # --- Inscription (date relative) ---
    reg2 = reg.merge(courses[CLE + ["cutoff_day"]], on=CLE, how="left")
    reg2["late_registration"] = reg2["date_registration"].fillna(0).clip(upper=0).abs()
    reg2["present_at_cutoff"] = (
        (reg2["date_registration"].fillna(-1e9) <= reg2["cutoff_day"])
        & ~(reg2["date_unregistration"] <= reg2["cutoff_day"]))

    # --- Assemblage ---
    df = info.merge(reg2[CLE + ["id_student", "date_registration",
                               "late_registration", "present_at_cutoff"]],
                    on=CLE + ["id_student"], how="left")
    if population == "active":
        df = df[df["present_at_cutoff"].fillna(True).astype(bool)]
    df = df.drop(columns="present_at_cutoff")
    df = df.merge(agg_vle, on=CLE + ["id_student"], how="left")
    df = df.merge(agg_sa, on=CLE + ["id_student"], how="left")

    # Valeurs manquantes d'engagement = aucune activité
    for c in ["total_clicks", "active_days", "distinct_sites",
              "clicks_per_active_day", "n_submissions"]:
        df[c] = df[c].fillna(0)
    df["mean_early_score"] = df["mean_early_score"].fillna(-1)  # -1 = non soumis
    df["first_activity"] = df["first_activity"].fillna(999)
    df["last_activity"] = df["last_activity"].fillna(-999)

    if relatives:
        df[VARIABLES_CONTEXTUELLES] = (df.groupby(CLE)[VARIABLES_CONTEXTUELLES]
                                       .rank(pct=True, method="average"))

    # --- Cible --- (final_result n'est jamais conservé comme prédicteur)
    df["at_risk"] = df["final_result"].isin(CIBLES[cible]).astype(int)
    return df.drop(columns="final_result").reset_index(drop=True)
