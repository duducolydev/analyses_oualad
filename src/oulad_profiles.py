# -*- coding: utf-8 -*-
"""
Descripteurs de CONTEXTE (module ou présentation OULAD) pour expliquer
l'hétérogénéité du transfert (T3).

Calculés au cutoff, sur la population retenue (table de variables) :
  structure d'évaluation : nb d'évaluations dues avant le cutoff, date de la
      première échéance (fraction de session), part de l'examen dans le poids total
  engagement observable  : part des étudiants avec une note précoce, clics médians,
      nb de types d'activité VLE
  public et cible        : taille, taux à risque, part des Withdrawn parmi les
      étudiants à risque, part sans A-level, part des plus de 35 ans
"""
import pandas as pd

from oulad_features import CLE


def profils(tables: dict[str, pd.DataFrame], feats: pd.DataFrame, cutoff: float,
            niveau: str = "code_module") -> pd.DataFrame:
    """Une ligne par contexte ; `niveau` = "code_module" ou "presentation"."""
    co = tables["courses"].copy()
    co["cutoff_day"] = (co["module_presentation_length"] * cutoff).round()
    a = tables["assessments"].merge(co, on=CLE)
    a["due_frac"] = a["date"] / a["module_presentation_length"]
    devoirs = a[a["assessment_type"] != "Exam"]

    pres = pd.DataFrame({
        "n_eval_avant_cutoff": devoirs[devoirs["date"] <= devoirs["cutoff_day"]].groupby(CLE).size(),
        "premiere_echeance": devoirs.groupby(CLE)["due_frac"].min(),
        "part_examen": a[a["assessment_type"] == "Exam"].groupby(CLE)["weight"].sum()
                       / a.groupby(CLE)["weight"].sum(),
    }).fillna({"n_eval_avant_cutoff": 0})
    vle = tables.get("vle")
    if vle is not None:
        pres["n_types_activite"] = vle.groupby(CLE)["activity_type"].nunique()

    f = feats.merge(tables["studentInfo"][CLE + ["id_student", "final_result",
                                                 "highest_education", "age_band"]]
                    .rename(columns={"highest_education": "_educ", "age_band": "_age"}),
                    on=CLE + ["id_student"], how="left")
    f["_note"] = f["n_submissions"] > 0
    f["_wd"] = f["final_result"] == "Withdrawn"
    f["_sans_alevel"] = f["_educ"].isin(["No Formal quals", "Lower Than A Level"])
    f["_plus35"] = f["_age"] != "0-35"

    if niveau == "presentation":
        cle_ctx = CLE
    else:
        cle_ctx = ["code_module"]
        pres = pres.groupby(level="code_module").mean()

    pop = f.groupby(cle_ctx).agg(
        n=("id_student", "size"),
        taux_risque=("at_risk", "mean"),
        part_note_precoce=("_note", "mean"),
        clics_medians=("total_clicks", "median"),
        part_sans_alevel=("_sans_alevel", "mean"),
        part_plus35=("_plus35", "mean"))
    risque = f[f["at_risk"] == 1].groupby(cle_ctx)["_wd"].mean().rename("part_withdrawn_parmi_risque")
    return pres.join(pop, how="inner").join(risque)
