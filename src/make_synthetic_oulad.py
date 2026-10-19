#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génère un petit jeu SYNTHÉTIQUE au schéma exact d'OULAD, pour tester la
chaîne d'analyse sans les vraies données. À REMPLACER par le vrai OULAD
(7 fichiers CSV) téléchargé sur votre machine :
    https://analyse.kmi.open.ac.uk/open_dataset   (licence CC-BY 4.0)

Ne JAMAIS utiliser ces données synthétiques pour un résultat de recherche :
elles ne servent qu'à vérifier que le code tourne.
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(__file__).parent / "data_synthetique"
OUT.mkdir(exist_ok=True)

modules = ["AAA", "BBB", "CCC", "DDD"]
pres = ["2013J", "2014J"]
n_par_module = 300

regions = ["Scotland", "Wales", "London Region", "East Anglian Region",
           "North Western Region", "South East Region"]
educ = ["No Formal quals", "Lower Than A Level", "A Level or Equivalent",
        "HE Qualification", "Post Graduate Qualification"]
imd = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
       "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
ages = ["0-35", "35-55", "55<="]
results = ["Pass", "Withdrawn", "Fail", "Distinction"]

rows_info, rows_reg, rows_vle, rows_sa = [], [], [], []
rows_assess, rows_vle_site, rows_courses = [], [], []

sid = 1000000
id_assessment = 1
id_site = 1
for m in modules:
    for p in pres:
        length = int(rng.integers(230, 270))
        rows_courses.append((m, p, length))
        # évaluations du module
        assess_ids = []
        for k in range(4):
            a_type = ["TMA", "CMA", "Exam"][min(k, 2)]
            date = int((k + 1) / 5 * length)
            weight = 100 if a_type == "Exam" else rng.choice([0, 10, 20])
            rows_assess.append((m, p, id_assessment, a_type, date, weight))
            assess_ids.append((id_assessment, date))
            id_assessment += 1
        # sites VLE
        site_ids = []
        for _ in range(8):
            rows_vle_site.append((id_site, m, p,
                                  rng.choice(["forumng", "resource", "quiz",
                                              "url", "oucontent"]),
                                  None, None))
            site_ids.append(id_site)
            id_site += 1
        for _ in range(n_par_module):
            sid += 1
            # profil
            reg = rng.choice(regions)
            ed = rng.choice(educ)
            band = rng.choice(imd)
            age = rng.choice(ages)
            prev = int(rng.integers(0, 3))
            credits = int(rng.choice([30, 60, 90, 120]))
            dis = rng.choice(["Y", "N"], p=[0.1, 0.9])
            # propension au décrochage (latente)
            risk = (0.5 * (ed in ["No Formal quals", "Lower Than A Level"])
                    + 0.3 * (prev > 0) + 0.2 * (dis == "Y")
                    + rng.normal(0, 0.4))
            engaged = max(0.05, 1 - 0.6 * risk + rng.normal(0, 0.2))
            # résultat
            if risk > 0.8:
                res = rng.choice(["Withdrawn", "Fail"], p=[0.6, 0.4])
            elif risk > 0.3:
                res = rng.choice(results, p=[0.4, 0.2, 0.2, 0.2])
            else:
                res = rng.choice(["Pass", "Distinction"], p=[0.6, 0.4])
            rows_info.append((m, p, sid, rng.choice(["M", "F"]), reg, ed,
                              band, age, prev, credits, dis, res))
            # inscription
            dreg = int(-rng.integers(0, 60))
            dunreg = ""
            if res == "Withdrawn":
                dunreg = int(rng.integers(5, length))
            rows_reg.append((m, p, sid, dreg, dunreg))
            # clics VLE (concentrés en début si engagé)
            n_days = int(rng.poisson(40 * engaged)) + 1
            for _ in range(n_days):
                d = int(min(length, max(-20, rng.normal(length * 0.3, length * 0.25))))
                rows_vle.append((m, p, sid, rng.choice(site_ids), d,
                                 int(rng.integers(1, 20))))
            # soumissions d'évaluation
            for aid, adate in assess_ids:
                if rng.random() < engaged:
                    rows_sa.append((aid, sid, adate + int(rng.integers(-5, 5)),
                                    0, round(float(np.clip(rng.normal(60 + 20*engaged, 15), 0, 100)), 1)))

pd.DataFrame(rows_courses, columns=["code_module", "code_presentation",
             "module_presentation_length"]).to_csv(OUT/"courses.csv", index=False)
pd.DataFrame(rows_info, columns=["code_module", "code_presentation", "id_student",
             "gender", "region", "highest_education", "imd_band", "age_band",
             "num_of_prev_attempts", "studied_credits", "disability",
             "final_result"]).to_csv(OUT/"studentInfo.csv", index=False)
pd.DataFrame(rows_reg, columns=["code_module", "code_presentation", "id_student",
             "date_registration", "date_unregistration"]).to_csv(OUT/"studentRegistration.csv", index=False)
pd.DataFrame(rows_assess, columns=["code_module", "code_presentation", "id_assessment",
             "assessment_type", "date", "weight"]).to_csv(OUT/"assessments.csv", index=False)
pd.DataFrame(rows_sa, columns=["id_assessment", "id_student", "date_submitted",
             "is_banked", "score"]).to_csv(OUT/"studentAssessment.csv", index=False)
pd.DataFrame(rows_vle, columns=["code_module", "code_presentation", "id_student",
             "id_site", "date", "sum_click"]).to_csv(OUT/"studentVle.csv", index=False)
pd.DataFrame(rows_vle_site, columns=["id_site", "code_module", "code_presentation",
             "activity_type", "week_from", "week_to"]).to_csv(OUT/"vle.csv", index=False)

print(f"Jeu synthétique écrit dans {OUT} ({len(rows_info)} étudiants, "
      f"{len(rows_vle)} lignes VLE).")
