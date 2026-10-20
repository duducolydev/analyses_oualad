# -*- coding: utf-8 -*-
"""
Garde-fou anti-fuite : les variables construites au cutoff ne doivent dépendre
QUE des données disponibles à cette date. Test d'invariance : on supprime des
tables brutes tout ce qui est postérieur au cutoff, et la table de variables
doit rester strictement identique (seule la cible peut venir du futur).
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
from oulad_features import CLE, charger, construire  # noqa: E402

# 0.20 et 0.40 tombent sur les échéances du jeu synthétique : des remises
# tardives franchissent alors le cutoff (cas de fuite à détecter).
CUTOFFS = [0.10, 0.20, 0.25, 0.40, 0.50]


@pytest.fixture(scope="session")
def tables(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetique")
    subprocess.run([sys.executable, str(SRC / "make_synthetic_oulad.py"), str(out)],
                   check=True, capture_output=True)
    return charger(out)


def tronquer(tables, cutoff):
    """Ne garde que ce qu'on sait au jour cutoff_day de chaque présentation."""
    c = tables["courses"].copy()
    c["cutoff_day"] = (c["module_presentation_length"] * cutoff).round().astype(int)
    jour = c[CLE + ["cutoff_day"]]
    t = dict(tables)
    v = t["studentVle"].merge(jour, on=CLE)
    t["studentVle"] = v[v["date"] <= v["cutoff_day"]].drop(columns="cutoff_day")
    a = t["assessments"].merge(jour, on=CLE)
    t["assessments"] = a[a["date"] <= a["cutoff_day"]].drop(columns="cutoff_day")
    s = t["studentAssessment"].merge(
        tables["assessments"][["id_assessment"] + CLE].merge(jour, on=CLE)[["id_assessment", "cutoff_day"]],
        on="id_assessment")
    t["studentAssessment"] = s[s["date_submitted"] <= s["cutoff_day"]].drop(columns="cutoff_day")
    r = t["studentRegistration"].merge(jour, on=CLE)
    r.loc[r["date_unregistration"] > r["cutoff_day"], "date_unregistration"] = float("nan")
    t["studentRegistration"] = r.drop(columns="cutoff_day")
    return t


def test_les_donnees_contiennent_des_remises_tardives(tables):
    """Sans ce cas, le test d'invariance ne pourrait pas détecter la fuite."""
    t = tronquer(tables, 0.20)
    assert len(t["studentAssessment"]) < len(
        tables["studentAssessment"].merge(t["assessments"][["id_assessment"]]))


@pytest.mark.parametrize("relatives", [False, True])
@pytest.mark.parametrize("population", ["active", "all"])
@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_invariance_aux_donnees_futures(tables, cutoff, population, relatives):
    cle = CLE + ["id_student"]
    complet = (construire(tables, cutoff, population, relatives)
               .sort_values(cle).reset_index(drop=True))
    passe = (construire(tronquer(tables, cutoff), cutoff, population, relatives)
             .sort_values(cle).reset_index(drop=True))
    pd.testing.assert_frame_equal(complet.drop(columns="at_risk"),
                                  passe.drop(columns="at_risk"), check_dtype=False)


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_population_active_exclut_les_partis(tables, cutoff):
    df = construire(tables, cutoff, "active")
    reg = tables["studentRegistration"].merge(tables["courses"], on=CLE)
    reg["cutoff_day"] = (reg["module_presentation_length"] * cutoff).round()
    partis = reg[reg["date_unregistration"] <= reg["cutoff_day"]]
    assert partis.merge(df, on=CLE + ["id_student"]).empty
    assert len(partis) > 0  # le test a bien quelque chose à exclure


@pytest.mark.parametrize("cible,positifs", [("withdrawn_fail", {"Withdrawn", "Fail"}),
                                            ("withdrawn", {"Withdrawn"})])
def test_cible(tables, cible, positifs):
    df = construire(tables, 0.25, cible=cible).merge(
        tables["studentInfo"][CLE + ["id_student", "final_result"]], on=CLE + ["id_student"])
    assert (df["at_risk"] == df["final_result"].isin(positifs)).all()
    variables = construire(tables, 0.25).drop(columns="at_risk")
    pd.testing.assert_frame_equal(variables, construire(tables, 0.25, cible=cible).drop(columns="at_risk"))


def test_pas_de_resultat_final_parmi_les_variables(tables):
    df = construire(tables, 0.25)
    assert "final_result" not in df.columns
    assert not any("unregistration" in c for c in df.columns)
