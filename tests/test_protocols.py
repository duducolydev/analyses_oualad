# -*- coding: utf-8 -*-
"""
Protocoles d'évaluation : aucun individu ne doit se trouver à la fois en
entraînement et en test, ni en (A) (plis groupés), ni en (B) strict.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from transfer_eval import plis_contexte  # noqa: E402


def donnees():
    """Trois contextes ; certains individus sont inscrits dans plusieurs contextes."""
    rng = np.random.default_rng(0)
    lignes = []
    for ctx in ["A", "B", "C"]:
        for i in rng.choice(60, size=30, replace=False):
            lignes.append({"contexte": ctx, "individu": int(i), "y": int(rng.random() < 0.4)})
    df = pd.DataFrame(lignes)
    assert df.groupby("individu").contexte.nunique().max() > 1  # le cas à tester existe
    return df


def test_lomo_strict_disjoint():
    df = donnees()
    for ctx, tr, te in plis_contexte(df, "contexte", exclure_communs="individu"):
        assert not (tr & te).any()
        assert set(df.loc[tr, "individu"]).isdisjoint(set(df.loc[te, "individu"]))
        assert (df.loc[te, "contexte"] == ctx).all() and (df.loc[tr, "contexte"] != ctx).all()


def test_lomo_standard_garde_les_communs():
    df = donnees()
    communs = [not set(df.loc[tr, "individu"]).isdisjoint(set(df.loc[te, "individu"]))
               for _, tr, te in plis_contexte(df, "contexte")]
    assert any(communs)  # c'est bien ce que le protocole strict corrige


def test_protocole_A_groupe_par_individu():
    df = donnees()
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
    for tr, te in sgkf.split(df, df.y, df.individu):
        assert set(df.individu.iloc[tr]).isdisjoint(set(df.individu.iloc[te]))
