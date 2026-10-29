#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Révision (étape 2 bis, revue 3) — Remèdes comparés à armes égales.

Pour chaque module, toutes les règles de signalement sont évaluées sur les
MÊMES étudiants et contre la MÊME référence :
  reference_A  modèle (A) à variables absolues, module vu, seuil 0,5
               (le modèle « local » habituel) ; dérive = part signalée − la sienne
  |signalés − risque|  écart à la part d'étudiants réellement à risque
               (mesure sans référence de modèle)
Règles (prédictions LOMO strict de 03_early_curve.py) :
  brut       variables absolues, seuil 0,5
  platt      Platt (pondéré équilibré) sur des étudiants étiquetés du module cible
  top_q      dans chaque cohorte (présentation), les q % de scores les plus élevés,
             q = taux à risque des six modules d'entraînement (sans étiquette cible)
  relatives  variables relatives, seuil 0,5 (sans étiquette cible)
  top_q_oracle  top-q avec q = vrai taux à risque du module (référence, inaccessible
             en pratique)
Plans d'évaluation :
  k       Platt calé sur k étudiants tirés au hasard (k = 25, 50, 100, 200 ;
          50 tirages) ; toutes les règles évaluées sur les étudiants restants
  pilote  Platt calé sur la première présentation ; toutes les règles évaluées
          sur les présentations suivantes

Sorties : results/remedes_k_c<cutoff>.csv (moyenne et écart-type sur les tirages),
          results/remedes_pilote_c<cutoff>.csv
Usage : python src/14_remedies.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from transfer_eval import recalibrer

RES = Path("results")
TARGET = "at_risk"
KS = [25, 50, 100, 200]
N_TIRAGES = 50


def mesures(y, f, f_ref):
    vp = (y & f).sum()
    prec = vp / max(f.sum(), 1)
    rap = vp / max(y.sum(), 1)
    return {"signalement": f.mean(), "derive_ref": f.mean() - f_ref.mean(),
            "ecart_risque": abs(f.mean() - y.mean()), "precision": prec, "rappel": rap,
            "F1": 2 * prec * rap / max(prec + rap, 1e-9)}


def regles_sans_etiquette(p_abs, p_rel, m):
    """Signalements des règles qui n'utilisent aucune étiquette du module m."""
    d = p_abs[p_abs.code_module == m]
    train = (p_abs.code_module != m) & ~p_abs.id_student.isin(set(d.id_student))  # LOMO strict
    q = p_abs.loc[train, TARGET].mean()
    rang = d.groupby("code_presentation")["p_lomo"].rank(pct=True, ascending=False)
    return d, q, {
        "reference_A": (d.p_aleatoire >= 0.5).values,
        "brut": (d.p_lomo >= 0.5).values,
        "top_q": (rang <= q).values,
        # référence seulement : suppose connu le vrai taux à risque du module cible
        "top_q_oracle": (rang <= d[TARGET].mean()).values,
        "relatives": (p_rel.loc[d.index, "p_lomo"] >= 0.5).values,
    }


def plan_k(p_abs, p_rel):
    rng = np.random.default_rng(0)
    lignes = []
    for m in sorted(p_abs.code_module.unique()):
        d, q, f = regles_sans_etiquette(p_abs, p_rel, m)
        y, pb = d[TARGET].values.astype(bool), d.p_lomo.values
        for k in KS:
            for r in range(N_TIRAGES):
                cal = rng.choice(len(d), size=k, replace=False)
                if y[cal].min() == y[cal].max():
                    continue  # une seule classe : recalibration impossible
                reste = np.setdiff1d(np.arange(len(d)), cal)
                fr = {nom: v[reste] for nom, v in f.items()}
                fr["platt"] = recalibrer(pb[cal], y[cal], pb[reste]) >= 0.5
                for nom, v in fr.items():
                    lignes.append({"module": m, "k": k, "tirage": r, "regle": nom, "q_train": q,
                                   **mesures(y[reste], v, fr["reference_A"])})
    t = pd.DataFrame(lignes).drop(columns="tirage")
    return t.groupby(["module", "k", "regle"]).agg(["mean", "std"])


def plan_pilote(p_abs, p_rel):
    lignes = []
    for m in sorted(p_abs.code_module.unique()):
        d, q, f = regles_sans_etiquette(p_abs, p_rel, m)
        pres = sorted(d.code_presentation.unique())  # 2013B < 2013J < 2014B < 2014J
        cal = (d.code_presentation == pres[0]).values
        test = ~cal
        y = d[TARGET].values.astype(bool)
        ft = {nom: v[test] for nom, v in f.items()}
        ft["platt"] = recalibrer(d.p_lomo.values[cal], y[cal], d.p_lomo.values[test]) >= 0.5
        for nom, v in ft.items():
            lignes.append({"module": m, "presentation_calibration": pres[0], "n_calibration": int(cal.sum()),
                           "n_test": int(test.sum()), "regle": nom, "q_train": q,
                           **mesures(y[test], v, ft["reference_A"])})
    return pd.DataFrame(lignes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, default=0.25)
    args = ap.parse_args()
    c = f"{round(args.cutoff * 100):02d}"
    p_abs = pd.read_parquet(RES / "preds" / f"preds_rf_active_c{c}.parquet")
    p_rel = pd.read_parquet(RES / "preds" / f"preds_rf_rel_active_c{c}.parquet")
    cle = ["code_module", "code_presentation", "id_student", TARGET]
    assert (p_abs[cle].values == p_rel[cle].values).all(), "prédictions désalignées"

    k = plan_k(p_abs, p_rel)
    k.to_csv(RES / f"remedes_k_c{c}.csv")
    pil = plan_pilote(p_abs, p_rel)
    pil.to_csv(RES / f"remedes_pilote_c{c}.csv", index=False)

    pd.set_option("display.width", 220)
    cols = ["derive_ref", "ecart_risque", "precision", "rappel", "F1"]
    km = k.xs("mean", axis=1, level=1)
    res = km.xs(100, level="k")[cols].assign(abs_derive=lambda t: t.derive_ref.abs())
    print("Plan k = 100 (moyennes sur les modules ; dérive en valeur absolue) :\n",
          res.groupby("regle")[["abs_derive", "ecart_risque", "precision", "rappel", "F1"]].mean().round(3).to_string())
    et = k.xs("std", axis=1, level=1).xs(100, level="k")
    print("\nPlatt k = 100, écart-type entre tirages (moyenne sur les modules) :\n",
          et.xs("platt", level="regle")[["signalement", "precision", "F1"]].mean().round(3).to_string())
    print("\nPlan pilote (moyennes sur les modules) :\n",
          pil.assign(abs_derive=pil.derive_ref.abs()).groupby("regle")[
              ["abs_derive", "ecart_risque", "precision", "rappel", "F1"]].mean().round(3).to_string())
    print("\nGGG, plan pilote :\n", pil[pil.module == "GGG"][["regle", "signalement", "derive_ref",
                                                             "ecart_risque", "precision", "rappel", "F1"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
