#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T6 — Équité par sous-groupe, à partir des prédictions de 03_early_curve.py.

Attributs : genre, quintile d'IMD (Q1 = le plus défavorisé ; « 10-20 » sans %
dans OULAD est rattaché à Q1), handicap. Pour chaque variante de variables
(absolues / relatives) et chaque protocole ((A) aléatoire hors-pli, (B) LOMO) :
  - métriques par groupe : signalement (p >= 0,5), TPR (rappel), FPR, PPV
    (précision), AUC ;
  - écarts entre groupes contrastés (F − M, Q1 − Q5, handicap O − N) avec IC 95 %
    par bootstrap groupé par étudiant (1000 tirages) ;
  - variation des écarts, avec les MÊMES tirages (bootstrap apparié) :
    transfert (B − A) et variables (relatives − absolues, en LOMO).

Sorties : results/fairness_groups<meta>.csv, fairness_gaps<meta>.csv,
          fairness_gap_changes<meta>.csv
Usage : python src/07_fairness.py --cutoff 0.25
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from oulad_features import CLE

TARGET = "at_risk"
N_BOOT = 1000
QUINTILE = {"0-10%": "Q1", "10-20": "Q1", "10-20%": "Q1", "20-30%": "Q2", "30-40%": "Q2",
            "40-50%": "Q3", "50-60%": "Q3", "60-70%": "Q4", "70-80%": "Q4",
            "80-90%": "Q5", "90-100%": "Q5"}
ATTRIBUTS = {"genre": ("F", "M"), "imd": ("Q1", "Q5"), "handicap": ("Y", "N")}
METRIQUES = ["signalement", "TPR", "FPR", "PPV"]
VARIANTES = {"absolues": "rf", "relatives": "rf_rel"}
PROTOCOLES = {"aleatoire": "p_aleatoire", "lomo": "p_lomo"}


def metriques_ponderees(w, y, f, g):
    """Métriques au seuil pour le masque de groupe g, pondérées par w (n_boot × n)."""
    wg = w * g
    vp = (wg * (y * f)).sum(axis=-1)
    return {"signalement": (wg * f).sum(axis=-1) / wg.sum(axis=-1),
            "TPR": vp / (wg * y).sum(axis=-1),
            "FPR": (wg * ((1 - y) * f)).sum(axis=-1) / (wg * (1 - y)).sum(axis=-1),
            "PPV": vp / (wg * f).sum(axis=-1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, default=0.25)
    ap.add_argument("--population", default="active")
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    suffixe = f"{args.population}_c{round(args.cutoff*100):02d}"

    info = pd.read_parquet(args.cache / "studentInfo.parquet")[CLE + ["id_student", "gender", "imd_band", "disability"]]
    preds = {}
    for v, tag in VARIANTES.items():
        preds[v] = pd.read_parquet(args.out / "preds" / f"preds_{tag}_{suffixe}.parquet")
    base = preds["absolues"]
    for v in preds:
        assert (preds[v][CLE + ["id_student", TARGET]].values == base[CLE + ["id_student", TARGET]].values).all()
    base = base.merge(info, on=CLE + ["id_student"], how="left", validate="one_to_one")
    groupes = {"genre": base["gender"].values,
               "imd": base["imd_band"].map(QUINTILE).fillna("inconnu").values,
               "handicap": base["disability"].values}
    y = base[TARGET].values.astype(float)

    # Bootstrap groupé par étudiant, tirages communs à toutes les configurations
    etu, idx = np.unique(base["id_student"].values, return_inverse=True)
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(etu), len(etu)), minlength=len(etu))[idx]
                  for _ in range(N_BOOT)]).astype(np.float32)
    un = np.ones((1, len(y)), dtype=np.float32)

    lignes_g, ecarts, boots = [], [], {}
    for v in VARIANTES:
        for proto, col in PROTOCOLES.items():
            p = preds[v][col].values
            f = (p >= 0.5).astype(float)
            for att, gr in groupes.items():
                for modalite in sorted(set(gr)):
                    g = (gr == modalite).astype(float)
                    m = {k: float(val[0]) for k, val in metriques_ponderees(un, y, f, g).items()}
                    lignes_g.append({"variables": v, "protocole": proto, "attribut": att, "groupe": modalite,
                                     "n": int(g.sum()), "taux_risque": y[g == 1].mean(),
                                     "AUC": roc_auc_score(y[g == 1], p[g == 1]), **m})
                g1, g2 = ATTRIBUTS[att]
                b1 = metriques_ponderees(W, y, f, (gr == g1).astype(float))
                b2 = metriques_ponderees(W, y, f, (gr == g2).astype(float))
                p1 = metriques_ponderees(un, y, f, (gr == g1).astype(float))
                p2 = metriques_ponderees(un, y, f, (gr == g2).astype(float))
                for met in METRIQUES:
                    d = b1[met] - b2[met]
                    boots[(v, proto, att, met)] = d
                    ecarts.append({"variables": v, "protocole": proto, "attribut": att,
                                   "contraste": f"{g1} − {g2}", "metrique": met,
                                   "ecart": float(p1[met][0] - p2[met][0]),
                                   "ic_bas": np.nanpercentile(d, 2.5), "ic_haut": np.nanpercentile(d, 97.5)})
    ecarts = pd.DataFrame(ecarts)

    # Variation des écarts (bootstrap apparié)
    est = ecarts.set_index(["variables", "protocole", "attribut", "metrique"])["ecart"]
    changements = []
    for att in ATTRIBUTS:
        for met in METRIQUES:
            for nom, (a, b) in {"transfert (B − A), absolues": (("absolues", "lomo"), ("absolues", "aleatoire")),
                                "transfert (B − A), relatives": (("relatives", "lomo"), ("relatives", "aleatoire")),
                                "relatives − absolues, LOMO": (("relatives", "lomo"), ("absolues", "lomo"))}.items():
                d = boots[(*a, att, met)] - boots[(*b, att, met)]
                changements.append({"comparaison": nom, "attribut": att, "metrique": met,
                                    "variation": est[(*a, att, met)] - est[(*b, att, met)],
                                    "ic_bas": np.nanpercentile(d, 2.5), "ic_haut": np.nanpercentile(d, 97.5)})
    changements = pd.DataFrame(changements)

    meta = f"_c{round(args.cutoff*100):02d}_{args.population}"
    pd.DataFrame(lignes_g).to_csv(args.out / f"fairness_groups{meta}.csv", index=False)
    ecarts.to_csv(args.out / f"fairness_gaps{meta}.csv", index=False)
    changements.to_csv(args.out / f"fairness_gap_changes{meta}.csv", index=False)

    pd.set_option("display.width", 220)
    print(pd.DataFrame(lignes_g).round(3).to_string(index=False))
    print("\nÉcarts (IC 95 % bootstrap groupé par étudiant) :")
    print(ecarts.round(3).to_string(index=False))
    sig = changements[(changements.ic_bas > 0) | (changements.ic_haut < 0)]
    print("\nVariations d'écart dont l'IC 95 % exclut 0 :")
    print(sig.round(3).to_string(index=False) if len(sig) else "aucune")


if __name__ == "__main__":
    main()
