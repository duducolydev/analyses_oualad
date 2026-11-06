#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Révision (étape 3, revue 1) — Taille minimale de cohorte pour les rangs.

Le modèle relatif est entraîné comme d'habitude (LOMO strict, rangs calculés
dans les présentations entières des six modules d'entraînement). Dans le module
testé, on simule des cohortes plus petites : chaque présentation est découpée au
hasard en groupes d'environ n étudiants, et les rangs sont recalculés DANS
chaque groupe, avec les valeurs absolues du même jour. On mesure la dégradation
par rapport aux rangs calculés sur la présentation entière (n = « complet »).

n ∈ {10, 20, 50, 100, 200}, 5 découpages aléatoires ; cutoff 25 %.
Mesures par module : AUC, part signalée, |signalés − risque|, précision, F1,
et écart de part signalée au cas « complet ».

Sortie : results/cohort_size.csv
Usage : python src/16_cohort_size.py --data data/OULAD
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from oulad_features import CLE, VARIABLES_CONTEXTUELLES, charger, construire, normaliser
from transfer_eval import SEED, construire_modele, plis_contexte, scores

warnings.filterwarnings("ignore")
ID_COLS = ["id_student", "code_module", "code_presentation"]
TARGET = "at_risk"
TAILLES = [10, 20, 50, 100, 200]
N_DECOUPAGES = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--cutoff", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    tables = charger(args.data, args.cache)
    rel = construire(tables, args.cutoff, "active", True)
    absol = construire(tables, args.cutoff, "active", False)
    assert (rel[ID_COLS].values == absol[ID_COLS].values).all()
    feats = [c for c in rel.columns if c not in ID_COLS + [TARGET]]
    rng = np.random.default_rng(SEED)

    lignes = []
    for mod, tr, te in plis_contexte(rel, "code_module", exclure_communs="id_student"):
        modele = construire_modele(rel.loc[tr, feats], "rf", SEED).fit(rel.loc[tr, feats], rel.loc[tr, TARGET].values)
        y = rel.loc[te, TARGET].values
        p_complet = modele.predict_proba(rel.loc[te, feats])[:, 1]
        f_complet = (p_complet >= 0.5).mean()

        def mesurer(p, n, r):
            s = scores(y, p)
            f = (p >= 0.5).mean()
            lignes.append({"module": mod, "taille": n, "decoupage": r, "AUC": s["AUC"], "signalement": f,
                           "ecart_risque": abs(f - y.mean()), "ecart_au_complet": f - f_complet,
                           "precision": s["Précision"], "F1": s["F1"]})

        mesurer(p_complet, "complet", 0)
        test = absol.loc[te].copy()
        for n in TAILLES:
            for r in range(N_DECOUPAGES):
                groupe = np.empty(len(test), dtype=int)
                for _, idx in test.groupby("code_presentation").indices.items():
                    perm = rng.permutation(idx)
                    k = max(1, round(len(idx) / n))
                    for g, morceau in enumerate(np.array_split(perm, k)):
                        groupe[morceau] = g
                x = test.assign(_groupe=groupe)
                x[VARIABLES_CONTEXTUELLES] = normaliser(x, VARIABLES_CONTEXTUELLES, CLE + ["_groupe"], "rang")
                mesurer(modele.predict_proba(x[feats])[:, 1], n, r)
        print(f"[{mod}] fait", flush=True)

    d = pd.DataFrame(lignes)
    d.to_csv(args.out / "cohort_size.csv", index=False)
    d["abs_ecart_au_complet"] = d.ecart_au_complet.abs()
    par_mod = d.groupby(["taille", "module"])[["AUC", "ecart_risque", "abs_ecart_au_complet", "precision", "F1"]].mean()
    ordre = [str(t) for t in TAILLES] + ["complet"]
    resume = par_mod.groupby(level="taille").mean()
    resume.index = resume.index.astype(str)
    print("\nMoyennes sur les modules (et sur les découpages) :\n", resume.reindex(ordre).round(3).to_string())


if __name__ == "__main__":
    main()
