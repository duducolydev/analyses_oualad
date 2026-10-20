#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4 — Stabilité des importances SHAP entre contextes (lit 05_shap_by_module.py).

Pour chaque variante de variables (absolues / relatives) :
  1. concordance transféré / local par module : Spearman entre les vecteurs
     d'importance (18 variables), recouvrement des 5 premières ;
  2. stabilité entre modules des importances LOCALES : Spearman moyen entre
     paires de modules, et similarité de chaque module aux autres ;
  3. variabilité par variable : part d'importance locale, min–max entre modules.

Sorties : results/shap_stability_c25_active.csv (par module et variante),
          results/shap_variable_spread_c25_active.csv (par variable et variante)
Usage : python src/06_shap_stability.py
"""
import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

VARIANTES = ["absolues", "relatives"]
TOP = 5


def analyse(imp: pd.DataFrame, variante: str):
    piv = imp.pivot_table(index="variable", columns=["modele", "code_module"], values="mean_abs_shap")
    loc, tra = piv["local"], piv["transfere"]
    mods = list(loc.columns)

    lignes = []
    for m in mods:
        top_l = set(loc[m].nlargest(TOP).index)
        top_t = set(tra[m].nlargest(TOP).index)
        autres = [spearmanr(loc[m], loc[o])[0] for o in mods if o != m]
        lignes.append({"variables": variante, "module": m,
                       "spearman_transfere_local": spearmanr(loc[m], tra[m])[0],
                       f"top{TOP}_communs": len(top_l & top_t),
                       "similarite_locale_aux_autres": sum(autres) / len(autres),
                       "top_local": ", ".join(loc[m].nlargest(3).index),
                       "top_transfere": ", ".join(tra[m].nlargest(3).index)})
    par_module = pd.DataFrame(lignes)

    paires = [spearmanr(loc[a], loc[b])[0] for a, b in combinations(mods, 2)]
    part = loc / loc.sum()
    etendue = pd.DataFrame({"variables": variante, "part_moyenne": part.mean(axis=1),
                            "part_min": part.min(axis=1), "part_max": part.max(axis=1),
                            "module_max": part.idxmax(axis=1), "module_min": part.idxmin(axis=1)})
    return par_module, etendue, sum(paires) / len(paires)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="_c25_active")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    pm, et = [], []
    for v in VARIANTES:
        imp = pd.read_csv(args.out / f"shap_by_module{args.meta}_{v}.csv")
        a, b, stab = analyse(imp, v)
        pm.append(a)
        et.append(b)
        print(f"\n=== Variables {v} ===  Spearman moyen entre paires de modules (local) : {stab:.3f}")
        print(a.round(3).to_string(index=False))
    pd.concat(pm).to_csv(args.out / f"shap_stability{args.meta}.csv", index=False)
    et = pd.concat(et)
    et.to_csv(args.out / f"shap_variable_spread{args.meta}.csv", index_label="variable")
    pd.set_option("display.width", 200)
    print("\nPart d'importance locale par variable (min–max entre modules) :")
    print(et.reset_index().pivot(index="variable", columns="variables",
                                 values=["part_moyenne", "part_min", "part_max"])
          .sort_values(("part_moyenne", "relatives"), ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
