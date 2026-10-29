#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Révision (étape 2) — Incertitude, calibration, seuil,
à partir des prédictions enregistrées (cutoff 25 %, population active) et des
relances multi-graines de 11_seed_variability.py.

1. IC 95 % par bootstrap groupé par étudiant (1000 tirages, APPARIÉ entre
   variantes et protocoles) : AUC (A), AUC (B), coût, |dérive| moyenne ; et
   différences relatives − absolues.
2. Niveau module (n = 7) : test de Wilcoxon apparié relatives vs absolues et
   nombre de modules améliorés.
3. Variabilité due à la graine : écart-type par module de l'AUC (B) et de la
   dérive ; différence relatives − absolues par module, moyenne ± écart-type
   sur les graines.
4. Calibration par module, (A) et (B) : Brier, risque prédit moyen, risque
   observé, calibration-in-the-large (intercept avec logit(p) en offset) et
   pente de calibration. Avec des poids de classe équilibrés, les probabilités
   ne sont pas calibrées même en (A) : on compare (B) à (A).
5. Seuil : |dérive| aux seuils 0,3–0,7, et décalage du score moyen (sans seuil).
(L'alternative top-q est comparée aux autres remèdes dans 14_remedies.py.)

Sorties : results/robustesse_*_c25.csv ; résumé à l'écran.
Usage : python src/12_robustness.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RES = Path("results")
TARGET = "at_risk"
VARIANTES = {"absolues": "rf", "relatives": "rf_rel"}
SEUILS = [0.3, 0.4, 0.5, 0.6, 0.7]
N_BOOT = 1000


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def charger_preds(c):
    p = {v: pd.read_parquet(RES / "preds" / f"preds_{t}_active_c{c}.parquet") for v, t in VARIANTES.items()}
    cle = ["code_module", "code_presentation", "id_student", TARGET]
    assert (p["absolues"][cle].values == p["relatives"][cle].values).all()
    return p


# 1. Bootstrap apparié -------------------------------------------------------
def quantites(p, w, mods, y):
    """AUC (A), AUC (B), coût, |dérive| moyennes sur les modules, pondérées par w."""
    out = {}
    for v, d in p.items():
        aA, aB, der = [], [], []
        for m in mods:
            k = m["masque"]
            wk = w[k]
            aA.append(roc_auc_score(y[k], d["p_aleatoire"].values[k], sample_weight=wk))
            aB.append(roc_auc_score(y[k], d["p_lomo"].values[k], sample_weight=wk))
            fA = np.average(d["p_aleatoire"].values[k] >= 0.5, weights=wk)
            fB = np.average(d["p_lomo"].values[k] >= 0.5, weights=wk)
            der.append(abs(fB - fA))
        out[v] = {"AUC (A)": np.mean(aA), "AUC (B)": np.mean(aB),
                  "Coût": np.mean(aA) - np.mean(aB), "|Dérive|": np.mean(der)}
    return out


def bootstrap(p):
    base = p["absolues"]
    y = base[TARGET].values
    mods = [{"nom": m, "masque": (base.code_module == m).values} for m in sorted(base.code_module.unique())]
    etu, idx = np.unique(base.id_student.values, return_inverse=True)
    rng = np.random.default_rng(0)
    point = quantites(p, np.ones(len(y)), mods, y)
    tirages = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(etu), len(etu)), minlength=len(etu))[idx].astype(float)
        tirages.append(quantites(p, w, mods, y))
    lignes = []
    for q in ["AUC (A)", "AUC (B)", "Coût", "|Dérive|"]:
        for v in VARIANTES:
            b = np.array([t[v][q] for t in tirages])
            lignes.append({"quantite": q, "comparaison": v, "estimation": point[v][q],
                           "ic_bas": np.percentile(b, 2.5), "ic_haut": np.percentile(b, 97.5)})
        b = np.array([t["relatives"][q] - t["absolues"][q] for t in tirages])
        lignes.append({"quantite": q, "comparaison": "relatives − absolues",
                       "estimation": point["relatives"][q] - point["absolues"][q],
                       "ic_bas": np.percentile(b, 2.5), "ic_haut": np.percentile(b, 97.5)})
    return pd.DataFrame(lignes)


# 2. Niveau module -------------------------------------------------------------
def par_module(p):
    lignes = []
    for v, d in p.items():
        for m, g in d.groupby("code_module"):
            y = g[TARGET].values
            fA, fB = (g.p_aleatoire >= 0.5).mean(), (g.p_lomo >= 0.5).mean()
            lignes.append({"variables": v, "module": m, "AUC (B)": roc_auc_score(y, g.p_lomo),
                           "Coût": roc_auc_score(y, g.p_aleatoire) - roc_auc_score(y, g.p_lomo),
                           "|Dérive|": abs(fB - fA)})
    t = pd.DataFrame(lignes).pivot(index="module", columns="variables")
    tests = []
    for q, mieux in [("AUC (B)", "plus"), ("Coût", "moins"), ("|Dérive|", "moins")]:
        diff = t[(q, "relatives")] - t[(q, "absolues")]
        ameliores = int((diff > 0).sum() if mieux == "plus" else (diff < 0).sum())
        tests.append({"quantite": q, "modules_ameliores": ameliores, "n_modules": len(diff),
                      "wilcoxon_p": wilcoxon(diff).pvalue})
    return t, pd.DataFrame(tests)


# 3. Graines -------------------------------------------------------------------
def graines(c):
    f = RES / f"seed_variability_c{c}.csv"
    if not f.exists():
        return None, None
    d = pd.read_csv(f)
    s = d.pivot_table(index=["variables", "graine", "module"], columns="protocole", values="signalement")
    der = (s["lomo"] - s["aleatoire"]).rename("derive")
    auc = d[d.protocole == "lomo"].set_index(["variables", "graine", "module"])["AUC"].rename("auc_B")
    t = pd.concat([auc, der], axis=1).reset_index()
    resume = t.groupby(["variables", "module"]).agg(auc_B_moy=("auc_B", "mean"), auc_B_et=("auc_B", "std"),
                                                    derive_moy=("derive", "mean"), derive_et=("derive", "std"))
    w = t.pivot_table(index=["graine", "module"], columns="variables", values="auc_B").dropna()
    if not {"absolues", "relatives"} <= set(w.columns) or w.empty:
        return resume, None  # relances encore partielles
    diff = (w["relatives"] - w["absolues"]).groupby("module").agg(["mean", "std", "min", "max"])
    return resume, diff.rename(columns=lambda x: f"rel_moins_abs_auc_B_{x}")


# 4. Calibration -----------------------------------------------------------------
def citl(y, p):
    """Intercept a tel que mean(sigmoid(a + logit p)) = mean(y) (MLE avec offset)."""
    lp = logit(p)
    f = lambda a: (1 / (1 + np.exp(-(a + lp)))).mean() - y.mean()
    return brentq(f, -20, 20)


def calibration(p):
    lignes = []
    for v, d in p.items():
        for proto, col in [("aleatoire", "p_aleatoire"), ("lomo", "p_lomo")]:
            for m, g in d.groupby("code_module"):
                y, q = g[TARGET].values, g[col].values
                lr = LogisticRegression(penalty=None).fit(logit(q)[:, None], y)
                lignes.append({"variables": v, "protocole": proto, "module": m,
                               "brier": np.mean((q - y) ** 2), "p_moyen": q.mean(), "risque_observe": y.mean(),
                               "citl": citl(y, q), "pente": lr.coef_[0, 0]})
    return pd.DataFrame(lignes)


# 5. Seuil -----------------------------------------------------------------------
def seuils(p):
    lignes = []
    for v, d in p.items():
        for s in SEUILS:
            f = d.assign(fA=d.p_aleatoire >= s, fB=d.p_lomo >= s).groupby("code_module")[["fA", "fB"]].mean()
            der = f.fB - f.fA
            lignes.append({"variables": v, "seuil": s, "derive_moy_abs": der.abs().mean(), "derive_GGG": der["GGG"]})
        sh = d.groupby("code_module").apply(lambda g: g.p_lomo.mean() - g.p_aleatoire.mean())
        lignes.append({"variables": v, "seuil": "score moyen", "derive_moy_abs": sh.abs().mean(),
                       "derive_GGG": sh["GGG"]})
    return pd.DataFrame(lignes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, default=0.25)
    ap.add_argument("--sans-bootstrap", action="store_true",
                    help="Réutilise results/robustesse_bootstrap_c<cutoff>.csv (~10 min sinon).")
    args = ap.parse_args()
    c = f"{round(args.cutoff * 100):02d}"
    pd.set_option("display.width", 220)
    p = charger_preds(c)

    if args.sans_bootstrap:
        b = pd.read_csv(RES / f"robustesse_bootstrap_c{c}.csv")
    else:
        b = bootstrap(p)
        b.to_csv(RES / f"robustesse_bootstrap_c{c}.csv", index=False)
    print("1. IC 95 % (bootstrap groupé par étudiant, apparié) :\n", b.round(3).to_string(index=False))

    t, tests = par_module(p)
    t.to_csv(RES / f"robustesse_modules_c{c}.csv")
    tests.to_csv(RES / f"robustesse_wilcoxon_c{c}.csv", index=False)
    print("\n2. Par module :\n", t.round(3).to_string(), "\n", tests.round(3).to_string(index=False))

    res_g, diff_g = graines(c)
    if res_g is not None:
        res_g.to_csv(RES / f"robustesse_graines_c{c}.csv")
        print("\n3. Graines (écart-type par module) :\n", res_g.round(3).to_string())
    if diff_g is not None:
        diff_g.to_csv(RES / f"robustesse_graines_diff_c{c}.csv")
        print("\nRelatives − absolues, AUC (B), sur les graines :\n", diff_g.round(3).to_string())

    cal = calibration(p)
    cal.to_csv(RES / f"robustesse_calibration_c{c}.csv", index=False)
    piv = cal.pivot_table(index=["variables", "module"], columns="protocole", values=["p_moyen", "citl", "pente", "brier"])
    print("\n4. Calibration :\n", piv.round(3).to_string())

    s = seuils(p)
    s.to_csv(RES / f"robustesse_seuils_c{c}.csv", index=False)
    print("\n5. Sensibilité au seuil :\n", s.round(3).to_string(index=False))



if __name__ == "__main__":
    main()
