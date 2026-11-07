#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Régénère TOUS les résultats, figures, tableaux et chiffres de l'article depuis
les 7 CSV OULAD placés dans data/OULAD/, puis compile paper/main.pdf.

Protocole (B) : LOMO strict, sans étudiants communs (D14).
Durée indicative (16 Go de RAM, 16 cœurs) : ~5 h, dont ~40 min pour SHAP,
~1 h 30 pour les modèles séquentiels (T5) et ~45 min pour les 10 graines.

  python run_all.py                    tout
  python run_all.py --sans-shap        réutilise les résultats SHAP existants
  python run_all.py --sans-sequences   réutilise les résultats T5 existants
  python run_all.py --sans-graines     réutilise les relances multi-graines existantes
  python run_all.py --sans-latex       ne compile pas l'article
  python run_all.py --depuis 10        reprend à l'étape 10 (numérotation de --liste)
  python run_all.py --liste            affiche les étapes numérotées

Seule exception : results/reference_rf_c25.txt, sortie des scripts initiaux
(commit 6ac9053), conservée telle quelle comme point de départ de l'audit.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent
PY = sys.executable
DATA = ["--data", "data/OULAD"]


def etapes(sans_shap: bool, sans_sequences: bool, sans_graines: bool):
    e = [("Tests anti-fuite", [PY, "-m", "pytest", "-q", "tests"])]
    # T2 / T3 : courbes et prédictions (5 cutoffs)
    e += [("Courbe, variables absolues", [PY, "src/03_early_curve.py", *DATA]),
          ("Courbe, variables relatives", [PY, "src/03_early_curve.py", *DATA,
                                           "--variables", "relatives", "--populations", "active"])]
    for v in ["relatives", "absolues"]:
        e.append((f"Courbe, cible Withdrawn, {v}", [PY, "src/03_early_curve.py", *DATA, "--variables", v,
                                                    "--cible", "withdrawn", "--populations", "active"]))
    for v in ["absolues", "relatives"]:
        for c in ["withdrawn_fail", "withdrawn"]:
            e.append((f"Analyse du transfert, {v}, {c}",
                      [PY, "src/04_transfer_analysis.py", *DATA, "--cutoff", "0.25", "--variables", v, "--cible", c]))
    # T4
    if not sans_shap:
        e += [(f"SHAP par module, {v}", [PY, "src/05_shap_by_module.py", *DATA, "--variables", v])
              for v in ["relatives", "absolues"]]
    e.append(("Stabilité SHAP", [PY, "src/06_shap_stability.py"]))
    # T6
    e.append(("Équité", [PY, "src/07_fairness.py", "--cutoff", "0.25"]))
    # T5
    if not sans_sequences:
        e.append(("Modèles séquentiels et boosting", [PY, "src/09_sequence_models.py", *DATA]))
    # Révision : robustesse et remèdes (D12–D14)
    e.append(("LOMO standard (comparaison au strict)", [PY, "src/10_overlap_check.py", *DATA]))
    if not sans_graines:
        e.append(("Variabilité due à la graine", [PY, "src/11_seed_variability.py", *DATA]))
    e.append(("Incertitude, calibration, seuil", [PY, "src/12_robustness.py"]))
    e.append(("Contrôle du chevauchement", [PY, "src/13_overlap_control.py", *DATA]))
    e.append(("Remèdes à armes égales", [PY, "src/14_remedies.py"]))
    e.append(("Normalisations alternatives", [PY, "src/15_normalisations.py", *DATA]))
    e.append(("Taille minimale de cohorte", [PY, "src/16_cohort_size.py", *DATA]))
    e.append(("Régression logistique", [PY, "src/15_normalisations.py", *DATA, "--model", "logit"]))
    e.append(("Forêt non pondérée calibrée", [PY, "src/15_normalisations.py", *DATA, "--model", "rf_cal",
                                             "--cutoffs", "0.25", "--variantes", "absolues", "rang"]))
    e.append(("Average Precision et fiabilité", [PY, "src/17_ap_reliability.py"]))
    e.append(("Équité : effets fixes et intersections", [PY, "src/18_fairness_fe.py"]))
    e.append(("Contrôle de la taille d'entraînement", [PY, "src/19_size_control.py", *DATA]))
    # Figures (thèse en français, article en anglais) puis chiffres de l'article
    for f in ["early_curve", "transfer", "shap", "fairness", "sequence", "reliability"]:
        for lang in ["fr", "en"]:
            e.append((f"Figure {f} ({lang})", [PY, f"src/fig_{f}.py", "--lang", lang]))
    e.append(("Chiffres et tableaux de l'article", [PY, "src/08_paper_assets.py"]))
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sans-shap", action="store_true")
    ap.add_argument("--sans-sequences", action="store_true")
    ap.add_argument("--sans-graines", action="store_true")
    ap.add_argument("--sans-latex", action="store_true")
    ap.add_argument("--depuis", type=int, default=1, help="Numéro de l'étape de reprise (voir --liste).")
    ap.add_argument("--liste", action="store_true", help="Affiche les étapes numérotées et s'arrête.")
    args = ap.parse_args()

    if not (RACINE / "data" / "OULAD" / "studentVle.csv").exists():
        sys.exit("Données absentes : placer les 7 CSV OULAD dans data/OULAD/ (voir README).")
    env = {**os.environ, "PYTHONUTF8": "1"}
    debut = time.time()
    liste = etapes(args.sans_shap, args.sans_sequences, args.sans_graines)
    if args.liste:
        for i, (nom, _) in enumerate(liste, 1):
            print(f"{i:3d}  {nom}")
        return
    for i, (nom, cmd) in enumerate(liste, 1):
        if i < args.depuis:
            continue
        t = time.time()
        print(f"\n=== {nom}", flush=True)
        r = subprocess.run(cmd, cwd=RACINE, env=env)
        if r.returncode != 0:
            sys.exit(f"ÉCHEC : {nom} (code {r.returncode})")
        print(f"    ({time.time() - t:.0f} s)", flush=True)

    if not args.sans_latex:
        if shutil.which("latexmk") is None:
            print("\nlatexmk introuvable : article non compilé.")
        else:
            print("\n=== Compilation de l'article", flush=True)
            try:
                # Délai : MiKTeX peut attendre en silence l'installation d'un paquet manquant.
                # main.tex : version JLA (biber) ; ieee/main.tex : version IEEE conservée.
                for doc in ["main.tex", "supplementary.tex", "ieee/main.tex"]:
                    r = subprocess.run(["latexmk", "-pdf", "-cd", "-interaction=nonstopmode", "-halt-on-error", doc],
                                       cwd=RACINE / "paper", stdout=subprocess.DEVNULL, timeout=600)
                    if r.returncode != 0:
                        break
            except subprocess.TimeoutExpired:
                sys.exit("ÉCHEC : compilation LaTeX bloquée (paquet MiKTeX manquant ? essayer "
                         "« pdflatex --disable-installer main.tex » dans paper/ pour le nommer)")
            if r.returncode != 0:
                sys.exit("ÉCHEC : compilation LaTeX (voir paper/main.log)")
    print(f"\nTerminé en {(time.time() - debut) / 60:.0f} min.")


if __name__ == "__main__":
    main()
