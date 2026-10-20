# Chaîne d'analyse OULAD — prédiction précoce et transférabilité

Préparation de l'article méthodologique sur OULAD, pensé comme répétition
du modèle qui sera appliqué aux données UN-CHK.

## Le créneau visé

La prédiction du décrochage sur OULAD est très publiée. Pour apporter du
neuf, l'angle retenu combine deux limites récurrentes de la littérature
récente :

1. **Prédiction précoce honnête.** N'utiliser que les données du **début du
   cours** (par défaut les premiers 25 % de la session), pas les variables
   calculées sur toute la session — reproche fait à la plupart des travaux
   (p. ex. l'audit d'équité MDPI *Sustainability* 2026, qui note son
   caractère rétrospectif).

2. **Généralisation à un contexte non vu.** Les articles courants utilisent
   un découpage aléatoire qui place les mêmes modules dans l'entraînement et
   le test : résultats optimistes et non transférables. Le Frontiers in
   Computer Science 2026 (DKBNE-AlexNet) liste explicitement « une seule
   institution » comme limite, et l'audit d'équité cite « tester le même
   protocole sur d'autres établissements » comme piste ouverte.

**Proposition :** comparer deux protocoles d'évaluation —
(A) découpage aléatoire, (B) *leave-one-module-out* (entraîner sur tous les
modules sauf un, tester sur le module laissé de côté). L'**écart de
performance (A − B)** mesure ce qui ne se transfère pas d'un contexte à un
autre. C'est à la fois :
- une contribution méthodologique nette sur OULAD (créneau peu exploité) ;
- la validation du protocole avant de passer aux données UN-CHK, où « le
  module non vu » devient « la formation, ou le pays, non vu ».

Cet angle est cohérent avec le titre de votre thèse (système « sensible au
contexte ») : il quantifie la dépendance au contexte au lieu de la supposer.

## Fichiers

| Fichier | Rôle |
|---|---|
| `make_synthetic_oulad.py` | Génère un échantillon au **schéma exact d'OULAD**, pour tester le code sans les vraies données. **Jamais pour un résultat de recherche.** |
| `01_build_features.py` | Construit la table de modélisation : variables démographiques + engagement précoce (VLE tronqué) + évaluations précoces ; cible `at_risk`. |
| `02_train_evaluate.py` | Entraîne (Random Forest / régression logistique) et compare les protocoles (A) et (B) ; importance des variables. |

## Installation

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Les vraies données

Télécharger OULAD (licence **CC-BY 4.0**) sur votre machine :
<https://analyse.kmi.open.ac.uk/open_dataset>

Décompresser les **7 CSV** (`studentInfo.csv`, `studentRegistration.csv`,
`studentVle.csv`, `studentAssessment.csv`, `assessments.csv`, `vle.csv`,
`courses.csv`) dans `data/OULAD/` (ignoré par git).

**Citation obligatoire :** Kuzilek, J., Hlosta, M., Zdrahal, Z. (2017).
*Open University Learning Analytics dataset.* Scientific Data 4, 170171.

## Exécution

**Tout régénérer** (résultats, figures FR/EN, chiffres et tableaux de l'article,
`paper/main.pdf`) : `python run_all.py` (~1 h ; `--sans-shap` pour réutiliser
les résultats SHAP). Les étapes détaillées :

Depuis la racine du dépôt, avec le venv activé :

```
# 0) garde-fou anti-fuite (données synthétiques)
python -m pytest tests

# 1) une configuration : fenêtre précoce à 25 %, population active au cutoff
python src/01_build_features.py --data data/OULAD --cache data/cache --out results/features_c25.parquet
python src/02_train_evaluate.py --features results/features_c25.parquet --model rf

# 2) T2 : courbe précocité vs performance (5 cutoffs × 2 populations, ~8 min)
python src/03_early_curve.py --data data/OULAD
python src/fig_early_curve.py

# 3) T3 : hétérogénéité du transfert (classement vs calibration, remèdes)
python src/03_early_curve.py --data data/OULAD --variables relatives --populations active
python src/04_transfer_analysis.py --data data/OULAD --cutoff 0.25
python src/04_transfer_analysis.py --data data/OULAD --cutoff 0.25 --variables relatives
python src/fig_transfer.py

# 4) T4 : SHAP par module (~16 et ~25 min, 3 modules en parallèle, ~2 Go de RAM chacun)
python src/05_shap_by_module.py --data data/OULAD --variables relatives
python src/05_shap_by_module.py --data data/OULAD --variables absolues
python src/06_shap_stability.py
python src/fig_shap.py

# 5) T6 : équité (prédictions déjà enregistrées) et cible « Withdrawn seul »
python src/07_fairness.py --cutoff 0.25
python src/fig_fairness.py
python src/03_early_curve.py --data data/OULAD --variables relatives --cible withdrawn --populations active
python src/03_early_curve.py --data data/OULAD --variables absolues --cible withdrawn --populations active
python src/04_transfer_analysis.py --data data/OULAD --cutoff 0.25 --variables relatives --cible withdrawn
python src/04_transfer_analysis.py --data data/OULAD --cutoff 0.25 --variables absolues --cible withdrawn

# 6) T7 : figures en anglais, chiffres et tableaux de l'article, compilation
python src/fig_early_curve.py --lang en   # idem fig_transfer, fig_shap, fig_fairness
python src/08_paper_assets.py             # paper/generated/macros.tex + tableaux
cd paper && latexmk -pdf main.tex
```

## Article (`paper/`)

`paper/main.tex` (IEEEtran, format revue, en anglais) ne contient aucun chiffre
écrit à la main : chaque nombre cité est une commande `\resXxx` définie dans
`paper/generated/macros.tex`, générée par `src/08_paper_assets.py` depuis
`results/`. Les figures de l'article sont dans `figures/en/`, celles de la
thèse (en français) dans `figures/`. Les choix de méthode sont justifiés dans
`DECISIONS.md`.

`--population all` reproduit le protocole initial (étudiants déjà désinscrits
au cutoff conservés) ; il redonne les chiffres de `results/reference_rf_c25.txt`
aux notes près (les remises postérieures au cutoff sont désormais exclues).

Code : `src/oulad_features.py` (variables, spécifique OULAD) et
`src/transfer_eval.py` (protocoles génériques, colonne de contexte en paramètre,
réutilisables sur UN-CHK).

## Pistes pour compléter avant soumission
- Ajouter des variables **séquentielles** (clics par semaine, tendance) et
  comparer à un modèle temporel (LSTM / GRU) — l'actuel est agrégé.
- Ajouter l'**interprétabilité SHAP** pour montrer si les variables
  importantes changent d'un module à l'autre (indice de non-transférabilité).
- Analyse d'**équité** par sous-groupe (genre, IMD) dans le protocole LOMO.
- Discuter le transfert vers l'UN-CHK : quelles variables OULAD n'existeront
  pas, lesquelles seront équivalentes.
