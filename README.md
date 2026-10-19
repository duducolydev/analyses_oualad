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
pip install pandas scikit-learn
# optionnel, pour l'interprétabilité fine : pip install shap pyarrow
```

## Les vraies données

Télécharger OULAD (licence **CC-BY 4.0**) sur votre machine :
<https://analyse.kmi.open.ac.uk/open_dataset>

Décompresser les **7 CSV** (`studentInfo.csv`, `studentRegistration.csv`,
`studentVle.csv`, `studentAssessment.csv`, `assessments.csv`, `vle.csv`,
`courses.csv`) dans un dossier, par exemple `./OULAD`.

**Citation obligatoire :** Kuzilek, J., Hlosta, M., Zdrahal, Z. (2017).
*Open University Learning Analytics dataset.* Scientific Data 4, 170171.

## Exécution

```
# 1) test du code sur données synthétiques
python make_synthetic_oulad.py
python 01_build_features.py --data ./data_synthetique --out features.csv
python 02_train_evaluate.py --features features.csv

# 2) sur les vraies données, fenêtre précoce à 25 %
python 01_build_features.py --data ./OULAD --out features_oulad.csv --cutoff 0.25
python 02_train_evaluate.py --features features_oulad.csv --model rf
```

Faire varier `--cutoff` (0.10, 0.25, 0.50) donne la courbe « précocité vs
performance », un résultat central de l'article.

## Pistes pour compléter avant soumission
- Ajouter des variables **séquentielles** (clics par semaine, tendance) et
  comparer à un modèle temporel (LSTM / GRU) — l'actuel est agrégé.
- Ajouter l'**interprétabilité SHAP** pour montrer si les variables
  importantes changent d'un module à l'autre (indice de non-transférabilité).
- Analyse d'**équité** par sous-groupe (genre, IMD) dans le protocole LOMO.
- Discuter le transfert vers l'UN-CHK : quelles variables OULAD n'existeront
  pas, lesquelles seront équivalentes.
