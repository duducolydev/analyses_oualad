# Brief pour Claude Code — Article OULAD : prédiction précoce et transférabilité

> À coller dans Claude Code au démarrage du projet (ou à enregistrer en
> `CLAUDE.md` à la racine du dépôt). Rédigé le 19/10/2026. Auteur : Doudou COLY,
> doctorant UN-CHK (thèse dirigée par le Pr El Hadji Mamadou NGUER).

---

## 1. Contexte et objectif

Je prépare un **article méthodologique** sur le jeu de données public **OULAD**
(Open University Learning Analytics Dataset, licence CC-BY 4.0). Cet article est
une étape de ma thèse sur la **prévention du décrochage** dans l'enseignement
supérieur en ligne (système d'accompagnement piloté par l'IA, « sensible au
contexte »). Il sert de **répétition méthodologique** avant l'application aux
données Moodle de l'UN-CHK, encore en cours d'obtention.

**Un premier article** (profils de vulnérabilité, enquête auprès de 450
étudiants UN-CHK) est déjà accepté au **CNRIA 2026** (publication IEEE). Le
présent article doit être une contribution **distincte et complémentaire**, pas
un doublon.

## 2. L'angle de recherche (le cœur, ne pas s'en écarter)

La prédiction du décrochage sur OULAD est **saturée** : un simple « j'entraîne un
classifieur, j'obtiens une bonne AUC » ne sera pas publiable. L'angle retenu
combine deux limites récurrentes de la littérature récente :

1. **Prédiction précoce honnête** : n'utiliser que les données disponibles au
   **début du cours** (fenêtre réglable, défaut = premiers 25 % de la session),
   pas les variables calculées sur toute la session.
2. **Transférabilité à un contexte non vu** : la plupart des travaux évaluent
   par découpage aléatoire, où les mêmes modules sont dans l'entraînement et le
   test (optimiste, non transférable). Je compare ce protocole à un
   **leave-one-module-out (LOMO)** : entraîner sur tous les modules sauf un,
   tester sur le module laissé de côté.

**La contribution = l'écart de performance (aléatoire − LOMO), et surtout son
hétérogénéité d'un module à l'autre.** Ce « module non vu » est un proxy du
« contexte non vu » (demain : une formation UN-CHK, ou le passage UK → Sénégal).
L'article quantifie ainsi la dépendance au contexte au lieu de la supposer, ce
qui colle au « sensible au contexte » de la thèse.

Références qui motivent le créneau (à citer et à positionner contre) :
- Frontiers in Computer Science 2026, *DKBNE-AlexNet* : liste « une seule
  institution » comme limite de généralisation.
- MDPI *Sustainability* 2026, audit d'équité régionale sur OULAD : signale son
  caractère rétrospectif et cite « tester sur d'autres établissements » comme
  piste ouverte.
- Revue systématique OULAD (ResearchGate, 2024) : à récupérer pour cadrer l'état
  de l'art.

## 3. Point de départ fourni (déjà écrit et testé)

Un pipeline minimal existe déjà (dossier `oulad_pipeline/`), **testé sur le vrai
OULAD**. À reprendre et étendre, pas à refaire :
- `make_synthetic_oulad.py` — génère un échantillon au schéma exact (tests).
- `01_build_features.py` — construit la table de modélisation (fenêtre précoce).
- `02_train_evaluate.py` — compare protocoles (A) aléatoire / (B) LOMO + importance.
- `README.md` — explique l'angle et l'usage.

### Résultats de référence déjà obtenus (Random Forest, cutoff 25 %)

À reproduire d'abord, puis à battre / analyser :

| Protocole | AUC | Rappel | Précision | F1 |
|---|---|---|---|---|
| (A) Aléatoire, groupé par étudiant | 0,882 | 0,748 | 0,857 | 0,799 |
| (B) LOMO, moyenne | 0,848 | 0,787 | 0,731 | 0,737 |

**Écart d'AUC moyen ≈ 0,033**, mais **très hétérogène selon le module** — c'est
l'histoire de l'article :

| Module testé (LOMO) | AUC | Rappel | Précision |
|---|---|---|---|
| AAA | 0,782 | 0,682 | 0,503 |
| BBB | 0,849 | 0,828 | 0,721 |
| CCC | 0,905 | 0,789 | 0,906 |
| DDD | 0,888 | 0,729 | 0,897 |
| EEE | 0,884 | 0,796 | 0,780 |
| FFF | 0,880 | 0,702 | 0,892 |
| GGG | 0,750 | 0,985 | 0,419 |

À noter pour la discussion : **GGG** (rappel 0,985 / précision 0,419) et **AAA**
(AUC 0,782) se transfèrent mal — le modèle y sur-prédit le risque. CCC/DDD se
transfèrent très bien. L'enjeu : **expliquer pourquoi** (profil du module,
poids des évaluations, calendrier, part VLE).

Variables les plus importantes à 25 % (permutation) : `mean_early_score`,
`last_activity`, `highest_education`, `active_days`, `studied_credits`.

## 4. Les données

- 7 CSV OULAD, schéma vérifié : `studentInfo` (32 593 étudiants), `studentVle`
  (10,6 M lignes, ~454 Mo), `studentRegistration`, `studentAssessment`,
  `assessments`, `vle`, `courses`. 7 modules, 22 présentations.
- Cible actuelle : `at_risk = 1` si `final_result ∈ {Withdrawn, Fail}`, sinon 0
  (taux à risque global 52,8 %).
- **Ne pas committer les CSV** dans le dépôt (volumineux + bonne pratique) :
  les mettre dans `data/OULAD/` et l'ajouter au `.gitignore`.
- **Citation obligatoire** : Kuzilek, J., Hlosta, M., Zdrahal, Z. (2017). *Open
  University Learning Analytics dataset.* Scientific Data 4, 170171.

## 5. Tâches à réaliser dans Claude Code (par ordre de priorité)

**T1. Mettre le projet en forme (dépôt propre)**
- Arborescence : `data/` (gitignore), `src/`, `results/`, `figures/`,
  `paper/`, `tests/`. `requirements.txt` ou `pyproject.toml`
  (pandas, scikit-learn, matplotlib ; optionnel : shap, pyarrow, xgboost).
- `.gitignore` (data, caches, `*.csv` de sortie). Init git, premier commit.
- Reprendre les 3 scripts fournis dans `src/` et vérifier qu'ils tournent.

**T2. Courbe précocité vs performance**
- Boucler `01_build_features.py` sur `cutoff ∈ {0.10, 0.20, 0.25, 0.33, 0.50}`.
- Pour chaque cutoff, lancer (A) et (B), stocker les métriques dans
  `results/early_curve.csv`.
- Figure : AUC et rappel en fonction du cutoff, une courbe par protocole. C'est
  un résultat central (« combien de performance perd-on en prédisant plus tôt »).

**T3. Analyser l'hétérogénéité du transfert (LOMO)**
- Tableau par module (déjà esquissé) + figure (barres AUC LOMO vs AUC aléatoire).
- Caractériser chaque module (nb d'évaluations, poids de l'examen final, part
  d'activité VLE, taille, taux de décrochage) et corréler avec la chute d'AUC.
- Hypothèse à tester : les modules dont la structure d'évaluation diffère le plus
  de la moyenne sont ceux qui se transfèrent le moins bien.

**T4. Interprétabilité comparée (SHAP)**
- Calculer l'importance SHAP globale **par module** (modèle LOMO).
- Question : les variables importantes **changent-elles** d'un module à l'autre ?
  Si oui, c'est une preuve directe de non-transférabilité des features — argument
  fort de l'article.

**T5. Modèle séquentiel (optionnel, si le temps le permet)**
- Variables hebdomadaires (clics par semaine jusqu'au cutoff) + un modèle
  temporel léger (GRU/LSTM, ou gradient boosting sur features séquentielles).
- Comparer au RF agrégé : le temporel améliore-t-il la prédiction **précoce** ?

**T6. Robustesse et équité**
- Variantes de la cible : `Withdrawn` seul vs `Withdrawn+Fail` (justifier le choix).
- Métriques par sous-groupe (genre, `imd_band`) dans le protocole LOMO, pour ne
  pas reproduire un biais (cf. audit d'équité cité).

**T7. Rédaction (dossier `paper/`)**
- Structure IMRaD, en LaTeX (viser un format IEEE, cohérent avec le CNRIA).
- Reproductibilité : chaque figure/tableau régénérable par un script (`make`
  ou un `run_all.py`). Fixer les graines aléatoires.

## 6. Conventions et garde-fous

- **Pas de fuite de données** : le découpage (A) est groupé par `id_student`
  (déjà fait via `StratifiedGroupKFold`) ; ne jamais utiliser `final_result` ni
  un score postérieur à la fenêtre comme prédicteur. Vérifier à chaque nouvelle
  feature qu'elle est bien calculée **avant** le cutoff.
- **Déterminisme** : `random_state` fixé partout ; journaliser les versions de
  librairies.
- **Python** : 3.11+. Le pipeline actuel bascule en CSV si `pyarrow` absent ;
  installer `pyarrow` pour des entrées/sorties plus rapides sur `studentVle`.
- **Performance** : `studentVle` fait 10,6 M lignes ; la construction des
  features prend ~10 s, l'entraînement LOMO ~3 min. Rien de lourd, mais éviter de
  recharger le VLE à chaque cutoff (le mettre en cache parquet).
- **Honnêteté méthodo** : le résultat vendeur n'est pas « forte AUC » mais
  « voici ce qui ne se transfère pas et pourquoi ». Ne pas masquer les modules où
  le modèle échoue (AAA, GGG) : ce sont eux l'intérêt.

## 7. Lien avec la thèse (à garder en tête, pas à coder tout de suite)

Le même protocole LOMO sera rejoué sur l'UN-CHK, où « module » devient
« formation » (MAI, SEG), puis « pays » (UK → Sénégal). Garder le code
**générique** : une fonction qui prend un jeu de features + une colonne de
« groupe de contexte », pour pouvoir changer OULAD → UN-CHK sans tout réécrire.

## 8. Première action suggérée pour Claude Code

Mettre en place T1 (dépôt + arborescence + requirements), reproduire les chiffres
de référence du §3, puis réaliser T2 (courbe précocité) et me montrer la figure.
