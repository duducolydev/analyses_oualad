# Décisions méthodologiques

Journal des choix qui s'écartent du brief initial, avec leur justification
chiffrée. Base de la section Méthodes de l'article.

## D1 — Population « active » au cutoff (19/10/2026)

Les étudiants déjà désinscrits au cutoff (ou pas encore inscrits) sont exclus.
À 25 % de la session, ils représentent 19,5 % des inscriptions et 62,5 % des
Withdrawn : leur « prédiction » est triviale (`last_activity`) et ne relève plus
de la prévention. Leur part varie de 5,4 % (GGG) à 28,0 % (CCC), ce qui
confondait l'hétérogénéité du transfert. L'ancien protocole reste disponible
(`--population all`) comme témoin.

## D2 — Notes précoces filtrées sur la date de remise (19/10/2026)

Une note n'est utilisée que si l'échéance ET la remise sont antérieures au
cutoff (4 % des remises retenues initialement étaient postérieures). Vérifié par
`tests/test_no_leakage.py` (invariance à la suppression des données futures).

## D3 — Même agrégation pour (A) et (B) (19/10/2026)

L'écart publié (A − B) comparait une AUC poolée (A, tous modules mélangés) à une
moyenne d'AUC intra-module (B). L'AUC poolée profite des différences de taux de
risque entre modules. On compare désormais les deux protocoles par la moyenne
des métriques intra-module, (A) étant évalué sur ses prédictions hors-pli.
À 25 % (population active) : écart 0,027 en poolé, 0,013 à agrégation égale.

## D4 — Recadrage de l'angle (validé le 19/10/2026)

Le classement du risque se transfère presque entièrement à un module non vu ;
le seuil de décision (calibration) ne se transfère pas. AAA et GGG sont des
modules intrinsèquement difficiles (AUC aléatoire ≈ 0,73), pas des échecs de
transfert du classement ; GGG voit son rappel passer de 0,66 à 0,99 et sa
précision de 0,55 à 0,38 en LOMO. T3 sépare donc difficulté intrinsèque, coût
de transfert du classement et dérive de calibration, et teste la recalibration
sur un petit échantillon du contexte cible.

## D5 — Mécanisme de la dérive et variables relatives (20/10/2026)

La dérive de GGG (+0,51 de part signalée) persiste à 33 % et 50 % : elle n'est
pas due seulement à l'absence de note précoce. L'étudiant médian de GGG a un
engagement VLE au 10e–16e centile des autres modules ; ailleurs ce niveau
signifie ~78 % de risque, dans GGG ~50 %. Les seuils absolus d'engagement ne se
transfèrent pas. Test : variante où engagement et notes sont remplacés par leur
rang centile dans la présentation (`--variables relatives`, sans étiquette,
calculable au cutoff, couverte par le test anti-fuite). À 25 % : |dérive|
moyenne 0,137 → 0,021 (GGG +0,51 → −0,01), AUC LOMO 0,787 → 0,809.
Précision apportée par T4 (SHAP signé, D8) : à 25 %, le modèle transféré
remonte le risque prédit de GGG de +0,30, dont +0,19 par les notes précoces
(moitié des étudiants sans note, devoirs formatifs notés plus bas) et +0,13 par
l'engagement VLE. Les deux relèvent du même problème d'échelle absolue.

## D6 — Statistiques des corrélations (20/10/2026)

Corrélations de Spearman aux niveaux module (n = 7) et présentation (n = 22,
non indépendantes : même modèle LOMO par module), q-valeurs de
Benjamini-Hochberg par niveau. La dissimilarité estimée par classifieur
(contexte vs reste) vaut 0,94–1,00 pour tous les modules : non informative,
écartée de l'interprétation.

## D7 — Variables relatives = modèle principal (validé le 20/10/2026)

Pour la suite (T2 en figure, T4–T6), le modèle principal utilise les variables
relatives ; le modèle à variables absolues (celui de la littérature) est gardé
comme référence. Il est meilleur sur tous les axes en LOMO (AUC, précision, F1,
dérive) et transforme le constat (« ce qui ne se transfère pas ») en remède
utilisable sans étiquette sur une nouvelle formation UN-CHK.

## D8 — T4 : mêmes variables, mauvais seuils (20/10/2026)

Hypothèse du brief : « si les variables importantes changent d'un module à
l'autre, c'est une preuve de non-transférabilité ». Résultat : les importances
SHAP locales sont assez stables entre modules (Spearman moyen 0,81 entre paires,
`mean_early_score` en tête partout). Le module qui dérive le plus (GGG) a la
MEILLEURE concordance transféré/local en variables absolues (0,79) : sa dérive
ne vient pas de variables différentes mais des mêmes variables lues avec des
seuils absolus inadaptés. L'importance globale (|SHAP| moyen) ne le voit pas ;
le SHAP signé moyen du modèle transféré, si (décomposition de la dérive,
corrélé à la dérive de signalement : ρ = 0,68 sur 7 modules). Les variables
relatives augmentent la concordance dans 6 modules sur 7 (moyenne 0,60 → 0,82 ;
exception : GGG 0,79 → 0,70) et annulent la décomposition (GGG +0,30 → −0,04).

Méthode : SHAP exact (TreeSHAP) estimé sur 100 arbres tirés parmi 300 (moyenne
de forêt ⇒ estimateur sans biais ; ρ = 0,985 avec l'exact), 400 étudiants par
module ; modèle local en validation croisée à 2 plis groupés par étudiant.

## D9 — Cible : Withdrawn + Fail, Withdrawn seul en robustesse (20/10/2026)

Cible principale conservée : `at_risk` = Withdrawn ou Fail. Raisons :
(1) dans un dispositif d'accompagnement, les deux issues appellent la même
intervention précoce ; (2) en population active, « Withdrawn seul » devient
« abandon futur », cible rare (13,7 % à 25 %) et bien plus difficile (AUC
intra-module ≈ 0,65–0,70), où les métriques au seuil 0,5 dégénèrent (rappel
0,10–0,13) ; (3) comparabilité avec la littérature OULAD. Robustesse (cutoff
25 %) : avec Withdrawn seul, les constats tiennent — coût de transfert de l'AUC
0,021 (absolues) → 0,006 (relatives), |dérive| 0,052 → 0,028 (GGG +0,18 →
+0,03), AUC LOMO 0,630 → 0,668. Pour cette cible, comparer sur l'AUC, pas au
seuil.

## D10 — Équité : la dérive de contexte devient un biais de groupe (20/10/2026)

Métriques au seuil par genre, quintile d'IMD et handicap ; IC 95 % par
bootstrap groupé par étudiant (1000 tirages, apparié entre configurations).
Genre : en variables absolues, le transfert fait passer l'écart de FPR
femmes − hommes de −0,007 [−0,019 ; 0,005] (A) à +0,188 [0,174 ; 0,201] (B).
À l'intérieur de chaque module, l'écart est ≤ 0 ou proche de 0 : l'écart global
est un effet de composition (femmes = 88 % de BBB et 81 % de GGG, les deux
modules qui dérivent vers le haut). Variables relatives : +0,016 [0,004 ; 0,027].
Conséquence méthodologique : auditer l'équité aussi à l'intérieur des contextes,
sinon une dérive de calibration se lit comme une discrimination (et inversement).
IMD Q1 − Q5 et handicap : écarts présents dans toutes les configurations
(risque de base plus élevé en Q1 : 51 % contre 33 %), FPR Q1 − Q5 de +0,11 à
+0,16 ; les variables relatives le réduisent en LOMO (−0,05 [−0,075 ; −0,026]).

## D11 — T5 : information temporelle et autres modèles (21/10/2026)

Séquences : K semaines complètes observées dans TOUTES les présentations au
cutoff (3, 6, 8, 11, 16), clics et jours actifs par semaine, clics avant le
début, 4 tendances ; rang dans la présentation en variante relative ; couvertes
par le test anti-fuite. Modèles : gradient boosting scikit-learn (agrégées ;
agrégées + séquences), GRU (h = 32, perte pondérée équivalente à « balanced »,
arrêt précoce sur 10 %). Résultats (population active, variables relatives,
protocole B) : la séquence ajoute au plus +0,004 d'AUC au boosting ; le GRU
dépasse la forêt de +0,004 à +0,009 selon le cutoff. Le signal précoce est
dans les comptes agrégés. La dérive du seuil en variables absolues est la même
pour les 4 modèles (0,118–0,137 à 25 % ; GGG +0,47 à +0,53) : le mécanisme est
une propriété des variables. Les variables relatives la réduisent partout (GRU
0,031, coût de classement 0,002 ; boosting 0,055, dérive résiduelle dispersée
et de signes alternés). La forêt reste le modèle principal (plus simple,
dérive la plus faible en relatif, SHAP déjà calculé) ; T5 entre dans l'article
comme vérification de généralité.

## D12 — Révision : étudiants inscrits dans plusieurs modules (21/10/2026)

Revue 2, point 1. 6,8 % des étudiants de la population active sont inscrits
dans plusieurs modules ; en LOMO, la part des étudiants du module testé aussi
présents en entraînement va de 0,3 % (AAA) à 48,7 % (CCC) et 30,3 % (EEE).
Le protocole (A) est groupé par étudiant, le LOMO standard ne l'était pas.
Test (`src/10_overlap_check.py`, option `exclure_communs` de
`eval_contexte_exclu`) : LOMO refait en retirant ces étudiants de
l'entraînement, 5 cutoffs × 2 variantes. AUC LOMO moyenne inchangée à ±0,002,
coût de transfert à ±0,002, dérive moyenne à ±0,007.

CORRIGÉ après l'étape 2 (D13) : j'avais d'abord attribué les écarts par module
au bruit de la forêt ; c'est faux pour CCC et EEE. L'écart-type dû à la graine
de l'AUC (B) y est de 0,001, et un contrôle (`src/13_overlap_control.py`,
5 graines) montre que retirer les étudiants communs fait baisser l'AUC (B) de
CCC de −0,007 ± 0,001 et d'EEE de −0,006 ± 0,002 en variables absolues, alors
que retirer le même nombre d'inscriptions au hasard ne change rien (+0,001,
−0,001). La forêt reconnaît des étudiants déjà vus : le LOMO standard est
légèrement optimiste pour les modules à fort chevauchement. Effet ≤ 0,002 en
variables relatives (de l'ordre du bruit), ≤ 0,002 sur les moyennes, et nul sur
la dérive. Conclusions inchangées ; l'écart de CCC entre variables relatives et
absolues tombe de −0,012 à −0,006 avec le protocole sans étudiants communs.
Décision sur le protocole principal : voir D13.

## D13 — Révision étape 2 : incertitude, calibration, seuil, top-q (21/10/2026)

Cutoff 25 %, population active ; `src/11_seed_variability.py`,
`src/12_robustness.py`, `src/13_overlap_control.py`.

1. **IC 95 %** (bootstrap groupé par étudiant, 1000 tirages, apparié) :
   relatives − absolues : AUC (B) +0,021 [0,015 ; 0,027], coût −0,005
   [−0,009 ; −0,001], |dérive| −0,117 [−0,123 ; −0,110]. Coût absolues 0,013
   [0,009 ; 0,017], relatives 0,008 [0,005 ; 0,010]. Ces IC ne couvrent que
   l'échantillonnage des étudiants, pas le choix des 7 modules.
2. **Niveau module (n = 7, Wilcoxon apparié)** : la dérive baisse dans 7 modules
   sur 7 (p = 0,016, minimum possible) ; l'AUC (B) monte dans 5 sur 7
   (p = 0,11, non significatif) : le gain d'AUC vient surtout d'AAA (+0,065) et
   BBB (+0,049) ; CCC baisse (−0,015 ± 0,002 sur 10 graines ; −0,006 sans
   étudiants communs), EEE à peine (−0,004 ± 0,002). À écrire tel quel.
3. **Graines (10)** : écart-type de l'AUC (B) par module 0,001–0,004, de la
   dérive 0,003–0,008 : les écarts discutés dépassent ce bruit.
4. **Calibration** ((B) − (A), moyenne des |écarts| sur les modules ;
   absolues → relatives) : Brier 0,039 → 0,006 (GGG +0,183 → +0,005),
   calibration-in-the-large 0,43 → 0,07 (GGG −1,87 → −0,06), score moyen
   0,077 → 0,013 (GGG +0,30 → +0,01), pente 0,18 → 0,08 (GGG −0,33 → −0,13).
   C'est donc aussi un défaut de calibration au sens strict, pas seulement de
   seuil. Les probabilités ne sont pas calibrées même en (A) (poids équilibrés) :
   on compare (B) à (A).
5. **Seuil** : |dérive| moyenne en absolues 0,088 / 0,111 / 0,137 / 0,139 /
   0,115 aux seuils 0,3–0,7 (GGG 0,16 à 0,60) ; en relatives 0,015–0,031.
   Le constat ne dépend pas du seuil.
6. **Top-q par cohorte** (q = taux à risque des 6 modules d'entraînement, sans
   étiquette cible) : |signalés − risque| 0,071, précision 0,634, F1 0,646 ;
   variables relatives au seuil 0,5 : 0,059, 0,684, 0,661 ; top-q oracle (vrai
   risque du module) : 0,001, 0,652, 0,651. Les variables relatives font mieux
   que top-q, et même que l'oracle en précision et F1. Exception : pour GGG,
   top-q a un meilleur F1 (0,598 contre 0,571) car le modèle relatif y
   sous-signale (30 % pour un risque de 37 %, rappel 0,52).

**À décider** : passer le protocole (B) principal au LOMO sans étudiants
communs (cohérent avec (A), groupé par étudiant), ce qui demande de relancer
toute la chaîne (~4 h) ; ou garder le LOMO standard et rapporter D12 en
robustesse. → Tranché en D14.

## D14 — LOMO strict, revue cible JLA, étape 2 bis (décidé le 29/10/2026)

**Protocole (B) = LOMO strict** : un étudiant inscrit dans le module testé
n'est jamais en entraînement (`plis_contexte(..., exclure_communs="id_student")`,
testé par `tests/test_protocols.py`). Appliqué dans 02, 03, 05, 09, 11. Le LOMO
standard devient la comparaison de robustesse (`10_overlap_check.py`).

**Revue cible : Journal of Learning Analytics** (gabarit et règles JLA à
appliquer à l'étape 4).

**Étape 2 bis (revue 3)** :
1. Remèdes comparés à armes égales (`14_remedies.py`) : toutes les règles (brut,
   Platt, top-q, variables relatives, et top-q oracle en référence) évaluées sur
   les MÊMES étudiants (non tirés pour Platt ; ou présentations postérieures pour
   le plan « pilote ») contre la MÊME référence, le modèle (A) à variables
   absolues ; plus |signalés − risque|, sans référence de modèle ; écart-type de
   Platt entre tirages. La recalibration quitte `04` ; la Fig. 2c et le tableau
   des remèdes lisent cette sortie. Sur les prédictions du LOMO standard (avant
   relance) : |dérive| brut 0,137, Platt-100 0,058 (±0,038 de part signalée
   entre tirages), top-q 0,059, relatives 0,043 ; précision 0,618 / 0,624 /
   0,633 / 0,684. Contre cette référence commune, les variables relatives
   SOUS-signalent GGG (−0,15) ; mais (A) y sur-signale lui-même (44 % pour un
   risque de 37 %) : d'où |signalés − risque| (relatives 0,059 = (A) 0,059).
2. Équité à l'intérieur de chaque module (`07_fairness.py`) : écarts et IC
   (mêmes tirages), effectifs par groupe (total, positifs, négatifs). Genre,
   variables absolues : aucun écart de FPR significativement en défaveur des
   femmes dans un module (3 en leur faveur) ; variables relatives : CCC +0,052
   [+0,006 ; +0,100] en défaveur des femmes. IMD : FPR plus élevé pour Q1 dans
   6 modules sur 7 — écart présent DANS les modules, pas un effet de composition.
   Justification du bootstrap groupé par étudiant écrite dans le script.

Après relance complète en LOMO strict (29/10–06/11/2026) : moyennes inchangées
à ±0,002 (AUC (B) 0,787 / 0,808 ; |dérive| 0,135 / 0,021) ; CCC relatives −
absolues −0,008 ± 0,002 (au lieu de −0,015) ; écart FPR F−H dans CCC en
relatives +0,035 [−0,014 ; +0,081], non significatif ; en absolues, écart
significativement en faveur des femmes dans 5 modules, contre elles dans aucun ;
dérive améliorée dans 6 modules sur 7 (Wilcoxon p = 0,047).

## D15 — Étape 3 : normalisations alternatives et taille de cohorte (06/11/2026)

`src/15_normalisations.py` (cutoffs 10/25/50 %, LOMO strict) : toute
normalisation par cohorte capte l'essentiel du gain. À 25 % — AUC (B) : absolues
0,787, rang 0,808, z-score 0,801, robuste 0,800 ; |dérive| (même variante) :
0,135 / 0,021 / 0,037 / 0,024 ; |signalés − risque| : 0,174 / 0,057 / 0,061 /
0,054 ; précision (B) : 0,618 / 0,680 / 0,666 / 0,664. Les rangs sont les
meilleurs ou à égalité à chaque cutoff, sans paramètre et insensibles aux
valeurs sentinelles ; les écarts entre normalisations (≤ 0,008 d'AUC) sont
faibles. Conséquence pour l'article : la contribution est la normalisation PAR
COHORTE ; le rang en est la forme recommandée, pas une condition nécessaire.

`src/16_cohort_size.py` (rangs recalculés dans des sous-cohortes aléatoires du
module testé, 5 découpages) : AUC (B) 0,778 / 0,793 / 0,802 / 0,805 / 0,806 /
0,808 pour n = 10 / 20 / 50 / 100 / 200 / présentation entière ; écart de part
signalée au cas complet 0,043 / 0,018 / 0,005 / 0,004 / 0,003. Les rangs sont
fiables à partir d'environ 50 étudiants ; en dessous de 20, le gain se dégrade
nettement (n = 10 : AUC sous celle des variables absolues, 0,787).

## D16 — Étape 4 : contenu de l'article mis à jour (06/11/2026)

Format IEEE conservé (le passage au gabarit JLA est reporté, décision de
Doudou). `paper/main.tex` réécrit pour intégrer D12–D15 et les trois
relectures : titre « Early At-Risk Prediction in Unseen Courses » (cible
composite), résumé et introduction distinguant trois niveaux d'affirmation
(gains établis sur 7 modules ; interprétation étayée ; transfert inter-
institutions non testé) ; protocole LOMO strict ; sections nouvelles
« calibration, pas seulement seuil » (tableau de calibration, seuils 0,3–0,7),
« remèdes sur les mêmes étudiants », « quelle normalisation, quelle taille de
cohorte » (tableau), « incertitude et hétérogénéité » (IC, Wilcoxon, graines,
CCC, protocole strict vs standard) ; équité réécrite (« atténue », intra-module,
IMD présent dans les modules) ; recommandation sur l'usage des alertes ;
positionnement vis-à-vis d'AdaBN et Tent ; URL du dépôt. Aucun chiffre de
résultat en dur : 257 macros générées par `src/08_paper_assets.py`.

## D17 — Analyses demandées par les relectures R1 et R2 (06–07/11/2026)

Décision de Doudou : soumettre avec un cadrage méthodologique assumé (sans
attendre les données UN-CHK). Résultats (cutoff 25 %, LOMO strict) :
- Baselines (14_remedies.py, plan k = 100) : correction de prévalence EM
  (Saerens et al., 2002), sans étiquette, PIRE que le modèle brut (|dérive|
  0,187, |signalés − risque| 0,224 ; GGG estimé à 99,9 % à risque) : la dérive
  n'est pas un changement de prévalence (pour un modèle pondéré équilibré au
  seuil 0,5, la règle compare des vraisemblances, invariantes sous pur changement
  de prévalence). Intercept seul (100 étiquetés) : 0,051 / 0,058, précision
  0,627, F1 0,647 ≈ Platt, et PAS plus stable (écart-type de la part signalée
  0,040 contre 0,039). Relatives : 0,043 / 0,057, précision 0,681, F1 0,659.
- Régression logistique (15_normalisations.py --model logit) : rangs → AUC (B)
  0,775 → 0,805, |dérive| 0,061 → 0,018 ; effet lié aux variables, pas aux arbres.
- Forêt NON pondérée calibrée (--model rf_cal) : absolues |dérive| 0,150,
  |p moyen − risque| 0,139 ; rangs 0,034 / 0,052, AUC (B) 0,786 → 0,810. Le
  choix pondéré + seuil 0,5 n'est pas la cause.
- Average Precision (17) : forêt (B) 0,720 → 0,755 ; Withdrawn seul 0,212 → 0,230
  (prévalence 0,137). Diagrammes de fiabilité : GGG absolu en (B) très sous la
  diagonale, relatif proche de (A).
- Taille d'entraînement (19) : à taille égale à (B), coût 0,014 / 0,008,
  identique à l'analyse principale (entraînement de (B) = 0,93–1,22 × (A)).
- Équité (18) : modèle linéaire avec effets fixes module, écart FPR F − H en
  (B) absolues +0,194 → −0,048 [−0,062 ; −0,033] ; relatives +0,015 → −0,022.
  Intersections : en relatives, l'écart restant suit l'IMD, pas le genre.
- Taille de cohorte, 20 découpages : n = 50 → AUC 0,802 [0,800 ; 0,804].
- CCC : la perte avec les rangs persiste dans chaque présentation (−0,014) — pas
  un effet entre présentations ; CCC a déjà la plus faible dérive absolue.
  BBB : coût de classement réel dans les présentations (0,033), ramené de 0,043
  à 0,011 par les rangs (effet d'échelle sur l'ordre lui-même).
