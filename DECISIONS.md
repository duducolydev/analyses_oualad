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
