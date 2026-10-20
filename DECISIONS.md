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

## D6 — Statistiques des corrélations (20/10/2026)

Corrélations de Spearman aux niveaux module (n = 7) et présentation (n = 22,
non indépendantes : même modèle LOMO par module), q-valeurs de
Benjamini-Hochberg par niveau. La dissimilarité estimée par classifieur
(contexte vs reste) vaut 0,94–1,00 pour tous les modules : non informative,
écartée de l'interprétation.
