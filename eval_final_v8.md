# Évaluation finale du modèle d'histoires (v8)

## Protocole
- Jeu scellé `prompts_holdout2.txt` : 20 débuts d'histoires écrits avant l'entraînement de la v8, jamais utilisés pour une décision.
- 3 graines (1234, 2000, 3000), température 0,4, top-k 50, top-p 0,95, 150 tokens : 60 textes par modèle.
- Notation à l'aveugle : pour chaque prompt, les trois textes sont mélangés (A/B/C), tronqués à la même longueur, correspondance révélée après notation.
- Grille : erreur majeure (confusion de personnages, prénom qui change, contradiction, logique incohérente, sujet abandonné), cas limite (compté dans le strict), erreur mineure (non comptée).
- Critère de décision (choisi par l'utilisateur) : v8 retenue si au moins égale à v7 et meilleure que LeCarnet-21M.

## Résultats

| Modèle | Majeures | Limites | Mineures | OK | Erreurs certaines | Comptage strict |
|---|---|---|---|---|---|---|
| v8 (140M, 18 couches) | 10 | 6 | 12 | 32 | 17 % | 27 % |
| v7 (97,5M, 12 couches) | 13 | 3 | 12 | 32 | 22 % | 27 % |
| LeCarnet-21M (référence publiée) | 42 | 15 | 2 | 1 | 70 % | 95 % |

## Conclusion
- v8 retenue : au moins égale à v7 (moins d'erreurs certaines), très nettement meilleure que LeCarnet-21M.
- Défaut résiduel principal : dérive de prénoms et confusion entre personnages sur des situations peu présentes dans le corpus (2M histoires stéréotypées).
- Suite : modèle généraliste entraîné sur un corpus français varié (roadmap, phases 7 à 9).

## Limites
- Un seul évaluateur (Claude), grille fixe mais jugement subjectif.
- 60 textes par modèle : marge d'environ ±6 points sur les pourcentages.
