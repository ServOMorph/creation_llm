# Cellule R&D — creation_llm

## Mission
Trouver des méthodes qui rendent un LLM français entraîné et exécuté sur une seule RTX 4060 8 Go nettement meilleur à matériel constant : méthodes publiées, méthodes encore en recherche, et idées nouvelles.

## Règle de preuve
Une idée ne vaut que par une expérience falsifiable sur ce matériel. Chaque proposition indique : mécanisme, gain attendu chiffré, coût (VRAM, temps), expérience minimale, critère d'échec, sources avec URL. Distinguer : résultat établi, prépublication, hypothèse de la cellule.

## Équipe (définitions dans `.claude/agents/`)
| Agent | Pôle | Livrable |
|---|---|---|
| `rd-veille-entrainement` | Veille web : entraînement efficace | `rd/veille_entrainement.md` |
| `rd-veille-inference` | Veille web : inférence et compression | `rd/veille_inference.md` |
| `rd-mathematiques` | Mathématiques : théorie, bornes, idées nouvelles | `rd/pole_mathematiques.md` |
| `rd-critique` | Contre-expertise et sélection | `rd/selection.md` |
| session principale | Direction : synthèse, programme, expériences | `rd/programme.md` |

## Cycle
1. Veille et mathématiques en parallèle.
2. Contre-expertise : tri, doublons, erreurs, classement par gain attendu / coût.
3. Programme d'expériences décidé avec l'utilisateur.
4. Expériences, résultats dans `rd/experiences/`, puis nouveau cycle.

## Contraintes
- GPU : aucun usage sans jeton VertIA ; le pré-entraînement du 369M est prioritaire.
- Les agents n'écrivent que dans `rd/` et ne modifient aucun fichier existant.
- Aucune information personnelle dans les requêtes web.
