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

## Recherches à la demande et veille d'actualité (Codex)
Définitions dans `rd/agents_codex/` (contexte commun `commun.md`), exécutées par Codex (modèle unique dans `rd/agents_codex/modele.sh`, `gpt-reserve` par défaut, recherche web, lecture seule) pour économiser les tokens Claude.
1. **Sujet de l'utilisateur** : ajouté à `rd/sujets_a_traiter.md`, puis `sh rd/agents_codex/sujet.sh <slug> "<sujet>"` (arrière-plan) : agent de recherche -> `rd/recherches/<slug>/rapport.md`, agent d'analyse -> `analyse.md` (fichiers et fonctions du dépôt, coût sur la 4060, expérience minimale, verdict À INTÉGRER / À TESTER / À SURVEILLER / ÉCARTÉ). Ligne ajoutée à `rd/recherches/INDEX.md`.
2. **Veille d'actualité** : `sh rd/agents_codex/veille.sh` couvre la période depuis la dernière veille -> `rd/veille_actu/<date>.md`, indexé dans `rd/veille_actu/INDEX.md`.
3. **Analyse par la session principale** : les nouveautés « à approfondir » de la veille passent dans `rd/sujets_a_traiter.md` après accord de l'utilisateur ; les verdicts À INTÉGRER / À TESTER sont rangés dans `rd/programme_cycle2.md` ou la roadmap, sur décision de l'utilisateur.
Limite : quota Codex ; `sh rd/agents_codex/file.sh` traite toute la file et s'arrête au premier échec ; un rapport déjà produit n'est pas refait.
