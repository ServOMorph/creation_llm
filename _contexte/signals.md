# Signals — creation_llm   (MAJ 2026-11-08)

## Actions ouvertes
- [P1|ouvert] Décider de la suite après l'assistant v1 : complément multi-tours et réduction des boucles de répétition (voir « Suite possible » de la roadmap)
  - fait quand: décision écrite (nouvelle roadmap, ou arrêt du projet à l'assistant v1).
  - réf: `roadmap_assistant.md`, `eval_assistant_blind.md`, `eval_assistant_grades.txt`.
- [P2|ouvert] Archiver `roadmap_assistant.md` (toutes les phases sont FAIT) : accord explicite de l'utilisateur requis
  - fait quand: roadmap archivée ou maintenue sur décision de l'utilisateur.
  - réf: `roadmap_assistant.md`.
- [P2|ouvert] Ajouter `checkpoints_sft/`, `checkpoints_sft_ab/` et `*.out` au `.gitignore` (non couverts, plusieurs Go) ; décider de la suppression de `checkpoints_sft_ab/` (comparatif de learning rate)
  - fait quand: `.gitignore` mis à jour et `git status` sans checkpoints.
  - réf: `.gitignore`, racine du projet.
- [P2|ouvert] Multi-tours : au 2e tour, `run_assistant.py` répète la 1re réponse mot pour mot (essai sur 2 tours, graine 1). Cause non analysée.
  - fait quand: cause identifiée et comportement corrigé, ou limite acceptée.
  - réf: `chat.py`, `run_assistant.py`, `tests/test_chat.py`.
- [P2|ouvert] Décider si les logs `eval_frenchbench_v8.log` / `eval_frenchbench_v9.log` (~650 Ko chacun) sont à commiter ou à ignorer
  - fait quand: fichiers ajoutés au commit ou à `.gitignore`.
  - réf: racine du projet.

## Contexte chaud
- VRAM libre, aucun entraînement en cours. Accord explicite requis avant tout lancement GPU.
- Assistant : `checkpoints_sft/ckpt.pt` (pas 5 142, val réponses 1,1427). Lancer : `python run_assistant.py`.

## Dernière session (2026-11-08)
# Session du 2026-11-08

## Décisions prises
- Phases 3 et 4 de la roadmap assistant menées à terme (SFT 2 passes, évaluation, interface).
- Learning rate 3e-4 retenu après comparatif sur 300 pas (1e-4, 3e-4, 1e-3).
- Notation à l'aveugle par l'assistant IA (Claude), clé ouverte après enregistrement des notes.

## Livrables produits ou modifiés
- `train_sft.py`, `run_sft.py`, `tests/test_train_sft.py`, `model.py` (doc_ids) : créés ou modifiés.
- `chat.py`, `run_assistant.py`, `eval_assistant.py`, `tests/test_chat.py` : créés.
- `generate.py` (arrêt multi-balises), `eval_frenchbench.py` (modèle sft) : modifiés.
- `eval_assistant_*`, `eval_frenchbench_sft_0shot.json` : résultats.

## Hypothèses validées / invalidées
- VALIDE : le SFT apporte la compréhension des consignes (assistant préféré 73/100 en compréhension, 63 en exactitude, 81 en français ; v9 de base 2, 3 et 2).
- VALIDE : pas de régression FrenchBench au-delà de la marge (grammaire −3,4 pts, vocabulaire −6,7 pts ; lecture +8,4 pts).
- INVALIDE : l'assistant ne répond pas encore de façon fiable (erreurs factuelles fréquentes, transformations de phrases recopiées, boucles dans certains cas).
- EN ATTENTE : comportement multi-tours (répétition de la 1re réponse au 2e tour).

## Prochaine étape exacte
Décider avec l'utilisateur de la suite (multi-tours, DPO, modèle plus grand) et de l'archivage de la roadmap.

## Question bloquante pour la session suivante
Quelle suite pour le projet : compléments de l'assistant v1 ou arrêt ?
