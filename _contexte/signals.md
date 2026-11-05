# Signals — creation_llm   (MAJ 2026-11-05)

## Actions ouvertes
- [P1|ouvert] Phase 1 de la roadmap assistant : balises `<|user|>`/`<|assistant|>` dans le tokenizer 32k, conversion du checkpoint v9 (vocab 32 832), tests
  - fait quand: `tokenizer_fr32k_chat.json` et checkpoint converti produits, tests verts (ids existants inchangés, sorties identiques sur texte sans balise).
  - réf: `roadmap_assistant.md` (Phase 1), `checkpoints_v9/ckpt.pt`, `prepare_corpus.py`, `model.py`.
- [P2|ouvert] Décider si les logs `eval_frenchbench_v8.log` / `eval_frenchbench_v9.log` (barres de progression, ~650 Ko chacun) sont à commiter ou à ignorer
  - fait quand: fichiers ajoutés au commit ou à `.gitignore`.
  - réf: racine du projet.

## Contexte chaud
- VRAM libre, aucun entraînement en cours. Accord explicite requis avant tout lancement GPU.
- Jeu SFT `kurakurai/luth-sft` téléchargé dans `data/sft/luth-sft/` (549 Mo). Analyse : ~475k exemples / ~177M tokens utilisables après exclusion de tulu3_math, scholar et des exemples > 1024 tokens.
- Débit de référence v9 : ~14k tokens/s ; SFT 2 passes estimé ~7 h de GPU.

## Dernière session (2026-11-05)
# Session du 2026-11-05

## Décisions prises
- Fin de l'entraînement v9 (pas 91 500, val 2,5445).
- v8 reste le modèle par défaut pour les histoires ; v9 retenue comme base généraliste.
- Objectif du projet : un assistant conversationnel français.
- SFT sur `kurakurai/luth-sft` (odc-by), sans tulu3_math ni scholar ; vocab agrandi à 32 832 pour les balises de chat.

## Livrables produits ou modifiés
- `eval_v9.py`, `eval_frenchbench.py`, `prompts_holdout3.txt` : créés.
- `eval_final_v9.md` : évaluation complète v9 (perplexité, histoires à l'aveugle, FrenchBench).
- `roadmap_assistant.md` : créé (4 phases).
- `run_v9.py` : modifié (reprise, tokenizer 32k).

## Hypothèses validées / invalidées
- INVALIDE : v9 meilleure que v8 sur les histoires (48 % d'erreurs certaines contre 32 %) -> v9 réorientée vers l'assistant.
- VALIDE : v9 nettement meilleure en français général (grammaire 74 %, vocabulaire 69 %, compréhension 48 %).
- EN ATTENTE : gain du SFT sur la compréhension des consignes.

## Prochaine étape exacte
Démarrer la phase 1 de `roadmap_assistant.md` (tokenizer et conversion du checkpoint, CPU uniquement).

## Question bloquante pour la session suivante
Aucune
