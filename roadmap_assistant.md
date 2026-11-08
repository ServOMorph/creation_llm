# Roadmap — Assistant français (SFT de la v9)

Objectif : transformer la v9 (modèle de base ~150M, contexte 1024, tokenizer 32k) en assistant qui répond en français à des questions simples.
Données : `kurakurai/luth-sft` (odc-by), téléchargé dans `data/sft/luth-sft/`.
Contrainte permanente : aucun lancement sur la VRAM sans accord explicite de l'utilisateur.

## Phase 1 — Format de conversation et tokenizer [FAIT]
- Format fixe : `<|user|>` question `<|assistant|>` réponse `<|endoftext|>`, tours enchaînés pour le multi-tours.
- Ajout des balises au tokenizer 32k (`tokenizer_fr32k_chat.json`), sans modifier les ids existants.
- Conversion du checkpoint v9 : embeddings et tête agrandis à 32 832 (multiple de 64), nouvelles lignes initialisées à la moyenne des embeddings existants.
- Tests : aller-retour du tokenizer avec balises, ids existants inchangés, sorties du modèle converti identiques à la v9 sur du texte sans balise.
- Gate : tests verts.

**⏸ Checkpoint** — Demander à l'utilisateur de faire `/compact` avant de continuer.
Attendre sa réponse écrite. Ne pas commencer la phase suivante sans confirmation.

## Phase 2 — Préparation des données [FAIT]
- Sources gardées : openhermes, smoltalk2, tulu3_persona_instruct, croissantllm, aya. Exclues : tulu3_persona_math, scholar (trop longues, hors de portée du modèle).
- Filtrage : exemples de plus de 1024 tokens écartés, déduplication exacte des questions. Volume attendu : ~475k exemples, ~177M tokens.
- Masque de loss : uniquement sur les tokens de réponse (balise `<|endoftext|>` finale comprise).
- Validation : 2 000 exemples tirés au hasard, retirés de l'entraînement.
- Jeu d'évaluation scellé `prompts_assistant_holdout.txt` : 50 questions variées (connaissances, consignes, rédaction, reformulation, conversation), écrites avant tout entraînement.
- Tests : masque aligné sur les réponses, aucun exemple de validation dans l'entraînement, longueurs bornées.
- Gate : tests verts, statistiques du jeu final cohérentes avec l'analyse initiale.

**⏸ Checkpoint** — Demander à l'utilisateur de faire `/compact` avant de continuer.
Attendre sa réponse écrite. Ne pas commencer la phase suivante sans confirmation.

## Phase 3 — Entraînement SFT [FAIT]
- Adaptation de l'entraînement : chargement du checkpoint converti, loss masquée, batches de conversations empaquetées dans des fenêtres de 1024 sans attention croisée entre exemples (ou une conversation par fenêtre si plus simple).
- Choix du learning rate par A/B courts (même budget de tokens) sur la loss de validation des réponses.
- Entraînement complet : 2 passes (~7 h de GPU estimées à ~14k tokens/s), reprenable, pause VRAM possible à tout moment.
- Lanceur `run_sft.py`, checkpoints dans `checkpoints_sft/`.
- Tests : loss masquée correcte sur un cas construit, reprise d'entraînement identique, surapprentissage d'un mini-jeu.
- Gate : loss de validation des réponses en baisse puis plateau, pas de divergence.

**⏸ Checkpoint** — Demander à l'utilisateur de faire `/compact` avant de continuer.
Attendre sa réponse écrite. Ne pas commencer la phase suivante sans confirmation.

## Phase 4 — Évaluation et interface [FAIT]
- Évaluation à l'aveugle sur le jeu scellé : assistant contre v9 de base (même consigne en texte libre), 50 questions, 2 graines, notation A/B sur trois critères : compréhension de la question, exactitude, qualité du français.
- Contrôle de non-régression : FrenchBench 0-shot (`eval_frenchbench.py`) sur l'assistant.
- Lanceur `run_assistant.py` : conversation en ligne de commande, historique multi-tours tronqué au contexte.
- Gate : assistant meilleur que la v9 de base en compréhension et en format ; grammaire et vocabulaire FrenchBench sans baisse au-delà de la marge (~8 points).

**⏸ Checkpoint** — Demander à l'utilisateur de faire `/compact` avant de continuer.
Attendre sa réponse écrite. Ne pas commencer la phase suivante sans confirmation.

## Suite possible
Complément multi-tours en français, préférence (DPO), annealing sur données de haute qualité avant SFT, contexte plus long, modèle plus grand.
