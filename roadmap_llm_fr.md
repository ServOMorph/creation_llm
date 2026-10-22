# Roadmap — LLM français from scratch

Matériel : RTX 4060 8 Go, 48 Go RAM, Ryzen 7 5700X. Installation sur D:.
Principe : v1 rapide et minimale, puis amélioration itérative (taille, données, qualité).
Exécution en autonomie : accord VRAM donné par l'utilisateur, enchaînement des phases sans checkpoint `/compact`.

## Phase 1 — Environnement et données [EN COURS]
- venv sur D:, PyTorch CUDA, tokenizers, datasets
- Corpus FR v1 : LeCarnet (histoires courtes FR, style TinyStories), fiche à vérifier avant usage
- Gate : `torch.cuda.is_available()` vrai, corpus nettoyé sur disque

## Phase 2 — Tokenizer [TODO]
- BPE entraîné sur le corpus, vocab ~8-16k
- Tests : aller-retour encode/decode, taux de compression
- Gate : tests passent

## Phase 3 — Modèle GPT (decoder-only, ~30-50M) [TODO]
- Implémentation from scratch : attention causale, RoPE, RMSNorm, QK-Norm, MLP ReLU² (briques modded-nanogpt)
- Tests : formes, masque causal, overfit sur un mini-batch
- Gate : tests passent, loss tend vers 0 sur un batch

## Phase 4 — Entraînement v1 [TODO]
- Boucle bf16, AdamW, cosine LR, checkpoints, reprise
- Entraînement quelques heures, suivi de la loss de validation
- Gate : loss de validation décroissante, checkpoint sauvegardé

## Phase 5 — Génération et évaluation v1 [TODO]
- `run.py` de génération (température, top-k/top-p)
- Évaluation : perplexité + benchmark de prompts verrouillés
- Gate : v1 génère du français lisible (sans garantie de cohérence)

## Phase 6+ — Améliorations itératives [TODO]
Pistes, à prioriser après v1 : corpus élargi (Wikipédia FR, FineWeb-2 FR), optimiseur Muon, modèle plus grand (~120M), SFT / forme assistant (v1 = complétion de texte, pas de chat), contexte plus long.
