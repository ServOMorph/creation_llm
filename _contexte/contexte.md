# Contexte — creation_llm

## Objectif (immuable sauf décision explicite)
Création d'un LLM français from scratch sur mon PC, dont le but final est un assistant conversationnel (décision du 2026-11-05).

## Stack / contraintes techniques (stable, rarement modifié)
- Machine : RTX 4060 8 Go, 48 Go de RAM, Ryzen 7 5700X, installation sur D:. VRAM partagée avec d'autres IA locales : accord explicite avant tout lancement GPU.
- Python (.venv), PyTorch 2.11 cu128, torch.compile, bf16 ; tokenizers HF (BPE 32k `tokenizer_fr32k.json`) ; lm-eval 0.4.13.
- Modèle GPT decoder-only maison (`model.py`) : RoPE, RMSNorm, QK-Norm, ReLU², soft-cap, cache KV ; optimiseur Muon + AdamW.

## État actuel (réécrit intégralement à chaque /close)
- Pré-entraînement 369M (24 x 1024, embeddings liés) en cours sous jeton VertIA : pas 26 390 sur 113 000 au 2026-11-17, perte de validation 2,83 contre 2,93 pour la v9 au même pas.
- Phases 3-4 préparées sur CPU : `compare_frenchbench.py`, `run_sft_369m.py` (SFT en deux étapes, pause VertIA), conversion au chat compatible embeddings liés.
- GPU : seul lanceur `gpu_task.py` ; garde `require_gpu_token` dans le code.
- Assistant retenu : `checkpoints_sft_v2/ckpt.pt` (49,7 %). R&D cycle 1 clos (`rd/bilan_cycle1.md`), cycle 2 à décider.

## Décisions structurantes (append only — 10 entrées max, 5 lignes max/entrée, archiver au-delà)
- 2026-10-22 : Initialisation du protocole vibecoding.
- 2026-11-05 : v9 non retenue pour les histoires (gate phase 9 non atteint), retenue comme base généraliste (FrenchBench nettement supérieur à v8).
- 2026-11-05 : Objectif final = assistant. SFT de la v9 sur `kurakurai/luth-sft` sans tulu3_math ni scholar, exemples ≤ 1024 tokens, loss sur les réponses seules, vocab agrandi à 32 832 pour `<|user|>`/`<|assistant|>`.
- 2026-11-08 : assistant v1 produit (SFT lr 3e-4, empaquetage sans attention croisée, loss sur réponses) ; gate phase 4 atteint.
- 2026-11-12 : assistant v2 retenu avec réserves (SFT 1 epoch depuis la v1, lr 1e-4) ; modèle plus grand = 369M from scratch avec recalcul des activations ; DPO non prioritaire.
- 2026-11-17 : mode orchestré VertIA (protocole v2) ; 369M à embeddings liés lancé ; GPU uniquement via `gpu_task.py` + garde `NoGpuToken` après un incident de protocole.
- 2026-11-17 : cellule R&D créée, cycle 1 clos sans gain (RAG automatique, CFG, TTT-NN) ; SFT 369M en deux étapes comme la v2 ; gate FrenchBench phase 4 conservé, à rediscuter.
