# Contexte — creation_llm

## Objectif (immuable sauf décision explicite)
Création d'un LLM français from scratch sur mon PC, dont le but final est un assistant conversationnel (décision du 2026-11-05).

## Stack / contraintes techniques (stable, rarement modifié)
- Machine : RTX 4060 8 Go, 48 Go de RAM, Ryzen 7 5700X, installation sur D:. VRAM partagée avec d'autres IA locales : accord explicite avant tout lancement GPU.
- Python (.venv), PyTorch 2.11 cu128, torch.compile, bf16 ; tokenizers HF (BPE 32k `tokenizer_fr32k.json`) ; lm-eval 0.4.13.
- Modèle GPT decoder-only maison (`model.py`) : RoPE, RMSNorm, QK-Norm, ReLU², soft-cap, cache KV ; optimiseur Muon + AdamW.

## État actuel (réécrit intégralement à chaque /close)
- Assistant v2 retenu : `checkpoints_sft_v2/ckpt.pt`, 49,7 % au benchmark (v1 : 35,7 %) ; réserves FrenchBench vocabulaire -9,2 points et conversation -2/30.
- `roadmap_assistant_v2.md` : 4 phases terminées, archivage en attente d'accord.
- Essai de taille : 369M tient avec recalcul des activations (environ 7 000 tok/s, 6,3 Go) ; pré-entraînement estimé à 12 jours.
- Roadmap en cours : `roadmap_modele_369m.md`, phase 1 (recalcul des activations, rodage) à lancer.

## Décisions structurantes (append only — 10 entrées max, 5 lignes max/entrée, archiver au-delà)
- 2026-10-22 : Initialisation du protocole vibecoding.
- 2026-11-05 : v9 non retenue pour les histoires (gate phase 9 non atteint), retenue comme base généraliste (FrenchBench nettement supérieur à v8).
- 2026-11-05 : Objectif final = assistant. SFT de la v9 sur `kurakurai/luth-sft` sans tulu3_math ni scholar, exemples ≤ 1024 tokens, loss sur les réponses seules, vocab agrandi à 32 832 pour `<|user|>`/`<|assistant|>`.
- 2026-11-08 : assistant v1 produit (SFT lr 3e-4, empaquetage sans attention croisée, loss sur réponses) ; gate phase 4 atteint.
- 2026-11-12 : assistant v2 retenu avec réserves (SFT 1 epoch depuis la v1, lr 1e-4) ; modèle plus grand = 369M from scratch avec recalcul des activations ; DPO non prioritaire.
