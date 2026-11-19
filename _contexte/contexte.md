# Contexte — creation_llm

## Objectif (immuable sauf décision explicite)
Création d'un LLM français from scratch sur mon PC, dont le but final est un assistant conversationnel (décision du 2026-11-05).

## Stack / contraintes techniques (stable, rarement modifié)
- Machine : RTX 4060 8 Go, 48 Go de RAM, Ryzen 7 5700X, installation sur D:. VRAM partagée avec d'autres IA locales : accord explicite avant tout lancement GPU.
- Python (.venv), PyTorch 2.11 cu128, torch.compile, bf16 ; tokenizers HF (BPE 32k `tokenizer_fr32k.json`) ; lm-eval 0.4.13.
- Modèle GPT decoder-only maison (`model.py`) : RoPE, RMSNorm, QK-Norm, ReLU², soft-cap, cache KV ; optimiseur Muon + AdamW.

## État actuel (réécrit intégralement à chaque /close)
- Pré-entraînement 369M en cours sous jeton VertIA : pas 37 690 sur 113 000 au 2026-11-19 ; au pas 34 230, perte de validation 2,80 contre 2,88 pour la v9 ; environ 11 jours restants (estimation au rythme réel).
- Phases 3-4 prêtes : `compare_frenchbench.py`, `eval_frenchbench.py --ckpt`, `run_sft_369m.py` (SFT en deux étapes, pause VertIA).
- Suivi : tableau de bord `python run.py`, artefact mobile sur demande ; R&D : programme du cycle 2 (`rd/programme_cycle2.md`) à valider, recherches et veille sur Codex (`gpt-reserve`).
- Racine rangée (`resultats/`, `logs/`) ; `ameliorations.md` tient l'état des 14 points relevés.

## Décisions structurantes (append only — 10 entrées max, 5 lignes max/entrée, archiver au-delà)
- 2026-10-22 : Initialisation du protocole vibecoding.
- 2026-11-05 : v9 non retenue pour les histoires (gate phase 9 non atteint), retenue comme base généraliste (FrenchBench nettement supérieur à v8).
- 2026-11-05 : Objectif final = assistant. SFT de la v9 sur `kurakurai/luth-sft` sans tulu3_math ni scholar, exemples ≤ 1024 tokens, loss sur les réponses seules, vocab agrandi à 32 832 pour `<|user|>`/`<|assistant|>`.
- 2026-11-08 : assistant v1 produit (SFT lr 3e-4, empaquetage sans attention croisée, loss sur réponses) ; gate phase 4 atteint.
- 2026-11-12 : assistant v2 retenu avec réserves (SFT 1 epoch depuis la v1, lr 1e-4) ; modèle plus grand = 369M from scratch avec recalcul des activations ; DPO non prioritaire.
- 2026-11-17 : mode orchestré VertIA (protocole v2) ; 369M à embeddings liés lancé ; GPU uniquement via `gpu_task.py` + garde `NoGpuToken` après un incident de protocole.
- 2026-11-17 : cellule R&D créée, cycle 1 clos sans gain (RAG automatique, CFG, TTT-NN) ; SFT 369M en deux étapes comme la v2 ; gate FrenchBench phase 4 conservé, à rediscuter.
- 2026-11-18 : recherches et veille R&D déléguées à Codex, modèle `gpt-reserve` uniquement (`gpt-6-astra` proscrit : abonnement) ; tableau de bord local par `run.py`, artefact mobile republié sur demande ; résultats rangés dans `resultats/`, journaux dans `logs/`.
