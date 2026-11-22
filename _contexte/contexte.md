# Contexte — creation_llm

## Objectif (immuable sauf décision explicite)
Création d'un LLM français from scratch sur mon PC, dont le but final est un assistant conversationnel (décision du 2026-11-05).

## Stack / contraintes techniques (stable, rarement modifié)
- Machine : RTX 4060 8 Go, 48 Go de RAM, Ryzen 7 5700X, installation sur D:. VRAM partagée avec d'autres IA locales : accord explicite avant tout lancement GPU.
- Python (.venv), PyTorch 2.11 cu128, torch.compile, bf16 ; tokenizers HF (BPE 32k `tokenizer_fr32k.json`) ; lm-eval 0.4.13.
- Modèle GPT decoder-only maison (`model.py`) : RoPE, RMSNorm, QK-Norm, ReLU², soft-cap, cache KV ; optimiseur Muon + AdamW.

## État actuel (réécrit intégralement à chaque /close)
- Pré-entraînement 369M en pause au pas 54 349 sur 113 000 (`etat.json`, 2026-11-21 21:25) ; reprise sur ordre de l'utilisateur ; VRAM libre (941 Mio utilisés le 2026-11-22).
- Phases 3-4 prêtes : `compare_frenchbench.py`, `eval_frenchbench.py --ckpt`, `run_sft_369m.py` (SFT en deux étapes, pause VertIA).
- R&D cycle 2 : étape a (CPU) livrée ; relecture A1 terminée (106 cas étiquetés, `rd/experiences/relecture_a1/resultats_a1.json`), A2 non commencée ; étapes b et c à valider après la base.
- Codex : modèle imposé par la politique centrale, binaire de l'application ; plus d'action ouverte.
- Suivi : `python run.py` ; artefacts listés dans `_contexte/artifacts.md` ; `ameliorations.md` tient l'état des 14 points.

## Décisions structurantes (append only — 10 entrées max, 5 lignes max/entrée, archiver au-delà)
- 2026-10-22 : Initialisation du protocole vibecoding.
- 2026-11-05 : v9 non retenue pour les histoires (gate phase 9 non atteint), retenue comme base généraliste (FrenchBench nettement supérieur à v8).
- 2026-11-05 : Objectif final = assistant. SFT de la v9 sur `kurakurai/luth-sft` sans tulu3_math ni scholar, exemples ≤ 1024 tokens, loss sur les réponses seules, vocab agrandi à 32 832 pour `<|user|>`/`<|assistant|>`.
- 2026-11-08 : assistant v1 produit (SFT lr 3e-4, empaquetage sans attention croisée, loss sur réponses) ; gate phase 4 atteint.
- 2026-11-12 : assistant v2 retenu avec réserves (SFT 1 epoch depuis la v1, lr 1e-4) ; modèle plus grand = 369M from scratch avec recalcul des activations ; DPO non prioritaire.
- 2026-11-17 : mode orchestré VertIA (protocole v2) ; 369M à embeddings liés lancé ; GPU uniquement via `gpu_task.py` + garde `NoGpuToken` après un incident de protocole.
- 2026-11-17 : cellule R&D créée, cycle 1 clos sans gain (RAG automatique, CFG, TTT-NN) ; SFT 369M en deux étapes comme la v2 ; gate FrenchBench phase 4 conservé, à rediscuter.
- 2026-11-18 : recherches et veille R&D déléguées à Codex, modèle `gpt-reserve` uniquement (`gpt-6-astra` proscrit : abonnement) ; tableau de bord local par `run.py`, artefact mobile republié sur demande ; résultats rangés dans `resultats/`, journaux dans `logs/`.
- 2026-11-21 : programme du cycle 2 validé pour l'étape a seule (audit, données RAFT/H2 synthétiques, contrôles CPU) ; étapes b et c après la base ; `data_c2/` régénérable donc ignoré par git.
- 2026-11-22 : process Codex réglé sans modification : modèle `gpt-6.2-sol` imposé par `~/.codex/politique_delegation.json`, binaire = `codex.exe` de l'application choisi par `_codex/run_codex.ps1` ; relecture A1 faite par page dédiée (`rd/experiences/relecture_a1/`).
