# creation_llm

## Objectif
LLM français entraîné from scratch sur un PC personnel (RTX 4060 8 Go), avec pour but final un assistant conversationnel.

## Stack
Python, PyTorch (bf16, torch.compile), tokenizers HF (BPE 32k), Muon + AdamW, lm-eval (FrenchBench).

## Structure
- `model.py`, `train.py`, `generate.py` : modèle GPT, entraînement, génération.
- `prepare_corpus.py`, `prepare_tokens.py` : préparation du corpus et tokenisation.
- `run.py`, `run_v9.py`, `run_sft.py`, `run_assistant.py` : lanceurs.
- `train_sft.py`, `chat.py`, `convert_chat.py`, `prepare_sft.py` : SFT et conversation.
- `eval_v9.py`, `eval_assistant.py`, `eval_frenchbench.py`, `eval_final_*.md` : évaluations.
- `roadmap_llm_fr.md`, `roadmap_assistant_v2.md`, `roadmap_modele_369m.md` : feuilles de route (`_archives/` pour les terminées).

## État actuel
- v9 (~150M, contexte 1024) : base généraliste ; assistant v1 puis v2 par SFT.
- Assistant v2 retenu (`checkpoints_sft_v2/ckpt.pt`) : 49,7 % au benchmark `bench_assistant.py` contre 35,7 % pour la v1 ; lancer `python run_assistant.py`.
- Réserves : conversation -2/30, FrenchBench vocabulaire -9,2 points par rapport à la v9, multi-tours à 50 %. Détail dans `bilan_assistant_v2.md`.
- En cours : pré-entraînement du modèle 369M (`roadmap_modele_369m.md`, phase 2), en pause au pas 54 349 sur 113 000 (2026-11-21 21:25), reprise à décider ; phases 3-4 préparées (`compare_frenchbench.py`, `run_sft_369m.py`).
- Suivi : `python run.py` ouvre le tableau de bord local (avancement, roadmaps, actions, recherche, résultats, améliorations).
- GPU partagé sous jeton VertIA : toute tâche GPU passe par `python gpu_task.py --gb <Go> --minutes <min> -- <commande>`.
- R&D : `rd/` (cycle 1 clos, programme du cycle 2 dans `rd/programme_cycle2.md`, étape a livrée dans `rd/experiences/c2_*.py`, relecture A1 par la page de `rd/experiences/relecture_a1/`, recherches dans `rd/recherches/`, veille dans `rd/veille_actu/`).
- Résultats d'évaluation dans `resultats/`, journaux dans `logs/`.
