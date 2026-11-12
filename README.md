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
- Prochaine étape : modèle 369M from scratch (`roadmap_modele_369m.md`), phase 1 (recalcul des activations et rodage).
