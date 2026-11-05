# creation_llm

## Objectif
LLM français entraîné from scratch sur un PC personnel (RTX 4060 8 Go), avec pour but final un assistant conversationnel.

## Stack
Python, PyTorch (bf16, torch.compile), tokenizers HF (BPE 32k), Muon + AdamW, lm-eval (FrenchBench).

## Structure
- `model.py`, `train.py`, `generate.py` : modèle GPT, entraînement, génération.
- `prepare_corpus.py`, `prepare_tokens.py` : préparation du corpus et tokenisation.
- `run.py`, `run_v9.py` : lanceurs.
- `eval_v9.py`, `eval_frenchbench.py`, `eval_final_*.md` : évaluations.
- `roadmap_llm_fr.md`, `roadmap_assistant.md` : feuilles de route.

## État actuel
- v9 (~150M, contexte 1024) entraînée sur corpus français mélangé ; évaluation dans `eval_final_v9.md`.
- v8 reste le modèle par défaut pour les histoires ; v9 sert de base à l'assistant.
- Prochaine étape : SFT de la v9 (`roadmap_assistant.md`, phase 1).
