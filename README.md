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
- `roadmap_llm_fr.md`, `roadmap_assistant.md` : feuilles de route.

## État actuel
- v9 (~150M, contexte 1024) entraînée sur corpus français mélangé ; évaluation dans `eval_final_v9.md`.
- v8 reste le modèle par défaut pour les histoires ; v9 est la base de l'assistant.
- Assistant v1 : SFT de la v9 terminé (`checkpoints_sft/ckpt.pt`), préféré à la v9 de base à l'aveugle ; lancer `python run_assistant.py`.
- Limites : erreurs factuelles, multi-tours à vérifier. Prochaine étape : décider de la suite.
