# Changelog

## v0.2 — 2026-11-08

### Ajouté
- SFT : `train_sft.py`, `run_sft.py` (empaquetage, attention par document, loss masquée, reprise), `tests/test_train_sft.py`.
- Assistant : `chat.py`, `run_assistant.py`, `eval_assistant.py`, `tests/test_chat.py` ; évaluation à l'aveugle et FrenchBench du modèle SFT.

### Modifié
- `model.py` : `doc_ids` facultatif ; `generate.py` : arrêt sur plusieurs balises ; `eval_frenchbench.py` : modèle `sft`.

## v0.1 — 2026-11-05

### Ajouté
- Évaluation complète de la v9 : `eval_v9.py`, `eval_frenchbench.py`, `prompts_holdout3.txt`, `eval_final_v9.md`.
- `roadmap_assistant.md` : SFT de la v9 vers un assistant français.

### Modifié
- `run_v9.py` : reprise et tokenizer 32k.
- Objectif du projet : assistant conversationnel ; v8 conservée pour les histoires.
