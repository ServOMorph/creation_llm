# Changelog

## v0.4 — 2026-11-17

### Ajouté
- Modèle 369M : `run_369m.py`, recalcul des activations et embeddings liés (`model.py`), jeton GPU VertIA (`vram_guard.py`, pause et reprise) ; pré-entraînement en cours.
- `gpu_task.py` (seul lanceur GPU) et garde `require_gpu_token` (`NoGpuToken`) dans `generate.py`, `train.py`, `train_sft.py`.
- Phases 3-4 du 369M : `run_sft_369m.py` (SFT en deux étapes), `compare_frenchbench.py` (gates), recalcul, plafond VRAM et pause VertIA dans `train_sft.py`.
- Cellule R&D : `rd/` (veilles, pôle mathématique, sélection, expériences RAG/CFG/TTT-NN, `bilan_cycle1.md`), agents `.claude/agents/rd-*.md`.
- Tests : `test_gpu_task.py`, `test_compare_frenchbench.py`, conversion liée, pause SFT (134 verts).

### Modifié
- `.claude/CLAUDE.md` (mode VertIA, règle GPU, journal de session), `start.md`, `close.md`, `.gitignore`, `eval_frenchbench.py`.

## v0.3 — 2026-11-12

### Ajouté
- Benchmark assistant : `bench_assistant.py`, `bench_assistant.jsonl` (300 cas), `bench_assistant_results/`, `tests/test_bench_assistant.py`.
- Données ciblées : `make_transfo.py`, `tests/test_make_transfo.py`, mélange `data/sft/prepared_v2`.
- SFT v2 (`checkpoints_sft_v2/ckpt.pt`) : 49,7 % au benchmark contre 35,7 % pour la v1 ; recopie des transformations de 52 à 13.
- `bilan_assistant_v2.md` (v1 contre v2, essai de taille, besoin en données) ; `roadmap_assistant_v2.md` (4 phases terminées) ; `roadmap_modele_369m.md` (modèle 369M from scratch).

### Modifié
- `prepare_sft.py` (troncature, filtre des refus, sources JSONL), `chat.py`, `generate.py`, `eval_frenchbench.py` (modèle `sft_v2`).
- `roadmap_assistant.md` déplacée vers `_archives/`.

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
