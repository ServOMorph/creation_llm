Rôle : agent d'analyse critique et d'innovation pour la cellule R&D. Le rapport de recherche sur les méthodes d'entraînement frugales est fourni ci-dessous (bloc stdin).
Mission : voir comment s'en servir pour notre projet, avec l'objectif d'être innovant et de trouver des méthodes efficaces sur une RTX 4060 8 Go.
1. Vérifier la solidité du rapport et signaler les erreurs ou extrapolations.
2. Relier chaque méthode au projet : fichier et fonction à modifier (`model.py`, `train.py`, `run_369m.py`, `prepare_data.py`, `train_sft.py`, `prepare_sft.py`), surcoût VRAM et débit estimés pour le 369M, compatibilité avec le pré-entraînement en cours, applicabilité au SFT et au défaut « critère absent » de l'assistant.
3. Proposer au moins 3 combinaisons originales (hypothèses de la cellule, signalées comme telles) qui n'existent pas telles quelles dans la littérature.
4. Pour chaque piste retenue : expérience minimale (CPU pendant le run du 369M, ou GPU court via `gpu_task.py`), critère d'échec, gain attendu.
5. Classement final par gain attendu / coût, et programme proposé en 3 étapes pour le cycle R&D 2, en tenant compte de `rd/bilan_cycle1.md` et `rd/selection.md`.
