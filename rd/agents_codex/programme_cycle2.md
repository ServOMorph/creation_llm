Rôle : direction de la cellule R&D. Lire `rd/analyse_lecun_jepa.md`, `rd/analyse_frugalite.md`, `rd/bilan_cycle1.md`, `rd/selection.md`, `roadmap_modele_369m.md` et vérifier dans le code (`model.py`, `train.py`, `train_sft.py`, `run_sft_369m.py`, `gpu_task.py`) chaque point d'intégration cité.
Mission : un programme unique pour le cycle R&D 2, qui fusionne les deux analyses sans doublon.
1. Liste fusionnée des pistes (STP, LLM-JEPA, CCE, audit des omissions, RAFT/H2, refroidissement + EMA, curriculum/H1, ResFormer, H3, etc.) : dédoublonner, signaler les contradictions entre les deux analyses et trancher en justifiant.
2. Programme en 3 étapes : (a) pendant le pré-entraînement 369M, CPU seulement, sans ralentir le run ; (b) après la base, autour du SFT en deux étapes déjà préparé (témoin = 369M + SFT standard) ; (c) choix d'une innovation pour un prochain pré-entraînement.
3. Pour chaque essai : fichier et fonction à créer ou modifier, coût CPU/GPU estimé (marqué estimation), commande `gpu_task.py` le cas échéant, critère de succès et d'échec chiffrés, dépendances.
4. Ordre d'exécution, et liste des décisions qui reviennent à l'utilisateur.
Format : titre, résumé de 8 lignes, tableau du programme, sections par étape. Maximum 250 lignes.
