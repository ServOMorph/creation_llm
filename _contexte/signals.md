# Signals — creation_llm   (MAJ 2026-11-12)

## Actions ouvertes
- [P1|ouvert] Phase 1 de `roadmap_modele_369m.md` : option `grad_checkpoint` (modèle, config, tests), `run_369m.py`, rodage d'environ 1 h
  - fait quand: tests verts, débit au moins 6 000 tok/s, VRAM au pic sous 6,5 Go, reprise sans saut de perte.
  - réf: `roadmap_modele_369m.md`, `bilan_assistant_v2.md` (section « Essai de taille »), `run_v9.py`, `train.py`, `model.py`.
- [P1|ouvert] Archiver `roadmap_assistant_v2.md` (4 phases FAIT) : accord explicite de l'utilisateur requis
  - fait quand: roadmap archivée dans `_archives/` ou maintenue sur décision de l'utilisateur.
  - réf: `roadmap_assistant_v2.md`.
- [P2|ouvert] Ajouter les checkpoints et logs au `.gitignore` : `checkpoints_sft/`, `checkpoints_sft_ab/`, `checkpoints_sft_v2/`, `checkpoints_sft_v2_ab/` (3,3 Go) et `*.out`, `*.err` ; décider de la suppression des dossiers A/B
  - fait quand: `.gitignore` mis à jour et `git status` sans checkpoints ni logs.
  - réf: `.gitignore`, racine du projet.
- [P2|ouvert] Multi-tours : ajouter des conversations multi-tours synthétiques (v2 à 50 %, pile à la limite du gate) ; DPO à réévaluer après le 369M (23 défauts corrigeables sur 151 échecs)
  - fait quand: décision écrite, après le benchmark du 369M (phase 5).
  - réf: `bilan_assistant_v2.md`, `roadmap_modele_369m.md` (phase 5).
- [P2|ouvert] Décider si les logs `eval_frenchbench_v8.log` / `eval_frenchbench_v9.log` (~650 Ko chacun) sont à commiter ou à ignorer
  - fait quand: fichiers ajoutés au commit ou à `.gitignore`.
  - réf: racine du projet.

## Contexte chaud
- VRAM libérée, aucun entraînement en cours ; Creation video prévenue deux fois (feu vert). Accord explicite requis avant tout lancement GPU ; après chaque usage GPU, libérer la VRAM et prévenir Creation video.
- Ollama est arrêté ; d'autres sessions (world model, creation_3d, Creation video) font des essais sur la VRAM : prévenir avant un lancement long.
- Assistant retenu : `checkpoints_sft_v2/ckpt.pt` (2 606 pas, 49,7 % au benchmark). Lancer : `python run_assistant.py` (pointe par défaut vers la v1 : ajouter `--ckpt checkpoints_sft_v2/ckpt.pt`).
- Essai de taille (données aléatoires) : 369M (24 x 1024) tient avec recalcul des activations, environ 7 000 tok/s et 6,3 Go ; 419M déborde ; durée estimée du pré-entraînement : environ 12 jours.

## Dernière session (2026-11-12)
# Session du 2026-11-12

## Décisions prises
- SFT v2 retenue (1 epoch depuis la v1) avec réserves : conversation -2/30, vocabulaire -9,2 points, multi-tours à 50 %.
- Modèle plus grand : 369M from scratch, avec recalcul des activations (décision de l'utilisateur).
- DPO non prioritaire : 23 défauts corrigeables sur 151 échecs ; réévaluation sur le 369M.

## Livrables produits ou modifiés
- `checkpoints_sft_v2/ckpt.pt`, `bench_assistant_results/v2*.json`, `eval_frenchbench_sft_v2_0shot.json` : résultats.
- `bilan_assistant_v2.md`, `roadmap_modele_369m.md`, `roadmap_assistant_v2.md` (4 phases FAIT) : créés ou mis à jour.
- `eval_frenchbench.py` (modèle `sft_v2`) : modifié.

## Hypothèses validées / invalidées
- VALIDE : la v2 divise la recopie des transformations par 4 (52 à 13) ; global 49,7 % contre 35,7 %.
- VALIDE : poursuivre depuis la v1 (48,3 %) vaut mieux que repartir de la v9 (39,7 %).
- INVALIDE : gate stricte de la phase 3 (conversation en baisse, vocabulaire au-delà de la marge) : acceptée avec réserves.
- EN ATTENTE : le défaut dominant `critere_absent` (139/151) relève d'un manque de capacité (hypothèse) ; à tester avec le 369M.

## Prochaine étape exacte
Phase 1 de `roadmap_modele_369m.md` : implémenter et tester `grad_checkpoint`, créer `run_369m.py`, puis rodage d'environ 1 h avec accord GPU.

## Question bloquante pour la session suivante
Aucune
