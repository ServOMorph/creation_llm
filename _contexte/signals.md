# Signals — creation_llm   (MAJ 2026-11-17)

## Actions ouvertes
- [P1|ouvert] Phase 2 de `roadmap_modele_369m.md` : pré-entraînement 369M jusqu'au pas 113 000
  - fait quand: `checkpoints_369m/etat.json` à « termine » et perte de validation finale < 2,5445 (v9).
  - réf: `run_369m.py`, `checkpoints_369m/log.csv`, `checkpoints_v9/log.csv`.
- [P1|ouvert] Phases 3-4 à la fin du pré-entraînement : FrenchBench + échantillons, conversion au chat, SFT en deux étapes, benchmark assistant
  - fait quand: `compare_frenchbench.py v9 369m` exécuté, puis `checkpoints_sft_369m/ckpt.pt` benchmarké.
  - réf: `compare_frenchbench.py` (commandes en tête), `run_sft_369m.py` (commandes en tête), `convert_chat.py`, `bench_assistant.py`.
- [P2|ouvert] Gate FrenchBench phase 4 (marge 8 points) : rediscuter sur le résultat réel du 369M (la v2 perd déjà 9,2 points de vocabulaire contre la v9)
  - fait quand: décision écrite après `compare_frenchbench.py 369m sft_369m --marge 8`.
  - réf: `roadmap_modele_369m.md` (phase 4), `compare_frenchbench.py`.
- [P2|ouvert] Taux d'apprentissage du SFT 369M repris de la v2 (3e-4 puis 1e-4), non recalés sur la taille
  - fait quand: lr confirmé ou modifié avant le lancement de l'étape 1.
  - réf: `run_sft_369m.py`.
- [P2|ouvert] R&D cycle 2 : RAFT, expériences 4 (fin de run) et 5 (ResFormer), défaut d'alignement des fenêtres sur les débuts de documents
  - fait quand: programme du cycle 2 décidé par l'utilisateur.
  - réf: `rd/bilan_cycle1.md`, `rd/selection.md`.
- [P1|ouvert] Archiver `roadmap_assistant_v2.md` (4 phases FAIT) : accord explicite de l'utilisateur requis
  - fait quand: roadmap archivée dans `_archives/` ou maintenue sur décision de l'utilisateur.
  - réf: `roadmap_assistant_v2.md`.
- [P2|ouvert] Multi-tours synthétiques et DPO à réévaluer après le benchmark du 369M (phase 5)
  - fait quand: décision écrite après la phase 4.
  - réf: `bilan_assistant_v2.md`, `roadmap_modele_369m.md` (phase 5).
- [P2|ouvert] Parcours `gpu_task.py --pause-369m` jamais exécuté de bout en bout (seuls les refus ont été essayés) ; `cap_vram` non couvert par les tests (chemin cuda)
  - fait quand: première tâche GPU réelle passée par gpu_task sans incident.
  - réf: `gpu_task.py`, `train.py` (`cap_vram`).

## Contexte chaud
- 369M en pré-entraînement continu sous jeton VertIA `creation_llm` : pas 26 390 sur 113 000 le 2026-11-17 18:47, perte de validation 2,83 (v9 : 2,93 au même pas), environ 10 jours restants (estimation au débit de 6 300 tok/s).
- Règle GPU : toute tâche GPU hors `run_369m.py` passe par `gpu_task.py` ; `load_model`, `train.py`, `train_sft.py` refusent `cuda` sans jeton (`NoGpuToken`). Origine : incident du 2026-11-17 14:04 (test lancé pendant que vertia détenait le jeton).
- R&D cycle 1 clos : le v2 lit bien un passage relu (71/90 contre 49/90), mais la recherche automatique (hybride e5-small + BM25) ne donne pas de gain de bout en bout (54/90) ; CFG et TTT-NN en échec. Aucune pause du 369M pour la R&D.
- Assistant retenu : `checkpoints_sft_v2/ckpt.pt` (49,7 %). Lancer : `python run_assistant.py --ckpt checkpoints_sft_v2/ckpt.pt`.

## Dernière session (2026-11-17)
# Session du 2026-11-17

## Décisions prises
- Mode orchestré VertIA (protocole v2) ; embeddings liés pour le 369M ; pré-entraînement lancé (gate phase 1 atteint le 13).
- Cellule R&D créée ; cycle 1 (RAG, CFG, TTT-NN) exécuté puis clos ; RAFT et expériences 4-5 reportées.
- Correctif GPU après incident : `gpu_task.py` seul lanceur, garde `require_gpu_token` dans le code, règle CLAUDE.md.
- SFT 369M en deux étapes comme la v2 ; pause VertIA ajoutée au SFT ; gate phase 4 conservé, à rediscuter.

## Livrables produits ou modifiés
- `gpu_task.py`, `vram_guard.py`, `generate.py`, `train.py`, `train_sft.py`, `run_sft_369m.py`, `compare_frenchbench.py`, `eval_frenchbench.py` : créés ou modifiés ; tests : 134 verts.
- `rd/` (rapports, expériences, `bilan_cycle1.md`, `suivi.html`), `.claude/agents/rd-*.md` : créés.

## Hypothèses validées / invalidées
- VALIDE : 369M en avance sur la v9 à pas égal (2,83 contre 2,93).
- VALIDE : la v2 exploite un passage pertinent (71/90).
- INVALIDE : RAG automatique, CFG, TTT-NN sur la v2 -> pivot : capacité du 369M d'abord.
- EN ATTENTE : SFT 369M (VRAM, débit, environ 8 h par epoch) non mesuré.

## Prochaine étape exacte
Laisser finir le pré-entraînement ; puis FrenchBench et échantillons via `gpu_task.py`, `compare_frenchbench.py v9 369m`.

## Question bloquante pour la session suivante
Aucune
