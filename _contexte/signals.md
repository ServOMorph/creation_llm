# Signals — creation_llm   (MAJ 2026-11-19)

## Actions ouvertes
- [P1|ouvert] Phase 2 de `roadmap_modele_369m.md` : pré-entraînement 369M jusqu'au pas 113 000
  - fait quand: `checkpoints_369m/etat.json` à « termine » et perte de validation finale < 2,5445 (v9).
  - réf: `run_369m.py`, `checkpoints_369m/log.csv`, tableau de bord (`python run.py`).
- [P1|ouvert] Phases 3-4 à la fin du pré-entraînement : FrenchBench + échantillons, conversion au chat, SFT en deux étapes, benchmark assistant
  - fait quand: `compare_frenchbench.py v9 369m` exécuté, puis `checkpoints_sft_369m/ckpt.pt` benchmarké.
  - réf: `compare_frenchbench.py`, `run_sft_369m.py`, `eval_frenchbench.py` (`--ckpt`), `convert_chat.py`, `bench_assistant.py`.
- [P2|ouvert] Valider le programme du cycle R&D 2, puis lancer son étape 1 sur CPU (audit des omissions, vérifications STP, corpus RAFT vérifié) sans ralentir le run
  - fait quand: programme validé par l'utilisateur et étape 1 commencée.
  - réf: `rd/programme_cycle2.md`, `rd/analyse_lecun_jepa.md`, `rd/analyse_frugalite.md`.
- [P2|ouvert] Gate FrenchBench phase 4 (marge 8 points) : rediscuter sur le résultat réel du 369M (la v2 perd déjà 9,2 points de vocabulaire contre la v9)
  - fait quand: décision écrite après `compare_frenchbench.py 369m sft_369m --marge 8`.
  - réf: `roadmap_modele_369m.md` (phase 4), `compare_frenchbench.py`.
- [P2|ouvert] Taux d'apprentissage du SFT 369M repris de la v2 (3e-4 puis 1e-4), non recalés sur la taille
  - fait quand: lr confirmé ou modifié avant le lancement de l'étape 1 du SFT.
  - réf: `run_sft_369m.py`.
- [P2|ouvert] Après le run : ranger les journaux du 369M (`run_369m*.out/.err`) dans `logs/` en adaptant `gpu_task.py`
  - fait quand: racine sans journaux du run et relance du 369M fonctionnelle.
  - réf: `gpu_task.py`, `run_369m.py`, `ameliorations.md` (point 6).
- [P2|ouvert] Parcours `gpu_task.py --pause-369m` jamais exécuté de bout en bout ; `cap_vram` non couvert par les tests (chemin cuda)
  - fait quand: première tâche GPU réelle passée par gpu_task sans incident.
  - réf: `gpu_task.py`, `train.py` (`cap_vram`).
- [P2|ouvert] Multi-tours synthétiques et DPO à réévaluer après le benchmark du 369M (phase 5)
  - fait quand: décision écrite après la phase 4.
  - réf: `bilan_assistant_v2.md`, `roadmap_modele_369m.md` (phase 5).
- [P2|ouvert] Process Codex à régler avec VibeObs : binaire (npm 0.154 refuse gpt-6.1-sol, binaire application 0.167 l'accepte) et modèle (`AGENTS.md` : gpt-6.1-sol ; `rd/agents_codex/modele.sh` : gpt-reserve)
  - fait quand: décision de l'utilisateur appliquée dans `modele.sh` et `AGENTS.md` finalisé par VibeObs ; commit local 2c4445e de VibeObs à ne pas pousser avant.
  - réf: `AGENTS.md`, `rd/agents_codex/modele.sh`.
- Backlog : 4 entrées (dette 1, piste 3) — `_contexte/backlog.md`

## Contexte chaud
- 369M : pas 37 690 sur 113 000 le 2026-11-19 (perte d'entraînement 2,655) ; au pas 34 230 le 2026-11-18, perte de validation 2,80 (v9 : 2,88 au pas 34 000) ; rythme réel environ 290 pas/h sur 24 h, soit environ 11 à 12 jours restants (estimation, pauses VertIA comprises).
- VRAM totale de la carte 7,1 à 7,7 Go pour une limite de 7,5 Go : le 369M reste à son plafond (réserve 6,10 Go), le surplus vient des applications de bureau (navigateurs, Docker Desktop, Parsec, RustDesk…).
- Tableau de bord : `python run.py` (ou `run.py ui`) ; artefact mobile republié sur demande seulement (`python -m ui.snapshot` puis publication), registre `_contexte/artifacts.md`.
- Recherches : file `rd/sujets_a_traiter.md`, `sh rd/agents_codex/file.sh` (verrou) ; veille `sh rd/agents_codex/veille.sh` ; modèle Codex `gpt-reserve` (`rd/agents_codex/modele.sh`), jamais `gpt-6-astra` (abonnement) ; le quota Codex peut bloquer une file.
- Règle GPU : toute tâche GPU hors `run_369m.py` passe par `gpu_task.py` ; garde `NoGpuToken` dans le code.
- Racine rangée : résultats dans `resultats/`, journaux dans `logs/` (sauf ceux du run 369M). `ameliorations.md` : 14 points, état par point.

## Dernière session (2026-11-19)
# Session du 2026-11-19

## Décisions prises
- Veille : liste de sources prioritaires ajoutée au prompt (lancement toujours manuel).
- Process Codex (binaire et modèle) laissé à l'utilisateur et VibeObs ; rien modifié en attendant.

## Livrables produits ou modifiés
- `rd/agents_codex/veille_actu.md` : sources prioritaires ajoutées.

## Hypothèses validées / invalidées
- INVALIDE : « gpt-6.1-sol refusé à cause du compte » -> cause = Codex npm 0.154 ; le binaire application 0.167 l'accepte.

## Prochaine étape exacte
Laisser finir le pré-entraînement ; appliquer le process Codex décidé avec VibeObs ; valider `rd/programme_cycle2.md`.

## Question bloquante pour la session suivante
Aucune
