# Signals — creation_llm   (MAJ 2026-11-22)

## Actions ouvertes
- [P1|ouvert] Phase 2 de `roadmap_modele_369m.md` : pré-entraînement 369M jusqu'au pas 113 000 (run en pause au pas 54 349, reprise à décider)
  - fait quand: `checkpoints_369m/etat.json` à « termine » et perte de validation finale < 2,5445 (v9).
  - réf: `run_369m.py`, `checkpoints_369m/log.csv`, `checkpoints_369m/etat.json`, tableau de bord (`python run.py`).
- [P1|ouvert] Phases 3-4 à la fin du pré-entraînement : FrenchBench + échantillons, conversion au chat, SFT en deux étapes, benchmark assistant
  - fait quand: `compare_frenchbench.py v9 369m` exécuté, puis `checkpoints_sft_369m/ckpt.pt` benchmarké.
  - réf: `compare_frenchbench.py`, `run_sft_369m.py`, `eval_frenchbench.py` (`--ckpt`), `convert_chat.py`, `bench_assistant.py`.
- [P2|ouvert] Cycle R&D 2, étape a (CPU) : relecture humaine d'A2 (200 familles RAFT/H2, non commencée) ; A1 relue (106 cas étiquetés) ; étapes b et c à valider après la base
  - fait quand: relectures faites et décision écrite sur les étapes b et c du programme.
  - réf: `rd/programme_cycle2.md`, `rd/experiences/c2_audit_v2.json`, `rd/experiences/relecture_a1/`, `rd/experiences/data_c2/`, `_contexte/artifacts.md`.
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
- Backlog : 6 entrées (dette 3, piste 3) — `_contexte/backlog.md`

## Contexte chaud
- 369M : en pause au pas 54 349 (`etat.json`, 2026-11-21 21:25, raison « pause_demandee », demandeur non établi) ; aucun processus `run_369m` le 2026-11-22 vers 07:20 ; reprise : relancer la même commande avec `--resume`, sur ordre de l'utilisateur et via les règles GPU. Dernier relevé de rythme : environ 6 000 tok/s (-4 % pendant les calculs CPU du 2026-11-20, cause non établie, seuil du programme 1 %).
- VRAM : 941 Mio utilisés sur 8 188 le 2026-11-22 après arrêt de ComfyUI-Qwen (processus hors projet, PID 24004 et 57148, à la demande de l'utilisateur) ; elle était à 5 688 Mio avant. L'utilisateur garde la VRAM libre, rien n'est relancé.
- Relecture A1 terminée le 2026-11-22 : 106 cas étiquetés (84 vrais échecs par l'utilisateur, 15 vrais échecs « avis assistant » de l'échantillon, 7 motif trop strict) dans `rd/experiences/c2_audit_v2.json` (`etiquette_humaine`) et `rd/experiences/relecture_a1/resultats_a1.json` ; relancer `c2_audit.py` efface les étiquettes du JSON. Page https://claude.ai/artifact/2evoVpypZMt1q3kpvjAUSn avec base de données (collection `a1`) ; source `rd/experiences/relecture_a1/` (`gen.py`, `tpl.html`, `relecture_a1.html`). 15 des 86 cas sont multi-tours : `c2_audit_v2.json` ne garde que le dernier tour, le dialogue complet vient de `bench_assistant.jsonl` et `resultats/bench_assistant_results/v2.json`.
- VertIA (message du 2026-11-22) demande de relancer le 369M après les générations du portail, via `vram_lock.py acquire creation_llm --wait` puis `run_369m.py` ; non appliqué, l'utilisateur voulait relancer lui-même ; en attente de son accord.
- Cycle 2, étape a : `rd/experiences/c2_audit.py`, `c2_data.py`, `c2_checks.py` livrés (13 tests) ; `data_c2/` est régénérable (graine 1234) et ignoré par git. Audit v2 : sur 20 cas « omission » relus, 20 sont de vrais échecs ; au moins 8 sont des réponses fausses, pas des omissions.
- Codex : modèle imposé par `~/.codex/politique_delegation.json` (`gpt-6.2-sol`) ; binaire choisi par `_codex/run_codex.ps1` : champ `binaire` de la politique (vide), puis `codex.exe` de l'application (`...\OpenAI\Codex\bin\f71a721e0e6193a9`, présent), puis npm ; version du binaire non lue ; quota sous les seuils au dernier relevé (2026-11-20 17:20 UTC). Recherches : file `rd/sujets_a_traiter.md`, `sh rd/agents_codex/file.sh` (verrou) ; veille `sh rd/agents_codex/veille.sh` ; jamais `gpt-6-astra` (abonnement) ; le quota Codex peut bloquer une file.
- `.gitattributes` (`* text=auto eol=lf`) créé par VibeObs et commité ; 88 fichiers de l'arbre restent en CRLF, aucune renormalisation lancée.
- Tableau de bord : `python run.py` (ou `run.py ui`) ; artefact mobile republié sur demande seulement, registre `_contexte/artifacts.md`.
- Règle GPU : toute tâche GPU hors `run_369m.py` passe par `gpu_task.py` ; garde `NoGpuToken` dans le code.
- Racine rangée : résultats dans `resultats/`, journaux dans `logs/` (sauf ceux du run 369M). `ameliorations.md` : 14 points, état par point.

## Dernière session (2026-11-22)
# Session du 2026-11-22

## Décisions prises
- Process Codex réglé par le code (modèle par la politique centrale, binaire de l'application) ; action close.
- Relecture A1 : 19 cas jugés par l'utilisateur au téléphone, puis page dédiée (« Relecture A1 », dialogue complet, sauvegarde en base) pour les 67 autres ; 106 cas étiquetés au total.
- VRAM libérée par arrêt de ComfyUI-Qwen à la demande ; 369M laissé en pause, VRAM gardée libre.

## Livrables produits ou modifiés
- `rd/experiences/relecture_a1/` (`gen.py`, `tpl.html`, `relecture_a1.html`, `resultats_a1.json`) et artefact publié : livrés ; `c2_audit_v2.json` : étiquettes humaines écrites.
- `_contexte/artifacts.md` : ligne Relecture A1 ajoutée.

## Hypothèses validées / invalidées
- VALIDE : modèle et binaire Codex déterminés par `_codex/run_codex.ps1` et la politique centrale.
- INVALIDE : « `modele.sh` à relire » -> le fichier n'existe plus (commit 7f71e65).
- INVALIDE : les cas A1 présentés avec leur question complète -> 15 sur 86 sont multi-tours, contexte absent de `c2_audit_v2.json`.
- VALIDE : A1 relue, 99 vrais échecs sur 106 (15 sans validation de l'utilisateur), 7 motif trop strict.
- EN ATTENTE : A2 non commencée ; cause de la pause du 369M.

## Prochaine étape exacte
Relire A2 (200 familles RAFT/H2) ; décider la reprise du 369M (en pause, VRAM libre, demande de VertIA en attente).

## Question bloquante pour la session suivante
Quand relancer le 369M, en pause au pas 54 349 avec la VRAM libre ?
