# Signals — creation_llm   (MAJ 2026-11-25)

## Actions ouvertes
- [P1|ouvert] Phase 2 de `roadmap_modele_369m.md` : pré-entraînement 369M jusqu'au pas 113 000 (run en pause au pas 63 085, aucun processus le 2026-11-25 à 22:43, à relancer)
  - fait quand: `checkpoints_369m/etat.json` à « termine » et perte de validation finale < 2,5445 (v9).
  - réf: `run_369m.py`, `checkpoints_369m/log.csv`, `checkpoints_369m/etat.json`, tableau de bord (`python run.py`).
- [P1|ouvert] Phases 3-4 à la fin du pré-entraînement : FrenchBench + échantillons, conversion au chat, SFT en deux étapes, benchmark assistant
  - fait quand: `compare_frenchbench.py v9 369m` exécuté, puis `checkpoints_sft_369m/ckpt.pt` benchmarké.
  - réf: `compare_frenchbench.py`, `run_sft_369m.py`, `eval_frenchbench.py` (`--ckpt`), `convert_chat.py`, `bench_assistant.py`.
- [P2|ouvert] Cycle R&D 2, étape a (CPU) : juger les 14 familles A2 limites (0 jugée), puis écrire les 200 étiquettes (186 « correctes, contrôle automatique », accord utilisateur) ; étapes b et c à valider après la base
  - fait quand: relectures faites et décision écrite sur les étapes b et c du programme.
  - réf: `rd/programme_cycle2.md`, `rd/experiences/c2_audit_v2.json`, `rd/experiences/relecture_a1/`, `rd/experiences/relecture_a2/`, `rd/experiences/data_c2/`, `_contexte/artifacts.md`.
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
- [P2|ouvert] Proposition VibeObs de passage au kit v8.1 : validation 1/3 accordée, validation 2/3 présentée sans accord ; aucune réponse envoyée à VibeObs, rien d'écrit
  - fait quand: trois validations obtenues et écriture faite par VibeObs, ou refus transmis à « VibeObs Général ».
  - réf: `.claude/CLAUDE.md`, `.claude/commands/start.md`, `.claude/commands/close.md`, `D:\ServOMorph\VibeObs\templates\.claude\CLAUDE.md`
- Backlog : 6 entrées (dette 3, piste 3) — `_contexte/backlog.md`

## Contexte chaud
- 369M : en pause au pas 63 085 (`etat.json`, 2026-11-25 01:36:40, « pause_demandee » ; journal `vram_lock` : demande de `vertia` à 01:32). PC redémarré le 2026-11-25 à 22:34 ; à 22:43 aucun processus d'entraînement ni veilleur, jeton libre, file vide, pas de pause utilisateur. Reprise : `.venv\Scripts\python.exe run_369m.py`, sur ordre de l'utilisateur (un lancement en arrière-plan par la session a été accepté le 2026-11-24). Rythme relevé le 2026-11-24 : 6 210 à 6 280 tok/s, environ 10,5 s par pas.
- Pause utilisateur `vram_lock` : tant qu'elle est active, `acquire` de `creation_llm` est refusé (code 7) et un `acquire --wait` expire après 240 min. Posée et levée plusieurs fois du 2026-11-24 au 25 ; auteur non relevé.
- Relecture A1 terminée le 2026-11-22 : 106 cas étiquetés (84 vrais échecs par l'utilisateur, 15 vrais échecs « avis assistant » de l'échantillon, 7 motif trop strict) dans `rd/experiences/c2_audit_v2.json` (`etiquette_humaine`) et `rd/experiences/relecture_a1/resultats_a1.json` ; relancer `c2_audit.py` efface les étiquettes du JSON. Page https://claude.ai/artifact/2evoVpypZMt1q3kpvjAUSn avec base de données (collection `a1`) ; source `rd/experiences/relecture_a1/` (`gen.py`, `tpl.html`, `relecture_a1.html`). 15 des 86 cas sont multi-tours : `c2_audit_v2.json` ne garde que le dernier tour, le dialogue complet vient de `bench_assistant.jsonl` et `resultats/bench_assistant_results/v2.json`.
- Renouvellement du jeton (`train.py` `token_hours` 23 h, `vram_guard.PauseMonitor`, code 75, release puis acquire et relance) : lu dans le code, jamais observé. Veilleur VertIA (`_orchestration/veilleur_reprise.py`) : annoncé pour relancer `run_369m.py` après pause, code non lu.
- Cycle 2, étape a : `rd/experiences/c2_audit.py`, `c2_data.py`, `c2_checks.py` livrés (13 tests) ; `data_c2/` est régénérable (graine 1234) et ignoré par git. Audit v2 : sur 20 cas « omission » relus, 20 sont de vrais échecs ; au moins 8 sont des réponses fausses, pas des omissions.
- Codex : modèle imposé par `~/.codex/politique_delegation.json` (`gpt-6.2-sol`) ; binaire choisi par `_codex/run_codex.ps1` : champ `binaire` de la politique (vide), puis `codex.exe` de l'application (`...\OpenAI\Codex\bin\f71a721e0e6193a9`, présent), puis npm ; version du binaire non lue ; quota sous les seuils au dernier relevé (2026-11-20 17:20 UTC). Recherches : file `rd/sujets_a_traiter.md`, `sh rd/agents_codex/file.sh` (verrou) ; veille `sh rd/agents_codex/veille.sh` ; jamais `gpt-6-astra` (abonnement) ; le quota Codex peut bloquer une file.
- `.gitattributes` (`* text=auto eol=lf`) créé par VibeObs et commité ; 88 fichiers de l'arbre restent en CRLF, aucune renormalisation lancée.
- Tableau de bord : `python run.py` (ou `run.py ui`) ; artefact mobile republié sur demande seulement, registre `_contexte/artifacts.md`.
- Règle GPU : toute tâche GPU hors `run_369m.py` passe par `gpu_task.py` ; garde `NoGpuToken` dans le code.
- Racine rangée : résultats dans `resultats/`, journaux dans `logs/` (sauf ceux du run 369M). `ameliorations.md` : 14 points, état par point.

## Dernière session (2026-11-25)
# Session du 2026-11-25

## Décisions prises
- A2 : contrôle automatique par l'assistant, l'utilisateur ne juge que les cas limites ; les 186 autres familles seront marquées « correctes (contrôle automatique) » après ses 14 jugements.
- 369M : relance demandée le 2026-11-24, bloquée par la pause utilisateur ; clôture demandée au lieu de lever la pause.

## Livrables produits ou modifiés
- `rd/experiences/relecture_a2/` (`gen.py`, `tpl.html`, `auto_checks.py`, `flags_a2.json`, `relecture_a2.html`) et artefact « Relecture A2 » : livrés.
- `rd/experiences/relecture_a1/resultats_a1.json`, étiquettes de `c2_audit_v2.json` : livrés (commit 8c935e9).

## Hypothèses validées / invalidées
- VALIDE : aucune contradiction détectée par le contrôle automatique sur les familles A2 ; 14 limites (noms ou valeurs proches).
- VALIDE : le 369M a repris et tourné le 2026-11-24 (pas 61 758 à 63 084) à environ 6 200 tok/s ; GPU relevé entre 44 et 100 % après un creux isolé à 12 %.
- EN ATTENTE : 14 familles A2 à juger (mots réels et plausibilité non contrôlés) ; renouvellement du jeton ; proposition kit v8.1.

## Prochaine étape exacte
Relancer le 369M (pas 63 085) ; juger les 14 familles A2 puis écrire les étiquettes.

## Question bloquante pour la session suivante
Relancer le 369M maintenant que le PC a redémarré et que la pause utilisateur est levée ?
