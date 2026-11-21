# Signals — creation_llm   (MAJ 2026-11-21)

## Actions ouvertes
- [P1|ouvert] Phase 2 de `roadmap_modele_369m.md` : pré-entraînement 369M jusqu'au pas 113 000
  - fait quand: `checkpoints_369m/etat.json` à « termine » et perte de validation finale < 2,5445 (v9).
  - réf: `run_369m.py`, `checkpoints_369m/log.csv`, tableau de bord (`python run.py`).
- [P1|ouvert] Phases 3-4 à la fin du pré-entraînement : FrenchBench + échantillons, conversion au chat, SFT en deux étapes, benchmark assistant
  - fait quand: `compare_frenchbench.py v9 369m` exécuté, puis `checkpoints_sft_369m/ckpt.pt` benchmarké.
  - réf: `compare_frenchbench.py`, `run_sft_369m.py`, `eval_frenchbench.py` (`--ckpt`), `convert_chat.py`, `bench_assistant.py`.
- [P2|ouvert] Cycle R&D 2, étape a (CPU) : relecture humaine d'A1 (106 échecs « omission ou réponse différente ») et d'A2 (200 familles RAFT/H2) ; étapes b et c à valider après la base
  - fait quand: relectures faites et décision écrite sur les étapes b et c du programme.
  - réf: `rd/programme_cycle2.md`, `rd/experiences/c2_audit_v2.json`, `rd/experiences/c2_audit_echantillon20.json`, `rd/experiences/data_c2/`.
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
- [P2|ouvert] Process Codex à régler avec VibeObs : binaire (npm 0.154 refuse gpt-6.1-sol, binaire application 0.167 l'accepte ; VibeObs juge l'action sur 0.154 probablement obsolète, non vérifié ici) et modèle (`AGENTS.md` : gpt-6.2-sol d'après le dernier commit ; `rd/agents_codex/modele.sh` : à relire)
  - fait quand: décision de l'utilisateur appliquée dans `modele.sh` et `AGENTS.md` finalisé par VibeObs.
  - réf: `AGENTS.md`, `rd/agents_codex/modele.sh`.
- Backlog : 5 entrées (dette 2, piste 3) — `_contexte/backlog.md`

## Contexte chaud
- 369M : pas 54 230 sur 113 000 le 2026-11-21 19:47 (perte d'entraînement 2,71 à ce pas, valeur bruitée) ; rythme environ 6 000 tok/s, environ 7 à 8 jours restants (estimation). Débit passé d'environ 6 250 à 6 000 tok/s pendant les calculs CPU auxiliaires du 2026-11-20 (-4 %, cause non établie, seuil du programme : 1 %) : aucun calcul auxiliaire relancé depuis.
- VRAM totale de la carte 7,1 à 7,7 Go pour une limite de 7,5 Go : le 369M reste à son plafond (réserve 6,10 Go), le surplus vient des applications de bureau.
- Tableau de bord : `python run.py` (ou `run.py ui`) ; artefact mobile republié sur demande seulement, registre `_contexte/artifacts.md`.
- Cycle 2, étape a : `rd/experiences/c2_audit.py`, `c2_data.py`, `c2_checks.py` livrés (13 tests) ; `data_c2/` est régénérable (graine 1234) et ignoré par git. Audit v2 : sur 20 cas « omission » relus, 20 sont de vrais échecs (aucun motif trop strict) ; au moins 8 sont des réponses fausses, pas des omissions.
- `.gitattributes` (`* text=auto eol=lf`) créé par VibeObs et commité ; 88 fichiers de l'arbre restent en CRLF, aucune renormalisation lancée.
- Recherches : file `rd/sujets_a_traiter.md`, `sh rd/agents_codex/file.sh` (verrou) ; veille `sh rd/agents_codex/veille.sh` ; jamais `gpt-6-astra` (abonnement) ; le quota Codex peut bloquer une file.
- Règle GPU : toute tâche GPU hors `run_369m.py` passe par `gpu_task.py` ; garde `NoGpuToken` dans le code.
- Racine rangée : résultats dans `resultats/`, journaux dans `logs/` (sauf ceux du run 369M). `ameliorations.md` : 14 points, état par point.

## Dernière session (2026-11-21)
# Session du 2026-11-21

## Décisions prises
- Programme du cycle 2 validé pour l'étape a seule (A1 à A3, CPU) ; étapes b et c à valider après la base.
- Échantillon A1 : 20 cas relus, tous de vrais échecs (cas 1-4 et 16 par l'utilisateur, les autres par l'assistant).
- Commit : scripts c2 et `.gitattributes` ; `data_c2/` ignoré par git (régénérable).

## Livrables produits ou modifiés
- `rd/experiences/c2_audit.py`, `c2_data.py`, `c2_checks.py` et leurs tests : livrés, 13 tests OK.
- `rd/experiences/data_c2/` : pilote 2 000, dev 120, test 900 ; 0 fuite, 0 erreur de validation.
- `.gitattributes` : créé par VibeObs, commité.

## Hypothèses validées / invalidées
- VALIDE : reprise identique (Muon et AdamW), masques et isolation des documents corrects sur petit GPT.
- INVALIDE : « les 106 omissions seraient surtout un motif trop strict » -> échantillon de 20 : aucun.
- INVALIDE : « phase 10+ de `roadmap_llm_fr.md` encore en cours » (affirmation de session, tirée d'`ameliorations.md` périmé) -> elle est [FAIT].
- EN ATTENTE : effet des calculs CPU sur le débit du run (-4 % observé, cause non établie).

## Prochaine étape exacte
Laisser finir le pré-entraînement ; relire A1 et A2 au PC ; décider le process Codex avec VibeObs.

## Question bloquante pour la session suivante
Aucune
