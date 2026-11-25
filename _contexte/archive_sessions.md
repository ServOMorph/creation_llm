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
---
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

---
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

---
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
