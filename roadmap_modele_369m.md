# Roadmap — modèle 369M from scratch

Objectif : un assistant plus capable que la v2 (`checkpoints_sft_v2/ckpt.pt`, 49,7 % au benchmark). Principal défaut actuel : `critere_absent`, c'est-à-dire l'information attendue qui manque.
Base chiffrée : `bilan_assistant_v2.md`, section « Essai de taille ».
Décision du 2026-11-12 : pré-entraînement from scratch d'un modèle 24 × 1024 (16 têtes, contexte 1 024, vocab 32 768), avec recalcul des activations.

## Phase 1 — Recalcul des activations et run de rodage [TODO]
- Ajouter une option `grad_checkpoint` à `TrainConfig` et à `GPT` : recalcul bloc par bloc, non réentrant, compatible avec compile. Elle doit être sans effet en génération.
- Tests :
  - pertes et gradients identiques avec et sans recalcul, sur un petit modèle ;
  - reprise `--resume` intacte.
- Créer `run_369m.py` sur le modèle de `run_v9.py` :
  - même mélange de corpus (fw2hq 0,58, wiki_fr 0,25, lecarnet 0,15, vikidia 0,02) ;
  - lot 4 × accumulation 16 = 65 536 tokens par pas ;
  - environ 113 000 pas, soit environ 7,4 milliards de tokens.
- Lr à rechercher, au départ celui de la v9 (3e-3, muon).
- Rodage d'environ 1 h (GPU, accord requis) :
  - débit et VRAM sur données réelles ;
  - perte qui décroît ;
  - point de sauvegarde puis reprise validés.
- Gate :
  - tests au vert ;
  - débit au moins égal à 6 000 tok/s ;
  - VRAM au pic sous 6,5 Go ;
  - reprise sans saut de perte.

**⏸ Checkpoint** — Demander à l'utilisateur de faire `/compact` avant de continuer.
Attendre sa réponse écrite. Ne pas commencer la phase suivante sans confirmation.

## Phase 2 — Pré-entraînement [TODO]
- Run complet d'environ 12 jours de calcul continu (GPU, accord requis), repris par `--resume` après chaque interruption.
- Partage de la VRAM :
  - la marge n'est que d'environ 1 Go : prévenir les autres sessions (Creation video, world model, creation_3d) au lancement ;
  - prévoir une pause et une reprise propres quand elles ont besoin du GPU.
- Suivi : perte de validation à comparer à la v9 au même nombre de tokens (`checkpoints_v9/log.csv`).
- Gate : perte de validation finale inférieure à celle de la v9 (2,5445).

**⏸ Checkpoint** — Demander à l'utilisateur de faire `/compact` avant de continuer.
Attendre sa réponse écrite. Ne pas commencer la phase suivante sans confirmation.

## Phase 3 — Évaluation de la base [TODO]
- FrenchBench 0-shot (`eval_frenchbench.py`) contre la v9 (grammaire 73,9, vocabulaire 68,9).
- Échantillons de génération libre.
- Gate : FrenchBench au moins égal à la v9 sur les deux tâches.

**⏸ Checkpoint** — Demander à l'utilisateur de faire `/compact` avant de continuer.
Attendre sa réponse écrite. Ne pas commencer la phase suivante sans confirmation.

## Phase 4 — SFT et benchmark assistant [TODO]
- Conversion au format de chat (`convert_chat.py`, vocab 32 832).
- SFT sur `data/sft/prepared_v2` (recette de la v2 : 1 epoch, warmup 50, lr à recaler sur la taille).
- Évaluations :
  - `bench_assistant.py` (300 cas, mêmes paramètres de génération) ;
  - FrenchBench contre la base de la phase 3.
- Gate :
  - global > 49,7 % ;
  - aucune catégorie sous la v2 ;
  - FrenchBench dans la marge d'environ 8 points par rapport à la base.

**⏸ Checkpoint** — Demander à l'utilisateur de faire `/compact` avant de continuer.
Attendre sa réponse écrite. Ne pas commencer la phase suivante sans confirmation.

## Phase 5 — Bilan et DPO [TODO]
- Rapport 369M contre v2 par catégorie et par cause d'échec.
- DPO réévaluée selon les défauts résiduels (recopie, boucle, répétition de tour, refus : 23 occurrences sur la v2).
- Question ouverte reprise : multi-tours synthétiques.

**⏸ Checkpoint** — Demander à l'utilisateur de faire `/compact` avant de continuer.
Attendre sa réponse écrite. Ne pas commencer la phase suivante sans confirmation.
