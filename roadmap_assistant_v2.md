# Roadmap — Assistant v2 (corrections ciblées de l'assistant v1)

Point de départ : assistant v1 `checkpoints_sft/ckpt.pt` (SFT de la v9 ~150M, contexte 1024).
Faiblesses constatées sur `eval_assistant_blind.md` : transformations de phrases recopiées (6/10), multi-tours (0,6 % d'exemples multi-tours dans le jeu SFT), refus hors de propos, erreurs factuelles (limite de taille, hors périmètre de cette roadmap).
Contrainte permanente : aucun lancement sur la VRAM sans accord explicite de l'utilisateur.

## Phase 1 — Benchmark absolu et réglages de génération [FAIT]
- Jeu verrouillé `bench_assistant.jsonl`, écrit avant tout nouvel entraînement, avec réponse attendue ou critère de réussite par item : faits simples, consignes de format, transformations de phrases, conversation, ~20 scénarios à 2 tours.
- Aucun item commun avec `prompts_assistant_holdout.txt` ni avec le jeu SFT (contrôle automatique).
- Mesures automatiques : taux de recopie de la consigne (similarité réponse/phrase d'entrée), refus hors de propos (motifs de refus sur des demandes anodines), boucles (phrases répétées).
- Notation correct / incorrect à l'aveugle, script `eval_bench.py`, graines fixes.
- Référence v1 chiffrée par catégorie.
- Essai de réglages de génération sur le benchmark : pénalité de répétition, température ; adoption dans `chat.py` uniquement si gain mesuré.
- Tests : chargement et validation du jeu, détecteurs (recopie, refus, boucles) sur cas construits.
- Gate : référence v1 reproductible (deux exécutions, mêmes chiffres).

**⏸ Checkpoint** — Demander à l'utilisateur de faire `/compact` avant de continuer.
Attendre sa réponse écrite. Ne pas commencer la phase suivante sans confirmation.

## Phase 2 — Données ciblées [FAIT]
- Multi-tours : mesurer les conversations multi-tours écartées pour longueur > 1024 ; les conserver tronquées aux premiers tours complets qui tiennent dans 1024 tokens.
- Transformations : quelques milliers d'exemples synthétiques en français (politesse, registre, temps, pluriel, question, négation, simplification, résumé, paraphrase, correction de fautes), générés par Ollama `gemma4:12b` (choisi le 2026-11-08 sur banc comparatif de 7 modèles locaux).
  - Phrases d'entrée tirées du corpus (LeCarnet, Vikidia, Wikipédia), nettoyées (références `[n]`, guillemets orphelins), jamais générées par le modèle : en génération libre, les modèles testés produisent des entrées déjà transformées.
  - Applicabilité de la transformation filtrée en amont (pas de « présent » sur une phrase déjà au présent, pas de « pluriel » sur un sujet propre ou déjà pluriel) : les modèles testés n'utilisent jamais l'option « impossible ».
  - Filtres automatiques : recopie (similarité entrée/sortie), forme attendue (« ? » pour une question, particule négative), une seule phrase pour un résumé, JSON valide.
  - Seconde passe de vérification par `gemma4:12b` (juge correct / incorrect), rejet des cas douteux.
  - Consignes formulées de plusieurs façons, pour éviter qu'un modèle de 150M n'apprenne une seule tournure.
  - Débit mesuré : ~12 tokens/s (modèle en partie sur CPU), soit environ 6 exemples/min, à doubler pour la vérification ; génération en tâche de fond reprenable, jamais en même temps qu'un entraînement GPU.
- Audit qualité : 200 exemples tirés au hasard, notés à la main ou par l'assistant IA.
- Filtrage des refus hors de propos dans le jeu SFT existant.
- Nouveau mélange préparé dans `data/sft/prepared_v2/`, statistiques par source.
- Tests : troncature au tour complet, masque aligné, aucune fuite vers le benchmark ni la validation, proportions du mélange conformes.
- Gate : tests verts, part multi-tours et volume de transformations chiffrés, audit qualité à 95 % de transformations correctes au moins.

**⏸ Checkpoint** — Demander à l'utilisateur de faire `/compact` avant de continuer.
Attendre sa réponse écrite. Ne pas commencer la phase suivante sans confirmation.

## Phase 3 — SFT v2 [FAIT]
- A/B court (même budget de tokens) : poursuite depuis l'assistant v1 sur le nouveau mélange contre SFT complet depuis la v9 convertie.
- Entraînement retenu, checkpoints dans `checkpoints_sft_v2/`, reprenable.
- Évaluation : benchmark de la phase 1, FrenchBench 0-shot (`eval_frenchbench.py`).
- Gate : sur le benchmark, recopie des transformations divisée par 2 au moins, 2e tour pertinent dans la majorité des scénarios, aucune catégorie en baisse ; FrenchBench grammaire et vocabulaire sans baisse au-delà de la marge (~8 points) par rapport à la v9.

**⏸ Checkpoint** — Demander à l'utilisateur de faire `/compact` avant de continuer.
Attendre sa réponse écrite. Ne pas commencer la phase suivante sans confirmation.

## Phase 4 — Bilan et décision [FAIT]
- Rapport v1 contre v2 par catégorie du benchmark.
- Préparation du modèle plus grand from scratch (décision de principe du 2026-11-08 : from scratch, pas de contrainte de durée) : court essai de débit et de VRAM sur 2 ou 3 tailles (~300M, ~350M), besoin en données chiffré, choix entre agrandissement de la v9 et nouveau pré-entraînement.
- Livrable : roadmap dédiée au modèle plus grand, rédigée à partir des mesures ; DPO réévalué selon les défauts résiduels de la v2.

**⏸ Checkpoint** — Demander à l'utilisateur de faire `/compact` avant de continuer.
Attendre sa réponse écrite. Ne pas commencer la phase suivante sans confirmation.
