# Incohérences et points de friction (relevé du 2026-11-18)

Relevé fait en construisant le tableau de bord. Chaque point : constat, effet, proposition. Rien n'est appliqué sans accord.

## Priorité haute

| # | Constat | Effet | Proposition |
|---|---|---|---|
| 1 | La durée restante du 369M dans `_contexte/signals.md` (« environ 10 jours ») repose sur le débit sans pause ; le rythme réel mesuré est d'environ 270 pas/h, soit 12 à 13 jours. | Planning faux entre deux /close. | Se fier au rythme réel du tableau de bord ; corriger au prochain /close. |
| 2 **[identifié le 2026-11-18 : le 369M reste à son plafond (réserve 6,10 Go + contexte CUDA) ; le surplus vient des applications de bureau, non chiffrable par processus sous Windows]** | VRAM totale de la carte entre 7,1 et 7,7 Go pendant le run, pour une limite de protocole de 7,5 Go (le rodage mesurait 7,2 Go). | Risque de dépassement selon les autres applications ouvertes. | Identifier les autres consommateurs (navigateur, bureau) ; abaisser `--vram-cap-gb` à la prochaine reprise si besoin. |
| 3 **[corrigé le 2026-11-18]** | `run_assistant.py` pointe par défaut sur `checkpoints_sft` (v1) alors que la v2 est retenue. | Lancer l'assistant sans option teste la mauvaise version. | Défaut sur `checkpoints_sft_v2/ckpt.pt`. |
| 4 **[corrigé le 2026-11-18]** | Deux files Codex ont tourné en même temps (relance de session et relance manuelle) : doublon dans `rd/recherches/INDEX.md` (corrigé à la main) et quota consommé deux fois. | Quota Codex gaspillé, index incohérent. | Verrou fichier dans `rd/agents_codex/file.sh`. |

## Priorité moyenne

| # | Constat | Effet | Proposition |
|---|---|---|---|
| 5 **[périmé le 2026-11-21 : phase 10+ déjà [FAIT], `roadmap_assistant_v2.md` déjà dans `_archives/`]** | `roadmap_llm_fr.md` garde une « Phase 10+ — Suite [EN COURS] » qui renvoie à `roadmap_assistant.md`, archivée ; `roadmap_assistant_v2.md` est terminée mais non archivée (action ouverte depuis le 2026-11-12). | Trois roadmaps, deux phases « en cours » : on ne sait pas laquelle fait foi. | Clore la phase 10+ en renvoyant à `roadmap_modele_369m.md` ; archiver `roadmap_assistant_v2.md`. |
| 6 **[fait le 2026-11-18 : resultats/ et logs/, journaux du run 369M laissés à la racine]** | Racine encombrée : environ 25 rapports `eval_report*.md`, des `.out`/`.err`, les JSON FrenchBench et d'évaluation à côté du code. | Difficile de distinguer code, résultats et journaux. | Dossier `resultats/` (rapports, JSON) et `logs/` ; adapter les chemins des scripts. |
| 7 **[fait le 2026-11-18 : v3 à v7 et dossiers A/B supprimés, environ 13,7 Go]** | Environ 22 Go de checkpoints obsolètes sur D: (v3 à v8, `checkpoints_sft_ab` 4,9 Go, `checkpoints_sft_v2_ab` 3,3 Go). | Espace disque. | Supprimer après accord explicite (irréversible) ; garder v9, sft_v2, 369m. |
| 8 **[en partie corrigé le 2026-11-18 : modèle unique `gpt-reserve` dans `rd/agents_codex/modele.sh`]** | Codex : le modèle de la configuration (`gpt-6.1-sol`) est refusé par le compte ; `gpt-6-astra` est codé en dur dans 3 scripts ; la limite d'usage bloque des files entières. | Échecs silencieux, relances manuelles. | Modèle dans une variable unique de `commun.md`/scripts ; corriger `~/.codex/config.toml` (hors projet, accord requis). |
| 9 **[corrigé le 2026-11-18 : --ckpt et --nom]** | `eval_frenchbench.py` : appareil `cuda` et dictionnaire `MODELS` en dur. | Chaque nouveau modèle demande d'éditer le code. | Accepter un chemin de checkpoint en argument. |
| 10 **[corrigé le 2026-11-18]** | `run.py generate` et `run.py eval` ont `checkpoints_v8` comme défaut. | Défaut obsolète. | Défaut sur la v9 ou option obligatoire. |

## Priorité basse

| # | Constat | Effet | Proposition |
|---|---|---|---|
| 11 **[corrigé le 2026-11-18]** | La docstring de `vram_guard.py` cite le « protocole version 1 », le projet applique la version 2. | Confusion à la lecture. | Corriger la docstring. |
| 12 **[reporté après le run 369M]** | `gpu_task.py` détecte le 369M par motif de ligne de commande (`run_369m.py`, `checkpoints_369m`). | Fragile si le lanceur change de nom. | Fichier PID écrit par `run_369m.py`. |
| 13 | `signals.md` et `contexte.md` ne changent qu'au /close : entre deux clôtures, l'état est figé. | État « officiel » périmé. | Le tableau de bord lit aussi le journal de session et les fichiers vivants (déjà fait). |
| 14 **[corrigé le 2026-11-18]** | `run.py` importe `torch` dès le démarrage, même pour le tableau de bord. | Ouverture de l'UI plus lente de quelques secondes. | Import paresseux de `torch` dans `run.py`. |
