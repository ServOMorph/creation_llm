# Bilan assistant v1 contre v2

Benchmark `bench_assistant.py` : 300 cas, scoring automatique, génération à température 0,5, top_k 50, top_p 0,95, 300 tokens au plus.
- v1 : `checkpoints_sft/ckpt.pt`, 5 142 pas.
- v2 : `checkpoints_sft_v2/ckpt.pt`, 2 606 pas, 1 epoch sur `data/sft/prepared_v2`, partie de la v1.

## Par catégorie

| Catégorie | v1 | v2 | Écart |
|---|---|---|---|
| Faits (90) | 47 (52,2 %) | 49 (54,4 %) | +2 |
| Consignes (45) | 17 (37,8 %) | 18 (40,0 %) | +1 |
| Transformations (75) | 7 (9,3 %) | 39 (52,0 %) | +32 |
| Conversation (30) | 15 (50,0 %) | 13 (43,3 %) | -2 |
| Multi-tours (60) | 21 (35,0 %) | 30 (50,0 %) | +9 |
| **Global (300)** | **107 (35,7 %)** | **149 (49,7 %)** | **+42** |

## Causes d'échec

| Raison | v1 | v2 |
|---|---|---|
| critere_absent | 173 | 139 |
| recopie | 52 | 13 |
| critere_interdit | 17 | 15 |
| repete_tour | 12 | 6 |
| critere_incomplet | 7 | 3 |
| refus | 5 | 2 |
| trop_long | 4 | 3 |
| boucle | 3 | 2 |
| trop_de_phrases | 0 | 2 |

## FrenchBench (0-shot, points)

| Tâche | v9 base | v1 | v2 |
|---|---|---|---|
| Grammaire | 73,9 | 70,6 | 69,7 |
| Vocabulaire | 68,9 | 62,2 | 59,7 |

## Lecture

- Le gain de la v2 vient des transformations et des multi-tours : la recopie est divisée par 4.
- Le défaut dominant est maintenant `critere_absent` (139 des 151 échecs ; un échec peut cumuler plusieurs raisons) : la réponse est bien formée, mais l'information attendue manque. Hypothèse : c'est un manque de connaissances et de capacité du modèle, plutôt qu'un défaut d'alignement.
- Réserves acceptées le 2026-11-12 : conversation -2/30, vocabulaire -9,2 points par rapport à la v9, multi-tours pile à 50 %.

## DPO

Les défauts qu'une DPO corrige bien (recopie, boucle, répétition de tour, refus) ne totalisent plus que 23 occurrences, pour 151 échecs. La DPO a donc un faible potentiel sur la v2 : elle n'est pas prioritaire et sera réévaluée sur le modèle plus grand.

## Essai de taille (2026-11-12)

Mesures sur données aléatoires : contexte 1 024, 16 384 tokens par pas, muon et compile, 6 pas mesurés. Le pic de VRAM est celui alloué par PyTorch, sans les quelque 0,6 à 2,4 Go pris par Windows et les autres applications.

| Modèle | Recalcul | Lot | Débit (tok/s) | VRAM au pic |
|---|---|---|---|---|
| v9, 192M (20 × 768) | non | 2 | 14 906 | 4,38 Go |
| v9, 192M | oui | 2 | 13 171 | 3,35 Go |
| 369M (24 × 1024) | non | 2 | 1 676 | 7,41 Go (déborde) |
| 369M | oui | 2 | 6 372 | 6,04 Go |
| 369M | oui | 4 | 6 994 | 6,33 Go |
| 419M (28 × 1024) | non | 2 | 498 | 8,38 Go (déborde) |
| 419M | oui | 2 | 548 | 6,79 Go (déborde) |

- La plus grande taille utilisable est 369M avec recalcul des activations (gradient checkpointing), à environ 7 000 tok/s, soit la moitié du débit de la v9.
- 419M déborde même avec recalcul : un bon débit demanderait un optimiseur 8 bits ou des états déchargés en RAM, à mesurer.
- Marge en VRAM du 369M : environ 1 Go. Si d'autres IA locales prennent de la VRAM pendant le pré-entraînement, il déborde et ralentit.

## Besoin en données et durée

- Corpus déjà encodé : environ 6,65 milliards de tokens (fw2hq 4,33 ; wiki_fr 1,89 ; lecarnet 0,37 ; vikidia 0,05).
- 369M avec 20 tokens par paramètre : environ 7,4 milliards de tokens, soit 1,1 passe sur le corpus. À 7 000 tok/s, cela fait environ 294 h, soit 12,2 jours de calcul continu.
- Pour comparaison, la v9 avait vu 6 milliards de tokens en environ 119 h.
