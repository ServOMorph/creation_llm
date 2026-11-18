# Évaluation finale du modèle généraliste (v9)

## Protocole
- Modèles : v9 (20 couches, n_embd 768, tokenizer 32k, pas 91 500, corpus mélangé fw2hq/wiki_fr/lecarnet/vikidia) contre v8 (18 couches, tokenizer 8k, pas 5 900, histoires seules).
- Perplexité : bits par octet (comparables entre tokenizers) sur 2 000 histoires de `data/validation.jsonl`, jamais vues par les deux modèles ; perplexité par domaine de la v9 sur 200 fenêtres de 1 024 tokens de chaque jeu de validation.
- Histoires : nouveau jeu scellé `prompts_holdout3.txt` (20 débuts écrits avant toute génération), 3 graines (1234, 2000, 3000), température 0,4, top-k 50, top-p 0,95, 150 tokens : 60 textes par modèle.
- Notation à l'aveugle : pour chaque prompt et graine, les deux textes sont mélangés (A/B), tronqués à la même longueur, correspondance révélée après notation (`eval_v9_key.json`).
- Grille identique à la v8 : erreur majeure (confusion de personnages, prénom qui change, contradiction, logique incohérente, sujet abandonné), cas limite (compté dans le strict), erreur mineure (non comptée).
- Gate (roadmap, phase 9) : meilleur que v8 sur les histoires nouvelles et perplexité FR générale en baisse continue.

## Résultats

### Perplexité

| Jeu | v8 | v9 |
|---|---|---|
| Histoires `validation.jsonl` (bits/octet) | 0,244 | 0,282 |

| Domaine (v9) | Loss | Perplexité | Bits/octet |
|---|---|---|---|
| fw2hq | 2,798 | 16,40 | 0,909 |
| wiki_fr | 2,578 | 13,17 | 0,866 |
| vikidia | 2,601 | 13,48 | 0,855 |
| lecarnet | 0,907 | 2,48 | 0,289 |

Loss de validation mélangée de la v9 pendant l'entraînement : 2,974 (pas 17 000), 2,816 (45 000), 2,660 (67 000), 2,598 (76 000), 2,545 (91 500) ; baisse continue, plateau sur les 2 500 derniers pas.

### Histoires (jeu scellé holdout3)

| Modèle | Majeures | Limites | Mineures | OK | Erreurs certaines | Comptage strict |
|---|---|---|---|---|---|---|
| v8 | 19 | 11 | 14 | 16 | 32 % | 50 % |
| v9 | 29 | 9 | 15 | 7 | 48 % | 63 % |

### FrenchBench (lm-eval 0.4.13, 0-shot, `eval_frenchbench.py`)

Précision (acc ; acc_norm entre parenthèses quand disponible). 0-shot pour les deux modèles, la v8 n'ayant que 512 tokens de contexte. Tâches génératives exclues (score attendu proche de zéro pour un modèle de base de cette taille), wikitext_fr exclu (jeu de données non chargeable avec datasets 5.x).

| Tâche | Exemples | Hasard | v8 | v9 |
|---|---|---|---|---|
| grammar | 119 | 25 % | 52,1 % | 73,9 % |
| vocab | 119 | 25 % | 41,2 % | 68,9 % |
| reading_comp | 71 | 25 % | 25,4 % | 47,9 % |
| hellaswag | 9 338 | 25 % | 25,7 % (25,6 %) | 29,0 % (31,7 %) |
| arc_challenge | 1 169 | 25 % | 20,8 % (24,5 %) | 20,1 % (25,1 %) |
| topic_based_nli | 600 | 33 % | 33,3 % | 36,2 % |
| xnli | 5 010 | 33 % | 33,3 % | 33,3 % |
| boolqa | 178 | 50 % | 45,5 % | 50,0 % |
| fquadv2_bool | 800 | 50 % | 49,8 % | 50,0 % |

- La v9 dépasse nettement la v8 et le hasard en grammaire, vocabulaire et compréhension de texte, et légèrement sur hellaswag.
- Raisonnement (arc_challenge), inférence (xnli, topic NLI) et questions oui/non : niveau du hasard pour les deux modèles. Les scores exacts de 33,3 % et 50,0 % indiquent une réponse constante, sans signal.
- Marges : environ ±8 points sur grammar/vocab, ±11 points sur reading_comp, ±1 point sur hellaswag.

## Conclusion
- Gate non atteint : la v9 est moins bonne que la v8 sur les histoires nouvelles (48 % d'erreurs certaines contre 32 %) et sur la perplexité des histoires (0,282 contre 0,244 bit/octet). Le second critère (perplexité générale en baisse continue) est rempli.
- v8 reste le modèle par défaut pour les histoires.
- En revanche, la v9 maîtrise nettement mieux le français général (FrenchBench : grammaire 74 % contre 52 %, vocabulaire 69 % contre 41 %, compréhension 48 % contre 25 %). Le raisonnement et l'inférence restent au niveau du hasard pour les deux modèles.
- Défauts observés chez la v9 : changements de prénom et confusions de personnages plus fréquents, et dérive vers un registre encyclopédique sur certains prompts (Pataud et Coin-Coin : texte de type Wikipédia avec section « Notes »), conséquence du corpus mélangé où les histoires ne pèsent que 15 %.
- Le jeu holdout3 semble plus difficile que holdout2 (v8 : 32 % contre 17 % d'erreurs certaines) ; seule la comparaison au sein d'une même lecture est significative.

## Limites
- Un seul évaluateur (Claude), grille fixe mais jugement subjectif.
- 60 textes par modèle : marge d'environ ±6 points sur les pourcentages ; l'écart observé (16 points en erreurs certaines) dépasse cette marge.
- FrenchBench limité aux tâches à choix multiples en 0-shot ; pas de comparaison avec des modèles publiés (protocoles différents).
