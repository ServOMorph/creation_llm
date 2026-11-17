# Pôle mathématiques — creation_llm

Date : 2026-11-17. Périmètre : budget d'information, lois de perte, idées. Aucun GPU utilisé. Scripts et sorties : `rd/experiences/`.

Légende : ÉTABLI (publié et reproduit), PRÉPUBLICATION, HYPOTHÈSE, non sourcé.
Sources : aucune recherche web n'a été faite dans cette session. Les références citées de mémoire sont marquées « non vérifié en ligne ». Aucune URL n'est donnée.

## 1. Où part le budget (mesures)

| Mesure | Valeur | Source |
|---|---|---|
| Perte de validation finale v9 (192M, 91 500 pas, 6,0 Md tokens) | 2,5445 nats/token = 3,67 bits/token | `checkpoints_v9/log.csv` |
| Octets par token (mélange, tokenizer 32k) | 4,45 | `math_exp1_budget.json` |
| Bits par octet v9 (approx.) | 0,82 | calcul à partir des deux lignes ci-dessus |
| Référence n-gramme lissé (bigramme, val, 3 M tokens) | 5,23 nats/token | `math_exp1_budget.json` |
| Même tokens à 1,37 Md : v9 vs 369M | v9 2,9417 ; 369M 2,8495 (écart 0,092) | logs, pas 21 000 et 21 000 |
| Perte moyenne v9 sur 112 séquences val, par source | fw2hq 2,76 ; wiki_fr 2,69 ; vikidia 2,60 ; lecarnet 0,88 | `math_exp3_v9_cpu_A.json` |

Lecture :
- Il n'existe pas ici de borne théorique établie de l'entropie du français sur ce corpus. Les références ci-dessus sont des points de comparaison, pas des bornes.
- Le « surcoût préquentiel » de la v9 (0,17 bit/token) mesure l'écart à sa propre perte finale, pas à une borne. Il ne doit pas être lu comme une distance au optimum.

### Où se concentre la perte (v9, mesure CPU)

- Tokens de début de mot : 62 % des tokens, 78 % de la perte moyenne (perte 3,34 nats contre 1,83 pour les suites de mots).
- 19 % des tokens ont une probabilité prédite inférieure à 0,01 pour la bonne cible. Ils représentent 55 % de la perte. Les tokens prédits à plus de 0,9 pèsent 0,1 % de la perte.
- Position dans la fenêtre : 4,4 nats pour les 4 premiers tokens, 3,4 pour les positions 4 à 16, 2,4 au-delà de 64.
- Les 16 premières positions pèsent environ 2,3 % de la perte totale. C'est la borne haute du gain d'un découpage aligné sur les documents (idée 4).

Chiffres ci-dessus : ÉTABLI au sens « mesuré sur ce checkpoint » (échantillon de 112 séquences, erreur-type entre séquences d'environ 0,05 nat).

## 2. Budget de données du 369M (mesure)

Couverture simulée par tirage avec remise (`math_exp1_budget.json`), concordante avec exp(-e) :

| Source | Tokens | Passes équivalentes sur 113 000 pas | Jamais vu | Vu 2 fois ou plus |
|---|---|---|---|---|
| fw2hq | 4,33 Md | 0,99 | 37 % | 26 % |
| wiki_fr | 1,89 Md | 0,98 | 38 % | 26 % |
| lecarnet | 0,37 Md | 3,0 | 5 % | 80 % |
| vikidia | 0,054 Md | 2,7 | 6 % | 75 % |

Les deux petites sources sont vues environ trois fois. Le 369M mémorise donc davantage ces sources que la v9. Le risque est réel mais non mesuré (HYPOTHÈSE).

## 3. Perte = progrès + plancher lié au taux d'apprentissage

Mesure : la perte de validation de la v9 est ajustée par `L = a + b·x^(-α) + κ·(lr/lr0)^p`, où `x` est le nombre de pas ou le temps intrinsèque (somme des lr). Le taux suit le cosinus du script (vérifié contre le log : lr 0,000305 au pas 89 000, calcul 0,000309).

Résultats clés (`math_exp2_*.json`, `math_exp2c_racine_lr.json`) :
- Un modèle sans plancher (M0) ne prédit pas la fin : erreur +0,18 à +0,27 selon l'ajustement.
- Un plancher linéaire (p = 1) ou un plancher en racine (p = 0,5) réduit l'erreur. Avec p = 0,5, l'ajustement sur 60 % du run donne une erreur finale de +0,018 sur la v9 (2,5445 réel). Avec p = 1, l'erreur est +0,096. Le choix de p a été fait sur 6 valeurs, donc l'erreur est légèrement optimiste.
- Le 369M prédit une perte finale d'environ 2,44 avec les α et κ de la v9 et p = 0,5. Une variation de κ de ±30 % donne 2,33 à 2,56.
- Le seuil de la roadmap (2,5445) est donc franchi dans la plupart des scénarios, mais pas de façon robuste. La marge dépend de κ, que le 369M ne fixe qu'à partir de sa fin de run.

Statut : mesure sur deux runs ; forme du plancher non publiée à ma connaissance, non vérifié en ligne. Statut de l'idée : HYPOTHÈSE.

## 4. Idées

Classement par gain attendu / coût. Chaque idée : mécanisme, gain, coût, expérience, critère d'échec, sources.

### I1. Décision d'arrêt ou de poursuite par extrapolation à 40 % du run (faible coût, gain d'opportunité)
- Mécanisme : ajuster `a + b·x^(-α) + κ·√(lr/lr0)` sur les premiers 40 à 60 % du run, puis décider de la poursuite.
- Gain : pas un gain de qualité. Permet de décider tôt si le run atteindra le gate, donc d'économiser des jours de GPU. Sur la v9, l'erreur hors échantillon à 60 % est de 0,018 (forme en racine).
- Coût : nul (CPU, quelques secondes).
- Expérience (faite) : backtest sur la v9 (`math_exp2c`). Expérience suivante : refaire le backtest à 40 % et à 60 % sur le 369M dès qu'il dépasse 60 000 pas.
- Critère d'échec : erreur finale supérieure à 0,05 sur le 369M à 60 % du run.
- Risque : la forme n'est validée que sur un run. Statut : PRÉPUBLICATION non vérifiée, HYPOTHÈSE sur la forme.

### I2. Plancher de lr : refroidissement plus long ou décroissance vers zéro (gain estimé 0,05 à 0,18 nat)
- Mécanisme : le terme κ·√lr est la part de perte attribuable au lr final non nul. Le 369M a κ ≈ 0,57 et un lr final de 0,1·lr0.
- Gain estimé : `κ·√(0,1) ≈ 0,18` nat au point final dans la forme en racine ; 0,047 dans la forme linéaire. L'écart entre formes est lui-même une incertitude.
- Coût : aucun coût de calcul supplémentaire, seulement un changement de schedule sur le run (pas de run neuf).
- Expérience : sur la v9 (run terminé), le schedule ne peut pas être changé. Test proposé : un run court de 30 min GPU, même lr de pic, deux schedules (cosinus à 0,1 et décroissance à 0,01), comparés à même nombre de tokens. Nécessite accord GPU.
- Critère d'échec : écart inférieur à l'écart-type du bruit de validation (0,005 mesuré sur les 20 derniers points du 369M).
- Statut : décroissance de lr en fin de run est une pratique connue (non sourcé en ligne). La quantification par le plancher est HYPOTHÈSE.

### I3. Taille de lot par rapport à l'échelle de bruit critique B* (non concluante)
- Mécanisme : sous B*, doubler le lot réduit le nombre de pas ; au-dessus, le gain par token disparaît. Le lot du 369M est 65 536 tokens.
- Mesure tentée : B* par U-statistique sur 8 groupes de 4 séquences (`math_exp3_v9_cpu_B.json`). La norme vraie estimée est 0,0007 avec une erreur-type de 0,0069. L'estimation est dominée par le bruit ; B* n'est pas mesurée. Les intervalles sont non bornés.
- Conclusion : non concluant. Pour une estimation utilisable, il faut environ 100 fois plus de séquences (plusieurs heures de CPU). Hors du cadre de quelques minutes par script.
- Idée restante : si B* est de l'ordre de quelques millions de tokens, un lot de 16 k tokens avec quatre fois plus de pas serait plus efficace par token. Non testé. Gain et coût dépendent de B*, inconnu.
- Source : la notion de B* vient de McCandlish et al. 2018, non vérifié en ligne.

### I4. Découpage des fenêtres sur les frontières de documents (gain borné à 2 %)
- Mécanisme : les fenêtres aléatoires commencent au milieu d'un document. Les 16 premières positions coûtent plus cher.
- Gain borné : 2,3 % de la perte totale, calcul à partir de la mesure de position. C'est une borne haute, car l'alignement change aussi la distribution des données.
- Coût : nul. L'option `align_docs` existe déjà dans `train.py`.
- Expérience : comparer la perte de validation avec et sans alignement sur un run court. Non faite (GPU).
- Critère d'échec : gain inférieur au bruit de validation.
- Statut : ÉTABLI sur le plan de la mesure ; HYPOTHÈSE sur le gain réel.

### I5. Pénalité de répétition des petites sources (lecarnet et vikidia vus 3 fois)
- Mécanisme : une source répétée trois fois a une part de mémorisation. Réduire son poids dans le mélange, ou ajouter des données, réduit ce risque.
- Gain : non chiffré. La mesure de répétition est faite, pas le coût en perte.
- Expérience : sur un modèle plus petit (v9 ou 192M), entraîner avec et sans répétition des petites sources, comparer la perte sur un jeu de validation tenu à l'écart. Non faite (GPU).
- Critère d'échec : pas d'écart de perte sur validation.
- Statut : la répétition de données est un sujet publié (non vérifié en ligne). Statut de l'application : HYPOTHÈSE.

### I6. Tokenizer : compression en bits par octet comme critère (idée de cadrage)
- Mécanisme : la perte par token n'est pas comparable entre tokenizers. Le bon critère est le nombre de bits par octet. Le v9 est à 0,82 bit par octet sur ce corpus.
- Gain : non mesuré. Permet de comparer deux tokenizers sans biais.
- Expérience : calculer bits par octet pour deux tokenizers sur le même corpus de validation (faisable sur CPU, nécessite un second tokenizer à entraîner).
- Statut : méthode de comparaison standard (non sourcé en ligne). Pas de gain annoncé.

## 5. Les trois idées les plus solides

1. I1 (décision d'arrêt par extrapolation) : coût nul, déjà testé sur la v9, critère d'échec clair.
2. I2 (schedule de lr en fin de run) : gain potentiel le plus grand (0,05 à 0,18 nat), test GPU de 30 min à faire après accord.
3. I4 (alignement des documents) : gain borné et mesuré, réalisable avec l'option existante.

## 6. Non vérifié

- Aucune recherche web : la nouveauté de I1, I2 et I5 n'est pas contrôlée.
- La forme du plancher de lr n'est validée que sur deux runs ; le choix de p est partiellement optimisé sur la v9.
- B* (I3) n'est pas mesurée.
- Les gains de I2, I4, I5 ne sont pas testés sur GPU.
- Le 369M n'a été comparé à la v9 qu'à un seul point (1,37 Md tokens).
