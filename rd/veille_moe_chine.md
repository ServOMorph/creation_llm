# Veille MoE et techniques chinoises sous contrainte matérielle — cellule R&D creation_llm

Date : 2026-11-17. Pôle : veille web (sujet 1 : mélange d'experts ; sujet 2 : apports chinois). Aucun calcul GPU lancé. Aucun fichier existant modifié. Les pistes déjà couvertes par `rd/veille_entrainement.md` et `rd/veille_inference.md` (Muon, ResFormer, tri de corpus, FP8, MTP, CFG, RAFT, etc.) ne sont reprises que par renvoi.

## Conventions

- **ÉTABLI** : publié et reproduit (reproduction indépendante vue dans la veille). Quand la reproduction n'est pas retrouvée : « ÉTABLI (publié, reproduction non retrouvée) ».
- **PRÉPUBLICATION** : arXiv ou rapport technique sans reproduction indépendante vue.
- **HYPOTHÈSE** : raisonnement de la cellule. Les coûts VRAM et temps sont des calculs arithmétiques, pas des mesures.
- « Lu » : chiffre lu dans la page primaire ouverte pendant cette veille. « Résumé seulement » : la page n'a donné que le résumé ; les tableaux n'ont pas été lus. « Non lu » : aucun chiffre obtenu.
- Limite de la veille : une limite d'usage a interrompu les recherches pendant une partie de la session ; les pages ont été reprises ensuite. Aucun PDF n'a été lu intégralement.

## Constats préalables sur le projet (lus dans le code)

- `train.py` ligne 87 : Muon est déjà appelé avec `adjust_lr_fn="match_rms_adamw"`, la mise à l'échelle des mises à jour attribuée à Moonlight (voir sujet 2). Ce point est donc déjà en place.
- `run_369m.py` : 24 couches, 1 024 de largeur, 16 têtes, `--tie-embeddings`, vocabulaire 32 768, lot 2, accumulation 32, soit 65 536 tokens par pas d'optimisation.
- Nom et taille : le nom « 369M » correspond au décompte sans liaison des embeddings (24 × 12 × 1024² + 2 × 33,5M). Avec `--tie-embeddings`, le décompte est d'environ 336M, qui est le chiffre demandé. Les deux chiffres coexistent dans les fichiers ; ce document utilise 336M.
- Constante VRAM : 5,8 Go de partie fixe pour 336M, soit environ 17,3 octets par paramètre stocké (5,8 × 10⁹ / 336 × 10⁶). Cette proportionnalité est une HYPOTHÈSE (linéarité supposée) ; elle sert à estimer les autres configurations.
- Incohérence à noter : le pic de 5,57 Go et la partie fixe de 5,8 Go sont donnés ensemble par la demande ; le `bilan_assistant_v2.md` mesure 6,04 Go de pic à lot 2 avec recalcul. Les deux ordres de grandeur sont retenus.

---

## Sujet 1 — Mélange d'experts (MoE) sur une RTX 4060 8 Go

### 1.1 État de l'art consulté (2024-2026)

| Famille | Source | Ce qui est lu | Statut |
|---|---|---|---|
| MoE clairsemé, experts fins et experts partagés (DeepSeekMoE) | https://arxiv.org/abs/2401.06066 (lu) | 2B : comparable à GShard 2,9B (1,5 fois les paramètres d'experts) et proche d'un dense de même total. 16B : comparable à LLaMA2 7B pour environ 40 % du calcul. 145B préliminaire : comparable à DeepSeek 67B pour 28,5 % (ou 18,2 %) du calcul. | PRÉPUBLICATION (chiffres des auteurs, reproduction indépendante non vue) |
| Équilibrage sans perte auxiliaire | https://arxiv.org/abs/2408.15664 (lu, résumé) | Biais d'expert ajouté au score de routage, mis à jour selon la charge. Modèles jusqu'à 3B et 200B tokens. Aucun chiffre de perplexité lu. | PRÉPUBLICATION |
| MoE à très petite échelle | https://arxiv.org/abs/2409.02060 (OLMoE, lu, résumé) | 7B total, environ 1B actif, 5T tokens. Affirme dépasser les modèles de même paramètres actifs. Pas de chiffre numérique dans le résumé. | PRÉPUBLICATION |
| Comparaison à budget égal, moins de 1B | https://arxiv.org/abs/2503.00245 (CoSMoEs, résumé) | Affirme qu'à l'échelle embarquée, les MoE dépassent les denses alignés sur le calcul, à mémoire et latence maîtrisées. Tailles et tableaux non lus. | PRÉPUBLICATION (résumé seulement) |
| Comparaison à budget égal, grande échelle | https://arxiv.org/abs/2506.12119 (lu, résumé) | Même total de paramètres, même calcul, même données : le MoE dépasse le dense sous une zone d'activation optimale. Environ 200 modèles à 2B et plus de 50 à 7B, 50 billions de tokens cumulés. Aucun taux d'activation numérique lu. ICLR 2026 selon la page. | PRÉPUBLICATION (2B et 7B seulement) |
| Lois d'échelle de parcimonie | https://arxiv.org/abs/2501.12370 (lu, résumé) | Existence d'une parcimonie optimale ; aucune valeur numérique lue. | PRÉPUBLICATION |
| Upcycling dense vers MoE | https://arxiv.org/abs/2410.07524 (lu, résumé) | Nemotron-4 15B : 67,6 % MMLU avec upcycling contre 65,3 % en poursuite dense, mêmes 1T tokens (+2,3 points). Initialisation « virtual group ». | PRÉPUBLICATION (échelle 15B) |
| MoE de couches partagées (MoEUT) | https://arxiv.org/abs/2405.16039 (lu, résumé ; accepté NeurIPS 2024 selon la page) | Couches partagées avec MoE dans l'attention et le MLP ; « nettement moins de calcul et de mémoire » sans chiffre lu. | PRÉPUBLICATION (chiffres non lus) |
| MoE d'adaptateurs | https://arxiv.org/abs/2404.13628 (MoLE, lu, résumé) | Fusion de LoRA par routage couche par couche avec perte d'équilibrage. Aucun chiffre lu. | PRÉPUBLICATION |
| Mélange de têtes d'attention | https://arxiv.org/abs/2410.11842 (MoH, lu, résumé ; ICML 2025 selon la page) | 50 à 90 % des têtes utilisées ; LLaMA3-8B : +2,4 points de précision moyenne sur 14 tâches à 75 % des têtes. Tailles petites non lues. | PRÉPUBLICATION |
| MoE de référence, équilibrage au niveau du lot | https://arxiv.org/abs/2502.16982 (Moonlight, lu) | 3B/16B MoE, 5,7T tokens, Muon environ 2 fois plus efficace qu'AdamW en calcul (lois d'échelle). | PRÉPUBLICATION |

Recherche de références à moins de 1 milliard de paramètres : une recherche a été lancée. Le résultat le plus proche est CoSMoEs (ci-dessus, tableaux non lus). Ling (Ant Group, https://arxiv.org/abs/2507.17702, repéré mais non ouvert) compare un MoE de 0,85B actif à un dense de 6,1B : hors de la tranche, et non lu. Aucune comparaison publiée, lue, à budget égal, en dessous de 1B et sur données françaises n'a été trouvée. **Il n'existe pas, à ma connaissance de cette veille, de résultat publié à budget égal sous 1B lu dans la source primaire.**

### 1.2 Question centrale : la VRAM, pas le calcul

Hypothèse de calcul : une partie fixe de 17,3 octets par paramètre stocké (5,8 Go / 336M). Le débit et le calcul par token sont évalués séparément.

Configurations (architecture de référence : 24 couches, largeur 1 024, attention 4,19M par couche, embeddings liés 33,5M ; MLP dense à 8,39M par couche) :

| Config | Params totaux | Params actifs / token | Partie fixe entraînement (17,3 o/param) | Tient dans 8 Go ? | Débit attendu |
|---|---|---|---|---|---|
| Dense 336M (référence) | 336M | 336M | 5,8 Go (mesuré) | Oui (pic 5,57 à 6,04 Go selon la source) | 6 100 à 7 000 tok/s (mesuré) |
| MoE-A : 8 experts de largeur 2 048, top-2 | 939M | 335M | 16,2 Go | Non, même avec activations recalculées | Calcul identique au dense ; limité par le transfert (voir 1.3) |
| MoE-B : 16 experts de largeur 1 024, top-2 | 939M | 235M | 16,2 Go | Non | Calcul inférieur au dense (environ 0,7 fois le MLP) ; qualité à budget égal non mesurée |
| MoE-T (témoin) : 12 couches, largeur 512, 8 experts de largeur 1 024, top-2 | 130M | 54,5M | 2,3 Go | Oui, large marge | Calcul identique au dense témoin (54,5M) |

Calculs : MoE-A : attention 100,7M + MLP 8 × 4,19M × 24 = 805M + embeddings 33,5M = 939M ; actif = 100,7M + 2 × 4,19M × 24 = 201M + 33,5M = 335M. MoE-T : attention 37,7M ; MLP 8 × 1,05M × 12 = 100,7M ; embeddings 16,8M ; total 130M ; actif 37,7M + 25,2M + 16,8M = 54,5M (le dense témoin de même largeur fait 54,5M).

**Verdict arithmétique** : tout MoE qui augmente les paramètres stockés au-delà d'environ 400M dépasse les 8 Go pour l'entraînement avec l'optimiseur actuel (Muon + AdamW, états en fp32). Le surplus de capacité ne peut pas être gardé sur le GPU. C'est une perte sur ce matériel, sauf contournement (1.3).

### 1.3 Contournements examinés

1. **Experts déchargés en RAM (48 Go).** Les paramètres d'experts (805M pour MoE-A) iraient en RAM, avec gradients et optimiseur. Un micro-lot de 2 048 tokens touche presque tous les experts (environ 512 tokens par expert), donc tous doivent être présents sur le GPU pendant le micro-lot. Transfert par micro-lot : 1,6 Go en bf16 aller et retour, 32 micro-lots par pas. Ordre de grandeur : 3 transferts de 1,6 Go par micro-lot, soit environ 5 Go par micro-lot. Bande passante PCIe effective pour la RTX 4060 : HYPOTHÈSE de 12 à 14 Go/s (lien x8 : non sourcé, à vérifier sur la machine). Le transfert seul coûte donc 0,35 à 0,4 s par micro-lot, comparable au temps de calcul du micro-lot (2 048 / 6 100 ≈ 0,34 s). **Débit estimé : 2 500 à 3 500 tok/s (HYPOTHÈSE), soit la moitié du dense.** L'optimiseur (AdamW sur 805M paramètres sur le CPU, un pas toutes les 65 536 tokens) : durée non mesurée, HYPOTHÈSE de plusieurs secondes par pas.
2. **Entraînement expert par expert.** Ne réduit pas la VRAM si la donnée est mélangée : chaque pas touche tous les experts. Le principe n'est pertinent que si le routage est hors ligne (cas des experts figés ou par domaine), voir 1.4.
3. **Experts gelés.** Les experts gelés ne stockent que 2 octets par paramètre (poids bf16) : 805M × 2 = 1,6 Go. Avec 2,3 Go de partie entraînée (cas MoE-T ou la partie non MoE du MoE-A : 134M × 17,3 ≈ 2,3 Go), la configuration tient dans 8 Go. Mais des experts gelés ne sont utiles qu'appris avant : c'est l'upcycling (1.4).
4. **États d'optimiseur allégés.** Muon garde un seul état par paramètre (quantité de mouvement) au lieu de deux pour Adam ; le passage de ces états en bf16 est la piste 9 de `veille_entrainement.md` (risque élevé sur les poids, moins sur les états). Gain sur MoE-A : environ 1,3 Go (HYPOTHÈSE), insuffisant pour 16,2 Go.

### 1.4 Ce qui est plausible sur ce matériel

- Un **témoin MoE-T (130M total, 54,5M actif)** tient largement dans la VRAM : 2,3 Go de partie fixe. Il permet de tester la question à budget égal sans risque. Coût : temps GPU, sous jeton VertIA.
- **Upcycling** (2410.07524) du témoin dense vers MoE-T : copier le MLP dense dans les 8 experts, bruiter, poursuivre l'entraînement. Gain publié : +2,3 MMLU à 15B (PRÉPUBLICATION) ; à 54M : HYPOTHÈSE.
- Pour le 336M : l'upcycling n'est possible qu'à partir d'un point de contrôle dense et exige de passer par MoE-A (939M) : écarté sur la VRAM.

### 1.5 Réponse à la question : gain ou perte ?

- **Perte** pour un pré-entraînement de 14 jours à 336M avec le recette actuelle : le surplus de paramètres ne tient pas dans 8 Go, et le contournement en RAM réduit le débit de moitié (HYPOTHÈSE) sans garantie de gain de qualité à budget égal sous 1B (aucune source lue).
- **Gain possible** dans les cas où la mémoire n'est pas la contrainte : SFT (1.6), ou un témoin MoE-T qui vérifie l'effet à budget égal avant toute décision.
- Points à ne pas confondre : la comparaison à budget égal (2506.12119) fixe le total de paramètres et le calcul ; sur ce matériel, la contrainte est la VRAM de stockage. Les deux ne coïncident pas, et les sources ne traitent pas le cas 8 Go.

### 1.6 Adaptateurs experts pour le SFT

MoLE (2404.13628) et les LoRA-MoE : quelques dizaines de millions de paramètres d'adaptateurs, poids de base gelés. Coût VRAM : faible. Gain chiffré : non lu. Pertinent seulement après la SFT du 336M (sans gain de pré-entraînement).

---

## Sujet 2 — Apports chinois (vérification de l'hypothèse « avance »)

### 2.1 Ce qui est confirmé, et sur quels points

| Affirmation | Source lue | Ce qu'elle établit | Ce qu'elle n'établit pas |
|---|---|---|---|
| MoE efficace à grande échelle (DeepSeek) | https://arxiv.org/abs/2401.06066 (lu) | Les chiffres de calcul économisé à 2B, 16B, 145B | Reproduction indépendante ; comparaison à 336M |
| Échelle et coût d'entraînement DeepSeek-V3 | https://arxiv.org/abs/2412.19437 (lu) | 671B total, 37B actifs, 14,8T tokens, 2,788M heures H800 | Les chiffres FP8, MTP et MLA ne figurent pas dans la page lue ; le rapport complet n'a pas été lu |
| Muon à l'échelle (Moonshot) | https://arxiv.org/abs/2502.16982 (lu) | Efficacité de calcul d'environ 2 fois sur AdamW, MoE 3B/16B, 5,7T tokens | Reproduction hors équipe non vue dans la veille ; l'affiliation n'a pas été lue sur la page |
| Planification WSD (OpenBMB) | https://arxiv.org/abs/2404.06395 (lu) | Modèles 1,2B et 2,4B non-embeddings ; scheduler Warmup-Stable-Decay | « Égalité avec les modèles de 7B à 13B » : affirmation des auteurs, aucun tableau lu |
| Équilibrage sans perte auxiliaire (DeepSeek) | https://arxiv.org/abs/2408.15664 (résumé) | Principe et échelle (3B, 200B tokens) | Gain chiffré non lu |

Constat : les équipes chinoises publient des résultats de **calcul économisé à grande échelle** (DeepSeekMoE, DeepSeek-V3, Moonlight) et des variantes d'optimiseur ou de planification. Ces résultats sont bien documentés. Ils ne sont pas transposables tels quels à 336M.

### 2.2 Ce qui n'est pas confirmé

- Aucune comparaison lue, en langue chinoise ou anglaise, entre une équipe chinoise et une équipe non chinoise **à budget égal sous 1B**, sur données françaises ou sur un matériel de type 8 Go.
- Les meilleurs résultats MoE à budget égal lus (2506.12119) viennent d'une équipe dont les affiliations n'ont pas été lues sur la page ; l'attribution géographique n'est donc pas établie ici.
- OLMoE (AI2, https://arxiv.org/abs/2409.02060) et MoEUT (https://arxiv.org/abs/2405.16039) sont des travaux non chinois qui traitent le même sujet à petite échelle, et MoLE (Microsoft Research Asia et Tsinghua selon la recherche) est une contribution mixte. Aucune source lue ne soutient que les équipes chinoises ont l'avance sur les petits modèles.
- Muon à 2 fois l'efficacité (Moonlight) : chiffre des auteurs ; le speedrun (modded-nanogpt, voir `veille_entrainement.md`) utilise Muon mais n'est pas une reproduction de ce chiffre.

### 2.3 Ce qui relève de la réputation, non d'un résultat mesuré

- « Les équipes chinoises sont en avance » : non mesurable avec les sources lues. Ce qui est mesuré concerne des modèles de 3B à 671B, et l'avance observée tient à l'ingénierie d'entraînement à grande échelle (MoE, FP8, cluster), pas à l'efficacité à 336M.
- Les modèles petits de MiniCPM (1,2B et 2,4B) sont présentés comme comparables à des modèles de 7B à 13B : affirmation des auteurs, non vérifiée par tableau.

### 2.4 Transposable à 336M, 8 Go

Retenus : WSD (scheduler), `match_rms_adamw` (déjà en place), Muon (déjà en place), MoH (mélange de têtes), MoEUT (couches partagées), MoE-T de test.

Écartés : DeepSeek-V3 (échelle), FP8 à grande échelle (déjà piste 15 de `veille_entrainement.md`), MTP (prédiction multi-tokens : déjà piste 8 de la même veille, dans le cadre de la TOP), équilibrage sans perte auxiliaire (seulement si un MoE est retenu), modèles de l'OpenBMB au-delà de 2,4B.

---

## 3. Pistes classées (gain attendu / coût sur ce matériel)

Classement : du meilleur rapport au plus faible. Chaque piste donne mécanisme, gain, coût, expérience, critère d'échec, risque, sources, statut.

### Piste 1 — Planification WSD (warmup-stable-decay) [B/C]
- **Mécanisme.** Phase de palier à lr constant, puis décroissance courte à la fin, au lieu d'une décroissance cosinus sur tout le run. Un run peut s'interrompre à tout moment et reprendre sans que la fin soit recalculée.
- **Gain publié.** Affirmation des auteurs (MiniCPM, 2404.06395) : performance de 1,2B et 2,4B comparable à des modèles de 7B à 13B ; aucun chiffre lu dans le tableau. HYPOTHÈSE : pas de gain de perte finale, mais meilleure compatibilité avec les pauses VertIA (contrôle de reprise).
- **Coût.** Nul en VRAM et en temps. Changer le planificateur en cours de run n'est pas neutre : à faire seulement sur un nouveau run, ou en SFT.
- **Expérience minimale.** Témoin 12 × 512 (54,5M actifs), 3 graines : cosinus contre WSD, même budget de tokens.
- **Critère d'échec.** Écart de perte de validation inférieur à l'écart-type entre graines.
- **Risque.** Faible. Réglage de la fraction de décroissance non sourcé.
- **Statut.** PRÉPUBLICATION (résumé lu, tableau non lu).
- **Sources.** https://arxiv.org/abs/2404.06395

### Piste 2 — Témoin MoE-T et upcycling (test à budget égal) [B]
- **Mécanisme.** 130M total, 54,5M actif ; 8 experts de largeur 1 024, top-2. Comparé au dense témoin 54,5M à calcul égal. Variante : upcycling à partir du dense témoin entraîné (copie et bruit).
- **Gain publié.** DeepSeekMoE, MoE comparable à un dense plus gros pour moins de calcul (chiffres à grande échelle, lus). Upcycling : +2,3 MMLU à 15B (2410.07524). Gain à 54M : HYPOTHÈSE, 0 à 0,03 nat de perte de validation.
- **Coût.** VRAM 2,3 Go de partie fixe ; temps : un témoin court. Un bras est mesuré sur la RTX 4060 sous jeton VertIA.
- **Expérience minimale.** MoE-T contre dense témoin, 3 graines, même nombre de tokens, débit mesuré. Puis upcycling.
- **Critère d'échec.** Gain inférieur à l'écart-type entre graines, ou débit inférieur à 70 % du dense : pas de MoE pour cette échelle.
- **Risque.** Un témoin à 54M ne prouve rien à 336M ; le routage top-2 n'est pas optimisé (les lois de parcimonie n'ont pas de valeurs lues).
- **Statut.** PRÉPUBLICATION (DeepSeekMoE, upcycling, 2506.12119 à 2B et 7B).
- **Sources.** https://arxiv.org/abs/2401.06066 ; https://arxiv.org/abs/2410.07524 ; https://arxiv.org/abs/2506.12119

### Piste 3 — Mélange de têtes d'attention (MoH) [C]
- **Mécanisme.** Chaque token choisit un sous-ensemble de têtes d'attention ; les sorties sont combinées par somme pondérée. Pas de paramètres ajoutés.
- **Gain publié.** LLaMA3-8B : +2,4 points de précision moyenne sur 14 tâches en utilisant 75 % des têtes (ICML 2025 selon la page ; résultats LLM lus dans le résumé). Gain à 336M : HYPOTHÈSE.
- **Coût.** Aucun paramètre ; code d'attention à réécrire, incompatible avec le checkpoint actuel. Calcul économisé en partie, VRAM d'activations possiblement réduite (non mesuré).
- **Expérience minimale.** Témoin 12 × 512, MoH contre attention standard, 3 graines.
- **Critère d'échec.** Perte inférieure à l'écart-type ou débit inférieur au dense.
- **Risque.** Les résultats de la source sont en vision et en LLM d'échelle 8B ; code non vérifié sous `torch.compile`.
- **Statut.** PRÉPUBLICATION.
- **Source.** https://arxiv.org/abs/2410.11842

### Piste 4 — Couches partagées (MoEUT) pour réduire les paramètres stockés [C]
- **Mécanisme.** Un jeu de couches réutilisé en boucle, avec MoE dans l'attention et le MLP. Moins de paramètres stockés pour une profondeur de calcul donnée.
- **Gain publié.** « Nettement moins de calcul et de mémoire » avec perplexité légèrement meilleure que le standard (résumé lu, chiffres non lus). Gain à 336M : HYPOTHÈSE, potentiellement le plus fort levier VRAM de cette veille, car il réduit directement les 17,3 octets par paramètre.
- **Coût.** Nouveau pré-entraînement (architecture changée). Incompatible avec le run 336M en cours.
- **Expérience minimale.** Témoin : même budget de paramètres stockés, MoEUT contre dense ; mesure de la perte et du pic VRAM.
- **Critère d'échec.** Perte supérieure de plus de 0,02 nat au dense à même VRAM, ou débit inférieur à 70 %.
- **Risque.** Chiffres non lus ; la profondeur effective est réduite par le partage.
- **Statut.** PRÉPUBLICATION.
- **Source.** https://arxiv.org/abs/2405.16039

### Piste 5 — Adaptateurs experts pour la SFT (MoLE, LoRA-MoE) [B]
- **Mécanisme.** Des adaptateurs LoRA par domaine, routés couche par couche, sur un modèle gelé.
- **Gain publié.** MoLE dépasse la fusion arithmétique de LoRA (résumé lu) ; aucun chiffre lu.
- **Coût.** Faible VRAM (quelques dizaines de millions de paramètres d'adaptateurs). Pertinent pour la SFT seulement.
- **Expérience minimale.** SFT v2 avec un seul LoRA contre MoLE à 4 experts ; même données.
- **Critère d'échec.** Gain inférieur à 2 points sur le benchmark d'assistant.
- **Risque.** Complexité pour un gain non chiffré ; le benchmark est petit (300 cas).
- **Statut.** PRÉPUBLICATION.
- **Source.** https://arxiv.org/abs/2404.13628

### Piste 6 — Équilibrage de charge sans perte auxiliaire [C, conditionnelle]
- **Mécanisme.** Biais d'expert ajouté au score de routage, mis à jour selon la charge, sans terme de perte dans le gradient.
- **Gain publié.** Résumé lu : meilleure performance et équilibre que la perte auxiliaire ; chiffres non lus.
- **Coût.** Nul en VRAM. Utile seulement si une piste MoE est retenue (pistes 2 et 4).
- **Expérience minimale.** MoE-T avec et sans biais, 3 graines, mesure de la fraction de tokens par expert.
- **Critère d'échec.** Déséquilibre (écart max/moyen) non réduit par rapport à la perte auxiliaire.
- **Risque.** Effet sur un petit nombre d'experts non sourcé.
- **Statut.** PRÉPUBLICATION.
- **Source.** https://arxiv.org/abs/2408.15664

### Piste 7 — Experts sur l'arrière-plan CPU (CoSMoEs, déchargement) [C, écartée pour le moment]
- **Mécanisme.** MoE à experts pondérés par décomposition, avec déchargement des experts en mémoire hôte.
- **Gain publié.** Résumé lu : MoE meilleur que les denses alignés sur le calcul à l'échelle embarquée ; efficacité de déchargement affirmée, non chiffrée dans le résumé.
- **Coût.** Élevé (voir 1.3).
- **Expérience minimale.** Non recommandée avant la piste 2.
- **Critère d'échec.** Débit inférieur à 50 % du dense.
- **Risque.** Tableaux non lus ; tailles inconnues.
- **Statut.** PRÉPUBLICATION (résumé seulement).
- **Source.** https://arxiv.org/abs/2503.00245

### Piste 8 — Muon et mise à l'échelle des mises à jour (Moonlight) [déjà en place]
- **Mécanisme.** `match_rms_adamw` et décroissance de poids dans Muon (Moonshot, 2502.16982).
- **Gain publié.** Environ 2 fois l'efficacité de calcul sur AdamW (lois d'échelle, auteurs).
- **Coût.** Nul : déjà appelé dans `train.py` ligne 87.
- **Statut.** PRÉPUBLICATION ; rien à faire pour le run en cours.
- **Source.** https://arxiv.org/abs/2502.16982

### Piste 9 — Renvois vers `veille_entrainement.md` et `veille_inference.md` (non repris ici)
- FP8 (piste 15 entraînement), prédiction multi-tokens et TOP (piste 8), sortie anticipée (inférence piste 13), tri de corpus (piste 4), mémoire par bigrammes (piste 11). Les résultats chinois de DeepSeek (Engram, janvier 2026) sont déjà couverts par la piste 11 de `veille_entrainement.md`.

---

## 4. Pistes écartées (motivées)

| Piste | Motif | Source |
|---|---|---|
| MoE-A (939M total, 335M actif) sur 8 Go | Partie fixe 16,2 Go ; contournement RAM : débit estimé à la moitié du dense (HYPOTHÈSE) ; aucun gain à budget égal sous 1B lu. | https://arxiv.org/abs/2506.12119 (2B et 7B) |
| MoE-B (16 experts, top-2) | Même stockage que MoE-A, calcul inférieur : gain de débit possible mais pas de gain de qualité démontré. | — |
| DeepSeek-V3 à grande échelle (671B, 37B actifs) | Échelle sans transposition ; chiffres FP8, MTP et MLA non lus dans la page. | https://arxiv.org/abs/2412.19437 |
| Équilibrage au niveau de l'expert en dehors du MoE | Sans objet sans MoE. | — |
| Experts gelés sans upcycling | Des experts gelés non entraînés n'ajoutent pas de capacité. | — |
| Modèles de la gamme OpenBMB au-delà de 2,4B | Hors budget VRAM. | https://arxiv.org/abs/2404.06395 |

---

## 5. Non vérifié

- Aucun chiffre de tableau de DeepSeekMoE, MoEUT, MoLE, MoH (au-delà du résumé), CoSMoEs, OLMoE, Moonlight, sans perte auxiliaire et WSD n'a été lu : seuls les résumés et les chiffres cités dans les pages ont été repris.
- Affiliations non lues pour la plupart des sources ; l'attribution géographique n'est pas établie hors DeepSeek (auteurs listés) et Moonshot (auteurs listés, affiliation non relue).
- Bande passante PCIe effective de la RTX 4060 (lien x8) : non sourcé, à mesurer.
- Débit de transfert et durée de l'optimiseur CPU sur MoE-A : HYPOTHÈSE, non mesuré.
- Partie fixe de 17,3 octets par paramètre : proportionnalité supposée, non vérifiée pour une autre taille.
- Ling (2507.17702) et MobileMoE (2605.27358) : repérés par la recherche, non ouverts.
- Aucun GPU utilisé ; aucun résultat expérimental dans ce document.
