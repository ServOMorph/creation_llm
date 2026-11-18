# NCM : Next-Concept Modeling — résultats, reproductions et intérêt pour le LLM français

*État documentaire au 18 novembre 2026. Aucun fichier modifié, aucun calcul GPU lancé.*

## Résumé en 10 lignes

1. Le sujet correspond à **Next-Concept Modeling (NCM)**, présenté par Hazy Research en octobre 2026.
2. L’équipe associe principalement **Stanford**, avec une affiliation **Together AI**.
3. NCM génère une hiérarchie de codes discrets, du général au détaillé, puis la décode en texte.
4. Les positions d’un même niveau sont produites en parallèle.
5. Le travail reste une **prépublication**, avec des reproductions indépendantes partielles.
6. Les accélérations annoncées concernent principalement des comparateurs à diffusion.
7. Le code d’évaluation et des checkpoints sont publics ; la recette officielle complète d’entraînement reste manquante.
8. Des essais sur GPU grand public confirment certains scores, mais révèlent des défauts d’implémentation.
9. Aucun résultat trouvé ne démontre une réduction des omissions de votre assistant français.
10. **Verdict : à surveiller ; un petit test d’inférence est justifié, un remplacement du pré-entraînement actuel ne l’est pas.**

## 1. Identification, auteurs et chronologie

### Quel « NCM » ?

| Candidat | Correspondance | Conclusion |
|---|---|---|
| **Next-Concept Modeling** | Stanford, nouveauté architecturale récente, génération parallèle hiérarchique | **Correspondance très forte** |
| **A Neural Conversational Model** | Sigle NCM également employé ; ancien article hébergé notamment sur un site pédagogique Stanford | Travail de **Vinyals et Le, Google, 2015** : homonyme |
| **Large Concept Models — LCM** | Nom voisin, génération dans un espace de représentations de phrases | Travail distinct de Meta |

L’homonyme de 2015 relève du dialogue séquence-à-séquence. LCM prédit des représentations de phrases dans l’espace SONAR. Ces deux pistes ne correspondent pas aussi bien à votre description. [Google Research, NCM 2015](https://research.google/pubs/a-neural-conversational-model/), [code officiel LCM](https://github.com/facebookresearch/large_concept_model/blob/main/README.md).

L’article recherché s’intitule **Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation**. Il est signé **Hermann Kumbong, Ishan S. Khare, Iddah Mlauzi, Andrew Woen, Charles Ding, E. Kelly Buchanan, Simran Arora, Azalia Mirhoseini et Christopher Ré**. Kumbong et Khare contribuent à égalité ; Arora, Mirhoseini et Ré partagent l’encadrement. [Annonce Hazy Research](https://hazyresearch.stanford.edu/blog/2026-10-12-ncm).

Les affiliations sont Stanford, sauf Simran Arora, indiquée chez Together AI ; Andrew Woen relève du département de génie électrique. Première version : **9 octobre 2026** ; annonce institutionnelle : **12 octobre**. Aucune acceptation en conférence vérifiée dans les sources consultées. [Article](https://arxiv.org/abs/2610.14162).

### Statuts employés

- **Établi, publié et reproduit** : publication scientifique et reproduction indépendante du résultat considéré.
- **Prépublication** : manuscrit public sans publication scientifique vérifiée.
- **Reproduction partielle** : exécution indépendante de checkpoints ou d’un sous-protocole.
- **Hypothèse locale** : transfert proposé au projet, sans résultat mesuré.

**NCM reste ici une prépublication avec reproductions partielles.** L’existence de ces reproductions renforce certaines observations, sans valider toute la recette ni toutes les comparaisons.

## 2. Rapport avec les recherches du dépôt

L’index consulté contient déjà **H-Net**, avec un verdict d’essai séparé après les priorités du cycle 2. Les cinq veilles ont été consultées : entraînement, inférence, MoE/Chine, frugalité et JEPA. Aucun dossier NCM n’apparaît dans l’index lu. `DOCUMENTATION/INDEX.md` est absent.

| Travail déjà étudié | Ce que NCM ajoute à la discussion |
|---|---|
| **H-Net** | Autre emploi d’une hiérarchie ; la question nouvelle est ici l’ordre de génération par niveaux |
| **SuperBPE et tokenisation** | Il faut évaluer une représentation apprise et sa fidélité, au-delà du nombre de tokens |
| **JEPA / STP** | Même intérêt pour les représentations ; le test final doit toujours porter sur les réponses effectivement décodées |
| **MTP/TOP et TST** | Autre façon d’organiser plusieurs prédictions ; leurs gains ne permettent pas d’estimer ceux de NCM |
| **CFG et vote entre réponses** | Techniques déjà étudiées : leur présence dans une recette NCM ne constitue pas une nouvelle preuve locale |
| **MoE et frugalité** | Même nécessité de compter tous les paramètres résidents et toutes les phases d’apprentissage |
| **Bits par octet** | Utile pour éviter les comparaisons trompeuses entre unités ; ne pas comparer directement une perte sur codes et votre perte BPE |

Le résultat local le plus pertinent demeure le passage de **49/90 à 71/90** avec des documents pertinents relus manuellement. Cela montre que `critere_absent` ne désigne pas uniquement une connaissance absente des poids. Une nouvelle architecture doit démontrer qu’elle améliore l’utilisation de l’information, sa conservation ou son acquisition. [Bilan du cycle 1](D:/ServOMorph/creation_llm/rd/bilan_cycle1.md).

## 3. Principe et mécanisme technique

### Deux apprentissages successifs

Le premier apprend un **tokenizer neuronal multirésolution** : chaque niveau dispose d’un dictionnaire de codes ; les niveaux fins complètent les précédents. Un abandon aléatoire des niveaux fins oblige les niveaux grossiers à rester utiles. Une perte de reconstruction exige de retrouver le texte.

Le second fige ce tokenizer et entraîne un Transformer à prédire chaque niveau à partir de la reconstruction cumulée des précédents. Une corruption des codes de conditionnement prépare le générateur aux erreurs rencontrées à l’inférence. [Présentation des auteurs](https://hazyresearch.stanford.edu/blog/2026-10-12-ncm).

Schéma de lecture :

```text
Apprentissage du tokenizer :
texte → représentation → codes multirésolutions → reconstruction du texte

Apprentissage du générateur :
reconstruction des niveaux précédents → codes du niveau suivant

Inférence :
niveau grossier → niveaux intermédiaires → niveau fin → texte
```

Il faut comprendre « concept » comme une **représentation apprise**. Rien n’autorise à identifier chaque code à un concept humain nommé, ni à supposer qu’un plan factuellement correct existe avant le décodage.

### Le rôle de l’attention

L’implémentation prévoit une attention par blocs. Dans le mode diagonal, les positions d’un niveau communiquent entre elles et peuvent consulter le préfixe ; le préfixe ne consulte pas les sorties à prédire. Les positions RoPE sont organisées par niveau. Les projections de sortie prédisent des identifiants de codes, avec plusieurs variantes de partage des têtes. [Implémentation `NextConceptTransformer`](https://github.com/HazyResearch/ncm/blob/main/models/next_concept.py).

**Conséquence pour votre code :** remplacer seulement le masque causal ne suffirait pas. Il faudrait aussi construire les cibles hiérarchiques, les dictionnaires, les entrées de reconstruction, les têtes adaptées et le décodeur de texte. Votre `train_sft.py` apprend actuellement des tokens assistant avec masquage des autres cibles : cette supervision ne devient pas automatiquement une supervision NCM.

### Pourquoi le parallélisme peut aider

**Analyse :** réduire les étapes séquentielles peut diminuer la latence lorsque chaque petit appel GPU coûte cher. Mais cela ne garantit ni moins d’opérations au total, ni moins de mémoire. Il faut compter la somme des positions traitées dans tous les niveaux, le traitement du préfixe et le décodage final.

Afficher un texte intermédiaire serait possible conceptuellement, mais garantir que les mots déjà affichés ne changeront plus demanderait un mécanisme supplémentaire. Ce point compte pour un assistant interactif.

## 4. Résultats chiffrés et budgets

### Résultats des auteurs

| Mesure | Résultat rapporté |
|---|---|
| LM1B, inférence | **21×** plus rapide que le comparateur diffusion principal |
| OpenWebText, inférence | **10×** |
| XSum, inférence | **31×** |
| Plateforme de chronométrage | **H100 SXM 80 Go** |

Ces rapports concernent une qualité comparable ou meilleure selon les métriques retenues, pas une comparaison avec votre GPT. [Annonce officielle](https://hazyresearch.stanford.edu/blog/2026-10-12-ncm).

| Configuration | Générateur + tokenizer | Longueur cible |
|---|---:|---:|
| TinyStories | 88,61M + 36,43M | 128 |
| LM1B | 97,07M + 111,40M | 128 |
| OpenWebText | 94,65M + 111,2M | 1 024 |
| XSum | 88,02M + 60,63M | 48 |
| WMT14 allemand–anglais | 119,96M + 144M | 64 |
| Sudoku | 25,46M + 14,90M | 89 |

Budgets cumulés tokenizer/générateur : **13,36 milliards de positions** sur LM1B ; **127,22 milliards** sur OpenWebText. Ils incluent répétitions et padding ; ce ne sont pas des FLOPs. WMT14 : **22,6 BLEU**, contre **25,2** pour la référence autorégressive ; avec sélection MBR, NCM atteint **25,2**. Sudoku difficile : **86,1 %**, contre **81,3 %** pour FMLM+. [Article, tableaux et annexes](https://arxiv.org/html/2610.14162v1).

### Ce qu’on peut en conclure pour le budget local

**Calcul documentaire :** les volumes ci-dessus représentent environ **1,8×** et **17,2×** votre budget de 7,4 milliards de tokens. Ces rapports comparent des compteurs différents ; ils montrent surtout pourquoi « moins de données que la diffusion » ne suffit pas à promettre un entraînement moins coûteux ici.

**Informations non trouvées sous une forme exploitable :**

- total complet des heures-GPU d’entraînement ;
- FLOPs cumulés incluant recherche de recette et ablations ;
- pic VRAM d’entraînement sur RTX 4060 ;
- temps d’entraînement français sous 6,5 Gio ;
- résultat sur votre tâche d’assistant.

Je ne convertis donc pas les budgets publiés en jours sur votre machine.

## 5. Code publié et actualités de reproduction

### État du dépôt officiel

Le dépôt public, sous **licence MIT**, fournit les modèles, les composants de tokenisation, les scripts d’évaluation et des liens vers poids et données d’évaluation. Le README actuel référence notamment XSum, WMT14, Sudoku, LM1B, TinyStories et OpenWebText. [Dépôt officiel](https://github.com/HazyResearch/ncm).

**La disponibilité d’une recette officielle complète d’entraînement n’est pas confirmée.** Les demandes restent visibles ; les auteurs avaient annoncé sa publication prochaine le 20 octobre. L’arborescence et les instructions consultées restent centrées sur l’évaluation. [Demande de code d’entraînement](https://github.com/HazyResearch/ncm/issues/2), [réponse des auteurs](https://github.com/HazyResearch/ncm/issues/9).

### Reproductions indépendantes pertinentes

| Source | Observation | Portée |
|---|---|---|
| **Contribution `velaia`, octobre** | WMT14 : **22,61 BLEU** ; XSum : **27,93 ROUGE-L** ; Sudoku difficile : **86,3 %** | Scores proches, obtenus avec checkpoints ; pas réentraînement |
| **RTX 5090 / Windows, 9 novembre** | Sudoku difficile : **86,25 %** ; LM1B 97M/317M/725M : perplexité générative **94,69 / 88,96 / 81,25** | Confirme partiellement qualité et tendance d’échelle |
| **RTX 3080, 16 novembre** | Lot 1 : **164,5 ms** NCM contre **787 ms** pour un décodeur AR de même taille ; lot 32 : environ **44,4 contre 44,2 exemples/s** | Comparateur AR aléatoire : test de vitesse, pas de qualité égale |

Sources primaires : [issue nº 9](https://github.com/HazyResearch/ncm/issues/9), [issue nº 26](https://github.com/HazyResearch/ncm/issues/26), [rapport Andrew Rose](https://andrewroseit.com/research/ncm/).

Le rapport RTX 5090 utilise une boucle AR PyTorch sans compilation ni CUDA graphs. Un commentaire sur RTX 4080 Super rapporte une inversion de l’avantage au grand lot. **Ces mesures ne justifient pas un facteur universel d’accélération.** [Protocole et discussion](https://github.com/HazyResearch/ncm/issues/26).

### Deux problèmes de reproductibilité identifiés

**1. Chronométrage et nombre de passes.** Avec les réglages publiés par défaut, un contributeur ne retrouvait pas la vitesse XSum. Cache KV, BF16 et CUDA graphs rapprochent sa mesure de la figure : **3,45 secondes contre environ 3,4** pour le lot d’évaluation. Les auteurs confirment ces optimisations internes. Ils reconnaissent également une probable erreur de comptage : **8 NFE annoncées, 9 passes observées pour XSum**. [Échange primaire](https://github.com/HazyResearch/ncm/issues/9).

**2. Cache KV.** La PR nº 5 corrige un masque de padding du préfixe omis dans la voie avec cache. Elle est encore ouverte sur la page consultée. Il faut comparer sorties et métriques avant de retenir une accélération. [Correctif proposé](https://github.com/HazyResearch/ncm/pull/5).

Sur RTX 3080, Andrew Rose observe un BLEU de **3,0** avec la voie défectueuse, rétabli à **17,93** après correction sur son sous-ensemble de 256 documents. Il ne présente pas ce sous-ensemble comme une reproduction complète du score de l’article. [Rapport](https://andrewroseit.com/research/ncm/).

## 6. Limites, critiques et résultats négatifs

### Conservation des informations rares

Un portage indépendant MLX rapporte une conservation des noms rares et nombres de **89 % sur XSum** et **96 % sur WMT14** dans son protocole. Sur une petite tâche MASSIVE-de de **11 500 exemples**, sa réimplémentation NCM de **27M** atteint **51,0 % d’exact match**, contre **57,6 %** pour l’AR. Les auteurs du portage précisent que leur entraînement n’a pas été vérifié contre la recette Stanford. **Rapport exploratoire, pas réfutation générale.** [Compte rendu primaire](https://github.com/HazyResearch/ncm/issues/19).

**Conséquence locale :** une représentation peut préserver le thème d’une réponse et perdre précisément le nom, le nombre ou la négation nécessaire à sa réussite. Ce risque est directement pertinent pour `critere_absent`.

### Les raffinements ne garantissent pas une amélioration monotone

Le dépôt indépendant `ncm-experiment-notes` rapporte que **25,04 %** des cellules corrigées aux niveaux 8–9 de ses essais Sudoku redeviennent incorrectes ensuite. Il propose des contrôles et des résultats négatifs, mais ne revendique ni validation par les pairs ni percée de performance. [Notes expérimentales](https://github.com/Quantix-eng/ncm-experiment-notes).

Cela motive une question expérimentale : **quels détails un niveau supplémentaire corrige-t-il, et lesquels détruit-il ?**

### Les petits spécialistes ne deviennent pas automatiquement des assistants

Andrew Rose rapporte **9,9 % d’exact match hors distribution** pour son meilleur petit décodeur de style NCM sur une tâche synthétique de proposition d’outils, malgré **94,2 % en validation dans la distribution**. Ce test concerne sa propre adaptation. Il illustre un problème de généralisation, sans isoler un défaut propre à NCM. [Rapport primaire](https://andrewroseit.com/research/ncm/).

### Points qui restent ouverts

- Une bonne perplexité attribuée par un évaluateur externe ne garantit pas la factualité.
- BLEU et ROUGE ne mesurent pas directement les omissions de l’assistant.
- Sudoku permet une vérification exacte ; le transfert au dialogue français doit être démontré.
- Une comparaison de latence exige des implémentations optimisées des deux côtés.
- Une comparaison d’apprentissage doit inclure tokenizer, générateur et préparation des codes.
- Une réduction du nombre de passes n’établit pas une réduction équivalente de l’énergie.

J’ai également trouvé des affirmations secondaires sur un contrôle Sudoku à décodage « plus confiant d’abord ». **La source expérimentale primaire n’a pas pu être consultée : je ne retiens pas ses chiffres comme preuves.**

## 7. Conséquences pour le projet et expériences minimales

### Contraintes confirmées dans le dépôt

Avec les embeddings liés, la configuration du run nommé « 369M » contient **335 544 320 paramètres uniques** : calcul issu de `model.py`. Le lanceur réserve **6,5 Gio**, tandis que son argument d’entraînement fixe **6,1 Gio**. [Modèle](D:/ServOMorph/creation_llm/model.py), [lanceur](D:/ServOMorph/creation_llm/run_369m.py).

NCM demanderait une filière distincte de préparation et d’apprentissage. Aucun transfert documenté trouvé ne permet de convertir directement le checkpoint courant en NCM en conservant ses capacités.

Les propositions suivantes sont des **hypothèses locales**. Les seuils sont des décisions expérimentales proposées, jamais des gains prédits.

### Proposition A — Vérifier l’inférence officielle sur la RTX 4060

| Élément | Protocole |
|---|---|
| **Mécanisme** | Exécuter un checkpoint WMT14, comparer voie standard et cache corrigé |
| **Gain attendu chiffré** | **Inconnu** ; seuil d’intérêt : latence réduite d’au moins **20 %**, qualité préservée |
| **VRAM** | Environ 264M paramètres : poids seuls ≈ **0,98 Gio en FP32**, hors activations et espace de travail ; pic à mesurer |
| **Temps** | Pilote plafonné à **1 heure GPU**, hors installation ; enveloppe proposée |
| **Expérience minimale** | 256 entrées fixes ; lots 1 et 8 ; échauffement, synchronisation, p50/p95, pic alloué/réservé, BLEU et sorties |
| **Échec** | Dépassement de 6,1 Gio, divergence inexpliquée des sorties ou absence du gain de latence fixé |
| **Sources** | [PR cache KV](https://github.com/HazyResearch/ncm/pull/5), [essai RTX 3080](https://andrewroseit.com/research/ncm/) |

Ce test validerait une compatibilité matérielle et une méthode de mesure. Il ne démontrerait pas la supériorité sur votre assistant.

### Proposition B — Tester d’abord la fidélité d’un tokenizer français

| Élément | Protocole |
|---|---|
| **Mécanisme** | Apprendre puis évaluer uniquement l’aller-retour texte → codes → texte |
| **Gain attendu chiffré** | **Inconnu** ; condition de poursuite proposée : **≥99,9 %** de récupération exacte des tokens et **≥99 %** des champs factuels |
| **VRAM** | Dépend de la configuration ; mesurer avant toute extrapolation. Plafond : **6,1 Gio** |
| **Temps** | Pilote plafonné à **4 heures GPU** ; durée nécessaire à la convergence inconnue |
| **Expérience minimale** | Textes français tenus à l’écart, avec noms, dates, nombres, accents et négations ; mesurer séparément chaque catégorie |
| **Échec** | Champs factuels sous le seuil, erreurs systématiques ou coût incompatible avec la suite |
| **Source** | [Portage et diagnostic de fidélité](https://github.com/HazyResearch/ncm/issues/19) |

Un échec dans cette enveloppe signifierait « piste non justifiée sous ce budget », pas « architecture impossible ».

### Proposition C — Comparaison française complète, uniquement après A et B

| Élément | Protocole |
|---|---|
| **Mécanisme** | Petit NCM contre petit GPT maison, sur les mêmes tâches conditionnelles |
| **Gain attendu chiffré** | **Inconnu** ; cible proposée : **+3 points** de réussite à temps total égal, ou qualité équivalente avec **≥20 %** de latence économisée |
| **VRAM** | Pics des deux phases mesurés séparément ; tous les composants comptés |
| **Temps** | Trois graines ; budget NCM total fixé égal au budget AR, incluant apprentissage du tokenizer et encodage |
| **Expérience minimale** | Même corpus source, test indépendant ; comparaison à temps égal et rapport séparé à exposition égale aux données |
| **Échec** | Gain non confirmé par comparaison appariée, factualité dégradée, ou avantage disparu après comptabilité complète |
| **Sources** | [Dépôt officiel](https://github.com/HazyResearch/ncm), [réimplémentation indépendante et réserves](https://github.com/HazyResearch/ncm/issues/19) |

Pour le temps, utiliser une comptabilité explicite :

\[
T_{\mathrm{NCM}}
=
T_{\mathrm{tokenizer}}
+
T_{\mathrm{encodage}}
+
T_{\mathrm{générateur}}
+
T_{\mathrm{évaluation}}.
\]

**Décision proposée :** conserver les priorités du cycle 2 et terminer le run actuel. NCM mérite un essai limité de système ; son adoption comme prochain modèle exige encore une preuve française à budget complet égal.

## 8. Tableau des sources

| Titre / ressource | Date | URL | Statut |
|---|---|---|---|
| Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation | 9 octobre 2026 | [Article](https://arxiv.org/abs/2610.14162) | Prépublication ; texte et annexes consultés |
| Next-Concept Modeling — Hazy Research | 12 octobre 2026 | [Annonce](https://hazyresearch.stanford.edu/blog/2026-10-12-ncm) | Source institutionnelle des auteurs |
| HazyResearch/ncm | Consulté le 18 novembre 2026 | [Code](https://github.com/HazyResearch/ncm) | Dépôt officiel MIT, évaluation et checkpoints |
| NextConceptTransformer | Consulté le 18 novembre 2026 | [Implémentation](https://github.com/HazyResearch/ncm/blob/main/models/next_concept.py) | Source primaire technique |
| Request for training codes — nº 2 | 15 octobre 2026 | [Discussion](https://github.com/HazyResearch/ncm/issues/2) | Demande de recette officielle |
| Fix the KV-cached generate() path — PR nº 5 | 17 octobre 2026 | [Correctif](https://github.com/HazyResearch/ncm/pull/5) | Contribution indépendante, ouverte |
| Inference settings for Figure 1 — nº 9 | 20–26 octobre 2026 | [Discussion](https://github.com/HazyResearch/ncm/issues/9) | Reproduction et réponse des auteurs |
| Independent MLX experiments — nº 19 | 31 octobre 2026 ; échanges en novembre | [Rapport](https://github.com/HazyResearch/ncm/issues/19) | Exploration indépendante, résultats négatifs inclus |
| Reproduction on an RTX 5090 — nº 26 | 9 novembre 2026 | [Rapport](https://github.com/HazyResearch/ncm/issues/26) | Reproduction partielle, Windows |
| Next-Concept Modeling on an RTX 3080 | 16 novembre 2026 | [Rapport Andrew Rose](https://andrewroseit.com/research/ncm/) | Mesures indépendantes, non évaluées par les pairs |
| NCM experiment notes | Date initiale non établie ; consulté le 18 novembre | [Code et notes](https://github.com/Quantix-eng/ncm-experiment-notes) | Études exploratoires indépendantes |
| A Neural Conversational Model | 2015 | [Google Research](https://research.google/pubs/a-neural-conversational-model/) | Homonyme historique, atelier ICML |
| Large Concept Models | Consulté le 18 novembre 2026 | [Dépôt Meta](https://github.com/facebookresearch/large_concept_model/blob/main/README.md) | Architecture distincte |

**Limite documentaire finale :** certaines métriques de checkpoints ont été reproduites, mais je n’ai trouvé aucune reproduction complète établissant un avantage d’entraînement NCM sur un GPT français avec Muon, 7,4 milliards de tokens et un plafond de 6,5 Gio.