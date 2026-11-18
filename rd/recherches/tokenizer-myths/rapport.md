# Tokenizer Myths : mieux mesurer la tokenisation du LLM français

*Recherche documentaire du 18 novembre 2026. Aucun fichier modifié, aucun calcul GPU lancé.*

## Résumé en 10 lignes

1. Le sujet correspond exactement à **Tokenizer Myths and Where To Find Them**, de Rylan Schaeffer et Sanmi Koyejo.
2. L’article est une **prépublication**, pas un résultat indépendamment reproduit établi dans cette recherche.
3. Pour le projet, son intérêt principal concerne le protocole d’évaluation des futurs tokenizers.
4. Je recommande de reformuler l’affirmation locale selon laquelle les bits par octet permettent une comparaison « sans biais ».
5. Les BPB restent utiles pour mesurer le coût de codage du texte par un couple tokenizer–modèle.
6. La qualité de l’assistant doit conserver ses mesures propres : réussite, omissions et respect des consignes.
7. Le vocabulaire actuel de 32 768 entrées ne doit pas être changé sur cette seule base.
8. Les futurs essais SuperBPE et H-Net doivent combiner qualité, données effectivement vues et temps réel.
9. Un audit CPU peut enrichir les expériences déjà proposées, sans concurrencer le pré-entraînement.
10. **Verdict : À INTÉGRER dans la méthode d’évaluation ; aucun changement du run actuel justifié.**

## 1. Identification et niveau de preuve

| Candidat | Correspondance |
|---|---|
| **Tokenizer Myths and Where To Find Them** — Rylan Schaeffer, Sanmi Koyejo, Stanford Computer Science ; 3 novembre 2026 | **Exacte : titre et identifiant fournis. Prépublication.** |
| **TokEval: A Tokenizer Evaluation Suite** — Clara Meister ; août 2026 | Travail voisin : outils et expériences d’évaluation, publié à COLM 2026. |
| **Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies** — Chaofan Tao et collaborateurs ; juillet 2024 | Travail antérieur sur le choix du vocabulaire, publié à NeurIPS 2024. |

Sources : [Tokenizer Myths](https://arxiv.org/abs/2611.04215), [TokEval](https://arxiv.org/abs/2608.18062), [Scaling Laws with Vocabulary](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cf5a019ae9c11b4be88213ce3f85d85c-Abstract-Conference.html).

**Convention de ce rapport :** « publié » indique une publication en conférence ; « reproduit » exige une vérification indépendante identifiée. Une publication et un dépôt de code ne suffisent pas, seuls, à établir la reproduction. Les propositions locales ci-dessous sont des **hypothèses de protocole**.

## 2. Résultats documentaires essentiels

Les trois conclusions des auteurs sont : BPB dépend du tokenizer ; sa baisse ne garantit pas une meilleure performance ; la normalisation unigramme ne supprime pas cette dépendance.

| Expérience | Résultat rapporté |
|---|---|
| BPE français, FineWeb-2 ; apprentissage tokenizer **10 Go**, validation **0,223 Go** | Modèle uniforme : minimum **3,3195 BPB**, vocabulaire **26 977** |
| TokEval : **85 tokenizers**, modèles **0,98–1,60B**, **9–10,5B tokens** | Corrélations BPB–score : BLiMP **+0,02** ; MBPP **−0,22** ; HumanEval **−0,28** ; GSM8K **−0,30** |
| Réanalyse de **56 points** de Tao et al. | Corrélation **0,989 → 0,057** après centrage par taille de modèle |

Leur trilemme rend incompatibles : optimum strict pour la vraie distribution, invariance aux représentations et égalité des références non entraînées. Sa portée dépend des hypothèses de représentabilité et de normalisation.

Le code spécifique est annoncé à venir ; aucune URL publique vérifiée trouvée. Heures GPU et pic VRAM : **non trouvés**. Le minimum français concerne des prédictions uniformes, pas un optimum d’entraînement. [Article, résultats et annexes](https://arxiv.org/html/2611.04215v1).

## 3. Comprendre le mécanisme dans le dépôt

Les calculs suivants sont une **vérification algébrique et une application au code local**, sans expérience nouvelle.

### 3.1 Ce que mesure la validation actuelle

Dans [model.py](/D:/ServOMorph/creation_llm/model.py), `cross_entropy` calcule une perte moyenne par token. [train.py](/D:/ServOMorph/creation_llm/train.py) agrège les pertes de lots de même taille.

Pour un ensemble fixe de cibles :

\[
L=-\sum_{i=1}^{N}\ln p(t_i\mid t_{<i}),
\qquad
\mathrm{NLL}=\frac{L}{N},
\qquad
\mathrm{BPB}=\frac{L}{B\ln 2},
\]

où \(B\) compte les octets du texte effectivement évalué.

Ainsi :

\[
\mathrm{BPB}=\frac{N}{B}\frac{\mathrm{NLL}}{\ln 2}.
\]

**À tokenizer, textes et fenêtres identiques, BPB et NLL classent les checkpoints exactement de la même manière.** Ajouter BPB ne constitue donc pas, à lui seul, un nouvel indicateur de progrès pour le run actuel.

En revanche, changer le tokenizer modifie \(N/B\), les unités prédites et potentiellement le contexte disponible.

### 3.2 Exemple calculé : un meilleur score sans apprentissage

Considérons deux distributions artificiellement uniformes :

| Hypothèse illustrative | Vocabulaire | Tokens/octet | BPB calculée |
|---|---:|---:|---:|
| A | \(2^{15}\) | 0,25 | \(15×0,25=3,75\) |
| B | \(2^{16}\) | 0,20 | \(16×0,20=3,20\) |

Ces valeurs sont **inventées pour l’exemple mathématique**, pas présentées comme des mesures. Aucun réseau n’a appris ; pourtant les coûts de codage diffèrent.

Ce constat n’invalide pas la mesure : il interdit de transformer automatiquement son amélioration en « connaissances apprises supplémentaires ».

### 3.3 Un score de texte et une réussite peuvent diverger

Exemple construit : pour deux réponses candidates, un modèle attribue les probabilités \((0,20;0,10)\), un autre \((0,40;0,20)\). Ils choisissent tous deux la première réponse. La probabilité de cette réponse a doublé, mais leur décision n’a pas changé.

Pour l’assistant, la difficulté est plus grande : plusieurs formulations peuvent transmettre la même information. La probabilité d’une réponse de référence particulière ne mesure pas directement la présence de l’information dans une génération libre.

Cela rejoint le diagnostic local : [bilan_assistant_v2.md](/D:/ServOMorph/creation_llm/bilan_assistant_v2.md) compte **139 occurrences de `critere_absent`**. Une baisse de perte ne permet pas de prédire combien disparaîtront.

### 3.4 Soustraire une référence ne résout pas toute la comparaison

Si l’on calcule une différence de pertes par token entre un modèle et une référence unigramme, on mesure son progrès **par rapport à cette référence particulière**. Lorsque le tokenizer change, la référence et le dénominateur changent aussi.

Pour une expérience locale, il faut donc nommer précisément l’objet évalué :

| Question | Mesure adaptée |
|---|---|
| Quel système code le même texte plus efficacement ? | BPB, avec protocole documenté |
| Quel modèle progresse depuis sa propre initialisation ? | Écart à une référence explicitement définie |
| Quel assistant répond mieux ? | Benchmark de réponses, erreurs et omissions |
| Quel système est préférable sur la RTX 4060 ? | Qualité à temps réel et mémoire contraints |

Aucune de ces colonnes ne remplace les autres.

## 4. Liens avec les travaux déjà présents

L’index et les veilles ont été consultés avant la conclusion ; les développements techniques déjà présents ne sont pas repris ici.

| Document local | Complément utile |
|---|---|
| [Pôle mathématiques, I6](/D:/ServOMorph/creation_llm/rd/pole_mathematiques.md:99) | Remplacer conceptuellement « comparer deux tokenizers sans biais » par « comparer le coût de codage de couples tokenizer–modèle sur un protocole commun ». |
| [Sélection R&D](/D:/ServOMorph/creation_llm/rd/selection.md) | Conserver BPB comme mesure, retirer toute interprétation de garantie de qualité générale. |
| [Veille entraînement, SuperBPE](/D:/ServOMorph/creation_llm/rd/veille_entrainement.md:194) | Le test CPU de réduction du nombre de tokens est une présélection économique ; il ne valide pas la qualité. |
| [Recherche H-Net](/D:/ServOMorph/creation_llm/rd/recherches/h-net/rapport.md) | Conserver ses contrôles de contexte et de données ; compléter les objectifs BPB par les tâches finales déjà prévues. |
| [Recherche LexisLM](/D:/ServOMorph/creation_llm/rd/recherches/lexislm/rapport.md) | Enrichir l’audit tokenizer et l’audit conjoint perte–omissions existants ; ne pas créer un programme concurrent. |
| Veilles inférence, frugalité, MoE et JEPA | Même séparation entre métrique intermédiaire, calcul et résultat utilisateur ; aucune nouvelle piste architecturale à ajouter. |

Le [bilan du cycle 1](/D:/ServOMorph/creation_llm/rd/bilan_cycle1.md) reste prioritaire : les passages pertinents relus portent les faits de **49/90 à 71/90**. Ce résultat local soutient l’étude de l’accès à l’information et de son exploitation. Il ne désigne pas la tokenisation comme cause principale des omissions.

## 5. Sources voisines, code et reproductibilité

### TokEval : une ressource complémentaire exploitable

TokEval examine notamment les frontières UTF-8, les nombres et la structure du code. Clara Meister rapporte des associations différentes selon les métriques et les tâches ; les mesures informationnelles atteignent une corrélation de Spearman allant jusqu’à **0,80** avec les capacités de modélisation du langage. Cela motive un tableau de mesures plutôt qu’un classement unique. [Article TokEval](https://arxiv.org/abs/2608.18062).

Le [dépôt officiel](https://github.com/cimeister/tokenizer-intrinsic-evals) fournit des commandes d’analyse, de visualisation et de contrôle. Il avertit que certaines définitions ont changé : comparer des versions différentes peut produire des résultats incompatibles.

Les [artefacts de modèles](https://huggingface.co/cmeister/tokenizer-lm-ablations) comprennent configurations, tokenizers, poids et provenance. Leur documentation indique aussi l’exclusion de familles défectueuses. **Disponibilité des artefacts vérifiée ; reproduction locale non effectuée.**

### Tao et al. : distinguer observation et prescription

L’étude implique notamment University of Hong Kong et Sea AI Lab, avec des collaborateurs de Contextual AI, Stanford et Ohio State. Elle couvre des modèles de **33M à 3B paramètres**, jusqu’à **500 milliards de caractères**.

Les auteurs rapportent, pour une comparaison à **\(2,3×10^{21}\) FLOPs**, une progression ARC-Challenge de **29,1 à 32,0** en passant de **32k à 43k**. Ce résultat publié ne donne pas directement une recommandation française pour le projet. [Publication NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cf5a019ae9c11b4be88213ce3f85d85c-Abstract-Conference.html), [article avec affiliations](https://papers.neurips.cc/paper_files/paper/2024/file/cf5a019ae9c11b4be88213ce3f85d85c-Paper-Conference.pdf).

Le [code officiel](https://github.com/sail-sg/scaling-with-vocab) est disponible. Ses ajustements n’ont pas été exécutés ici.

### Reproductions et critiques indépendantes

**Aucune reproduction indépendante ni réponse technique des auteurs critiqués n’a été identifiée dans les recherches effectuées.** Les résumés automatiques et reprises éditoriales trouvés ne constituent pas des reproductions.

TokEval est un travail antérieur indépendant pertinent, mais ce n’est pas une reproduction de *Tokenizer Myths*.

## 6. Limites et contre-expertise pour la cellule

Les points suivants sont **mon analyse**, pas de nouveaux résultats expérimentaux.

- **La neutralité dépend de la question.** Récompenser un tokenizer qui exploite des régularités du corpus est légitime si l’objectif est le meilleur système complet.
- **Une corrélation faible ne prouve pas l’absence de causalité.** Bruit des benchmarks, plage de performances réduite et variables concomitantes peuvent compliquer son interprétation.
- **Même architecture ne signifie pas même budget.** Le vocabulaire affecte les embeddings, la projection de sortie, le nombre de tokens et les octets visibles dans une fenêtre.
- **Le français exige son propre contrôle.** Accents, apostrophes, noms rares, nombres et normalisation doivent être vérifiés sur les textes du projet.
- **Les propriétés d’un modèle uniforme ne décrivent pas toute l’initialisation locale.** Les embeddings liés et leur initialisation dans `model.py` doivent être considérés avant d’assimiler le réseau initial à cette référence.
- **La compression reste une propriété utile.** Il serait excessif d’abandonner BPB ou de conclure que le tokenizer n’a pas d’importance.

## 7. Conséquences matérielles vérifiées dans le code

Le lanceur réserve **6,5 Go** via le garde GPU et passe un plafond interne de **6,1 Go** à l’entraînement. Ces deux valeurs coexistent dans [run_369m.py](/D:/ServOMorph/creation_llm/run_369m.py).

Avec des embeddings liés de largeur 1 024 :

\[
P_{\mathrm{vocabulaire}}=1\,024V.
\]

Passer de 32 768 à 65 536 entrées ajouterait **33 554 432 paramètres**. Sous l’hypothèse poids, gradients et deux moments AdamW en FP32, cela représente **512 Mio supplémentaires**, hors activations et temporaires. C’est un calcul conditionnel, pas une mesure VRAM.

Autres contraintes :

- `train.py` lit les corpus en **`uint16`** : les identifiants au-delà de 65 535 exigent un changement de format.
- Une fenêtre de 1 024 tokens couvre une quantité de texte variable selon le tokenizer.
- `train_sft.py` évalue uniquement les cibles non masquées : une éventuelle BPB SFT doit compter les octets de ces mêmes cibles, pas ceux du prompt complet.
- Le modèle appelé « 369M » contient, d’après les matrices du code et les embeddings liés, **335 544 320 paramètres uniques**. Cette distinction compte pour les budgets comparatifs.

## 8. Propositions falsifiables

### A. Enrichir l’audit CPU existant — priorité immédiate

**Statut : hypothèse de protocole ; complément aux audits LexisLM et SuperBPE.**

- **Mécanisme :** mesurer séparément fidélité du décodage, tokens/octet, segmentation des nombres et coût uniforme calculé.
- **Expérience minimale :** échantillon tenu à l’écart de chaque source ; tokenizer actuel et un candidat ; mêmes textes UTF-8 et conventions de normalisation. Commencer par **1 Mio par source**, puis étendre si utile.
- **Gain attendu chiffré :** **0 point direct** sur l’assistant, puisque ses poids ne changent pas ; bénéfice décisionnel non quantifiable actuellement.
- **Coût :** **0 Go VRAM** ; durée CPU non mesurée. Mesurer le débit sur l’échantillon initial avant de fixer le volume final.
- **Critère d’échec :** toute altération non autorisée du texte élimine le candidat ; une amélioration de compression seule ne déclenche aucun réentraînement.
- **Sources :** [TokEval et ses outils](https://github.com/cimeister/tokenizer-intrinsic-evals), [Tokenizer Myths](https://arxiv.org/abs/2611.04215).

### B. Relier les mesures aux omissions — priorité après les évaluations prévues

**Statut : hypothèse locale ; prolongement de la roadmap SFT.**

- **Mécanisme :** confronter perte, succès et `critere_absent` sur les mêmes exemples ; séparer faits, consignes et transformations.
- **Expérience minimale :** réutiliser les sorties disponibles ; compléter lors du benchmark prévu du 369M. Comparaison appariée par question ; ne pas traiter plusieurs graines d’une même question comme des questions indépendantes.
- **Gain attendu chiffré :** **0 point direct** pour l’analyse ; gain d’une décision ultérieure inconnu.
- **Coût :** **0 Go VRAM** pour l’analyse des sorties ; scoring supplémentaire éventuel à mesurer. Aucun temps GPU fiable disponible.
- **Critère d’échec :** si une métrique intermédiaire ne prédit pas les omissions sur des données distinctes de celles ayant servi à la choisir, ne pas l’utiliser comme substitut au benchmark.
- **Sources :** [TokEval](https://arxiv.org/abs/2608.18062), [bilan assistant](/D:/ServOMorph/creation_llm/bilan_assistant_v2.md).

### C. Comparer deux tokenizers entraînés — seulement pour un prochain témoin

**Statut : hypothèse ; aucune exécution recommandée pendant le run prioritaire.**

- **Mécanisme :** tester le bénéfice complet à contrainte matérielle réelle.
- **Expérience minimale :** deux tokenizers, même famille de modèle réduit, trois graines ; une comparaison à textes identiques et une comparaison à temps égal. Rapporter paramètres, octets traités, contexte, BPB et tâches.
- **Gain attendu chiffré :** **inconnu**. Seuil d’intérêt proposé, non prédiction : **10 % de temps économisé** à qualité finale non dégradée.
- **Coût :** six entraînements. Pour \(D\) octets par bras et des débits mesurés \(q_A,q_B\), durée approximative \(3D/q_A+3D/q_B\), hors préparation et évaluation. VRAM à mesurer ; aucune garantie de tenue donnée.
- **Critère d’échec :** pas d’avantage reproductible sur qualité–temps, régression des omissions ou dépassement du plafond mémoire.
- **Sources :** [protocole et artefacts TokEval](https://huggingface.co/cmeister/tokenizer-lm-ablations), [étude à budget de calcul contrôlé](https://arxiv.org/abs/2407.13623).

## 9. Tableau des sources

| Titre / ressource | Date | URL | Statut |
|---|---|---|---|
| Tokenizer Myths and Where To Find Them | 03/11/2026 | [Notice](https://arxiv.org/abs/2611.04215) ; [texte et annexes](https://arxiv.org/html/2611.04215v1) | Prépublication ; reproduction indépendante non trouvée |
| TokEval: A Tokenizer Evaluation Suite | 18/08/2026 ; v3 09/09/2026 | [Article](https://arxiv.org/abs/2608.18062) | Publié à COLM 2026 ; reproduction locale non effectuée |
| TokEval — outils | Consulté le 18/11/2026 | [GitHub](https://github.com/cimeister/tokenizer-intrinsic-evals) | Code officiel disponible |
| TokEval — modèles et provenance | Consulté le 18/11/2026 | [Hugging Face](https://huggingface.co/cmeister/tokenizer-lm-ablations) | Artefacts officiels disponibles |
| Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies | 18/07/2024 ; NeurIPS 2024 | [Article](https://arxiv.org/abs/2407.13623) ; [publication](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cf5a019ae9c11b4be88213ce3f85d85c-Abstract-Conference.html) | Publié ; résultats auteurs, extrapolation locale non établie |
| Scaling with Vocabulary — code | Consulté le 18/11/2026 | [GitHub](https://github.com/sail-sg/scaling-with-vocab) | Code officiel disponible ; non exécuté |

**Décision proposée : conserver le tokenizer et le pré-entraînement actuels, intégrer ces contrôles aux audits existants, et réserver toute sélection de nouveau tokenizer à une comparaison de qualité et de coût effectivement mesurés.**