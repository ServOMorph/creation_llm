# LFM3-1.7B de Liquid AI : architecture hybride, preuves disponibles et intérêt pour le projet

*Recherche arrêtée au 18 novembre 2026. Aucun fichier modifié, aucun calcul GPU lancé.*

## Résumé en 10 lignes

1. **Le nom est exact : LFM3-1.7B**, annoncé le 27 octobre 2026, correspond bien au sujet indiqué. [Annonce](https://huggingface.co/blog/LiquidAI/lfm3)
2. Il faut distinguer le modèle assistant **LFM3-1.7B** de son checkpoint **LFM3-1.7B-Base**. [Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base)
3. Son architecture combine convolutions courtes, attention locale et attention globale.
4. Les poids, la configuration et une implémentation sont disponibles ; la reproduction complète du pré-entraînement ne l’est pas dans les sources trouvées.
5. Le français figure parmi les langues annoncées, sans résultat spécifique retrouvé sur votre benchmark.
6. Les performances publiées sont principalement des évaluations de Liquid AI, pas des reproductions indépendantes.
7. Le budget annoncé de **22 000 milliards de tokens** interdit d’attribuer les résultats à la seule architecture.
8. La mémoire des poids quantifiés ne représente pas la mémoire totale nécessaire à l’inférence.
9. Pour le projet, l’usage prioritaire serait un **modèle de comparaison local**, éventuellement un enseignant après validation.
10. **Verdict : À TESTER pour l’inférence française ; À SURVEILLER pour une nouvelle architecture ; aucun changement du run actuel.**

## 1. Identification et niveau de preuve

| Candidat | Correspondance avec le sujet |
|---|---|
| **LFM3-1.7B** | Meilleur candidat : modèle textuel post-entraîné, destiné notamment aux agents locaux |
| **LFM3-1.7B-Base** | Même architecture ; checkpoint de départ pour adaptation importante |
| **LFM2 / LFM2.5** | Générations précédentes : pertinentes pour comprendre la filiation technique |
| **LFM3-VL, Audio, Omni et d1** | Déclinaisons de la famille ; leurs résultats ne caractérisent pas directement l’assistant textuel |

L’article fourni est une annonce collective de **Liquid AI**, datée du **27 octobre 2026**. Parmi ses signataires figurent Leonie Monigatti, Jimmy T. H. Smith, Anthony Maltsev, Maxime Labonne, Alexander Amini, Ramin Hasani et Mathias Lechner. Cette liste est partielle et ne permet pas d’attribuer chaque innovation à une personne particulière. [Article collectif](https://huggingface.co/blog/LiquidAI/lfm3)

**Aucun rapport scientifique autonome LFM3 avec ablations complètes n’a été trouvé** dans les recherches effectuées. Le rapport technique de référence trouvé concerne **LFM2**, déposé le 28 novembre 2025, signé Alexander Amini et 32 coauteurs. Il décrit notamment la recherche d’architecture sous contraintes matérielles. Il ne constitue pas une validation expérimentale de LFM3. [LFM2 Technical Report](https://arxiv.org/abs/2511.23404)

Conventions utilisées :

- **Établi documentairement** : vérifiable dans un artefact publié, par exemple la configuration.
- **Résultat établi et reproduit** : résultat expérimental confirmé indépendamment dans un protocole identifiable.
- **Annonce / rapport technique non reproduit** : chiffres de l’équipe, sans reproduction indépendante retrouvée.
- **Prépublication** : manuscrit arXiv ; publication évaluée par les pairs non vérifiée ici.
- **Hypothèse locale** : transposition au modèle maison ou au matériel du projet.

L’existence du code est établie documentairement. **L’amélioration de votre assistant grâce à cette architecture ne l’est pas.**

## 2. Liens avec les recherches déjà réalisées

L’index des recherches et les cinq veilles `rd/veille_*.md` ont été consultés, avec lecture ciblée des passages pertinents. Les fichiers de modèle, de pré-entraînement, de SFT, la roadmap, le bilan assistant, la charte, la sélection, le bilan du cycle 1 et le pôle mathématiques ont également été consultés.

| Travail existant | Apport spécifique de cette recherche |
|---|---|
| `veille_frugalite.md`, hybrides Gated DeltaNet / Olmo Hybrid | LFM3 propose un autre mélange : convolution courte et deux portées d’attention. Les résultats d’Olmo Hybrid ne lui sont pas transférables. |
| `veille_frugalite.md`, distillation textuelle | LFM3 fournit un candidat concret d’enseignant local ; sa compétence française reste à mesurer. La méthode de distillation n’est pas redéveloppée ici. |
| `veille_inference.md`, RAFT et passages distracteurs | Le bon test reste la complétude des réponses avec documents imparfaits, pas seulement un benchmark général. |
| `bilan_cycle1.md` | Le passage pertinent fait passer les faits de 49/90 à 71/90 ; la difficulté de recherche et de résistance aux distracteurs reste centrale. |
| `veille_moe_chine.md` | « Hybride » ne signifie pas MoE : aucun routage entre experts dans le modèle textuel examiné. |
| `veille_lecun_jepa.md` | Sujet distinct : LFM3 reste un générateur autorégressif ; il ne remplace pas la prédiction de tokens par un objectif JEPA. |
| Recherches H-Net et NCM indexées | Leur existence et leurs verdicts ont été relevés dans l’index ; leurs rapports complets ne sont pas réexaminés ici. |

Deux précisions déjà relevées par la veille frugalité sont confirmées par le code :

- Le run nommé « 369M » possède **335 544 320 paramètres uniques** avec embeddings liés.
- `113000 × 65536 = 7 405 568 000` tokens représentent **13,6 jours à 6 300 tok/s**, hors interruptions. Les quelque 12 jours correspondent au débit initial proche de 7 000 tok/s.

Ces corrections servent aux comparaisons de budget ; elles ne changent pas la priorité du pré-entraînement.

## 3. Architecture et mécanisme

### 3.1 Configuration vérifiable

| Élément | LFM3-1.7B |
|---|---:|
| Largeur cachée | 2 048 |
| Couches | 24 |
| Convolutions courtes | 13 |
| Attention à fenêtre glissante | 5, fenêtre de 256 tokens |
| Attention globale | 6 |
| Têtes de requêtes / clés-valeurs | 16 / 4 |
| Dimension d’une tête | 128 |
| Noyau convolutionnel | 7 |
| Largeur intermédiaire du MLP | 7 680 |
| Vocabulaire | 128 000 |
| Contexte configuré | 262 144 tokens |
| Embeddings entrée/sortie | Liés |

Les couches globales occupent les positions **3, 7, 11, 15, 19 et 23**, en numérotation commençant à 1. Le partage de quatre têtes clés-valeurs entre seize têtes de requêtes correspond à la **Grouped Query Attention**. [Configuration publiée](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/config.json)

### 3.2 Les trois opérateurs

**Lecture du code**, sous forme simplifiée :

\[
[B,C,X]=W_{\mathrm{in}}h,\qquad
y=W_{\mathrm{out}}\left[C\odot\operatorname{Conv}_{7}(B\odot X)\right].
\]

La convolution est causale et indépendante par canal. Ses portes multiplicatives ne sont pas les portes sigmoïdes de l’attention.

Dans l’attention :

\[
g=\sigma(W_g h),\qquad
y=W_o\operatorname{Concat}_j(g_j\,\operatorname{Attention}_j).
\]

Chaque tête reçoit un coefficient dépendant du token. Le MLP utilise **SwiGLU**. Les normalisations sandwich agissent sur l’entrée et la sortie des branches avant addition résiduelle ; Q et K sont également normalisés. [Implémentation](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/modeling_lfm3.py)

L’intuition est de réserver les interactions globales coûteuses à certaines couches, tout en traitant davantage les motifs proches par des opérateurs locaux. **Déduction :** conserver six attentions globales laisse une composante quadratique en longueur lors du préremplissage ; le modèle entier n’est donc pas un réseau à coût strictement linéaire.

### 3.3 RoPE partiel et filiation scientifique

Dans les attentions globales, la moitié des dimensions reste sans rotation. L’implémentation conserve la progression fréquentielle correspondant à la dimension complète puis annule les fréquences restantes. Les attentions locales conservent RoPE complet. [Configuration et paramètres RoPE](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/config.json)

Trois travaux antérieurs sont cités par Liquid :

- **Bondarenko, Nagel et Blankevoort, 2023** : permettre aux têtes de produire une mise à jour proche de zéro réduit certains extrêmes d’activation nuisibles à la quantification. [Quantizable Transformers](https://arxiv.org/abs/2306.12929)
- **Qiu et al., 2025** : étude de plus de trente variantes de portes, sur modèles denses 1,7B et MoE 15B ; amélioration annoncée de stabilité et de performances. [Gated Attention](https://arxiv.org/abs/2505.06708)
- **Barbero et al., 2024, révisé en 2025** : analyse du rôle positionnel et sémantique des différentes fréquences RoPE. [Round and Round We Go!](https://arxiv.org/abs/2410.06205)

Ces travaux soutiennent les mécanismes. Ils ne quantifient pas le gain isolé de chacun dans LFM3, ni dans votre modèle.

## 4. Entraînement, données et calcul

La base est annoncée à **1,73 milliard de paramètres**, entraînée sur **22 000 milliards de tokens**, dont les **3 500 derniers milliards** constituent un *mid-training* sur données sélectionnées et trajectoires d’agents, avec extension du contexte. La fiche décrit du web, du synthétique, du code, des mathématiques et des sciences. [Fiche Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base)

La recherche d’architecture annoncée couvre **14 combinaisons appareil/backend**. Elle optimise donc aussi des contraintes de déploiement matériel, au-delà de la seule perte linguistique. [Présentation technique de Liquid](https://www.liquid.ai/blog/lfm3)

| Information nécessaire à une reproduction complète | État trouvé |
|---|---|
| Liste exhaustive des corpus et proportions | Non trouvée |
| Quantité et proportion de français | Non trouvées |
| Données et recette intégrales de post-entraînement | Non trouvées |
| Nombre et modèle des GPU de pré-entraînement | Non trouvés |
| GPU-heures, énergie, coût financier | Non trouvés |
| FLOPs mesurés de pré-entraînement | Non trouvés |
| Ablation architecture seule, à données et calcul égaux | Non trouvée |

**Calculs de cadrage**, à partir des tailles annoncées et du dépôt :

- paramètres : environ **5,16 fois** ceux du modèle maison ;
- tokens : environ **2 971 fois** le budget prévu ;
- tokens par paramètre : environ **12 700**, contre **22** dans le projet.

Les tokenizers étant différents, le rapport de tokens n’est pas un rapport exact de quantité de texte. Il suffit néanmoins à montrer que les deux modèles occupent des régimes d’entraînement très éloignés.

**Aucun gain de temps de pré-entraînement sur RTX 4060 ne peut être déduit de ces chiffres.**

## 5. Résultats publiés et portée réelle

### 5.1 Évaluations de Liquid AI

Sélection des scores publiés :

| Benchmark | LFM3-1.7B | LFM2.5-2.6B | Qwen3.5-9B |
|---|---:|---:|---:|
| WebArena-Verified | **38,42** | 3,02 | 28,94 |
| BFCL V4 | **71,39** | 55,89 | 62,83 |
| IFEval | **96,66** | 95,39 | 92,08 |
| SWE-bench Verified | 43,40 | 18,00 | **51,20** |
| GPQA Diamond | 63,37 | 57,78 | **80,40** |
| RULER 128K | 65,56 | 1,15 | **87,11** |

**Statut : évaluations publiées par le fournisseur, non reproduites indépendamment dans cette recherche.** [Tableau officiel](https://huggingface.co/LiquidAI/LFM3-1.7B)

Le protocole emploie notamment vLLM en BF16, les contextes complets et, sauf exceptions, une température de 0,6 avec `top_p=0,95`. Les limites de génération sont beaucoup plus généreuses que les 300 tokens de votre benchmark. [Conditions d’évaluation](https://huggingface.co/LiquidAI/LFM3-1.7B)

**Analyse :** comparer directement IFEval à vos 49,7 % n’aurait aucun sens. Les tâches, budgets, formats et critères diffèrent. Les défaites sur GPQA et RULER montrent également qu’une réussite sur les agents n’implique pas une supériorité générale.

### 5.2 Vitesse

Liquid annonce, à **32K de contexte**, au moins **2,5× en préremplissage** et **1,8× en décodage** face aux modèles comparés, sur Ryzen AI Max+ 395, Apple M5 Max et iPhone 17 Pro, en GGUF Q4_0. **Ce n’est pas une mesure sur Ryzen 5700X ou RTX 4060.** [Protocole et résultats](https://huggingface.co/LiquidAI/LFM3-1.7B)

### 5.3 Français

Le français fait partie des **16 langues annoncées**. Aucun score spécifique retrouvé ne permet d’établir la qualité sur vos faits, consignes, transformations ou conversations. [Langues prises en charge](https://huggingface.co/LiquidAI/LFM3-1.7B-Base)

La présence d’une langue dans une fiche indique une cible de prise en charge ; elle ne prouve ni une bonne complétude, ni une résistance aux distracteurs.

## 6. Code, ouverture et reproductions indépendantes

### Artefacts disponibles

Le dépôt public fournit poids, configuration, tokenizer et code personnalisé. L’implémentation Python porte un en-tête Apache 2.0 ; les poids relèvent de la **LFM Open License v1.0**. Cette dernière contient une limitation d’usage commercial liée à un seuil de chiffre d’affaires de 10 millions de dollars. Il faut donc distinguer la licence du code de celle du modèle. [Fichiers](https://huggingface.co/LiquidAI/LFM3-1.7B/tree/main), [licence des poids](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE)

Le support LFM3 apparaît aussi dans le dépôt de llama.cpp. Cela constitue une voie d’exécution distincte, **pas une reproduction du pré-entraînement ou des benchmarks**. [Architecture dans llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp)

### Mesure tierce trouvée

Le rapport public de **crusaderky**, dans `pixi-llm-recipes`, compare plusieurs quantifications des poids et du cache. Il rapporte notamment une configuration à **3,7 Gio**, décomposée en environ **1,4 Gio de poids, 1,2 Gio de contexte et 1,1 Gio de buffers**, pour 256K.

La fidélité est évaluée par divergence KL, accords de tokens et perplexité. L’expression « sans dégradation discernable » doit donc rester limitée à ces mesures. Certains réglages nécessitent BeeLLaMA.cpp ; le rapport relève aussi un ralentissement important pour une variante de cache très compressée. Les tableaux mémoire excluent explicitement les buffers temporaires. [Rapport indépendant](https://github.com/crusaderky/pixi-llm-recipes/blob/main/perplexity/LFM3-1.7B/README.md)

**Statut : mesure tierce publique, non exécutée ici.** Ce rapport soutient la faisabilité locale ; il ne reproduit pas la qualité française ou les scores d’agents. Sa date initiale exacte n’a pas été vérifiée.

### Ce qui manque encore

Aucune reproduction indépendante complète retrouvée pour :

- les gains de benchmarks annoncés ;
- le gain propre au mélange des trois opérateurs ;
- la qualité de l’assistant français ;
- l’efficacité d’un pré-entraînement à quelques centaines de millions de paramètres et quelques milliards de tokens.

## 7. Coût plausible sur le matériel du projet

Les valeurs suivantes sont des **calculs**, pas des mesures locales.

### Poids

À 1,73B paramètres :

- BF16 : environ **3,46 Go décimaux**, soit **3,22 Gio** ;
- 4 bits théoriques : environ **0,865 Go**, avant métadonnées et tenseurs conservés à précision supérieure.

### Cache d’attention

Pour une séquence, des clés/valeurs en 16 bits et une fenêtre locale réellement bornée :

\[
M_{\mathrm{KV}}\approx
2_{\mathrm{K,V}}\times4_{\mathrm{têtes}}\times128
\times2_{\mathrm{octets}}
\times[6T+5\min(T,256)].
\]

Cela donne approximativement :

| Contexte | Cache KV calculé |
|---|---:|
| 1 024 | 14,5 Mio |
| 8 192 | 98,5 Mio |
| 262 144 | 3,00 Gio |

Ces chiffres excluent états convolutionnels, buffers, activations et allocations du moteur. Ils dérivent de la [configuration publiée](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/config.json).

**Conséquence :** BF16 à court contexte paraît compatible avec une enveloppe de 6,5 Gio, mais reste à mesurer. À 256K, les poids et le cache seuls approchent déjà **6,23 Gio**. La quantification change fortement ce bilan.

Pour l’entraînement complet, poids et gradients en FP32 représenteraient déjà **12,9 Gio**, avant optimiseur et activations. Une adaptation légère avec base gelée est une question différente ; **son coût exact sous 6,5 Go n’est pas établi ici**.

## 8. Limites et critiques pour votre objectif

1. **Attribution impossible des gains.** Architecture, données, volume, contexte et post-entraînement changent ensemble. Une comparaison entre produits ne mesure pas l’effet causal d’une convolution ou d’une porte.

2. **Contexte disponible ≠ information correctement utilisée.** La taille du cache mesure une capacité de stockage ; elle ne garantit pas l’extraction fidèle d’une information éloignée.

3. **Les portes ne rendent pas automatiquement le calcul sparse.** Multiplier une tête par un petit coefficient ne supprime pas le calcul de cette tête.

4. **Le contexte 1 024 réduit l’intérêt potentiel du remplacement de l’attention.** Les projections et MLP restent coûteux ; l’économie obtenue à 32K ne se reporte pas proportionnellement.

5. **Le transfert au code maison comporte des interactions concrètes.** Les normalisations actuelles sont sans paramètres et les projections résiduelles sont initialisées à zéro. Ajouter une normalisation de sortie modifie cette dynamique. Par ailleurs, `build_optimizer` affecte tous les paramètres de `blocks.*` à Muon : de nouveaux vecteurs de normalisation ou noyaux convolutionnels exigeraient un traitement explicite.

6. **Le packing SFT doit rester isolé par document.** Une convolution nouvelle ne doit pas propager l’état d’un exemple au suivant. Le masque d’attention existant ne suffit pas à assurer cette propriété pour un nouvel opérateur.

7. **La priorité reste documentée par le cycle 1.** Le défaut dominant concerne l’information absente et la distraction par les passages. Aucun résultat LFM3 ne justifie de remplacer cette priorité par une refonte immédiate.

## 9. Expériences minimales proposées

**Tous les gains locaux attendus sont inconnus.** Les chiffres ci-dessous sont des seuils de décision proposés, non des prédictions. Les expériences GPU seraient ultérieures, sous jeton VertIA, sans concurrence avec le pré-entraînement.

### A. Évaluer LFM3 comme référence française — À TESTER

- **Mécanisme :** utiliser le checkpoint existant pour mesurer la marge accessible avec un modèle local plus entraîné.
- **Gain attendu :** inconnu ; aucun point de benchmark promis.
- **Coût :** poids BF16 ≈ 3,22 Gio ; pic réel et débit à mesurer. Temps total : préremplissage, génération et chargement, relevés séparément.
- **Expérience minimale :** présélection sur 30 cas difficiles, puis les 300 cas du benchmark et le sous-ensemble à passages distracteurs. Comparaison appariée, mêmes textes et critères ; documenter template, traces de raisonnement et troncatures.
- **Critère d’échec :** dépassement mémoire, ou absence d’amélioration robuste de `critere_absent`. Seuil de sélection proposé : **au moins 15 réussites supplémentaires sur 300**, avec intervalle apparié et sans régression majeure par catégorie.
- **Sources :** [modèle](https://huggingface.co/LiquidAI/LFM3-1.7B), [mesure tierce mémoire](https://github.com/crusaderky/pixi-llm-recipes/blob/main/perplexity/LFM3-1.7B/README.md).
- **Statut :** hypothèse locale. Même positif, ce test ne prouverait aucun avantage architectural à budget égal.

### B. Isoler la porte d’attention — À TESTER après les priorités actuelles

- **Mécanisme :** ajouter uniquement une porte sigmoïde par tête après l’attention.
- **Gain attendu :** baisse de perte ou meilleure robustesse à la quantification ; amplitude inconnue.
- **Coût :** sur 24 couches, largeur 1 024 et 16 têtes, **393 216 paramètres supplémentaires**, soit environ **1,5 Mio par copie FP32**. Surcoût temporel et activations à mesurer.
- **Expérience minimale :** témoin 12 × 512, avec/sans porte, même tokenizer et données ; présélection à 100M tokens par bras, confirmation sur trois graines.
- **Critère d’échec :** absence de gain à temps égal ; seuil proposé de **0,01 nat** de baisse, confirmé au-delà du bruit entre graines, sans régression française.
- **Sources :** [mécanisme LFM3](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/modeling_lfm3.py), [étude des portes](https://arxiv.org/abs/2505.06708).
- **Statut :** hypothèse locale ; changement isolable, mais pas correctif d’un checkpoint existant.

### C. Tester le mélange convolution/local/global — À SURVEILLER

- **Mécanisme :** remplacer une partie des attentions globales par les opérateurs locaux.
- **Gain attendu :** économie de temps ou mémoire ; gain chiffré local inconnu.
- **Coût :** nouvelle architecture, caches et gestion des frontières de documents. Pic VRAM inconnu. Pour 100M tokens et débit mesuré \(q\), durée d’un bras : \(100\,000\,000/q\) secondes, plus compilation et validation.
- **Expérience minimale :** vérifier d’abord causalité, isolation des documents et équivalence préremplissage/décodage ; puis dense contre hybride sur témoin, à paramètres comparables et à temps égal.
- **Critère d’échec :** bugs de frontières, dépassement mémoire ou absence de **10 % d’économie de temps à qualité comparable**, seuil proposé. Une amélioration uniquement à nombre de tokens égal ne suffit pas.
- **Sources :** [architecture](https://www.liquid.ai/blog/lfm3), [configuration](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/config.json).
- **Statut :** hypothèse locale, priorité inférieure aux essais SFT et données déjà retenus.

## 10. Tableau des sources

*Les pages évolutives ont été consultées le 18 novembre 2026. Une date non vérifiée est explicitement signalée.*

| Titre / ressource | Date | URL | Statut et usage |
|---|---|---|---|
| Build specialized multimodal models with LFM3 | 27/10/2026 | [Hugging Face](https://huggingface.co/blog/LiquidAI/lfm3) | Annonce collective officielle |
| Introducing LFM3: Every Modality for Specialized Intelligence | Lancement octobre 2026 | [Liquid AI](https://www.liquid.ai/blog/lfm3) | Présentation technique ; résultats fournisseur |
| LFM3-1.7B | Page évolutive | [Fiche](https://huggingface.co/LiquidAI/LFM3-1.7B) | Benchmarks et protocole officiels |
| LFM3-1.7B-Base | Page évolutive | [Fiche Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base) | Taille, données annoncées, langues |
| Configuration LFM3 | Commit visible `40ebe0a`, date non vérifiée | [config.json](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/config.json) | Artefact primaire inspecté |
| Implémentation LFM3 | Commit visible `40ebe0a`, date non vérifiée | [modeling_lfm3.py](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/modeling_lfm3.py) | Code primaire inspecté |
| LFM Open License v1.0 | Date non vérifiée | [LICENSE](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE) | Licence des poids |
| LFM2 Technical Report — Amini et al. | 28/11/2025 | [arXiv](https://arxiv.org/abs/2511.23404) | Rapport technique du prédécesseur |
| Quantizable Transformers — Bondarenko et al. | 22/06/2023 ; v2 09/11/2023 | [arXiv](https://arxiv.org/abs/2306.12929) | Manuscrit consulté ; filiation des portes |
| Gated Attention for Large Language Models — Qiu et al. | 10/05/2025 | [arXiv](https://arxiv.org/abs/2505.06708) | Prépublication consultée ; étude des portes |
| Round and Round We Go! — Barbero et al. | 08/10/2024 ; v3 13/05/2025 | [arXiv](https://arxiv.org/abs/2410.06205) | Manuscrit consulté ; analyse de RoPE |
| LFM3-1.7B quantization report — crusaderky | Date initiale non vérifiée | [GitHub](https://github.com/crusaderky/pixi-llm-recipes/blob/main/perplexity/LFM3-1.7B/README.md) | Mesure indépendante de quantification, non reproduite ici |
| Registre des architectures llama.cpp | Branche évolutive | [Code](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp) | Support d’exécution vérifiable |

**Décision recommandée : évaluer le checkpoint comme référence française, puis n’envisager une reprise architecturale qu’après un essai isolé à temps égal. Les sources disponibles ne démontrent ni une réduction du coût de votre pré-entraînement, ni une correction du défaut `critere_absent`.**