# LexisLM : intérêt et limites pour le pré-entraînement français du « 369M »

*Recherche documentaire arrêtée au 18 novembre 2026. Aucun fichier modifié, aucun calcul GPU lancé.*

## Résumé en 10 lignes

1. **LexisLM est bien le sujet recherché** : la référence donnée correspond aux modèles grecs 502M et 1,09B.
2. Son intérêt pour le projet tient surtout à la proximité de la démarche : apprendre une langue depuis zéro.
3. **Statut : prépublication** ; aucune reproduction indépendante des résultats n’a été trouvée.
4. Les versions Instruct disposent de fiches officielles accessibles et d’exemples d’utilisation.
5. Le code complet de préparation, d’entraînement et d’évaluation n’a pas été retrouvé.
6. Les performances doivent être interprétées séparément pour la prédiction du texte, les questionnaires et le dialogue.
7. Aucun résultat ne démontre une réduction des omissions de votre assistant français.
8. Aucun coût publié vérifié ne permet de promettre une reproduction sur RTX 4060.
9. Les contrôles utiles prolongent les travaux locaux sur les données, la tokenisation et l’évaluation SFT.
10. **Verdict : à surveiller comme référence expérimentale ; aucune raison de modifier le pré-entraînement en cours.**

## 1. Identification, auteurs et statut

Le titre exact est **« Compact Greek Language Models from Scratch: Pretraining 500M- and 1B-Parameter Decoder-Only Transformers »**, déposé le **6 novembre 2026**.

Auteurs : **Konstantinos Perifanos, Anne Laure Poite, Giannis Tsilikas, Ilias Katis, Stavros Deligiannidis et Dionysis Goutsos**. Affiliations indiquées : **National and Kapodistrian University of Athens**, et **lightsaber.gr** ; celle d’Anne Laure Poite n’est pas précisée. [Article original](https://arxiv.org/html/2611.10209v1)

Les candidats se distinguent ainsi :

| Candidat | Correspondance |
|---|---|
| **LexisLM-500M / LexisLM-1B, modèles de base** | Correspondance exacte avec le sujet et les tailles annoncées |
| **LexisLM-500m-Instruct / LexisLM-1B-Instruct** | Variantes après adaptation aux instructions, à distinguer des bases |
| Autre projet portant un nom voisin | Aucun candidat concurrent crédible trouvé pour cette référence |

Les fiches Instruct confirment explicitement leur filiation avec les bases LexisLM. [Fiche 500M](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-500m-Instruct), [fiche 1B](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-1B-Instruct)

**Convention de preuve :**

- **Établi et reproduit** : résultat confirmé indépendamment ; aucun gain propre à LexisLM ne reçoit cette qualification ici.
- **Prépublication** : résultats déclarés par les auteurs, sans validation indépendante retrouvée.
- **Hypothèse locale** : proposition ou extrapolation concernant votre modèle.
- **Calcul** : conséquence arithmétique d’hypothèses explicites, distincte d’une mesure matérielle.

## 2. Principe et mécanisme technique

Le mécanisme général reste celui de votre projet : une séquence de tokens entre dans un décodeur causal ; chaque position apprend à prédire la suivante. La spécialisation linguistique intervient dans les textes d’apprentissage et dans leur découpage en tokens.

**Recette documentée :** grec, Wikipédia/FineWeb2, **22B tokens par modèle**, passage unique ; **RMSNorm, RoPE, SwiGLU, embeddings liés**. Entraînement : **AdamW**, bf16, **335 733 pas × 65 536 tokens**, un dispositif CUDA. [Méthode](https://arxiv.org/html/2611.10209v1#S5)

Les fiches des exports Instruct donnent les dimensions suivantes :

| Propriété | 500M-Instruct | 1B-Instruct |
|---|---:|---:|
| Paramètres annoncés | 502 millions | 1,093 milliard |
| Couches | 16 | 20 |
| Largeur | 1 536 | 2 048 |
| Têtes d’attention | 16 | 16 |
| Contexte | 4 096 | 4 096 |
| Vocabulaire | 32 004 | 32 004 |
| Export | BF16, compatible Llama/Transformers | BF16, compatible Llama/Transformers |

Le tokenizer est un **BPE au niveau des octets**, spécialisé pour le grec et complété par des tokens de dialogue. [Configuration 500M](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-500m-Instruct), [configuration 1B](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-1B-Instruct)

Pour le dépôt, l’intérêt technique se situe dans la comparaison des recettes. `model.py` possède déjà normalisation RMS, RoPE et embeddings liés ; il emploie **ReLU²**, QK-Norm et soft-cap. `train.py` utilise **Muon sur les blocs et AdamW sur les autres paramètres**. Passer à une autre activation ou à AdamW seul constituerait donc une expérience nouvelle : la comparaison entre deux familles de modèles ne permet pas d’isoler le bénéfice de chacun de ces choix.

## 3. Résultats chiffrés et portée réelle

### Pré-entraînement

| Mesure | 500M | 1B |
|---|---:|---:|
| Perplexité de validation | 11,31 | 9,77 |
| ARC-Easy, brut (%) | 44,4 | 46,3 |
| ARC-Challenge, normalisé (%) | 25,3 | 26,7 |
| MMLU (%) | 25,3 | 24,5 |
| TruthfulQA MC1 (%) | 24,7 | 23,9 |
| TruthfulQA MC2 (%) | 42,1 | 41,4 |

**Réserve : exposition au train ARC signalée ; protocole traduit, sans comparaison française contrôlée.** [Résultats et protocole](https://arxiv.org/html/2611.10209v1#S6)

Ces métriques répondent à des questions différentes. La perplexité mesure la probabilité attribuée au texte de référence. Un questionnaire mesure la sélection d’une réponse dans un format particulier. Votre benchmark assistant exige, lui, que le modèle **produise spontanément l’information attendue**.

Par conséquent, aucun de ces chiffres ne peut être converti en gain attendu sur les **49,7 %** de l’assistant v2. Même une amélioration nette de perte locale doit être accompagnée d’une évaluation des réponses générées.

### Après SFT

Les fiches officielles décrivent un mélange **EuroAlpaca grec, Aya grec et données conversationnelles grecques**, avec un checkpoint au pas **3 802**. Les modèles Instruct sont annoncés sous **CC BY-NC 4.0**. [Fiche 500M-Instruct](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-500m-Instruct), [fiche 1B-Instruct](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-1B-Instruct)

| Mesure publiée sur les fiches | 500M-Instruct | 1B-Instruct |
|---|---:|---:|
| ARC-Easy, normalisé | 41,58 % | 44,15 % |
| ARC-Challenge, normalisé | 25,26 % | 27,91 % |
| MMLU | 26,04 % | 24,67 % |
| TruthfulQA MC1 | 23,13 % | 22,52 % |
| TruthfulQA MC2 | 41,35 % | 40,90 % |

ARC emploie **25 démonstrations**, MMLU **5**, TruthfulQA **0**. Ces évaluations ne constituent pas un test de suivi d’instructions. [Protocole et scores 500M](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-500m-Instruct), [scores 1B](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-1B-Instruct)

**Conséquence pour votre SFT :** choisir un checkpoint uniquement sur sa perte peut manquer une régression de connaissances ou de comportement. Votre roadmap prévoit déjà FrenchBench et le benchmark assistant : cette séparation reste pertinente.

## 4. Code publié et reproductibilité

L’accès documentaire obtenu est incomplet :

| Élément | Vérification effectuée |
|---|---|
| Article intégral | HTML et extraction du PDF consultés |
| Fiches officielles Instruct | Accessibles, exemples de chargement présents |
| Fichiers des modèles et configurations brutes | Navigation non aboutie avec l’outil web |
| Fiches des modèles de base | Accès direct non abouti |
| Code complet d’entraînement | Non trouvé |
| Scripts complets d’évaluation | Non trouvés |
| Reproduction indépendante | Non trouvée |
| Critique indépendante avec nouvelles expériences | Non trouvée |

La fiche 500M mentionne un export par `scripts/export_hf.py`. **Une référence à un chemin de script ne prouve pas que ce script soit publiquement accessible.** [Provenance de l’export](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-500m-Instruct)

Les recherches par nom, identifiant arXiv, GitHub et termes de reproduction ont surtout retrouvé des reprises du papier. Elles ne permettent pas d’affirmer que les artefacts manquants n’existent pas ; elles permettent seulement de dire qu’ils n’ont pas été vérifiés.

## 5. Compute et faisabilité sur votre machine

**GPU précis, durée, débit, pic VRAM et consommation énergétique : non trouvés dans le rapport consulté.** [Rapport](https://arxiv.org/html/2611.10209v1)

### Rectifications locales importantes

Le code donne **335 544 320 paramètres uniques** au run appelé « 369M » :

\[
N=32768\times1024+24\times12\times1024^2.
\]

Les normalisations du modèle ne portent pas de paramètres appris ; les embeddings d’entrée et de sortie sont liés. Ce décompte concorde avec les remarques de la veille frugalité.

Le budget exact configuré vaut :

\[
113000\times2\times32\times1024
=7\,405\,568\,000\ \text{tokens}.
\]

À **6 300 tok/s**, cela représente **326,5 heures**, soit **13,6 jours continus**. Les quelque 12 jours correspondent à un débit plus proche de 7 000 tok/s. Le lanceur réserve 6,5 Go et passe un plafond PyTorch de **6,1 Gio**. [Modèle](D:/ServOMorph/creation_llm/model.py), [lanceur](D:/ServOMorph/creation_llm/run_369m.py), [gestion mémoire](D:/ServOMorph/creation_llm/train.py)

### Mémoire : estimation explicite, sans mesure GPU

Pour distinguer stockage des poids et entraînement, prenons deux scénarios arithmétiques :

- inférence : **2 octets par paramètre**, poids BF16 seuls ;
- entraînement AdamW classique : **16 octets par paramètre**, avec poids, gradients et deux moments FP32.

| Taille nominale | Poids BF16 seuls | États du scénario AdamW |
|---|---:|---:|
| 502M | 1,00 Go | 8,03 Go |
| 1,093B | 2,19 Go | 17,49 Go |

Ces montants excluent activations, buffers temporaires et contexte CUDA. Le second scénario **n’est pas une mesure de l’implémentation LexisLM** : d’autres choix de précision ou de déchargement changeraient le résultat.

Il suffit néanmoins à montrer pourquoi la taille des poids téléchargeables ne permet pas de conclure à la faisabilité d’un pré-entraînement. Avec votre plafond, une reproduction demanderait une stratégie mémoire différente, puis un test de débit.

À titre de simple conversion, **22B tokens à 6 300 tok/s demanderaient 40,4 jours**. Ce n’est pas une prévision pour LexisLM : son débit local est inconnu.

La relation entre taille et budget de données relève d’un compromis expérimental. Les travaux Chinchilla étudient ce compromis à calcul fixé ; ils ne fournissent pas un optimum universel applicable automatiquement au français et à Muon. [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)

## 6. Limites et contre-expertise

Les points suivants sont **mon analyse**, sans prétendre constituer une critique indépendante publiée.

**Comparaison causale insuffisante pour choisir une recette locale.** Pour attribuer un gain à la spécialisation monolingue, il faudrait un témoin multilingue à architecture, budget, qualité des données et protocole comparables. Pour attribuer un gain au tokenizer, il faudrait une ablation correspondante. Une recette complète ne permet pas de départager ces facteurs.

**Le budget pertinent est le temps total sur votre machine.** Augmenter simultanément taille, contexte et volume peut améliorer un score tout en devenant défavorable sous une contrainte de plusieurs semaines sur un seul GPU.

**Les perplexités entre langues ne classent pas les modèles.** Leur valeur dépend du tokenizer et de la distribution du texte. Même les bits par octet doivent être comparés sur des textes et un protocole cohérents ; ils ne transforment pas deux corpus linguistiques différents en test identique.

**Le diagnostic « critère absent = capacité insuffisante » reste incomplet.** Dans le bilan local, fournir un passage pertinent relu fait passer les faits de **49/90 à 71/90**. Une partie du problème concerne donc l’accès à l’information et son exploitation. L’augmentation de taille reste une hypothèse utile, mais elle ne remplace pas ce diagnostic. [Bilan du cycle 1](D:/ServOMorph/creation_llm/rd/bilan_cycle1.md)

**L’absence de reproduction limite la confiance dans les petites différences.** Avant toute décision coûteuse, il faudrait disposer des révisions des données, prompts exacts, checkpoints et sorties par question, puis refaire le scoring.

## 7. Liens avec les recherches déjà présentes

L’index et les cinq veilles ont été consultés, avec lecture ciblée de leurs sections pertinentes. Aucun dossier LexisLM n’apparaît dans l’index consulté.

| Travail local | Relation avec ce rapport |
|---|---|
| `veille_entrainement.md` | Tokenisation, préparation des séquences et sélection des données sont déjà étudiées : aucune nouvelle priorité architecturale justifiée ici |
| `veille_frugalite.md` | Les distinctions données uniques/répétitions et temps total/tokens restent centrales |
| `pole_mathematiques.md`, idée I6 | Prolonger les mesures de tokenisation ; éviter une comparaison directe des perplexités |
| `veille_inference.md` et `bilan_cycle1.md` | Conserver le diagnostic des omissions et la piste RAFT issue des expériences locales |
| `veille_moe_chine.md` | Maintenir la distinction entre paramètres stockés, calcul et mémoire d’entraînement |
| `veille_lecun_jepa.md` | Même exigence : une meilleure métrique intermédiaire doit se traduire en réponses meilleures |
| Index H-Net, NCM, LFM3 | LexisLM ajoute un point de comparaison documentaire ; il ne justifie pas de rouvrir les conclusions de ces dossiers |

Ces rapprochements évitent de présenter comme nouveauté une nouvelle proposition d’EMA, de filtrage, de distillation ou de changement de tokenizer. [Index R&D](D:/ServOMorph/creation_llm/rd/recherches/INDEX.md), [sélection critique](D:/ServOMorph/creation_llm/rd/selection.md)

## 8. Propositions falsifiables pour la cellule

**Tous les gains locaux ci-dessous sont inconnus.** Les nombres proposés sont des seuils de décision ou des budgets expérimentaux, pas des prévisions.

### A. Compléter l’audit CPU de tokenisation — priorité raisonnable

- **Mécanisme :** mesurer la fragmentation par occurrence et par forme distincte, les tokens par caractère, les textes rares et les cas Unicode ; distinguer compression et compréhension.
- **Gain attendu chiffré :** inconnu ; l’audit seul ne change pas les performances du modèle.
- **Coût :** **0 VRAM** ; temps CPU à mesurer sur un premier lot, puis extrapoler au volume retenu.
- **Expérience minimale :** échantillon tenu à l’écart, stratifié par les quatre sources ; conserver accents, ponctuation, nombres et formes rares.
- **Critère d’échec :** aucune anomalie reproductible ni différence exploitable ; ne pas en déduire la nécessité de remplacer le tokenizer.
- **Sources :** [tokenizer LexisLM](https://arxiv.org/html/2611.10209v1#S3), [travail local I6](D:/ServOMorph/creation_llm/rd/pole_mathematiques.md).

### B. Auditer conjointement perte SFT et omissions — priorité haute, déjà compatible avec la roadmap

- **Mécanisme :** vérifier si les checkpoints améliorant la perte améliorent aussi la complétude, les connaissances et les consignes.
- **Gain attendu chiffré :** inconnu ; objectif de décision proposé : **au moins +10 réussites sur 300**, soit **+3,33 points**, sans régression importante par catégorie.
- **Coût :** aucun entraînement supplémentaire si les checkpoints existent ; coût de génération \(T=N_{\text{tokens}}/q\), avec débit \(q\) à mesurer. Pic VRAM d’inférence à mesurer.
- **Expérience minimale :** comparer les étapes SFT prévues, mêmes items et paramètres de génération ; trois graines ; relecture aveugle d’un échantillon des omissions.
- **Critère d’échec :** amélioration de perte sans progrès des réponses, gain fragile au rééchantillonnage des questions, ou régression FrenchBench.
- **Sources :** [limites du protocole Instruct](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-500m-Instruct), [roadmap locale](D:/ServOMorph/creation_llm/roadmap_modele_369m.md).

Le test final ne doit pas servir à fabriquer les données SFT. Les générations répétées d’une question ne sont pas des questions indépendantes.

### C. Comparer couverture avec et sans remise — priorité secondaire

- **Mécanisme :** vérifier si une meilleure couverture du corpus apporte davantage que des fenêtres tirées avec remise.
- **Gain attendu chiffré :** inconnu ; seuil proposé : **0,01 nat/token** de mieux, confirmé à temps comparable.
- **Coût :** modèle inchangé ; index CPU supplémentaire. Pour deux bras de 100M tokens et trois graines : **600M tokens**, soit \(600\,000\,000/q\) secondes hors évaluations.
- **Expérience minimale :** témoin 12 × 512 déjà envisagé par la cellule ; mêmes proportions de sources, longueur, optimiseur et schedule ; seul le mode de parcours change.
- **Critère d’échec :** gain absent, inférieur à la variabilité entre graines, ou obtenu seulement par un changement involontaire du mélange.
- **Sources :** [parcours des données LexisLM](https://arxiv.org/html/2611.10209v1#S4), [audit local des répétitions](D:/ServOMorph/creation_llm/rd/veille_frugalite.md).

Cette expérience doit rester séparée de l’alignement documentaire et du masque inter-documents, faute de quoi les causes du gain seraient indissociables.

## 9. Tableau des sources

| Titre / ressource | Date | URL | Statut et vérification |
|---|---|---|---|
| Article LexisLM, titre complet en section 1 | 06/11/2026, v1 | [arXiv](https://arxiv.org/abs/2611.10209) | Prépublication ; identité vérifiée |
| Texte intégral LexisLM | 06/11/2026 | [HTML](https://arxiv.org/html/2611.10209v1) · [PDF](https://arxiv.org/pdf/2611.10209) | Source primaire consultée |
| LexisLM-500m-Instruct | Date initiale non vérifiée ; consulté le 18/11/2026 | [Hugging Face](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-500m-Instruct) | Fiche officielle ; configurations et scores déclarés |
| LexisLM-1B-Instruct | Date initiale non vérifiée ; consulté le 18/11/2026 | [Hugging Face](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-1B-Instruct) | Fiche officielle ; configurations et scores déclarés |
| LexisLM-500m, base | Non vérifiée | [Dépôt indiqué](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-500m) | Accès non abouti ; fichiers et licence non vérifiés |
| LexisLM-1B, base | Non vérifiée | [Dépôt lié par la fiche Instruct](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-1B) | Accès non abouti ; fichiers et licence non vérifiés |
| *Training Compute-Optimal Large Language Models* | 29/03/2022, dépôt initial | [Hoffmann et al.](https://arxiv.org/abs/2203.15556) | Source primaire de cadrage ; aucun optimum local déduit |

**Décision proposée : À SURVEILLER.** LexisLM enrichit les références de pré-entraînement monolingue à petite échelle. Les actions immédiatement défendables concernent la qualité des mesures ; les priorités expérimentales restent celles du bilan local et de la SFT du 369M.