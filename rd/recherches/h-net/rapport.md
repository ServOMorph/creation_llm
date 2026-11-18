# H-Net : découpage dynamique et hiérarchie apprise — intérêt pour le LLM français

*Recherche documentaire au 18 novembre 2026. Aucun fichier du dépôt modifié ; aucun calcul GPU lancé.*

## Résumé en 10 lignes

1. **H-Net est bien le sujet recherché** : un réseau hiérarchique qui apprend à regrouper les octets pendant l’entraînement.
2. Le travail initial est signé **Sukjun Hwang, Brandon Wang et Albert Gu**, de Carnegie Mellon et Cartesia.
3. Diffusé en juillet 2025, il est désormais **publié à ICLR 2026**.
4. Il remplace la tokenisation fixe par des représentations continues, compressées à plusieurs niveaux.
5. Le code initial et des poids sont publics ; cela ne constitue pas une recette complète reproduite sur votre matériel.
6. Les résultats initiaux favorisent surtout la version à deux niveaux, à données et FLOPs comparables.
7. Une nouvelle version annoncée par Cartesia en octobre 2026 revendique des économies beaucoup plus fortes.
8. Son annonce « environ 10× » reste un résultat des auteurs, avec des différences de données et une architecture améliorée non intégralement publiée.
9. Des expériences indépendantes à petite échelle sont encourageantes, mais montrent aussi un surcoût en temps réel.
10. **Verdict : à tester séparément pour un prochain pré-entraînement ; aucune justification pour remplacer le run actuel.**

## 1. Identification et niveau de preuve

### Quel candidat correspond ?

| Candidat | Correspondance avec la description | Conclusion |
|---|---|---|
| **H-Net — Dynamic Chunking for End-to-End Hierarchical Sequence Modeling** | Découpage appris, dépendant du contexte, hiérarchie récursive | **Identification principale** |
| **H-Net amélioré — From Tokens to Bytes**, octobre 2026 | Suite récente du même programme, avec nouveaux résultats d’efficacité | Probablement pertinent si la veille mentionnait « 10× » |
| **H-Net++** | Extension indépendante orientée langues à morphologie riche | À distinguer de la suite officielle |
| **Byte Latent Transformer — BLT** | Modèle par octets avec groupes de taille variable déterminés par l’entropie | Alternative proche, mais mécanisme différent |

Sources : [publication H-Net](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f1aa53cac69ef6980ca4a911ffcf278b-Abstract-Conference.html), [suite Cartesia](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1), [H-Net++](https://arxiv.org/abs/2508.05628), [BLT](https://arxiv.org/abs/2412.09871).

### Auteurs, organisations et chronologie

Le travail initial associe **Sukjun Hwang — Carnegie Mellon University**, **Brandon Wang — Cartesia AI**, et **Albert Gu — Carnegie Mellon University et Cartesia AI**. Première version arXiv : **10 juillet 2025** ; deuxième version : **15 juillet 2025**. La publication à **ICLR 2026** est confirmée par les actes officiels. Les deux billets récents sont signés collectivement **Cartesia Research**, datés du **28 octobre 2026**. [Article initial](https://arxiv.org/abs/2507.07955), [actes ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f1aa53cac69ef6980ca4a911ffcf278b-Abstract-Conference.html), [billet II](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-2).

Conventions utilisées dans ce rapport :

- **Établi** : résultat publié et reproduit indépendamment dans un cadre identifiable.
- **Publié, reproduction partielle** : publication confirmée ; expériences indépendantes connexes, sans réplication exacte du résultat principal.
- **Prépublication / rapport technique** : résultat accessible, sans validation scientifique indépendante établie.
- **Hypothèse locale** : proposition pour votre projet ; aucun gain mesuré.

**H-Net relève ici de « publié, reproduction partielle ». Les gains de la nouvelle version de 2026 restent des résultats de communication technique des auteurs.**

## 2. Rapport avec les recherches déjà présentes

L’[index des recherches](D:/ServOMorph/creation_llm/rd/recherches/INDEX.md) ne contient encore aucune entrée. Les cinq veilles présentes ont été consultées : entraînement, inférence, MoE/Chine, frugalité et JEPA. H-Net n’y fait pas l’objet d’une étude dédiée. `DOCUMENTATION/INDEX.md` est absent.

Les liens utiles, sans reprendre ces veilles :

| Travail déjà étudié | Apport distinct de H-Net |
|---|---|
| **SuperBPE**, veille entraînement | Même question : quelle unité donner au modèle ? H-Net rend le regroupement dépendant du contexte et de l’apprentissage du modèle. |
| **Token-Superposition Training**, veille frugalité | Même levier de réduction des positions coûteuses ; ici, la hiérarchie fait partie du modèle génératif final. |
| **Hybrides attention–récurrence**, veille frugalité | Les couches récurrentes peuvent servir aux niveaux fins, l’attention aux niveaux compressés. |
| **MoE**, veille Chine | Question commune du calcul conditionnel ; il faut toujours distinguer paramètres stockés et calcul effectivement exécuté. |
| **JEPA**, veille LeCun | Vocabulaire commun d’abstraction, mais H-Net reste un modèle génératif prédisant les octets. |
| **Bits par octet**, pôle mathématiques | Métrique indispensable pour comparer votre BPE et un modèle par octets. |

Le [bilan du cycle 1](D:/ServOMorph/creation_llm/rd/bilan_cycle1.md) reste déterminant : avec un passage pertinent, l’assistant passe de **49/90 à 71/90** en faits. Le défaut `critere_absent` ne démontre donc pas à lui seul un problème de tokenisation. **H-Net ne constitue pas une réponse établie à ce défaut.**

## 3. Principe et mécanisme technique

### 3.1 Architecture générale

Le schéma conceptuel est :

```text
Octets UTF-8
    ↓
Encodeur causal
    ↓
Sélection apprise de positions représentatives
    ↓
Réseau principal sur une séquence raccourcie
    ↓
Reconstruction à la résolution des octets
    ↓  + connexion provenant de l’encodeur
Décodeur causal
    ↓
Distribution du prochain octet
```

Le réseau principal peut lui-même être un H-Net : on obtient alors plusieurs niveaux de compression. Les groupes ne sont pas des identifiants dans un dictionnaire BPE ; ils correspondent à des **vecteurs contextuels**. La présentation institutionnelle décrit cette combinaison encodeur–réseau principal–décodeur et son apprentissage conjoint. [Présentation Cartesia](https://www.cartesia.ai/blog/hierarchical-modeling).

**Conséquence pratique :** exporter une liste de frontières puis continuer à utiliser le GPT actuel ne reproduirait pas cette architecture. Les représentations, leur calcul et la génération changent ensemble.

### 3.2 Routeur et reconstruction

Dans le code public, le routeur compare deux représentations voisines projetées et normalisées. Avec une notation adaptée au code :

\[
p_t=\frac{1-\cos(W_qh_{t-1},W_kh_t)}{2}.
\]

Une décision binaire autour de \(0{,}5\) détermine les positions conservées ; le début de séquence est forcé comme frontière. Le réseau interne reçoit les vecteurs sélectionnés.

La reconstruction utilise un lissage causal :

\[
\bar z_j=P_jz_j+(1-P_j)\bar z_{j-1},
\]

puis redistribue les représentations aux positions d’origine. Le code gère également des séquences empaquetées et des états de génération. [Implémentation officielle du découpage](https://github.com/goombalab/hnet/blob/main/hnet/modules/dc.py).

Il faut distinguer **octet**, **caractère Unicode** et **mot**. Un accent peut occuper plusieurs octets ; aucune frontière linguistique correcte n’est garantie par le simple choix d’une entrée UTF-8.

### 3.3 Comment apprendre une décision discrète ?

Le code combine le chemin de reconstruction avec un **estimateur de gradient direct**, ou STE : une opération garde une valeur constante en passe avant, mais transmet un gradient en passe arrière. Le modèle conserve aussi un chemin résiduel entre encodeur et décodeur. Il s’agit d’une construction d’optimisation, pas d’une dérivation exacte de la sélection discrète. [Structure officielle et STE](https://github.com/goombalab/hnet/blob/main/hnet/models/hnet.py).

Une perte auxiliaire contrôle la fréquence des frontières. Pour un facteur de compression cible \(N\), une fraction effectivement sélectionnée \(F\), et une moyenne des scores \(G\), sa forme initiale est :

\[
\mathcal L_{\mathrm{ratio}}
=\frac{N}{N-1}\left[(N-1)FG+(1-F)(1-G)\right].
\]

L’analyse indépendante de **deklan**, publiée le 23 septembre 2025, étudie son gradient et ses liens avec une pénalisation du taux de sélection. Elle souligne aussi le rôle des mécanismes de confiance du routeur. C’est une **analyse technique indépendante**, pas une reproduction des scores linguistiques. [Analysis of H-Net Router](https://www.deklan.dev/hnet-router).

**Interprétation :** les frontières s’apprennent, mais le budget de compression reste en partie choisi par le concepteur. « Appris de bout en bout » ne signifie pas « sans hyperparamètres ».

## 4. Résultats chiffrés : séparer les générations

### 4.1 Travail initial, publié à ICLR 2026

Entraînement principal : sous-ensemble **100 milliards de tokens de FineWeb-Edu**, AdamW et WSD ; fenêtres de **8 192 octets**, contre **1 792 tokens GPT-2**. Comparaison à données et FLOPs rapprochés.

| Modèle | Paramètres | GFLOPs/octet, évaluation | Validation BPB ↓ | Moyenne de sept tâches ↑ |
|---|---:|---:|---:|---:|
| Transformer L | 760M | 0,42 | 0,756 | 53,3 |
| H-Net L, un niveau | 680M | 0,43 | 0,755 | 53,6 |
| H-Net L, deux niveaux | 870M | 0,43 | 0,743 | 55,5 |
| Transformer XL | 1,3B | 0,69 | 0,730 | 55,5 |
| H-Net XL, deux niveaux | 1,6B | 0,69 | 0,715 | 58,2 |

Autres résultats : **59,9 → 66,3 %** sur XWinograd chinois ; **3,6×** d’efficacité en données sur l’expérience ADN. Les auteurs signalent une implémentation pouvant être **jusqu’à environ 2× plus lente** à l’entraînement. [Article, tableaux et limites](https://arxiv.org/html/2507.07955v2) ; [version publiée, tableau 2 concordant](https://proceedings.iclr.cc/paper_files/paper/2026/file/f1aa53cac69ef6980ca4a911ffcf278b-Paper-Conference.pdf).

**Lecture critique :** le rapprochement de scores 55,5 contre 55,5 concerne ici **870M contre 1,3B paramètres totaux**, soit environ **1,49×**, calculé à partir du tableau. La formule « deux fois plus grand » ne doit donc pas devenir une promesse de division par deux de la mémoire.

Le gain du modèle à un niveau est faible dans cette comparaison. Cela justifie de tester la hiérarchie elle-même, plutôt que de supposer que supprimer BPE suffit.

**Non trouvé :** bilan complet en heures-GPU, matériel et pics VRAM permettant une conversion fiable vers la RTX 4060.

### 4.2 Nouvelle version Cartesia, octobre 2026

Deux expériences doivent être distinguées :

| Expérience | Résultat annoncé et cadre |
|---|---|
| Échelle contrôlée, un niveau | Muon ; modèles BPE de 221M à 2,62B ; mélange anglais/code/math/multilingue ; **1,28×** en efficacité calcul sur validation, **1,83×** sur Paloma et **2,06×** sur OLMES |
| Deux niveaux contre SmolLM3 | **3,67B contre 3,08B paramètres** ; **4,44 contre 4,36 GFLOPs/octet d’entraînement** ; score moyen final rejoint vers **1T tokens équivalents**, contre environ **11T** |

Le grand essai utilise AdamW. Les données reconstruisent au mieux les sources publiques de la première phase de SmolLM3 ; les phases ultérieures du témoin et son refroidissement diffèrent. Le nouveau mécanisme n’est pas intégralement publié ; Cartesia appelle rétrospectivement l’ancien **LH-Net**, pour *lazy/lookahead*. [Rapport Cartesia I](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1).

**Statut : rapport technique des auteurs, non reproduction indépendante du facteur 10.** Les multiplicateurs de l’échelle contrôlée concernent des courbes de perte ; ils ne représentent pas des multiplications de précision ni des accélérations murales.

### 4.3 Comparaison récente avec BLT

Le second billet rapporte un multiplicateur calcul-vers-OLMES de **1,84× pour “BLT-like”** et **2,06× pour BPE**, relativement à H-Net. Mais « BLT-like » conserve l’organisation de leur H-Net et change la règle de découpage : ce n’est pas une reproduction intégrale de BLT. Le coût du modèle auxiliaire d’entropie est même exclu, à l’avantage de ce comparateur. [Rapport Cartesia II](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-2).

Cette ablation soutient l’intérêt du découpage conjointement appris dans ce cadre. Elle ne permet pas de déclarer H-Net universellement supérieur à toutes les versions de BLT.

## 5. Code, reproductions et critiques indépendantes

### 5.1 Ressources officielles

Le [dépôt `goombalab/hnet`](https://github.com/goombalab/hnet) fournit l’architecture, les configurations et un script de génération. Il annonce des checkpoints anglais L/XL à un ou deux niveaux, ainsi que chinois et code. **Je n’ai pas identifié dans la structure consultée un pipeline complet de pré-entraînement reproduisant toutes les expériences.**

Les dépendances comprennent PyTorch, Triton, Mamba, FlashAttention et `causal_conv1d`. Leur présence dans le manifeste ne garantit pas une installation fonctionnelle dans votre environnement Windows. [Manifeste officiel](https://github.com/goombalab/hnet/blob/main/pyproject.toml).

### 5.2 Réimplémentation indépendante sur une RTX 5090

Le dépôt **`bigwolfeman/hnet-research`** annonce des modèles d’environ **42M**, entraînés sur **12,58 milliards d’octets**, soit environ **\(8\times10^{17}\) FLOPs**.

Résultats déclarés :

- H-Net à deux niveaux : **1,0905 / 1,0979 / 1,0913 BPB**, trois graines ;
- BPE : **1,1156 / 1,1109 BPB**, deux graines ;
- H-Net à un niveau : **1,1581 BPB** ;
- temps : **73 contre 50 ms/pas**, soit **1,46× plus long**, à FLOPs rapprochés.

L’auteur ne retrouve pas certains comportements de segmentation contextuelle de la nouvelle version, et utilise l’ancien routeur public. **Statut : expérience indépendante avec code et résultats déclarés, non évaluée par les pairs ; non exécutée par moi.** [Dépôt et comptes rendus](https://github.com/bigwolfeman/hnet-research).

C’est une piste utile pour construire un témoin réduit, mais la RTX 5090 ne valide ni votre plafond mémoire ni votre débit.

### 5.3 Expériences Parameter Golf

**Darius Feher** publie des essais sous contrainte d’artefact de **16 Mo**, généralement sur **8 H100** :

| Essai déclaré | BPB ↓ |
|---|---:|
| H-Net par octets, 10 minutes | 1,4116 ± 0,013 |
| H-Net sur sous-mots, 10 minutes | 1,3734 |
| H-Net par octets, 4 heures | 1,3595 |

Le dépôt rapporte une amélioration ultérieure réduisant l’écart. Il s’agit d’une exploration indépendante à petite taille, pas d’une réplication exacte du papier. **16 Mo d’artefact ne signifie pas 16 Mo de mémoire d’entraînement.** [Code et protocoles](https://github.com/DariusFeher/parameter-golf-byte-hnet).

### 5.4 H-Net++ et autres implémentations

**H-Net++**, de Mehrdad Zakershahrak et Samira Ghodratnama, annonce sur un corpus persan de **1,4B tokens** une réduction de **0,159 BPB** face à GPT-2-fa et **+5,4 points sur ParsGLUE**. C’est une **prépublication indépendante**, avec modifications propres ; je n’ai pas vérifié de reproduction externe ni de code officiel exploitable. Elle ne valide pas le gain de H-Net en français. [H-Net++](https://arxiv.org/abs/2508.05628).

Une [implémentation de lucidrains](https://github.com/lucidrains/h-net-dynamic-chunking) existe également. L’existence d’un code tiers ne suffit pas à établir une reproduction des résultats.

### 5.5 Limites à retenir

**Analyse pour la cellule :**

1. Une meilleure BPB n’établit pas une amélioration des réponses factuelles après SFT.
2. Des frontières visuellement « sémantiques » ne démontrent pas une meilleure capacité de raisonnement.
3. À FLOPs égaux, les opérations irrégulières peuvent pénaliser les petits GPU.
4. La longueur interne dépend des données : le pic mémoire importe davantage que le ratio moyen.
5. Comparer des pertes par token serait invalide ; comparer des nombres de tokens lus le serait également.
6. Les résultats 2026 ne doivent pas être attribués automatiquement au code public de 2025.
7. Aucun résultat consulté n’établit le bénéfice sur votre combinaison français, Muon, ReLU², QK-Norm, soft-cap et budget de données.

## 6. Application concrète au dépôt

### 6.1 Trois corrections de cadrage

La lecture de [model.py](D:/ServOMorph/creation_llm/model.py), [train.py](D:/ServOMorph/creation_llm/train.py) et [run_369m.py](D:/ServOMorph/creation_llm/run_369m.py) confirme :

- **335 544 320 paramètres uniques** avec embeddings liés ; « 369M » est le nom du run.
- Réservation GPU de **6,5**, mais plafond PyTorch passé à **6,1 Gio**.
- **7 405 568 000 tokens** prévus ; à 6 300 tok/s constants, environ **13,6 jours**, hors interruptions.

Ce sont des calculs sur la configuration, pas de nouvelles mesures.

### 6.2 Tokens, octets et contexte

Le [pôle mathématiques](D:/ServOMorph/creation_llm/rd/pole_mathematiques.md) rapporte **4,45 octets/token** sur le mélange.

En première approximation :

\[
1024\ \text{tokens}\approx4557\ \text{octets},
\]

\[
7{,}406\ \text{milliards de tokens}\approx33{,}0\ \text{milliards d’octets traités}.
\]

Ces conversions doivent être recalculées sur les documents exacts du futur essai. **Un H-Net à contexte 1 024 octets disposerait de beaucoup moins de texte que votre GPT à 1 024 tokens.**

Autre piège : conserver les proportions `0,58/0,25/0,15/0,02` en passant directement à un échantillonnage par octets modifierait le mélange effectif. Il faut sélectionner les mêmes documents ou segments avant les deux encodages.

### 6.3 Mémoire : économie partielle, bilan inconnu

À largeur 1 024, remplacer théoriquement une table liée de 32 768 entrées par 256 entrées retire :

\[
(32768-256)\times1024=33\,292\,288\ \text{paramètres}.
\]

Cela représente **127 Mio de poids FP32**, ou **63,5 Mio en BF16**, avant gradients et états d’optimiseur.

**Ce calcul n’est pas une économie nette prédite pour H-Net** : il faut ajouter encodeurs, décodeurs, routeurs, activations sur les longues séquences et espaces de travail des kernels.

### 6.4 Changements nécessaires pour une expérimentation

L’analyse du code local indique plusieurs points d’intégration :

- `GPT.forward` et son cache supposent une séquence homogène ; une hiérarchie demande des états par niveau.
- `build_optimizer` affecte Muon selon le préfixe `blocks.` ; une nouvelle arborescence imposerait un classement explicite des paramètres.
- La boucle écrase le taux d’apprentissage de tous les groupes ; des multiplicateurs par niveau demanderaient une adaptation.
- [train_sft.py](D:/ServOMorph/creation_llm/train_sft.py) masque les cibles et isole les exemples avec `doc_ids` : les états récurrents et le découpage devraient respecter les mêmes frontières.
- Les budgets de génération devraient être comparés en texte produit, pas avec un identique « 300 tokens » devenu « 300 octets ».

**Une conversion directe du checkpoint actuel n’est pas disponible dans votre code.** Une éventuelle transplantation de poids serait une expérience distincte.

## 7. Propositions falsifiables

Tous les chiffres de décision ci-dessous sont des **seuils proposés**, pas des gains attendus démontrés. Les expériences GPU resteraient soumises au jeton VertIA et à la priorité du run actuel.

### Proposition A — Vérifier la faisabilité d’un prototype réduit

| Élément | Proposition |
|---|---|
| **Statut** | Hypothèse locale |
| **Mécanisme** | Petit H-Net public à un puis deux niveaux ; mêmes textes français que le témoin |
| **Gain attendu chiffré** | **Inconnu** ; cette étape vérifie fonctionnement et coût |
| **VRAM** | Pic inconnu ; respecter 6,1 Gio alloués et l’enveloppe de réservation de 6,5 |
| **Temps** | Enveloppe proposée : **30 minutes GPU maximum**, hors installation CPU |
| **Expérience minimale** | Passe avant/arrière, gradients du routeur, 200 pas, préremplissage puis génération incrémentale ; tester aussi des séquences provoquant beaucoup de frontières |
| **Critère d’échec** | Dépassement mémoire, gradients non finis, fuite entre documents ou incohérence entre génération incrémentale et passe complète |
| **Sources** | [Code officiel](https://github.com/goombalab/hnet), [dépendances](https://github.com/goombalab/hnet/blob/main/pyproject.toml) |

Un échec d’installation ne réfuterait pas l’architecture ; il invaliderait sa faisabilité immédiate dans cet environnement.

### Proposition B — Comparer qualité et temps sur un petit modèle français

| Élément | Proposition |
|---|---|
| **Statut** | Hypothèse locale, expérience principale |
| **Mécanisme** | Comparer GPT-BPE, H-Net à un niveau et H-Net à deux niveaux ; ajouter un découpage fixe pour isoler l’effet du routeur |
| **Gain attendu chiffré** | **Inconnu** ; seuil d’intérêt proposé : **10 % de temps économisé** vers une BPB cible, ou **0,01 BPB gagnée à temps égal** |
| **VRAM** | À mesurer pour chaque bras ; paramètres totaux et activations inclus |
| **Temps** | Pour \(D\) octets par bras et débits \(q_i\), coût d’une graine : \(\sum_i D/q_i\), plus évaluations et compilation |
| **Expérience minimale** | Présélection sur **\(10^9\) octets traités par bras** ; confirmer les finalistes sur trois graines |
| **Critère d’échec** | Gain absent à temps égal, gain non confirmé entre graines, ou régression persistante sur une source |
| **Sources** | [H-Net](https://arxiv.org/abs/2507.07955), [essai indépendant réduit](https://github.com/bigwolfeman/hnet-research) |

Mesures nécessaires :

\[
\mathrm{BPB}
=\frac{\sum -\log p(\text{cible}\mid\text{contexte})}
{\ln 2\times\text{nombre d’octets évalués}}.
\]

Utiliser les mêmes textes, frontières documentaires et horizons de contexte. Rapporter séparément données uniques, octets traités, FLOPs et temps réel. Pour BPE, cette métrique décrit le codage via sa segmentation canonique ; ne pas la présenter comme une marginalisation exacte sur toutes les segmentations possibles.

Le budget court sert au tri. Un résultat négatif à ce stade ne prouverait pas l’absence de gain à plus grande échelle.

### Proposition C — Vérifier le transfert à l’assistant, seulement après B

| Élément | Proposition |
|---|---|
| **Statut** | Hypothèse locale conditionnelle |
| **Mécanisme** | SFT des deux meilleures bases, avec mêmes exemples et masquage des réponses adapté aux octets |
| **Gain attendu chiffré** | **Inconnu** ; seuil proposé : **+10 réussites sur 300**, soit +3,33 points |
| **VRAM** | Pic SFT à mesurer ; ne pas déduire sa valeur du seul pré-entraînement |
| **Temps** | Deux SFT et évaluations ; durée non estimable avant mesure des débits |
| **Expérience minimale** | Comparaison appariée par question, trois graines de génération, audit des omissions et des erreurs sur accents/noms/nombres |
| **Critère d’échec** | BPB améliorée sans progrès des réponses, ou régression confirmée d’une catégorie importante |
| **Sources** | [Discussion du transfert](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-2), [bilan assistant local](D:/ServOMorph/creation_llm/bilan_assistant_v2.md) |

Les graines de génération d’une même question ne doivent pas être comptées comme des questions indépendantes.

## 8. Verdict pour la cellule

**À TESTER pour le prochain pré-entraînement**, sous réserve de franchir d’abord la vérification technique. **À SURVEILLER pour la version améliorée de 2026**, dont la publication du mécanisme complet conditionne une réplication fidèle.

La priorité est de mesurer **la qualité obtenue par heure sur la RTX 4060**, avec le même texte et le même plafond mémoire. Ni la réduction des positions internes, ni les multiplicateurs annoncés en FLOPs ne permettent aujourd’hui de promettre un raccourcissement du run français.

## 9. Tableau des sources

| Titre / ressource | Date | URL | Statut |
|---|---|---|---|
| Dynamic Chunking for End-to-End Hierarchical Sequence Modeling | 10 juillet 2025 ; v2 15 juillet | [ArXiv](https://arxiv.org/abs/2507.07955) | Prépublication initiale ; travail ensuite publié |
| Même travail — actes et version finale | ICLR 2026 | [Actes](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f1aa53cac69ef6980ca4a911ffcf278b-Abstract-Conference.html), [PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/f1aa53cac69ef6980ca4a911ffcf278b-Paper-Conference.pdf) | Publication confirmée |
| H-Net — code et checkpoints | Consulté le 18 novembre 2026 | [GitHub](https://github.com/goombalab/hnet) | Ressource officielle ; non exécutée |
| Hierarchical modeling | 2025 ; jour non vérifié | [Cartesia](https://www.cartesia.ai/blog/hierarchical-modeling) | Présentation institutionnelle |
| From Tokens to Bytes I: Scaling Hierarchical Networks | 28 octobre 2026 | [Cartesia I](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1) | Rapport technique des auteurs |
| From Tokens to Bytes II: Learning from Raw Data | 28 octobre 2026 | [Cartesia II](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-2) | Rapport technique et ablations |
| Analysis of H-Net Router | 23 septembre 2025 | [Analyse](https://www.deklan.dev/hnet-router) | Analyse indépendante, non évaluée par les pairs |
| H-Net research reimplementation | Date initiale non vérifiée ; consulté le 18 novembre 2026 | [Dépôt](https://github.com/bigwolfeman/hnet-research) | Expériences indépendantes déclarées |
| Byte-Level H-Net Dynamic Chunking for Parameter Golf | Étude initiale référencée au 29 mars 2026 | [Dépôt](https://github.com/DariusFeher/parameter-golf-byte-hnet) | Expériences indépendantes à petite échelle |
| H-Net dynamic chunking — lucidrains | Date initiale non vérifiée | [Code](https://github.com/lucidrains/h-net-dynamic-chunking) | Réimplémentation ; gain non vérifié |
| H-Net++: Hierarchical Dynamic Chunking for Tokenizer-Free Language Modelling in Morphologically-Rich Languages | 7 août 2025 | [Article](https://arxiv.org/abs/2508.05628) | Prépublication indépendante |
| Byte Latent Transformer: Patches Scale Better Than Tokens | 13 décembre 2024 | [Article](https://arxiv.org/abs/2412.09871) | Source primaire du candidat voisin ; statut final non vérifié ici |

**Limite documentaire :** aucune reproduction exacte du facteur 10, aucun coût RTX 4060 et aucun gain sur l’assistant français n’ont été établis dans les sources consultées.