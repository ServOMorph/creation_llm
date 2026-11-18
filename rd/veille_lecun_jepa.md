# « GPA » chez Yann LeCun : JEPA, ses variantes et leur intérêt pour votre LLM français

*Veille arrêtée au 17 novembre 2026. Aucun fichier modifié, aucun calcul GPU lancé.*

## Résumé en 10 lignes

1. Le nom recherché est très probablement **JEPA**, pour *Joint Embedding Predictive Architecture*, phonétiquement proche de « jépa ».
2. JEPA désigne une famille d’objectifs d’apprentissage : prédire des représentations apprises plutôt que reconstruire directement les observations.
3. I-JEPA concerne les images ; V-JEPA et ses successeurs concernent la vidéo et, avec des actions, la planification.
4. **LeJEPA** introduit SIGReg, une régularisation qui vise à empêcher l’effondrement des représentations.
5. **LLM-JEPA** conserve l’apprentissage autorégressif et lui ajoute un objectif de représentation ; il reste un générateur de texte.
6. Son successeur **Semantic Tube Prediction — STP**, publié en 2026, mérite particulièrement l’attention sous contrainte de calcul.
7. Les gains annoncés portent sur des protocoles précis ; aucun ne prouve une amélioration de votre assistant français à budget égal.
8. Le récent résultat « 109M contre 72B » concerne l’alignement avec des représentations visuelles, pas la qualité générale d’un assistant.
9. Les publications, les reproductions indépendantes et les hypothèses de transfert doivent rester trois catégories distinctes.
10. Ma recommandation est de terminer le pré-entraînement actuel, puis d’envisager un petit essai SFT contrôlé de STP, éventuellement suivi de LLM-JEPA.

## 1. Identification : quelle méthode correspond à « gpa » ?

**JEPA est le meilleur candidat au nom entendu.** La date et le contenu de l’annonce permettraient d’identifier la variante exacte.

| Si l’annonce évoquait… | Candidat probable | Pourquoi |
|---|---|---|
| « Prédire dans un espace abstrait » | **JEPA** | Principe commun à la famille |
| Des zones cachées d’une image | **I-JEPA** | Première réalisation majeure en vision |
| Des vidéos, un robot, les conséquences des actions | **V-JEPA 2**, **V-JEPA 2.1** ou **LeWorldModel** | Représentations visuelles et dynamique |
| Une nouvelle méthode plus simple, sans enseignant EMA | **LeJEPA / SIGReg** | Simplification de l’apprentissage et traitement de l’effondrement |
| Améliorer l’entraînement d’un modèle de langage | **LLM-JEPA** ou **STP** | Applications directement linguistiques |
| « 109 millions battent 72 milliards », « 42 fois moins d’exemples » | **Proving and Improving the Platonic Representation Hypothesis with JEPAs** | Prépublication du 29 octobre 2026, affiliations AMI Labs |

Les références primaires correspondantes sont [I-JEPA](https://arxiv.org/abs/2301.08243), [V-JEPA 2](https://arxiv.org/abs/2506.09985), [LeJEPA](https://arxiv.org/abs/2511.08544), [LLM-JEPA](https://arxiv.org/abs/2509.14252), [STP](https://proceedings.mlr.press/v306/huang26s.html) et [le travail d’octobre 2026](https://arxiv.org/abs/2610.39302).

Je n’ai pas identifié, parmi les sources consultées, de méthode centrale de LeCun officiellement appelée « GPA ». L’identification de **la famille JEPA** est solide ; celle de **l’annonce particulière** reste incertaine.

### Convention de preuve

Pour éviter les ambiguïtés présentes entre les anciennes veilles du dépôt :

- **Publié** signifie qu’une publication ou une acceptation est vérifiée ; cela ne prouve pas une reproduction.
- **Établi et reproduit** exige une reproduction indépendante du résultat considéré.
- **Prépublication** désigne un résultat arXiv sans publication vérifiée ici.
- **Hypothèse locale** désigne toute transposition au français, au modèle maison ou à la RTX 4060.

**Aucun gain sur votre assistant n’est établi par cette veille.**

## 2. Principe, architecture et objectifs mathématiques

### 2.1 Apprendre ce qu’il faut prédire

Un objectif génératif demande de prédire une observation : pixels, tokens ou distribution sur ces éléments. JEPA apprend d’abord une représentation de chaque observation, puis prédit une représentation à partir d’une autre.

L’intuition est de pouvoir ignorer certains détails imprévisibles : par exemple, conserver le mouvement d’un objet sans devoir reconstruire chaque variation de texture. Le choix de ce qui est conservé dépend toutefois de l’objectif, des données et des vues utilisées. Le texte fondateur présente cette approche dans une architecture plus large comprenant modèle du monde, objectifs et planification. **JEPA seul ne constitue pas cet agent complet.** [LeCun, 2022](https://openreview.net/pdf/315d43ba26f55357a84cec9a7ed15a6610094f79.pdf)

### 2.2 Schéma avec encodeur cible EMA

Une écriture générique est :

\[
z_x=f_\theta(x),\qquad
z_y=f_{\bar\theta}(y),\qquad
\widehat z_y=g_\phi(z_x,m,a),
\]

\[
\mathcal L_{\mathrm{pred}}
=\mathbb E\!\left[d\!\left(\widehat z_y,
\operatorname{sg}(z_y)\right)\right].
\]

- \(x\) : contexte observé ;
- \(y\) : observation cible ;
- \(m\) : positions ou masque des éléments à prédire ;
- \(a\) : action éventuelle ;
- \(\operatorname{sg}\) : arrêt du gradient sur la cible.

Dans I-JEPA, l’encodeur de contexte voit les régions disponibles, le prédicteur produit les représentations des régions cibles et l’encodeur cible traite l’image complète. Ses poids suivent une moyenne exponentielle :

\[
\bar\theta\leftarrow \tau\bar\theta+(1-\tau)\theta.
\]

Le code officiel utilise notamment une normalisation des cibles et une perte `smooth_l1_loss`. **Il ne faut donc pas identifier toute implémentation JEPA à une simple MSE.** [Code officiel I-JEPA](https://github.com/facebookresearch/ijepa/blob/main/src/train.py)

Cette EMA est un mécanisme de construction de cibles pendant l’apprentissage. Elle diffère de l’EMA de fin de pré-entraînement envisagée dans votre cellule pour évaluer des poids moyennés.

### 2.3 Le problème de l’effondrement

Si les deux encodeurs renvoient toujours le même vecteur, la prédiction peut devenir parfaite sans contenir d’information utile. Il faut donc empêcher :

- l’effondrement complet : toutes les observations deviennent identiques ;
- l’effondrement dimensionnel : seules quelques directions restent utilisées.

**VICReg**, de Bardes, Ponce et LeCun, combine trois termes :

\[
\mathcal L
=\lambda_{\mathrm{inv}}\mathcal L_{\mathrm{inv}}
+\lambda_{\mathrm{var}}\mathcal L_{\mathrm{var}}
+\lambda_{\mathrm{cov}}\mathcal L_{\mathrm{cov}}.
\]

Ils rapprochent les vues correspondantes, maintiennent une dispersion suffisante par dimension et pénalisent les corrélations redondantes. Une écriture schématique des deux derniers est :

\[
\mathcal L_{\mathrm{var}}
=\frac1d\sum_j\max(0,\gamma-\operatorname{std}(Z_j)),
\qquad
\mathcal L_{\mathrm{cov}}
\propto\sum_{i\ne j}\operatorname{Cov}(Z)_{ij}^{2}.
\]

VICReg est publié à ICLR 2022. Ce n’est pas une composante obligatoire de toutes les JEPA. [Article VICReg](https://arxiv.org/abs/2105.04906)

### 2.4 LeJEPA et SIGReg

LeJEPA vise une distribution latente gaussienne isotrope. SIGReg compare des projections unidimensionnelles aléatoires des embeddings à une gaussienne standard, à l’aide d’un test fondé sur les fonctions caractéristiques.

Une représentation simplifiée est :

\[
\mathrm{SIGReg}(Z)
\approx \frac1K\sum_{k=1}^{K}
\int
\left|
\frac1B\sum_{b=1}^{B}e^{it u_k^\top z_b}
-e^{-t^2/2}
\right|^2w(t)\,dt.
\]

L’objectif combine rapprochement des vues et SIGReg :

\[
\mathcal L_{\mathrm{LeJEPA}}
=(1-\lambda)\mathcal L_{\mathrm{vues}}
+\lambda\,\mathrm{SIGReg}.
\]

L’implémentation présentée rapproche les vues de centres calculés à partir des vues globales ; elle ne nécessite ni cible EMA ni arrêt du gradient. Les auteurs rapportent notamment **79 % sur ImageNet-1k en évaluation linéaire avec ViT-H/14**. La justification d’optimalité dépend des hypothèses théoriques étudiées. [LeJEPA](https://arxiv.org/html/2511.08544v1)

**Nuance pratique :** « sans heuristiques » ne signifie pas « sans choix de recette ». Le dépôt conserve augmentations, nombre de vues, optimiseur, taux d’apprentissage et calendrier ; il documente même un warmup suivi d’un cosinus. La complexité linéaire annoncée pour la régularisation ne rend pas linéaire toute l’attention du Transformer. [Dépôt LeJEPA](https://github.com/galilai-group/lejepa)

## 3. Historique et résultats : ce que les chiffres mesurent réellement

| Étape | Résultat ou apport vérifié | Portée et coût |
|---|---|---|
| **JEPA, 2022** | Programme de recherche sur représentations, modèles du monde et planification | Texte de position, pas démonstration d’intelligence générale |
| **I-JEPA, 2023** | Prédiction de régions dans l’espace latent | ViT-H/14 sur ImageNet : **16 A100, moins de 72 h**, soit moins de 1 152 heures-GPU par multiplication |
| **V-JEPA, 2024** | Apprentissage vidéo sans reconstruction des pixels | **2 millions de vidéos** ; ViT-H/16 : **81,9 % Kinetics-400**, **72,2 % SSv2**, **77,9 % ImageNet1K**, encodeur gelé |
| **V-JEPA 2, 2025** | Passage à une plus grande échelle et post-entraînement conditionné par actions | Plus d’un million d’heures vidéo ; **77,3 % SSv2**, **39,7 Recall@5 EPIC-KITCHENS-100** |
| **LeJEPA, 2025** | Régularisation SIGReg et simplification du dispositif | Résultats de représentation visuelle ; aucune réduction démontrée des jours de pré-entraînement d’un GPT français |
| **VL-JEPA, 2025–2026** | Prédiction d’embeddings textuels à partir de vision/langage, décodage à la demande | **50 % de paramètres entraînables en moins** dans la comparaison contrôlée ; **2,85× moins d’opérations de décodage** |
| **V-JEPA 2.1, 2026** | Supervision dense et à plusieurs profondeurs | **77,7 SSv2**, **40,8 Recall@5 EPIC-KITCHENS**, **+20 points** de réussite en préhension contre V-JEPA 2-AC |
| **LeWorldModel, 2026** | Dynamique latente entraînée de bout en bout depuis les pixels | Environ **15M paramètres**, un GPU et quelques heures selon les auteurs ; planification jusqu’à **48× plus rapide** que les comparateurs considérés |
| **H-JEPA, 2026** | Hiérarchie de prédictions à plusieurs horizons | Visual AntMaze : **18 % → 73 %** de réussite, avec moins de calcul de planification |

Sources primaires : [texte fondateur](https://openreview.net/pdf/315d43ba26f55357a84cec9a7ed15a6610094f79.pdf), [I-JEPA, actes CVPR](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html), [V-JEPA](https://arxiv.org/abs/2404.08471), [V-JEPA 2](https://arxiv.org/html/2506.09985v1), [VL-JEPA](https://arxiv.org/abs/2512.10942), [V-JEPA 2.1](https://arxiv.org/abs/2603.14482), [LeWorldModel](https://arxiv.org/abs/2603.19312), [H-JEPA](https://arxiv.org/abs/2610.06805).

### Trois confusions sur l’efficacité à éviter

**Le facteur 1,5× à 6×** vient notamment de la communication de Meta sur **V-JEPA en 2024**, concernant l’efficacité d’apprentissage et d’échantillonnage. Ce n’est pas une mesure sur un modèle de langage. [Annonce Meta](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture?trk=public_post_comment-text)

**Le facteur 8,4× de V-JEPA 2** concerne l’entraînement progressif en résolution comparé à un entraînement intégral à haute résolution. Le comparateur à pleine résolution est projeté autour de 60 années-GPU. Ce chiffre ne mesure pas le seul bénéfice de l’objectif JEPA. Les quelque **62 heures de données robotiques** servent au post-entraînement et s’ajoutent au vaste pré-entraînement visuel. [V-JEPA 2, sections 2.4 et 3](https://arxiv.org/html/2506.09985v1)

**Un modèle entraînable sur “un GPU” ne signifie pas qu’il tient sous 6,5 Gio.** Pour LeWorldModel, le résumé ne fournit pas un pic VRAM utilisable pour votre configuration. Je ne transforme donc pas cette annonce en garantie de compatibilité RTX 4060.

## 4. Application au langage

### 4.1 LLM-JEPA : résultat réel, mais objectif hybride

Huang, LeCun et Balestriero ajoutent à la perte linguistique une distance cosinus entre représentations de vues associées :

\[
\mathcal L
=\mathcal L_{\mathrm{NTP}}
+\lambda\left[1-\cos\!\left(
P(E(x)),E(y)\right)\right].
\]

L’encodeur réutilise le LLM ; la représentation vient du dernier token. Des tokens `[PRED]` peuvent servir de prédicteur à poids partagés. Voici des moyennes sur cinq exécutions, dans la version arXiv v2 :

| Modèle / tâche | Témoin | LLM-JEPA | Écart en points |
|---|---:|---:|---:|
| Llama-3.2-1B / NL-RX-SYNTH | 57,29 | 71,46 | +14,17 |
| Gemma-2-2B / NL-RX-SYNTH | 33,65 | 43,12 | +9,47 |
| OpenELM-1,1B / NL-RX-SYNTH | 12,07 | 25,40 | +13,33 |
| OLMo-2-1B / NL-RX-SYNTH | 87,09 | 87,52 | +0,43 |
| Llama-3.2-1B / GSM8K | 32,36 | 36,36 | +4,00 |
| Llama-3.2-1B / Spider | 47,52 | 50,55 | +3,03 |
| Llama-3.2-1B / NQ-Open | 20,12 | 21,59 | +1,47 |

Le pré-entraînement exploratoire sur NL-RX-SYNTH donne **54,38 → 60,59 %**, avec un critère acceptant une sortie commençant par la cible. Ce n’est pas un pré-entraînement généraliste à plusieurs milliards de tokens. [LLM-JEPA v2, tableaux 2, 4, 12 et 13](https://arxiv.org/html/2509.14252v2)

**Coût et code.** La première implémentation nécessitait deux passes supplémentaires. Le dépôt actuel regroupe les deux vues avec un masque d’attention, ramenant le total à deux passes. Avec une perte JEPA omise sur 75 % des lots, le coût annoncé devient **1,25×** celui du témoin. L’inférence conserve le fonctionnement ordinaire. Les expériences du dépôt utilisent quatre époques et cinq graines ; le choix du dernier token dépend du modèle. [Code et protocole officiels](https://github.com/galilai-group/llm-jepa)

La publication ICLR 2026 est indiquée dans le document indexé par OpenReview et annoncée par le premier auteur. L’ouverture directe d’OpenReview a rencontré une vérification de navigateur ; les chiffres ci-dessus sont donc explicitement ceux de **l’arXiv v2**, pas une lecture intégrale de la version finale. [Document ICLR indexé](https://openreview.net/pdf?id=GbXKPo9QfH), [annonce de Hai Huang](https://www.linkedin.com/posts/hai-huang-ml_im-happy-to-share-that-our-paper-llm-jepa-activity-7421620757941694464-defM)

### 4.2 STP : la suite pertinente pour votre contrainte matérielle

**Semantic Tube Prediction** régularise les trajectoires d’états cachés. Pour trois positions \(s<r<t\) :

\[
\mathcal L_{\mathrm{STP}}
=1-\cos(h_t-h_r,\ h_r-h_s),
\qquad
\mathcal L=\mathcal L_{\mathrm{NTP}}+\lambda\mathcal L_{\mathrm{STP}}.
\]

Ces états existent déjà dans la passe linguistique : les auteurs annoncent un surcoût limité aux opérations auxiliaires, sans construction explicite de deux vues.

Le résultat **16× moins de données** concerne NL-RX-SYNTH. **Les époques sont multipliées par 16 lorsque le jeu est divisé par 16**, pour compenser le nombre de pas. Il s’agit donc d’efficacité en exemples distincts, pas d’une division par 16 du temps GPU. La géométrie des trajectoires repose sur une hypothèse de géodésiques localement linéaires. [Article STP](https://arxiv.org/html/2602.22617v1)

La publication **ICML 2026** est vérifiée dans les actes ; le code est dans `stp.py` du dépôt LLM-JEPA. [Actes ICML](https://proceedings.mlr.press/v306/huang26s.html), [code](https://github.com/galilai-group/llm-jepa)

### 4.3 Le résultat récent d’AMI Labs : « 109M contre 72B »

La prépublication du 29 octobre 2026 porte les affiliations AMI Labs. Elle entraîne un encodeur texte avec plusieurs descriptions d’une même scène comme vues positives.

Les auteurs rapportent **42× moins d’échantillons**, mais **14× moins de tokens**, pour atteindre un niveau comparable d’alignement. Le modèle de 109M à grande échelle a lu **116 milliards de tokens**, sur **8 ou 16 GPU**. La métrique principale mesure des voisinages communs avec un encodeur visuel.

Les limites déclarées comprennent l’origine synthétique de nombreuses descriptions, la sensibilité à une graine/régularisation et des hypothèses théoriques non exactement satisfaites par les expériences. **Ce résultat ne démontre pas qu’un assistant de 109M remplace un LLM de 72B.** [Article et limites](https://arxiv.org/html/2610.39302v1)

Le [code est public](https://github.com/galilai-group/prh-proof). Le [site d’AMI Labs](https://amilabs.xyz/) confirme l’orientation vers des modèles du monde prédisant dans l’espace des représentations ; il ne fournit pas une recette prête à remplacer votre GPT.

## 5. Limites, critiques et questions ouvertes

### 5.1 Une représentation mieux organisée peut ne pas améliorer la génération

Un audit indépendant de **Biswa Sengupta**, publié en prépublication le 14 mai 2026, étudie 22 auxiliaires sur Llama-3.2-1B. Aucun gain ne survit aux corrections de comparaisons multiples retenues.

Pour sa construction « decoder-visible JEPA », l’entraînement complet sur cinq graines donne :

- TURK : **+0,04 point**, \(p=0,96\) ;
- SYNTH : **+0,52 point**, \(p=0,28\).

Cela soutient la nécessité de mesurer les sorties décodées. **Ce n’est pas une réfutation exacte de toute la recette originale LLM-JEPA**, car l’audit examine plusieurs constructions particulières. Statut : **prépublication indépendante**. [Representation Without Reward](https://arxiv.org/abs/2605.15394)

À l’inverse, **ER-JEPA**, de Pu, Fan et Lian, reprend LLM-JEPA et ajoute une mémoire de répétition d’exemples, avec des améliorations annoncées sur plusieurs tâches. C’est une extension indépendante intéressante ; je n’en ai pas audité les tableaux assez complètement pour certifier une reproduction exacte des gains originaux. [ER-JEPA, septembre 2026](https://arxiv.org/abs/2609.36952)

### 5.2 Les garanties mathématiques sont conditionnelles

Le travail *When Does LeJEPA Learn a World Model?* établit une identifiabilité linéaire dans une classe de mondes à transitions stationnaires avec bruit additif, avec un rôle particulier des latents gaussiens. Sa version de novembre indique « NeurIPS 2026 camera-ready ».

Cette garantie ne signifie pas que tout langage, toute vidéo ou tout environnement satisfait ces conditions. [Article](https://arxiv.org/abs/2605.26379)

**Analyse critique :** une perte peut rendre certaines propriétés faciles à prédire tout en supprimant des détails indispensables à une tâche future. Pour votre assistant, les noms, dates, quantités et négations sont parfois précisément les détails qu’il faut préserver.

### 5.3 La position de LeCun sur les LLM

La position à retenir est un programme de recherche : développer des systèmes possédant mémoire persistante, modèle du monde et capacité de planification, au-delà de la seule production autorégressive. AMI reconnaît explicitement le succès des approches génératives pour le langage tout en contestant leur adéquation aux observations sensorielles fortement imprévisibles. [AMI Labs](https://amilabs.xyz/)

**Mon appréciation :** les résultats examinés justifient cette recherche, mais ne démontrent ni l’impossibilité générale du raisonnement autorégressif ni la supériorité universelle de JEPA. Le débat doit porter sur des tâches et des budgets comparables.

### 5.4 Ce qui reste ouvert pour votre cas

Je n’ai pas trouvé de preuve directement applicable sur :

- un GPT français de 336–369M entraîné depuis zéro avec environ 7,4 milliards de tokens ;
- Muon + AdamW, ReLU², QK-Norm et soft-cap dans la configuration du dépôt ;
- le score d’assistant et le défaut `critere_absent` ;
- un plafond réel de 6,5 Gio sous Windows ;
- une économie de temps incluant recherche d’hyperparamètres et préparation des vues ;
- une reproduction indépendante complète garantissant les gains de STP sur un assistant généraliste.

Les résultats visuels, linguistiques spécialisés et robotiques ne permettent pas de remplir ces cases par extrapolation.

## 6. Conséquences pour le dépôt et propositions falsifiables

### 6.1 Constats issus des fichiers demandés

J’ai consulté les fichiers de code, la roadmap, le bilan assistant et les documents R&D cités. `DOCUMENTATION/INDEX.md` est absent.

Trois éléments conditionnent la décision :

1. **« 369M » est le nom du run.** Avec les embeddings liés, le code correspond à **335 544 320 paramètres uniques**, soit environ 336M.
2. `model.py` expose les logits et la perte, mais pas directement les états cachés. Une expérimentation JEPA/STP demanderait de rendre ceux-ci accessibles.
3. `train_sft.py` utilise déjà les identifiants de documents pour isoler les exemples. Toute perte auxiliaire devra respecter ces frontières.

Le bilan du cycle 1 nuance aussi l’interprétation de `critere_absent` : les passages pertinents relus manuellement ont fait passer les faits de **49/90 à 71/90**. Une partie des connaissances peut donc être exploitée lorsque le contexte convient. Cela n’établit pas que la régularisation des représentations résoudra les échecs restants. [Bilan assistant](D:/ServOMorph/creation_llm/bilan_assistant_v2.md), [bilan du cycle 1](D:/ServOMorph/creation_llm/rd/bilan_cycle1.md)

### 6.2 Priorité proposée : STP en SFT, après le run actuel

**Statut : hypothèse locale**, motivée par une publication ; aucun gain local mesuré.

| Élément de preuve | Proposition |
|---|---|
| Mécanisme | Ajouter la perte STP aux états cachés de positions appartenant au même exemple, en conservant la perte SFT |
| Gain attendu chiffré | **Inconnu.** Aucun chiffre transférable défendable ; seuil d’intérêt proposé : **+10 réussites sur 300**, soit +3,33 points |
| VRAM | Pas de copie d’enseignant. Un tenseur \(2\times1024\times1024\) en bf16 représente **4 Mio** ; ce calcul n’est pas une estimation du pic complet |
| Temps | À mesurer. Pour 500 pas, deux bras et trois graines : **\(1500(t_A+t_B)\) secondes**, où \(t_A,t_B\) sont les temps mesurés par pas |
| Expérience minimale | Vérification de coût sur 200 pas ; puis témoin SFT contre SFT+STP, même checkpoint, mêmes données et budgets, trois graines |
| Critère d’échec | Dépassement de 6,5 Gio ; absence de gain apparié convaincant ; ou régression nette en faits, consignes ou FrenchBench |
| Source | [STP](https://arxiv.org/html/2602.22617v1) |

Le seuil de +10 est **un critère de décision proposé**, pas une prévision. Choisir les hyperparamètres sur un jeu de développement séparé ; conserver le benchmark final pour l’évaluation. Les trois générations d’une même question ne doivent pas être considérées comme trois questions indépendantes.

### 6.3 Deuxième piste : LLM-JEPA sur paires françaises contrôlées

**Statut : hypothèse locale.**

| Élément de preuve | Proposition |
|---|---|
| Mécanisme | Apprentissage conjoint de la génération et d’une relation entre deux vues : consigne/transformation, formulation/reformulation ou question/réponse |
| Gain attendu chiffré | **Inconnu** ; même seuil d’intérêt de +10/300 pour permettre la comparaison |
| VRAM | Poids partagés possibles ; pic des graphes et activations supplémentaires inconnu. Aucun argument ne permet d’affirmer que le pic reste identique |
| Temps | Avec omission de la perte sur 75 % des lots : modèle de coût annoncé **1,25×**, à vérifier localement ; ce n’est pas une mesure de temps mural |
| Expérience minimale | Témoin, paires correctes, puis contrôle avec paires permutées ; mêmes données et comparaison à temps égal |
| Critère d’échec | Gain absent à temps égal ; paires permutées aussi efficaces ; hausse des réponses vides/boucles ; dépassement VRAM |
| Source | [Implémentation officielle](https://github.com/galilai-group/llm-jepa) |

Le contrôle par permutation est utile : il vérifie si le bénéfice provient réellement de la relation entre les vues, plutôt que d’une régularisation ou d’un calcul supplémentaire quelconque.

### 6.4 LeJEPA/SIGReg pur : piste différée

**Mécanisme :** apprendre un encodeur de textes à partir de vues associées, puis mesurer son utilité pour la recherche de passages.

**Gain attendu :** inconnu ; cible expérimentale proposée **+5 points de Recall@5** sur un jeu indépendant suffisamment large.

**Coût :** pour une copie cible de 336M paramètres, les poids seuls coûteraient environ **0,625 Gio en bf16** ou **1,25 Gio en fp32**, hors activations. LeJEPA permet précisément d’explorer une formulation sans cette copie ; le coût d’entraînement total reste à mesurer.

**Expérience minimale :** petit encodeur séparé, comparé au système de recherche déjà évalué ; mesurer à la fois récupération du passage pertinent et réponses finales.

**Échec :** amélioration de la perte latente sans amélioration de récupération ou de réponse. Une statistique calculée sur deux embeddings de séquences par micro-lot serait particulièrement fragile : l’accumulation des gradients ne constitue pas automatiquement un grand lot pour SIGReg.

**Source :** [LeJEPA et SIGReg](https://github.com/galilai-group/lejepa).

**Décision proposée : terminer le pré-entraînement actuel.** Les preuves disponibles justifient des essais auxiliaires isolés ; elles ne justifient pas de remplacer maintenant son objectif ni de promettre une économie de plusieurs jours.

## 7. Tableau des sources et du code

*« Reproduction non vérifiée » ne signifie pas qu’aucune reproduction n’existe.*

| Titre / ressource | Date | URL | Statut retenu |
|---|---|---|---|
| A Path Towards Autonomous Machine Intelligence | 27 juin 2022 | [Texte](https://openreview.net/pdf/315d43ba26f55357a84cec9a7ed15a6610094f79.pdf) | Texte de position |
| VICReg | 11 mai 2021 ; ICLR 2022 | [Article](https://arxiv.org/abs/2105.04906) | Publié ; reproduction non vérifiée ici |
| I-JEPA | 19 janvier 2023 ; CVPR juin 2023 | [Actes](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html), [code](https://github.com/facebookresearch/ijepa) | Publié, code officiel |
| Revisiting Feature Prediction — V-JEPA | 2024 ; annonce le 15 février | [Article](https://arxiv.org/abs/2404.08471), [code](https://github.com/facebookresearch/jepa) | Article public ; publication finale non vérifiée |
| V-JEPA 2 | Juin 2025 | [Article](https://arxiv.org/abs/2506.09985), [code](https://github.com/facebookresearch/vjepa2) | Rapport public et code ; reproduction indépendante non vérifiée |
| LLM-JEPA | 11 septembre 2025 ; v2 le 7 octobre ; ICLR 2026 | [ArXiv](https://arxiv.org/abs/2509.14252), [ICLR](https://openreview.net/pdf?id=GbXKPo9QfH), [code](https://github.com/galilai-group/llm-jepa) | Publication indiquée ; chiffres audités dans arXiv v2 |
| LeJEPA | 11 novembre 2025 | [Article](https://arxiv.org/abs/2511.08544), [code](https://github.com/galilai-group/lejepa) | Prépublication ; code public |
| VL-JEPA | 11 décembre 2025 ; v2 le 2 février 2026 | [Article](https://arxiv.org/abs/2512.10942) | Version arXiv consultée ; statut final non recoupé |
| Semantic Tube Prediction | 26 février 2026 ; ICML juillet 2026 | [Article](https://arxiv.org/abs/2602.22617), [actes](https://proceedings.mlr.press/v306/huang26s.html) | Publié ; reproduction indépendante exacte non vérifiée |
| LeWorldModel | 13 mars 2026 ; v3 le 3 juin | [Article](https://arxiv.org/abs/2603.19312) | Prépublication consultée |
| V-JEPA 2.1 | 15 mars 2026 ; v3 le 11 juin | [Article](https://arxiv.org/abs/2603.14482), [code commun](https://github.com/facebookresearch/vjepa2) | Prépublication consultée, code officiel |
| Representation Without Reward | 14 mai 2026 | [Article](https://arxiv.org/abs/2605.15394) | Audit indépendant, prépublication |
| When Does LeJEPA Learn a World Model? | 25 mai 2026 ; v2 le 9 novembre | [Article](https://arxiv.org/abs/2605.26379) | Mention NeurIPS 2026 camera-ready ; garanties conditionnelles |
| ER-JEPA | 29 septembre 2026 | [Article](https://arxiv.org/abs/2609.36952) | Extension indépendante, prépublication |
| H-JEPA | 5 octobre 2026 ; v2 le 1er novembre | [Article](https://arxiv.org/abs/2610.06805) | Prépublication |
| Proving and Improving the Platonic Representation Hypothesis with JEPAs | 29 octobre 2026 | [Article](https://arxiv.org/abs/2610.39302), [code](https://github.com/galilai-group/prh-proof) | Prépublication, affiliations AMI Labs |
| AMI Labs — programme scientifique | Page consultée le 17 novembre 2026 | [Site officiel](https://amilabs.xyz/) | Source institutionnelle ; pas une preuve expérimentale |

**Limite documentaire :** les pics VRAM, durées GPU et volumes exacts de chaque configuration ne sont pas tous disponibles dans les passages vérifiés. Ils restent explicitement inconnus ; aucun coût RTX 4060 ni gain sur votre assistant n’a été mesuré.