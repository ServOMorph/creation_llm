**Je conserverais le pré-entraînement actuel et réorienterais le cycle 2 vers trois leviers : mieux exploiter les données déjà disponibles, réduire le coût de la tête de sortie, puis apprendre au SFT à sélectionner et restituer l’information pertinente.** Les nouvelles architectures et initialisations sont des expériences pour le prochain modèle.

Le rapport est globalement prudent et utile. Ses principales références existent, et plusieurs chiffres importants sont confirmés. Son principal défaut est de placer au même niveau des pistes immédiatement testables et des programmes exigeant plusieurs entraînements complets.

Aucun fichier modifié, aucun calcul GPU lancé. Les fichiers demandés ont été consultés, ainsi que `prepare_corpus.py`, `prepare_tokens.py`, `gpu_task.py` et le scoring de `bench_assistant.py`. `DOCUMENTATION/INDEX.md` est absent.

**1. Solidité du rapport et corrections**

J’utilise quatre statuts :

- **Établi** : résultat publié et reproduit indépendamment dans un cadre identifié.
- **Publié, reproduction indépendante non vérifiée** : publication scientifique confirmée ; le gain précis reste celui des auteurs.
- **Prépublication** : manuscrit ou rapport technique ; une acceptation annoncée ne constitue pas une reproduction.
- **Hypothèse locale** : adaptation, calcul de coût ou objectif expérimental proposé ici.

**Aucun gain des nouveautés n’est établi sur votre configuration.** Les mesures du cycle 1 sont des résultats locaux documentés, distincts d’une reproduction indépendante.

| Point du rapport | Vérification et conséquence |
|---|---|
| Taille réelle du modèle | **Correct.** Avec RMSNorm sans paramètres et embeddings liés : \(24×12×1024²+32768×1024=335\,544\,320\) paramètres. |
| Durée du run | **Correct.** 113 000 × 65 536 = **7 405 568 000 tokens** ; à 6 300 tok/s : **326,5 h, soit 13,60 jours** hors interruptions. |
| Budget mémoire | `cap_vram()` interprète bien 6,1 en **Gio**. Ne pas confondre cette limite avec la réservation déclarée de 6,5 ni avec les 8 Go physiques. Le pic historique de 6,04 laisse seulement **environ 61 Mio** sous 6,1, si ces valeurs sont comparées dans les mêmes conditions. |
| Curriculum | Le résultat EACL 2026 est confirmé : **18–45 % de pas en moins vers certains niveaux de performance**, surtout tôt et à mi-parcours. Le rejet général du cycle 1 doit être remplacé par un essai conditionnel. Ce n’est pas une promesse de −45 % sur le run complet. [Article](https://aclanthology.org/2026.eacl-long.271/) |
| Répétition | La distinction est fondée : la réutilisation d’un domaine dans un mélange et la duplication interne ne sont pas équivalentes. Les résultats « 15–20 répétitions » et « 67 % des FLOPs » correspondent à des protocoles différents ; ils ne permettent pas de recalculer directement vos poids de sources. [Mélanges](https://arxiv.org/html/2605.12715v2), [doublons](https://arxiv.org/abs/2606.24998) |
| Rho-1 | La sélection porte sur une **perte excédentaire relativement à une référence**, pas sur les seuls tokens difficiles. Le Transformer traite toujours le contexte. Une référence v9 partageant le tokenizer facilite l’implémentation, mais sa compétence par domaine reste une condition expérimentale. [Article](https://arxiv.org/html/2404.07965v4) |
| APOLLO | Les gains système sont principalement comparés à **AdamW**, et dépendent du lot rendu possible. Rien ne justifie ×3 contre votre Muon. [Article](https://arxiv.org/abs/2412.05270) |
| SOAP / FOAM | L’avertissement mémoire est juste. FOAM conserve davantage d’états que Muon selon son propre tableau. « Moitié de SOAP » ne signifie pas « léger pour votre GPU ». [SOAP, publication](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e988664070e9591f93fdcf605f7dc623-Abstract-Conference.html), [FOAM, prépublication](https://arxiv.org/html/2611.16905v1) |
| Cut Cross-Entropy | **Réserve à actualiser :** l’API officielle expose déjà `softcap`, `ignore_index`, `reduction` et `filter_eps`. Le soft-cap n’est donc pas une fonctionnalité absente. La compatibilité de bout en bout reste à vérifier. Le filtrage de gradients impose aussi de distinguer mode non filtré et mode approximatif. [API officielle](https://raw.githubusercontent.com/apple-aiml-research/ml-cross-entropy/main/cut_cross_entropy/linear_cross_entropy.py) |
| Soft-MTP | Le tableau est correctement lu : à 332M, NTP obtient **13,073** de perplexité contre **13,551** pour Soft-MTP. La moyenne des tâches est également légèrement meilleure pour NTP : **50,79 contre 50,46**. Aucune raison de le prioriser ici. [Tableau 3](https://arxiv.org/html/2610.38331v1) |
| BPT | Les **206M entraînables / 494M**, losses **3,014 / 3,022**, sont confirmés. L’implémentation avec Muon serait toutefois une nouvelle expérience : il faut traiter séparément matrices d’adaptateurs, diagonales et remises à zéro des états. [Article](https://arxiv.org/html/2610.37934v1) |
| Olmo Hybrid | Le déséquilibre de paramètres est confirmé : colonne nominale 370M, **371M contre 523M hors embeddings**. Cela n’invalide pas toute l’étude, mais interdit d’en faire une comparaison locale à taille égale. [Tableau 22](https://arxiv.org/html/2604.03444v4) |
| Initialisations structurées | Le résultat ACL des langages formels et les chiffres du résumé NCA sont confirmés. Le premier est publié ; NCA reste ici classé prépublication. Ni l’un ni l’autre ne démontre un remède direct aux omissions factuelles françaises. [Langages formels](https://aclanthology.org/2025.acl-long.478/), [NCA](https://arxiv.org/abs/2603.10055) |
| TST « sortie seule » | **Extrapolation importante :** supprimer la compression des entrées retire le mécanisme principal d’accélération. Cette variante devient un objectif auxiliaire à tester, pas une méthode bénéficiant déjà du ×2,5 annoncé. [TST](https://arxiv.org/abs/2605.06546) |

Les chiffres secondaires des six familles de modèles — SmolLM2/3, MiniCPM4, Phi-4-mini, TinyLlama, MNN-Pocket — n’ont pas tous été revérifiés dans leurs tableaux. Je ne les utilise donc pas pour chiffrer un gain local. Leur existence et leurs sources principales sont confirmées ; leur qualité finale n’isole pas causalement un ingrédient de recette.

Deux limites documentaires persistent : le PDF BabyLM cité n’a pas été récupéré ; la page explicative de TST a échoué à l’ouverture. Les affirmations correspondantes ne sont retenues qu’au niveau vérifiable dans leurs résumés primaires.

**Deux corrections méthodologiques supplémentaires sont nécessaires.**

Premièrement, `critere_absent` signifie dans `bench_assistant.score()` qu’aucune expression régulière attendue n’a été trouvée. Cela peut correspondre à une connaissance absente, une omission, une mauvaise sélection du passage ou une réponse correcte formulée autrement. **Ce n’est pas un diagnostic causal.** Le passage oracle du cycle 1 — 71/90 contre 49/90 — montre un potentiel de lecture, pas un plafond garanti pour RAFT.

Deuxièmement, les **2,3 % de perte des premières positions** dans `rd/pole_mathematiques.md` ne constituent pas une borne générale sur le gain d’un entraînement aligné : changer le découpage modifie toute la distribution et la trajectoire d’apprentissage. Et deux checkpoints successifs ne suffisent pas à estimer un seuil de significativité de 0,005 nat.

**2. Points d’intégration réellement présents dans le dépôt**

| Zone | Fonctions et implications |
|---|---|
| Modèle | `GPT.forward()` calcule systématiquement les logits complets ; `CausalSelfAttention.forward()` gère le masque ; `Block.forward()` porte le recalcul. Ce sont les points d’entrée pour CCE, MIR, MTP, récurrence et hybrides. |
| Pré-entraînement | `MixedSource.batch()` tire des fenêtres avec remise. `get_batch()`, `train()`, `evaluate()`, `build_optimizer()`, `get_lr()` et `save_checkpoint()` concentrent les changements proposés. |
| Lanceur | `run_369m.spec()`, `MIX` et `ARGS` règlent corpus et budget. Les variantes expérimentales devraient avoir des configurations séparées ; le lanceur de production ne doit pas devenir un catalogue d’essais. |
| Préparation générale | **`prepare_data.main()` ne prépare que LeCarnet.** Pour les quatre sources, les fonctions pertinentes sont surtout `prepare_corpus.parquet_texts()`, `write_batch()` et `encode_sources()`. Modifier seulement `prepare_data.py` manquerait le pipeline réellement utilisé. |
| Préparation SFT | `prepare()`, `build_example()`, `write_split()` et `load_split()` portent corpus, masques et éventuels poids de tokens. |
| SFT | `PackedData.batch()` passe déjà des identifiants de conversations ; `masked_loss_sum()` applique la loss aux réponses ; `train()` normalise sur le nombre total de cibles du pas accumulé. **Cette normalisation doit être conservée avec CCE.** |
| GPU | `gpu_task.main()` contrôle jeton et absence du 369M. Une réservation `--minutes` ne remplace pas une limite explicite de pas dans l’expérience. |

Le masque SFT isole les **conversations**, pas les passages d’un même exemple RAFT : la réponse doit pouvoir consulter tous ces passages.

Autres pièges à traiter avant les expériences :

- `prepare_sft.prepare()` sépare actuellement les exemples après mélange. Des variantes d’un même fait peuvent donc traverser train/validation : séparer **par document, fait ou famille de gabarits avant augmentation**.
- `--drop-refusals` peut supprimer certaines réponses légitimes indiquant que l’information manque.
- `train_sft.train()` charge le checkpoint initial sur GPU, y compris les objets présents dans ce checkpoint, avant de supprimer `src`. Un chargement initial sur CPU mérite d’être vérifié pour éviter un pic transitoire inutile.
- `train.get_lr()` utilise le numéro absolu du pas. Prolonger simplement `max_steps` peut modifier le taux de reprise : l’expérience de refroidissement exige un **schedule local de continuation** explicite.

**3. Transposition de chaque méthode : coûts et intérêt**

Les coûts ci-dessous sont des **calculs analytiques ou hypothèses**, jamais des mesures nouvelles.

Référence : 335,54M paramètres, micro-lot 2 × 1024, accumulation 32, débit indicatif \(q_0=6300\) tok/s. « ≈ \(q_0\) » signifie calcul neuronal inchangé ; les lectures disque et traitements CPU peuvent dégrader ce débit. Pour le SFT, le débit de référence doit être remesuré : masque, remplissage et fraction de réponses diffèrent.

**Données et supervision**

| Méthode, mécanisme, source | Fichiers / fonctions | VRAM et débit estimables | Run actuel ; SFT et omissions |
|---|---|---|---|
| **Curriculum initial** : difficulté croissante puis mélange ordinaire. Publié, reproduction non vérifiée. [Source](https://aclanthology.org/2026.eacl-long.271/) | `prepare_corpus.write_batch()` pour les index ; `MixedSource.batch()` ; configuration de `run_369m` pour un futur run. | **0 état GPU supplémentaire** ; ≈ \(q_0\) si index préchargés. Temps CPU à mesurer. | Pas d’amorçage rétroactif sur le run actuel. SFT possible via ordre des exemples, mais « facile » doit être défini par tâche. Lien avec omissions indirect. |
| **Déduplication et répétition contrôlée** : supprimer les doublons accidentels, conserver les réutilisations utiles. Prépublications. [Mélanges](https://arxiv.org/html/2605.12715v2), [doublons](https://arxiv.org/abs/2606.24998) | `prepare_corpus.encode_sources()` ; `MixedSource` ; `prepare_sft.prepare()`. | **0 VRAM** ; ≈ \(q_0\). Index CPU/disque proportionnels aux documents ou fenêtres, pas au vocabulaire. | Audit CPU immédiat ; nouveau corpus ou échantillonnage sur branche séparée. SFT directement pertinent pour diversité et fuite entre partitions. |
| **Rho-1** : privilégier la perte excédentaire. Publié selon le rapport ; gain local hypothétique. [Source](https://arxiv.org/html/2404.07965v4) | `GPT.forward()` pour loss par token ; `MixedSource.batch()` pour scores ; `train.train()` ; éventuellement `masked_loss_sum()`. | Scores d’un micro-lot : quelques Kio ; **14,8 Go sur disque** à 2 octets × 7,4B. Contexte dense inchangé ; scoring additionnel \(N/q_{\rm ref}\). | Compatible avec une continuation expérimentale, mais change l’objectif. Au SFT, risque de privilégier rareté/bruit plutôt que faits omis ; référence à valider. |
| **Distillation textuelle** : réponses vérifiées d’un professeur. Publication BabyLM ; adaptation locale hypothétique. [Source](https://aclanthology.org/2025.babylm-main.33/) | `prepare_sft.source_files()` accepte déjà `--extra` ; `prepare()` ; entraînement SFT existant. | **0 surcoût architectural étudiant**. Génération séparée ; poids BF16 du professeur 1,7B ≈ **3,4 Go**, pic total inconnu. | Préparation possible maintenant ; génération GPU hors run concurrent. Très pertinente si ciblée sur sélection, complétude et transformations. |
| **MIR** : loss supplémentaire sur entrée masquée. Prépublication. [Source](https://arxiv.org/html/2606.06888v2) | `train.train()`, éventuellement `GPT.forward()` pour le masquage ; adaptation de `masked_loss_sum()`. | Deux traitements complets : hypothèse simple **≈ 2× le temps**, soit ≈ 3 150 tokens originaux/s. Pic proche du témoin seulement si les deux graphes sont traités successivement. | Pas recommandé pour le run. En SFT, ne pas masquer la preuve puis exiger la même réponse factuelle : cela peut enseigner à inventer. |
| **Recettes SmolLM/Phi/MiniCPM** : mélange progressif et données ciblées | `MixedSource.batch()`, `get_lr()`, configurations des lanceurs ; `prepare_sft.prepare()`. | **0 état GPU ajouté** à architecture constante ; ≈ \(q_0\), plus préparation et génération. | Une phase finale enrichie est testable après la base, sans copier leurs proportions. Gain sur omissions inconnu. [SmolLM2](https://arxiv.org/abs/2502.02737), [SmolLM3](https://huggingface.co/blog/smollm3), [MiniCPM4](https://arxiv.org/html/2506.07900v1), [Phi](https://arxiv.org/abs/2503.01743) |

**Mémoire et optimisation**

Le stockage persistant théorique FP32 vaut ici environ **3,875 Gio** : poids, gradients, momentum Muon des blocs et deux moments AdamW des embeddings. Ce calcul exclut activations, copies temporaires, espaces de travail et allocations du compilateur ; il n’est donc pas un pic mémoire.

| Méthode et source | Intégration | Coût / économie pour cette architecture | Compatibilité et SFT |
|---|---|---|---|
| **CCE**, loss sans matrice globale de logits. Publié, reproduction non vérifiée. [Code](https://github.com/apple-aiml-research/ml-cross-entropy) | `GPT.forward()` : chemin loss sans logits ; `masked_loss_sum()` : réduction somme. | Matrice actuelle : **128 Mio BF16**, **256 Mio FP32**. Le vocabulaire chat 32 832 augmente légèrement ces valeurs. Économie nette inconnue ; objectif d’essai **≥128 Mio**. Débit inconnu. | Compatible avec les poids existants si comportement numérique validé. Très utile au SFT ; **aucun gain sémantique direct**. |
| **APOLLO-Mini**, états projetés. Publié MLSys. [Source](https://arxiv.org/abs/2412.05270) | `build_optimizer()`, configuration et sauvegarde des nouveaux états. | Momentum Muon remplaçable : **1,125 Gio**. C’est une borne sur l’économie de cet état, avant états APOLLO et temporaires. Débit non prédictible. | Poids réutilisables, états Muon non réutilisables tels quels. Possible au SFT, sans raison spécifique d’améliorer les omissions. |
| **SOAP / FOAM**, préconditionnement structuré. [SOAP](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e988664070e9591f93fdcf605f7dc623-Abstract-Conference.html), [FOAM](https://arxiv.org/html/2611.16905v1) | `build_optimizer()` ; découpage des facteurs à concevoir. | FOAM non découpé : les facteurs des seules projections MLP descendantes ajoutent **3 Gio** ; les autres matrices aggravent le bilan. Débit local inconnu, comparaison directe hors plafond. | Non retenus sur 369M. SFT également contraint ; aucune preuve ciblée. |
| **BPT**, base gelée, diagonales et adaptateurs fusionnés. Prépublication avec acceptation annoncée. [Source](https://arxiv.org/html/2610.37934v1) | `CausalSelfAttention.__init__()`, `MLP.__init__()`, `build_optimizer()`, `train()` pour fusions. | Illustration locale, **pas reproduction BPT** : rang 64 sur les quatre matrices par bloc → **25,17M paramètres bas rang**. Avec AdamW FP32 et blocs de base gelés, économie persistante d’environ **1,88 Gio**, avant diagonales/temporaires. Débit inconnu. | Conversion possible sur branche, pas reprise identique. SFT envisageable ; vérifier les projections initialement nulles et les états après fusion. |
| **COAT**, stockage FP8 d’activations/états. Publié ICLR. [Code](https://github.com/NVlabs/COAT) | `Block.forward()`, attention/MLP et `build_optimizer()`. | Gain total inconnu. À titre de borne, un momentum Muon FP32→8 bits économiserait **0,844 Gio**, avant métadonnées — mais ce serait une adaptation nouvelle, pas COAT livré. | Kernels Windows/4060/Muon non vérifiés. Aucun transfert en cours ; intérêt SFT purement matériel. |

**Objectifs, architecture et initialisation**

| Méthode et source | Fonctions concernées | Estimation locale | Position |
|---|---|---|---|
| **MTP progressif / Soft-MTP** : plusieurs futurs. [Curriculum publié](https://aclanthology.org/2025.acl-long.1243/), [Soft-MTP prépublication](https://arxiv.org/html/2610.38331v1) | `GPT.__init__()`, `forward()`, `train.train()` ; masques SFT à étendre. | Une tête indépendante de 32 768×1024 coûte **512 Mio** poids+gradient+AdamW FP32, hors logits ; partager la projection réduit ce coût, pas celui des blocs éventuels. Débit inférieur ou inconnu selon variante. | Futur témoin seulement. Le soft-cap casse l’application directe de la dérivation linéaire. Pas de gain démontré à votre taille. |
| **CGLS** : croissance de profondeur liée au curriculum. [Source](https://arxiv.org/html/2506.11389v5) | `GPT.blocks`, `train()` pour croissance/recompilation ; `build_optimizer()`, `MixedSource`. | À 12 couches au lieu de 24, **1,688 Gio** de stockage poids/gradients/momentum de blocs en moins ; mémoire finale inchangée. Moins de FLOPs au début, débit exact inconnu. | Nouveau run. SFT sur architecture finale possible, sans effet causal isolé sur omissions. |
| **Schedule de continuation** : refroidissement adapté à l’horizon. [Source](https://arxiv.org/abs/2501.18965) | `train.get_lr()`, `train()`, `save_checkpoint()` ; équivalent SFT `get_lr()`. | **0 VRAM supplémentaire**, ≈ \(q_0\). EMA CPU : **1,25 Gio de RAM** pour les poids FP32, transferts à mesurer. | Prioritaire à la fin du run, conformément au cycle 1. Gain qualité inconnu ; hypothèse de −0,18 nat non validée. |
| **RecursiveGPT** : couches partagées et embeddings factorisés. Prépublication. [Source](https://arxiv.org/html/2608.26973v1) | `GPT.__init__()`, `forward()`, `build_optimizer()`. | Variante illustrative : 12 blocs distincts exécutés deux fois économisent **1,688 Gio persistants**, mais conservent 24 exécutions. Factorisation liée de rang 256 : environ **0,371 Gio** supplémentaires économisés. Pic et débit inconnus. | Nouveau modèle ; pas une conversion neutre du checkpoint. SFT ultérieur possible. Pas de preuve d’efficacité en temps. |
| **Gated DeltaNet / Olmo Hybrid** : attention et récurrence alternées. Rapport technique. [Source](https://arxiv.org/html/2604.03444v4) | Nouveau bloc dans `model.py`, `GPT.forward()`, gestion des états et `build_optimizer()`. | **VRAM et débit non estimables honnêtement sans configuration précise et kernels testés.** À contexte 1024, l’avantage asymptotique ne suffit pas. | Nouveau pré-entraînement ; basse priorité cycle 2. |
| **Langages formels** : préparer les dépendances hiérarchiques. Publié ACL. [Source](https://aclanthology.org/2025.acl-long.478/) | Générateur CPU relié à la préparation ; `train.train()` pour transfert ; `GPT` si transfert partiel. | **0 état permanent ajouté**. Chaque 100M tokens coûterait **4,41 h** à \(q_0\), conversion indicative seulement. | Aucun amorçage du run déjà commencé. Effet SFT indirect, surtout structure/syntaxe. |
| **NCA** : initialisation spatiotemporelle synthétique. Prépublication. [Source](https://arxiv.org/abs/2603.10055) | Générateur distinct, `train.train()` et transfert contrôlé des poids. | **0 état permanent ajouté** après transfert ; 164M tokens ≈ **7,23 h** à \(q_0\), plus génération. Pic générateur et débit réels inconnus. | Futur run ; après essai des langages formels. |
| **TST** : superposition des entrées puis récupération NTP. Prépublication. [Source](https://arxiv.org/abs/2605.06546) | `GPT.forward()`, `MixedSource.batch()`, loss et phases de `train()`. | À \(k\) tokens regroupés et longueur brute fixe, positions internes divisées par \(k\) ; stockage des paramètres inchangé. Pas de facteur de débit local fiable. Sortie seule : **aucune réduction du tronc**. | Branche dédiée, jamais activation opportuniste sur le run. SFT à réponses exactes particulièrement sensible à la perte d’ordre. |

TinyLlama reste une référence d’ingénierie ; MNN-Pocket une référence de déploiement. Leurs résultats ne justifient ni une quantification d’entraînement immédiate ni un changement de tokenizer en cours. [TinyLlama](https://github.com/jzhang38/TinyLlama), [MNN-Pocket](https://arxiv.org/abs/2611.12311)

**4. Expériences retenues et critères de décision**

Les gains chiffrés ci-dessous sont des **objectifs de sélection**, pas des prédictions. Un gain attendu statistiquement crédible n’est pas disponible sans mesure locale.

Protocole commun :

- Validation fixe non filtrée, ventilée par source ; test assistant indépendant des données générées.
- Comparaisons à temps total égal, tokens traités égaux et données uniques égales, selon la question.
- Temps total comprenant préparation, scoring, génération, compilation et évaluations.
- Tests appariés ; incertitude calculée par documents ou questions, pas en considérant trois générations de la même question comme trois questions indépendantes.
- Un court essai mesure faisabilité, stabilité et débit. **Il ne démontre pas l’efficacité d’apprentissage.**

Le témoin 12×512 lié contient **54 525 952 paramètres**. À 100M tokens, il n’a vu que **1,83 token par paramètre** ; à 300M, **5,50**. Le 369M vise **22,07**. Les essais courts doivent donc présélectionner ; une confirmation d’horizon plus long est nécessaire avant de lancer un nouveau modèle de deux semaines.

| Piste retenue | Expérience minimale | Gain visé | Échec / abandon |
|---|---|---|---|
| **Audit couverture et doublons** | CPU, échantillon stratifié des quatre sources ; mesurer doublons exacts, quasi-doublons, recouvrement train/validation et exposition. Pilote plafonné à 1 h CPU avant extrapolation. | Aucun gain qualité annoncé à ce stade ; déterminer le calcul gaspillé et la diversité réellement vue. | Audit incapable de séparer doublons, répétitions volontaires et collisions ; ou préparation dont le coût ne s’amortit pas. |
| **CCE** | CPU : interfaces, décalages et référence numérique réduite. GPU : compilation puis 100 micro-lots avant/arrière, et plusieurs vrais pas accumulés avec optimiseur. Comparer loss, gradients liés et pic. | **≥128 Mio économisés**, débit non inférieur. Ensuite vérifier si le lot 4 devient utilisable. | Incompatibilité locale, gradients incorrects, dépassement du plafond, économie sans intérêt opérationnel ou ralentissement. |
| **Curriculum initial par source** | CPU : scorer un échantillon, contrôler corrélation avec longueur et doublons. GPU court : chargement et trajectoire. Puis aléatoire / curriculum initial / permanent sur témoin. | **≥10 % de temps total économisé** vers une qualité commune. | Avantage disparu après retour aléatoire ; régression d’une source ; gain dû uniquement à la sélection de textes. |
| **SFT ciblé avec distracteurs** | CPU : 200 exemples vérifiés, puis pilote 2 000. GPU : essai 30–45 min pour stabilité, avant entraînement complet comparatif. | **+5 points** sur test indépendant ciblé, sans régression générale confirmée. | Réponses fausses, recopie accrue, abstention excessive, amélioration seulement lexicale du score. |
| **Refroidissement + EMA** | CPU : vérifier le schedule de reprise. À la fin du run, deux continuations de 2 000 pas : lr constant contre refroidissement ; EMA dans les deux bras. | **≥0,005 nat** favorable, avec intervalle apparié excluant zéro ; qualité française préservée. | Gain non confirmé ou seulement effet des tokens supplémentaires. **262,1M tokens au total ≈11,56 h** à 6 300 tok/s, hors frais annexes. |
| **ResFormer reporté du cycle 1** | CPU : gradients/recalcul/cache. GPU court : surcoût ; puis témoin standard contre résidu de valeurs. | **≥0,01 nat** à temps égal, confirmé entre graines. | Gain inférieur au bruit ou annulé par le coût. Un tenseur de valeurs BF16 de lot 2 vaut **4 Mio** ; ce n’est pas tout le surcoût autograd. [Source](https://arxiv.org/abs/2410.17897) |
| **Langages formels** | CPU : générateur et contrôle aléatoire à fréquences comparables. GPU court : apprentissage et transfert fonctionnels. Puis français seul / formel→français / aléatoire→français. | **≥10 % de temps total économisé**, initialisation comprise. | Gain uniquement syntaxique avec régression factuelle, ou coût d’amorçage non amorti. |
| **APOLLO, conditionnel** | CPU : compter les états exacts. GPU court : Muon contre APOLLO-Mini, même micro-lot puis meilleur lot permis. | **≥0,5 Gio économisé**, et pas de dégradation confirmée >0,01 nat ; avantage opérationnel réel. | Mémoire libérée inutilisable, projections trop lentes ou convergence inférieure à temps égal. |

Pour CCE, commencer avec le filtrage de gradients désactivé, `shift=0` puisque les cibles sont déjà décalées, `softcap=30`, et `reduction="sum"` au SFT. Vérifier notamment que le poids partagé reçoit les contributions de l’entrée **et** de la sortie. Le filtrage pourra devenir une expérience séparée. [Implémentation officielle](https://raw.githubusercontent.com/apple-aiml-research/ml-cross-entropy/main/cut_cross_entropy/cce.py)

Un point matériel rend cet essai particulièrement intéressant : le bilan historique mesure **6 372 tok/s au lot 2**, contre **6 994 au lot 4**, soit **+9,8 %** dans cet essai. Mais le lot 4 atteignait 6,33 Gio : il faudrait économiser au moins **236 Mio**, plus une marge, pour passer sous 6,1. **128 Mio économisés ne suffisent donc pas à promettre le lot 4.**

Les essais GPU courts proposés passent exclusivement par `gpu_task.py`, avec contrôle du détenteur du jeton et limite de pas dans le script. Ils restent séquentiels avec le 369M.

**5. Trois combinaisons originales proposées**

Ce sont des **hypothèses de la cellule**. Je n’ai pas trouvé leur formulation exacte dans les sources consultées et les recherches ciblées. Cela ne permet pas d’affirmer leur inexistence dans toute la littérature ; leur originalité porte sur le protocole combiné, pas sur chaque ingrédient.

**H1 — Curriculum par source avec couverture contrôlée**

**Mécanisme.** Combiner un curriculum initial par compression avec des permutations de fenêtres au sein de chaque source. Conserver les quotas 0,58/0,25/0,15/0,02 en tokens ; éviter les revisites accidentelles d’une fenêtre avant épuisement du réservoir pertinent. Les petites sources continuent de se répéter volontairement.

Cette combinaison traite un problème local distinct du « manque de données » : avec tirage avec remise, environ une passe équivalente ne couvre qu’une partie des tokens. Le cycle 1 estime environ **37–38 % jamais vus** pour les deux grandes sources.

**Implémentation.** Index de fenêtres dans `prepare_corpus.write_batch()` ; permutations et compteurs dans `MixedSource` ; état du sampler sérialisé par `save_checkpoint()`. Garder exactement la même segmentation dans les bras comparés pour ne pas confondre couverture, alignement et curriculum.

**Coût.** 0 VRAM supplémentaire. Un index de 6,65B/1024 fenêtres avec offset 64 bits et score 32 bits représente environ **78 Mo bruts**, hors permutations et métadonnées. Débit neuronal inchangé ; coût CPU et I/O à mesurer.

**Expérience minimale.** Simulation CPU puis plan factoriel :

1. tirage actuel ;
2. couverture seule ;
3. curriculum seul ;
4. combinaison.

**Gain visé.** ≥10 % de temps total économisé à qualité égale ; ce seuil ne découle pas des 37 % de tokens non vus.

**Échec.** La combinaison ne dépasse pas le meilleur ingrédient seul, pénalise les textes longs ou abaisse la qualité d’une source.

**Lien assistant.** Potentiel indirect de meilleure couverture factuelle ; aucune promesse de correction des 139 omissions.

Fondements : [curriculum](https://aclanthology.org/2026.eacl-long.271/), [mélanges avec répétition](https://arxiv.org/html/2605.12715v2).

**H2 — RAFT avec paires contrefactuelles et pondération des unités de réponse**

**Mécanisme.** Pour une même question synthétique, construire des variantes contrôlées :

- passage contenant la réponse ;
- même ensemble avec distracteurs ;
- preuve retirée, avec réponse signalant précisément l’information manquante ;
- fait fictif modifié, avec réponse modifiée en conséquence.

Annoter les unités indispensables de la réponse — valeur, relation, condition, négation — et leur donner un poids modéré, par exemple **2 contre 1** pour le reste. Les annotations viennent du générateur vérifiable, jamais des expressions régulières du benchmark.

Cela associe **sélection de preuve, sensibilité au contenu et complétude**. Le modèle doit changer de réponse lorsque la preuve change, et conserver les éléments indispensables.

**Implémentation.** `prepare_sft.prepare()` pour séparation par familles ; `build_example()` et `write_split()` pour poids ; `PackedData.batch()` et `masked_loss_sum()` pour loss pondérée. Normaliser par la somme des poids sur tout le pas accumulé. Les variantes d’une famille restent dans la même partition.

**Coût.** Aucun nouveau paramètre. Les poids FP32 d’un micro-lot représentent **8 Kio**. Coût du tronc inchangé par token traité ; les passages allongent les exemples et diminuent la proportion de tokens supervisés. Génération CPU possible pour faits fictifs et opérations déterministes ; professeur facultatif pour reformulations vérifiées.

**Expérience minimale.** 200 familles CPU, puis quatre bras à même budget total :

1. données SFT supplémentaires ordinaires ;
2. RAFT simple ;
3. RAFT avec variantes contrefactuelles ;
4. combinaison avec pondération.

**Gain visé.** +5 points sur complétude sémantique indépendante, avec amélioration confirmée du comportement face aux distracteurs.

**Échec.** Gain obtenu par copie, réponses insensibles à la preuve modifiée, abstention systématique ou régression en livre fermé. Ne pas recopier automatiquement les longues chaînes de raisonnement de RAFT dans un contexte limité à 1024.

**Lien cycle 1.** C’est la suite la plus directe de « oracle utile, récupération imparfaite déstabilisante ».

Fondements : [RAFT](https://arxiv.org/abs/2403.10131), [supervision synthétique multitâche](https://aclanthology.org/2025.babylm-main.33/). La pondération proposée est une hypothèse locale, pas un résultat de ces articles.

**H3 — Amorçage formel avec transfert d’attention et résidu de valeurs**

**Mécanisme.** Pré-apprendre des dépendances hiérarchiques sur données formelles, transférer les modules d’attention vers le témoin français et combiner ce transfert avec le résidu de valeurs de ResFormer. Réinitialiser embeddings et MLP dans cette variante afin d’isoler le transfert des mécanismes d’attention.

L’hypothèse est que les relations apprises pendant l’amorçage deviennent plus accessibles aux couches suivantes grâce au résidu de valeurs.

**Implémentation.** Générateur CPU ; transfert explicite dans `train.train()` ; modification de `CausalSelfAttention.forward()`, `Block.forward()` et `GPT.forward()`. Vérifier le recalcul et l’initialisation des projections de sortie nulles.

**Coût.** Phase GPU d’amorçage à inclure intégralement. Exemple de pilote : **5M tokens formels sur 100M tokens totaux**, valeur expérimentale proposée, non issue du papier. Mémoire supplémentaire : tenseur de valeurs et graphe associé ; le tenseur seul vaut 4 Mio dans la configuration 369M.

**Expérience minimale.** Vérification CPU des transferts, puis quatre bras : standard, amorçage seul, ResFormer seul, combinaison. Ajouter un contrôle d’amorçage aléatoire si l’effet formel est positif.

**Gain visé.** ≥10 % de temps total économisé par rapport au standard **et** avantage confirmé sur le meilleur ingrédient seul.

**Échec.** Amorçage non amorti, absence d’interaction positive, progrès syntaxique payé par une régression factuelle.

**Lien assistant.** Potentiel sur dépendances, consignes et conservation des relations ; effet sur les connaissances absentes très incertain. Réservé au prochain pré-entraînement.

Fondements : [langages formels](https://aclanthology.org/2025.acl-long.478/), [ResFormer](https://arxiv.org/abs/2410.17897). NCA fournit un indice supplémentaire sur la transférabilité de l’attention, sans établir cette combinaison. [NCA](https://arxiv.org/abs/2603.10055)

**6. Classement final par gain plausible / coût**

Ce classement est décisionnel : les probabilités de succès locales ne sont pas connues.

| Rang | Piste | Pourquoi maintenant |
|---:|---|---|
| **1** | Audit des omissions, partitions et couverture | CPU ; empêche de choisir une méthode sur un mauvais diagnostic et prépare plusieurs essais. |
| **2** | CCE | Bénéfice mémoire directement mesurable ; réutilisable en pré-entraînement et SFT ; compatibilité fonctionnelle plus favorable que ne le suggère le rapport. |
| **3** | SFT avec distracteurs, puis H2 | Meilleur lien avec les résultats du cycle 1 et le défaut visé. |
| **4** | Refroidissement + EMA | Expérience déjà reportée ; exploite le modèle terminé sans recommencer le pré-entraînement. |
| **5** | Curriculum initial, puis H1 | Faible complexité GPU ; cible un défaut réel d’exposition des données. |
| **6** | ResFormer | Conserver l’essai décidé au cycle 1 ; coût inférieur à une refonte architecturale. |
| **7** | Langages formels, puis H3 | Innovation plausible, sans professeur ; amortissement encore inconnu. |
| **8** | APOLLO-Mini | À remonter seulement si la mémoire libérée permet un régime effectivement meilleur. |
| **9** | Rho-1, NCA, BPT | Intérêt réel, mais coût de scoring, génération ou intégration plus élevé. |
| **10** | COAT, CGLS, RecursiveGPT, Olmo Hybrid, TST | Chantiers distincts ; trop de variables pour le cœur du cycle 2. |
| **11** | MIR, SOAP/FOAM, Soft-MTP | Mauvais alignement avec la rareté du calcul, le plafond mémoire ou les résultats à petite taille. |

**7. Programme proposé en trois étapes pour le cycle R&D 2**

**Étape 1 — Pendant le 369M : préparer les preuves sur CPU.**

Produire quatre éléments : audit sémantique des omissions, partitions sans fuite, index de couverture/curriculum, petit corpus RAFT vérifié. Préparer les références numériques CCE et les contrôles de schedule. Limiter les tâches CPU/I/O pour ne pas ralentir le chargement du run.

**Livrable de décision :** coûts de préparation mesurés, jeu d’évaluation figé, et deux expériences GPU courtes prêtes : CCE et faisabilité SFT. Aucune modification de la trajectoire actuelle.

**Étape 2 — Après la base : obtenir un assistant de référence, puis tester les améliorations proches.**

Conserver l’évaluation de base et le SFT en deux étapes prévus par la roadmap. Effectuer le refroidissement/EMA depuis un checkpoint final commun avec contrôle à tokens égaux. Tester CCE séparément, puis comparer le SFT de référence à RAFT et à H2.

**Critère principal :** complétude sémantique et résistance aux distracteurs sur questions indépendantes, avec conservation des catégories générales et de FrenchBench. Les 49,7 % de v2 restent un repère historique ; le vrai témoin est le **369M avec SFT standard**.

**Étape 3 — Choisir une seule innovation pour le prochain pré-entraînement.**

Présélectionner sur témoin le curriculum et ResFormer ; ne tester H1 ou H3 que si leurs ingrédients fonctionnent. Confirmer le meilleur candidat à trois graines, puis sur un horizon plus proche du budget final — environ **1,2B tokens pour le témoin 54,5M** donne un ratio tokens/paramètres voisin du run actuel.

La durée se calcule après mesure du débit \(q_{\rm témoin}\) : chaque bras coûte \(N/q_{\rm témoin}\), auquel s’ajoutent préparation et évaluations. Ne pas reprendre les 12–21 h du cycle 1 comme un coût acquis.

**Décision proposée : CCE pour la marge matérielle, RAFT/H2 pour l’assistant, curriculum/H1 pour le prochain pré-entraînement.** Conserver refroidissement/EMA et ResFormer dans le programme ; réserver NCA et les refontes architecturales à un cycle ultérieur si ces essais simples ne suffisent pas.