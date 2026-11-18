**H-Net mérite un prototype séparé, mais ne justifie aucune modification du pré-entraînement en cours.** L’adaptation la plus immédiate est son protocole de comparaison en octets ; la hiérarchie apprise exige une nouvelle architecture et une nouvelle base. Aucun résultat consulté ne démontre qu’elle corrige le défaut « critère absent ».

Analyse au 18 novembre 2026, après consultation du code, des documents demandés et des sources ci-dessous. `DOCUMENTATION/INDEX.md` est absent. Aucun fichier modifié, aucune expérience lancée.

**Statuts employés :** **P** = publication vérifiée, réplication exacte non établie ; **PP** = prépublication ; **T** = compte rendu technique des auteurs ou d’un tiers ; **H** = hypothèse locale. Aucun gain H-Net n’est **établi, publié et reproduit sur notre configuration**. Les seuils expérimentaux proposés sont des décisions de sélection, pas des gains prédits.

**Solidité du rapport fourni**

| Point vérifié | Conclusion |
|---|---|
| Identification, auteurs, publication | Correct : *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling*, Sukjun Hwang, Brandon Wang, Albert Gu ; actes ICLR 2026 confirmés. **P**. [Actes officiels](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f1aa53cac69ef6980ca4a911ffcf278b-Abstract-Conference.html). |
| Résultats initiaux | Les tableaux 1–2 confirment notamment 0,756 → 0,743 BPB et 53,3 → 55,5 pour Transformer L contre H-Net L à deux niveaux ; 870M contre 1,3B pour les deux modèles atteignant 55,5. Aucune division par deux de la VRAM n’en découle. [Article](https://arxiv.org/html/2507.07955v2). |
| Résultats chinois et ADN | 59,9 → 66,3 et efficacité en données ×3,6 confirmés dans l’article. Ce sont des expériences distinctes ; aucune extrapolation quantitative au français n’est justifiée. [Article](https://arxiv.org/html/2507.07955v2). |
| Annonce d’octobre 2026 | Les multiplicateurs 1,28/1,83/2,06 et la comparaison 3,67B/3,08B, 4,44/4,36 GFLOPs/octet figurent bien dans le billet. Le rapprochement vers 1T contre 11T est une déclaration des auteurs avec limites de contrôle reconnues. **T**, pas reproduction indépendante du ×10. [Cartesia I](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1). |
| Comparateur BLT | Le rapport distingue correctement « BLT-like » d’une reproduction complète de BLT. Les multiplicateurs concernent des courbes calcul/perte, pas directement la précision de l’assistant. [Cartesia II](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-2). |
| Réimplémentation RTX 5090 | Taille, volume et BPB déclarés concordent avec le dépôt. Celui-ci précise tester les tendances avec l’ancien routeur, pas reproduire les chiffres Cartesia. **T indépendant**, pas réplication exacte. [Dépôt](https://github.com/bigwolfeman/hnet-research). |
| Parameter Golf | Chiffres initiaux concordants, mais bilan incomplet : la seconde étude annonce **1,2070 BPB en quatre heures**, rejoignant ses comparateurs. Cela ne valide toujours pas la RTX 4060. [Dépôt](https://github.com/DariusFeher/parameter-golf-byte-hnet). |
| H-Net++ | Auteurs, corpus et gains annoncés confirmés dans le résumé ; **PP**, sans reproduction externe vérifiée ici. [Prépublication](https://arxiv.org/abs/2508.05628). |

Les corrections importantes portent surtout sur l’interprétation et le protocole :

- **« Critère absent » est un résultat de regex, pas un diagnostic cognitif.** `bench_assistant.score()` vérifie l’absence de motifs `ok`. Une paraphrase correcte, une omission, une erreur ou une distraction peuvent produire ce même label. Le cycle 2 corrige déjà cette ambiguïté.
- **300 résultats ne signifient pas 300 questions indépendantes.** `bench_assistant.run()` répète les items pour trois graines. Les intervalles doivent regrouper les observations par question, voire famille.
- **Un milliard d’octets par bras n’est pas une expérience GPU courte.** Au débit indicatif du 369M, `6 300 × 4,45 = 28 035 octets/s`, cela représente **9,91 h par bras**, soit **39,6 h pour quatre bras**, hors évaluations. Un petit modèle pourrait aller plus vite : son débit reste à mesurer.
- **Le ratio cible n’est pas le ratio obtenu.** Dans le tableau 1 initial, les cibles 6 et 3×3 donnent environ 4,8 et 7,0 au niveau L. Dimensionner la mémoire sur la cible seule serait une erreur. [Article](https://arxiv.org/html/2507.07955v2).
- **256 entrées est une simplification pour le chat.** Il faut aussi représenter rôles et fins de tour, par identifiants réservés ou séquences d’octets explicites.
- **Le rapport ne fournit pas les estimations locales demandées.** « À mesurer » est honnête, mais doit être accompagné d’une décomposition du coût ; donner un pic H-Net global précis sans architecture arrêtée serait inventer une mesure.

**Contraintes effectivement vérifiées dans le dépôt**

| Élément | Conséquence |
|---|---|
| [model.py](/D:/ServOMorph/creation_llm/model.py), `GPT.__init__()`, `num_params()` | **335 544 320 paramètres uniques**, embeddings liés, normalisations sans paramètres appris. « 369M » reste le nom du run. |
| [run_369m.py](/D:/ServOMorph/creation_llm/run_369m.py), `ARGS` | Lot 2, accumulation 32, contexte 1 024, 113 000 pas : **7 405 568 000 tokens**, soit **13,60 jours à 6 300 tok/s**, hors interruptions. |
| [train.py](/D:/ServOMorph/creation_llm/train.py), `cap_vram()`, `train()` | Plafond **6,1 Gio** de l’allocateur PyTorch ; réservation VertIA 6,5. Poids FP32 avec autocast BF16, pas poids intégralement BF16. |
| `MixedSource.batch()`, `train()` | Fenêtres non alignées ; `doc_ids` absent du pré-entraînement. Un nouveau comparatif devra appliquer la même politique documentaire aux deux bras. |
| `build_optimizer()` | Muon sélectionné par préfixe `blocks.` ; les nouveaux modules tomberaient sinon dans AdamW. La boucle remplace aussi tous les LR par une valeur commune. |
| [train_sft.py](/D:/ServOMorph/creation_llm/train_sft.py), `PackedData.batch()`, `masked_loss_sum()` | Masquage des cibles et isolation des conversations déjà présents ; ils devront être conservés à tous les niveaux hiérarchiques. |

Le pic historique **6,04 Gio** laisse environ **61 Mio** sous 6,1 Gio, si l’on reprend les unités du programme cycle 2. Ce n’est ni une mesure du pic actuel, ni une marge garantie pour le SFT.

**Adaptation 1 — Comparaison en BPB et audit de segmentation : utile dès maintenant**

**Mécanisme et statut :** **H locale**, inspirée du protocole publié : mesurer la somme des pertes sur les mêmes textes et la diviser par leur nombre exact d’octets UTF-8. Cela évite de confondre meilleure tokenisation, contexte plus long et meilleur modèle. [Protocole H-Net](https://arxiv.org/html/2507.07955v2).

**Points du code :** `train.evaluate()` fournit actuellement une moyenne par token ; `MixedSource.batch()` ne conserve pas les octets correspondants. Prévoir un évaluateur externe utilisant `GPT.forward()` et des segments textuels figés, sans changer la métrique historique du run.

**Gain attendu chiffré :** **0 gain direct de qualité** ; gain expérimental recherché : comparaisons valides. Aucun gain indirect quantifiable établi.

**Coût estimé pour le 369M :** **0 VRAM supplémentaire et 0 baisse de débit GPU** pour le seul audit CPU hors boucle, sous réserve de contention CPU/disque. Échantillon proposé : 1 000 segments, un thread, enveloppe CPU ≤30 min, RAM ≤1 Gio. Une nouvelle évaluation du modèle aurait un coût distinct, non compris.

**Expérience minimale :** mesurer octets/token par source, longueurs des réponses et round-trip texte→tokens→texte ; définir explicitement le traitement des marqueurs spéciaux et de la normalisation. Pour les unités effectivement évaluées :
\[
\mathrm{BPB}=\frac{\sum \mathrm{NLL}}{\ln(2)\times \text{octets correspondants}}.
\]
Ne pas multiplier une moyenne globale par un ratio issu d’un autre échantillon.

**Succès / échec proposés :** 100 % des segments expliqués sans perte silencieuse de texte ; accord des deux calculs de BPB à **10⁻⁶** sur un jeu synthétique. Échec si dénominateurs, cibles ou horizons diffèrent. Arrêter le travail auxiliaire si le débit du run baisse de **>1 % sur deux périodes comparables de 15 min**, conformément au cycle 2.

**Compatibilité et SFT :** compatible avec le run inchangé ; permet de distinguer troncature, fragmentation et omission réelle. Ne corrige pas les connaissances.

**Adaptation 2 — Hiérarchie sur les tokens BPE : ablation locale moins coûteuse à préparer**

**Mécanisme :** conserver le tokenizer et redistribuer les 24 blocs en encodeur **2**, cœur **20**, décodeur **2**, avec routeur causal, reconstruction et chemin résiduel. Réduire les positions du cœur, tout en prédisant chaque token original. **H**, pas reproduction du H-Net officiel : ses encodeurs/décodeurs utilisent principalement Mamba-2. [Architecture officielle](https://arxiv.org/html/2507.07955v2), [ablation sous-mots indépendante](https://github.com/DariusFeher/parameter-golf-byte-hnet).

**Points du code :** `GPT.__init__()/forward()`, `Block.forward()`, `CausalSelfAttention.forward()` pour les positions et caches ; `build_optimizer()` pour les groupes ; `train()` pour la perte de ratio. Prévoir une classe expérimentale distincte : `GPT.encode()` n’existe pas actuellement.

**Gain attendu chiffré : inconnu.** La proposition isole l’intérêt de la hiérarchie sans refaire tout le corpus.

**Estimation mémoire explicitement conditionnelle, largeur 1 024 :**
- Deux projections du routeur plus une projection résiduelle : **3 × 1 024² = 3 145 728 paramètres**.
- Avec poids, gradients et deux moments FP32 sous AdamW : **48 Mio supplémentaires**, hors temporaires.
- Un tenseur BF16 de forme `[2,1024,1024]` représente **4 Mio** ; conserver deux tels tenseurs ajoute **8 Mio**.
- Ces **56 Mio de composants** consommeraient presque la marge historique ; ce n’est pas un pic net, car compression et recalcul changent simultanément les activations. Pic global non estimable fiablement avant prototype.

**Estimation de débit :** à compression exactement 2, le travail linéaire des blocs passe idéalement de 24 à `4+20/2=14` blocs équivalents par position. Cela donne **×1,71 sur cette seule composante**, avant routage, reconstruction, padding et compilation ; **pas une prévision de 10 800 tok/s**. Le débit mural peut diminuer malgré cette économie.

**Expérience minimale :** CPU, largeur 128 et profondeur réduite, 50–100 pas ; comparer routeur appris et sélection fixe causale à même ratio. Puis, après la base, créneau GPU **≤30 min**, avec au moins 20 pas de chauffe et 100 pas mesurés si le budget le permet.

**Succès technique proposé :** gradients finis à 100 %, aucun accès au futur ni mélange inter-documents ; écart maximal de logits passe complète/incrémentale **≤10⁻⁴ en FP32** sur le petit témoin ; pic ≤6,1 Gio. Échec dès qu’un contrôle échoue ; test incomplet si la compilation absorbe le créneau.

**Succès qualitatif ultérieur :** **≥0,01 BPB gagnée à temps égal**, ou **≥10 % de temps total économisé** vers une BPB fixée à l’avance ; confirmation sur trois graines d’entraînement. Sinon ne pas retenir l’architecture.

**Compatibilité et SFT :** incompatible avec une reprise ordinaire du checkpoint actuel. Garder le vocabulaire simplifie les données SFT, mais une nouvelle base reste nécessaire ; aucun gain démontré sur « critère absent ».

**Adaptation 3 — H-Net par octets : candidat de recherche pour une future base**

**Mécanisme et statut :** encodeurs/décodeurs fins, routeur appris, cœur compressé ; tester un niveau avant deux. **P pour le mécanisme initial, H pour notre adaptation française.** Le routeur public compare des états voisins, sélectionne au seuil strict `>0,5`, puis reconstruit causalement ; le STE ne rend pas la sélection discrète exactement différentiable. [Routeur](https://github.com/goombalab/hnet/blob/main/hnet/modules/dc.py), [structure et STE](https://github.com/goombalab/hnet/blob/main/hnet/models/hnet.py).

**Points du code :** mêmes fonctions modèle/optimiseur que ci-dessus ; remplacer aussi les hypothèses `uint16`/BPE de `load_data()` et `MixedSource`, adapter `prepare_sft.build_example()`, `PackedData.batch()`, `masked_loss_sum()`. `convert_chat.expand_vocab()` seul ne convertit pas un GPT en H-Net.

**Gain attendu chiffré : inconnu.** Potentiel sur les formes rares, accents et variations orthographiques ; aucun résultat établi sur la complétude des réponses françaises.

**Estimation mémoire décomposée pour la taille actuelle :**
- Remplacer théoriquement la table liée 32 768×1 024 par 256×1 024 retire **33 292 288 paramètres**, soit **127 Mio de poids FP32**.
- Avec gradient et deux moments AdamW FP32 : économie de composants d’environ **508 Mio**, hors marqueurs de chat.
- À texte comparable, **1 024 tokens ≈4 557 octets** selon le ratio local moyen. Un seul tenseur BF16 `[2,4557,1024]` occupe **17,8 Mio**, contre 4 Mio actuellement.
- Les nouveaux poids, activations et espaces de travail peuvent dépasser l’économie. **Aucune estimation nette fiable du pic 369M n’est disponible** ; ne pas budgéter un gain mémoire.

**Estimation de débit par scénario, pas prévision :** si le ralentissement mural était ×1,46 comme dans l’essai tiers, le débit équivalent serait **4 315 tokens BPE/s**, et le volume actuel prendrait **19,9 jours** ; à ×2, **3 150/s et 27,2 jours**. Les matériels et architectures diffèrent : ces calculs illustrent le risque, sans constituer une fourchette prédictive. [Essai tiers](https://github.com/bigwolfeman/hnet-research).

**Expérience minimale :** référence CPU minuscule, puis prototype GPU réduit, budget **30 min maximum**. Vérifier UTF-8, frontières, gradients du routeur, ratio réalisé, génération incrémentale et cas où presque toutes les positions sont conservées. Le code officiel dépend de kernels spécialisés : une référence CPU réécrite valide les invariants, pas leurs performances. [Dépendances](https://github.com/goombalab/hnet/blob/main/pyproject.toml).

**Succès / échec :** mêmes contrôles numériques que l’adaptation 2 ; ratio moyen à **±15 % de la cible** après le pilote, absence de dépassement même sous forte densité de frontières. Une instabilité du ratio ou un dépassement mémoire arrête ce candidat local, sans réfuter H-Net en général.

**Comparaison qualitative après faisabilité :** mêmes documents, mêmes octets traités, contexte textuel comparable et contrôle à découpage fixe ; seuils **0,01 BPB / 10 %** ci-dessus. Reporter séparément paramètres, temps total, FLOPs et octets ; ne pas promettre leur égalité simultanée.

**Compatibilité et SFT :** aucun branchement sur le run actuel. Réinitialiser les états à chaque conversation, mais conserver le contexte entre ses tours ; masquer les octets du prompt, gérer les marqueurs de rôle et comparer des budgets de texte produit équivalents.

**Validation assistant commune aux adaptations 2–3**

Elle n’est pertinente qu’après obtention d’une base compétitive. Adapter `bench_assistant.run()` et le moteur de génération ; conserver `score()` comme historique et ajouter l’audit sémantique A1 du cycle 2.

**Coût estimé :** deux recettes SFT complètes et évaluations, puis confirmation du finaliste ; durée `T_BPE + T_HNet`, inconnue avant mesure. Pic SFT à mesurer séparément ; aucune marge ne se déduit du pré-entraînement.

**Gain attendu : inconnu. Succès proposé :** **+10/300 résultats historiques**, **+5 points de complétude réelle** sur les familles dédiées, intervalle apparié à 95 % favorable en regroupant par question ; aucune catégorie sous la référence, conformément à la roadmap. Échec si seule la BPB progresse, si l’amélioration est un effet de regex, ou si les réponses deviennent plus incomplètes.

Les futures tâches GPU passent par `gpu_task.main()` : acquisition et contrôle du jeton, vérification de l’absence du 369M, puis exécution. **`--minutes 30` ne constitue pas à lui seul un arrêt forcé du programme** : le banc d’essai devra imposer sa propre limite de temps. Aucun lancement n’est proposé pendant le run prioritaire.

**Idées inadaptables dans le périmètre actuel**

| Idée | Motif d’exclusion |
|---|---|
| Remplacer maintenant BPE par les octets | Tables, données, positions, caches et distributions changent ; aucune conversion de checkpoint validée. |
| Ajouter seulement le routeur au GPT | Sans reconstruction, supervision à résolution originale et résidu, ce n’est pas H-Net ; risque de supprimer l’information nécessaire. |
| Utiliser tout le GPT actuel sur 4 557 octets | Longueur ×4,45 ; terme quadratique de l’attention ≈×19,8 à largeur constante. Ce calcul concerne le travail, pas nécessairement la mémoire des kernels SDPA. |
| Déduire « 12 jours → 1,2 jour » de l’annonce ×10 | Confusion entre efficacité en données/FLOPs et temps mural ; nouvelle version non reproduite localement. |
| Copier H-Net++ pour traiter les accents français | Résultat persan avec plusieurs changements simultanés ; aucun effet isolé démontré pour le français. |
| Présenter les groupes appris comme une solution au raisonnement ou aux faits manquants | Aucune preuve causale ; la lisibilité des frontières ne mesure pas la complétude de l’assistant. |

**Articulation avec les décisions du projet**

- [Programme cycle 2](/D:/ServOMorph/creation_llm/rd/programme_cycle2.md) : l’adaptation 1 **complète A1/A3**. Les adaptations 2–3 seraient un candidat architectural supplémentaire **hors budget principal**, après la phase assistant. Elles ne remplacent ni CCE, ni le témoin SFT, ni RAFT/H2.
- [Bilan cycle 1](/D:/ServOMorph/creation_llm/rd/bilan_cycle1.md) : **49/90 →71/90 avec passage pertinent** soutient d’abord le travail sur les preuves et leur sélection. H-Net ne contredit pas ce résultat et n’en traite pas directement la cause.
- [Sélection](/D:/ServOMorph/creation_llm/rd/selection.md) et veilles : H-Net **complète la piste tokenisation/SuperBPE**, mais change l’architecture entière. Il ne réhabilite pas les pistes d’inférence closes.
- [Roadmap](/D:/ServOMorph/creation_llm/roadmap_modele_369m.md) : aucune modification des phases en cours ; préserver évaluation de base et SFT standard. Les corrections de durée, taille réelle et marge mémoire sont déjà reprises dans le cycle 2.
- Pour le prochain pré-entraînement, H-Net devra **battre le meilleur candidat simple à temps et mémoire comparables**, puis occuper l’unique place d’innovation retenue. Cumuler d’emblée H-Net, ResFormer et curriculum empêcherait d’attribuer le gain.

À TESTER — sur prototype séparé après les priorités du cycle 2 ; mécanisme crédible, mais bénéfice français, coût RTX 4060 et réduction des omissions restent non démontrés.