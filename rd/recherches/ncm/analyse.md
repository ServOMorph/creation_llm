**NCM est une piste crédible de génération parallèle, mais le rapport ne fournit pas de raison suffisante pour modifier le 369M ou son SFT.** Les adaptations utiles concernent surtout la mesure de fidélité et la validation de l’inférence ; elles complètent largement le cycle 2 déjà prévu.

Analyse au 18 novembre 2026, après consultation des fichiers cités, du code concerné et des sources primaires. `DOCUMENTATION/INDEX.md` est absent. Aucun fichier modifié, aucun calcul GPU lancé.

**1. Solidité du rapport**

| Point vérifié | Conclusion |
|---|---|
| Identité | **Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation** ; prépublication du 9 octobre 2026. Les neuf auteurs et les affiliations Stanford/Together AI du rapport sont confirmés. [Article](https://arxiv.org/abs/2610.14162) |
| Mécanisme | Tokenizer neuronal à quantification vectorielle résiduelle multirésolution, puis générateur de codes par niveaux ; parallélisme à l’intérieur d’un niveau. « Concept » ne signifie pas concept humain explicite. [Présentation institutionnelle](https://hazyresearch.stanford.edu/blog/2026-10-12-ncm) |
| Accélérations | **21× LM1B, 10× OpenWebText, 31× XSum**, sur H100 SXM 80 Go, face aux comparateurs à diffusion indiqués. Aucun de ces facteurs ne prédit le gain face au GPT maison. [Annonce](https://hazyresearch.stanford.edu/blog/2026-10-12-ncm) |
| Tailles et exposition | Les six couples générateur/tokenizer, longueurs cibles et budgets **13,36 / 127,22 milliards de positions** concordent avec les annexes. Les ratios **1,8× / 17,2×** sont arithmétiquement corrects, sans équivalence en FLOPs ou en durée. [Annexes C](https://arxiv.org/html/2610.14162v1#A3) |
| Reproductions | WMT14 **22,61 BLEU**, XSum **27,93 ROUGE-L**, Sudoku difficile **86,3 %** : valeurs présentes dans la contribution indépendante. Elles reproduisent des checkpoints, pas l’apprentissage. [Issue 9](https://github.com/HazyResearch/ncm/issues/9) |
| RTX 5090 | Les valeurs **86,25 %** et **94,69 / 88,96 / 81,25** sont confirmées ; les comparaisons AR emploient des poids aléatoires et une boucle sans compilation. [Issue 26](https://github.com/HazyResearch/ncm/issues/26) |
| Disponibilité | Dépôt MIT, checkpoints et évaluation disponibles ; recette complète officielle d’entraînement toujours non trouvée dans l’arborescence consultée. [Dépôt](https://github.com/HazyResearch/ncm), [demande ouverte](https://github.com/HazyResearch/ncm/issues/2) |
| Résultats négatifs | Fidélité noms/nombres **89 % / 96 %**, MASSIVE-de **51,0 % contre 57,6 %** : présents dans le rapport MLX, explicitement non validé contre la recette officielle. [Issue 19](https://github.com/HazyResearch/ncm/issues/19) |
| Raffinement | Les **25,04 %** concernent des cellules corrigées aux niveaux 8–9 qui redeviennent fausses, dans une exploration Sudoku particulière ; ce n’est pas un taux général d’erreur NCM. [Notes expérimentales](https://github.com/Quantix-eng/ncm-experiment-notes) |

**Statuts à conserver :** NCM = **prépublication** ; exécutions tierces = **reproductions partielles déclarées** ; adaptations proposées ci-dessous = **hypothèses locales**. Aucun avantage NCM ne satisfait ici « résultat publié et reproduit » pour l’assistant français.

**Corrections et réserves qui changent la décision :**

- **Le cache corrigé n’accélère pas nécessairement une requête isolée.** Sur RTX 3080, Andrew Rose mesure **201,3 ms avec cache corrigé contre 164,5 ms sans cache**, soit environ **22 % plus lent** ; l’intérêt observé porte sur le débit en lot. Le protocole A du rapport doit séparer ces deux objectifs. [Mesures primaires](https://andrewroseit.com/research/ncm/)
- **Exactitude numérique et qualité doivent être séparées.** Dans l’issue 9, le cache FP32 conserve les sorties ; avec BF16 et CUDA graphs, seulement **53,5 %** des sorties XSum restent identiques, malgré un ROUGE-L voisin. Une différence de texte n’est donc pas automatiquement un bogue. L’erreur probable **8 contre 9 passes** est bien reconnue par un auteur. [Discussion](https://github.com/HazyResearch/ncm/issues/9)
- **Le risque de perte factuelle figure déjà dans l’article.** La reconstruction WMT14 est à **98,79 % par token**. Même une fidélité de 99,9 % par token ne garantit pas une réponse entière intacte : sous une hypothèse illustrative d’erreurs indépendantes, `0,999^300 ≈ 74 %`. Il faut mesurer directement les champs et les réponses complètes. [Tableau 1](https://arxiv.org/html/2610.14162v1#S4.SS1)
- **Le bilan hors distribution mérite son contrôle AR.** Les **9,9 %** du petit NCM adapté sont à rapprocher des **8,0 %** du petit AR, également médiocre. Cela indique surtout un échec des petits systèmes entraînés dans ce protocole, sans démontrer une faiblesse spécifique de NCM. [Rapport primaire](https://andrewroseit.com/research/ncm/)
- **Les budgets proposés ne sont pas des estimations de faisabilité.** Une heure d’inférence et quatre heures d’apprentissage sont des enveloppes arbitraires. Dans `gpu_task.py`, au-delà de **45 minutes**, le jeton passe en mode long ; `--minutes` ne force pas l’arrêt du processus.
- **Le rapport minimise le travail préalable de la proposition B.** Sans recette d’entraînement officielle disponible, apprendre un tokenizer français demande d’abord une implémentation et sa validation. Quatre heures GPU n’incluent pas ce chantier.
- **« Critère absent » n’est pas un diagnostic de connaissance manquante.** `bench_assistant.score()` signale l’absence d’une expression attendue. Paraphrase correcte, omission, distraction et ignorance restent à distinguer ; le cycle 2 le prévoit déjà.

**2. Contraintes vérifiées dans le dépôt**

| Ancrage réel | Conséquence |
|---|---|
| [model.py](/D:/ServOMorph/creation_llm/model.py), `GPT.__init__()`, `num_params()` | **335 544 320 paramètres uniques** : `32768×1024 + 24×12×1024²`. « 369M » reste le nom du run. |
| [run_369m.py](/D:/ServOMorph/creation_llm/run_369m.py), `ARGS`, `main()` | Lot **2**, accumulation **32**, contexte **1024**, **113 000 pas** ; réservation 6,5, plafond PyTorch **6,1 Gio**. |
| [train.py](/D:/ServOMorph/creation_llm/train.py), `cap_vram()`, `train()` | Le plafond est calculé en `2**30` octets. À 6 300 tok/s, **7 405 568 000 tokens prennent 13,60 jours**, hors interruptions ; les 12 jours correspondent à une hypothèse de débit supérieure. |
| `model.CausalSelfAttention.forward()` | Le cache accepte un seul nouveau token après préremplissage. NCM demanderait plusieurs positions par niveau : changement structurel, pas simple remplacement du masque. |
| [train_sft.py](/D:/ServOMorph/creation_llm/train_sft.py), `PackedData.batch()`, `masked_loss_sum()` | Cibles assistant décalées et masquées ; conversations isolées par `doc_ids`. Ces cibles ne constituent pas une supervision de codes NCM. |
| `train.MixedSource.batch()` et `train.train()` | Fenêtres aléatoires ; `doc_ids` non transmis en pré-entraînement. NCM ne corrige pas automatiquement cette question documentaire. |
| [generate.py](/D:/ServOMorph/creation_llm/generate.py), `prefill()`, `sample()` | Cache KV déjà présent. Le chemin de génération consulté n’emploie pas d’autocast ; ne pas compter les poids comme BF16 par défaut. |

Le pic historique **6,04 Gio** laisse seulement environ **61 Mio** sous le plafond PyTorch de 6,1 Gio. C’est une mesure antérieure, pas une garantie du pic courant ni du SFT.

**3. Adaptations retenues, avec expériences falsifiables**

Tous les coûts nouveaux ci-dessous sont des **estimations ou plafonds de protocole**, non des mesures. Les gains de qualité et de vitesse attendus sont **inconnus** ; les seuils chiffrés sont des conditions de sélection.

**A — Étendre l’audit de fidélité factuelle du cycle 2 : utile maintenant sur CPU**

- **Mécanisme :** suivre séparément noms, nombres, dates, négations et relations dans les données préparées et les réponses. Distinguer information absente à l’entrée, perdue en préparation, ignorée à la génération et seulement manquée par le score automatique. Le diagnostic de codec NCM motive cette décomposition, sans prouver un gain pour notre GPT. [Source](https://github.com/HazyResearch/ncm/issues/19)
- **Code concerné :** `prepare_sft.build_example()` et `fitting_turns()` ; `train_sft.PackedData.batch()` ; `bench_assistant.score()`, `rescore()`, `summarize()`. Ajouter ultérieurement les contrôles au pilote d’audit **déjà proposé**, mais non implémenté, `rd/experiences/c2_audit.py`.
- **Coût estimé 369M :** **0 Gio VRAM supplémentaire**, aucun changement du calcul du modèle ; **≤1 h CPU**, RAM **≤2 Gio**, relecture humaine supplémentaire **2–4 h**. Un thread, lectures limitées. Débit pré-entraînement théoriquement inchangé, contention CPU/disque à surveiller.
- **Expérience minimale :** annoter les **139 cas historiques `critere_absent`** et vérifier **200 familles contrôlées**, avec changements unitaires de nom, nombre ou négation. Conserver le score historique et ajouter l’évaluation sémantique ; séparer les familles avant augmentation.
- **Succès :** **100 %** des cas reçoivent une catégorie, y compris « indéterminé » ; **0 champ essentiel altéré** par préparation sur les familles contrôlées ; **0 fuite de famille** entre partitions. Toute violation bloque l’usage du lot concerné.
- **Compatibilité/intérêt :** compatible avec le run en cours ; intérêt direct pour RAFT/H2 et pour interpréter « critère absent ». **Gain modèle immédiat : 0**, puisqu’aucun poids n’est changé.
- **Suite SFT :** reprendre B2 du cycle 2, avec son seuil **+5 points de complétude** et comparaison appariée. Cette suite conserve son propre budget ; l’audit ne permet pas de promettre ce gain.

**B — Vérifier l’équivalence du cache maison : transférer la méthode de contrôle**

- **Mécanisme :** comparer les logits obtenus avec préremplissage/cache à ceux du préfixe complet, avec les mêmes tokens imposés. Le bogue NCM justifie le contrôle, **pas la présomption d’un bogue identique ici**. [Correctif documenté](https://github.com/HazyResearch/ncm/pull/5)
- **Code concerné :** `model.CausalSelfAttention.forward()`, `GPT.forward()` ; `generate.prefill()` et `sample()`, notamment le renouvellement du contexte à 1024 tokens.
- **Coût estimé :** contrôle initial sur petit GPT CPU, **≤30 min**, **0 VRAM**. Confirmation ultérieure sur 369M : **≤15 min GPU**, deux chemins exécutés successivement ; aucun paramètre ajouté.
- **Mémoire calculée :** cache complet lot 1 = `2×24×1024×1024×octets`, soit **192 Mio FP32** ou **96 Mio BF16**. Le cache existe déjà : ce n’est pas un surcoût permanent nouveau. Pic total du contrôle à mesurer, plafond **6,1 Gio**.
- **Débit/gain :** aucun gain revendiqué ; aucune modification du débit du run. Le coût de l’évaluation sans cache est supplémentaire et limité au test.
- **Expérience minimale :** **32 séquences**, préfixes courts et proches de la limite ; comparaison des mêmes positions, dont le passage à une fenêtre glissante. Référence FP32 d’abord.
- **Succès proposé :** erreur relative L2 des logits **≤10⁻⁴**, valeurs toutes finies, **0 désaccord d’argmax** lorsque la marge des deux meilleurs logits dépasse deux fois l’erreur maximale. Sinon investiguer avant d’évaluer la vitesse.
- **Compatibilité/intérêt :** CPU pendant le run ; confirmation GPU après sa fin. Aucun effet démontré sur le SFT ou « critère absent », mais garantit que les mesures aval portent sur le bon calcul.

**C — Réduire la proposition A du rapport à un essai NCM externe de 30 minutes : option de veille**

- **Mécanisme :** checkpoint WMT14 officiel, génération standard contre cache corrigé, précision identique. Tester une voie BF16 séparément. Épingler le commit et le correctif. [Code officiel](https://github.com/HazyResearch/ncm), [PR 5](https://github.com/HazyResearch/ncm/pull/5)
- **Ancrage dépôt :** orchestration par `gpu_task.main()`/`run_watched()` et plafond par `train.cap_vram()` dans un futur adaptateur. **Aucune fonction de `model.py` ou `train_sft.py` à remplacer** ; modèle externe dans un processus distinct.
- **Coût estimé RTX 4060 :** environ **0,98 Gio de poids FP32** pour 263,96M paramètres ; enveloppe de travail indicative **1,5–4 Gio** aux lots 1–8, très incertaine, arrêt à **6,1 Gio**. Installation et téléchargement hors budget GPU.
- **Débit :** nombre de requêtes/s et accélération sur 4060 **non estimables de façon fiable** à partir des seules mesures 3080/H100. Surcoût du 369M **nul après sa fin** ; interrompre le run 30 minutes représenterait environ **11,34M tokens de calcul différé**, raison de reporter l’essai.
- **Expérience minimale :** **128 entrées fixes**, lots 1 et 8, échauffement, trois répétitions ; p50/p95, temps complet jusqu’au texte, mémoire allouée/réservée et BLEU sur le même sous-ensemble.
- **Succès technique :** **100 % de tokens identiques** entre voies FP32 déterministes ; mémoire sous plafond. **Intérêt système : ≥20 % de latence économisée au lot 1**, ou **≥20 % de débit supplémentaire au lot 8**, explicitement séparés.
- **Échec :** divergence FP32 inexpliquée, dépassement mémoire, ou aucun gain atteignant ces seuils. En BF16, tolérance proposée **≤0,5 BLEU perdu**, sans confondre ce petit échantillon avec une reproduction complète.
- **Compatibilité/intérêt :** aucune conversion du checkpoint 369M ; aucun bénéfice SFT français démontré. Cet essai vérifie une faisabilité d’inférence seulement.

Pour B/C, le futur pilote doit gérer son propre délai, les pauses et la libération du jeton ; `gpu_task.py --minutes` ne suffit pas. Attendre la fin du pré-entraînement, conformément au cycle 2.

**4. Idées inadaptées ou prématurées**

| Idée | Pourquoi l’écarter du programme actuel |
|---|---|
| Convertir directement le 369M en NCM | Nouveaux codes, entrées reconstruites, têtes et décodeur ; aucun transfert préservant les capacités trouvé. `GPT.forward()`, cache et objectifs d’apprentissage devraient changer ensemble. [Implémentation](https://raw.githubusercontent.com/HazyResearch/ncm/main/models/next_concept.py) |
| Remplacer uniquement le masque causal | Les positions d’un niveau prédisent des codes à partir de représentations antérieures ; elles ne reçoivent pas les mêmes entrées que notre GPT. Le masque seul ne construit ni cibles ni hiérarchie. [Implémentation](https://raw.githubusercontent.com/HazyResearch/ncm/main/models/next_concept.py) |
| Tokenizer français entraîné en quatre heures — proposition B du rapport | Convergence, recette reproductible et pic mémoire inconnus. Le seuil de fidélité est utile ; le budget ne démontre aucune faisabilité. À rouvrir après disponibilité d’une recette validée. [Demande de code](https://github.com/HazyResearch/ncm/issues/2) |
| Comparaison française complète — proposition C | Étude légitime, mais nouveau chantier : tokenizer, encodage, générateur, SFT et évaluations à compter. Aucun débit ni budget total crédible sur 4060 trouvé. Ne pas l’inscrire comme expérience minimale. [Dépôt](https://github.com/HazyResearch/ncm) |
| Utiliser les codes grossiers comme plan factuel garanti | Représentations apprises, raffinements non monotones ; une information correcte peut disparaître. Aucune preuve de réduction des omissions françaises. [Observations](https://github.com/Quantix-eng/ncm-experiment-notes) |
| Réouvrir CFG/MBR grâce aux résultats NCM | Le résultat d’une autre architecture n’annule pas les essais locaux. MBR ajoute des candidats et ne garantit pas la présence du fait recherché. [Présentation des auteurs](https://hazyresearch.stanford.edu/blog/2026-10-12-ncm) |

**5. Articulation avec les décisions du projet**

- **[Programme cycle 2](/D:/ServOMorph/creation_llm/rd/programme_cycle2.md)** : A complète **A1/A2**, B complète **A3**. C reste hors du budget principal, après le témoin et les priorités de complétude. NCM ne remplace ni CCE, ni RAFT/H2, ni STP.
- **[Bilan cycle 1](/D:/ServOMorph/creation_llm/rd/bilan_cycle1.md)** : **49/90 → 71/90** avec passages relus soutient la priorité donnée à la sélection et à l’utilisation des preuves. Cela ne prouve pas que toute omission vient de la récupération ; les **54/90** avec récupération hybride maintiennent la question de robustesse.
- **[Sélection](/D:/ServOMorph/creation_llm/rd/selection.md)** : conserver la comptabilité complète des composants et les comparaisons à temps égal. Les petits écarts de score sans intervalle favorable restent insuffisants.
- **[Roadmap 369M](/D:/ServOMorph/creation_llm/roadmap_modele_369m.md)** : conserver base → évaluation → conversion chat → SFT standard en deux étapes. Corriger seulement l’interprétation documentaire des **12 jours**, de la marge mémoire et du lot historique **4×16**, alors que le lanceur actuel utilise **2×32**.
- **Veilles antérieures et pôle mathématiques** : NCM ajoute un problème de fidélité d’un codec appris ; il ne remplace ni TOP/MTP, ni SuperBPE, ni les stratégies MoE. Une perte sur codes, même divisée par des octets, ne devient pas automatiquement comparable à la vraisemblance du texte : le décodeur et la reconstruction doivent entrer dans la mesure.

À SURVEILLER — architecture prometteuse pour l’inférence parallèle, mais aucun avantage démontré sur la complétude française ou le budget global du 369M ; reprendre ses contrôles de fidélité sans changer le run ni les priorités SFT.