**LexisLM conforte les audits prévus au cycle 2, sans justifier de changement du pré-entraînement actuel.** Le rapport est globalement prudent et ses principaux chiffres sont corrects ; ses propositions doivent toutefois être resserrées et certaines limites mieux explicitées.

Analyse au 18 novembre 2026, fondée sur les sources primaires et les fichiers locaux cités, avec lecture ciblée des veilles. `DOCUMENTATION/INDEX.md` est absent. Aucun fichier modifié, aucun calcul GPU lancé.

**Statuts :** **E** = résultat publié et reproduit indépendamment ; **PP** = prépublication ; **H** = hypothèse locale ; **L** = constat du code ou mesure locale documentée. Aucun gain LexisLM applicable au projet ne relève de **E**. Tous les seuils expérimentaux ci-dessous sont des décisions proposées, pas des gains prédits.

**1. Solidité du rapport**

| Point vérifié | Conclusion |
|---|---|
| Identification | Titre, six auteurs et dépôt du 6 novembre 2026 confirmés. Statut **PP**. [Notice arXiv](https://arxiv.org/abs/2611.10209) |
| Recette et résultats de base | Tailles, architecture, budget et tableaux de scores concordent. Les perplexités concernent les **meilleurs checkpoints évalués**, pas nécessairement les poids finaux. [Article](https://arxiv.org/html/2611.10209v1) |
| Résultats Instruct | Les deux tableaux du rapport concordent avec les fiches ; licences CC BY-NC 4.0 annoncées. [500M](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-500m-Instruct), [1B](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-1B-Instruct) |
| Reproductibilité | Recherche complémentaire sans code complet ni reproduction indépendante identifiés. Configurations brutes toujours non récupérées par l’outil : dimensions vérifiées dans l’article et les fiches uniquement. |
| Accès aux bases | À actualiser : la fiche **1B est accessible** et annonce Apache-2.0. Accès à la base 500M non abouti. [Base 1B](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-1B) |
| Matériel LexisLM | GPU précis, durée, débit et pic VRAM non trouvés dans les sources consultées. Aucun coût RTX 4060 démontré. |

**Corrections et réserves scientifiques :**

- Le papier signale une exposition aux données d’entraînement ARC, mais sa section corpus n’en précise pas la quantité malgré le renvoi interne. C’est une lacune de traçabilité, pas une preuve de contamination du test. [Article, §4 et §6.2](https://arxiv.org/html/2611.10209v1)
- Pour comparer base et SFT, utiliser ARC-Easy **normalisé contre normalisé** : 41,9 → 41,58 et 44,6 → 44,15. Les valeurs brutes du premier tableau ne sont pas directement comparables aux fiches Instruct. [Article](https://arxiv.org/html/2611.10209v1), [500M-Instruct](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-500m-Instruct), [1B-Instruct](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-1B-Instruct)
- Le passage unique est une recette, **sans ablation démontrant sa supériorité**. La fertilité est partiellement calculée sur des mots filtrés par fréquence ; elle ne mesure pas toute la longue traîne. [Article, §3 et §8](https://arxiv.org/html/2611.10209v1)
- « Critère absent = information manquante » est un diagnostic trop fort : `bench_assistant.score()` constate l’absence d’un motif regex. Une paraphrase correcte peut échouer. Le cycle 2 corrige déjà cette interprétation.
- Les **71/90 contre 49/90** du cycle 1 sont des résultats locaux utiles, mais concernent 30 questions et trois graines de génération : ce ne sont pas 90 questions indépendantes.
- Les **600M tokens** proposés pour la couverture constituent une confirmation coûteuse, pas une expérience minimale courte : à 6 300 tok/s, **26,46 h** pour le 369M, hors évaluations. Le débit du témoin réduit reste à mesurer.
- « Checkpoints existants » doit être vérifié : `train.save_checkpoint()` et les boucles d’entraînement écrasent `ckpt.pt`. Une trajectoire historique complète n’est pas automatiquement disponible.

**2. Base matérielle et logicielle vérifiée**

Dans [model.py](/D:/ServOMorph/creation_llm/model.py), `GPT.num_params()`, `GPT.__init__()` et `MLP.__init__()` donnent :

\[
N=32768\times1024+24\times12\times1024^2=335\,544\,320.
\]

Le nom « 369M » reste l’identifiant du run ; son nombre de paramètres uniques est **335,54M**, les embeddings étant liés et les RMSNorm sans paramètres appris.

[run_369m.py](/D:/ServOMorph/creation_llm/run_369m.py) configure **2 × 32 × 1 024**, et non le lot 4 × 16 encore mentionné dans la phase initiale de la roadmap. Budget : **7 405 568 000 tokens**, soit **13,60 jours à 6 300 tok/s**, hors interruptions.

`train.cap_vram()` interprète 6,1 en **Gio**. Le pic historique de 6,04 laisse environ **61 Mio sous ce plafond**, sans garantie sur les futurs pics. La réservation de 6,5 ne constitue pas une marge disponible de 1 Go.

Les 16 octets/paramètre du rapport décrivent correctement un scénario poids, gradients et deux moments FP32, **pas toute implémentation BF16**. Localement, `train.train()` conserve des paramètres FP32 avec autocast BF16. Remplacer Muon par AdamW sur les blocs ajouterait approximativement un moment FP32 : **1,125 Gio**, hors temporaires. Ce changement ne tient pas dans la marge historique.

**3. Adaptations proposées**

**A — Audit CPU de tokenisation et de provenance : compléter A1/A3, priorité raisonnable**

- **Mécanisme — H :** mesurer séparément occurrences, formes distinctes, mots rares, caractères et octets ; vérifier réversibilité et stabilité des IDs. Cela détecte une fragmentation ou une partition trompeuse, sans améliorer directement le modèle.
- **Code concerné — L :** [prepare_corpus.py](/D:/ServOMorph/creation_llm/prepare_corpus.py), `train_tokenizer()`, `parquet_texts()`, `write_batch()`, `encode_sources()` ; [convert_chat.py](/D:/ServOMorph/creation_llm/convert_chat.py), `build_chat_tokenizer()`.
- **Point nouveau concret :** le tokenizer est appris sur les premiers documents des sources ; `write_batch()` réserve également les premiers documents à la validation. Le code permet donc un recouvrement entre apprentissage du tokenizer et validation du modèle. Ce n’est pas une exposition des poids aux réponses, mais l’audit de tokenisation doit utiliser un échantillon réellement extérieur.
- **Expérience minimale :** 1 000 textes par source, plafonnés à 10 000 caractères chacun, plus 200 cas Unicode ; vérifier les 32 768 correspondances ID→token avant/après extension chat. Conserver les métriques par source et une moyenne pondérée au mélange réel.
- **Coût estimé 369M :** **0 VRAM supplémentaire**, budget CPU **≤30 min**, RAM **≤1 Gio** ; effet direct sur le débit modèle **0 %**, contention CPU/disque à surveiller.
- **Succès :** 0 erreur de réversibilité, 0 ID de base modifié, 100 % des textes d’audit avec provenance documentée ; signal exploitable si anomalie reproduite sur un second échantillon indépendant.
- **Échec :** ≥1 corruption ou ID modifié bloque une future conversion ; aucune anomalie reproductible signifie **aucun changement de tokenizer**. Une différence de fragmentation ≥10 % déclenche seulement une investigation.
- **Gain attendu chiffré :** **0 point directement par l’audit** ; gain d’un éventuel futur tokenizer inconnu.
- **Compatibilité et SFT :** réalisable pendant le run en lecture seule ; utile aux accents, nombres et termes rares, sans preuve de réduction de `critere_absent`. Ne pas introduire NFC dans le tokenizer courant sans réentraînement compatible.
- **Sources :** [LexisLM, analyse du tokenizer](https://arxiv.org/html/2611.10209v1#S3) ; [I6 local](/D:/ServOMorph/creation_llm/rd/pole_mathematiques.md).

**B — Auditer les omissions puis sélectionner les checkpoints sur les réponses : priorité haute, déjà prévue**

- **Mécanisme — H :** distinguer erreur du score, omission réelle, erreur factuelle et distraction ; confronter perte SFT, complétude et connaissances.
- **Code concerné — L :** [bench_assistant.py](/D:/ServOMorph/creation_llm/bench_assistant.py), `score()`, `rescore()`, `run()`, `summarize()` ; [train_sft.py](/D:/ServOMorph/creation_llm/train_sft.py), `evaluate()`, `train()` ; `train.save_checkpoint()` pour de futures sauvegardes distinctes.
- **Expérience minimale CPU :** annoter les 139 occurrences historiques `critere_absent`, aveuglément quant à la variante ; préserver le score historique et ajouter un score sémantique séparé. Double annotation de 30 cas.
- **Coût estimé :** **0 VRAM**, CPU **≤1 h**, relecture **4–8 h** ; effet direct sur le débit **0 %**.
- **Succès de l’audit :** 100 % des cas classés et accord initial ≥90 % sur les 30 doubles annotations ; sinon clarifier la grille avant toute conclusion.
- **Extension GPU après la base :** comparer les deux étapes SFT sur les mêmes questions et graines ; pilote de 30 questions, puis benchmark complet si le profil passe.
- **Coût GPU estimé :** **0 surplus structurel** par rapport au benchmark existant ; même débit par checkpoint, mais temps total multiplié par le nombre de checkpoints. Pilote borné dans son code à **30 min**.
- **Mémoire :** poids locaux FP32 ≈**1,25 Gio**, cache KV FP32 à batch 1 et 1 024 tokens ≈**0,188 Gio**, hors logits et temporaires. Le pic peut être dominé par `generate.load_model()`, qui charge aussi le checkpoint et ses états sur le GPU : ne pas promettre une évaluation « sous 2 Go ». Réserver 6,5 et mesurer sous plafond 6,1 Gio.
- **Succès d’une sélection :** gain ≥**10/300**, soit **3,33 points**, contre le témoin 369M SFT ; intervalle apparié à 95 % excluant zéro, aucune catégorie sous la v2 et perte FrenchBench ≤8 points face à la base.
- **Échec :** gain inférieur au seuil, intervalle incluant zéro, régression d’un garde-fou ou simple amélioration de perte. Rééchantillonner par question ; trois générations ne valent pas trois entraînements.
- **Gain attendu chiffré :** inconnu ; l’audit seul apporte **0 gain modèle**. C’est la piste la plus directement pertinente pour le défaut observé.
- **Sources :** [limite des évaluations Instruct](https://huggingface.co/ComputationalLinguisticsPlatform/LexisLM-500m-Instruct), [bilan local](/D:/ServOMorph/creation_llm/rd/bilan_cycle1.md), [cycle 2](/D:/ServOMorph/creation_llm/rd/programme_cycle2.md).

**C — Couverture sans remise : simulation CPU maintenant, intégration à C1 ensuite**

- **Mécanisme — H :** parcourir une permutation de blocs pour réduire les répétitions accidentelles, tout en conservant les répétitions intentionnelles des petites sources.
- **Code concerné — L :** [train.py](/D:/ServOMorph/creation_llm/train.py), `MixedSource.batch()`, `get_batch()`, `train()`, `save_checkpoint()` ; mélange défini par `run_369m.spec()`.
- **Contrôle causal indispensable :** comparer avec/sans remise sur **la même grille de blocs**. Comparer des positions aléatoires actuelles à des blocs fixes changerait aussi la segmentation. Garder alignement documentaire, masque et curriculum constants.
- **Expérience minimale CPU :** 100 000 fenêtres simulées, mêmes quotas de sources ; vérifier couverture, nombre de répétitions et reprise exacte. Budget **≤30 min**, RAM **≤1 Gio**, **0 VRAM**.
- **Succès fonctionnel :** 0 bloc répété avant épuisement de sa source, quotas à **±0,5 point**, reprise identique sur les **1 000 indices suivants**. Sinon échec avant GPU.
- **Coût estimé sur 369M :** **0 état GPU supplémentaire**, variation de débit supposée **0 à −1 %** si les accès CPU restent masqués ; hypothèse, pas mesure. Arrêt si baisse persistante >1 %.
- **Validation matérielle courte :** après la base, deux bras de **50 pas** chacun : 6 553 600 tokens au total, soit **17,3 min arithmétiques** à 6 300 tok/s ; prévoir jusqu’à **45 min**, compilation et évaluations comprises. Ce test vérifie débit et mémoire, pas la qualité.
- **Validation scientifique :** reprendre le témoin **54,53M de C1** ; pilote 100M tokens par bras, une graine, puis confirmation selon le cycle 2. Les deux bras de 100M sur le 369M coûteraient déjà **8,82 h** hors évaluations.
- **Succès qualitatif proposé :** ≥**0,01 nat/token** gagnée à temps total égal, intervalle apparié favorable ; sélection finale selon C1 : ≥**10 % de temps total économisé** à qualité commune.
- **Échec :** absence du gain, gain annulé à temps égal, modification involontaire des proportions ou coût d’indexation absorbant l’économie.
- **Gain attendu chiffré :** inconnu. Effet SFT indirect et non démontré ; ne pas remplacer RAFT/H2 par cette piste.
- **Compatibilité :** simulation seule pendant le run ; changement du sampler réservé à un futur témoin.
- **Sources :** [LexisLM, parcours des données](https://arxiv.org/html/2611.10209v1#S4), [cycle 2, A3/C1](/D:/ServOMorph/creation_llm/rd/programme_cycle2.md).

**D — Détecter les réponses SFT contradictoires : petit complément à A1/A2**

- **Mécanisme — H :** identifier les prompts équivalents associés à des réponses incompatibles avant de conserver arbitrairement un doublon.
- **Code concerné — L :** [prepare_sft.py](/D:/ServOMorph/creation_llm/prepare_sft.py), `question_key()`, `prepare()`, `load_conversations()` ; `bench_assistant.check_overlap()`. La clé actuelle utilise seulement le premier message utilisateur et conserve le premier exemple.
- **Expérience minimale :** audit CPU de 10 000 conversations, clé normalisée réservée à l’audit ; comparer réponses et historique complet, relire jusqu’à 100 groupes suspects. Ne pas fusionner deux conversations dont la suite diffère.
- **Coût estimé :** CPU **≤1 h**, relecture **1–2 h**, RAM **≤2 Gio**, **0 VRAM**, **0 % de coût par pas** à données figées.
- **Succès :** tous les groupes détectés étiquetés, 0 fuite de famille entre partitions, 0 contradiction non résolue dans les données pilotes retenues. Échec si une normalisation fusionne des consignes distinctes ou si des conflits subsistent.
- **Gain attendu chiffré :** inconnu, **0 directement pour l’audit** ; tout gain après nettoyage doit satisfaire le protocole B.
- **Compatibilité et SFT :** compatible en lecture seule ; peut réduire des cibles incohérentes, sans démontrer une correction des omissions. Complète A2, ne crée pas un nouveau bras prioritaire.
- **Source :** [LexisLM, préparation SFT](https://arxiv.org/html/2611.10209v1#S7).

**Règles communes aux expériences**

Les audits CPU suivent A1–A3 : un processus, un thread, priorité basse ; suspendre si le débit baisse de plus de **1 % sur deux périodes comparables de 15 min**. Les estimations CPU sont des budgets de pilote, pas des mesures réalisées.

Les pilotes GPU attendent la fin du pré-entraînement et passent par [gpu_task.py](/D:/ServOMorph/creation_llm/gpu_task.py), `main()`, avec contrôle du jeton. **`--minutes` réserve le jeton, mais n’arrête pas le calcul** : limite de durée et sauvegarde doivent être gérées par le pilote. Aucun nouveau pilote décrit ici n’a été créé.

**4. Idées inadaptées ou déjà présentes**

| Idée | Décision et raison |
|---|---|
| Copier les modèles grecs ou leurs données comme solution française | Écarté : transfert linguistique et gain sur les omissions non démontrés ; les poids ne s’insèrent pas dans le modèle courant. |
| Copier SwiGLU et AdamW | Écarté pour ce run : modification de `MLP` et de `build_optimizer()`, sans ablation causale ; AdamW ajouterait ≈1,125 Gio d’état sur les blocs locaux. |
| Passer immédiatement à 4 096 tokens | Écarté : activations et logits croissent avec la longueur ; coût d’attention supérieur, marge mémoire insuffisamment caractérisée. Aucun besoin démontré par le benchmark actuel. |
| Porter le budget à 22B | Écarté : ≈40,4 jours au débit actuel, sans preuve de rendement local ; le débit des modèles LexisLM est inconnu. |
| Remplacer le tokenizer pendant le run | Écarté : incompatibilité des IDs avec embeddings et corpus déjà encodés. |
| Ajouter extension chat et masque de réponse comme « nouveautés » | Déjà présents : `convert_chat.build_chat_tokenizer()`, `expand_vocab()`, `prepare_sft.build_example()`, `train_sft.masked_loss_sum()`. |
| Supprimer le packing ou tronquer à gauche | Écarté : le packing local possède déjà des `doc_ids` ; retirer des débuts de consigne peut aggraver précisément l’information manquante. |
| Assimiler bits/octet et compression du tokenizer | À corriger : les tokens/octet sont mesurables sans modèle ; les bits/octet nécessitent des probabilités. I6 ne doit pas promettre une comparaison de modèles par simple audit CPU du tokenizer. |

**5. Articulation avec les décisions locales**

| Document | Apport de cette analyse |
|---|---|
| [Programme cycle 2](/D:/ServOMorph/creation_llm/rd/programme_cycle2.md) | **Complète** A1/A3 par provenance tokenizer, invariance des IDs et audit des conflits ; B s’intègre à B1/B2 ; C reste dans C1. Aucun nouveau chantier prioritaire. |
| [Bilan cycle 1](/D:/ServOMorph/creation_llm/rd/bilan_cycle1.md) | **Confirme** la priorité donnée à l’exploitation des preuves ; LexisLM ne remplace pas l’argument local en faveur de RAFT/H2. |
| [Sélection](/D:/ServOMorph/creation_llm/rd/selection.md) | **Confirme** l’exigence d’ablation et de coût mesuré ; ne réhabilite ni changement d’optimiseur ni changement d’architecture. |
| [Roadmap 369M](/D:/ServOMorph/creation_llm/roadmap_modele_369m.md) | **Complète** les évaluations après SFT ; **rectifie** taille effective, lot exécuté, durée et marge mémoire. Aucun changement des étapes autorisées. |
| Veilles entraînement, inférence, MoE et I6 | Aucun résultat LexisLM ne justifie de rouvrir leurs choix architecturaux ; l’apport est méthodologique, surtout sur les mesures et leur interprétation. |

À SURVEILLER — LexisLM justifie quelques compléments aux audits du cycle 2, mais aucun gain démontré ne motive de modifier le pré-entraînement actuel ni de déplacer la priorité SFT/RAFT.