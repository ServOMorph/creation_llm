**Recommandation : conserver le pré-entraînement actuel, préparer les vérifications sur CPU, puis tester une petite adaptation de STP au SFT.** LLM-JEPA sur paires françaises vient ensuite. La prédiction latente multi-pas et SIGReg restent des expériences exploratoires : aucune preuve consultée ne permet de leur attribuer un gain sur votre assistant.

Le rapport fourni est globalement prudent et ses principales références linguistiques sont authentiques. Ses faiblesses concernent surtout le coût réel d’intégration, la sélection des meilleurs hyperparamètres et la différence entre amélioration des représentations et amélioration des réponses.

Aucun fichier modifié, aucun entraînement ni calcul GPU lancé. J’ai lu les fichiers de code demandés, la roadmap, le bilan assistant, la charte, le bilan du cycle 1, la sélection et consulté les passages pertinents des veilles antérieures. `DOCUMENTATION/INDEX.md` est absent.

**1. Solidité du rapport et corrections**

Je distingue quatre statuts, afin de corriger l’ambiguïté de certaines veilles du dépôt :

- **Établi et reproduit** : résultat publié avec reproduction indépendante du résultat considéré.
- **Publié, reproduction non vérifiée** : publication confirmée, sans garantie de réplication indépendante.
- **Prépublication** : résultat public sans publication finale vérifiée.
- **Hypothèse locale** : transposition, estimation technique ou protocole proposé pour ce dépôt.

**Aucun gain JEPA sur votre configuration n’appartient à la première catégorie.**

| Affirmation du rapport | Verdict de contre-expertise |
|---|---|
| JEPA prédit des représentations plutôt que les observations directement | Correct. Cela ne fournit pas, à lui seul, un générateur, une mémoire ou un système de planification. |
| Chiffres LLM-JEPA : SYNTH, GSM8K, Spider, NQ-Open | Confirmés dans l’arXiv v2. **Omission importante : plusieurs tableaux donnent la meilleure configuration parmi une recherche d’hyperparamètres**, chaque configuration étant exécutée cinq fois. Ce ne sont pas des gains obtenus avec une recette unique choisie d’avance. |
| LLM-JEPA : publication ICLR 2026 | L’accès direct à OpenReview a rencontré une vérification de navigateur. Je conserve les chiffres comme résultats de l’arXiv v2, sans prétendre avoir audité la version finale. |
| STP publié à ICML 2026 | Confirmé par les actes PMLR. |
| STP : « 16× moins de données » | Correct dans le protocole concerné ; les époques augmentent proportionnellement. **Ce résultat ne démontre pas une économie de calcul de 16×.** |
| LLM-JEPA : coût 1,25× avec 75 % d’omission de l’auxiliaire | Confirmé comme modèle de coût annoncé dans le dépôt officiel. Ce n’est ni une mesure RTX 4060, ni une garantie de pic mémoire. |
| Audit indépendant : gains non robustes après correction statistique | Confirmé. Il concerne des constructions et un protocole particuliers ; il ne réfute pas toutes les variantes JEPA. |
| ER-JEPA | Référence et auteurs confirmés ; prépublication d’extension, pas preuve suffisante de reproduction exacte de la recette initiale. |
| « 109M contre 72B », 42× exemples et 14× tokens | Confirmés pour l’alignement intermodal étudié. Ce résultat ne mesure pas la qualité d’un assistant généraliste. |

Sources : [LLM-JEPA v2](https://arxiv.org/html/2509.14252v2), [STP dans les actes ICML](https://proceedings.mlr.press/v306/huang26s.html), [protocole STP](https://arxiv.org/html/2602.22617v1), [dépôt LLM-JEPA](https://github.com/galilai-group/llm-jepa), [audit indépendant](https://arxiv.org/abs/2605.15394), [ER-JEPA](https://arxiv.org/abs/2609.36952), [travail sur l’alignement intermodal](https://arxiv.org/html/2610.39302v1).

Les références visuelles et robotiques citées existent également. En revanche, je **ne recertifie pas ici tous leurs tableaux chiffrés** : notamment les facteurs 8,4×, 48×, les coûts d’entraînement et tous les scores vidéo. Leur validation exhaustive ne changerait pas la décision locale : ils ne permettent pas d’estimer le gain d’entraînement de ce GPT.

Les garanties de LeJEPA restent conditionnelles. Le résultat d’identifiabilité porte sur une classe définie de transitions stationnaires avec bruit additif ; il ne démontre pas que les états d’un décodeur français doivent devenir gaussiens pour mieux répondre. [Article théorique](https://arxiv.org/abs/2605.26379)

**Corrections propres au projet :**

1. **336M paramètres uniques, effectivement.** Le calcul depuis `model.py` donne :
   \[
   24\times12\times1024^2+32768\times1024
   =335\,544\,320.
   \]
   Le nom « 369M » correspond au décompte sans partage des embeddings. Après conversion au vocabulaire chat de 32 832 entrées, le nombre change légèrement.

2. **La durée de 12 jours ne correspond pas à 6 300 tok/s.**
   \[
   113000\times65536=7\,405\,568\,000\ \text{tokens}.
   \]
   À 6 300 tok/s : **326,5 heures, soit 13,6 jours continus**, hors pauses et surcoûts non inclus dans ce débit. Les 12,2 jours du bilan correspondent à environ 7 000 tok/s.

3. **Deux limites mémoire distinctes.** [run_369m.py](/D:/ServOMorph/creation_llm/run_369m.py) réserve 6,5 Gio auprès de l’orchestrateur, mais transmet **6,1 Gio** au plafond PyTorch. La marge expérimentale doit être évaluée sous ce plafond effectif.

4. **Le calcul « états cachés = 4 Mio » est seulement conditionnel.** Il vaut pour \(2\times1024\times1024\) valeurs bf16. Ici, les poids sont créés en fp32 ; l’autocast ne convertit pas tout le modèle. Le flux résiduel peut rester en fp32 : **8 Mio pour ce tenseur**, sans compter gradients et temporaires. Il faut mesurer `h.dtype`.

5. **Le score 49,7 % appartient à la v2.** Un auxiliaire appliqué au 369M devra être comparé au **SFT témoin du même 369M**, pas seulement dépasser la v2.

6. **Un gain de 10/300 n’est pas automatiquement significatif.** Il constitue un seuil d’intérêt pratique. La conclusion dépend des gains et pertes appariés, des graines d’entraînement et du bruit de génération.

Enfin, le passage de 49/90 à 71/90 avec documents relus ne prouve pas que les connaissances étaient déjà mémorisées : **le contexte apporte aussi l’information**. Il établit surtout une capacité de lecture exploitable. C’est la conclusion utile de [rd/bilan_cycle1.md](/D:/ServOMorph/creation_llm/rd/bilan_cycle1.md).

**2. Conditions communes d’implémentation**

Les changements ci-dessous sont des **spécifications proposées**, pas des modifications effectuées.

| Fichier / fonction | Changement nécessaire |
|---|---|
| [model.py — `GPT.forward`](/D:/ServOMorph/creation_llm/model.py:102) | Ajouter un accès optionnel aux états de la dernière couche, en conservant le retour actuel par défaut. Distinguer résidu brut et état après RMSNorm. Ne pas conserver les sorties des 24 couches. |
| `GPT.forward` ou nouvelle méthode interne `GPT.encode` | Permettre une passe d’encodage sans projection vocabulaire pour les vues JEPA. Éviter de calculer des logits inutiles. |
| [train_sft.py — `PackedData.batch`](/D:/ServOMorph/creation_llm/train_sft.py:86) | Fournir les segments de messages/réponses utiles aux auxiliaires. `doc_ids` sépare les exemples, mais ne décrit pas tous les changements de rôle. |
| `masked_loss_sum`, `train`, `evaluate` | Séparer CE, auxiliaire et objectif total. Conserver la validation CE seule pour comparer les modèles. |
| [train.py — `TrainConfig`, `train`, `save_checkpoint`](/D:/ServOMorph/creation_llm/train.py) | Ajouter mode auxiliaire, coefficient, fréquence, graine et journalisation. Sauvegarder les éventuels prédicteurs et leurs états d’optimisation. |
| `MixedSource.batch`, `get_batch` | Pour le pré-entraînement expérimental, reconstruire les frontières depuis le séparateur réel du tokenizer, puis empêcher les couples/triplets de les traverser. |
| [run_369m.py](/D:/ServOMorph/creation_llm/run_369m.py) | Laisser la recette actuelle intacte ; les essais doivent avoir leur propre répertoire de checkpoints et leurs arguments explicites. |

Point sensible : le SFT normalise actuellement la CE par le **nombre total de tokens supervisés du pas accumulé**. Ajouter naïvement une moyenne auxiliaire à chaque micro-lot changerait son poids avec l’empaquetage.

Pour un auxiliaire défini sur des triplets, utiliser par exemple :

\[
L_{\text{pas}}=
\frac{\sum \mathrm{CE}}{N_{\text{tokens supervisés}}}
+\lambda\frac{\sum \ell_{\text{aux}}}{N_{\text{triplets valides}}}.
\]

Préparer les indices et leur dénominateur avant les passes ; un micro-lot sans triplet contribue zéro. Pour les pertes distributionnelles comme SIGReg, cette décomposition ne reproduit pas une statistique calculée sur un grand lot.

Autre piège : `y[t]` supervise le token **suivant**. Le masque `y != -100` ne doit pas être interprété directement comme le rôle du token d’entrée `x[t]`.

**3. Coûts : ce que l’on peut raisonnablement estimer**

Les fourchettes suivantes sont des **hypothèses d’ingénierie**, sans benchmark local. Elles supposent : lot 2, contexte 1 024, recalcul conservé, dernière couche seulement, auxiliaire vectorisé, aucune nouvelle projection vers les 32k tokens.

| Variante proposée | Surplus de pic envisagé | Surcroît de temps par pas | Débit équivalent depuis 6 300 tok/s |
|---|---:|---:|---:|
| STP sur un petit échantillon de triplets | **16–64 Mio** | **1–5 %** | **6 000–6 240 tok/s** |
| Prédiction latente, trois horizons, prédicteurs de rang 64 | **32–96 Mio** | **2–8 %** | **5 830–6 180 tok/s** |
| SIGReg sur un échantillon projeté | **16–64 Mio** | **2–10 %** | **5 730–6 180 tok/s** |
| Paires JEPA sur 25 % des micro-lots, passes organisées séquentiellement | **0,1–0,6 Gio** | **12,5–35 %** | **4 670–5 600 tok/s** |

Ces intervalles servent à **dimensionner un essai**, pas à garantir qu’il passera. Une recompilation, un changement de noyau d’attention ou une rétention involontaire des graphes peut les dépasser.

Le débit SFT de référence n’a pas été mesuré ici. Son attention masquée et sa densité de tokens supervisés diffèrent du pré-entraînement : les débits de la dernière colonne ne sont donc **pas des prédictions de débit SFT**.

Pour calibrer : mesurer d’abord 20 micro-pas de chauffe puis 100 micro-pas utiles, avec le même lot, la même accumulation et les mêmes données. Relever temps synchronisé, mémoire allouée/réservée et consommation GPU globale. L’omission de JEPA sur certains lots réduit le coût moyen, **pas le pic du lot où JEPA est actif**.

**4. Piste prioritaire : STP pendant le SFT**

**Statut : publié, reproduction indépendante exacte non vérifiée ; adaptation locale hypothétique.**

Mécanisme :

\[
L=L_{\mathrm{SFT}}+
\lambda\,\mathbb E_{s<r<t}
\left[1-\cos(h_t-h_r,\ h_r-h_s)\right].
\]

La formule encourage des directions successives compatibles dans l’espace caché. Elle ne fournit aucun fait supplémentaire au modèle. [Article STP](https://arxiv.org/html/2602.22617v1)

**Implémentation précise.**

- Dans `GPT.forward`, exposer l’état final normalisé utilisé par la tête linguistique.
- Dans `PackedData.batch`, identifier les segments continus de réponse.
- Dans `masked_loss_sum`, calculer l’auxiliaire sur, par exemple, **32 triplets au maximum par fenêtre**, au sein d’un même segment ; exclure padding et marqueurs de contrôle.
- Calculer différences et cosinus en fp32 avec protection des normes nulles.
- Dans `train`, conserver la normalisation globale décrite plus haut.
- Ne pas ajouter de paramètres, de cible EMA ou de `detach` absent de cette formulation.

Cette sélection limitée aux réponses est une **variante locale**. Le code officiel gère des segments utilisateur/assistant, des différences de représentations et des pondérations ; reprendre seulement la formule ne constitue pas une reproduction exacte. [Implémentation officielle](https://raw.githubusercontent.com/galilai-group/llm-jepa/main/stp.py)

**Gain attendu chiffré : inconnu.** Seuil d’intérêt proposé : **+10 réussites sur 300**, contre le SFT témoin du même modèle, avec une amélioration appariée robuste. Ce seuil n’est pas une prévision.

**Coût :** première ligne du tableau, aucun paramètre supplémentaire. À 500 pas de 65 536 tokens, le témoin représenterait environ 1,44 h à 6 300 tok/s ; trois graines et deux bras demanderaient environ **8,7–8,9 h**, hors évaluations, si ce débit était atteint. Cela montre pourquoi « 500 pas » n’est pas un essai GPU court sur le 369M.

**Expérience minimale.**

- CPU : vérifier gradients finis, frontières, comportement avec zéro triplet et identité avec le témoin pour \(\lambda=0\).
- Petit modèle : CE seule contre CE+STP, mêmes données et initialisation ; choisir \(\lambda\) sur développement, pas sur les 300 cas finaux.
- Après la base 369M : appliquer la comparaison à une seule étape SFT clairement choisie, puis confirmer le gagnant dans la recette complète en deux étapes.

**Échec :** perte STP réduite sans gain des réponses ; hausse de `critere_absent`, des boucles ou des erreurs de nombres/négations ; dépassement mémoire ; ralentissement supérieur à 5 % sans bénéfice à temps égal.

**Reprise :** compatible structurellement avec les poids et l’optimiseur existants, puisqu’aucun paramètre n’est ajouté. Mais activation = **changement d’objectif** : acceptable dans une branche expérimentale depuis un checkpoint, pas comme continuation transparente du run actuel.

**5. Deuxième piste : LLM-JEPA sur paires françaises fiables**

**Statut : résultats linguistiques publics ; transfert local hypothétique.**

Mécanisme : conserver la CE et apprendre une relation entre deux vues encodées séparément. Les premiers candidats sont des transformations déterministes ou des paraphrases contrôlées. Une question et sa réponse ne sont pas nécessairement deux formulations du même contenu.

**Implémentation précise.**

- Ajouter l’encodage sans logits dans `model.py`.
- Étendre les données SFT pour fournir deux vues et leurs dernières positions valides.
- Encoder les vues avec des `doc_ids` distincts : la vue source ne doit pas lire la cible.
- Pour le premier essai, utiliser la variante sans token `[PRED]`, afin de préserver vocabulaire et checkpoints.
- Dans `train_sft.train`, effectuer le backward CE, libérer son graphe, puis effectuer la passe auxiliaire lorsqu’elle est sélectionnée ; clipper et mettre à jour les poids après toutes les contributions.
- Fixer explicitement la convention de coefficient lorsque seuls 25 % des lots utilisent l’auxiliaire. Ne pas multiplier silencieusement la perte par quatre.

Une réponse encodée dans la concaténation normale question+réponse a déjà lu la question. L’utiliser comme si elle était une vue indépendante modifie l’expérience.

**Gain attendu chiffré : inconnu.** Même seuil pratique de **+10/300**, avec victoire également à temps égal. Les gains publiés ne constituent pas une fourchette prédictive pour le français. [LLM-JEPA](https://arxiv.org/html/2509.14252v2)

**Coût :** 0,1–0,6 Gio et +12,5–35 % de temps dans l’organisation proposée. La plage suppose une passe auxiliaire coûtant environ 0,5–1,4 fois la passe d’entraînement principale, exécutée un quart du temps. C’est une hypothèse à remplacer par mesure ; le dépôt officiel annonce 1,25× dans sa propre recette. [Protocole et coût officiels](https://github.com/galilai-group/llm-jepa)

**Expérience minimale : trois bras.**

- CE seule, exposée aux mêmes textes.
- CE + paires correctes.
- CE + paires permutées, de longueurs comparables.

Le témoin doit bénéficier des mêmes données supplémentaires : autrement, on attribuerait à JEPA un effet provenant des exemples.

**Échec :** paires permutées aussi efficaces ; gain perdu à temps égal ; dégradation des détails exacts ; pic mémoire excessif.

**Reprise :** possible depuis les poids actuels pour une variante sans nouveau paramètre. Avec `[PRED]` ou prédicteur appris, il faut migrer les poids et éventuellement l’optimiseur. `load_state_dict` strict et la restauration actuelle des groupes d’optimiseur ne gèrent pas cette migration automatiquement.

**6. Troisième piste : prédiction latente à plusieurs horizons**

**Statut : hypothèse de la cellule inspirée de JEPA ; aucune validation trouvée pour cette recette.**

Proposition :

\[
L=L_{\mathrm{CE}}+
\frac{\lambda}{|K|}\sum_{k\in K}
\left[1-\cos\left(P_k(h_t),
\operatorname{sg}(h_{t+k})\right)\right],
\qquad K=\{4,16,64\}.
\]

Les horizons sont exprimés en tokens, pas en unités sémantiques. Ils sont des choix expérimentaux.

**Implémentation précise.**

- `GPT.__init__` : ajouter trois prédicteurs \(1024\rightarrow64\rightarrow1024\), sans biais.
- `GPT.forward` : calculer leurs entrées sur les positions échantillonnées uniquement ; cibles détachées issues de la même passe causale.
- `train.build_optimizer` : les noms hors `blocks.*` les enverront actuellement vers AdamW ; rendre ce choix explicite.
- `MixedSource.batch` : fournir les frontières ; n’échantillonner aucun horizon traversant un document.
- `train.train` : journaliser séparément chaque horizon et la CE.

Les trois prédicteurs totalisent **393 216 paramètres**, soit **6 Mio** pour poids, gradients et deux moments AdamW fp32, hors temporaires. D’où l’estimation totale de **32–96 Mio**, +2–8 % de temps.

**Risque central : la copie.** \(h_{t+k}\) contient déjà une représentation du préfixe. Le prédicteur peut réussir en reproduisant ce qui était connu à \(t\), sans apprendre l’information nouvelle. La prédiction de plusieurs futurs n’est donc pas automatiquement de la planification.

**Gain attendu chiffré : inconnu.** Seuil proposé : **au moins −0,01 nat de CE de validation à temps égal**, reproduit sur trois graines, sans dégradation aval.

**Expérience minimale :** témoin CE, auxiliaire, et contrôle \(P(h_t)=h_t\). Comparer aussi l’auxiliaire à horizon nul et mesurer si les futurs réels sont mieux distingués de futurs permutés.

**Échec :** proximité latente améliorée sans progrès CE ; prédiction pas meilleure que la copie ; effondrement des représentations ; gain disparu à temps égal.

**Reprise :** poids du tronc réutilisables ; nouveaux prédicteurs et états d’optimisation à initialiser. **Pas de reprise directe inchangée** avec le chargeur actuel. Aucun nouveau pré-entraînement complet n’est intrinsèquement obligatoire, mais une branche dédiée l’est.

Sources d’inspiration, sans transfert chiffré revendiqué : [I-JEPA](https://arxiv.org/abs/2301.08243), [H-JEPA](https://arxiv.org/abs/2610.06805). Le résultat de navigation de H-JEPA ne valide pas cette proposition linguistique.

**7. Quatrième piste : régulariser les représentations avec SIGReg**

**Statut : méthode source publique ; usage comme auxiliaire de ce GPT hypothétique.**

Mécanisme : ajouter une faible pénalité de distribution sur des représentations échantillonnées, afin de limiter concentration et directions inutilisées. SIGReg emploie des projections et une comparaison à une distribution gaussienne ; ce n’est pas simplement une pénalité de covariance. [LeJEPA](https://arxiv.org/html/2511.08544v1)

**Implémentation proposée.**

- Exposer le résidu final **avant RMSNorm** dans `GPT.forward`.
- Dans une fonction auxiliaire appelée par `train` ou `masked_loss_sum`, sélectionner au maximum 256 positions valides et espacées.
- Employer une projection fixe vers 256 dimensions, puis une petite configuration SIGReg, par exemple 32 directions et 17 points d’intégration.
- Ne pas régulariser simultanément toutes les couches et ne pas ajouter immédiatement une perte d’invariance.

Ces nombres définissent un prototype local, pas la recette validée de LeJEPA. Une adaptation sur tokens corrélés ne doit pas être appelée « LeJEPA reproduit ».

**Pourquoi avant RMSNorm ?** Après RMSNorm, la norme par token est presque fixée. Imposer une gaussienne isotrope complète à ces vecteurs introduirait une tension géométrique évitable.

**Coût :** hypothèse de 16–64 Mio et +2–10 % de temps. Le seul tableau \(256\times32\times17\) fp32 occupe environ 0,53 Mio ; plusieurs intermédiaires, les gradients et l’état caché portent le coût réel au-delà.

**Gain attendu chiffré : inconnu.** Même seuil exploratoire de **−0,01 nat à temps égal**, puis vérification des réponses.

**Expérience minimale :** CE seule contre CE+SIGReg, avec suivi de variance, rang effectif et CE ; tester un petit coefficient choisi sur développement.

**Échec :** statistiques de représentation améliorées sans bénéfice CE/assistant ; dégradation des faits rares ; instabilité selon la composition du micro-lot.

L’accumulation de 32 micro-lots **ne transforme pas** des statistiques calculées séparément en statistiques sur 64 séquences. Les positions d’un même document restent corrélées. Ce problème réduit fortement la priorité de cette piste.

**Reprise :** compatible si la projection est fixe et sans paramètres entraînables ; toujours un changement expérimental d’objectif. Une version avec projecteur appris demande la migration de l’optimiseur.

**8. Pistes à écarter pour le run actuel**

| Piste | Motif |
|---|---|
| Remplacer la CE par JEPA | Aucun objectif n’imposerait alors correctement les sorties tokenisées attendues. L’ablation sans perte linguistique de LLM-JEPA produit des sorties vides dans son protocole. [Source](https://arxiv.org/html/2509.14252v2) |
| Ajouter un enseignant EMA complet sur GPU | Poids seuls : **0,625 Gio en bf16 ou 1,25 Gio en fp32**, plus passe cible et activations. Incompatible avec la marge prudente actuelle, sans refonte mémoire. |
| Décharger cet enseignant sur CPU à chaque micro-lot | Faisable conceptuellement, mais coût de transfert ou d’inférence CPU inconnu ; aucune justification de priorité face aux variantes sans enseignant. |
| Transposer directement I-JEPA/V-JEPA/VL-JEPA | Modalités, vues, cibles et tâches différentes. Les gains visuels ne chiffrent pas une amélioration de génération française. [V-JEPA 2](https://arxiv.org/abs/2506.09985), [VL-JEPA](https://arxiv.org/abs/2512.10942) |
| Ajouter immédiatement une planification hiérarchique | Le corpus ne fournit pas les transitions action–état et le critère de planification des expériences robotiques. [H-JEPA](https://arxiv.org/abs/2610.06805) |
| Produire des paraphrases de milliards de tokens | Coût de génération et qualité des vues non établis ; risque d’altérer noms, nombres et négations. Commencer par un petit corpus contrôlé. |
| Entraîner immédiatement un nouvel encodeur de recherche | Cela améliore potentiellement la récupération, pas directement l’entraînement du GPT. Le cycle 1 dispose déjà d’un témoin hybride ; tout nouvel encodeur devrait gagner aussi sur les réponses finales. |
| Changer simultanément auxiliaire, refroidissement, EMA et données | Attribution causale impossible. Les expériences différées du cycle 1 doivent garder leurs témoins propres. |

**9. Programme minimal et classement final**

Pendant le run actuel, le travail utile sur CPU est limité et concret :

- Petit GPT **4 couches × 128**, contexte 128, vocabulaire actuel : environ **4,98M paramètres** avec embeddings liés.
- Vérifications de masques, causalité, gradients et reprise ; 50–100 pas suffisent à détecter des erreurs d’implémentation, pas à mesurer la qualité.
- Petit jeu de transformations françaises comportant nombres, négations et contraintes explicites, séparé par familles entre entraînement et évaluation.
- Budget proposé : **30 minutes CPU maximum par vérification**, peu de threads pour préserver l’alimentation du run GPU. Durée réelle inconnue.

Après le run, premier passage GPU sur un témoin **8 × 256**, puis seulement sur le 369M si les contrôles passent. Réserver initialement **30 minutes de faisabilité**, sans présenter ce rodage comme une preuve de gain.

Le lancement futur doit utiliser [gpu_task.py](/D:/ServOMorph/creation_llm/gpu_task.py), avec vérification du détenteur du jeton conformément au bilan du cycle 1. Le lanceur possède désormais un contrôle avant exécution. Aucun essai concurrent au 369M n’est recommandé.

| Rang | Essai | Gain local quantifiable aujourd’hui | Coût / intérêt | Quand |
|---:|---|---|---|---|
| **1** | STP sur réponses SFT | Inconnu ; seuil +10/300 | Faible surcoût, aucune nouvelle donnée ni nouveau paramètre | Vérifications CPU maintenant ; SFT après la base |
| **2** | LLM-JEPA sur paires contrôlées | Inconnu ; seuil +10/300 à temps égal | Plus coûteux, mais relation entre vues directement testable | Après STP, ou si un corpus de paires fiable existe |
| **3** | Prédiction latente multi-horizons | Inconnu ; seuil −0,01 nat | Faible ajout de paramètres, mécanisme de copie à exclure | Petit témoin après le run |
| **4** | SIGReg auxiliaire | Inconnu ; seuil −0,01 nat | Coût modéré, justification linguistique et estimation de distribution fragiles | Seulement après diagnostic des représentations |

Ce classement porte sur **l’intérêt d’expérimenter par unité de coût**, pas sur un rapport numérique de gains attendus que les preuves disponibles ne permettent pas de calculer.

Pour sélectionner un résultat : mêmes checkpoints de départ, mêmes données, mêmes graines ; comparaison à tokens égaux puis à temps égal ; hyperparamètres choisis sur développement ; évaluation finale appariée par question. Les graines de génération ne remplacent pas plusieurs graines d’entraînement. Une absence de différence concluante reste une absence de preuve, pas la preuve d’un effet nul.

La [roadmap](/D:/ServOMorph/creation_llm/roadmap_modele_369m.md) reste donc cohérente : terminer la base, évaluer, établir le SFT témoin, puis ajouter l’auxiliaire. Le [bilan du cycle 1](/D:/ServOMorph/creation_llm/rd/bilan_cycle1.md) justifie parallèlement de conserver le SFT avec distracteurs parmi les priorités : il vise plus directement le défaut observé que la seule géométrie latente. **STP mérite le premier essai JEPA ; il ne justifie pas de détourner maintenant le budget du pré-entraînement.**