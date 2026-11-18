# Cycle R&D 2 — améliorer la complétude, puis choisir une innovation frugale

**Résumé — 8 lignes**  
1. Terminer le pré-entraînement actuel sans modifier ses données, son objectif ni son ordonnanceur.  
2. Pendant ce run, limiter le travail CPU à l’audit, aux petits corpus contrôlés et aux vérifications fonctionnelles.  
3. Après la base, tester CCE comme optimisation matérielle, sans lui attribuer de gain sémantique.  
4. Établir le témoin obligatoire : base 369M + SFT standard en deux étapes, conformément au lanceur existant.  
5. Prioriser RAFT/H2 pour la sélection des preuves et la complétude ; tester STP séparément ensuite.  
6. Garder LLM-JEPA conditionnel à des paires fiables et à une marge mémoire mesurée.  
7. Isoler refroidissement et EMA dans une expérience de continuation ; présélectionner curriculum/H1 et ResFormer pour le futur modèle.  
8. Ne retenir qu’une innovation pour le prochain pré-entraînement, ou aucune si les résultats restent inconclusifs.

## Arbitrage et règles de preuve

Les deux analyses, le bilan du cycle 1, la sélection et la roadmap ont été lus, ainsi que les cinq fichiers de code demandés et les préparateurs pertinents. `DOCUMENTATION/INDEX.md` est absent. **Aucun fichier modifié, aucun entraînement lancé.** Les modifications et commandes nouvelles ci-dessous sont des spécifications futures.

**Statuts :** **E** = résultat publié et reproduit indépendamment ; **P** = publication vérifiée, reproduction indépendante non vérifiée ; **PP** = prépublication consultée ; **H** = hypothèse locale. Aucun gain proposé ici n’est **E sur cette configuration**. Les résultats du cycle 1 sont des mesures locales documentées.

**Gain attendu chiffré : inconnu pour toutes les innovations.** Les chiffres du programme sont des **seuils de sélection proposés**, jamais des prévisions. Tous les coûts nouveaux sont des **estimations**, à remplacer par des mesures.

| Pistes fusionnées / divergence | Décision de direction et justification |
|---|---|
| STP premier dans l’analyse JEPA ; audit/CCE/RAFT premiers dans l’analyse frugalité | Pas de contradiction scientifique : les classements portent sur des périmètres différents. **Audit → CCE → témoin → RAFT/H2 → STP.** STP reste le premier auxiliaire géométrique testé. |
| RAFT, SFT avec distracteurs, données synthétiques ciblées, H2 | Une seule famille expérimentale, avec ablations successives. Le cycle 1 montre **71/90 avec passage relu contre 49/90**, mais seulement **54/90 avec récupération hybride à trois passages** : la robustesse aux documents imparfaits est prioritaire. |
| « Critère absent » assimilé à connaissance manquante | Correction : `bench_assistant.score()` constate l’absence d’un motif attendu. Cela peut aussi être une paraphrase correcte, une omission ou une mauvaise sélection. Audit sémantique avant diagnostic causal. |
| Curriculum rejeté dans `selection.md`, réhabilité par l’analyse frugalité | Réouvrir **sur futur témoin**, grâce au résultat publié plus récent ; ne pas appliquer rétroactivement un curriculum initial au run actuel. Les économies publiées de pas ne prédisent pas celles du projet. [Curriculum, P](https://aclanthology.org/2026.eacl-long.271/) |
| Couverture, déduplication, répétition et H1 | Audit commun ; **couverture et curriculum restent deux facteurs séparés**. Ne pas supprimer indistinctement les répétitions intentionnelles des petites sources. |
| Refroidissement, schedule de continuation, EMA | Un seul essai à deux trajectoires et quatre sorties évaluées. **EMA de poids CPU ≠ enseignant EMA JEPA** ; le second ajoute une passe cible et n’est pas retenu. |
| ResFormer, amorçage formel, H3 | ResFormer seul d’abord. H3 devient admissible seulement après comparaison de ses ingrédients ; son amorçage doit être compté dans le temps total. |
| LLM-JEPA, prédiction latente multi-horizons, SIGReg | Trois mécanismes distincts, pas trois noms d’un même essai. LLM-JEPA reste conditionnel ; multi-horizons et SIGReg sortent du budget principal. |
| Gain géométrique assimilé à amélioration de l’assistant | Rejeté : exiger une amélioration des sorties. Un audit indépendant rapporte des résultats non robustes après correction des comparaisons, sans réfuter toutes les variantes. [Audit, PP](https://arxiv.org/abs/2605.15394) |
| Marge « environ 1 Go » ; durée « 12 jours » | Le code impose **6,1 Gio PyTorch**, avec réservation de 6,5. Le pic historique de 6,04 laisse environ **61 Mio sous ce plafond**, sans garantir le pic SFT. Les 7,406 milliards de tokens demandent **13,60 jours à 6 300 tok/s**, hors interruptions. |

Le « 369M » contient **335 544 320 paramètres uniques** avant conversion chat, d’après les embeddings liés et les RMSNorm sans paramètres. Son nom reste conservé pour identifier les artefacts.

## Tableau du programme

Les coûts CPU désignent du temps machine, sauf mention de relecture humaine. `T₁` et `T₂` désignent les durées mesurées des deux étapes SFT standard pour une graine ; `q₅₄` le débit mesuré du témoin 12 × 512.

| ID / étape | Essai minimal et mécanisme | Coût estimé CPU / GPU / mémoire | Seuil de succès ; échec ou arrêt | Dépendances |
|---|---|---|---|---|
| A1 / a | Audit sémantique des omissions, partitions et évaluation figée | CPU ≤1 h ; relecture humaine 4–8 h ; GPU 0 ; RAM ≤2 Gio | 100 % des cas audités étiquetés, 0 fuite de famille ; sinon correction avant entraînement | Résultats v2 et cycle 1 |
| A2 / a | 200 familles RAFT/H2 et paires contrôlées ; pilote de 2 000 exemples | CPU 1–4 h fractionnées ; relecture 4–8 h ; GPU 0 ; RAM ≤4 Gio | 0 contradiction détectée dans les 200 familles relues ; 0 exemple >1 024 tokens ; sinon corriger | A1 |
| A3 / a | Simulation couverture/curriculum ; contrôles masques, gradients, reprise et schedule | CPU 2–6 h fractionnées, ≤30 min par contrôle ; GPU 0 ; RAM ≤4 Gio | 0 frontière interdite franchie ; reprise identique ; quotas ±0,5 point sur simulation | A1 ; index échantillonné |
| B0 / b | CCE contre CE dense : éviter la matrice globale de logits | CPU 1–2 h ; GPU 30–45 min ; économie nette inconnue | ≥128 Mio économisés, temps/pas ≤1,05×, critères numériques ci-dessous ; sinon conserver CE dense | Base terminée ; A3 |
| B1 / b | Témoin SFT standard en deux étapes | GPU `T₁+T₂` par graine, plus évaluations ; plafond 6,1 Gio | Gates de la roadmap ; sinon diagnostiquer le témoin avant innovations | B0 décidé ; base évaluée |
| B2 / b | Quatre bras : données ordinaires, RAFT, contrefactuels, H2 pondéré | CPU préparation A2 ; GPU ≈4 budgets d’étape 2 par graine, débit à mesurer ; 0 nouveau paramètre | ≥+5 points de complétude ciblée ; garde-fous communs ; sinon rejeter le bras | B1, A2 |
| B3 / b | STP : CE + alignement des directions entre états de réponse | CPU ≤2 h ; GPU ≈`T₂×1,01–1,05` par bras STP ; surplus 16–64 Mio | ≥+10/300 au benchmark ; surcoût ≤5 % ou victoire à temps égal ; sinon rejeter | B1, A3 |
| B4 / b, conditionnel | LLM-JEPA : CE + relation entre vues indépendantes ; contrôle paires permutées | CPU 2–4 h ; GPU +12,5–35 % par bras auxiliaire ; surplus 0,1–0,6 Gio | ≥+10/300 contre CE et avantage sur paires permutées, à temps égal ; sinon arrêt | Paires A2 fiables ; marge B0/B1 |
| B5 / b | Deux continuations de 2 000 pas : LR constant / refroidi ; EMA CPU dans les deux | CPU EMA concomitant ; RAM +1,25 Gio ; GPU ≈11,56 h hors EMA/évaluations ; pas de copie EMA GPU persistante | Gain ≥0,005 nat avec intervalle apparié favorable ; sinon non retenu | Checkpoint final intact ; A3 |
| C1 / c | Plan factoriel couverture × curriculum initial, témoin 54,5M | CPU index 2–8 h estimées ; GPU `4N/q₅₄` par graine ; 0 état GPU ajouté | ≥10 % de temps total économisé à qualité commune ; H1 doit battre le meilleur ingrédient | A3 ; B terminé |
| C2 / c | Témoin standard contre ResFormer : réinjecter les valeurs initiales | CPU contrôles 1–3 h ; GPU `2N/q₅₄` corrigé du surcoût ; pic inconnu à mesurer | ≥0,01 nat gagnée à temps égal, confirmation trois graines ; sinon rejet | A3 ; B terminé |
| C3 / c, conditionnel | Formel seul / ResFormer seul / H3 / standard, puis contrôle aléatoire | CPU génération 1–4 h ; GPU ≥`4N/q₅₄`, amorçage inclus ; pic à mesurer | ≥10 % de temps économisé et avantage confirmé sur le meilleur ingrédient ; sinon H3 rejeté | C2 ; budget optionnel |

## Étape a — pendant le pré-entraînement : CPU seulement

**Protection du run.** Un seul processus auxiliaire, un thread de calcul par défaut, priorité basse, lectures par petits blocs ; pas de scan intégral ni d’encodage massif. Commencer par un échantillon plafonné à 100 000 fenêtres. Comparer les médianes du débit journalisé sur des périodes comparables de 15 minutes, en excluant sauvegardes/évaluations. Une baisse persistante **>1 % sur deux périodes** entraîne l’arrêt du travail auxiliaire. Ce contrôle ne prouve pas une absence absolue d’effet ; si le bruit empêche de trancher, différer le calcul lourd.

**A1 — créer `rd/experiences/c2_audit.py` : `classify_failures()`, `check_family_splits()`, `paired_report()`.** Lire les sorties existantes ; distinguer motif trop strict, omission réelle, erreur factuelle, distraction et cas indéterminé. Conserver le score historique intact et ajouter une annotation sémantique indépendante. Ne pas transformer les réponses du benchmark en données d’entraînement. Source locale : [bilan du cycle 1](/D:/ServOMorph/creation_llm/rd/bilan_cycle1.md) et [scoring](/D:/ServOMorph/creation_llm/bench_assistant.py) ; cette méthode d’audit est **H**, sans gain modèle revendiqué.

**A2 — créer `rd/experiences/c2_data.py` : `make_families()`, `split_by_family()`, `validate_evidence()`.** Chaque famille comporte preuve correcte, distracteurs, preuve retirée et fait fictif modifié ; la réponse suit la preuve. Les paires JEPA préservent explicitement nombres, négations et relations. Séparer train/développement/test **avant augmentation**, puis produire 300 familles de test indépendantes, distinctes du pilote d’entraînement. Le mécanisme de distracteurs vient de [RAFT, PP consultée](https://arxiv.org/abs/2403.10131) ; les variantes contrefactuelles et pondérations H2 sont **H**.

**A3 — créer `rd/experiences/c2_checks.py` : `check_masks()`, `check_resume()`, `check_schedule()`, `simulate_sampling()`.** Petit GPT 4 × 128, contexte 128 ; 50–100 pas maximum par contrôle, aucune prétention de mesure qualitative. Vérifier causalité, segments de réponse, gradients finis, cas sans cible/triplet et identité du chemin témoin lorsque l’auxiliaire est nul. Pour CCE, le CPU prépare la référence dense ; il ne valide pas les kernels GPU. Pour H1, simuler les permutations à segmentation constante. Fondement du curriculum : [publication EACL, P](https://aclanthology.org/2026.eacl-long.271/) ; simulation locale **H**.

## Étape b — après la base : construire puis améliorer l’assistant

### Protocole commun

- **Témoin scientifique :** checkpoint final du 369M, conversion chat, `prepared` pendant 2 epochs à `3e-4`, puis `prepared_v2` pendant 1 epoch à `1e-4`. Aucun refroidissement, aucune nouvelle donnée ni perte auxiliaire dans ce témoin.
- **Base :** appliquer les gates de la roadmap : validation <2,5445 ; FrenchBench grammaire ≥73,9 et vocabulaire ≥68,9. Un échec déclenche une analyse, pas une réécriture opportuniste du seuil.
- **Assistant standard :** global >49,7 %, aucune catégorie sous la v2, baisse FrenchBench ≤8 points face à sa base. Ces repères historiques ne remplacent pas les comparaisons contre B1.
- **Sélection :** réglages sur développement, un seul réglage final par famille ; pilote à une graine, confirmation des finalistes à **trois graines d’entraînement**. Les trois graines de génération habituelles sont conservées, mais ne comptent pas comme trois entraînements.
- **Preuve de gain :** seuil pratique atteint et intervalle apparié à 95 % excluant zéro ; rééchantillonnage par question/famille, sans traiter les variantes ou graines de génération comme indépendantes. Corriger les comparaisons finales multiples ; publier aussi la dispersion entre graines. Avec trois graines, une conclusion peut rester inconclusive.
- **Garde-fous proposés :** non-infériorité à −2 points sur le score général à livre fermé ; aucune catégorie perdant >3 points ; FrenchBench ≤2 points sous B1 ; hausse des boucles, recopies ou abstentions injustifiées ≤1 point. Si les intervalles ne permettent pas de conclure, ne pas déclarer la variante validée.
- **Coût :** mesurer 20 micro-pas de chauffe puis 100 utiles, mémoire allouée/réservée et usage GPU global. Mesurer séparément chaque mode et le débit SFT ; ne pas lui substituer 6 300 tok/s.

### B0 — CCE : prérequis matériel, pas amélioration sémantique

Créer `model.GPT.encode()` pour retourner l’état final normalisé sans logits et `GPT.loss()` pour choisir CE dense ou CCE. Préserver le contrat de `GPT.forward()` pour la génération. Modifier `train_sft.masked_loss_sum()` et, pour les essais de pré-entraînement, `train.train()`/`evaluate()`.

L’API expose bien `softcap`, `shift`, `ignore_index` et les réductions : première version avec **softcap 30, shift 0, ignore_index −100 et filtrage désactivé**. Le SFT conserve une somme divisée par le total des cibles du pas accumulé. Vérifier les deux contributions au poids partagé, entrée et sortie. [API officielle](https://raw.githubusercontent.com/apple-aiml-research/ml-cross-entropy/main/cut_cross_entropy/linear_cross_entropy.py)

**Critères numériques proposés :** en référence FP32, erreur relative de perte ≤`1e-5` et erreur relative L2 des gradients ≤`1e-4` ; en BF16, respectivement ≤`5e-3` et ≤`2e-2`, avec gradients finis. Ces tolérances sont **H**, à annoncer avant comparaison, sans les assouplir après échec.

La compatibilité Windows/4060/compile reste **non vérifiée**. Une économie de 128 Mio valide l’intérêt mémoire minimal ; elle ne garantit pas le lot 4, dont le pic historique nécessiterait environ **236 Mio d’économie, plus une marge**. Si CCE passe, fixer le même backend pour tous les bras comparés et conserver une courte référence dense.

### B1 — conserver le SFT préparé

`run_sft_369m.build_args()` transmet déjà les paramètres des deux étapes et accepte des arguments supplémentaires. `train_sft.train()` charge actuellement le checkpoint directement sur le périphérique : prévoir un chargement initial CPU pour réduire le risque de pic transitoire, puis vérifier l’identité des poids chargés.

Le lanceur impose `--resume` : **chaque bras et chaque graine doivent avoir un répertoire distinct**, vide au premier lancement. L’étape 2 doit pointer explicitement sur l’étape 1 correspondante. Un même checkpoint d’étape 1 peut servir aux branches appariées d’une graine.

### B2 — RAFT/H2 : essai prioritaire de qualité

Ne modifier que l’étape 2 dans le pilote. Comparer quatre bras recevant le même budget de tokens traités : données supplémentaires ordinaires ; RAFT simple ; RAFT avec contrefactuels ; H2 avec poids **2 sur les unités indispensables, 1 ailleurs**. Le témoin standard B1 reste séparé. Le contrôle « données ordinaires » empêche d’attribuer au mécanisme un simple effet d’exposition supplémentaire.

Modifier `prepare_sft.prepare()` pour la partition par familles, `build_example()`/`write_split()`/`load_split()` pour les poids ; étendre `PackedData.batch()` et `masked_loss_sum()`. Normaliser la perte par **la somme globale des poids du pas**, pas par micro-lot. Les poids d’un micro-lot 2 × 1 024 occupent seulement 8 Kio FP32 ; les passages augmentent surtout le coût par réponse supervisée.

Le masque `doc_ids` doit isoler les conversations, **sans isoler leurs passages de la réponse**. Préserver les abstentions légitimes : `--drop-refusals` ne convient pas automatiquement aux exemples sans preuve. Ne pas reproduire de longues chaînes de raisonnement dans le contexte 1 024.

Succès ciblé : **≥+5 points de complétude**, amélioration également sur les variantes avec distracteurs, garde-fous communs respectés. Pour retenir la pondération H2, demander en plus **≥+2 points face aux contrefactuels non pondérés** ; sinon retenir le bras plus simple. Confirmer le gagnant sur la recette complète en deux étapes. [Fondement RAFT](https://arxiv.org/abs/2403.10131) ; H2 et ses seuils : **H**.

### B3 — STP : premier auxiliaire géométrique

**P** pour STP publié ; adaptation SFT locale **H**. Ajouter `stp_loss()` dans `rd/experiences/c2_losses.py`, l’accès aux états dans `GPT.encode()`, et les identifiants de segments de réponse dans `PackedData.batch()`. Échantillonner au plus 32 triplets ordonnés par fenêtre, dans un même segment ; minimiser `1−cos(h_t−h_r, h_r−h_s)` sur l’état final normalisé.

Les cibles sont déjà décalées : `y[t] != -100` n’identifie pas directement le rôle de `x[t]`. Calculer les cosinus en FP32 ; aucun enseignant ni nouveau paramètre. Dans `train_sft.train()`, normaliser l’auxiliaire par le **nombre total de triplets du pas**, indépendamment du dénominateur CE ; `evaluate()` garde la CE seule.

Comparer B1 à STP sur les mêmes données ; tester d’abord l’étape 2. Succès : **+10/300**, preuve appariée et garde-fous communs ; échec si seule la perte géométrique baisse. Le résultat publié « 16× moins de données » n’est pas une promesse de calcul divisé par 16. [STP, ICML 2026](https://proceedings.mlr.press/v306/huang26s.html)

### B4 — LLM-JEPA : option conditionnelle

Source utilisée : **PP arXiv v2** ; publication finale non recertifiée ici. Variante locale sans nouveau token ni prédicteur : encoder séparément deux vues contrôlées via `GPT.encode()`, puis ajouter leur objectif latent en conservant la CE. Ne pas traiter automatiquement question et réponse comme deux paraphrases. [LLM-JEPA](https://arxiv.org/html/2509.14252v2)

Créer `paired_loss()` et un chargeur de paires ; modifier `train_sft.train()` pour terminer le backward CE avant la passe auxiliaire, puis effectuer **un seul** clipping et pas optimiseur. Commencer sur 25 % des micro-lots avec coefficient explicitement défini ; cette fréquence réduit le coût moyen, pas le pic actif.

Trois bras : CE exposée aux mêmes textes, paires correctes, paires permutées à longueurs comparables. Exiger **+10/300 contre CE**, avantage apparié sur le contrôle permuté et victoire à temps égal. Échec si le gain vient seulement des données ajoutées, disparaît à temps égal ou altère nombres/négations. Ne lancer qu’après mesure démontrant le respect des 6,1 Gio.

### B5 — refroidissement et EMA : branche indépendante

Depuis le même checkpoint final **avec états optimiseur**, deux bras de 2 000 pas et mêmes tirages : A à `3e-4` constant ; B décroissant de `3e-4` à `3e-5`. Ajouter `continuation_lr()` dans `train.py` et un compteur local : augmenter simplement `max_steps` modifierait le cosinus existant en fonction du pas absolu.

Créer `update_ema_cpu()` ; proposition **H** : mise à jour tous les 10 pas, coefficient `β=0,99` par mise à jour, initialisée au checkpoint de départ. Sauvegarder EMA, compteur, schedule et états dans `save_checkpoint()`. Le coût des transferts reste à mesurer ; aucun second modèle GPU persistant.

Comparer les sorties A, EMA(A), B et EMA(B), d’abord sur développement puis sur blocs de validation indépendants. Le gain de refroidissement se mesure **B contre A** ; celui d’EMA, contre les poids instantanés du même bras. **0,005 nat est un seuil pratique, pas une estimation du bruit issue de deux checkpoints.** Source de continuation : [étude du refroidissement, PP consultée](https://arxiv.org/abs/2501.18965) ; recette EMA exacte : **H**.

Une amélioration de CE ne suffit pas à remplacer la base de l’assistant : refaire le SFT standard sur la base gagnante et comparer à B1, avec son coût supplémentaire `T₁+T₂`. Ne pas fusionner silencieusement cette branche avec RAFT ou STP.

### Commandes et orchestration

Commandes existantes, **à exécuter seulement après la fin du pré-entraînement et la conversion chat** :

```powershell
.venv/Scripts/python gpu_task.py --gb 6.5 --minutes 1440 -- .venv/Scripts/python run_sft_369m.py --etape 1
.venv/Scripts/python gpu_task.py --gb 6.5 --minutes 1440 -- .venv/Scripts/python run_sft_369m.py --etape 2
```

Créer un pilote unique `rd/experiences/c2_run.py`, fonctions `main()`, `run_trial()`, `profile()`, `evaluate_trial()`. Interface **proposée, non disponible aujourd’hui** :

```powershell
.venv/Scripts/python gpu_task.py --gb 6.5 --minutes 45 -- .venv/Scripts/python -m rd.experiences.c2_run --essai B0 --phase profil
.venv/Scripts/python gpu_task.py --gb 6.5 --minutes 1440 -- .venv/Scripts/python -m rd.experiences.c2_run --essai B2 --phase pilote --seed 1234
```

Même commande pour **B3, B4, B5, C1, C2 ou C3**, en remplaçant l’identifiant ; chaque protocole fixe ses bras, budgets et répertoires. Le profil s’arrête à 20+100 micro-pas ; les pilotes SFT à 100 pas au maximum ; B5 à 2 000 pas par bras ; les essais C au budget de tokens déclaré.

`gpu_task.main()` vérifie le détenteur après acquisition et l’absence du 369M. **`--minutes` réserve le jeton ; il ne borne pas l’expérience.** Le pilote doit gérer sauvegarde/reprise et `TrainingPaused` comme le lanceur SFT. Éviter `--pause-369m` dans ce programme, puisque les essais GPU attendent sa fin. Le détecteur `pids_369m()` cherche aussi la chaîne `checkpoints_369m` dans les lignes de commande : employer des chemins de travail expérimentaux distincts et tester cette détection avant lancement.

## Étape c — choisir une seule innovation pour le prochain pré-entraînement

**Témoin commun :** 12 couches × 512, embeddings liés, vocabulaire 32 768, soit **54,53M paramètres**, même mélange et contexte. Commencer à **100M tokens par bras, une graine** pour éliminer les échecs ; confirmer à **300M, trois graines**, puis comparer le seul finaliste au standard à **1,2B tokens, trois graines**. Ce dernier horizon approche le ratio tokens/paramètres du run actuel.

À titre d’**estimation conditionnelle**, si `q₅₄=25–40k tok/s`, un bras de 300M coûte 2,1–3,3 h ; six bras de 1,2B coûtent **50–80 h**, hors évaluations. Ce débit n’a pas été mesuré. Le budget long exige donc une décision distincte après les pilotes.

**C1 — curriculum/H1.** Créer `rd/experiences/c2_sampling.py`, fonctions `build_window_index()`, `make_curriculum()`, `sampler_state_dict()`. Étendre `train.MixedSource.batch()` et `save_checkpoint()` pour permutations, compteurs et reprise. Index brut indicatif : environ **78 Mo** pour offsets 64 bits et scores 32 bits sur 6,65B/1 024 fenêtres, hors métadonnées.

Comparer tirage avec remise, couverture seule, curriculum seul et combinaison ; fixer le retour au mélange ordinaire après les premiers **20 % du budget**, choix **H**. Conserver les quotas en tokens et la même segmentation dans les quatre bras. Ne pas ajouter simultanément alignement documentaire ou masque inter-documents : ce serait un facteur supplémentaire. Succès : **≥10 % de temps total économisé**, préparation comprise, vers une CE cible commune ; aucune source dégradée de >0,01 nat. H1 doit dépasser son meilleur ingrédient avec intervalle favorable. [Curriculum, P](https://aclanthology.org/2026.eacl-long.271/)

**C2 — ResFormer.** Modifier `CausalSelfAttention.forward()`, `Block.forward()` et `GPT.forward()` pour transmettre les valeurs de la première couche ; intégrer ces valeurs explicitement aux appels de recalcul. Vérifier cache KV, génération incrémentale et gradients. Ne pas confondre ResFormer avec le partage complet des valeurs de SVFormer. [Value Residual Learning, PP consultée](https://arxiv.org/abs/2410.17897)

À lot 2, contexte 1 024 et largeur 512, le tenseur de valeurs seul représente **2 Mio BF16 ou 4 Mio FP32** ; le surplus total du graphe est inconnu. Profil GPU obligatoire, plafond 6,1 Gio. Succès : **−0,01 nat à temps égal**, confirmation entre graines et absence de dégradation aval ; les économies publiées ne sont pas transposées au projet.

**C3 — amorçage formel/H3, option.** Créer `rd/experiences/c2_formal.py` : `generate_formal()`, `generate_random_control()`, `transfer_attention()`. Appeler le transfert avant `train.build_optimizer()` ; copier explicitement les modules d’attention, réinitialiser embeddings/MLP et états optimiseur. Le transfert d’attention seul est une variante **H**, pas une reproduction du papier sur les langages formels. [Amorçage formel, ACL 2025, P](https://aclanthology.org/2025.acl-long.478/)

Pilote : **5M tokens d’amorçage inclus dans 100M totaux**, puis français. Comparer standard, amorçage seul, ResFormer seul et combinaison ; si l’amorçage est positif, ajouter le contrôle aléatoire à fréquences comparables. Exiger **≥10 % de temps total économisé** et un avantage confirmé sur le meilleur ingrédient ; rejeter H3 si l’amorçage n’est pas amorti ou si le progrès grammatical s’accompagne d’une régression factuelle.

**Choix final :** sélectionner le candidat qui atteint une qualité commune le plus vite, préparation et amorçage compris, sous le plafond mémoire. À résultats indiscernables, retenir le plus simple. H1 est la préférence initiale pour son coût matériel ; ResFormer peut le remplacer ; H3 n’obtient aucun privilège d’originalité.

## Pistes sorties du budget principal

- **Multi-horizons latents et SIGReg : réserve scientifique.** La première risque d’apprendre la copie du préfixe ; la seconde estime une distribution sur des tokens corrélés. Gains locaux inconnus ; aucune dépense GPU autorisée par ce programme. Leurs coûts précédemment annoncés restent des hypothèses, non des mesures. [Inspiration I-JEPA](https://arxiv.org/abs/2301.08243), [LeJEPA](https://arxiv.org/html/2511.08544v1).
- **APOLLO-Mini, Rho-1, BPT, NCA : cycle ultérieur.** Respectivement changement d’optimiseur, scoring de référence, adaptation structurelle et amorçage alternatif ; ils introduiraient des facteurs supplémentaires avant résolution des essais principaux. Coût local et gain non établis. [APOLLO](https://arxiv.org/abs/2412.05270), [Rho-1](https://arxiv.org/html/2404.07965v4), [BPT](https://arxiv.org/html/2610.37934v1), [NCA](https://arxiv.org/abs/2603.10055).
- **COAT, CGLS, RecursiveGPT, Olmo Hybrid, TST : chantiers distincts reportés.** Aucun budget ni promesse de gain local. **MIR, SOAP/FOAM et Soft-MTP** restent non prioritaires selon l’analyse frugalité ; leurs bénéfices ne sont pas démontrés face au coût et au régime local.
- **CFG, TTT-NN et nouvelle campagne de récupération seule : clos pour ce cycle.** Réutiliser les résultats du cycle 1 ; réévaluer la récupération uniquement avec le nouvel assistant RAFT. DPO reste conditionnelle aux défauts résiduels de celui-ci.

## Ordre d’exécution et décisions utilisateur

**Ordre ferme :** A1 → A2/A3, séquentiels sur CPU → fin de base et évaluation → B0 → B1 → B2 → B3 → B4 si admissible → B5 → C1/C2 → C3 si autorisé → confirmation longue du seul finaliste. Les échecs techniques bloquent le bras concerné ; les résultats inconclusifs ne déclenchent pas automatiquement davantage de calcul.

Les décisions à prendre par l’utilisateur portent sur :

1. **Le budget GPU par tranche**, après mesure de `T₁`, `T₂` et `q₅₄` ; inclure les évaluations et les répétitions, pas seulement l’entraînement.
2. **Les options B4 et C3**, puis la confirmation longue estimée à 50–80 h sous l’hypothèse de débit donnée.
3. **Les tolérances de régression proposées** et la priorité entre assistant à livre fermé et assistant utilisant des documents.
4. **L’éventuel recours à un professeur externe**, absent du programme initial de données déterministes.
5. **L’adoption finale** de CCE, d’une variante SFT, d’une base refroidie/EMA et d’une seule innovation de pré-entraînement, chacune après sa comparaison propre.

Ces décisions concernent l’exécution future ; elles ne conditionnent pas la remise du présent programme.