# Veille entraînement — cellule R&D creation_llm

Date : 2026-11-16. Pôle : `rd-veille-entrainement`. Aucun calcul GPU lancé. Aucun fichier existant modifié.

## 0. Cadre et limites de cette veille

**Fichiers du projet lus** : `rd/README.md`, `model.py`, `train.py`, `run_369m.py`, `bilan_assistant_v2.md`, fin de `run_369m.out`.

**Faits relevés dans le code (lus, non mesurés par moi)**
- `model.py` contient déjà : RoPE, RMSNorm sans paramètre, QK-Norm, ReLU², projections initialisées à zéro, soft-cap des logits à 30, embeddings liés, recalcul des activations. Il ne contient pas : résidu de valeurs, connexions de saut entre couches, perte auxiliaire, moyenne de poids.
- `train.py` : Muon (`torch.optim.Muon`) sur `blocks.*`, AdamW sur le reste, décroissance cosinus jusqu'à 0,1 x lr, weight decay 0,1 classique, clip de gradient à 1,0.
- `run_369m.py` passe par `MixedSource` : les fenêtres sont tirées à une position aléatoire, sans alignement sur les débuts de documents (`align_docs` n'a pas d'effet sur une source mixte).
- `run_369m.out` (dernière ligne lue) : pause au pas 10 235 sur 113 000, environ 6 100 à 6 300 tok/s, pic 5,70 Go.

**Contrainte majeure** : le pré-entraînement du 369M est en cours et prioritaire sur le GPU. Trois classes de pistes en découlent, indiquées pour chacune :
- **[A]** applicable au run en cours sans le perturber ;
- **[B]** applicable à la phase SFT ou à un petit modèle témoin (GPU requis, jeton VertIA) ;
- **[C]** exige un nouveau pré-entraînement (architecture ou tokenizer changés).

**Méthode et réserves sur les sources**
- Recherche web du 2026-11-16. Les pages ont été lues par un outil d'extraction automatique (résumé par un petit modèle) : les chiffres ci-dessous sont ceux rendus par cet outil à partir de la page indiquée, pas une lecture intégrale des PDF. Chaque source porte la mention « ouverte » (page primaire extraite) ou « résumé de recherche seulement ».
- Presque tous les gains publiés sont mesurés en anglais, sur 8 x H100 ou plus, souvent à plus grande échelle. Leur transfert au français, à 336M paramètres et à 8 Go est une HYPOTHÈSE partout où ce n'est pas dit autrement.
- Étiquettes : **ÉTABLI** = publié en conférence et repris par au moins un tiers indépendant (ici, le plus souvent, le speedrun modded-nanogpt ou nanochat). **PRÉPUBLICATION** = arXiv, ou publié sans reproduction indépendante trouvée. **HYPOTHÈSE** = extrapolation de la cellule.

**Témoin commun proposé pour les expériences** (HYPOTHÈSE de protocole) : un modèle réduit de même famille (par exemple 12 x 512, même tokenizer 32k, même mélange de données), budget fixe en tokens, 3 graines par bras, métrique = perte de validation sur le même jeu. Justification : nanochat rapporte que des changements trouvés sur un modèle de profondeur 12 se sont transférés à la profondeur 24 (source en piste 13). Durée d'un bras sur la RTX 4060 : à mesurer, non estimée ici.

---

## 1. Classement (rapport gain / coût sur ce matériel)

| Rang | Piste | Classe | Coût | Gain publié (source) | Confiance transfert |
|---|---|---|---|---|---|
| 1 | Moyenne de poids en fin de run (EMA sur CPU) | A | quasi nul | non chiffré isolément | moyenne |
| 2 | Cautious Weight Decay | B/C | nul | perte val 2,51 -> 2,42 (OLMo-1B, Muon) | moyenne |
| 3 | Résidu de valeurs (ResFormer) et sauts | C | nul en VRAM | -16,11 % paramètres, -20,3 % données à perte égale | bonne |
| 4 | Tri du corpus par perplexité / MiniPLM | A/C | dizaines d'heures d'inférence | jusqu'à 1,45x moins de pas ; 2,4x moins de données | moyenne |
| 5 | Planification : alignement EoS, lot et longueur croissants, plancher de lr | C | nul | 1 à 2 % de temps chacun (speedrun) | moyenne |
| 6 | Connexions denses dynamiques (MUDD) | C | +0,4 % calcul | équivalent 1,8 à 2,4x de calcul | moyenne |
| 7 | NorMuon et Polar Express | C | réécriture de Muon | +11,31 % d'efficacité contre Muon (1,1B) | faible à moyenne |
| 8 | Objectif auxiliaire TOP | C | +33,5M paramètres | Lambada PPL 30,34 -> 28,76 (340M) | moyenne |
| 9 | Suppression du recalcul par états en bf16 | B/C | nul | +13 % de débit (mesure locale sur le 192M) | faible |
| 10 | Reformulation synthétique du corpus | B/C | très élevé en génération | +4,2 points à 20 milliards de tokens | faible à ce volume |
| 11 | Embedding de bigrammes hachés (type Engram) | C | table en RAM ou VRAM | -5 % de temps (speedrun) | faible |
| 12 | Distillation de logits en pré-entraînement | C | très élevé | +1,6 % à +8 % (élève 1,9B) | faible |
| 13 | Recherche automatique sur modèle réduit | B | temps GPU | -11 % de temps (nanochat) | moyenne |
| 14 | Tokenisation SuperBPE | C | ré-encodage complet | -33 % de tokens (vocabulaire 200k, anglais) | faible |
| 15 | FP8 et entraînement quantifié | C | incertain sur RTX 4060 | +4,3 % de vitesse (nanochat, H100) | très faible |

---

## 2. Pistes détaillées

### Piste 1 — Moyenne de poids en fin de run (EMA tenue sur CPU) [A]
- **Mécanisme** : conserver en RAM une moyenne exponentielle des poids, mise à jour tous les N pas, et évaluer cette moyenne en plus du point final. Le record 92 du speedrun liste « Tail weight averaging (EMA) with a raised LR floor ».
- **Gain attendu** : non chiffré isolément dans la source (le record 92 cumule plusieurs changements). HYPOTHÈSE de la cellule : 0,005 à 0,02 nat de perte de validation, d'autant plus que le lr final reste élevé (ici 0,1 x 3e-3).
- **Coût** : environ 1,35 Go de RAM (336M x 4 octets), 0 Go de VRAM, une copie GPU -> CPU tous les N pas (coût à mesurer). Aucune modification de la trajectoire d'entraînement.
- **Expérience minimale** : activer l'EMA sur le dernier tiers du run 369M (ou sur un run témoin), évaluer `val` avec et sans moyenne au même pas, sur les mêmes 40 lots.
- **Critère d'échec** : perte EMA >= perte du point final, ou débit réduit de plus de 1 %.
- **Risque** : faible. Seule piste qui ne perturbe pas le run en cours ; demande toutefois de modifier `train.py` et de relancer depuis le checkpoint (décision hors de mon périmètre).
- **Statut** : HYPOTHÈSE pour le gain ; technique listée dans une source ouverte.
- **Source** : https://github.com/KellerJordan/modded-nanogpt (ouverte, liste des techniques du record 92).

### Piste 2 — Cautious Weight Decay (CWD) [B/C]
- **Mécanisme** : n'appliquer la décroissance de poids qu'aux coordonnées où la mise à jour et le paramètre ont le même signe : `x <- x - lr * (u + lambda * 1[u * x >= 0] * x)`. Aucun hyperparamètre ajouté.
- **Gain publié** : OLMo-1B, 100 milliards de tokens : perte de validation 2,65 -> 2,56 avec AdamW et 2,51 -> 2,42 avec Muon (tableau 2). À 338M avec AdamW sur C4, 20 tokens par paramètre : 3,0136 -> 3,0059 seulement (tableau 6). Speedrun : record 43, 2,313 -> 2,284 min (1,3 %), étendu aux paramètres Adam au record 50.
- **Gain attendu ici** : HYPOTHÈSE, entre 0,008 nat (chiffre 338M AdamW) et 0,09 nat (chiffre 1B Muon). L'écart entre les deux tailles est grand : ne pas retenir le chiffre haut sans mesure.
- **Coût** : nul en VRAM et en temps. `torch.optim.Muon` applique sa propre décroissance : il faut la mettre à 0 et appliquer le masque à la main, ou réécrire le pas.
- **Expérience minimale** : modèle témoin, 3 graines, CWD contre décroissance standard, même budget.
- **Critère d'échec** : écart de perte de validation inférieur à l'écart-type entre graines.
- **Risque** : changer la règle de décroissance au milieu du run 369M n'est couvert par aucune source ; à réserver à un nouveau run ou au SFT.
- **Statut** : ÉTABLI pour l'existence de l'effet (ICLR 2026 d'après le résumé de recherche, repris par le speedrun) ; ampleur à 336M : faible d'après la source elle-même.
- **Sources** : https://arxiv.org/abs/2510.12402 (v2 HTML ouverte, 100 000 premiers caractères) ; https://github.com/KellerJordan/modded-nanogpt (ouverte).

### Piste 3 — Résidu de valeurs (ResFormer) et connexions de saut [C]
- **Mécanisme** : ajouter aux vecteurs de valeurs de chaque couche d'attention ceux de la première couche (mélange appris). Variante du speedrun : tables d'embeddings supplémentaires injectées dans les valeurs, sauts de l'embedding vers chaque bloc.
- **Gain publié** : même perte de validation avec 16,11 % de paramètres en moins et 20,3 % de données en moins (résumé arXiv v5, juin 2025). Speedrun : record 9 (sauts de valeurs et d'embedding, avec d'autres changements) 10,8 -> 8,2 min ; record 14 (embeddings de valeurs) 4,66 -> 4,41 min.
- **Gain attendu ici** : HYPOTHÈSE, équivalent de 15 à 20 % de tokens, soit 2 à 3 jours sur 14.
- **Coût** : la variante ResFormer n'ajoute aucun paramètre et garde un tenseur de valeurs de plus en mémoire (négligeable à lot 2). La variante « embeddings de valeurs » coûte 33,5M paramètres par table (32 768 x 1 024), soit environ 0,4 Go avec gradient et état d'optimiseur : à éviter sous 8 Go. Le cache KV d'inférence et le recalcul des activations doivent transmettre les valeurs de la couche 1.
- **Expérience minimale** : témoin avec et sans résidu de valeurs, 3 graines.
- **Critère d'échec** : gain inférieur à 0,01 nat ou inférieur à l'écart-type entre graines.
- **Risque** : faible ; incompatible avec la reprise du checkpoint actuel.
- **Statut** : ÉTABLI (article arXiv v5 et reprise par le speedrun et nanochat).
- **Sources** : https://arxiv.org/abs/2410.17897 (résumé ouvert) ; https://github.com/KellerJordan/modded-nanogpt (ouverte).

### Piste 4 — Tri du corpus par un modèle de référence [A pour le tri, C pour le bénéfice complet]
- **Mécanisme** : (a) élagage par perplexité : un petit modèle de référence note chaque document, on écarte une tranche choisie de la distribution ; (b) MiniPLM : on garde les documents où un professeur est bien meilleur qu'un petit modèle de référence (« difference sampling »), un seul nombre par séquence, calculé hors ligne, indépendant du tokenizer.
- **Gain publié** : (a) référence de 125M paramètres, modèle cible de 3B : jusqu'à +2,04 de performance moyenne et jusqu'à 1,45x moins de pas (résumé arXiv ; le résumé de recherche signale des chiffres plus bas dans la version ICLR : 1,35 % et 1,36x). (b) MiniPLM, professeur Qwen-1.5 1,8B, référence 104M : élève de 200M 39,9 -> 41,3 ; 500M 43,2 -> 44,8 ; besoin en données réduit de 2,4x dans le régime limité en données ; fonctionne entre familles et tokenizers différents (Llama 212M : 41,0 -> 41,8).
- **Gain attendu ici** : HYPOTHÈSE, 1 à 1,5 point sur une moyenne de tâches, ou 20 à 30 % de pas en moins à perte égale.
- **Coût** : (a) le projet dispose déjà d'un modèle de 192M (v9) utilisable comme référence ; noter 6,65 milliards de tokens demande une passe d'inférence complète, dont la durée est à mesurer. (b) demande en plus un professeur maîtrisant le français tenant dans 8 Go : modèle à choisir, non sourcé ici. Stockage : un flottant par séquence.
- **Expérience minimale** : noter 200 millions de tokens avec la v9, entraîner le témoin sur le sous-ensemble trié contre un tirage aléatoire de même taille, évaluer sur le jeu de validation non trié et sur FrenchBench.
- **Critère d'échec** : pas de gain sur le jeu de validation non trié (le tri déplace la distribution : un gain mesuré sur le jeu trié ne compte pas).
- **Risque** : le corpus ne fait que 6,65 milliards de tokens pour 7,4 prévus ; en écarter 25 à 50 % impose plusieurs passes sur le reste. MiniPLM a mesuré ce régime (alpha 0,25, environ 4 époques, meilleur résultat), mais en anglais. Risque de biaiser contre les textes simples (vikidia, lecarnet).
- **Statut** : PRÉPUBLICATION (MiniPLM publié à ICLR 2025, reproduction indépendante non trouvée ; élagage par perplexité : arXiv 2024).
- **Sources** : https://arxiv.org/abs/2410.17215 et https://arxiv.org/html/2410.17215v3 (ouvertes, 100 000 premiers caractères) ; https://arxiv.org/abs/2405.20541 (résumé ouvert).

### Piste 5 — Planification : alignement EoS, lot et longueur croissants, plancher de lr, soft-cap [C]
- **Mécanisme** : ensemble de réglages du speedrun, chacun isolé dans un record : démarrer les fenêtres sur un début de document (record 26) ; lot croissant en trois phases (record 46) ; longueur de séquence croissante et plancher de lr relevé (record 72) ; soft-cap des logits abaissé de 30 à 15 (record 18).
- **Gain publié** : record 26 : 2,896 -> 2,863 min (1,1 %, avec changement du refroidissement) ; record 46 : 2,248 -> 2,203 (2,0 %) ; record 72 : 1,516 -> 1,496 (1,3 %) ; record 18 : 3,57 -> 3,4 (4,8 %).
- **Gain attendu ici** : HYPOTHÈSE, 3 à 6 % cumulés. Les gains du speedrun sont mesurés sur un run de moins de 3 minutes vers une perte cible de 3,28 : leur valeur sur 14 jours n'est pas établie.
- **Coût** : nul en VRAM. La longueur croissante réduit le coût d'attention en début de run. L'alignement EoS demande d'étendre `MixedSource` (les débuts de documents existent déjà pour une source simple).
- **Expérience minimale** : un bras par réglage sur le témoin, 3 graines.
- **Critère d'échec** : aucun réglage ne dépasse l'écart-type entre graines.
- **Risque** : faible ; réglages à retoucher à chaque taille.
- **Statut** : ÉTABLI dans le cadre du speedrun (records validés statistiquement d'après les règles du dépôt) ; transfert : HYPOTHÈSE.
- **Sources** : https://github.com/KellerJordan/modded-nanogpt (ouverte) ; https://metr.org/notes/2026-04-21-ai-rd-nanogpt-progress/ (ouverte).

### Piste 6 — Connexions denses dynamiques (MUDD) [C]
- **Mécanisme** : chaque bloc reçoit une combinaison des sorties de tous les blocs précédents, avec des poids calculés dynamiquement par position et distincts pour les entrées requête, clé, valeur et résidu.
- **Gain publié** : performance d'un Transformer entraîné avec 1,8 à 2,4x plus de calcul ; MUDDPythia-2.8B au niveau de Pythia-6.9B pour +0,23 % de paramètres et +0,4 % de calcul (résumé arXiv, ICML 2025). Speedrun : record 81, 1,406 -> 1,363 min (3,1 %) ; record 85, 1,320 -> 1,271 (avec une attention composable).
- **Gain attendu ici** : HYPOTHÈSE, 5 à 20 % de tokens équivalents. L'écart entre le chiffre des auteurs (1,8 à 2,4x) et celui du speedrun (3 %) est grand : la base du speedrun contient déjà des sauts, celle des auteurs non.
- **Coût** : garder les 24 sorties de bloc : 24 x 2 x 1 024 x 1 024 valeurs en bf16, environ 0,1 Go. Interaction avec le recalcul des activations à coder (chaque bloc recalculé dépend de toutes les sorties précédentes).
- **Expérience minimale** : témoin avec et sans MUDD, 3 graines, en mesurant aussi le débit.
- **Critère d'échec** : gain en perte inférieur à la perte de débit, à temps égal.
- **Risque** : complexité de code ; cache KV d'inférence à adapter.
- **Statut** : ÉTABLI (ICML 2025 et reprise par le speedrun) ; ampleur incertaine.
- **Sources** : https://arxiv.org/abs/2502.12170 (résumé ouvert) ; https://github.com/KellerJordan/modded-nanogpt (ouverte).

### Piste 7 — NorMuon et Polar Express [C]
- **Mécanisme** : NorMuon ajoute à Muon une normalisation par neurone (second moment par ligne, appliqué après l'orthogonalisation). Polar Express remplace l'itération de Newton-Schulz par des polynômes optimaux au sens minimax, stables en bf16.
- **Gain publié** : NorMuon : 21,74 % d'efficacité en plus contre Adam et 11,31 % contre Muon en pré-entraînement à 1,1B (résumé arXiv), empreinte mémoire « comparable à Muon ». Speedrun : record 41 (NorMuon) 2,358 -> 2,345 min (0,6 %), record 42 (réglage) -> 2,313 ; record 38 (Polar Express) 2,483 -> 2,476 (0,3 %).
- **Gain attendu ici** : HYPOTHÈSE, 1 à 10 % de pas en moins. Les deux sources divergent d'un ordre de grandeur.
- **Coût** : `torch.optim.Muon` ne propose pas ces variantes (affirmation non contrôlée dans la documentation PyTorch 2.11) : il faudrait un Muon maison. Mémoire : un scalaire par ligne de matrice, négligeable.
- **Expérience minimale** : témoin, Muon PyTorch contre NorMuon, 3 graines, balayage de lr sur 3 valeurs par bras.
- **Critère d'échec** : gain inférieur à 2 % de pas à perte égale.
- **Risque** : un Muon maison non compilé peut être plus lent que celui de PyTorch et annuler le gain.
- **Statut** : ÉTABLI pour NorMuon (ICML 2026 d'après le résumé de recherche, repris par le speedrun) ; Polar Express : PRÉPUBLICATION (la page arXiv ouverte ne nomme aucune conférence ; le résumé de recherche indique ICLR 2026, non recoupé).
- **Sources** : https://arxiv.org/abs/2510.05491 (résumé ouvert) ; https://arxiv.org/abs/2505.16932 (résumé ouvert) ; https://github.com/KellerJordan/modded-nanogpt (ouverte).

### Piste 8 — Objectif auxiliaire : prédiction de l'ordre des tokens (TOP) [C]
- **Mécanisme** : en plus du token suivant, une seconde tête apprend à classer les tokens à venir selon leur proximité (perte de classement). Une seule couche de sortie ajoutée, retirée à l'inférence.
- **Gain publié** : modèle de 340M, 52 milliards de tokens de FineWeb-Edu : Lambada PPL 30,34 -> 28,76 ; HellaSwag 42,53 -> 43,57 ; SciQ 74,90 -> 79,80 ; mais TriviaQA 4,93 -> 4,37 et Social IQa 39,82 -> 39,00 (TOP ne gagne pas partout). Speedrun : une prédiction multi-tokens au record 53, 2,037 -> 1,988 min (avec d'autres changements).
- **Gain attendu ici** : HYPOTHÈSE, environ 5 % de perplexité. Le modèle testé a la taille du 369M, mais avec 7 fois plus de tokens.
- **Coût** : tête supplémentaire de 32 768 x 1 024 = 33,5M paramètres, environ 0,4 Go avec gradient et état AdamW, pour une marge actuelle d'environ 0,4 Go sous le plafond de 6,1 Go. Surcoût de calcul : une projection de sortie de plus.
- **Expérience minimale** : témoin avec et sans TOP, 3 graines, mesure de la perte du token suivant seule.
- **Critère d'échec** : perte du token suivant non améliorée, ou dépassement du plafond de VRAM sur le 369M.
- **Risque** : VRAM ; résultats mitigés par tâche ; une seule équipe.
- **Statut** : PRÉPUBLICATION (ICML 2026 d'après le résumé de recherche, pas de reproduction indépendante trouvée).
- **Source** : https://arxiv.org/abs/2508.19228 (v2 HTML ouverte).

### Piste 9 — Gestion mémoire : supprimer le recalcul des activations [B/C]
- **Mécanisme** : le recalcul des activations coûte une passe avant supplémentaire. Deux voies pour s'en passer sous 8 Go : (a) stocker gradients et quantité de mouvement de Muon en bf16 (le speedrun passe les poids d'attention et de MLP en bf16 et Muon en précision mixte au record 57) ; (b) blocs réversibles, qui reconstruisent les activations au lieu de les stocker.
- **Gain attendu** : mesure locale du `bilan_assistant_v2.md` sur le 192M : 14 906 tok/s sans recalcul contre 13 171 avec, soit +13 %. HYPOTHÈSE : gain du même ordre sur le 369M si l'on récupère environ 1,3 Go (deux tampons de 336M paramètres passant de 4 à 2 octets). Le bilan mesure 7,41 Go sans recalcul : on arriverait vers 6,1 Go, exactement au plafond, donc sans marge. Voie (b) : en GPT-2 Large, perte de validation 2,6988 -> 2,6183 et lot maximal multiplié par 9 à 10, mais GPT-2 Small légèrement dégradé (2,9022 -> 2,9148).
- **Coût** : (a) quelques lignes, mais un Muon maison ; (b) réécriture des blocs, incompatible avec le checkpoint.
- **Expérience minimale** : essai de taille à la manière du bilan (6 pas mesurés, données aléatoires) pour le pic de VRAM et le débit, puis comparaison de perte sur le témoin entre tampons fp32 et bf16.
- **Critère d'échec** : pic de VRAM au-dessus de 6,1 Go, ou perte dégradée de plus de 0,01 nat.
- **Risque** : élevé pour les poids en bf16 (8 bits de mantisse : une mise à jour relative plus petite que 0,4 % est perdue ; sur un run long à lr décroissant, c'est plausible). La voie (b) est signalée « marginalement stable » par ses auteurs et l'article ne dit rien du bf16.
- **Statut** : HYPOTHÈSE pour (a) ; PRÉPUBLICATION pour (b).
- **Sources** : https://github.com/KellerJordan/modded-nanogpt (ouverte) ; https://arxiv.org/html/2512.02056v1 (ouverte).

### Piste 10 — Données synthétiques par reformulation du corpus [B/C]
- **Mécanisme** : faire reformuler des documents existants par un petit modèle (résumé, changement de format, questions-réponses) et mélanger le résultat au corpus réel.
- **Gain publié** : BeyondWeb : à 20 milliards de tokens, mélange 50/50, 50,4 % contre 46,2 % pour les données naturelles seules (+4,2 points) ; répéter deux fois les données réelles donne 45,5 %. Un générateur de 1B suffit à battre la base (+1,8 point), le passage de 3B à 8B n'apporte que 0,4 point.
- **Gain attendu ici** : HYPOTHÈSE, non chiffrable sans connaître le volume générable. Intérêt qualitatif : le bilan attribue 139 des 151 échecs de la v2 à une information manquante, et le corpus (6,65 milliards) est plus petit que le budget (7,4 milliards).
- **Coût** : dominant. Le débit de génération d'un modèle local de 1 à 3B sur la RTX 4060 n'a pas été mesuré ; à l'ordre de grandeur de 100 tokens par seconde en flux unique (HYPOTHÈSE), un milliard de tokens demanderait plus de 100 jours. Seul un volume de 10 à 100 millions de tokens paraît atteignable, soit 0,1 à 1,5 % du corpus, loin des 40 à 50 % testés par la source.
- **Expérience minimale** : mesurer d'abord le débit de génération par lot. Si moins de 50 millions de tokens par semaine : abandonner pour le pré-entraînement et réserver au SFT.
- **Critère d'échec** : débit insuffisant, ou pas de gain du témoin avec 1 % de synthétique.
- **Risque** : erreurs factuelles du générateur en français ; une autre étude (arXiv 2510.01631, résumé de recherche seulement) signale qu'un corpus synthétique de type manuel utilisé seul dégrade la perte.
- **Statut** : PRÉPUBLICATION.
- **Source** : https://arxiv.org/html/2508.10975v2 (ouverte, 100 000 premiers caractères).

### Piste 11 — Embedding de bigrammes hachés (type Engram) [C]
- **Mécanisme** : une table indexée par le hachage des 2 ou 3 derniers tokens est ajoutée au flux résiduel : la mémorisation locale passe par une consultation de table au lieu de couches.
- **Gain publié** : speedrun, record 62 : 1,748 -> 1,655 min (5,3 %). Engram (DeepSeek, janvier 2026) : à 27B paramètres contre un MoE de même taille, MMLU +3,4, ARC-Challenge +3,7 ; table déportable en mémoire hôte avec un surcoût dit « négligeable » (non chiffré dans le résumé).
- **Gain attendu ici** : HYPOTHÈSE, 3 à 5 % de tokens équivalents. La table est un bon usage des 48 Go de RAM, seule ressource abondante de la machine.
- **Coût** : table en RAM avec gradients creux, ou en VRAM si petite (1 million de lignes x 256 dimensions = 256M paramètres, trop pour la VRAM restante). Transferts CPU-GPU à chaque pas : coût à mesurer.
- **Expérience minimale** : témoin avec une table de 2^20 lignes en RAM, mesure du débit et de la perte.
- **Critère d'échec** : perte de débit supérieure au gain de perte à temps égal.
- **Risque** : l'inférence doit embarquer la table (taille du modèle livré) ; mise à jour creuse sous Windows à valider.
- **Statut** : PRÉPUBLICATION (Engram) ; record de speedrun validé.
- **Sources** : https://arxiv.org/abs/2601.07372 (résumé ouvert) ; https://github.com/KellerJordan/modded-nanogpt (ouverte) ; https://metr.org/notes/2026-04-21-ai-rd-nanogpt-progress/ (ouverte).

### Piste 12 — Distillation de logits pendant le pré-entraînement [C]
- **Mécanisme** : mélanger la perte de modélisation avec une divergence vers les logits tronqués d'un professeur calculés hors ligne.
- **Gain publié** : professeur GLM-4-9B, élève 1,9B, 100 milliards de tokens : +1,6 % en configuration simple (37,7 -> 38,3, qualifié de « marginal » par les auteurs), 41,2 en configuration optimisée (top-0,95-50, température 2, poids 0,9 avec planification). Les élèves plus gros en profitent davantage ; l'étude descend à 330M. Stockage : environ 15 To pour 100 milliards de tokens.
- **Gain attendu ici** : HYPOTHÈSE, faible (élève de 336M dans la tranche qui profite le moins).
- **Coût** : rédhibitoire. Le professeur doit partager le tokenizer BPE 32k du projet, ce qu'aucun modèle public ne fait ; à tokenizers différents il faut une méthode dédiée (ALM, NeurIPS 2025, non testée en pré-entraînement d'après le résumé de recherche). Stockage proportionnel : environ 1 To pour 7,4 milliards de tokens (règle de trois sur le chiffre de la source).
- **Expérience minimale** : aucune recommandée avant les pistes 4 et 10, qui exploitent un professeur sans contrainte de tokenizer.
- **Critère d'échec** : sans objet.
- **Risque** : coût d'inférence du professeur supérieur au gain.
- **Statut** : ÉTABLI pour le résultat à grande échelle (ACL 2025 d'après le résumé de recherche) ; non pertinent sur ce matériel (opinion de la cellule).
- **Sources** : https://arxiv.org/html/2410.16215v1 (ouverte) ; https://arxiv.org/abs/2503.20083 (résumé de recherche seulement).

### Piste 13 — Recherche automatique d'hyperparamètres sur modèle réduit [B]
- **Mécanisme** : un agent enchaîne de courts entraînements sur un petit modèle, garde les changements qui baissent la perte, puis on les reporte sur le grand modèle.
- **Gain publié** : nanochat, classement « time to GPT-2 » : 2,02 h -> 1,80 h (9 mars 2026, changements trouvés sur un modèle de profondeur 12 et reportés en profondeur 24), puis 1,65 h (14 mars 2026, ajout de « backout » et « smear » venus du speedrun). Les articles secondaires parlent d'environ 700 essais dont une vingtaine retenus (non recoupé sur source primaire).
- **Gain attendu ici** : HYPOTHÈSE, 5 à 10 % de tokens équivalents sur le prochain run.
- **Coût** : uniquement du temps GPU, en concurrence directe avec le pré-entraînement prioritaire. Faisable seulement dans les fenêtres de pause ou après le run.
- **Expérience minimale** : c'est le témoin commun de la section 0, automatisé.
- **Critère d'échec** : les 3 meilleurs changements du témoin ne se retrouvent pas sur un modèle intermédiaire (par exemple 192M).
- **Risque** : sur-ajustement au petit modèle et au budget court ; bruit entre graines pris pour un gain.
- **Statut** : PRÉPUBLICATION (résultat de dépôt, non publié).
- **Source** : https://raw.githubusercontent.com/karpathy/nanochat/master/dev/LEADERBOARD.md (ouverte).

### Piste 14 — Tokenisation SuperBPE [C]
- **Mécanisme** : apprendre d'abord des sous-mots classiques, puis lever la contrainte d'espace pour créer des tokens couvrant plusieurs mots.
- **Gain publié** : vocabulaire de 200k, modèles de 8B en anglais : jusqu'à 33 % de tokens en moins, +4,0 % en moyenne sur 30 tâches, +8,2 % sur MMLU, 27 % de calcul d'inférence en moins.
- **Gain attendu ici** : HYPOTHÈSE, non chiffrable : aucun résultat à 32k de vocabulaire ni en français dans le résumé ouvert. Si le gain de compression se maintenait à moitié, le même texte coûterait environ 15 % de tokens en moins, donc 15 % de temps en moins à corpus égal.
- **Coût** : nouveau tokenizer, ré-encodage des 6,65 milliards de tokens, nouveau pré-entraînement, format de chat à refaire. Un vocabulaire plus grand coûte de la VRAM : 1 024 x 4 octets par ligne, triplé par le gradient et l'état d'optimiseur.
- **Expérience minimale** : sans GPU : entraîner un SuperBPE 32k sur un échantillon français et mesurer le nombre de tokens par caractère contre `tokenizer_fr32k.json`.
- **Critère d'échec** : moins de 10 % de tokens en moins à 32k.
- **Risque** : gain de compression sans gain de qualité à petite échelle.
- **Statut** : PRÉPUBLICATION pour le français et pour cette taille.
- **Source** : https://arxiv.org/abs/2503.13423 (résumé ouvert).

### Piste 15 — FP8 et entraînement quantifié [C]
- **Mécanisme** : (a) produits matriciels en FP8 pour la tête puis les MLP et projections (speedrun, records 19, 84, 89, 90) ; nanochat `--fp8`. (b) entraînement conscient de la quantification vers 1,58 bit (BitNet).
- **Gain publié** : (a) nanochat : 10 949 s -> 10 493 s, soit 4,3 % (2 février 2026, avec passage de la profondeur 24 à 26). (b) démarrer en 16 bits puis basculer en 1,58 bit est préférable au 1,58 bit dès le départ, sur 11 tâches (résumé arXiv 2502.11895).
- **Gain attendu ici** : (a) inconnu. Le support FP8 de `torch._scaled_mm` sur l'architecture Ada (RTX 4060) n'a pas pu être établi : un fil du forum PyTorch de janvier 2025 indique un support conditionné à la compilation, absent des versions standard à cette date (résumé de recherche seulement). (b) aucun gain d'entraînement : les poids latents restent en pleine précision ; l'intérêt est en inférence, à transmettre au pôle `rd-veille-inference`.
- **Coût** : (a) un test de 10 minutes sans entraînement ; (b) un pré-entraînement dédié.
- **Expérience minimale** : (a) appeler `torch._scaled_mm` en FP8 sur la machine et chronométrer contre bf16.
- **Critère d'échec** : opération non supportée ou plus lente que bf16.
- **Risque** : (a) temps perdu si non supporté ; (b) hors mission entraînement.
- **Statut** : ÉTABLI sur H100 ; HYPOTHÈSE sur RTX 4060.
- **Sources** : https://raw.githubusercontent.com/karpathy/nanochat/master/dev/LEADERBOARD.md (ouverte) ; https://github.com/KellerJordan/modded-nanogpt (ouverte) ; https://arxiv.org/abs/2502.11895 (résumé ouvert) ; https://discuss.pytorch.org/t/sm-89-not-listed-in-the-torch-cuda-get-arch-list/215827 (résumé de recherche seulement).

---

## 3. Pistes écartées ou résultats négatifs utiles

- **Curriculum par difficulté** : le rapport du premier défi BabyLM indique que les tentatives de curriculum, nombreuses, ont été « largement infructueuses » (résumé de recherche seulement, https://arxiv.org/abs/2504.08165). Le rapport du troisième défi (novembre 2025) conclut que ce sont les nouveaux objectifs et architectures qui donnent les meilleurs résultats, et que FLOPs d'entraînement et performance ne sont pas complètement corrélés (page de résumé ouverte, https://aclanthology.org/2025.babylm-main.28/). Les gagnants par piste n'ont pas pu être lus.
- **Distillation de logits classique** : voir piste 12, bloquée par le tokenizer.
- **Record 92 du speedrun** (30 août 2026, 1,126 -> 0,665 min) : softmax échantillonné, table de n-grammes hachés de 84,6 millions de lignes, optimiseur « ANVIL », capture en graphe CUDA. Trop récent, cumul de changements non isolés, et dépendant de 8 x H100 : non retenu comme piste. Le chiffre « 65B parameters » rendu par l'outil pour cette table est à recontrôler sur le dépôt.

## 4. État de l'art des concours (repères)

- **modded-nanogpt** (perte de validation 3,28 sur FineWeb, 8 x H100) : 45 min (mai 2024) -> 2,933 min (janvier 2025) -> 1,878 min (4 janvier 2026) -> 1,126 min (6 août 2026) -> 0,665 min (30 août 2026). Source ouverte : https://github.com/KellerJordan/modded-nanogpt. METR attribue l'essentiel de l'accélération à des contributions « superficielles et modérées » (environ 21x sur 31x en mars 2026) : https://metr.org/notes/2026-04-21-ai-rd-nanogpt-progress/ (ouverte).
- **nanochat** (score CORE de GPT-2, 8 x H100) : 3,04 h (29 janvier 2026) -> 1,65 h (14 mars 2026). Le plus gros saut vient d'un changement de jeu de données (FineWeb-EDU -> ClimbMix, 2,76 h -> 2,02 h), pas de l'architecture. Source ouverte : https://raw.githubusercontent.com/karpathy/nanochat/master/dev/LEADERBOARD.md. État du classement après mars 2026 : non lu.
- **Enseignement commun** : sur ces deux concours, le choix et le tri des données pèsent autant que les changements d'architecture. C'est l'argument principal pour la piste 4.

## 5. Non vérifié

- Aucun gain n'a été mesuré sur ce matériel : tous les « gains attendus ici » sont des HYPOTHÈSES.
- Lecture par extraction automatique, souvent limitée aux 100 000 premiers caractères ; aucun PDF lu intégralement. Le PDF du rapport BabyLM 2025 n'a pas pu être lu (outil PDF absent).
- Conférences de publication de CWD, NorMuon, TOP, Polar Express et de la distillation en pré-entraînement : tirées des résumés de recherche, non recoupées sur les sites des conférences.
- Support FP8 sur RTX 4060 avec PyTorch 2.11 : non établi.
- Présence ou absence de NorMuon, Polar Express ou CWD dans `torch.optim.Muon` de PyTorch 2.11 : non contrôlée.
- Existence d'un professeur français de 1 à 3B tenant dans 8 Go et de qualité suffisante (pistes 4 et 10) : non recherchée.
- Débits d'inférence et de génération sur la RTX 4060 (pistes 4 et 10) : non mesurés.
- AdaMuon et les variantes de Muon à variance adaptative : trouvées seulement dans des agrégateurs, non retenues.
