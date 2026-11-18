# Entraîner un petit LLM avec moins de calcul et de données : compléments de veille 2025–2026

*Recherche documentaire au 17 novembre 2026. Aucun fichier modifié, aucun calcul GPU lancé.*

## Résumé en 15 lignes

1. Les pistes nouvelles les plus pertinentes concernent l’organisation des données, l’initialisation et la mémoire d’entraînement.
2. Aucun résultat consulté ne garantit un gain sur votre modèle français avec Muon et une RTX 4060.
3. Le rejet général du curriculum dans la première veille mérite d’être révisé : des résultats positifs sont désormais publiés à EACL 2026.
4. Répéter une petite source utile et conserver des doublons internes sont deux situations différentes.
5. Une étude de 2026 traite précisément les mélanges limités en données, avec des expériences en français.
6. Rho-1 mérite un essai distinct de MiniPLM : il sélectionne les cibles au niveau du token.
7. Masquer des cibles n’économise cependant pas automatiquement les passes du Transformer.
8. La distillation textuelle depuis un enseignant local contourne le problème des tokenizers différents.
9. Les données synthétiques doivent être évaluées en incluant leur coût de génération et de vérification.
10. Une initialisation sur des grammaires formelles est une piste publiée, simple et encore peu exploitée.
11. Les automates cellulaires neuronaux constituent une piste de rupture de 2026, testée dès 400M paramètres.
12. APOLLO et Cut Cross-Entropy offrent des pistes mémoire plus directement pertinentes que remplacer Muon par SOAP.
13. À une taille proche du projet, Soft-MTP améliore MTP mais reste derrière l’objectif classique dans le tableau consulté.
14. Les hybrides attention–récurrence disposent maintenant de résultats d’efficacité en pré-entraînement, notamment avec Olmo Hybrid.
15. Je recommande de conserver le run actuel et de tester ces nouveautés séparément, avec un budget comptant tous les coûts annexes.

## 1. Cadre, preuves et particularités du dépôt

Les trois veilles demandées ont été consultées en premier, puis les fichiers de modèle, d’entraînement, de SFT, la roadmap, le bilan assistant et les documents de clôture/sélection R&D. Les propositions déjà documentées — EMA, CWD, ResFormer, MiniPLM, MUDD, NorMuon, Polar Express, TOP, BeyondWeb, Engram, SuperBPE, MoEUT — ne sont pas redéveloppées.

Trois précisions issues du code influencent cette sélection :

- **Taille réelle : 335 544 320 paramètres** pour la configuration 24 × 1024 avec embeddings liés. « 369M » reste le nom du run.
- **Mémoire : réservation de 6,5 Go, plafond PyTorch de 6,1 Gio** dans les arguments de `run_369m.py`. La marge disponible ne doit donc pas être calculée simplement par rapport aux 8 Go physiques.
- **Durée : 113 000 × 65 536 = 7,406 milliards de tokens.** À 6 300 tok/s constants, cela représente **13,6 jours**, hors interruptions ; les quelque 12 jours correspondent au débit initialement envisagé de 7 000 tok/s.

### Lecture des statuts

- **Établi** : résultat publié et reproduit indépendamment dans un cadre identifiable.
- **Publié, non reproduit** : publication confirmée, mais reproduction indépendante du gain précis non trouvée. Ce n’est pas une prépublication, ni une preuve de transfert.
- **Prépublication** : manuscrit ou rapport technique ; éventuelle acceptation signalée séparément.
- **Hypothèse** : adaptation proposée au projet.

**Aucun gain local n’est établi pour les nouveautés ci-dessous.** Lorsqu’une estimation crédible manque, j’indique « inconnu » plutôt que d’inventer une fourchette. Les seuils expérimentaux sont des **objectifs de décision proposés**, pas des gains prédits.

### Protocole commun proposé

Pour les méthodes de pré-entraînement : témoin **12 × 512, environ 54,5M paramètres**, tokenizer et mélange actuels, validation fixe non filtrée, résultats séparés par source. Présélection à 100M tokens par bras, puis confirmation des finalistes à 300M tokens et trois graines.

Comparer séparément :

1. qualité à **temps total égal** ;
2. qualité à **nombre de tokens traités égal** ;
3. qualité à **quantité de données uniques égale**.

Le temps total comprend tri, génération, enseignant, initialisation et recherche d’hyperparamètres. Pour un débit mesuré \(q\), un bras de 300M tokens coûte \(300\,000\,000/q\) secondes ; **le débit du témoin reste inconnu**.

Seuil proposé pour une piste d’accélération : **au moins 10 % de temps total économisé à qualité comparable**, sans régression confirmée sur les évaluations françaises. Tout dépassement du plafond mémoire applicable invalide sa transposition directe.

## 2. Efficacité en données

### 2.1. Curriculum initial, puis échantillonnage aléatoire — EACL 2026

**Principe.** Ordonner les documents selon une difficulté peu coûteuse à calculer — compression, diversité lexicale, lisibilité — puis revenir au mélange aléatoire. C’est un complément au filtrage : les documents restent disponibles.

**Résultat revendiqué.** Plus de 200 modèles, tailles **0,5B à 3B**, jusqu’à **100B tokens**. Les auteurs annoncent **18–45 % de pas en moins** pour atteindre la performance de référence pendant certaines phases. Le curriculum utilisé comme amorçage conserve ensuite des améliorations ; à 100B tokens, l’annexe rapporte environ **+3,1 points** sur la moyenne évaluée. Ce n’est pas une garantie de raccourcir tout entraînement de 45 %. **Statut : publié à EACL 2026, reproduction indépendante non trouvée.** [Article et PDF](https://aclanthology.org/2026.eacl-long.271/)

**Application proposée.** Trier *à l’intérieur de chaque source*, afin de ne pas confondre difficulté et changement du mélange. Tester un amorçage par compression, puis retour au tirage habituel.

**Coût.** Pas de paramètres supplémentaires ; scores et index sur CPU. Temps Ryzen et surcoût du chargement inconnus. Code officiel non trouvé dans le PDF.

**Expérience / échec.** Trois bras : aléatoire, curriculum permanent, curriculum initial. Échec si l’avantage disparaît après la phase aléatoire ou ne satisfait pas le seuil commun.

**Nouveauté par rapport à la veille :** les résultats négatifs BabyLM ne suffisent plus à écarter toute cette famille.

### 2.2. Répétition d’époques et mélange de sources : distinguer deux effets — 2026

**Premier résultat : répéter une source utile peut fonctionner.** *Scaling Laws for Mixture Pretraining Under Data Constraints* étudie plus de **2 000 entraînements**, approximativement **100M–900M paramètres**, dont des mélanges avec du **français**. Les auteurs observent que certaines petites sources peuvent être réutilisées **15–20 fois**, grâce notamment à la présence de données générales. L’optimum dépend du budget et de la source ; leur loi extrapole imparfaitement les poids optimaux. **Prépublication ; code officiel non trouvé.** [Article, expériences et limites](https://arxiv.org/html/2605.12715v2)

**Second résultat : les doublons internes peuvent gaspiller bien davantage que leur proportion.** Dans une expérience contrôlée sur **344M**, avec **10 % du calcul** consacré à des documents répétés, le cas le plus défavorable atteint la perte d’un entraînement sans répétition utilisant seulement **67 % des FLOPs**. Cela ne signifie pas que trois passages sur Vikidia causent cette perte. **Prépublication ; code officiel non trouvé dans les passages consultés.** [Étude des répétitions internes](https://arxiv.org/abs/2606.24998)

**Application proposée.** Séparer déduplication, réutilisation volontaire d’un domaine et couverture des documents. Les trois passages équivalents sur les petites sources ne justifient pas, seuls, une réduction de leurs poids.

**Coût.** Audit et index CPU ; aucun état GPU supplémentaire. Temps de déduplication inconnu.

**Expérience / échec.** À tokens constants, comparer répétitions 1/3/6 d’un sous-corpus réduit, avec et sans mélange général. Échec si le gain est limité à la perte d’entraînement ou obtenu au détriment des autres sources. Gain attendu local : **inconnu**.

### 2.3. Rho-1 : pondérer les tokens par leur perte excédentaire — version 2025

**Principe.** Sélectionner les tokens dont la perte du modèle dépasse fortement celle d’une référence compétente. La différence distingue approximativement ce qui reste apprenable du bruit intrinsèquement difficile. Ce n’est pas simplement conserver les tokens de perte maximale.

**Résultat revendiqué.** Sur **1B et 7B**, après pré-entraînement continu mathématique puis SFT, les scores MATH atteignent **40,6 % et 51,8 %**. L’article rapporte aussi **+6,8 % d’amélioration moyenne** sur 15 tâches après 80B tokens généraux. Le chiffre « 3 % des tokens de DeepSeekMath » concerne une comparaison spécialisée, avec des modèles déjà pré-entraînés. **Publié à NeurIPS 2024 ; version révisée en janvier 2025 ; gain indépendant reproduit non établi ici.** [Article](https://arxiv.org/html/2404.07965v4), [dépôt officiel](https://github.com/microsoft/rho)

**Application proposée.** Employer une référence du projet partageant le tokenizer, puis comparer perte uniforme, sélection aléatoire et sélection par perte excédentaire. La compétence de cette référence doit être vérifiée par domaine.

**Coût.** Une passe de scoring supplémentaire. Un score FP16 par token représenterait **14,8 Go pour 7,4B tokens**, calcul de stockage brut. Masquer la loss ne supprime pas le calcul des états contextuels.

**Expérience / échec.** Essai sur 100M tokens, fractions de cibles retenues 50/75/100 %. Échec si le gain disparaît lorsque le scoring entre dans le budget, ou si seuls les scores sur les tokens sélectionnés progressent. Gain français attendu : **inconnu**.

### 2.4. Distillation textuelle locale et tâches synthétiques explicites — 2025

**Principe.** Faire produire au professeur des exercices, réponses et transformations vérifiables, puis entraîner l’élève sur le texte. Aucun alignement de logits ou de vocabulaire n’est nécessaire.

Un travail BabyLM 2025 propose **56 tâches générées automatiquement**, avec supervision par leurs réponses, et rapporte des performances compétitives face à des corpus humains de même taille. **Publié en atelier, non reproduit.** Le PDF n’a pas pu être récupéré : taille exacte des modèles, gains numériques et code restent **non vérifiés**. [Publication](https://aclanthology.org/2025.babylm-main.33/)

**Enseignant local concret.** **Qwen3-1.7B** fournit des poids ouverts, un mode sans raisonnement explicite et une couverture multilingue. Sa qualité pour enseigner vos tâches françaises reste à mesurer. Les poids BF16 représentent environ **3,4 Go**, hors cache et espace de travail ; cette arithmétique ne constitue pas une mesure de VRAM. [Fiche officielle et exemples de code](https://huggingface.co/Qwen/Qwen3-1.7B)

**Hypothèse pour le projet.** Produire un petit corpus ciblé sur les informations omises, à partir de passages sources et d’exercices vérifiables, sans reprendre les items du benchmark.

**Coût.** Génération puis SFT séparées. Pour \(G\) tokens produits à \(g\) tok/s, coût minimal \(G/g\), hors préremplissage et vérification. Le débit local est inconnu.

**Expérience / échec.** Pilote de 2 000 exemples ; comparaison avec un ajout de données existantes de même budget. Objectif proposé : **+5 points** sur un test indépendant ciblé, sans régression générale. Échec si les réponses sont erronées, le gain non confirmé, ou le coût total supérieur à l’alternative.

### 2.5. MIR : apprendre aussi à partir d’entrées masquées — juin 2026

**Principe.** Ajouter à la prédiction causale normale une loss causale sur une version aléatoirement masquée du même texte.

**Résultat revendiqué.** Modèles **72M–1,4B**, notamment sur **100M tokens uniques** répétés. Amélioration de validation d’environ **0,006 à 0,03 nat** selon la taille ; une loi ajustée estime un équivalent d’environ **1,3× les données uniques**. Les auteurs précisent que la méthode exige **deux passes avant** et vise un régime où le calcul est abondant. **Prépublication.** [Article](https://arxiv.org/html/2606.06888v2), [code](https://github.com/yixinw-lab/dc_pretrain)

**Coût local.** Deux calculs de loss et leurs gradients ; coût proche de deux traitements du lot dans une réalisation simple. Exécution séquentielle possible pour limiter le pic mémoire. Aucun gain de temps démontré.

**Expérience / échec.** Comparer MIR à un témoin avec régularisation correctement réglée, à données uniques constantes **et à temps constant**. Échec pour votre objectif si davantage de passages ordinaires donnent un meilleur résultat dans le même temps.

**Avis :** intéressante pour un petit corpus spécialisé, basse priorité pour le pré-entraînement général actuel.

## 3. Optimisation, mémoire et faible précision

### 3.1. APOLLO : compresser l’état de l’optimiseur — MLSys 2025

**Principe.** Approcher l’adaptation du taux d’apprentissage par des statistiques projetées dans un espace de faible rang, avec projections aléatoires. Les poids du modèle ne deviennent pas nécessairement de faible rang.

**Résultat revendiqué.** Performance comparable à AdamW ; jusqu’à **3× le débit** sur **8 A100 80 Go**, principalement parce que la mémoire libérée autorise un lot quatre fois plus grand. Variante quantifiée : entraînement de **7B sous 12 Go**. Ces chiffres ne sont pas des gains sur Muon ni une démonstration sous 6,1 Gio. **Publié à MLSys 2025, reproduction indépendante du gain précis non trouvée.** [Article](https://arxiv.org/abs/2412.05270), [code](https://github.com/zhuhanqing/APOLLO)

**Coût local.** Projections supplémentaires. À titre de repère, le seul tampon de momentum FP32 des quelque 302M paramètres des blocs représente **1,125 Gio** ; l’économie réelle dépend des états conservés par la variante APOLLO.

**Expérience / échec.** Muon contre APOLLO-Mini, avec réglage du lr et mesure du pic mémoire. Objectif proposé : libérer **au moins 0,5 Gio**, sans dégradation supérieure à 0,01 nat confirmée. Échec si les projections annulent le bénéfice en temps.

**Avis :** candidat mémoire raisonnable, pas remplacement automatique de Muon.

### 3.2. SOAP et FOAM : préconditionnement plus structuré — 2025–novembre 2026

**SOAP.** Adam appliqué dans une base adaptée à la courbure. Sur **360M/660M**, à grand lot, le papier annonce **plus de 40 % de pas et 35 % de temps économisés contre AdamW**. **Publié à ICLR 2025.** [Publication](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e988664070e9591f93fdcf605f7dc623-Abstract-Conference.html), [code](https://github.com/nikhilvyas/SOAP)

**FOAM.** Préconditionnement sur un seul côté et second moment factorisé. Le manuscrit du **12 novembre 2026** rapporte des pertes proches de SOAP/Muon sur **124M–760M**, environ **la moitié des états de SOAP**, et un temps de l’ordre de Muon. **Prépublication, mention NeurIPS 2026 ; recul indépendant insuffisant.** [Article](https://arxiv.org/html/2611.16905v1), [code](https://github.com/naver-ai/foam)

**Point décisif :** moitié de SOAP ne signifie pas moins de mémoire que Muon. Le tableau donne pour FOAM un état de taille \(mn+2n^2+(m+n)\), contre \(mn\) pour Muon. Sans découpage, les seuls facteurs des 24 projections MLP descendantes de largeur d’entrée 4096 ajouteraient environ **3 Gio en FP32**.

**Expérience / échec.** D’abord compter les états pour les matrices exactes du projet ; éventuellement témoin réduit avec facteurs découpés. Échec immédiat si le plafond mémoire est dépassé ; ensuite seuil commun contre **Muon**. Gain supplémentaire attendu : **inconnu**.

**Avis :** SOAP non prioritaire ; FOAM à surveiller, pas à intégrer maintenant.

### 3.3. Bi-Phase Training : faible rang avec accumulation de mises à jour — octobre 2026

**Principe.** Combiner transformations diagonales et adaptateurs de faible rang, puis fusionner périodiquement leurs mises à jour dans les poids de base. Le rang cumulé n’est donc pas limité au rang d’un seul adaptateur.

**Résultat revendiqué.** Qwen2.5 **100M–1,5B**, OLMo-2-1B, **10B/100B tokens**. À 0,5B et 10B tokens, une ablation obtient **3,014 de loss avec 206M paramètres entraînables**, contre **3,022 avec 494M** entièrement entraînables. À 100B tokens, le modèle complet reprend l’avantage moyen. Le papier rapporte aussi **5,8–7,9 Go de pic mémoire économisés par GPU** dans un montage GH200 à 1,5B. **Manuscrit indiquant Findings AACL-IJCNLP 2026 ; reproduction non trouvée.** [Article et tableaux](https://arxiv.org/html/2610.37934v1)

**Coût local.** Poids de base toujours stockés ; matrices additionnelles, fusions et gestion du momentum. Réduction de paramètres entraînables ≠ réduction proportionnelle des FLOPs. Code officiel non trouvé.

**Expérience / échec.** Témoin complet contre BPT, même corpus, plusieurs cycles de fusion. Objectif proposé : **20 % de mémoire totale économisée** avec perte à moins de 0,01 nat du témoin. Échec si le ralentissement empêche toute amélioration à temps égal.

### 3.4. COAT : quantifier les activations et états, pas seulement les multiplications — ICLR 2025

**Principe.** Stockage FP8 des activations et états d’optimiseur, avec règles de quantification adaptées.

**Résultat revendiqué.** **1,54× moins de mémoire totale** et jusqu’à **1,43× d’accélération** dans les configurations étudiées. Le dépôt fournit des intégrations Llama 2/3 et un optimiseur AdamW FP8 ; un exemple Llama-2-7B utilise **8 H100**. **Publié à ICLR 2025 ; gain local non établi.** [Code et résultats officiels](https://github.com/NVlabs/COAT)

**Fait nouveau par rapport à la veille :** une partie de l’intérêt vient de la compression du stockage, distincte du support matériel des GEMM FP8.

**Coût local.** Conversions, kernels et adaptation substantielle du modèle maison. Compatibilité Windows, RTX 4060 et Muon **non vérifiée**. Aucun chiffre de VRAM transférable directement.

**Expérience / échec.** Microbenchmark sur un bloc avant tout entraînement, puis trajectoire courte. Objectif proposé : **15 % de mémoire totale économisée**, sans divergence ni ralentissement supérieur à 5 %. Échec si seule la consommation théorique baisse.

### 3.5. Cut Cross-Entropy : éviter de matérialiser tous les logits — ICLR 2025

**Principe.** Fusionner projection de sortie et calcul de loss, en calculant le log-sum-exp par blocs.

**Résultat revendiqué.** Pour Gemma 2 **2B**, mémoire du calcul de loss **24 Go → 1 Mo**, et empreinte d’entraînement de la tête **28 Go → 1 Go**, sans détériorer la convergence dans les expériences rapportées. Ces tailles de vocabulaire et de lots dépassent celles du projet. **Publié à ICLR 2025.** [Publication](https://proceedings.iclr.cc/paper_files/paper/2025/hash/aa963ac256590bb7ad5fc26c68229a3a-Abstract-Conference.html), [code officiel](https://github.com/apple-aiml-research/ml-cross-entropy)

**Intérêt local calculable.** Un tenseur de logits `[2, 1024, 32768]` occupe **128 Mio en BF16**, **256 Mio en FP32**. `train_sft.py` effectue explicitement une conversion des logits en FP32.

**Coût.** Intégration d’un kernel ; exigences annoncées : GPU Ampere ou ultérieur et Triton. Compatibilité exacte Windows et conservation du **soft-cap tanh** à vérifier. Économie totale et accélération inconnues.

**Expérience / échec.** Comparer loss et gradients, puis 100 pas chronométrés. Objectif proposé : **au moins 128 Mio économisés** sans divergence numérique significative ni ralentissement. La version retenue doit préserver embeddings liés, masque SFT et soft-cap.

**Avis :** première piste d’ingénierie mémoire à examiner.

## 4. Objectifs, croissance et taux d’apprentissage

### 4.1. MTP progressif et Soft-MTP : intérêt réel, mais contre-exemple à petite taille

**Curriculum MTP, ACL 2025.** Passer progressivement de NTP à la prédiction de plusieurs tokens aide les petits modèles à exploiter l’objectif. Le curriculum inverse améliore aussi certaines évaluations, mais perd l’avantage du décodage spéculatif. Les gains numériques précis et le code n’ont pas été vérifiés ici. **Publié, reproduction non trouvée.** [Article](https://aclanthology.org/2025.acl-long.1243/)

**Soft-MTP, octobre 2026.** Modifier la circulation des gradients pour conserver une composante d’auto-distillation et retirer un résidu bruité. Tests **167M–1,2B**. À **332M**, perplexité FineWeb-Edu :

| Objectif | Perplexité |
|---|---:|
| NTP classique | **13,073** |
| MTP | 13,602 |
| Soft-MTP | 13,551 |

Soft-MTP gagne sur MTP, **pas sur NTP dans cette comparaison**. **Prépublication.** [Tableau 3](https://arxiv.org/html/2610.38331v1), [code](https://github.com/JaydenTeoh/Soft-MTP)

**Coût local.** Les têtes/blocs MTP restent à ajouter. Le surcoût annoncé négligeable est relatif à MTP. De plus, la dérivation suppose une projection de logits linéaire : votre soft-cap introduit une non-linéarité à prendre en compte.

**Expérience / échec.** NTP/MTP/Soft-MTP à temps égal sur témoin. Échec si NTP reste meilleur, même si Soft-MTP bat MTP. **Gain attendu sur votre configuration : aucune preuve positive trouvée.**

### 4.2. Croissance progressive synchronisée avec les données : CGLS — 2025–2026

**Principe.** Commencer avec moins de couches, puis empiler des couches au moment où les données deviennent plus complexes.

**Résultat revendiqué.** À environ **100M**, gain de **4,1 points sur ARC-Easy** face au témoin aléatoire. À l’échelle **1,2B**, ARC-Easy **41,37 % contre 36,36 %** pour le meilleur autre bras, à calcul comparable. Mais le témoin aléatoire conserve la meilleure perplexité sur The Pile. **Article signalé accepté à ICML 2026 ; reproduction indépendante non trouvée.** [Résultats](https://arxiv.org/html/2506.11389v5), [code](https://github.com/su-karanps/cgls)

**Limite importante.** Une autre étude de 2025 montre que l’avantage de réutiliser un checkpoint se sature lorsque la base a été longuement entraînée. On ne peut pas supposer que grossir la v9 aurait été systématiquement optimal. [Étude du bootstrap](https://arxiv.org/abs/2510.06548)

**Coût local.** Moins de calcul au début ; mémoire finale inchangée, recompilation et initialisation des nouveaux états. Gain net inconnu.

**Expérience / échec.** Quatre bras : profondeur fixe, croissance seule, curriculum seul, combinaison. Comparer tous les calculs depuis l’initialisation. Échec si le gain ne dépasse pas le meilleur changement simple ou si les évaluations françaises régressent.

### 4.3. Schedules fondés sur l’horizon de poursuite — ICML 2025

**Fait nouveau au-delà du WSD déjà recensé.** Une analyse relie les schedules empiriques à des bornes d’optimisation et propose d’adapter le palier ou la durée de refroidissement lorsqu’on prolonge un entraînement.

**Résultat revendiqué.** Sur des modèles Llama de **124M et 210M**, les auteurs décrivent un gain équivalent à environ **7 % de pas supplémentaires** dans leurs expériences de poursuite. Ce résultat ne valide pas la formule locale \(\kappa\sqrt{lr}\), ni son estimation d’un gain de 0,18 nat. **Publié à ICML 2025, reproduction indépendante non trouvée.** [Article](https://arxiv.org/abs/2501.18965), [explication des auteurs](https://fabian-sp.github.io/posts/2025/09/horizon/), [code](https://github.com/fabian-sp/lr-scheduling)

**Coût.** Aucun paramètre supplémentaire ; calcul du schedule négligeable. Coût expérimental : plusieurs branches de continuation.

**Expérience / échec.** Depuis un checkpoint commun de témoin, prolonger avec le même nombre de tokens et comparer les schedules à lr correctement réglés. Échec si le bénéfice ne dépasse pas la variabilité entre graines. Gain local attendu : **inconnu**.

**Avis :** complément documentaire à l’expérience de refroidissement déjà décidée, pas motif pour la remplacer.

## 5. Architectures frugales : profondeur, récurrence et mémoire

### 5.1. RecursiveGPT avec embeddings factorisés — août 2026

**Principe.** Réutiliser un bloc en profondeur et réduire la taille de la table de vocabulaire par factorisation. On dissocie paramètres stockés et profondeur de calcul.

**Résultat revendiqué.** À **10M mots**, RecursiveGPT **27,6M** obtient une moyenne de **45,80**, contre **44,93** pour le Transformer standard **41,1M**. À 100M mots, le récurrent **404,1M** atteint **63,93**, contre **63,47** pour un standard **1,22B**. Résultats moyennés sur trois graines ; les comparaisons ne doivent pas être interprétées comme une réduction identique du temps. **Prépublication.** [Article](https://arxiv.org/html/2608.26973v1), [code](https://github.com/serdardoesml/recursive-lm)

**Coût local.** Moins de poids et d’états, mais chaque récurrence coûte du calcul et des activations. L’étude vise explicitement des données rares avec calcul relativement abondant.

**Expérience / échec.** Comparer partage de couches et Transformer ordinaire sous une même limite de mémoire, puis à temps égal. Objectif proposé : **25 % de pic mémoire économisé** sans perte de qualité à temps égal. Échec si la réduction de paramètres est payée par un ralentissement excessif.

**Avis :** piste de capacité sous contrainte mémoire ; gain simultané en données et calcul non acquis.

### 5.2. Gated DeltaNet et Olmo Hybrid — résultats nouveaux en 2026

**Principe.** Alterner attention et récurrence à état matriciel, qui mémorise et met à jour des associations. Cette mémoire appartient à l’architecture entraînée ; elle diffère de l’adaptation TTT-NN testée puis abandonnée dans le cycle 1.

**Résultat revendiqué.** Olmo Hybrid **7B**, entraîné jusqu’à **6T tokens**, atteint la même perte Common Crawl avec **35 % de tokens en moins**, et le même score MMLU avec **49 % de tokens en moins** qu’Olmo 3. Les ablations descendent à de petites tailles. Attention : à la colonne nominale « 370M », le Transformer contient **371M paramètres hors embeddings**, contre **523M** pour l’hybride 3:1 ; cette colonne n’est pas une comparaison à paramètres égaux. **Rapport 2026 ; reproduction indépendante du gain précis non trouvée.** [Article et tableaux](https://arxiv.org/html/2604.03444v4)

**Code.** [GatedDeltaNet officiel](https://github.com/NVlabs/GatedDeltaNet), avec exemple 0,4B/15B tokens ; [intégration Olmo Hybrid](https://huggingface.co/docs/transformers/v5.10.2/en/model_doc/olmo_hybrid).

**Coût local.** Nouvelle architecture et kernels ; mémoire d’inférence réduite ne garantit pas un pic d’entraînement inférieur. Débit sous Windows/RTX 4060 à contexte 1024 inconnu.

**Expérience / échec.** Dense contre hybride 3:1, largeur ajustée pour égaliser réellement paramètres et mémoire. Échec si le gain disparaît à temps égal. **Avis :** candidat pour le prochain modèle, après les essais plus simples.

## 6. Idées de rupture

### 6.1. Initialisation sur des langages formels — ACL 2025

**Principe.** Avant le français, apprendre des séquences artificielles structurées, notamment des dépendances hiérarchiques de type parenthèses. L’objectif est de préparer les mécanismes d’attention, pas d’acquérir des faits.

**Résultat revendiqué.** Pour un modèle **1B**, dans une expérience autour de **1,6B tokens de langue naturelle**, même loss et meilleure généralisation linguistique avec un **budget de tokens inférieur de 33 %**. **Publié à ACL 2025 ; reproduction indépendante du chiffre non trouvée.** Le dépôt propose également des checkpoints Pythia 160M, en précisant qu’ils ne reproduisent pas strictement les résultats du papier. [Article](https://aclanthology.org/2025.acl-long.478/), [code](https://github.com/michahu/pre-pretraining)

**Coût local.** Données générables sur CPU ; pas de professeur. Une phase GPU supplémentaire, sans augmentation durable des paramètres. Gain attendu en français : **inconnu**.

**Expérience minimale.** Trois bras à calcul total égal : français seul ; données formelles puis français ; données aléatoires puis français. Ce dernier contrôle permet de vérifier que la structure compte.

**Échec.** Aucune économie nette après inclusion de l’initialisation, ou gain limité à la syntaxe avec régression factuelle.

**Avis :** meilleure piste originale à faible complexité pour un prochain pré-entraînement.

### 6.2. Initialisation sur des automates cellulaires neuronaux — mars–novembre 2026

**Principe.** Générer des séquences non linguistiques à structure spatiotemporelle, puis transférer le modèle vers le texte.

**Résultat revendiqué.** **164M tokens synthétiques** avant des entraînements de modèles **400M–3B** : jusqu’à **6 % d’amélioration** en modélisation et **1,6× de convergence accélérée**. Les auteurs rapportent aussi un avantage face à une initialisation utilisant 1,6B tokens naturels. La page du projet annonce une acceptation NeurIPS 2026 ; reproduction indépendante non trouvée. [Article, version du 11 novembre](https://arxiv.org/abs/2603.10055), [présentation des auteurs](https://hanseungwook.github.io/blog/nca-pre-pre-training/), [code](https://github.com/danihyunlee/nca-pre-pretraining)

**Coût local.** Génération synthétique plus initialisation GPU. À titre de conversion seulement, **164M tokens à 6 300 tok/s représentent 7,2 heures** ; le débit réel de cette phase et le coût du générateur sont inconnus.

**Expérience / échec.** Comparer NCA, langage formel et français seul ; comptabiliser l’ensemble des phases et contrôler le traitement des embeddings lors du transfert. Échec si le gain n’amortit pas l’initialisation.

**Avis :** piste de rupture la plus intéressante trouvée à une échelle voisine du projet, mais encore risquée.

### 6.3. Token-Superposition Training : accélérer en regroupant les tokens — mai 2026

**Principe.** Moyenner les embeddings de plusieurs tokens contigus, prédire le sac de tokens suivant, puis revenir à NTP pour récupérer un modèle génératif normal.

**Résultat revendiqué.** Tests **270M/600M/3B/10B-A1B** ; jusqu’à **2,5× moins de temps** à perte comparable. Mais le grand exemple consomme **2T tokens contre 1,05T** pour le témoin. Les auteurs précisent que, à quantité totale de texte égale, TST peut perdre son avantage. **Prépublication.** [Article](https://arxiv.org/pdf/2605.06546), [mécanisme, exemples de code et limites](https://nousresearch.net/token-superposition/)

**Coût local.** Pas de tête supplémentaire pour la variante principale ; davantage de tokens lus et phase de récupération obligatoire. Dépôt complet de reproduction non vérifié.

**Hypothèse plus adaptée au projet.** Tester d’abord la variante **sortie seule** : prédire une distribution sur quelques tokens futurs, sans compresser l’entrée ni consommer plus de texte. Son avantage face à MTP/TOP n’est pas établi.

**Expérience / échec.** NTP contre TST et sortie seule, avec deux comparaisons : données uniques constantes et temps constant. Échec si le gain exige d’augmenter fortement les répétitions ou disparaît après récupération. Gain local : **inconnu**.

## 7. Ce que les petits modèles performants enseignent réellement

Leur qualité finale démontre souvent l’intérêt d’une recette et d’un surentraînement massif. Elle ne démontre pas automatiquement une efficacité à votre budget de 7,4B tokens.

| Famille | Méthode et résultat documenté | Échelle et ressources ouvertes | Statut et implication |
|---|---|---|---|
| **SmolLM2, 2025** | Mélanges révisés entre phases, données spécialisées et instructions ; dépasse plusieurs petits modèles contemporains. Gain causal isolé de chaque ingrédient non établi. | Rapport centré sur **1,7B / environ 11T tokens** ; données et modèles ouverts. [Rapport](https://arxiv.org/abs/2502.02737) | Rapport technique ; copier la méthode d’ablation, pas supposer que 11T tokens sont évitables. |
| **SmolLM3, 2025** | Mélange progressif web/code/math, français inclus ; modèle 3B compétitif avec certains 4B. | **3B / 11,2T tokens**, **384 H100 pendant 24 jours** ; recettes et configurations publiées. [Présentation officielle](https://huggingface.co/blog/smollm3) | Résultat des auteurs ; modèle frugal à exécuter, entraînement très coûteux. |
| **MiniCPM4, 2025** | UltraClean : filtrage et génération ciblés. Ablation anglaise : moyenne **45,89 contre 44,56** pour FineWeb-Edu. Comparaison globale 8B revendiquée avec **22 %** des tokens de Qwen3. | Versions **0,5B et 8B**, rapport annonçant **8T tokens** ; [article](https://arxiv.org/html/2506.07900v1), [code](https://github.com/openbmb/minicpm). | Rapport technique. Le ratio de 22 % compare des recettes différentes ; ce n’est pas une ablation démontrant −78 % de calcul. |
| **Phi-4-mini, 2025** | Filtrage, synthétique math/code, mélange ajusté ; **67,3 au MMLU 5-shot** sur la fiche consultée. | **3,8B / 5T tokens** ; poids et code d’inférence ouverts, corpus complet de reproduction non fourni dans les sources consultées. [Rapport](https://arxiv.org/abs/2503.01743), [fiche](https://huggingface.co/microsoft/Phi-4-mini-instruct) | Rapport technique. La qualité synthétique compte, mais aucune recette complète reproductible à votre budget n’est établie. |
| **TinyLlama, référence 2024** | Kernels fusionnés et entraînement prolongé ; environ **24k tok/s par A100-40G**, selon les auteurs. | **1,1B / 3T tokens**, environ trois passages sur le corpus ; code et checkpoints. [Dépôt officiel](https://github.com/jzhang38/TinyLlama) | Référence ouverte utile pour l’ingénierie, pas preuve d’apprentissage avec peu de données. |
| **MNN-Pocket-0.3B, novembre 2026** | Architecture adaptée au matériel, vocabulaire élagué, distillation multi-professeurs et QAT. | **300M / environ 4,1T tokens** chinois-anglais ; **173 Mo en INT4**, poids et paquets de déploiement annoncés. [Rapport](https://arxiv.org/abs/2611.12311) | Prépublication très récente. Les 173 Mo concernent l’inférence ; code complet de pré-entraînement non vérifié. |

**Expérience minimale pour transférer ces recettes.** Sur le témoin, comparer un mélange fixe à une phase finale enrichie en données pertinentes, à tokens et temps égaux. Les données nouvelles doivent être préparées sans consulter le test final. Coût GPU du modèle inchangé ; coût CPU/génération à ajouter. **Gain français attendu : inconnu.** Échec si la moyenne progresse en sacrifiant une catégorie importante, ou si les données générées coûtent plus que le calcul économisé.

Cette expérience est distincte d’une copie des proportions de SmolLM ou Phi : celles-ci répondent à d’autres langues, tâches et volumes.

## 8. Tableau récapitulatif et priorités

Les chiffres ci-dessous sont ceux des sources, **jamais des promesses sur la RTX 4060**.

| Piste | Gain documenté principal | Ressource principalement économisée | Coût ou réserve déterminante | Priorité proposée |
|---|---|---|---|---|
| Curriculum initial | 18–45 % de pas vers certains niveaux de référence | Calcul, éventuellement tokens | Généralisation du gain final à vérifier | **Haute** |
| Mélange tenant compte des répétitions | Réutilisation utile jusqu’à 15–20 fois dans certains régimes | Données uniques | Ne pas confondre domaine répété et doublons | **Haute pour l’audit** |
| Rho-1 | Gains spécialisés importants ; +6,8 % moyen annoncé en généraliste | Signal d’apprentissage | Scoring supplémentaire ; calcul dense conservé | Moyenne |
| Distillation textuelle locale | Gain local non connu | Données SFT pertinentes | Génération et contrôle qualité | **Haute après la base** |
| MIR | Équivalent ajusté à 1,3× les données uniques | Données | Deux passes ; davantage de calcul | Basse |
| APOLLO | États fortement réduits ; gain système jusqu’à 3× contre AdamW | VRAM | Comparaison locale nécessaire contre Muon | Moyenne |
| SOAP / FOAM | SOAP : −35 % temps contre AdamW ; FOAM allège SOAP | Convergence | États potentiellement trop volumineux | Basse |
| BPT | 206M entraînables contre 494M dans une ablation | États et gradients | Fusions ; avantage non uniforme à long terme | Moyenne |
| COAT | Mémoire /1,54 ; vitesse ×1,43 dans les tests | VRAM et temps | Kernels et compatibilité locale | Conditionnelle |
| Cut Cross-Entropy | Suppression de la grande matrice globale de logits | VRAM | Soft-cap, SFT et Windows à valider | **Haute** |
| Soft-MTP | À 332M, meilleur que MTP, moins bon que NTP | Aucun gain établi ici | Surcoût MTP et dérivation avec soft-cap | Basse |
| CGLS | +4,1 points ARC-Easy à petite échelle | Qualité par calcul | Croissance et curriculum à isoler | Moyenne |
| Schedule de continuation | Environ 7 % de pas équivalents | Calcul | Réglage dépendant de l’horizon | Complément au programme |
| RecursiveGPT | 27,6M dépasse 41,1M dans un régime limité | Paramètres et données | Récurrence non gratuite en temps | Moyenne |
| Olmo Hybrid | −35 % tokens à perte comparable, à 7B | Données et calcul | Nouvelle architecture ; budgets exacts à égaliser | Moyenne |
| Langages formels | −33 % de budget tokens dans le test publié | Données et calcul | Transfert français à démontrer | **Haute en R&D** |
| Automates cellulaires | Jusqu’à ×1,6 de convergence | Données naturelles, calcul potentiel | Initialisation et génération à amortir | **Exploratoire prioritaire** |
| TST | Jusqu’à ×2,5 de vitesse | Calcul | Consomme davantage de texte | Sortie seule à explorer |

## 9. Programme recommandé pour le prochain cycle

Je retiendrais **quatre expériences principales**, plutôt qu’un balayage de toutes les méthodes :

1. **Cut Cross-Entropy : vérification fonctionnelle et mémoire.** C’est la modification dont le bénéfice matériel est le plus directement mesurable. Elle doit préserver exactement les particularités du modèle.
2. **Curriculum initial par source.** Peu de changements d’architecture, pas de professeur, et un résultat publié récent qui justifie de rouvrir cette piste.
3. **Initialisation structurée.** Commencer par les langages formels ; n’engager NCA qu’après une reproduction minimale du générateur et un budget complet.
4. **Distillation textuelle ciblée après la base 369M.** Viser les omissions d’information sur un test indépendant, avec réponses vérifiables et professeur évalué avant génération massive.

L’audit des répétitions peut préparer ces essais sur CPU. APOLLO devient prioritaire si les mesures montrent que libérer les états d’optimiseur permet effectivement un meilleur régime d’entraînement.

**La principale limite documentaire demeure le transfert au français, à Muon et à 6,1 Gio.** Les nouveautés les plus séduisantes doivent franchir ce test précis : une amélioration à budget total égal, mesurée sur vos tâches, et non seulement un meilleur chiffre que leur propre témoin publié.