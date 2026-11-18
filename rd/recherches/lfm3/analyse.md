**LFM3-1.7B : intérêt pour le projet — analyse au 18 novembre 2026**

Le rapport est globalement solide sur les artefacts et les chiffres officiels. Son application la plus utile est **un modèle de comparaison français, puis éventuellement un producteur de données SFT contrôlées**. Il ne justifie pas de modifier le pré-entraînement actuel. La porte d’attention et GQA sont des candidats séparables pour un futur modèle ; l’architecture hybride complète reste secondaire.

Analyse en lecture seule : aucun fichier modifié, aucun modèle téléchargé, aucun calcul GPU lancé. Fichiers demandés consultés, dont `rd/programme_cycle2.md`, ainsi que les chemins d’évaluation et de préparation SFT. `DOCUMENTATION/INDEX.md` est absent.

**1. Solidité du rapport et corrections**

Statuts employés : **D** = artefact documentaire vérifié ; **P** = résultat publié par ses auteurs, sans reproduction indépendante vérifiée ; **PP** = prépublication consultée ; **H** = hypothèse locale. Aucun gain LFM3 sur notre configuration n’est un **résultat établi, publié et reproduit**.

| Point contrôlé | Conclusion |
|---|---|
| Identité et date | **D** : LFM3 existe bien ; annonce du 27 octobre 2026. La distinction assistant/Base est correcte. [Annonce](https://huggingface.co/blog/LiquidAI/lfm3) |
| Architecture | **D** : 24 couches, 13 convolutions, 5 attentions locales de fenêtre 256, 6 globales ; largeur 2 048, 16/4 têtes Q/KV, noyau 7, vocabulaire 128 000, contexte 262 144. Positions globales annoncées correctes. [Configuration](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/config.json) |
| Mécanismes | **D** : portes par tête, convolution multiplicative, SwiGLU et normalisations sandwich correspondent au code. Le RoPE partiel conserve bien la progression fréquentielle de la dimension complète. [Implémentation](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/modeling_lfm3.py) |
| Budget | **P** : 1,73B paramètres, 22 000 milliards de tokens dont les derniers 3 500 milliards de mid-training ; français annoncé. Ni volume français ni recette complète reproductible retrouvés. [Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base) |
| Benchmarks | Les six lignes du rapport correspondent à la fiche officielle. Ce sont des résultats fournisseur ; aucune reproduction indépendante de cet ensemble retrouvée. [Scores et protocole](https://huggingface.co/LiquidAI/LFM3-1.7B) |
| Vitesse | Les facteurs ≥2,5×/≥1,8× concernent préremplissage/décodage à 32K, en GGUF Q4_0, sur les trois appareils cités. Ils ne mesurent ni notre GPU ni l’entraînement. [Protocole](https://huggingface.co/LiquidAI/LFM3-1.7B) |
| Mémoire tierce | Les 3,7 Gio incluent les buffers dans le résumé ; les totaux du tableau les excluent. Le rapport restitue correctement cette distinction. Ce réglage utilise des poids à **6,56 bits/poids**, pas simplement du « 4 bits ». [Mesure tierce](https://github.com/crusaderky/pixi-llm-recipes/blob/main/perplexity/LFM3-1.7B/README.md) |
| Références scientifiques | Titres, auteurs principaux et dates vérifiés pour [LFM2](https://arxiv.org/abs/2511.23404), [Quantizable Transformers](https://arxiv.org/abs/2306.12929), [Gated Attention](https://arxiv.org/abs/2505.06708) et [RoPE](https://arxiv.org/abs/2410.06205). Manuscrits **PP** dans cet audit ; ils ne constituent pas des ablations de LFM3. |
| Licence | Distinction code Apache 2.0/poids LFM correcte. Le seuil commercial vise un revenu annuel **d’au moins** 10 M$, selon les conditions de la licence. [Licence](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE) |

**Corrections importantes pour décider :**

- **« Critère absent » n’est pas un diagnostic de connaissance manquante.** `bench_assistant.score()` cherche des expressions régulières dans la dernière réponse. Paraphrase correcte, omission, erreur factuelle et distraction peuvent produire le même indicateur. Le cycle 2 corrige déjà cette interprétation du bilan v2.
- **300 observations ne sont pas 300 questions indépendantes.** `bench_assistant.jsonl` contient **100 scénarios**, exécutés avec trois graines par `bench_assistant.run()`. Les 90 observations factuelles représentent 30 questions. Un intervalle de confiance doit regrouper les répétitions par scénario.
- **Le pilote architectural proposé n’est pas court.** À l’hypothèse du cycle 2 de 25–40k tok/s sur le témoin, 100M tokens coûtent **42–67 minutes par bras**, hors compilation/évaluation. Deux bras puis trois graines représentent plusieurs heures.
- **La mémoire des portes est sous-décrite.** Les 393 216 paramètres et 1,5 Mio par copie FP32 sont corrects ; poids, gradients et momentum Muon représentent ensemble environ **4,5 Mio**, avant temporaires et activations.
- **« Pas correctif d’un checkpoint existant » est trop absolu.** Une porte constante à 0,5, accompagnée du doublement de la projection de sortie, peut préserver algébriquement la fonction initiale. Cela ne préserve pas automatiquement la trajectoire optimiseur et ne démontre aucun bénéfice : **aucune greffe sur le run actuel**.
- Les calculs **335 544 320 paramètres**, **7 405 568 000 tokens**, **13,60 jours à 6 300 tok/s**, et les rapports approximatifs de budgets sont corrects. Le code actuel utilise lot **2 × accumulation 32**, contrairement au lot 4 × 16 encore écrit dans la roadmap.
- `train.cap_vram()` compte en **Gio**, malgré l’étiquette « Go ». Le plafond PyTorch est **6,1 Gio** ; le pic historique de 6,04 ne laisse qu’environ **61 Mio**, pas 1 Go de marge opérationnelle.

Le cache LFM3 calculé dans le rapport est juste **si le moteur borne effectivement le cache local**. La formule ne prouve pas que tous les backends adoptent cette allocation. La disponibilité de llama.cpp constitue un support d’exécution, pas une reproduction scientifique.

**2. Cadre commun des expériences**

**Gain de qualité attendu chiffré : inconnu pour toutes les adaptations.** Les seuils ci-dessous sont des décisions proposées, jamais des gains promis.

Les estimations architecturales supposent le 24 × 1 024 actuel, contexte 1 024, lot 2, recalcul des activations, FP32 pour les paramètres et autocast BF16. Les plages de débit sont des **hypothèses de dimensionnement**, non des mesures RTX 4060 ; une sélection de kernel différente peut les invalider.

Avant tout entraînement : contrôle CPU sur petit modèle, puis profil GPU de **20 micro-pas de chauffe + 100 mesurés**, avec arrêt propre au plus tard à 30–45 minutes. Mesurer temps, mémoire allouée/réservée et usage GPU global. Le débit SFT doit être mesuré séparément des 6 300 tok/s de pré-entraînement.

Le lanceur existant est `gpu_task.main()` : acquisition du jeton, vérification du détenteur et absence du 369M. **`--minutes` réserve le jeton ; il ne limite pas la durée du programme.** Les pilotes futurs doivent assurer leur propre arrêt. Les essais GPU attendent la fin du run conformément au cycle 2.

Pour la qualité : même partition, mêmes textes, comparaison appariée par scénario/famille ; intervalle à 95 % excluant zéro et confirmation des finalistes sur trois graines **d’entraînement**. Garde-fous du cycle 2 : global ≥ témoin −2 points, aucune catégorie perdant >3 points, FrenchBench ≥ témoin −2 points, défauts indésirables n’augmentant pas de >1 point.

**3. Adaptations prioritaires**

**A — Référence locale française : à tester en premier**

- **Mécanisme et intérêt SFT :** comparer LFM3 assistant aux réponses v2 puis B1, à livre fermé, avec preuve correcte, distracteurs et preuve retirée. Cela mesure sa capacité à fournir des réponses complètes et à suivre une preuve ; cela ne démontre pas la valeur de son architecture. Source **P** : [modèle assistant](https://huggingface.co/LiquidAI/LFM3-1.7B).
- **Code concerné :** réutiliser `bench_assistant.load_bench()`, `score()`, `summarize()` et `rescore()` ; prévoir un adaptateur distinct du chargement maison dans `bench_assistant.run()`. `generate.load_model()` et `chat.Chat` attendent notre GPT/tokenizer et ne chargent pas directement LFM3.
- **Comparabilité :** mêmes scénarios et paramètres d’échantillonnage ; désactiver explicitement le raisonnement pour le test principal, enregistrer le template et les troncatures. Les 300 tokens ont des longueurs textuelles différentes selon le tokenizer. Fournir les mêmes passages visibles ; l’historique maison est tronqué par `chat.build_prompt()`.
- **Coût estimé :** **0 VRAM supplémentaire et 0 variation du débit du 369M** en exécution CPU séparée et limitée. Pour LFM3 CPU quantifié : provision indicative **2–6 Gio RAM**, pas une mesure. En BF16 GPU : **3,22 Gio de poids**, provision totale **4–6 Gio à court contexte**, à vérifier ; jamais simultanément avec le run.
- **Temps :** présélection de **30 scénarios**, une graine, au maximum 300 tokens par réponse. Pour 30 réponses simples : temps de décodage ≤`9 000/q` secondes ; `q` est à mesurer. Aucun débit 5700X/4060 fiable trouvé.
- **Succès proposé :** ensuite ≥**15 réussites nettes supplémentaires sur les 300 observations**, baisse ≥5 points des omissions réelles sur le jeu ciblé, preuve appariée et garde-fous respectés. Comparer au B1 dès qu’il existe, pas seulement aux 49,7 % historiques.
- **Échec :** seuil non atteint, amélioration expliquée seulement par les regex, ou dépassement mémoire. Un intervalle inconclusif n’est pas une validation.
- **Compatibilité :** totale avec les poids actuels ; aucune amélioration automatique de notre assistant. L’évaluation d’un enseignant n’est pas une autorisation de produire un corpus.

**B — Enseignant hors ligne pour RAFT/H2 : conditionnel au test A**

- **Mécanisme :** générer des réponses françaises courtes à partir de preuves vérifiées, avec distracteurs et faits contrefactuels ; conserver uniquement les réponses fidèles. L’intérêt vise directement la sélection de l’information et la complétude. Fondement **PP** : [RAFT](https://arxiv.org/abs/2403.10131) ; choix de LFM3 comme enseignant : **H**.
- **Code concerné :** `prepare_sft.prepare()` pour séparer les familles avant augmentation ; `build_example()` et `write_split()` pour réencoder les textes avec **notre** tokenizer ; `train_sft.PackedData.batch()` et `masked_loss_sum()` restent inchangés pour une distillation textuelle simple.
- **Point bloquant concret :** les tokens sont enregistrés en `uint16` ; les identifiants du vocabulaire LFM3 à 128k ne peuvent pas être copiés dans ces fichiers. Échanger du texte, pas les identifiants ni directement les logits.
- **Coût estimé :** **0 nouveau paramètre, ΔVRAM≈0 à formes identiques** ; débit brut du SFT voisin du témoin, hypothèse **±5 %**. Le temps par réponse augmente avec les passages : facteur approximatif `tokens_traités_nouveaux/tokens_traités_témoin`. Ajouter séparément génération et validation humaine.
- **Expérience minimale :** reprendre les **200 familles/2 000 exemples** A2 ; présélection CPU sur 30 familles, puis validation des familles conservées. Un profil SFT ≤100 pas vérifie seulement la faisabilité. Le gain exige un bras B2 à budget de tokens comparable, face aux données déterministes déjà prévues.
- **Succès :** 0 contradiction dans les familles relues, 0 fuite train/test, 0 exemple >1 024 tokens ; puis **+5 points de complétude** contre le contrôle B2, y compris avec distracteurs, et garde-fous communs.
- **Échec :** hallucinations non éliminées, gain limité au style, ou absence d’avantage face aux données déterministes. Ne pas entraîner sur les réponses du benchmark.
- **Compatibilité :** aucune modification du pré-entraînement ; intervention seulement dans une branche SFT ultérieure. Le cycle 2 demande déjà une décision distincte concernant un enseignant externe.

**4. Adaptations architecturales isolables, pour un futur modèle**

**C — Porte sigmoïde par tête : candidate secondaire**

- **Mécanisme :** multiplier la sortie SDPA de chaque tête par une porte dépendant du token, avant `proj`. Fondement **PP** : [Qiu et al.](https://arxiv.org/abs/2505.06708). Le transfert à 336M est **H** ; aucun effet démontré sur `critere_absent`.
- **Code :** `model.CausalSelfAttention.__init__()` et `forward()` ; option dans `GPTConfig`. La matrice sans biais est compatible avec le classement actuel de `train.build_optimizer()` vers Muon.
- **Coût 369M estimé :** **+393 216 paramètres**, **+4,5 Mio persistants** avec gradient/momentum ; provision **+5–30 Mio au pic**. Débit envisagé **0 à −5 %**, soit environ **5 985–6 300 tok/s**. Le calcul matriciel ajouté est petit ; kernels et recalcul dominent l’incertitude.
- **Expérience minimale :** CPU : porte neutralisée et gradients/cache vérifiés ; GPU court : profil mémoire/débit. Puis témoin 12 × 512 avec/sans porte, à temps égal, sans changement de données ni de taux d’apprentissage.
- **Succès :** perte diminuée d’au moins **0,01 nat à temps égal**, confirmée entre graines ; plafond 6,1 Gio et garde-fous aval respectés. Échec si ralentissement >5 % sans gain compensateur, instabilité, ou seule amélioration de quantification sans intérêt de déploiement.
- **Compatibilité :** réserver au futur pré-entraînement. Un bénéfice SFT doit être démontré après un SFT identique sur les deux bases.

**D — GQA 16 têtes Q / 4 KV : candidate matérielle plus simple que le mélange complet**

- **Mécanisme :** partager K/V entre quatre groupes de requêtes. Présent dans la [configuration LFM3](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/config.json), **D** ; intérêt local **H**, sans gain de complétude attendu démontré.
- **Code :** ajouter `n_kv_head` à `GPTConfig` ; modifier dimensions et découpage de `CausalSelfAttention.qkv`, `forward()` et cache. Vérifier `generate.prefill()`/`sample()` et le chemin masqué de `train_sft.masked_loss_sum()`.
- **Coût 369M calculé :** conserver 16 têtes Q de dimension **64**, pas 128 comme LFM3. Retrait de **37 748 736 paramètres** ; économie persistante théorique **432 Mio** avec poids/gradients/momentum FP32. Cache BF16 à 1 024 tokens, lot 1 : **96 → 24 Mio**.
- **Débit estimé :** hypothèse **0 à +10 %**, soit **6 300–6 930 tok/s**, uniquement si le backend conserve un chemin efficace ; répétition explicite K/V ou repli de kernel peut annuler le gain. Le pic exact n’est pas déductible des seuls états persistants.
- **Expérience minimale :** CPU : comparer GQA à une référence répétant explicitement K/V, erreur maximale FP32 ≤`1e-5`, causalité et cache ; GPU court : profiler pré-entraînement **et** masque SFT.
- **Succès :** ≥**300 Mio économisés**, temps/pas ≤1,05× ; puis sur témoin, dégradation CE ≤**0,01 nat** et garde-fous SFT respectés. Échec technique ou qualitatif sinon.
- **Compatibilité :** pas de remplacement transparent des poids QKV existants. Ne pas profiter immédiatement de l’économie pour agrandir le MLP : ce serait une deuxième expérience.

**E — RoPE partiel : réserve peu coûteuse, priorité basse**

- **Mécanisme :** laisser la moitié des dimensions sans rotation pour conserver une composante d’appariement indépendante de la distance. Inspiration **PP** : [Barbero et al.](https://arxiv.org/abs/2410.06205) ; adaptation maison **H**.
- **Code :** `precompute_rope()`, option `GPTConfig`, éventuellement sélection des tables dans `GPT.forward()` ; `apply_rope()` peut conserver sa forme avec fréquences nulles. Garder d’abord la base 10 000 pour isoler le facteur de rotation.
- **Coût estimé :** **0 paramètre**, ΔVRAM≈0 si les tables gardent leurs dimensions ; débit **≈inchangé, hypothèse ±1 %**. Aucun gain sémantique chiffrable.
- **Minimum :** CPU sur tête de dimension 64 : 16 fréquences tournantes et 16 nulles ; contrôler dimensions non tournées et cache à `1e-5` près. Profil GPU court, puis essai isolé sur témoin.
- **Succès :** −**0,01 nat à temps égal**, sans recul >2 points sur tâches d’ordre/grammaire. Échec sinon. Pas de promesse d’extension à 256K.
- **Compatibilité :** les tables ne sont pas persistantes dans le checkpoint maison : changer leur construction modifie silencieusement le comportement au rechargement. Réserver à une nouvelle configuration ; intérêt pour les omissions non démontré.

**5. Idées à écarter du programme immédiat**

| Idée | Pourquoi |
|---|---|
| Copier le mélange 13 convolutions/5 locales/6 globales | Nouveau chemin de calcul, caches et segmentation. `GPT.forward()` transmet aujourd’hui un masque d’attention, pas les identifiants nécessaires à une convolution segmentée. Une convolution naïve ferait communiquer deux exemples SFT. |
| En attendre 2,5× d’accélération à contexte 1 024 | Les projections et MLP restent présents. Avec notre MLP, remplacer l’attention par la convolution conserve approximativement les **4d²** paramètres de projections et ajoute le noyau ; aucun effondrement du coût fixe. Les résultats à 32K ne sont pas extrapolables. |
| Normalisation sandwich immédiate | `Block.forward()` utilise des RMSNorm sans paramètres et des projections résiduelles initialisées à zéro. Une normalisation après ces projections change fortement les gradients initiaux. Les nouveaux vecteurs seraient actuellement envoyés à Muon : classement à corriger avant tout essai. |
| Copier SwiGLU sans ajuster la largeur | Trois matrices au lieu de deux : à largeur intermédiaire 4d, coût MLP accru de 50 %. Comparaison à budget égal nécessaire ; aucun gain isolé LFM3 trouvé. |
| Vocabulaire 128k et contexte 256K | Rupture des embeddings/tokenizer et du stockage `uint16`, hausse des logits et du calcul ; aucun besoin établi pour corriger nos omissions à contexte 1 024. |
| Remplacer le projet par le fine-tuning complet de LFM3 | Change l’objectif « from scratch ». Poids+gradients FP32 seuls ≈12,9 Gio ; l’annonce « un GPU » ne signifie pas RTX 4060 sous notre plafond. |
| Distillation en ligne des logits | Tokenizers incompatibles et enseignant supplémentaire en mémoire ; la distillation textuelle hors ligne est l’option concrète à examiner. |
| Quantification comme remède au SFT | La mesure tierce concerne l’inférence. Elle ne prouve aucune économie de mémoire pour notre entraînement Muon, ni amélioration des informations présentes dans les réponses. |

Pour rouvrir l’hybride complet : exiger d’abord **0 fuite entre documents**, équivalence cache/préremplissage à `1e-5` en FP32, puis **≥10 % de temps total économisé à qualité comparable**. Son pic VRAM et son débit RTX 4060 ne sont pas estimables avec confiance sans implémentation/backend définis ; aucun budget GPU long ne se justifie avant ce profil. Source architecturale **P/D** : [Liquid AI](https://www.liquid.ai/blog/lfm3).

**6. Articulation avec les décisions du dépôt**

| Document | Ce que cette analyse complète, remplace ou contredit |
|---|---|
| `rd/programme_cycle2.md` | **Complète A1** par une référence externe ; **complète conditionnellement A2/B2** par un enseignant, sans remplacer les données déterministes. Confirme l’ordre audit → CCE → B1 → RAFT/H2. |
| Même programme, C1/C2 | Portes/GQA sont des candidats supplémentaires éventuels, pas des ajouts automatiques à curriculum/H1 et ResFormer. Maintenir le choix final d’**une seule innovation**, ou aucune. |
| `rd/bilan_cycle1.md` | Confirme la priorité de lecture robuste : **71/90 avec preuve relue contre 49/90**, mais **54/90 avec trois passages récupérés**. LFM3 ne résout pas à lui seul la sélection des preuves de notre modèle. |
| `roadmap_modele_369m.md` | Ne remplace aucune phase ni aucun seuil. Conserver base → évaluation → SFT standard deux étapes → benchmark. Corriger seulement l’interprétation des durées, du lot courant et de la marge mémoire dans les décisions futures. |
| Veilles et `rd/selection.md` | Prolonge RAFT et les questions de données ; ne rouvre ni CFG ni TTT-NN. LFM3 n’est pas un MoE, et ses budgets ne réfutent aucune conclusion locale sur leur coût. |
| `rd/pole_mathematiques.md` | Complète l’analyse des budgets : l’écart d’exposition aux données empêche d’attribuer la qualité à l’architecture. Ne modifie ni les quotas actuels ni les hypothèses de couverture. |

À TESTER — LFM3 comme référence française puis enseignant RAFT conditionnel ; aucun résultat vérifié ne justifie de modifier le pré-entraînement en cours.