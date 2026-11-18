**Le rapport justifie une amélioration du protocole d’évaluation, sans changement du tokenizer ni du pré-entraînement actuel.** Ses chiffres principaux sont exacts ; plusieurs précisions statistiques et deux contraintes du code doivent compléter ses propositions.

Lecture seule effectuée ; aucun fichier modifié, aucun calcul GPU lancé. `DOCUMENTATION/INDEX.md` est absent. Les fichiers cités ont été consultés, avec lecture ciblée des veilles et recherches connexes.

**Solidité du rapport**

Convention : **établi** = publié et reproduit indépendamment ; **publié** = publication vérifiée, reproduction non vérifiée ; **prépublication** = résultats des auteurs non encore établis ; **hypothèse** = proposition locale. Aucun gain sur notre configuration n’est établi.

| Affirmation | Vérification |
|---|---|
| Identité de *Tokenizer Myths* | Schaeffer et Koyejo, 3 novembre 2026 : confirmé ; prépublication. |
| Expérience française | 10 Go d’apprentissage, 0,223 Go de validation ; minimum uniforme 3,3195 BPB à 26 977 entrées : confirmé. Aucun optimum de vocabulaire entraîné démontré. |
| Corrélations +0,02 / −0,22 / −0,28 / −0,30 | Confirmées : **Pearson**, sur les textes des benchmarks. |
| 85 tokenizers, 0,98–1,60B, 9–10,5B tokens | Confirmé ; 102 entraînements avant agrégation, effectifs évaluables variables. |
| 0,989 → 0,057 sur 56 points | Confirmé, mais la figure réanalysée mesure **BPC**, bits par caractère ; points numérisés, pas journaux originaux. |

Ces vérifications proviennent du [texte et des annexes](https://arxiv.org/html/2611.04215v1). Deux réserves supplémentaires : les intervalles ne propagent pas toute l’incertitude des entraînements ; sur le mélange de validation, la corrélation de Spearman avec GSM8K atteint −0,65. Généraliser « BPB prédit peu » à tous les protocoles serait excessif. Le code spécifique reste annoncé, reproduction indépendante non trouvée. Une réponse de Limisiewicz est déjà discutée dans l’annexe F : la recherche de critiques du rapport est incomplète. Le trilemme reste conditionné par son cadre de représentabilité et normalisation.

Les sources voisines sont correctement identifiées :

- **TokEval : publié**, COLM 2026 indiqué dans la [notice officielle](https://arxiv.org/abs/2608.18062). Le maximum de corrélation 0,80 concerne des mesures informationnelles et la modélisation du langage, pas une prédiction universelle de réussite.
- Les [outils officiels](https://github.com/cimeister/tokenizer-intrinsic-evals) existent ; leurs définitions ont changé entre versions. Épingler un commit est nécessaire. La compatibilité Windows locale n’est pas démontrée.
- Les [artefacts actuels](https://huggingface.co/cmeister/tokenizer-lm-ablations) ont évolué : leur inventaire courant ne doit pas être confondu avec la révision analysée.
- **Tao et al. : publié**, NeurIPS 2024. Les tailles 33M–3B, 500 milliards de caractères et ARC-Challenge 29,1 → 32,0 à budget identique sont confirmés dans la [source primaire](https://arxiv.org/abs/2407.13623). Ce résultat ne prescrit ni 43k ni 64k pour notre modèle français.

**Corrections et implications vérifiées dans le dépôt**

1. **Le calcul BPB est juste, sous protocole fixe.** Dans [train.py](/D:/ServOMorph/creation_llm/train.py), `evaluate()` réinitialise le générateur à zéro et moyenne des lots de même taille ; [model.py](/D:/ServOMorph/creation_llm/model.py), `GPT.forward()`, calcule une CE moyenne. À mêmes cibles et fenêtres :
   \[
   \mathrm{BPB}=\mathrm{NLL}\,N/(B\ln 2).
   \]
   Le classement des checkpoints reste identique. Il faut sommer les pertes et les octets correspondants, **pas moyenner sans pondération les BPB des documents**.

2. **Les octets ne sont pas directement disponibles dans les `.bin`.** `MixedSource.batch()` prélève des fenêtres pouvant couper des documents et des caractères UTF-8. Décoder chaque token séparément peut introduire des caractères de remplacement ; compter leur encodage UTF-8 serait faux. Préférer un jeu documentaire brut dédié et conserver la correspondance texte–cibles.

3. **Le SFT inclut des cibles de structure.** [prepare_sft.py](/D:/ServOMorph/creation_llm/prepare_sft.py), `build_example()`, supervise les réponses **et EOT** ; `PackedData.batch()` décale les cibles ; `masked_loss_sum()` ignore −100. Une BPB de contenu doit séparer la perte EOT de celle des octets textuels. Diviser toute la perte SFT par les seuls octets de réponse exige une convention explicitement nommée.

4. **`critere_absent` n’est pas un diagnostic causal.** [bench_assistant.py](/D:/ServOMorph/creation_llm/bench_assistant.py), `score()`, constate qu’aucune expression régulière attendue ne correspond. Cela inclut potentiellement une paraphrase correcte. Les 139 occurrences sont confirmées, mais ne prouvent pas 139 connaissances manquantes.

5. **Les paramètres et états mémoire sont correctement calculés.** Avec embeddings liés :
   \[
   24\times12\times1024^2+32768\times1024=335\,544\,320.
   \]
   Doubler le vocabulaire ajoute 33 554 432 paramètres ; poids, gradients et deux moments FP32 ajoutent **512 Mio**, hors temporaires. `build_optimizer()` affecte bien les embeddings à AdamW.

6. **Le risque `uint16` apparaît dès la conversion chat.** Un vocabulaire de base de 65 536 entrées tient exactement, mais les deux tokens ajoutés par [convert_chat.py](/D:/ServOMorph/creation_llm/convert_chat.py), `build_chat_tokenizer()`, dépasseraient la limite. `CHAT_VOCAB=32832` et les formats de préparation devraient aussi être adaptés.

7. **La marge matérielle est faible.** `cap_vram()` utilise des **Gio** : plafond interne 6,1 Gio ; réservation du lanceur 6,5. Le pic historique de 6,04 laisse environ **61 Mio**, pas 1 Go de marge utilisable sous ce plafond. Les 113 000 × 65 536 tokens représentent **13,60 jours à 6 300 tok/s**, hors interruptions ; 12 jours correspond à un débit supérieur.

**Adaptations proposées — toutes des hypothèses locales**

Les gains directs des audits sont **0 point de qualité et 0 % d’accélération du modèle**, puisque les poids et l’entraînement restent inchangés. Le bénéfice décisionnel est inconnu. Les durées ci-dessous sont des **estimations de planification**, pas des mesures ; les seuils sont proposés avant expérience.

**A — Audit CPU commun : fidélité, compression et conservation des exemples SFT**

- **Mécanisme :** séparer économie de tokens, intégrité du texte et quantité d’information réellement conservée sous 1 024 tokens.
- **Ancrages :** [prepare_corpus.py](/D:/ServOMorph/creation_llm/prepare_corpus.py), `train_tokenizer()`, `write_batch()`, `encode_sources()` ; `prepare_sft.prepare()`, `fitting_turns()`, `build_example()` ; `convert_chat.build_chat_tokenizer()`.
- **Expérience minimale :** 1 Mio de textes complets par source, tenus hors apprentissage du candidat ; tokenizer actuel et candidat existant. Ajouter 1 000 conversations de développement et 200 cas accents, apostrophes, nombres, noms rares et Unicode.
- **Mesures :** tokens/octet, coût uniforme `log2(V) × tokens/octet`, fidélité aller-retour, débit CPU, conversations rejetées et tours supprimés. Mesurer par source avant toute agrégation.
- **Coût estimé :** 5–30 min CPU pour l’encodage et les contrôles, hors apprentissage d’un candidat ; RAM plafonnée à 2 Gio ; **VRAM +0 Mio**. Effet direct sur le débit 369M : **0 %** ; contention CPU/disque possible.
- **Succès :** 0 altération non autorisée, 0 dépassement d’identifiant, résultats reproductibles ; pour promouvoir le candidat, ≥10 % de tokens en moins, sans hausse >1 point des conversations rejetées/tronquées ni régression >2 % de tokens/octet sur une source.
- **Échec :** une violation d’intégrité élimine le candidat ; compression insuffisante ou résultats contradictoires arrêtent sa promotion, sans invalider l’audit.
- **SFT / omissions :** intérêt concret si davantage de tours ou de preuves tiennent dans la fenêtre ; aucun lien causal avec les omissions n’est encore démontré.
- **Compatibilité :** immédiate en petits lots CPU. Arrêt si débit du run baisse >1 % sur deux fenêtres comparables de 15 min, comme prévu au cycle 2.
- **Sources :** [TokEval](https://arxiv.org/abs/2608.18062), [définitions et outils](https://github.com/cimeister/tokenizer-intrinsic-evals).

**B — Contrat de mesure BPB, sans instrumenter le run en cours**

- **Mécanisme :** conserver explicitement somme des pertes, nombre de cibles, octets textuels et pertes des tokens spéciaux ; figer les documents et la politique de contexte.
- **Ancrages :** `GPT.forward()`, `train.evaluate()`, `train_sft.masked_loss_sum()` et `evaluate()`. Un futur évaluateur externe devrait réutiliser ces conventions ; il n’existe pas encore.
- **Expérience minimale CPU :** 100 documents courts et 20 exemples SFT synthétiques ; distribution uniforme calculée analytiquement, puis petit GPT 2 × 64, contexte 128. Vérifier masques, EOT, décalage des cibles et frontières UTF-8.
- **Coût estimé :** 5–30 min CPU, **VRAM +0 Mio**, débit 369M inchangé hors contention. Gain qualitatif direct : **0 point**.
- **Succès :** comptes entiers exacts ; erreur relative ≤`1e-6` sur l’identité algébrique en FP64 et ≤`1e-5` sur les réductions FP32 ; invariance à l’ordre des documents et au regroupement des lots.
- **Échec :** un octet mal attribué, une cible masquée comptée ou un changement de classement à cibles fixes bloque l’usage comparatif.
- **SFT / omissions :** améliore l’interprétation des pertes ; ne remplace jamais le benchmark de réponses.
- **Compatibilité :** totale comme contrôle CPU séparé. Une BPB du 369M nécessiterait ses pertes réelles ; le test uniforme ne les fournit pas.
- **Sources :** [formalisation BPB](https://arxiv.org/html/2611.04215v1), [protocole local H-Net](/D:/ServOMorph/creation_llm/rd/recherches/h-net/rapport.md).

**C — Relier fragmentation, pertes et omissions sémantiques**

- **Mécanisme :** vérifier si fragmentation ou longueur explique une difficulté après contrôle de la catégorie, puis tester si la perte apporte un signal supplémentaire.
- **Ancrages :** `bench_assistant.score()`, `rescore()`, `summarize()` ; `prepare_sft.build_example()` ; `train_sft.masked_loss_sum()`. Les sorties actuelles du benchmark **ne contiennent pas de perte par réponse**.
- **Expérience minimale :** réutiliser les 300 questions et sorties existantes ; annoter les 139 occurrences signalées et 30 réussites témoins. Regrouper toutes les graines d’une question dans la même partition ; réserver un tiers des familles au test.
- **Coût estimé :** 5–30 min CPU, 4–8 h de relecture ; **VRAM +0 Mio**, débit 369M inchangé hors contention. Gain direct : **0 point**.
- **Succès de l’audit :** 100 % des cas retenus annotés, 0 fuite de famille, désaccord de double annotation ≤10 % sur 30 cas ; sinon arbitrage avant analyse.
- **Succès d’un indicateur prédictif :** sur test réservé, amélioration d’AUROC ≥0,05 contre catégorie + longueur, avec intervalle apparié à 95 % excluant zéro. Sinon : indicateur exploratoire, pas substitut au benchmark.
- **Complément GPU facultatif après la base :** mesurer les pertes de références sur un pilote de 30 questions via `gpu_task.main()`, réservation 6,5, plafond interne 6,1 Gio, budget maximal 15 min. Prévoir 20 passages de chauffe puis 100 mesurés ; débit de scoring inconnu, **6 300 tok/s n’est pas une estimation valide pour cette tâche**.
- **Surcoût estimé du reporting seul :** +0–1 Mio au-delà du scoring dense ; débit du même évaluateur attendu à ±1 %, à vérifier. Le pic total de l’évaluation reste à mesurer.
- **SFT / omissions :** intérêt prioritaire pour distinguer omission réelle, paraphrase et distraction ; une corrélation ne prouverait toujours pas que changer le tokenizer corrige le défaut.
- **Sources :** [TokEval](https://arxiv.org/abs/2608.18062), [programme A1](/D:/ServOMorph/creation_llm/rd/programme_cycle2.md), [bilan assistant](/D:/ServOMorph/creation_llm/bilan_assistant_v2.md).

**D — Comparaison entraînée, seulement après présélection et cycle 2 prioritaire**

- **Mécanisme :** comparer deux systèmes à mêmes textes, puis à temps égal ; contrôler séparément l’avantage de contexte obtenu par compression.
- **Ancrages :** `prepare_corpus.train_tokenizer()` / `encode_sources()`, `GPTConfig`, `train.train()` / `MixedSource.batch()` / `evaluate()`, puis conversion chat et SFT.
- **Correction du rapport :** « six entraînements » couvre deux tokenizers × trois graines pour **un régime**. Deux régimes distincts peuvent demander douze runs ; des checkpoints communs ne suffisent que si parcours et ordonnanceurs permettent réellement les deux lectures.
- **Expérience minimale :** candidat 32 768 contre actuel, témoin 12 × 512, une graine pilote ; contrôle documentaire commun, mêmes frontières, mêmes proportions **d’octets** par source. `MixedSource` pondère actuellement des fenêtres de tokens : garder ses seuls coefficients ne garantit pas le même mélange textuel.
- **GPU court initial :** faisabilité et débit seulement, ≤15 min via `gpu_task.py`, après la base. Ce créneau ne valide aucune qualité.
- **Coût 369M estimé à vocabulaire et formes identiques :** **+0 Mio structurel**, environ **0 % de variation tok/s** ; repère 6 300 tok/s, à re-mesurer. Si le nombre de tokens baisse de 10 %, temps théorique à texte égal −10 %, uniquement si le débit reste constant.
- **Budget complet :** avec `D` octets par bras et `qA,qB` en octets/s, trois graines coûtent `3D/qA + 3D/qB`, plus compilation, préparation, évaluations et SFT. Durée absolue et gain de qualité : **inconnus**.
- **Succès proposé :** ≥10 % de temps total économisé ; non-infériorité assistant à −2 points, aucune catégorie à moins de −3 points, hausse des omissions sémantiques ≤1 point ; confirmation sur trois graines et intervalles appariés.
- **Échec :** plafond 6,1 Gio dépassé, avantage absent, garde-fou violé ou incertitude trop grande. Un succès du petit modèle autorise un essai futur, pas une modification du checkpoint actuel.
- **Sources :** [Tao et al.](https://arxiv.org/abs/2407.13623), [artefacts contrôlés TokEval](https://huggingface.co/cmeister/tokenizer-lm-ablations).

**Idées inadaptables ou conclusions à rejeter**

| Idée | Motif |
|---|---|
| Choisir 26 977 comme vocabulaire optimal | Minimum d’un modèle uniforme sur un corpus précis ; aucune prescription d’apprentissage. |
| Remplacer le tokenizer pendant le run ou uniquement au SFT | Les identifiants structurent embeddings, sorties et corpus encodés ; transfert non étudié ici. |
| Passer directement à 64k | +512 Mio d’états estimés, plus logits et temporaires ; dépasse largement la marge historique et casse les identifiants chat `uint16`. Débit fiable non estimable avant adaptation mémoire. |
| Déduire une qualité supérieure d’une BPB ou perte normalisée plus faible | Ces mesures ne suffisent pas à établir la complétude des réponses. |
| Abandonner BPB | Elle conserve un objet utile : coût de codage sous protocole explicite. |
| Ajouter H-Net ou MoE pour appliquer cet article | Aucune nécessité architecturale ; coûts et effets causaux distincts. |
| Assimiler −10 % de tokens à −10 % de temps garanti | Valable seulement à débit constant, sans changement de vocabulaire, de préparation ou de contexte. |

**Articulation avec les décisions existantes**

| Document | Complète, remplace ou contredit |
|---|---|
| [Programme cycle 2](/D:/ServOMorph/creation_llm/rd/programme_cycle2.md) | **Complète A1/A3** par A–C ; conserve base → CCE → SFT témoin → RAFT/H2 → STP. D reste optionnel après ces priorités. |
| [Bilan cycle 1](/D:/ServOMorph/creation_llm/rd/bilan_cycle1.md) | **Conforte** l’audit sémantique : 49/90 → 71/90 avec passages relus, contre 54/90 en RAG hybride à trois passages. Aucun indice isolant la tokenisation comme cause principale. |
| [Roadmap 369M](/D:/ServOMorph/creation_llm/roadmap_modele_369m.md) | **Conserve** le run, ses gates et le SFT en deux étapes ; complète l’évaluation. Rectifie seulement les repères de durée et marge mémoire. |
| [Pôle mathématiques I6](/D:/ServOMorph/creation_llm/rd/pole_mathematiques.md:99) | **Remplace** « sans biais » par une comparaison du coût de codage sous protocole commun. Deux tokenizers seuls ne donnent pas deux BPB de modèles entraînés. |
| [Sélection](/D:/ServOMorph/creation_llm/rd/selection.md) et [SuperBPE](/D:/ServOMorph/creation_llm/rd/veille_entrainement.md:194) | **Complète** le filtre CPU existant ; compression seule insuffisante pour sélectionner un nouveau run. |
| [H-Net](/D:/ServOMorph/creation_llm/rd/recherches/h-net/rapport.md) et [LexisLM](/D:/ServOMorph/creation_llm/rd/recherches/lexislm/rapport.md) | **Fusionne** les audits déjà proposés ; conserve les contrôles de contexte et l’évaluation assistant. Évite un programme concurrent. |
| Veilles inférence et MoE | Aucune modification technique motivée ; seule la discipline de comparaison s’applique. |

À INTÉGRER — renforcer les audits et les critères de comparaison, en conservant le tokenizer et le pré-entraînement actuels.