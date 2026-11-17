# Veille inférence, compression et usage — pôle `rd-veille-inference`

Date : 2026-11-16. Périmètre : méthodes 2024-2026 applicables à l'inférence d'un GPT maison de 24 x 1024 (environ 336M de paramètres, contexte 1 024, BPE 32k) sur RTX 4060 8 Go.
Aucun calcul GPU n'a été lancé. Aucune expérience ci-dessous n'a été exécutée : tous les gains « attendus sur ce projet » sont des hypothèses.

## Conventions

- **ÉTABLI** : publié avec comité de lecture et reproduction retrouvée. Quand la reproduction n'a pas été retrouvée, c'est écrit : « ÉTABLI (publié, reproduction non retrouvée) ».
- **PRÉPUBLICATION** : arXiv ou rapport technique, sans comité de lecture retrouvé.
- **HYPOTHÈSE** : raisonnement de la cellule, non sourcé.
- « Source ouverte » : la page primaire a été ouverte pendant cette veille et le chiffre y a été lu. « Extrait de recherche » : chiffre lu seulement dans un résultat de moteur de recherche, à revérifier.
- Les coûts en VRAM et en temps sont des estimations arithmétiques (HYPOTHÈSE), sauf mention contraire. Aucun débit d'inférence du 369M n'a été mesuré.

## Constats préalables

1. **Défaut à traiter** (lu dans `bilan_assistant_v2.md`) : `critere_absent` représente 139 occurrences pour 151 échecs sur 300 cas. Le scoring (`bench_assistant.py`) cherche une expression régulière dans la dernière réponse : l'échec signifie que le mot attendu n'apparaît pas. La répartition de ces 139 occurrences par catégorie n'a pas été lue (fichiers de `bench_assistant_results/` non ouverts) : à établir avant de chiffrer un plafond de gain.
2. **La VRAM n'est pas la contrainte à l'inférence** (HYPOTHÈSE, arithmétique) : 336M de paramètres en bf16 pèsent environ 0,67 Go ; le cache KV d'une séquence de 1 024 tokens pèse environ 24 x 2 x 1 024 x 1 024 x 2 octets, soit 0,1 Go. La quantification n'apporte donc rien en VRAM à l'inférence sur ce modèle. La contrainte de 8 Go porte sur l'entraînement, que la quantification n'allège pas (voir piste 12).
3. **Avertissement central sur le RAG à cette taille** : une étude de mars 2026 mesure qu'un modèle de 360M généraliste (SmolLM2-360M) n'exploite pas un passage fourni, même exact : 0,0 % d'exact match avec passage oracle sur les questions qu'il ne sait pas résoudre seul, et 100 % de ses échecs sont de la génération hors sujet. Les modèles de 1,5B à 7B plafonnent à 10,0 %, 12,8 % et 14,6 % dans la même condition. PRÉPUBLICATION, un seul auteur, modèles quantifiés en 4 bits, source ouverte : https://arxiv.org/html/2603.11513. Conséquence : brancher un moteur de recherche sur l'assistant v2 sans réentraînement est probablement inutile ; la lecture de contexte doit être apprise en SFT (pistes 1 et 2).
4. **Contre-exemple** : la famille Pleias-RAG (350M et 1B, entraînés spécifiquement pour le RAG avec citations, contexte 4 096) revendique des scores supérieurs aux modèles de moins de 4B sur HotPotQA et 2Wiki, et une dégradation « négligeable » en français, là où les autres modèles de moins de 1B perdent 10 à 35 %. PRÉPUBLICATION, évaluation faite par les auteurs avec un juge LLM (Gemma 3 12B), scores chiffrés présents seulement dans des figures, non lus. Source ouverte : https://arxiv.org/html/2504.18225v1.
5. **Loi d'échelle** : « To Memorize or to Retrieve » (avril 2026, modèles OLMo-2 de 30M à 3B, jusqu'à 100 milliards de tokens) conclut que la récupération améliore la perplexité à toutes les tailles, que les plus petits modèles en profitent le plus par token récupéré, et que le gain sature vers 3B. Le protocole utilise un encodeur de 8B de paramètres (Qwen3-Embedding-8B), des passages de 900 tokens et k = 5 : non transposable tel quel à un contexte de 1 024. PRÉPUBLICATION, source ouverte : https://arxiv.org/html/2604.00715v1.

## Classement (rapport gain / coût décroissant)

| Rang | Piste | Cible | Coût | Statut de la preuve |
|---|---|---|---|---|
| 1 | Diagnostic à passage oracle et RAG lexical sans entraînement | critere_absent | très faible | prépublications contradictoires |
| 2 | SFT « à livre ouvert » avec distracteurs (type RAFT) | critere_absent | une SFT | prépublication |
| 3 | Récupération sélective par marge de confiance | réponses connues détruites | très faible | prépublication, une publication TMLR |
| 4 | Décodage contrastif au contexte (CAD, AdaCAD) | fidélité au passage | 2 passes avant | publié |
| 5 | Vote entre N échantillons | faits | N passes en lot | publié sur grands modèles |
| 6 | Guidage sans classifieur sur la consigne (CFG) | consignes | 2 passes avant | prépublication |
| 7 | Arrêts durs pour les contraintes de longueur | trop_long, trop_de_phrases | nul | hypothèse |
| 8 | Mémoire factuelle externalisée dès le pré-entraînement (LMLM) | critere_absent | un pré-entraînement | publié ICLR 2026 |
| 9 | Entraînement au moment du test sur voisins (TTT-NN) | connaissances | 20 pas de gradient par requête | publié et reproduit |
| 10 | kNN-LM sur magasin restreint | perplexité, faits | index de plusieurs Go | publié et reproduit |
| 11 | Décodage contrastif expert / amateur (369M contre v9) | qualité du texte | 2 modèles | publié sur grands modèles |
| 12 | Quantification : 4 bits par QAT, ternaire déconseillé | déploiement CPU | QAT de 10 à 30 milliards de tokens | publié |
| 13 | Sortie anticipée et auto-spéculation (LayerSkip) | vitesse | modification de l'entraînement | publié |

---

## Piste 1 — Diagnostic à passage oracle, puis RAG lexical sans entraînement

- **Mécanisme.** Indexer en BM25 (CPU) les corpus déjà présents localement (vikidia, puis wiki_fr), découpés en passages courts (100 à 150 tokens, le contexte de 1 024 l'impose). Pour chaque question « Faits » du benchmark, construire automatiquement un passage oracle : le premier passage récupéré qui satisfait l'expression `ok` de l'item. Injecter le passage dans le tour utilisateur et mesurer le modèle existant, sans réentraînement.
- **Gain attendu.** HYPOTHÈSE : nul ou négatif sur l'assistant v2, d'après le constat 3 (0,0 % avec oracle pour un 360M généraliste). L'intérêt est de mesurer deux bornes : (a) le taux de questions pour lesquelles un passage oracle existe dans le corpus local (plafond du RAG) ; (b) la capacité de lecture actuelle du modèle.
- **Coût.** Index BM25 de vikidia (0,05 milliard de tokens) : quelques minutes de CPU, moins de 1 Go de RAM (HYPOTHÈSE). Génération : 90 items x 3 graines, quelques minutes de GPU sous jeton. VRAM inférieure à 2 Go.
- **Expérience minimale.** Trois conditions sur les 90 items « Faits », 3 graines : sans passage (référence 49/90 pour la v2), passage oracle, 3 meilleurs passages BM25.
- **Critère d'échec.** Si moins de 60 des 90 questions ont un passage oracle dans vikidia + wiki_fr, le corpus local ne suffit pas comme magasin : abandonner les pistes 1 à 4 sous cette forme. Si la condition oracle dépasse la référence de moins de 9 items sur 90 (environ deux écarts-types binomiaux), la lecture de contexte n'est pas acquise : passer à la piste 2.
- **Risque.** L'oracle automatique par expression régulière peut sélectionner un passage qui contient le mot sans répondre à la question. Le format de chat actuel n'a pas de rôle « document » : le passage dans le tour utilisateur peut déclencher le défaut de recopie.
- **Sources.** https://arxiv.org/html/2603.11513 ; https://arxiv.org/html/2604.00715v1.

## Piste 2 — SFT « à livre ouvert » avec distracteurs (type RAFT)

- **Mécanisme.** Ajouter au jeu SFT des exemples (question, passages, réponse) où le passage utile est mêlé à des distracteurs, et où une fraction des exemples ne contient aucun passage utile. Le modèle apprend à extraire la réponse et à ignorer le bruit.
- **Gain publié.** RAFT sur LLaMA2-7B : HotPotQA 35,28 contre 4,41 pour un réglage fin classique + RAG ; PubMed 73,30 contre 71,6 ; HuggingFace 74,00 contre 42,59. Quatre distracteurs par exemple ; la part optimale d'exemples avec passage utile varie de 40 à 100 % selon le jeu. PRÉPUBLICATION, source ouverte : https://arxiv.org/html/2403.10131v2. Aucun résultat RAFT à moins de 1B n'a été trouvé ; Pleias-RAG-350M (constat 4) est le seul indice de faisabilité à cette taille, chiffres non lus.
- **Gain attendu ici.** HYPOTHÈSE : « Faits » de 49/90 à au moins 65/90 avec les 3 meilleurs passages BM25, si la piste 1 montre un plafond suffisant.
- **Coût.** Une SFT de l'ordre de celle de la v2 (2 606 pas ; durée non relue). Données françaises candidates, à vérifier avant usage : SQuAD-fr (environ 90 000 questions, CC BY 4.0), PIAF (environ 9 224 paires, licence non vérifiée), FQuAD (CC BY-NC-SA 3.0, usage non commercial) — extraits de recherche, fiches non ouvertes. Avec un contexte de 1 024, la limite pratique est de 3 à 4 passages de 150 tokens.
- **Expérience minimale.** SFT v2 + 10 000 à 20 000 exemples à livre ouvert (dont 30 % sans passage utile), puis les trois conditions de la piste 1 et le benchmark complet de 300 cas.
- **Critère d'échec.** Moins de 65/90 sur « Faits » avec les 3 meilleurs passages BM25, ou perte de plus de 5 items sur les 210 cas des autres catégories, ou baisse de la condition sans passage sous 45/90.
- **Risque.** Régression en livre fermé (le modèle attend un passage) ; hausse de la recopie, défaut que la v2 venait de diviser par 4 ; données traduites automatiquement de qualité inégale. La piste est à rejouer sur le 369M : un résultat négatif sur le 192M ne tranche pas pour le modèle plus grand.
- **Sources.** https://arxiv.org/html/2403.10131v2 ; https://arxiv.org/html/2504.18225v1 ; https://arxiv.org/html/2603.11513.

## Piste 3 — Récupération sélective par marge de confiance

- **Mécanisme.** Générer un court brouillon sans contexte, mesurer l'incertitude (écart entre les deux premiers logits, ou variance entre quelques échantillons), ne récupérer que si elle dépasse un seuil. Évite de dégrader les réponses que le modèle connaît déjà.
- **Gain publié.** L'injection de contexte détruit 42 à 100 % des réponses connues pour les modèles de 7B et moins (PRÉPUBLICATION, source ouverte : https://arxiv.org/abs/2603.11513). TARG (publié TMLR 2026 d'après l'extrait de recherche, non ouvert) rapporte que l'entropie sur-déclenche et que la marge top-1/top-2 et la variance entre échantillons discriminent mieux. Sur Qwen3-4B et TriviaQA, récupérer pour 25 % des requêtes donne +9,09 points contre +17,77 en récupérant toujours (extrait de recherche, arXiv 2603.06604, non ouvert).
- **Gain attendu ici.** HYPOTHÈSE : supprime la régression sur les questions déjà réussies sans passage ; aucun gain propre.
- **Coût.** Une génération courte supplémentaire par requête ; aucune VRAM additionnelle notable.
- **Expérience minimale.** Sur les 90 « Faits », tracer la précision sans passage en fonction de la marge moyenne des 10 premiers tokens ; vérifier que la marge sépare réussites et échecs (AUROC).
- **Critère d'échec.** AUROC inférieure à 0,65 : la confiance du modèle n'est pas exploitable, récupérer systématiquement.
- **Risque.** Un petit modèle peut être sûr de lui et faux ; le signal n'a pas été validé sous 1,5B dans les sources lues.
- **Sources.** https://arxiv.org/abs/2603.11513 ; https://arxiv.org/html/2511.09803v1 ; https://arxiv.org/pdf/2603.06604.

## Piste 4 — Décodage contrastif au contexte (CAD, AdaCAD)

- **Mécanisme.** Deux passes avant par token : avec et sans passage. Les logits finaux amplifient la différence, ce qui force l'usage du passage. AdaCAD règle l'intensité par la divergence de Jensen-Shannon entre les deux distributions, pour ne pas dégrader les cas sans conflit.
- **Gain publié.** CAD : +14,3 % sur les métriques de factualité en résumé pour LLaMA ; testé sur OPT, GPT-Neo, LLaMA, FLAN-T5 (publié NAACL 2024 d'après l'extrait de recherche ; page arXiv ouverte : https://arxiv.org/abs/2305.14739). AdaCAD : +14,21 points de précision en questions-réponses par rapport à un contrastif statique, +6,19 AlignScore en résumé (NAACL 2025, source ouverte : https://arxiv.org/abs/2409.07394). ÉTABLI (publié, reproduction non retrouvée).
- **Gain attendu ici.** HYPOTHÈSE : +3 à +8 items sur 90 « Faits » au-dessus de la piste 2, uniquement si le modèle lit déjà le contexte.
- **Coût.** Débit de génération divisé par 2 environ ; VRAM doublée pour le cache KV, soit +0,1 Go. Vingt à trente lignes dans `generate.py`.
- **Expérience minimale.** Modèle de la piste 2, 3 meilleurs passages BM25, alpha parmi 0 ; 0,5 ; 1, 3 graines.
- **Critère d'échec.** Gain inférieur à 3 items sur 90 pour tout alpha, ou hausse de `recopie` ou `boucle` de plus de 3 occurrences.
- **Risque.** Amplifie aussi les erreurs d'un mauvais passage ; texte dégradé à alpha élevé.
- **Sources.** https://arxiv.org/abs/2305.14739 ; https://arxiv.org/abs/2409.07394.

## Piste 5 — Vote entre N échantillons

- **Mécanisme.** Générer N réponses en lot, garder la plus consensuelle. `generate.py` contient déjà un `best_of` qui choisit par log-probabilité moyenne : la variante à tester est le choix par accord entre échantillons. Integrative Decoding va plus loin : chaque échantillon est replacé en préfixe et les prédictions sont agrégées à chaque pas.
- **Gain publié.** Integrative Decoding : TruthfulQA +11,2 %, Biographies +15,4 %, LongFact +8,5 %, sur LLaMA2/3, Mistral, Gemma2, Qwen2, GLM4 (ICLR 2025, source ouverte : https://arxiv.org/abs/2410.01556). Plus petit modèle testé non lu. ÉTABLI (publié, reproduction non retrouvée).
- **Gain attendu ici.** HYPOTHÈSE : +2 à +5 items sur 90 « Faits ». Le vote stabilise une connaissance présente mais instable ; il ne crée pas une connaissance absente. Si `critere_absent` vient d'un manque réel, le gain sera nul.
- **Coût.** N = 8 en lot : environ 0,8 Go de cache KV, débit par réponse divisé par un facteur inférieur à 8 grâce au lot (non mesuré).
- **Expérience minimale.** Sans GPU supplémentaire : les 3 graines déjà enregistrées par le benchmark donnent, par item, le nombre de graines réussies. Si la plupart des items sont à 0/3 ou 3/3, le vote ne peut rien. Ensuite seulement, N = 8 avec vote sur les « Faits ».
- **Critère d'échec.** Moins de 10 items « Faits » à 1/3 ou 2/3 sur les graines existantes : abandonner sans lancer de calcul.
- **Risque.** Le vote renforce une erreur majoritaire.
- **Sources.** https://arxiv.org/abs/2410.01556.

## Piste 6 — Guidage sans classifieur sur la consigne (CFG)

- **Mécanisme.** Deux passes avant : avec la consigne et sans elle (ou avec une consigne neutre). Logits = sans + gamma x (avec − sans), gamma autour de 1,5. Renforce l'adhésion à la consigne.
- **Gain publié.** Les auteurs annoncent des gains « équivalents à un modèle de taille double » sur Pythia, GPT-2 et LLaMA, et une préférence humaine de 75 % pour GPT4All avec CFG. PRÉPUBLICATION (2023), source ouverte : https://arxiv.org/abs/2306.17806. Plus petit modèle testé non lu.
- **Gain attendu ici.** HYPOTHÈSE : +3 à +6 items sur les 45 « Consignes » (18/45 pour la v2). Effet incertain sur les faits.
- **Coût.** Identique à la piste 4, et le code est commun (même combinaison de deux jeux de logits).
- **Expérience minimale.** Benchmark complet à gamma = 1 ; 1,25 ; 1,5 ; 2, 3 graines, sur la v2 puis sur le 369M.
- **Critère d'échec.** Gain inférieur à 3 items sur 45 en « Consignes » pour tout gamma, ou perte de plus de 3 items sur le reste.
- **Risque.** Texte dégradé à gamma élevé ; le soft-cap des logits à 30 du modèle modifie l'échelle de la différence, effet inconnu.
- **Sources.** https://arxiv.org/abs/2306.17806.

## Piste 7 — Arrêts durs pour les contraintes de longueur, sans décodage contraint par grammaire

- **Mécanisme.** Quand la consigne fixe un nombre de mots ou de phrases, l'appliquer dans la boucle de génération (arrêt à la fin de phrase). Ne pas contraindre le contenu par grammaire.
- **Résultat publié qui motive la prudence.** « The Constraint Tax » (mai 2026, 15 000 générations, Qwen2.5-0.5B, 1.5B, SmolLM2-1.7B) : le décodage contraint par schéma fait passer la validité de 61,5 % à 100,0 %, mais la précision de 19,7 % à 11,0 %. PRÉPUBLICATION, un seul auteur, source ouverte : https://arxiv.org/abs/2605.26128. StructureBench (IJCAI 2026, extrait de recherche, non ouvert) conclut dans le même sens.
- **Gain attendu ici.** Plafond de 5 occurrences sur 151 échecs (`trop_long` 3, `trop_de_phrases` 2, lus dans le bilan). HYPOTHÈSE : gain inférieur, une réponse tronquée pouvant perdre le critère attendu.
- **Coût.** Nul en calcul. Nécessite d'extraire la contrainte de la consigne, ce que le benchmark fournit (`max_mots`, `max_phrases`) mais pas un usage réel.
- **Expérience minimale.** Rejouer les items porteurs de `max_mots` ou `max_phrases` avec arrêt dur.
- **Critère d'échec.** Moins de 3 items récupérés.
- **Risque.** Optimisation du benchmark plutôt que du modèle : gain non transférable à l'usage réel. À faire en dernier, ou pas du tout.
- **Sources.** https://arxiv.org/abs/2605.26128 ; https://www.ijcai.org/proceedings/2026/267.

## Piste 8 — Mémoire factuelle externalisée dès le pré-entraînement (LMLM)

- **Mécanisme.** Annoter le corpus de pré-entraînement avec des triplets (entité, relation, valeur) ; pendant l'entraînement, le modèle émet une requête à la base (jetons spéciaux), la valeur renvoyée est insérée et exclue de la perte. Le modèle apprend à interroger au lieu de mémoriser. À l'inférence, la recherche se fait par similarité cosinus (all-MiniLM-L6-v2, seuil 0,6).
- **Gain publié, à l'échelle exacte du projet.** Modèles pré-entraînés depuis zéro sur environ 3 milliards de tokens de Wikipédia, contexte 1 024, base de 54,6 millions de triplets. LLaMA2-382M : FactScore 14,0 → 31,9 ; T-REx 52,0 → 58,1 ; PopQA 22,7 → 50,8. GPT2-355M : FactScore 14,4 → 23,9 ; PopQA 21,4 → 52,0. Publié ICLR 2026, source ouverte : https://arxiv.org/html/2505.15962v3. ÉTABLI (publié, reproduction non retrouvée). Évaluation en anglais uniquement.
- **Variante.** « Knowledgeless LM » (juillet 2026) : pré-entraîner sur un corpus aux entités nommées anonymisées ; jusqu'à 20 à 25 % de gain relatif avec des preuves imparfaites en contexte. PRÉPUBLICATION, source ouverte : https://arxiv.org/abs/2607.12831.
- **Gain attendu ici.** HYPOTHÈSE : le plus fort potentiel sur `critere_absent` de toute cette veille, mais hors de portée du pré-entraînement 369M en cours.
- **Coût.** Élevé : annoter 6,65 milliards de tokens en français exige un annotateur (les auteurs distillent GPT-4o en un petit modèle), puis un pré-entraînement complet (environ 12 jours d'après le bilan). L'annotation locale par un modèle Ollama est un chantier en soi, non chiffré.
- **Expérience minimale.** Réplique réduite : annoter vikidia (0,05 milliard de tokens), entraîner deux modèles de petite taille (ordre de 50M de paramètres) avec et sans base, comparer sur 200 questions factuelles tirées de vikidia.
- **Critère d'échec.** Gain inférieur à 10 points de précision factuelle sur la réplique réduite, ou taux d'annotations fausses supérieur à 20 % sur un échantillon de 200 triplets relus.
- **Risque.** Qualité de l'annotateur en français ; coût en tokens des requêtes à la base (les auteurs le signalent) ; ne couvre que les faits liés à des entités.
- **Sources.** https://arxiv.org/html/2505.15962v3 ; https://arxiv.org/abs/2607.12831.

## Piste 9 — Entraînement au moment du test sur voisins (TTT-NN)

- **Mécanisme.** Pour chaque requête, récupérer 20 passages voisins et appliquer un pas de gradient par passage sur une copie des poids, puis répondre. La connaissance entre par les poids et non par le contexte : contourne à la fois la limite de 1 024 tokens et l'incapacité de lecture du constat 3.
- **Gain publié.** 20 voisins, un pas chacun, réduisent fortement l'écart entre un petit GPT-2 et un GPT-Neo plus de 10 fois plus grand, sur plus de 20 tâches de The Pile (ICLR 2024, source ouverte : https://arxiv.org/abs/2305.18466). Un rapport de reproduction de novembre 2025 confirme la baisse de perplexité sur GPT-2 117M et 774M (source ouverte : https://arxiv.org/abs/2511.16691). ÉTABLI. Les gains chiffrés n'ont pas été lus (absents des résumés), et ils portent sur la perplexité, pas sur des réponses d'assistant.
- **Gain attendu ici.** HYPOTHÈSE : baisse de perplexité probable ; effet sur le benchmark d'assistant inconnu, car les voisins sont du texte brut et le modèle est en format de chat.
- **Coût.** 20 passes avant-arrière par requête. Sur le 369M avec recalcul des activations, de l'ordre de 6 Go de VRAM (pic mesuré à l'entraînement avec lot de 2, lu dans le bilan) et plusieurs secondes par requête (non mesuré). Ne pas mettre à jour les poids de façon persistante.
- **Expérience minimale.** Sur la v2 (192M) : 90 « Faits », 20 passages BM25 de vikidia, un pas AdamW chacun, taux d'apprentissage parmi 3 valeurs, poids restaurés après chaque requête.
- **Critère d'échec.** Gain inférieur à 5 items sur 90, ou dégradation du format de chat (hausse de `boucle` ou réponses vides).
- **Risque.** Un pas de gradient sur du texte brut peut casser l'alignement SFT ; latence incompatible avec un usage interactif.
- **Sources.** https://arxiv.org/abs/2305.18466 ; https://arxiv.org/abs/2511.16691.

## Piste 10 — kNN-LM sur magasin restreint

- **Mécanisme.** Stocker pour chaque token du corpus le vecteur de contexte de la dernière couche et le token suivant ; à l'inférence, interpoler la distribution du modèle avec celle des k plus proches voisins.
- **Gain publié.** Perplexité 15,79 sur WikiText-103, soit 2,9 points de mieux sans entraînement (ICLR 2020, source ouverte : https://arxiv.org/abs/1911.00172). Travaux d'efficacité (2021) : jusqu'à 6 fois plus rapide, magasin réductible de 95 % (extrait de recherche, https://arxiv.org/pdf/2109.04212). ÉTABLI. Aucun résultat 2025-2026 sur petits modèles n'a été trouvé.
- **Coût (HYPOTHÈSE, arithmétique).** Vikidia : 50 millions de clés x 1 024 dimensions en 16 bits = 102 Go bruts, soit environ 3,2 Go après quantification par produit à 64 octets. Wiki_fr (1,89 milliard de tokens) : environ 121 Go, au-delà des 48 Go de RAM. La construction exige une passe avant sur tout le magasin : environ 50 millions de tokens, quelques minutes de GPU pour vikidia. Recherche FAISS sur CPU à chaque token généré.
- **Gain attendu ici.** HYPOTHÈSE : baisse de perplexité sur du texte proche de vikidia ; gain sur l'assistant incertain, les clés étant calculées sur du texte brut et interrogées en format de chat.
- **Expérience minimale.** Magasin vikidia, mesure de perplexité sur la validation vikidia avec lambda parmi 0 ; 0,1 ; 0,25 ; puis 90 « Faits ».
- **Critère d'échec.** Baisse de perplexité inférieure à 5 % sur la validation vikidia, ou gain inférieur à 3 items sur 90 « Faits ».
- **Risque.** Le magasin doit être reconstruit à chaque nouveau point de contrôle ; latence par token.
- **Sources.** https://arxiv.org/abs/1911.00172 ; https://arxiv.org/pdf/2109.04212.

## Piste 11 — Décodage contrastif expert / amateur (369M contre v9 192M)

- **Mécanisme.** Soustraire aux log-probabilités de l'expert celles d'un modèle plus faible, sous contrainte de plausibilité. Le projet dispose d'un amateur naturel : la v9 de 192M, même tokenizer à vérifier (la v9 et le 369M utilisent-ils le même BPE 32k ? non relu).
- **Gain publié.** Texte de meilleure qualité que nucleus et top-k en évaluation automatique et humaine, avec OPT-13B comme expert et OPT-125M comme amateur (ACL 2023, source ouverte : https://arxiv.org/abs/2210.15097). ÉTABLI (publié, reproduction non retrouvée). L'article vise la génération ouverte, pas l'exactitude factuelle ; le rapport de taille y est de 100, contre moins de 2 ici.
- **Gain attendu ici.** HYPOTHÈSE faible : réduction des boucles et répétitions (8 occurrences en v2) ; effet sur `critere_absent` probablement nul.
- **Coût.** Deux modèles en mémoire : environ 1,1 Go de poids, débit réduit d'environ un tiers (non mesuré).
- **Expérience minimale.** Possible seulement après la SFT du 369M. Benchmark complet avec et sans contraste.
- **Critère d'échec.** Gain global inférieur à 5 items sur 300.
- **Risque.** Amateur trop proche de l'expert : la soustraction retire aussi le bon signal.
- **Sources.** https://arxiv.org/abs/2210.15097.

## Piste 12 — Quantification : 4 bits par QAT si déploiement CPU, ternaire déconseillé

- **Mécanisme.** Entraînement conscient de la quantification (QAT) en fin de parcours ; ternaire (1,58 bit) ou 2 à 4 bits.
- **Résultats publiés.** ParetoQ (NeurIPS 2025, source ouverte : https://arxiv.org/html/2502.02631v2), MobileLLM-350M, moyenne sur 8 tâches : précision pleine 54,8 ; 4 bits 55,0 ; 3 bits 54,7 ; 2 bits 52,8 ; ternaire 51,4. Le réglage fin sature vers 10 milliards de tokens pour 3 et 4 bits, 30 milliards pour 2 bits et moins ; partir d'un modèle en précision pleine bat l'entraînement quantifié depuis zéro. BitNet b1.58 2B4T (rapport technique, source ouverte : https://arxiv.org/html/2504.12285v2) : 0,4 Go, 29 ms par token sur CPU, moyenne 54,19 contre 55,23 pour Qwen2.5 1.5B, mais 4 000 milliards de tokens d'entraînement, et les auteurs écrivent que les GPU courants ne sont pas conçus pour ces modèles. « Compute-Optimal QAT » (ICLR 2026, source ouverte : https://arxiv.org/abs/2509.22935) : la part optimale de QAT croît avec le budget de calcul ; fusion du refroidissement du taux d'apprentissage et de la QAT.
- **Lecture pour ce projet.** Le ternaire coûte 3,4 points à 350M et demande 30 milliards de tokens de QAT, soit 4 fois le budget total du pré-entraînement (7,4 milliards) : à écarter. La QAT ne réduit pas la VRAM d'entraînement (poids latents en précision pleine, HYPOTHÈSE conforme aux méthodes citées), donc elle ne permet pas d'entraîner un modèle plus grand sur 8 Go. À l'inférence GPU, le modèle pèse 0,67 Go : aucun besoin. Seul cas utile : un déploiement CPU ou embarqué, où 4 bits est quasi sans perte.
- **Expérience minimale.** Quantification 4 bits après entraînement (sans QAT) du point de contrôle v2, mesure de la perplexité de validation et du benchmark sur CPU.
- **Critère d'échec.** Hausse de perplexité supérieure à 2 % ou perte de plus de 5 items sur 300 : il faudrait une QAT, hors budget.
- **Risque.** Travail sans effet sur le défaut dominant.
- **Sources.** https://arxiv.org/html/2502.02631v2 ; https://arxiv.org/html/2504.12285v2 ; https://arxiv.org/abs/2509.22935 ; https://arxiv.org/pdf/2407.09527.

## Piste 13 — Sortie anticipée et auto-spéculation (LayerSkip)

- **Mécanisme.** Entraîner avec abandon de couches croissant en profondeur et une perte de sortie anticipée partagée ; à l'inférence, les premières couches proposent des tokens que le modèle complet vérifie.
- **Gain publié.** Accélération jusqu'à 2,16 fois en résumé, 1,82 en code, 2,0 en analyse sémantique (ACL 2024, source ouverte : https://arxiv.org/abs/2404.16710). ÉTABLI (publié, intégration dans une bibliothèque tierce d'après un extrait de recherche). Gain de vitesse uniquement, aucune hausse de qualité.
- **Coût.** Exige de modifier l'entraînement (depuis zéro, en poursuite ou en réglage fin). Le pré-entraînement 369M est lancé sans ce dispositif ; l'introduire en SFT est possible selon les auteurs, effet non chiffré à cette taille.
- **Expérience minimale.** Sonde linéaire : appliquer `lm_head` à la sortie de chaque couche du modèle actuel et mesurer à quelle profondeur le token final est déjà prédit. Sans réentraînement.
- **Critère d'échec.** Moins de 50 % des tokens correctement prédits à la couche 12 sur 24 : pas de gain de vitesse à espérer sans réentraînement.
- **Risque.** La vitesse d'inférence n'est pas le problème mesuré du projet ; priorité basse.
- **Sources.** https://arxiv.org/abs/2404.16710 ; https://huggingface.co/blog/layerskip.

---

## Pistes examinées et écartées

| Piste | Raison | Source |
|---|---|---|
| Échantillonnage min-p | Une réanalyse de 2025 conclut qu'à effort de réglage égal, min-p ne surpasse ni top-p ni les autres échantillonneurs (GSM8K, 9 modèles) ; 1 comparaison humaine sur 12 significative. | https://arxiv.org/html/2506.13681v2 (ouverte) |
| DoLa (contraste entre couches) | Les auteurs rapportent un échec sur GPT2-Medium (335M) : pas assez de connaissances pour que les couches diffèrent. Taille identique à celle du projet. | https://arxiv.org/pdf/2309.03883 (extrait de recherche, non ouverte) |
| Recherche guidée par vérificateur (PRM) | Un modèle de 1B dépasse un 405B sur MATH-500, mais avec un modèle de récompense externe et sur des mathématiques uniquement ; aucun vérificateur français disponible. | https://arxiv.org/abs/2502.06703 (ouverte) ; https://arxiv.org/abs/2504.04718 (ouverte) |
| Profondeur récurrente, calcul adaptatif | Preuve de concept à 3,5B de paramètres et 800 milliards de tokens (extrait de recherche) : hors budget, et change l'architecture. | https://arxiv.org/abs/2502.05171 (non ouverte) |
| Mémoires paramétriques hiérarchiques | Un modèle de 160M avec 18M de mémoire tirée d'une banque de 4,6B équivaut à un modèle plus de 2 fois plus grand (ICLR 2026), mais à l'échelle de mille milliards de tokens. | https://arxiv.org/abs/2510.02375 (ouverte) |
| Élagage | Sans objet : le modèle est entraîné depuis zéro et tient en 0,67 Go ; entraîner plus petit est plus simple. Gain chiffré sur petits modèles : non sourcé. | https://arxiv.org/html/2502.03460v3 (non ouverte) |
| Mélange d'experts LoRA et routage | Aucun gain chiffré sur petit modèle retrouvé : non sourcé. | — |
| Encodeur dense pour la recherche | À n'envisager que si BM25 échoue (piste 1). multilingual-e5-small autour de 0,50 à 0,52 nDCG@10 en recherche française (extraits de recherche, non ouverts) ; aucune comparaison française avec BM25 retrouvée. | https://arxiv.org/html/2405.20468v2 (non ouverte) |

## Enchaînement recommandé

1. Sans GPU : compter les items instables entre graines (piste 5) et la répartition de `critere_absent` par catégorie.
2. Piste 1 (quelques minutes de GPU sous jeton) : elle décide du sort des pistes 2, 3 et 4.
3. Pistes 2, puis 4 et 6 (code commun), sur la v2 pour mise au point, puis sur le 369M après sa SFT.
4. Piste 8 seulement pour un prochain pré-entraînement, après la réplique réduite.

## Non vérifié

- Scores chiffrés de Pleias-RAG-350M (figures non lisibles) et nombre exact de paramètres.
- Gains chiffrés de TTT-NN et de sa reproduction ; plus petits modèles testés pour CFG et Integrative Decoding.
- Licences et tailles de FQuAD, PIAF et SQuAD-fr (extraits de recherche seulement).
- Pages primaires de TARG, AB-RAG, arXiv 2603.06604, StructureBench, DoLa, Geiping et al., MTEB-French.
- Une incohérence relevée dans ParetoQ : le texte donne 58,7 pour le modèle ternaire de 600M, la table 55,5.
- Tout coût en temps et en VRAM sur ce matériel : aucune mesure faite.
