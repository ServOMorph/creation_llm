# Sélection — pôle critique (rd-critique)

Date : 2026-11-17. Aucun GPU utilisé. Aucun fichier existant modifié. Seul fichier écrit : ce document.

Lus : `rd/README.md`, `rd/veille_inference.md`, `rd/veille_entrainement.md`, `rd/veille_moe_chine.md`, `rd/pole_mathematiques.md`, `model.py`, `train.py`, `run_369m.py`, `bilan_assistant_v2.md`, `checkpoints_369m/log.csv` (lu jusqu'au pas 22 420), `checkpoints_v9/log.csv` (lu jusqu'au pas 91 500).
Non lus : `run.py`, `vram_guard.py`, `bench_assistant.py`, `bench_assistant_results/`, `rd/experiences/`, `run_369m.out`.
Pages ouvertes avec WebFetch (la lecture passe par un résumé automatique : un chiffre n'est compté comme « présent » que s'il est cité mot pour mot ou ligne de tableau dans la version HTML ; sinon « absent de la page ouverte »).

Étiquettes utilisées :
- ÉTABLI : publié, chiffre confirmé à la source, et reproduction tierce vue dans une source ouverte.
- PRÉPUBLICATION : arXiv ou publié sans reproduction tierce vue (un résultat publié non reproduit est donc PRÉPUBLICATION dans ce document, et non ÉTABLI).
- HYPOTHÈSE : raisonnement ou arithmétique de la cellule.
- non sourcé : aucune référence retrouvée.

---

## 0. Erreurs relevées dans les rapports précédents

### Veille inférence
1. Constat 3 : « une seule auteure » est faux. La page 2603.11513 nomme une première auteure et « 2 other authors ». Les valeurs 10,0 % (Qwen2.5-1.5B), 12,8 % (3B) et 14,6 % (7B) sont confirmées en HTML ; 0,0 % (360M) et 100 % des échecs hors sujet aussi.
2. Piste 3 : l'article cité pour « +9,09 contre +17,77 » (arXiv 2603.06604) est « Know When You're Wrong » (calibration de confiance, Qwen3-4B). Ces chiffres n'y figurent pas. TARG n'a pas été ouvert : « publié TMLR 2026 » est non sourcé.
3. Piste 8 (LMLM) : « publié ICLR 2026 » non confirmé, la page est un arXiv v3 (octobre 2025) sans conférence. Les chiffres du tableau 2, le corpus de 3 Md tokens et les 54,6 M de triplets sont confirmés (HTML).
4. Piste 4 (CAD) : « publié NAACL 2024 » non confirmé, la page arXiv ne nomme aucune conférence. Le gain de 14,3 % est confirmé (résumé : « 14.3% gain for LLaMA », tâches de résumé). Le texte cite « GPT », non « GPT-Neo ».
5. Constat 4 (Pleias-RAG) : « dégradation négligeable en français » non confirmé. La page donne une fourchette de 10 à 30-35 % pour quatre langues ensemble (français, espagnol, italien, allemand). Les scores sont dans des figures non lues. Le second modèle est 1B dans le résumé et 1,2B dans les résultats.
6. Piste 5 (Integrative Decoding) : ICLR 2025 et les trois gains (+11,2 / +15,4 / +8,5) confirmés. La liste des modèles et le plus petit modèle testé ne figurent pas dans la page ouverte.
7. Piste 6 (CFG) : « équivalent à deux fois les paramètres » et « 75 % GPT4All » confirmés. Gamma et plus petit modèle absents de la page.
8. Piste 9 (TTT-NN) : les gains chiffrés ne sont pas dans les pages ouvertes. La reproduction (arXiv 2511.16691) est confirmée.
9. Constat 5 (« To Memorize or to Retrieve ») : OLMo-2 30M-3B, 100 Md tokens, k = 5, passages de 900 tokens, Qwen3-Embedding-8B confirmés en HTML. Le rapport écrit « améliore la perplexité » ; la page parle de performance. Mineur.

### Veille entraînement
9. Constat « run_369m.out : pause au pas 10 235 » : état daté. Le journal `checkpoints_369m/log.csv` atteint le pas 22 420 à la lecture. Ce chiffre doit être remplacé.
10. Piste 2 (CWD) : ICLR 2026, et les chiffres 2,51 → 2,42 et 3,0136 → 3,0059, ne figurent pas dans la page ouverte (arXiv abs). Non confirmé.
11. Piste 7 (NorMuon) : « ICML 2026 » non confirmé (aucune conférence sur la page). Les chiffres 21,74 % et 11,31 % sont confirmés (préprint).
12. Piste 8 (TOP) : tableau 2 confirmé en HTML (NTP 30,34 → TOP 28,76 ; SciQ 74,90 → 79,80 ; TriviaQA 4,93 → 4,37). « ICML 2026 » non confirmé : la page ne donne que le mot-clé ICML.
13. Piste 3 (ResFormer) : 16,11 % de paramètres et 20,3 % de données confirmés (résumé v5). « Record 9 » non vérifié.
14. Piste 4 : MiniPLM confirmé (ICLR 2025, tous les chiffres du tableau 1 et du tableau 3 en HTML). L'élagage par perplexité (référence 125M, 1,45x, +2,04) n'est pas dans MiniPLM : il vient d'un autre article (arXiv 2405.20541), non ouvert.
15. Piste 11 (Engram) : « DeepSeek » non confirmé (21 auteurs, aucune affiliation). MMLU +3,4 et ARC-Challenge +3,7 à 27B confirmés dans le résumé.
16. Piste 13 : « 700 essais » absent du classement nanochat (non sourcé). Le classement donne 5 à 7 répétitions par run.
17. Piste 9 : « speedrun record 57 » non vérifié.
18. Piste 10 (BeyondWeb) : 50,4 % contre 46,2 %, 45,5 %, +1,8 et +0,4 confirmés en HTML. « 40 à 50 % » n'est pas une plage de la source (40 % de synthétique dans une configuration, mélange 50/50 dans une autre).
19. Record 92 : « 65 B parameters » et 84,6 M lignes confirmés dans le classement du dépôt.

### Veille MoE
20. MoE-T : « attention 37,7M » est faux. Attention d'un bloc de 512 : 4 × 512² = 1,05 M, soit 12,6 M pour 12 couches. 37,7 M est le dense témoin entier (attention + MLP). Le total de 130 M et le « actif 54,5 M » sont justes, mais le calcul écrit (37,7 + 25,2 + 16,8) ne donne pas 54,5. La conclusion tient, le calcul écrit doit être corrigé.
21. MoEUT : « perplexité légèrement meilleure » non confirmé. La page parle de BLiMP et PIQA, et de « significantly less compute and memory », ce qui est confirmé.
22. Moonlight : l'arXiv 2502.16982 est « Muon is Scalable for LLM Training » ; Moonlight est le modèle qui en découle. Affiliation non lue.

### Pôle mathématiques
23. « Lr 0,000305 vérifié contre le log » : exact (recalcul manuel du cosinus au pas 89 000 : 0,000305, comme le log). Le « calcul 0,000309 » du rapport est une erreur d'arithmétique de la cellule ; la conclusion ne change pas.
24. I2 : « κ ≈ 0,57 » est l'estimation de la v9, non du 369M. Forme linéaire : κ × 0,1 = 0,057, et non 0,047.
25. I3 : la notion de bruit de gradient de McCandlish et al. (arXiv 1812.06162) est confirmée (« largest useful batch size »), mais le terme « B* » n'apparaît pas dans la page.
26. Affirmation « 2,850 contre 2,942 au même nombre de tokens » : chiffres confirmés, voir section 2. La comparaison n'est pas à modèle, schedule ou jeu de validation identiques.

---

## 1. Vérifications sur le code et les journaux

- `run_369m.py` et `train.py` : le rapport d'entraînement est exact. `MixedSource.batch` tire une position uniforme dans chaque fichier (`torch.randint`), et `train_starts = None if isinstance(train_data, MixedSource)` désactive `align_docs`. ARGS de `run_369m.py` ne contient pas `--align-docs`. Les fenêtres ne sont donc pas alignées sur les débuts de documents.
- Angle mort non signalé par les rapports : `model.py` accepte `doc_ids` (masque inter-documents, `forward`, lignes 102-107), mais `train.py` ne le passe jamais. L'attention traverse donc les frontières de documents. Ni l'alignement ni le masque ne sont en place.
- `train.py`, ligne 87 : Muon est appelé avec `adjust_lr_fn="match_rms_adamw"` (déjà en place). Cosinus vers 0,1 × lr (lignes 93-98). Clip 1,0. Warmup 1000 pas (run_369m).
- `save_checkpoint` écrase toujours `ckpt.pt`. Aucune moyenne de checkpoints successifs n'est possible sans modifier `train.py` (ajout de copies ou d'un EMA).
- Déjà présent dans `model.py` : RoPE, RMSNorm, QK-Norm, ReLU², soft-cap 30, embeddings liés (via `--tie-embeddings` de `run_369m.py`), recalcul des activations (`--grad-checkpoint`), cache KV. Absents : soft-cap 15, WSD, résidu de valeurs, EMA, MoE, prédiction multi-tokens, masque inter-documents.
- Batch effectif 369M : 2 × 32 × 1024 = 65 536 tokens par pas (lu dans `run_369m.py`). 113 000 pas = 7,4 Md tokens.
- Journal 369M (lu) : pas 21 000, val 2,8495 ; pas 22 000, val 2,8441 ; pas 22 250, val 2,8500 ; débit 5 100 à 6 600 tok/s (moyenne lue environ 6 000 à 6 300). Le lr du pas 22 250 se recalcule à 0,002766 avec la formule de `get_lr` ; le log donne 0,002767 (conforme).
- Journal v9 (lu) : pas 21 000, val 2,9417 ; pas 91 500, val 2,5445 (valeur finale de la v9 confirmée).
- Comparaison à 21 000 pas : 2,8495 (369M) contre 2,9417 (v9), écart 0,092. Les deux journaux sont à 21 000 pas. Le nombre de tokens par pas de la v9 n'a pas été lu dans sa config : il est supposé égal à 65 536, cohérent avec 6,0 Md tokens pour 91 500 pas (91 500 × 65 536 = 6,00 Md). Limites : modèles de taille différente (336M contre 192M), schedules différents (lr au pas 21 000 : 0,002793 pour le 369M, 0,002688 pour la v9), jeu de validation identique non vérifié.
- Répartition des données du 369M (table de couverture du pôle mathématiques) : recalcul des passes (0,58 × 7,4 Md / 4,33 Md = 0,99 ; 0,15 × 7,4 / 0,37 = 3,0 ; 0,02 × 7,4 / 0,054 = 2,7) : cohérent.

---

## 2. Verdicts par piste

Légende : retenue (chiffre clé confirmé, applicable à 336M sur 8 Go, falsifiable à coût raisonnable) ; à tester (conditions non remplies, ou test nécessaire avant décision) ; rejetée (motif indiqué).

### 2.1 Veille inférence

| # | Piste | Verdict | Motif | Étiquette |
|---|---|---|---|---|
| 1 | Diagnostic à passage oracle, RAG BM25 sans entraînement | retenue | Coût minime. Mesure directement le plafond du RAG et la capacité de lecture. Constat 3 confirmé (0,0 % à 360M, 100 % d'échecs hors sujet). | PRÉPUBLICATION |
| 2 | SFT à livre ouvert avec distracteurs (RAFT) | à tester | Chiffres RAFT confirmés en HTML (35,28 contre 4,41 ; 73,30 contre 71,6 ; 74,00 contre 42,59 ; 4 distracteurs ; 40 à 100 %). Préprint sur LLaMA2-7B, aucun résultat sous 1B. Données françaises : SQuAD-fr, PIAF, FQuAD non ouvertes ; licences non sourcées. | PRÉPUBLICATION |
| 3 | Récupération sélective par marge | à tester (test AUROC seul) | Source citée ne porte pas les chiffres (erreur 2). TARG non ouvert. Le test AUROC sur 90 items est peu coûteux. | non sourcé (chiffres), HYPOTHÈSE (gain) |
| 4 | Décodage contrastif au contexte (CAD, AdaCAD) | à tester | +14,3 % en résumé confirmé ; conférence non confirmée ; AdaCAD non ouvert ; gain en QA non confirmé. Deux passes avant : coût GPU × 2 à la génération. | PRÉPUBLICATION |
| 5 | Vote entre N échantillons | à tester (gate sans GPU d'abord) | Integrative Decoding ICLR 2025 et gains confirmés. Reproduction non vue. Le gate (cohérence entre les 3 graines existantes) ne coûte pas de GPU. | PRÉPUBLICATION |
| 6 | Guidage sans classifieur sur la consigne (CFG) | à tester | « Taille double » et 75 % confirmés, mais gamma et plus petit modèle absents ; effet du soft-cap inconnu. | PRÉPUBLICATION |
| 7 | Arrêts durs pour les contraintes de longueur | rejetée | Optimise le score du benchmark, non le modèle (le rapport le dit lui-même). Hors programme. | HYPOTHÈSE |
| 8 | Mémoire factuelle externalisée (LMLM) | rejetée pour le 369M ; réplique réduite à tester (basse priorité) | Chiffres confirmés (tableau 2, 3 Md tokens, 54,6 M triplets). Conférence non confirmée. Annotation du corpus non chiffrée ; nouveau pré-entraînement de 12 jours. | PRÉPUBLICATION |
| 9 | Entraînement au moment du test sur voisins (TTT-NN) | à tester (priorité moyenne) | ICLR (page). Reproduction confirmée (arXiv 2511.16691 : baisse de perplexité sur GPT-2 117M et 774M). Gains chiffrés non lus. Transfert à l'assistant : non sourcé. | ÉTABLI (qualitatif, perplexité) ; transfert HYPOTHÈSE |
| 10 | kNN-LM sur magasin restreint | à tester (basse priorité) | ICLR 2020 et 15,79 confirmés. Reproduction non vue. Magasin vikidia ≈ 3,2 Go en RAM (arithmétique, HYPOTHÈSE) ; wiki_fr hors de portée (≈ 121 Go). | PRÉPUBLICATION |
| 11 | Décodage contrastif expert/amateur (369M contre v9) | à tester (basse priorité, après SFT) | ACL 2023 et exemples OPT-13B / OPT-125M confirmés. Le rapport de taille (≈ 100) n'est pas celui du projet (≈ 2). Objectif de texte ouvert, non factuel. | PRÉPUBLICATION |
| 12 | Quantification ; ternaire ; QAT | rejetée (ternaire et QAT) ; PTQ 4 bits seulement pour un déploiement CPU | ParetoQ confirmé (54,8 / 55,0 / 54,7 / 52,8 / 51,4). BitNet 2B4T confirmé (0,4 Go, 29 ms contre 65 ms, 54,19 contre 55,23, 4 000 Md tokens). QAT ne réduit pas la VRAM d'entraînement (HYPOTHÈSE de la cellule, méthode non lue). | PRÉPUBLICATION |
| 13 | Sortie anticipée et auto-spéculation (LayerSkip) | rejetée | Accélérations confirmées (ACL 2024). La vitesse d'inférence n'est pas le problème mesuré. Exige un réentraînement. | PRÉPUBLICATION |
| — | Écartées : min-p, DoLa, PRM, profondeur récurrente, mémoires hiérarchiques, élagage, LoRA-MoE | rejetées | Sources non ouvertes ici, ou non sourcé pour le LoRA-MoE et l'élagage à cette taille. Rejet retenu pour l'échelle. | non sourcé / PRÉPUBLICATION |
| — | Encodeur dense pour la recherche | à tester (conditionnel) | Seulement si BM25 échoue (piste 1). Chiffres français non ouverts. | non sourcé |

### 2.2 Veille entraînement

| # | Piste | Verdict | Motif | Étiquette |
|---|---|---|---|---|
| 1 | Moyenne de poids (EMA) en fin de run | à tester (dans l'expérience E3) | Technique présente dans le record modded-nanogpt (« Tail weight averaging (EMA) with a raised LR floor »), ampleur non chiffrée. Demande une modification de `train.py` (checkpoint écrasé). Coût VRAM nul, 1,35 Go de RAM. | PRÉPUBLICATION ; gain HYPOTHÈSE |
| 2 | Cautious Weight Decay | à tester | Effet qualitatif confirmé (résumé : « consistently improves »). Chiffres non confirmés. Ampleur à 336M : 0,008 nat selon le rapport (chiffre non vérifié). `torch.optim.Muon` applique sa propre décroissance : non contrôlé. | PRÉPUBLICATION |
| 3 | Résidu de valeurs (ResFormer) | retenue (témoin, prochain pré-entraînement) | 16,11 % de paramètres et 20,3 % de données confirmés (résumé v5). Reprise « value embeddings » présente dans le speedrun. Incompatible avec le checkpoint actuel. Variante embeddings de valeurs à éviter sous 8 Go (0,4 Go). | PRÉPUBLICATION |
| 4 | Tri du corpus (MiniPLM, élagage par perplexité) | à tester (basse priorité) | MiniPLM ICLR 2025 et tous les chiffres confirmés, mais sur l'anglais, teacher de 1,8B. Un teacher français tenant dans 8 Go : non sourcé. Scorer 6,65 Md tokens : coût non mesuré. Élagage (2405.20541) non ouvert. Risque de biais contre les textes simples. | PRÉPUBLICATION |
| 5 | Planification (lr floor, longueur croissante, soft-cap 15) | à tester (témoin) | Records du speedrun présents dans le dépôt. Gains isolés non chiffrés. Soft-cap 15 changerait le modèle (actuellement 30). Longueur croissante : coût nul. | PRÉPUBLICATION (records) ; gain HYPOTHÈSE |
| 6 | Connexions denses dynamiques (MUDD) | à tester (témoin, prochain pré-entraînement) | ICML 2025 confirmé ; 1,8 à 2,4 × de calcul et +0,23 % de paramètres confirmés. Stocke 24 sorties de bloc (≈ 0,1 Go). Incompatible avec le recalcul tel qu'il est codé. | PRÉPUBLICATION |
| 7 | NorMuon et Polar Express | à tester | NorMuon : 21,74 % et 11,31 % confirmés (préprint, aucune conférence ; « ICML 2026 » non confirmé). Polar Express : nom et bf16 confirmés, aucune conférence. Variante absente de `torch.optim.Muon` non contrôlée : Muon maison requis. | PRÉPUBLICATION |
| 8 | Prédiction de l'ordre des tokens (TOP) | rejetée pour le 369M ; à tester sur témoin | Tableau confirmé. +33,5 M de paramètres (≈ 0,4 Go). Pic mesuré 6,04 Go avec cap 6,1 : marge insuffisante. | PRÉPUBLICATION |
| 9 | Recalcul supprimé ou états en bf16 | rejetée pour les poids ; états Muon à tester (faible) | Risque sur les poids démontré par l'arithmétique (bf16 : 8 bits de mantisse). Le bilan mesure 7,41 Go sans recalcul, au-delà du cap de 6,1 Go : aucune marge. Record 57 non vérifié. | HYPOTHÈSE |
| 10 | Données synthétiques (BeyondWeb) | à tester (gate sur le débit de génération) | Chiffres confirmés en HTML (50,4 contre 46,2 ; 45,5 ; +1,8 ; +0,4). Débit de génération local non mesuré : 1 Md tokens hors de portée si le débit reste de l'ordre de 100 tok/s (HYPOTHÈSE). | PRÉPUBLICATION |
| 11 | Table de bigrammes et trigrammes hachés (Engram) | à tester (basse priorité) | Résumé confirmé (27B, MMLU +3,4, ARC +3,7). DeepSeek non confirmé. La table du speedrun (84,6 M lignes, 65 B paramètres, 8 GPU) n'est pas transposable. Table en RAM de 2^20 lignes : débit à mesurer. | PRÉPUBLICATION |
| 12 | Distillation de logits en pré-entraînement | rejetée | Teacher et élève doivent partager le tokenizer (aucun modèle public ne le fait). Stockage ≈ 1 To. Le rapport lui-même juge la piste non pertinente. | PRÉPUBLICATION (2410.16215 non ouvert ici) |
| 13 | Recherche automatique sur modèle réduit (nanochat) | à tester (après le run) | Classement confirmé (3,04 h → 2,91 h FP8 → 1,80 h → 1,65 h, backout et smear). « 700 essais » non sourcé. Coût GPU élevé en concurrence avec le pré-entraînement. | PRÉPUBLICATION |
| 14 | Tokenisation SuperBPE | à tester (gate CPU) | COLM 2025 confirmé ; −33 % de tokens à 200k sur l'anglais, +4,0 % et +8,2 % confirmés. Aucun résultat à 32k ni en français. Nouveau tokenizer = nouveau run. | PRÉPUBLICATION |
| 15 | FP8 et quantification à l'entraînement | à tester (gate de 10 minutes) | Gain FP8 nanochat de 4,3 % confirmé (sur H100, non transposable). Support Ada/RTX 4060 : non établi. | PRÉPUBLICATION (H100) ; HYPOTHÈSE (RTX 4060) |
| — | Curriculum par difficulté | rejetée | Résultats négatifs du défi BabyLM (résumé lu). | PRÉPUBLICATION |
| — | Record 92 du speedrun (cumul) | rejetée comme piste | Cumul non isolé, 8 × H100. Chiffres de la table confirmés. | ÉTABLI (dépôt) pour les chiffres ; non transposable |

### 2.3 Veille MoE et apports chinois

| # | Piste | Verdict | Motif | Étiquette |
|---|---|---|---|---|
| 1 | Planification WSD | à tester (témoin) | Résumé et « 1,2B et 2,4B » confirmés. Tableau de pertes absent de la page. Gain non chiffré. | PRÉPUBLICATION |
| 2 | Témoin MoE-T et upcycling | à tester (basse priorité) | DeepSeekMoE 2B/16B/145B confirmé (ni conférence ni reproduction). Upcycling Nemotron-4 15B : 67,6 contre 65,3 confirmé (préprint). MoE-T 54 M : non sourcé. Calcul corrigé (erreur 20). | PRÉPUBLICATION |
| 3 | Mélange de têtes (MoH) | à tester (témoin, basse priorité) | ICML 2025 confirmé ; +2,4 points avec 75 % des têtes confirmé sur LLaMA3-8B (anglais). Réécriture de l'attention. Pas de résultat à petite taille. | PRÉPUBLICATION |
| 4 | Couches partagées (MoEUT) | à tester (témoin) | NeurIPS 2024 confirmé. Moins de calcul et de mémoire confirmé ; perplexité non (erreur 21). Levier VRAM réel, mais nouvelle architecture. | PRÉPUBLICATION |
| 5 | Adaptateurs experts pour la SFT (MoLE, LoRA-MoE) | à tester (SFT) | Résumé seulement. Aucun chiffre lu. Gain non sourcé. | PRÉPUBLICATION (résumé) ; gain non sourcé |
| 6 | Équilibrage sans perte auxiliaire | rejetée pour l'instant | Sans objet sans MoE. Article non ouvert ici. | PRÉPUBLICATION |
| 7 | Experts déchargés en RAM (CoSMoEs) | rejetée | Tailles et tableaux non lus. Débit estimé à la moitié du dense (HYPOTHÈSE) ; débit PCIe non sourcé. | PRÉPUBLICATION (résumé) |
| 8 | Muon et mise à l'échelle des mises à jour (Moonlight) | déjà présente | `train.py` ligne 87. Efficacité ≈ 2 × sur AdamW confirmée (résumé, auteurs). | PRÉPUBLICATION |
| 9 | Renvois vers les autres rapports | sans objet | Traités dans leurs sections. | — |
| — | MoE-A (939 M, 335 M actifs) sur 8 Go | rejetée | Partie fixe ≈ 939 M × 17,3 o = 16,2 Go (arithmétique de la cellule, HYPOTHÈSE de proportionnalité). | HYPOTHÈSE |
| — | MoE-B (16 experts) | rejetée | Même stockage que MoE-A : 16,2 Go. | HYPOTHÈSE |
| — | DeepSeek-V3 (671 B) | rejetée | Échelle non transposable. | ÉTABLI (chiffres de la page) ; transposition rejetée |
| — | Experts gelés sans upcycling | rejetée | Pas de capacité ajoutée. | HYPOTHÈSE |
| — | Modèles OpenBMB au-delà de 2,4B | rejetée | Hors budget VRAM. | HYPOTHÈSE |

### 2.4 Pôle mathématiques

| # | Idée | Verdict | Motif | Étiquette |
|---|---|---|---|---|
| I1 | Décision d'arrêt par extrapolation à 40-60 % du run | retenue comme outil de décision (coût nul) | Ne prouve aucun gain. Backtest sur une seule courbe (v9). Choix de p sur 6 valeurs : erreur optimiste. | HYPOTHÈSE |
| I2 | Refroidissement final plus long (terme κ√lr) | à tester (test de fin de run, E3) | Calcul κ√0,1 = 0,18 exact. Mais κ vient de la v9 (non du 369M), et la forme linéaire donne 0,057 (non 0,047). Estimation à 2 runs. | HYPOTHÈSE |
| I3 | Taille de lot par rapport au batch critique | rejetée (non concluante) | Estimation dominée par le bruit (erreur-type 0,0069 pour une norme de 0,0007). Concept de McCandlish confirmé (1812.06162), terme B* absent de la page. | PRÉPUBLICATION (concept) |
| I4 | Fenêtres alignées sur les débuts de documents | à tester (basse priorité) | Gain borné à 2,3 % (mesure de la cellule). `align_docs` inopérant avec `MixedSource` (confirmé). Ajouter le masque inter-documents (`doc_ids`, déjà dans `model.py`). | HYPOTHÈSE (gain) ; ÉTABLI (mesure locale) |
| I5 | Pénalité de répétition des petites sources | à tester | Passes simulées (3,0 et 2,7) arithmétiquement justes. Aucune référence : non sourcé. | non sourcé |
| I6 | Bits par octet comme critère de comparaison | retenue (méthode de mesure) | Calcul exact : 3,67 bits / 4,45 octets = 0,82 bit par octet. Pas un gain. | HYPOTHÈSE (méthode) |

---

## 3. Classement des cinq expériences à lancer en premier

Contraintes communes : le 369M est prioritaire (VRAM cap 6,1 Go, pic mesuré 6,04 Go, marge d'environ 1 Go selon le bilan). Toute expérience GPU concurrente exige une pause et un jeton VertIA. Tous les débits et coûts ci-dessous sont des HYPOTHÈSES : aucun débit d'inférence ou de témoin n'a été mesuré.
Point de comparaison de référence (lu dans `bilan_assistant_v2.md`) : v2 (`checkpoints_sft_v2/ckpt.pt`, 192M) : Faits 49/90, Consignes 18/45, 300 cas au total, 3 graines.
Note de bruit : sur 90 items, l'écart-type binomial est ≈ 4,7 items ; sur 45, ≈ 3,3. Les seuils de 3 items sur 45 des pistes du rapport sont donc sous le bruit. Les tests doivent être appariés (mêmes items, mêmes graines) et les critères relus en conséquence.

### Rang 1 — Diagnostic RAG à passage oracle (inférence, v2)
- Piste source : inférence 1 (retenue).
- Étape 0, sans GPU : répartir les 139 `critere_absent` par catégorie dans `bench_assistant_results/` (non lu). Construire l'index BM25 sur vikidia (passages de 100 à 150 tokens, CPU).
- Protocole : 90 items « Faits » × 3 graines × 3 conditions : sans passage (référence), passage oracle (premier passage contenant l'expression `ok` de l'item), 3 meilleurs passages BM25. Le passage est injecté dans le tour utilisateur (pas de rôle « document » dans le format).
- Run ou référence : v2, référence 49/90.
- Coût GPU (HYPOTHÈSE) : 810 générations de 300 tokens au plus sur 192M : 15 à 30 min ; VRAM inférieure à 2 Go ; index CPU de quelques minutes.
- Critère d'échec : couverture oracle inférieure à 60/90 (étape 0, CPU) : abandonner les pistes RAG 1 à 4. Condition oracle inférieure à la référence de 9 items : lecture de contexte non acquise, passer à RAFT (inférence 2).
- Risque : l'expression `ok` peut sélectionner un passage qui contient le mot sans répondre.

### Rang 2 — Guidage sans classifieur sur la consigne (CFG, v2)
- Piste source : inférence 6 (à tester).
- Protocole : 135 items (90 Faits, 45 Consignes) × 3 graines × γ ∈ {1 (témoin), 1,25, 1,5}. Logits = sans + γ × (avec − sans). Consigne « sans » : même gabarit de chat, consigne vide (choix de la cellule, non sourcé). Appariement par item.
- Run ou référence : v2, référence 49/90 et 18/45.
- Coût GPU (HYPOTHÈSE) : 1 215 générations, deux passes avant par token : 2 à 4 h.
- Critère d'échec : gain inférieur à 5 items sur 45 Consignes (seuil de bruit) pour tout γ, ou perte supérieure à 5 items hors Consignes. Le seuil du rapport (3 items) est sous le bruit.
- Risque : soft-cap 30 modifie l'échelle de la différence ; texte dégradé à γ élevé.

### Rang 3 — Adaptation au moment du test sur voisins (TTT-NN, v2)
- Piste source : inférence 9 (à tester ; ÉTABLI qualitativement).
- Protocole : 90 items Faits ; pour chacun, 20 passages BM25 de vikidia (index du rang 1) ; un pas AdamW par passage sur une copie des poids ; restauration des poids après chaque item ; taux d'apprentissage parmi 3 valeurs ; 3 graines.
- Run ou référence : v2 sans passage (49/90) et v2 avec passages sans adaptation (rang 1, condition 3).
- Coût GPU (HYPOTHÈSE) : 20 passes avant-arrière de 1 024 tokens par item sur 192M : 3 à 6 s par item ; 810 items : 0,7 à 1,4 h.
- Critère d'échec : gain inférieur à 5 items sur 90, ou hausse de `boucle` ou de réponses vides.
- Risque : un pas de gradient sur du texte brut peut casser l'alignement SFT ; latence incompatible avec l'usage interactif.

### Rang 4 — Refroidissement final prolongé et EMA (369M, fin de run)
- Pistes sources : math I2 et veille entraînement 1 (à tester).
- Précondition : fin du pré-entraînement (pas 113 000, lr = 3e-4). Nécessite un ajout à `train.py` (flag de lr final et sauvegarde des poids EMA), non fait.
- Protocole : depuis le checkpoint final, deux bras de 2 000 pas (≈ 131 M tokens chacun) : A, lr constant à 3e-4 (contrôle, même nombre de tokens) ; B, décroissance cosinus de 3e-4 à 3e-5. EMA des poids (décroissance par pas : choix de la cellule à fixer) maintenue dans les deux bras. Validation sur les 40 lots fixes tous les 250 pas, points et EMA comparés.
- Run ou référence : checkpoint final du 369M et bras A (le bras A évite de confondre le gain du refroidissement avec celui de 2 000 pas de calcul supplémentaires).
- Coût GPU (HYPOTHÈSE) : 2 × 2 000 pas × 65 536 tokens ≈ 262 M tokens, soit environ 12 h à 6 000 tok/s (débit lu dans le journal, 5 100 à 6 600 tok/s).
- Critère d'échec : écart B − A inférieur à 0,005 nat (bruit de validation, lu sur les points 22 000 à 22 250 du journal) : abandonner le refroidissement. EMA inférieure à 0,005 nat au point final : abandonner l'EMA.
- Risque : résultat disponible seulement en fin de run ; le gain prédit (0,047 à 0,18 nat) vient d'un ajustement à deux runs.

### Rang 5 — Résidu de valeurs (ResFormer) sur témoin 12 × 512
- Piste source : veille entraînement 3 (retenue pour témoin).
- Protocole : témoin 12 × 512 (54,5 M de paramètres au total, lecture du rapport et calcul vérifié : 37,7 M de blocs + 16,8 M d'embeddings), même tokenizer 32k et même mélange de données. Deux bras (baseline et résidu de valeurs), 3 graines, même budget de tokens (300 M proposés, à fixer). Perte de validation sur le même jeu.
- Run ou référence : bras baseline du même témoin (pas le 369M, dont la reprise est incompatible avec ce changement).
- Coût GPU (HYPOTHÈSE de débit, 25 000 à 40 000 tok/s, non mesuré) : 6 bras, 2 à 3,5 h chacun : 12 à 21 h. Pause du 369M obligatoire (VRAM).
- Critère d'échec : écart inférieur à l'écart-type entre graines, ou inférieur à 0,01 nat (seuil du rapport).
- Risque : le chiffre de 16,11 % concerne un autre contexte d'échelle (résumé, non lu en texte intégral) ; sa transposition au témoin est une HYPOTHÈSE.

Ordre d'exécution : rangs 1 et 2 (inférence, sans réentraînement) puis rang 3 (dépend de l'index du rang 1). Les rangs 4 et 5 ne peuvent pas être lancés pendant le pré-entraînement sans pause : le rang 4 n'est possible qu'en fin de run, le rang 5 demande une pause.

---

## 4. Non vérifié et angles morts

- Pages ouvertes par résumé automatique : les chiffres validés sont ceux cités mot pour mot. Pages non ouvertes dans cette passe : TARG, AdaCAD (2409.07394), StructureBench, DoLa, MTEB-French, CWD (HTML), 2405.20541, 2410.16215, 2503.20083, 2502.11895, 2509.22935, 2408.15664, 2503.00245, 2506.13681, 2502.06703, 2504.04718, 2502.05171, 2510.02375, 2502.03460, 2405.20468, 2511.09803, 2507.17702, 2605.27358, ALM (NeurIPS 2025), Pleias (figures).
- Non lus : `run.py` et `vram_guard.py` (mapping des arguments `--tie-embeddings` et VRAM), `bench_assistant.py` et ses résultats (répartition des 139 `critere_absent`), `rd/experiences/`, `run_369m.out`, config de la v9 (tokens par pas).
- Torch 2.11 : présence de NorMuon, Polar Express, CWD dans `torch.optim.Muon` : non contrôlée.
- Licences et tailles de SQuAD-fr, PIAF, FQuAD : non sourcées.
- Débits de génération et d'entraînement des témoins, coûts VRAM des nouvelles pistes : aucune mesure ; tous les coûts GPU ci-dessus sont des HYPOTHÈSES.
- Les seuils d'échec de plusieurs rapports (3 items) sont sous le bruit binomial : à relever avant décision.
- Le 369M est à 2,8495 au pas 21 000 ; cette lecture, comme les autres, est datée du journal lu.
