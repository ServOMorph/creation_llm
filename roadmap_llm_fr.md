# Roadmap — LLM français from scratch

Matériel : RTX 4060 8 Go, 48 Go RAM, Ryzen 7 5700X. Installation sur D:.
Principe : v1 rapide et minimale, puis amélioration itérative (taille, données, qualité).
Exécution en autonomie : accord VRAM donné par l'utilisateur, enchaînement des phases sans checkpoint `/compact`.

## Phase 1 — Environnement et données [EN COURS]
- venv sur D:, PyTorch CUDA, tokenizers, datasets
- Corpus FR v1 : LeCarnet (histoires courtes FR, style TinyStories), fiche à vérifier avant usage
- Gate : `torch.cuda.is_available()` vrai, corpus nettoyé sur disque

## Phase 2 — Tokenizer [TODO]
- BPE entraîné sur le corpus, vocab ~8-16k
- Tests : aller-retour encode/decode, taux de compression
- Gate : tests passent

## Phase 3 — Modèle GPT (decoder-only, ~30-50M) [TODO]
- Implémentation from scratch : attention causale, RoPE, RMSNorm, QK-Norm, MLP ReLU² (briques modded-nanogpt)
- Tests : formes, masque causal, overfit sur un mini-batch
- Gate : tests passent, loss tend vers 0 sur un batch

## Phase 4 — Entraînement v1 [TODO]
- Boucle bf16, AdamW, cosine LR, checkpoints, reprise
- Entraînement quelques heures, suivi de la loss de validation
- Gate : loss de validation décroissante, checkpoint sauvegardé

## Phase 5 — Génération et évaluation v1 [TODO]
- `run.py` de génération (température, top-k/top-p)
- Évaluation : perplexité + benchmark de prompts verrouillés
- Gate : v1 génère du français lisible (sans garantie de cohérence)

## Phase 6 — Itérations modèle d'histoires (v2 à v8) [TODO]
- v1-v7 : 33,6M puis 97,5M, augmentation des prénoms, 6 époques cumulées ; v8 : extension de profondeur 12 -> 18 couches (LLaMA Pro, arXiv 2401.02415), présentations variées
- Évaluation : 3 jeux de 20 prompts x 3 graines, lecture avec grille fixe (erreur majeure / limite / mineure)
- Constat : ~40 % d'erreurs strictes sur des prompts nouveaux, contre ~85 % pour la référence LeCarnet-21M ; plafond lié à la pauvreté du corpus (2M histoires stéréotypées, 365M tokens répétés)
- Gate (décidé par l'utilisateur) : v8 retenue si au moins égale à v7 et meilleure que LeCarnet-21M sur le jeu scellé

## Phase 7 — Corpus français généraliste [TODO]
- Sources : FineWeb2-HQ fra_Latn (epfml, top 10 % qualité, 32M docs, colonne text seule), Wikipédia FR (wikimedia/wikipedia 20231101.fr, 2,56M articles), Vikidia FR et FR simple (FrancophonIA/Vikidia-EnFr), LeCarnet (histoires)
- Pipeline : téléchargement par lots, nettoyage, déduplication exacte, mélange pondéré, tokenisation en .bin
- Tokenizer BPE 32k entraîné sur le mélange (8k insuffisant hors histoires)
- Gate : >= 5 milliards de tokens prêts, tests tokenizer (aller-retour, compression)

## Phase 8 — Recette d'entraînement efficace [TODO]
- A/B courts (même budget de tokens) : AdamW vs Muon (torch.optim.Muon, matrices cachées) + AdamW (embeddings, tête) ; torch.compile si disponible sous Windows
- Fenêtres de contexte 1024, micro-batch adapté à ~4 Go de VRAM, reprise testée
- Gate : recette retenue sur loss de validation à budget égal

## Phase 9 — Modèle généraliste v9 (~150M, plusieurs jours) [TODO]
- Pré-entraînement sur 5-10 milliards de tokens, par tranches reprenables (pause VRAM possible à tout moment)
- Évaluation : perplexité par domaine, benchmarks FR (FrenchBench, CroissantLLM), jeux de prompts d'histoires
- Gate : meilleur que v8 sur les histoires nouvelles et perplexité FR générale en baisse continue

## Phase 10+ — Suite [TODO]
Pistes : annealing final sur données de haute qualité, SFT / forme assistant, contexte plus long, value embeddings / U-net (modded-nanogpt).
