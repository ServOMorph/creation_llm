Contexte du projet (dépôt courant, lire les fichiers cités avant de conclure) :
- LLM français from scratch sur une seule RTX 4060 8 Go (VRAM partagée, plafond 6,5 Go), 48 Go RAM, Ryzen 7 5700X.
- Modèle maison `model.py` : GPT decoder-only, RoPE, RMSNorm, QK-Norm, ReLU², soft-cap des logits, embeddings liés, recalcul des activations ; optimiseur Muon + AdamW (`train.py`).
- En cours : pré-entraînement 369M (24 x 1024, contexte 1024, vocab 32 768) sur environ 7,4 milliards de tokens (fw2hq 0,58, wiki_fr 0,25, lecarnet 0,15, vikidia 0,02), environ 6 300 tok/s, 12 jours de calcul (`run_369m.py`, `roadmap_modele_369m.md`).
- Puis SFT assistant (`train_sft.py`, `bilan_assistant_v2.md` : assistant v2 à 49,7 %, défaut dominant « critère absent » = information manquante).
- Cellule R&D : charte `rd/README.md`, cycle 1 clos `rd/bilan_cycle1.md`, sélection `rd/selection.md`, veilles déjà faites `rd/veille_entrainement.md`, `rd/veille_inference.md`, `rd/veille_moe_chine.md`, `rd/pole_mathematiques.md`.
Règles :
- Écrire en français. Ne modifier aucun fichier du dépôt : ta réponse finale est le livrable complet, en Markdown.
- Règle de preuve de la cellule : pour chaque proposition, mécanisme, gain attendu chiffré, coût (VRAM, temps), expérience minimale, critère d'échec, sources avec URL.
- Distinguer explicitement : résultat établi (publié, reproduit), prépublication, hypothèse. Ne jamais inventer de chiffre, d'auteur ou d'URL ; si une information n'est pas trouvée, l'écrire.
- Aucune information personnelle dans les requêtes web.
