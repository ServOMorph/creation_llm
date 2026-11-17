# Bilan du cycle 1 de la cellule R&D

Date : 2026-11-17. Clos sur décision de l'utilisateur.

## Expériences menées (assistant v2, 90 items de faits ou 45 de consignes, 3 graines, mêmes items et graines que la référence)

| Expérience | Résultat | Verdict |
|---|---|---|
| RAG, passage choisi par motif (`rag_exp1.py`) | 48/90 contre 49/90 sans passage ; 3 passages BM25 : 28/90 | Non concluant : passages souvent hors sujet |
| RAG, passages relus à la main (`rag_exp1b.py`) | 71/90 contre 49/90 (31 gagnés, 9 perdus) | v2 sait lire un passage pertinent |
| Recherche sur CPU (`rag_exp1c_recherche.py`, `rag_exp1d_dense.py`) | Rang 1 contenant la réponse : 15/30 (BM25, tous passages) ; 19/30 (BM25, 3 premiers passages par article) ; 22/30 (hybride BM25 + e5-small) ; environ 18/30 pertinents à la relecture | Gain de recherche réel |
| RAG hybride de bout en bout (`rag_exp1e.py`) | 1 passage 47/90 ; 3 passages 54/90 (17 gagnés, 12 perdus) | Pas de gain hors bruit |
| CFG (`cfg_exp2.py`) | Consignes 18/18/16 sur 45 pour gamma 1/1,25/1,5 ; faits 49/50/54 sur 90 | Critère non atteint |
| TTT-NN (`ttt_exp3.py`) | 49/46/37/25 sur 90 pour lr 0/1e-5/3e-5/1e-4 ; boucles 1/7/26/41 | Échec, dégrade |

## Enseignements

- Le défaut `critere_absent` n'est pas une incapacité à lire : avec un passage pertinent, v2 gagne 22 items sur 90. Avec des passages imparfaits, il se laisse distraire.
- Environ 8 questions sur 30 (calcul, grammaire) n'ont pas de réponse dans Vikidia : un RAG Vikidia ne peut pas les couvrir.

## Reporté au cycle suivant

- RAFT (SFT à livre ouvert avec distracteurs) : seule suite cohérente des pistes à documents ; plusieurs heures de GPU.
- Expérience 4 (refroidissement final et EMA) : en fin de pré-entraînement du 369M.
- Expérience 5 (ResFormer sur témoin 12 × 512) : 12 à 21 h de GPU estimées, non mesurées.
- Fenêtres du 369M non alignées sur les débuts de documents et sans masque inter-documents : défaut confirmé, non corrigé.

## Incident

Le 2026-11-17 à 14:04, le test `rag_exp1e.py` a tourné sans jeton GPU pendant que vertia le détenait : la chaîne de lancement a confondu une pause demandée par vertia avec une sauvegarde. Règle pour les cycles suivants : vérifier le détenteur du jeton (`vram_lock.py status`) avant tout lancement GPU.
