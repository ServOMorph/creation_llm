# Backlog — creation_llm

- [dette] Détection du 369M par fichier PID écrit par `run_369m.py` au lieu du motif de ligne de commande (point 12 d'`ameliorations.md`, reporté après le run)
  - fait quand: `gpu_task.py` lit un fichier PID et les tests passent.
  - réf: `gpu_task.py`, `run_369m.py`, `ameliorations.md`
- [piste] H-Net : prototype séparé, après les priorités du cycle 2 (verdict À TESTER)
  - fait quand: décision de prototype prise ou piste écartée.
  - réf: `rd/recherches/h-net/analyse.md`
- [piste] LFM3-1.7B : référence française, puis enseignant RAFT conditionnel (verdict À TESTER)
  - fait quand: comparaison au 369M faite ou piste écartée.
  - réf: `rd/recherches/lfm3/analyse.md`
- [piste] NCM (Next-Concept Modeling) et LexisLM : à surveiller, aucun changement du run ni du SFT
  - fait quand: nouvelle publication ou reproduction qui change le verdict.
  - réf: `rd/recherches/ncm/analyse.md`, `rd/recherches/lexislm/analyse.md`
- [dette] Fins de ligne : `rd/agents_codex/file.sh` en CRLF (risque d'échec sous bash) et 88 fichiers en CRLF malgré `.gitattributes` ; renormalisation (`git add --renormalize`) non décidée
  - fait quand: `file.sh` s'exécute sous bash et décision de renormalisation prise.
  - réf: `rd/agents_codex/file.sh`, `.gitattributes`
- [dette] `classify_failures` n'enregistre que le dernier tour des cas multi-tours (question et réponse) : sortie d'audit sans dialogue complet
  - fait quand: les enregistrements d'`c2_audit_v2.json` portent tous les tours et les tests passent.
  - réf: `rd/experiences/c2_audit.py`, `rd/experiences/relecture_a1/gen.py`
