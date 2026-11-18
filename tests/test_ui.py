import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ui import server  # noqa: E402


def test_training_reads_state_and_compares_v9(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PROGRESS", tmp_path / "p.json")
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "etat.json").write_text(json.dumps({"step": 500, "raison": "en_cours", "date": "x"}), encoding="utf-8")
    (ck / "log.csv").write_text("step,train_loss,val_loss,lr,tokens_per_s\n250,3.0,3.1,0.1,6000\n500,2.9,2.8,0.1,6200\n510,2.7,,0.1,6400\n")
    ref = tmp_path / "v9.csv"
    ref.write_text("step,train_loss,val_loss,lr,tokens_per_s\n500,3.0,2.95,0.1,13000\n")
    t = server.training(ck, max_steps=1000, ref_log=ref)
    assert t["step"] == 510 and t["pct"] == 51.0 and t["val"] == 2.8
    assert t["comparaison_v9"] == (500, 2.8, 2.95)
    assert t["rate_theorique"] == round(6200 * 3600 / 65536) and t["actif"]


def test_real_rate_needs_half_hour():
    assert server.real_rate([[0, 0], [600, 10]]) is None
    assert server.real_rate([[0, 0], [3600, 270]]) == 270


def test_record_progress_dedups(tmp_path):
    p = tmp_path / "p.json"
    server.record_progress(5, 100, p)
    server.record_progress(5, 200, p)
    assert server.record_progress(6, 300, p) == [[100, 5], [300, 6]]


def test_roadmaps_and_signals(tmp_path):
    (tmp_path / "roadmap_x.md").write_text("# X\n## Phase 1 — A [FAIT]\n## Phase 2 — B [EN COURS]\n", encoding="utf-8")
    r = server.roadmaps(tmp_path)
    assert r[0]["phases"][1] == {"titre": "Phase 2 — B", "statut": "EN COURS"}
    s = tmp_path / "signals.md"
    s.write_text("# S (MAJ 2026-11-17)\n\n## Actions ouvertes\n- [P1|ouvert] Faire A\n  - fait quand: A fait\n  - réf: `a.py`\n\n## Contexte chaud\n- chaud\n", encoding="utf-8")
    a = server.actions(s)
    assert a["maj"] == "2026-11-17" and a["actions"][0]["fait_quand"] == "A fait" and a["contexte"] == ["chaud"]


def test_safe_path_blocks_escape(tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "s.bin").write_text("x", encoding="utf-8")
    assert server.safe_path("a.md", tmp_path)
    assert server.safe_path("../a.md", tmp_path / "sub") is None
    assert server.safe_path("s.bin", tmp_path) is None


def test_state_on_real_project():
    s = server.state()
    assert {"entrainement", "roadmaps", "signals", "recherche", "benchmarks"} <= set(s)
    json.dumps(s, ensure_ascii=False)
