import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from bench_assistant import (  # noqa: E402
    CATS,
    has_loop,
    is_copy,
    is_refusal,
    load_bench,
    score,
    summarize,
)
from generate import penalize  # noqa: E402


def test_bench_file_is_valid_and_balanced():
    items = load_bench()
    counts = {c: sum(it["cat"] == c for it in items) for c in CATS}
    assert counts == {"faits": 30, "consignes": 15, "transformations": 25, "conversation": 10, "multi_tours": 20}
    assert all(it.get("src") for it in items if it["cat"] == "transformations")


def test_bench_has_no_holdout_question():
    holdout = Path(__file__).parent.parent / "prompts_assistant_holdout.txt"
    questions = {" ".join(q.split()).lower() for q in holdout.read_text(encoding="utf-8").splitlines() if q.strip()}
    turns = {" ".join(t.split()).lower() for it in load_bench() for t in it["turns"]}
    assert not questions & turns


def test_load_bench_rejects_bad_items(tmp_path):
    bad = tmp_path / "b.jsonl"
    bad.write_text(json.dumps({"id": "X", "cat": "multi_tours", "turns": ["a"], "ok": ["a"]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_bench(bad)
    bad.write_text(json.dumps({"id": "X", "cat": "faits", "turns": ["a"]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_bench(bad)


def test_copy_detected_with_or_without_quotes():
    src = "Le petit chien noir court dans le jardin."
    assert is_copy(src, "Le petit chien noir court dans le jardin.")
    assert is_copy(src, "Voici la phrase : « Le petit chien noir court dans le jardin. »")
    assert not is_copy(src, "Les petits chiens noirs courent dans les jardins.")


def test_refusal_and_loop_detectors():
    assert is_refusal("Je suis désolé, mais je ne peux pas répondre à cette demande.")
    assert not is_refusal("Avec plaisir, n'hésite pas si tu as d'autres questions.")
    loop = "Le ciel est bleu pendant la nuit. " * 4
    assert has_loop(loop)
    assert not has_loop("Le ciel est bleu. Il fait beau aujourd'hui. Les oiseaux chantent.")


def test_score_rules():
    item = {"id": "T", "cat": "transformations", "turns": ["q"], "src": "Le chat dort.", "ok": ["les chats dorment"]}
    assert score(item, ["Les chats dorment."])["correct"]
    assert score(item, ["Le chat dort."])["raisons"] == ["critere_absent", "recopie"]
    item = {"id": "C", "cat": "consignes", "turns": ["q"], "ok_all": ["bleu", "rouge"], "ko": ["vert"], "max_mots": 3}
    assert score(item, ["bleu rouge"])["correct"]
    assert set(score(item, ["bleu vert et encore"])["raisons"]) == {"critere_incomplet", "critere_interdit", "trop_long"}
    item = {"id": "M", "cat": "multi_tours", "turns": ["a", "b"], "ok": [".{3,}"]}
    assert score(item, ["Voici une blague sur les chats.", "Voici une blague sur les chats !"])["raisons"] == ["repete_tour"]


def test_case_sensitive_pattern_in_bench():
    item = next(it for it in load_bench() if it["id"] == "C06")
    assert score(item, ["BONJOUR"])["correct"]
    assert not score(item, ["bonjour"])["correct"]


def test_summarize_counts():
    recs = [{"cat": "faits", "correct": True, "raisons": []}, {"cat": "faits", "correct": False, "raisons": ["refus"]}]
    s = summarize(recs)
    assert s["par_categorie"]["faits"] == {"correct": 1, "total": 2, "taux": 0.5}
    assert s["raisons"] == {"refus": 1}


def test_repetition_penalty_lowers_seen_tokens_only():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    seen = torch.tensor([[True, True, False]])
    out = penalize(logits, seen, 2.0)
    assert torch.allclose(out, torch.tensor([[1.0, -4.0, 1.0]]))
