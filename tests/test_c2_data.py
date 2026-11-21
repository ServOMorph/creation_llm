import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rd.experiences import c2_data  # noqa: E402


def families():
    return c2_data.make_families(60, 7)


def test_families_are_deterministic_and_unique():
    a, b = c2_data.make_families(30, 3), c2_data.make_families(30, 3)
    assert a == b
    assert len({f["entite"] for f in a}) == 30


def test_split_by_family_is_disjoint():
    fams = families()
    splits = c2_data.split_by_family(fams, {"train": 30, "dev": 10, "test": 20}, 1)
    ids = [f["famille"] for v in splits.values() for f in v]
    assert len(ids) == len(set(ids)) == 60
    assert c2_data.check_disjoint(splits) == []


def test_examples_validate_for_every_variant():
    for f in families():
        for v in c2_data.VARIANTS:
            ex = c2_data.build_example(f, v, 0)
            assert c2_data.validate_evidence(ex, f) == []


def test_validate_detects_contradiction_and_missing_evidence():
    f = families()[0]
    ex = c2_data.build_example(f, "gold", 0)
    bad = dict(ex, messages=[ex["messages"][0], {"role": "assistant", "content": "Je ne sais pas."}])
    assert "reponse_hors_preuve" in c2_data.validate_evidence(bad, f)
    contradict = c2_data.render(f["passage"], f["entite"], f["valeur_contrefactuelle"])
    mixed = dict(ex, messages=[{"role": "user", "content": ex["messages"][0]["content"] + "\n" + contradict}, ex["messages"][1]])
    assert "contradiction" in c2_data.validate_evidence(mixed, f)


def test_no_evidence_variant_has_no_gold_passage():
    for f in families():
        ex = c2_data.build_example(f, "sans_preuve", 0)
        assert f["entite"] not in "\n".join(ex["messages"][0]["content"].split("Question")[0].splitlines())
        assert ex["messages"][1]["content"] == c2_data.ABSTAIN


def test_pilot_size_and_balance():
    pilot = c2_data.build_pilot(families()[:20], 100)
    assert len(pilot) == 100
    counts = {v: sum(ex["variante"] == v for _, ex in pilot) for v in c2_data.VARIANTS}
    assert max(counts.values()) - min(counts.values()) <= 1
