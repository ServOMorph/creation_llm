import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rd.experiences import c2_audit  # noqa: E402


def rec(reasons, answer="x"):
    return {"id": "I", "cat": "faits", "seed": 1, "reponses": [answer], "correct": not reasons, "raisons": reasons}


def test_literal_tokens_strip_regex():
    assert c2_audit.literal_tokens([r"\b8\b|huit", "loire"]) == ["8", "huit", "loire"]


def test_classify_labels():
    item = {"ok": ["loire"]}
    assert c2_audit.classify_record(item, rec([])) is None
    assert c2_audit.classify_record(item, rec(["critere_interdit"])) == "erreur_factuelle_probable"
    assert c2_audit.classify_record(item, rec(["boucle"])) == "defaut_forme"
    assert c2_audit.classify_record(item, rec(["trop_long"])) == "contrainte_format"
    assert c2_audit.classify_record(item, rec(["critere_absent"], "C'est la Loîre.")) == "motif_trop_strict_probable"
    assert c2_audit.classify_record(item, rec(["critere_absent"], "C'est la Seine.")) == "omission_ou_reponse_differente"


def test_family_splits_detect_leak():
    assert c2_audit.check_family_splits([("a", "train"), ("a", "test"), ("b", "train")]) == ["a"]
    assert c2_audit.check_family_splits([("a", "train"), ("b", "test")]) == []
