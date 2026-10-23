import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from augment_names import EXCLUDED, FEM, MASC, NEUTRE, augment  # noqa: E402


def test_replaces_all_occurrences_with_gender():
    text = "Il était une fois un lapin nommé Lulu. Lulu aimait les carottes. La maman de Lulu riait."
    out = augment(text, random.Random(0), p=1.0)
    assert "Lulu" not in out
    new = out.split("nommé ")[1].split(".")[0]
    assert new in MASC
    assert out.count(new) == 3


def test_feminine_and_neutral():
    rng = random.Random(1)
    out = augment("Une fille appelée Lila jouait. Lila riait.", rng, p=1.0)
    assert out.split("appelée ")[1].split(" ")[0] in FEM
    out = augment("Le chat s'appelait Pipo. Pipo dormait.", rng, p=1.0)
    assert out.split("s'appelait ")[1].split(".")[0] in NEUTRE


def test_never_uses_benchmark_names():
    rng = random.Random(2)
    for _ in range(500):
        out = augment("Un garçon nommé Lulu et une fille nommée Mimi.", rng, p=1.0)
        assert not any(name in out.split() for name in EXCLUDED)


def test_word_boundaries_and_no_name():
    text = "Un ours nommé Lu mangeait. Lulu dormait."
    out = augment(text, random.Random(3), p=1.0)
    assert "Lulu dormait." in out
    plain = "Il était une fois un petit chat qui dormait."
    assert augment(plain, random.Random(0), p=1.0) == plain


def test_probability_zero_keeps_text():
    text = "Un lapin nommé Lulu. Lulu sautait."
    assert augment(text, random.Random(0), p=0.0) == text
