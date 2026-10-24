import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from augment_names import EXCLUDED, FEM, MASC, NEUTRE, augment, vary_intro  # noqa: E402


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


def test_never_uses_excluded_names():
    rng = random.Random(2)
    for _ in range(500):
        out = augment("Un garçon nommé Lulu et une fille nommée Mimi.", rng, p=1.0)
        assert not any(name in out.split() for name in EXCLUDED)


def test_name_pools_have_no_duplicates_across_genders():
    assert not set(MASC) & set(FEM)


def test_word_boundaries_and_no_name():
    text = "Un ours nommé Lu mangeait. Lulu dormait."
    out = augment(text, random.Random(3), p=1.0)
    assert "Lulu dormait." in out
    plain = "Il était une fois un petit chat qui dormait."
    assert augment(plain, random.Random(0), p=1.0) == plain


def test_probability_zero_keeps_text():
    text = "Un lapin nommé Lulu. Lulu sautait."
    assert augment(text, random.Random(0), p=0.0) == text


def only(kind):
    weights = [0, 0, 0, 0]
    weights[kind] = 1
    return weights


def test_vary_intro_variants():
    text = "Il était une fois un petit lapin nommé Lulu. Lulu aimait les carottes."
    rng = random.Random(0)
    assert vary_intro(text, rng, only(0)) == text
    assert vary_intro(text, rng, only(1)) == "Il était une fois Lulu, un petit lapin. Lulu aimait les carottes."
    assert vary_intro(text, rng, only(2)) == "Il était une fois un petit lapin qui s'appelait Lulu. Lulu aimait les carottes."
    assert vary_intro(text, rng, only(3)) == "Il était une fois le petit lapin Lulu. Lulu aimait les carottes."


def test_vary_intro_articles_and_elision():
    rng = random.Random(0)
    assert vary_intro("Elle vit une petite fille nommée Lila.", rng, only(3)) == "Elle vit la petite fille Lila."
    assert vary_intro("Il rencontra un écureuil nommé Noisette.", rng, only(3)) == "Il rencontra l'écureuil Noisette."


def test_vary_intro_avoids_double_relative():
    text = "Il était une fois un petit lapin nommé Lulu qui adorait jouer."
    assert vary_intro(text, random.Random(0), only(2)) == text
