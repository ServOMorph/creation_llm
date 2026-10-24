from pathlib import Path

import pytest
from tokenizers import Tokenizer

TOK = Path(__file__).parent.parent / "tokenizer_fr32k.json"

pytestmark = pytest.mark.skipif(not TOK.exists(), reason="tokenizer_fr32k.json absent")


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.from_file(str(TOK))


def test_vocab_and_eot(tok):
    assert tok.get_vocab_size() == 32768
    assert tok.token_to_id("<|endoftext|>") == 0


@pytest.mark.parametrize(
    "text",
    [
        "L'œuvre de Molière, né en 1622, reste jouée aujourd'hui.",
        "« Où vas-tu ? » demanda-t-elle. — Au marché !",
        "Ligne 1\nLigne 2\n\n\tTabulation, émoji ☀ et 3,14 €.",
        "Il était une fois un petit lapin nommé Lulu.",
    ],
)
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text).ids) == text


def test_compression_on_general_french(tok):
    text = "La Révolution française est une période de bouleversements politiques et sociaux en France. " * 20
    assert len(text) / len(tok.encode(text).ids) > 4.0
