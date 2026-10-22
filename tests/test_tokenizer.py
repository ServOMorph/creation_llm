import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer

ROOT = Path(__file__).parent.parent
TOK = ROOT / "tokenizer.json"
VAL = ROOT / "data" / "validation.jsonl"

pytestmark = pytest.mark.skipif(not TOK.exists(), reason="tokenizer.json absent")


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.from_file(str(TOK))


def test_vocab_size(tok):
    assert tok.get_vocab_size() == 8192


def test_eot_token(tok):
    assert tok.token_to_id("<|endoftext|>") == 0


@pytest.mark.parametrize(
    "text",
    [
        "Il était une fois un petit garçon nommé Léo.",
        "Où est l'œuf ? Ça va très bien, merci !",
        "Ligne 1\nLigne 2\n\nÉlève, forêt, naïve, Noël.",
        "« Bonjour », dit-elle.",
    ],
)
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text).ids) == text


def test_compression_validation(tok):
    texts = []
    with open(VAL, encoding="utf-8") as f:
        for _, line in zip(range(2000), f):
            texts.append(json.loads(line)["text"])
    chars = sum(len(t) for t in texts)
    ntok = sum(len(e.ids) for e in tok.encode_batch(texts))
    assert chars / ntok >= 3.0
    for t in texts[:200]:
        assert tok.decode(tok.encode(t).ids) == t
