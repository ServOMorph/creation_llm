import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from convert_chat import ASSISTANT, CHAT_VOCAB, EOT, SRC_TOKENIZER, USER, build_chat_tokenizer, expand_vocab  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

V9 = ROOT / "checkpoints_v9" / "ckpt.pt"
V9_CHAT = ROOT / "checkpoints_v9" / "ckpt_chat.pt"
CHAT_TOK = ROOT / "tokenizer_fr32k_chat.json"
CFG = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=4, n_embd=64)
TEXTS = [
    "L'œuvre de Molière, né en 1622, reste jouée aujourd'hui.",
    "« Où vas-tu ? » demanda-t-elle. — Au marché !",
    "Ligne 1\nLigne 2\n\n\tTabulation, émoji ☀ et 3,14 €.",
    "Écris un poème sur <|la mer|> et user|>.",
]

needs_tok = pytest.mark.skipif(not SRC_TOKENIZER.exists(), reason="tokenizer_fr32k.json absent")


@pytest.fixture(scope="module")
def toks(tmp_path_factory):
    from tokenizers import Tokenizer

    dst = tmp_path_factory.mktemp("tok") / "chat.json"
    build_chat_tokenizer(SRC_TOKENIZER, dst)
    return Tokenizer.from_file(str(SRC_TOKENIZER)), Tokenizer.from_file(str(dst))


@needs_tok
def test_existing_ids_unchanged(toks):
    base, chat = toks
    base_vocab = base.get_vocab()
    chat_vocab = chat.get_vocab()
    assert all(chat_vocab[t] == i for t, i in base_vocab.items())
    assert chat.get_vocab_size() == base.get_vocab_size() + 2
    assert chat.token_to_id(EOT) == 0


@needs_tok
def test_chat_token_ids(toks):
    _, chat = toks
    assert chat.token_to_id(USER) == 32768
    assert chat.token_to_id(ASSISTANT) == 32769
    assert chat.get_vocab_size() <= CHAT_VOCAB
    assert CHAT_VOCAB % 64 == 0


@needs_tok
@pytest.mark.parametrize("text", TEXTS)
def test_plain_text_encoding_unchanged(toks, text):
    base, chat = toks
    assert chat.encode(text).ids == base.encode(text).ids


@needs_tok
def test_roundtrip_with_tags(toks):
    _, chat = toks
    text = f"{USER}Quelle est la capitale de la France ?{ASSISTANT}Paris.{EOT}{USER} Et de l'Italie ?{ASSISTANT}Rome.{EOT}"
    ids = chat.encode(text).ids
    assert ids.count(32768) == 2 and ids.count(32769) == 2 and ids.count(0) == 2
    assert chat.decode(ids, skip_special_tokens=False) == text


@needs_tok
def test_tags_are_single_tokens_next_to_text(toks):
    _, chat = toks
    ids = chat.encode(f"Bonjour{USER}salut{ASSISTANT}").ids
    assert ids[-1] == 32769
    assert 32768 in ids
    assert chat.decode(ids, skip_special_tokens=True) == "Bonjoursalut"


def make_model():
    torch.manual_seed(0)
    m = GPT(CFG)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.1)
    return m.eval()


def test_expand_vocab_keeps_logits():
    m = make_model()
    state = expand_vocab(m.state_dict(), 128)
    big = GPT(replace(CFG, vocab_size=128)).eval()
    big.load_state_dict(state)
    x = torch.randint(0, CFG.vocab_size, (2, 20))
    with torch.no_grad():
        assert torch.equal(m(x)[0], big(x)[0][..., : CFG.vocab_size])


def test_new_rows_are_mean():
    m = make_model()
    sd = m.state_dict()
    state = expand_vocab(sd, 128)
    for key in ("wte.weight", "lm_head.weight"):
        assert state[key].shape == (128, CFG.n_embd)
        assert torch.equal(state[key][: CFG.vocab_size], sd[key])
        assert torch.allclose(state[key][CFG.vocab_size :], sd[key].mean(0).expand(64, -1))


def test_expanded_model_trains_on_new_tokens():
    m = make_model()
    big = GPT(replace(CFG, vocab_size=128))
    big.load_state_dict(expand_vocab(m.state_dict(), 128))
    x = torch.tensor([[1, 64, 2, 65, 3]])
    _, loss = big(x, x)
    loss.backward()
    assert big.wte.weight.grad[64].abs().sum() > 0
    assert big.lm_head.weight.grad[65].abs().sum() > 0


def test_expand_vocab_rejects_shrink():
    with pytest.raises(ValueError):
        expand_vocab(make_model().state_dict(), 32)


@pytest.mark.skipif(not (V9.exists() and V9_CHAT.exists() and CHAT_TOK.exists()), reason="checkpoints v9 absents")
def test_real_v9_conversion_identical_on_plain_text():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(CHAT_TOK))
    ids = tok.encode(" ".join(TEXTS[:3])).ids
    x = torch.tensor([ids])
    outs = []
    for path in (V9, V9_CHAT):
        ckpt = torch.load(path, map_location="cpu", mmap=True)
        model = GPT(GPTConfig(**ckpt["model_cfg"])).eval()
        model.load_state_dict(ckpt["model"])
        with torch.no_grad():
            outs.append(model(x)[0])
        del ckpt, model
    assert outs[1].size(-1) == CHAT_VOCAB
    assert torch.equal(outs[0], outs[1][..., :32768])
