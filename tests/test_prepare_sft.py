import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from convert_chat import ASSISTANT, CHAT_TOKENIZER, EOT, USER  # noqa: E402
from prepare_sft import (  # noqa: E402
    HOLDOUT,
    MAX_LEN,
    SFT_OUT,
    build_example,
    fitting_turns,
    has_reserved,
    is_refusal,
    is_valid,
    load_holdout,
    load_split,
    prepare,
    source_files,
    write_split,
)

needs_tok = pytest.mark.skipif(not CHAT_TOKENIZER.exists(), reason="tokenizer_fr32k_chat.json absent")
needs_out = pytest.mark.skipif(not (SFT_OUT / "train_offsets.npy").exists(), reason="données SFT préparées absentes")


def conv(*contents):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": c} for i, c in enumerate(contents)]


@pytest.fixture(scope="module")
def tok():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(CHAT_TOKENIZER))


@pytest.fixture(scope="module")
def ids(tok):
    return tuple(tok.token_to_id(t) for t in (USER, ASSISTANT, EOT))


def write_parquet(path, conversations):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"messages": conversations}), path)


def test_is_valid():
    assert is_valid(conv("q", "r"))
    assert is_valid(conv("q", "r", "q2", "r2"))
    assert not is_valid(conv("q"))
    assert not is_valid(conv("q", "r", "q2"))
    assert not is_valid(conv("q", ""))
    assert not is_valid([{"role": "assistant", "content": "r"}, {"role": "user", "content": "q"}])


def test_has_reserved():
    assert not has_reserved(conv("Bonjour", "Salut <|la mer|>"))
    for r in (USER, ASSISTANT, EOT):
        assert has_reserved(conv("q", f"texte {r} suite"))


@needs_tok
def test_mask_aligned_on_answers(tok, ids):
    c = conv("Quelle est la capitale ?", "Paris.", "Et de l'Italie ?", "Rome, bien sûr.")
    turn_ids = [tok.encode(m["content"]).ids for m in c]
    seq, mask = build_example(turn_ids, *ids)
    assert len(seq) == len(mask)
    assert tok.decode(seq, skip_special_tokens=False) == f"{USER}Quelle est la capitale ?{ASSISTANT}Paris.{EOT}{USER}Et de l'Italie ?{ASSISTANT}Rome, bien sûr.{EOT}"
    target = [t for t, m in zip(seq, mask) if m]
    assert tok.decode(target, skip_special_tokens=False) == f"Paris.{EOT}Rome, bien sûr.{EOT}"
    assert seq[0] == ids[0] and mask[0] == 0
    assert seq[-1] == ids[2] and mask[-1] == 1
    assert all(m == 0 for t, m in zip(seq, mask) if t in ids[:2])


@needs_tok
def test_prepare_filters_and_splits(tok, tmp_path):
    long_answer = " ".join(["mot"] * 400)
    data = [conv(f"Question numéro {i} ?", f"Réponse {i}.") for i in range(30)]
    data += [conv("Question numéro 3 ?", "Autre réponse."), conv("  Question   numéro 5 ? ", "Encore.")]
    data += [conv("Longue ?", long_answer), conv("Balise ?", f"x {EOT} y"), conv("Scellée ?", "Non.")]
    write_parquet(tmp_path / "luth_a-00000-of-00001.parquet", data)
    write_parquet(tmp_path / "luth_b-00000-of-00001.parquet", [conv("Question numéro 7 ?", "Doublon inter-sources.")])
    files = {"a": [tmp_path / "luth_a-00000-of-00001.parquet"], "b": [tmp_path / "luth_b-00000-of-00001.parquet"]}
    splits, stats = prepare(files, tok, max_len=100, val_size=5, holdout={"Scellée ?"}, batch=7)
    assert stats["a"] == {"lignes": 35, "invalides": 0, "balises": 1, "refus": 0, "trop_longs": 1, "tronqués": 0, "doublons": 2, "holdout": 1, "gardés": 30, "répétition": 1, "train": 25}
    assert stats["b"]["doublons"] == 1 and stats["b"]["gardés"] == 0
    assert len(splits["val"]) == 5 and len(splits["train"]) == 25
    assert not {ex[1] for ex in splits["val"]} & {ex[1] for ex in splits["train"]}
    assert all(len(ex[2]) <= 100 for split in splits.values() for ex in split)
    again, _ = prepare(files, tok, max_len=100, val_size=5, holdout={"Scellée ?"}, batch=3)
    assert [ex[1] for ex in again["val"]] == [ex[1] for ex in splits["val"]]


def test_is_refusal():
    assert is_refusal(conv("Météo ?", "Je suis désolé, mais je ne peux pas accéder à internet."))
    assert is_refusal(conv("Q ?", "En tant que modèle de langage, je n'ai pas d'avis."))
    assert not is_refusal(conv("Q ?", "Voici la réponse : je ne peux pas dire mieux que Paris."))
    assert not is_refusal(conv("Désolé, je ne peux pas venir.", "Ce n'est pas grave."))


@needs_tok
def test_prepare_drops_refusals(tok, tmp_path):
    data = [conv("Météo ?", "Désolé, je ne peux pas accéder aux prévisions."), conv("Capitale ?", "Paris.")]
    write_parquet(tmp_path / "luth_a-00000-of-00001.parquet", data)
    files = {"a": [tmp_path / "luth_a-00000-of-00001.parquet"]}
    _, stats = prepare(files, tok, val_size=0, drop_refusals=True)
    assert stats["a"]["refus"] == 1 and stats["a"]["gardés"] == 1
    _, stats = prepare(files, tok, val_size=0)
    assert stats["a"]["refus"] == 0 and stats["a"]["gardés"] == 2


@needs_tok
def test_prepare_reads_jsonl_extra_source(tok, tmp_path):
    import json

    path = tmp_path / "transfo.jsonl"
    path.write_text("".join(json.dumps({"id": str(i), "messages": conv(f"Mets au pluriel : chat {i}", f"chats {i}")}, ensure_ascii=False) + "\n" for i in range(4)), encoding="utf-8")
    files = source_files(raw=tmp_path, extra=[path])
    assert files["transfo"] == [path]
    splits, stats = prepare({"transfo": files["transfo"]}, tok, val_size=1)
    assert stats["transfo"]["gardés"] == 4 and len(splits["train"]) == 3
    splits, stats = prepare({"transfo": files["transfo"]}, tok, val_size=1, repeat={"transfo": 4})
    assert len(splits["val"]) == 1 and len(splits["train"]) == 12 and stats["transfo"]["train"] == 12
    val_ids = {tuple(ex[2]) for ex in splits["val"]}
    assert not val_ids & {tuple(ex[2]) for ex in splits["train"]}
    assert len({tuple(ex[2]) for ex in splits["train"]}) == 3


def test_fitting_turns_keeps_complete_pairs():
    t = [[1] * 10, [2] * 10, [3] * 10, [4] * 100]
    assert fitting_turns(t, 23) == 2
    assert fitting_turns(t, 22) == 0
    assert fitting_turns(t, 1000) == 4


@needs_tok
def test_prepare_truncates_long_multi_turn(tok, ids, tmp_path):
    long_answer = " ".join(["mot"] * 400)
    data = [conv("Bonjour ?", "Salut.", "Raconte ?", long_answer), conv("Long ?", long_answer, "Et ?", "Oui.")]
    write_parquet(tmp_path / "luth_a-00000-of-00001.parquet", data)
    files = {"a": [tmp_path / "luth_a-00000-of-00001.parquet"]}
    splits, stats = prepare(files, tok, max_len=100, val_size=0, truncate=True)
    assert stats["a"]["tronqués"] == 1 and stats["a"]["trop_longs"] == 1 and stats["a"]["gardés"] == 1
    seq, mask = splits["train"][0][2], splits["train"][0][3]
    assert tok.decode(seq.tolist(), skip_special_tokens=False) == f"{USER}Bonjour ?{ASSISTANT}Salut.{EOT}"
    assert seq[-1] == ids[2] and mask[-1] == 1 and len(seq) == len(mask)
    _, stats = prepare(files, tok, max_len=100, val_size=0)
    assert stats["a"]["trop_longs"] == 2 and stats["a"]["tronqués"] == 0


@needs_tok
def test_write_and_load_roundtrip(tok, tmp_path):
    data = [conv(f"Q{i} ?", f"R{i}.") for i in range(10)]
    write_parquet(tmp_path / "luth_a-00000-of-00001.parquet", data)
    splits, _ = prepare({"a": [tmp_path / "luth_a-00000-of-00001.parquet"]}, tok, val_size=3)
    info = write_split(splits["train"], tmp_path, "train")
    tokens, mask, offsets = load_split(tmp_path, "train")
    assert info["exemples"] == 7 and len(offsets) == 8 and offsets[-1] == len(tokens) == len(mask)
    for i, ex in enumerate(splits["train"]):
        assert np.array_equal(tokens[offsets[i] : offsets[i + 1]], ex[2])
        assert np.array_equal(mask[offsets[i] : offsets[i + 1]], ex[3])


def first_questions(tokens, offsets, assistant_id):
    pos = np.flatnonzero(tokens == assistant_id)
    firsts = pos[np.searchsorted(pos, offsets[:-1])]
    return [tokens[s + 1 : a].tobytes() for s, a in zip(offsets[:-1], firsts)]


@needs_tok
@needs_out
def test_real_output_structure(ids):
    user_id, assistant_id, eot_id = ids
    for split in ("train", "val"):
        tokens, mask, offsets = load_split(SFT_OUT, split)
        assert len(tokens) == len(mask) == offsets[-1]
        assert np.diff(offsets).max() <= MAX_LEN and np.diff(offsets).min() >= 4
        starts, ends = offsets[:-1], offsets[1:] - 1
        assert (tokens[starts] == user_id).all() and (mask[starts] == 0).all()
        assert (tokens[ends] == eot_id).all() and (mask[ends] == 1).all()
        assert not mask[(tokens == user_id) | (tokens == assistant_id)].any()
        assert (mask[tokens == eot_id] == 1).all()
        assert tokens.max() < 32770


@needs_tok
@needs_out
def test_real_val_and_holdout_disjoint_from_train(tok, ids):
    assistant_id = ids[1]
    val = first_questions(*load_split(SFT_OUT, "val")[::2], assistant_id)
    tokens, _, offsets = load_split(SFT_OUT, "train")
    train = set(first_questions(tokens, offsets, assistant_id))
    assert len(val) == 2000
    assert not set(val) & train
    holdout = {np.array(tok.encode(q).ids, dtype=np.uint16).tobytes() for q in load_holdout(HOLDOUT)}
    assert len(holdout) == 50
    assert not holdout & train and not holdout & set(val)
