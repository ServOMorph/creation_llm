import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare_corpus import write_batch  # noqa: E402

TOK = Tokenizer.from_file(str(Path(__file__).parent.parent / "tokenizer.json"))


def test_write_batch_routes_first_docs_to_validation(tmp_path):
    texts = ["Le chat dort.", "Le chien court.", "Il pleut.", "Fin."]
    counts = {"train": 0, "val": 0}
    with open(tmp_path / "t.bin", "wb") as ft, open(tmp_path / "v.bin", "wb") as fv:
        seen = write_batch(TOK, texts, 0, 0, 2, ft, fv, counts)
    assert seen == 4
    val = np.fromfile(tmp_path / "v.bin", dtype=np.uint16).tolist()
    train = np.fromfile(tmp_path / "t.bin", dtype=np.uint16).tolist()
    assert val.count(0) == 2 and train.count(0) == 2
    assert val[-1] == 0 and train[-1] == 0
    assert TOK.decode(val[: val.index(0)]) == "Le chat dort."
    assert counts == {"train": len(train), "val": len(val)}


def test_write_batch_continues_document_count(tmp_path):
    counts = {"train": 0, "val": 0}
    with open(tmp_path / "t.bin", "wb") as ft, open(tmp_path / "v.bin", "wb") as fv:
        seen = write_batch(TOK, ["a.", "b."], 0, 0, 3, ft, fv, counts)
        seen = write_batch(TOK, ["c.", "d."], 0, seen, 3, ft, fv, counts)
    assert seen == 4
    assert np.fromfile(tmp_path / "v.bin", dtype=np.uint16).tolist().count(0) == 3
    assert np.fromfile(tmp_path / "t.bin", dtype=np.uint16).tolist().count(0) == 1
