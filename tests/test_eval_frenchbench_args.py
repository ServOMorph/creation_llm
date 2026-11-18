import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

eval_frenchbench = pytest.importorskip("eval_frenchbench")


def test_known_name_and_fewshot():
    name, shot, ckpt = eval_frenchbench.parse(["v9", "5"])
    assert (name, shot) == ("v9", 5) and ckpt == eval_frenchbench.MODELS["v9"]


def test_free_checkpoint_named_after_folder():
    name, shot, ckpt = eval_frenchbench.parse(["--ckpt", "checkpoints_x/ckpt.pt"])
    assert (name, shot, ckpt) == ("checkpoints_x", 0, Path("checkpoints_x/ckpt.pt"))
    assert eval_frenchbench.parse(["--ckpt", "a/ckpt.pt", "--nom", "essai"])[0] == "essai"


def test_unknown_name_refused():
    with pytest.raises(SystemExit):
        eval_frenchbench.parse(["inconnu"])
