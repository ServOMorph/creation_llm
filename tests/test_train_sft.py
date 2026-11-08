import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import GPT, GPTConfig  # noqa: E402
from prepare_sft import write_split  # noqa: E402
from train_sft import IGNORE, PackedData, SFTConfig, Schedule, masked_loss_sum, pack, train  # noqa: E402

MCFG = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=4, n_embd=64)
USER, ASSISTANT, EOT = 1, 2, 0


def example(q, r):
    ids = [USER, *q, ASSISTANT, *r, EOT]
    mask = [0] * (len(q) + 2) + [1] * (len(r) + 1)
    return ("t", "", np.array(ids, dtype=np.uint16), np.array(mask, dtype=np.uint8))


def make_data(tmp_path, n=24, seed=0):
    rng = np.random.default_rng(seed)
    exs = []
    for _ in range(n):
        q = rng.integers(3, 64, rng.integers(2, 8)).tolist()
        r = [(t * 7 + 3) % 61 + 3 for t in q] + [q[0]]
        exs.append(example(q, r))
    write_split(exs, tmp_path, "train")
    write_split(exs[:8], tmp_path, "val")
    return exs


def make_init(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "init.pt"
    torch.save({"model": GPT(MCFG).state_dict(), "model_cfg": asdict(MCFG), "step": 0}, path)
    return path


def make_cfg(tmp_path, **kw):
    base = dict(
        data_dir=str(tmp_path),
        init_from=str(make_init(tmp_path)),
        ckpt_dir=str(tmp_path / "ckpt"),
        device="cpu",
        batch_size=2,
        accum_steps=2,
        epochs=40,
        lr=3e-3,
        warmup_steps=5,
        eval_interval=20,
        save_interval=10,
        log_interval=10,
        optimizer="adamw",
    )
    base.update(kw)
    return SFTConfig(**base)


def test_pack_uses_every_example_once_within_capacity():
    lengths = np.random.default_rng(0).integers(4, 17, 500)
    windows = pack(lengths, 32, seed=1)
    assert sorted(i for w in windows for i in w) == list(range(500))
    assert all(lengths[w].sum() <= 32 for w in windows)
    assert lengths.sum() / (len(windows) * 32) > 0.95
    assert pack(lengths, 32, seed=1) == windows
    assert pack(lengths, 32, seed=2) != windows
    with pytest.raises(ValueError):
        pack([40], 32, seed=0)


def test_batch_targets_follow_answer_mask(tmp_path):
    exs = [example([10, 11], [20, 21, 22]), example([12], [23])]
    write_split(exs, tmp_path, "train")
    data = PackedData(tmp_path, "train", 16)
    x, y, doc = data.batch([[0, 1]], "cpu")
    assert x[0, :12].tolist() == [1, 10, 11, 2, 20, 21, 22, 0, 1, 12, 2, 23] and x[0, 13:].eq(0).all()
    expected = [IGNORE, IGNORE, IGNORE, 20, 21, 22, 0, IGNORE, IGNORE, IGNORE, 23, 0] + [IGNORE] * 4
    assert y[0].tolist() == expected
    assert doc[0].tolist() == [0] * 8 + [1] * 5 + [-1] * 3


def test_masked_loss_matches_manual_computation(tmp_path):
    exs = [example([10, 11], [20, 21, 22]), example([12], [23])]
    write_split(exs, tmp_path, "train")
    x, y, doc = PackedData(tmp_path, "train", 16).batch([[0, 1]], "cpu")
    torch.manual_seed(0)
    model = GPT(MCFG).eval()
    with torch.no_grad():
        loss, n = masked_loss_sum(model, x, y, doc)
        logits, _ = model(x, doc_ids=doc)
    keep = [3, 4, 5, 6, 10, 11]
    manual = sum(F.cross_entropy(logits[0, p], y[0, p]) for p in keep)
    assert n == 6
    assert torch.allclose(loss, manual, atol=1e-5)


def test_packed_documents_do_not_see_each_other():
    torch.manual_seed(0)
    model = GPT(MCFG).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.1)
    a = torch.randint(3, 64, (1, 10))
    b = torch.randint(3, 64, (1, 14))
    packed = torch.cat([a, b], dim=1)
    doc = torch.tensor([[0] * 10 + [1] * 14])
    with torch.no_grad():
        out = model(packed, doc_ids=doc)[0]
        alone_a, alone_b = model(a)[0], model(b)[0]
        leaky = model(packed)[0]
    assert torch.allclose(out[:, :10], alone_a, atol=1e-4)
    assert torch.allclose(out[:, 10:], alone_b, atol=1e-4)
    assert not torch.allclose(leaky[:, 10:], alone_b, atol=1e-4)


def test_schedule_covers_epochs(tmp_path):
    make_data(tmp_path, n=40)
    cfg = make_cfg(tmp_path, epochs=3)
    data = PackedData(tmp_path, "train", MCFG.block_size)
    sched = Schedule(data, cfg)
    assert sched.total == sum(len(data.windows(cfg.seed + e)) // 4 for e in range(3))
    assert sched.locate(sched.steps_per_epoch[0]) == (1, 0)
    assert Schedule(data, make_cfg(tmp_path, epochs=3, max_steps=2)).total == 2


def test_overfits_small_dataset(tmp_path):
    make_data(tmp_path)
    cfg = make_cfg(tmp_path, epochs=60)
    train_loss, val_loss = train(cfg)
    assert val_loss < 0.3
    log = (Path(cfg.ckpt_dir) / "log.csv").read_text().splitlines()
    assert float(log[1].split(",")[2]) > 3.0


def test_resume_is_identical(tmp_path):
    make_data(tmp_path)
    full = make_cfg(tmp_path, epochs=4, ckpt_dir=str(tmp_path / "full"), save_interval=1000)
    train(full)
    part = make_cfg(tmp_path, epochs=4, max_steps=7, ckpt_dir=str(tmp_path / "part"), save_interval=1000)
    train(part)
    resumed = make_cfg(tmp_path, epochs=4, ckpt_dir=str(tmp_path / "part"), save_interval=1000, resume=True)
    train(resumed)
    a = torch.load(tmp_path / "full" / "ckpt.pt")
    b = torch.load(tmp_path / "part" / "ckpt.pt")
    assert a["step"] == b["step"] > 7
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k


def test_resume_identical_with_muon(tmp_path):
    make_data(tmp_path)
    kw = dict(epochs=3, optimizer="muon", lr=1e-2, save_interval=1000)
    train(make_cfg(tmp_path, ckpt_dir=str(tmp_path / "full"), **kw))
    train(make_cfg(tmp_path, ckpt_dir=str(tmp_path / "part"), max_steps=5, **kw))
    train(make_cfg(tmp_path, ckpt_dir=str(tmp_path / "part"), resume=True, **kw))
    a = torch.load(tmp_path / "full" / "ckpt.pt")["model"]
    b = torch.load(tmp_path / "part" / "ckpt.pt")["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
