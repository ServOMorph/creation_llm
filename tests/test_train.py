import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import GPTConfig  # noqa: E402
from train import TrainConfig, doc_starts, get_batch, get_lr, train  # noqa: E402

MCFG = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=4, n_embd=64)


def make_cfg(tmp_path, **kw):
    pattern = np.tile(np.arange(16, dtype=np.uint16), 4000)
    pattern.tofile(tmp_path / "train.bin")
    pattern.tofile(tmp_path / "val.bin")
    base = dict(
        train_bin=str(tmp_path / "train.bin"),
        val_bin=str(tmp_path / "val.bin"),
        ckpt_dir=str(tmp_path / "ckpt"),
        device="cpu",
        batch_size=8,
        accum_steps=2,
        max_steps=40,
        lr=3e-3,
        warmup_steps=5,
        eval_interval=20,
        eval_batches=4,
        save_interval=10,
        log_interval=10,
    )
    base.update(kw)
    return TrainConfig(**base)


def test_batches_start_at_document_boundary():
    data = np.array([0, 5, 6, 7, 0, 8, 9, 0, 1, 2, 3, 4, 5, 6], dtype=np.uint16)
    starts = doc_starts(data, block_size=4)
    assert starts.tolist() == [0, 4, 7]
    gen = torch.Generator().manual_seed(0)
    x, y = get_batch(data, 16, 4, "cpu", gen, starts)
    assert torch.all(x[:, 0] == 0)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_lr_schedule():
    cfg = TrainConfig(lr=1e-3, warmup_steps=10, max_steps=100, min_lr_ratio=0.1)
    assert get_lr(0, cfg) < get_lr(9, cfg) <= 1e-3
    assert abs(get_lr(10, cfg) - 1e-3) < 1e-9
    assert abs(get_lr(100, cfg) - 1e-4) < 1e-9


def test_training_reduces_loss(tmp_path):
    cfg = make_cfg(tmp_path)
    train_loss, val_loss = train(cfg, MCFG)
    assert val_loss < 0.5 * np.log(MCFG.vocab_size)
    assert (Path(cfg.ckpt_dir) / "ckpt.pt").exists()
    assert (Path(cfg.ckpt_dir) / "log.csv").exists()


def test_init_from(tmp_path):
    cfg = make_cfg(tmp_path, max_steps=40)
    _, trained_val = train(cfg, MCFG)
    cfg2 = make_cfg(tmp_path, max_steps=1, lr=1e-9, warmup_steps=1, ckpt_dir=str(tmp_path / "ckpt2"), init_from=str(Path(cfg.ckpt_dir) / "ckpt.pt"))
    _, init_val = train(cfg2, MCFG)
    assert abs(init_val - trained_val) < 0.05
    assert torch.load(tmp_path / "ckpt2" / "ckpt.pt")["step"] == 1


def test_resume(tmp_path):
    cfg = make_cfg(tmp_path, max_steps=20)
    train(cfg, MCFG)
    ckpt = torch.load(Path(cfg.ckpt_dir) / "ckpt.pt")
    assert ckpt["step"] == 20
    cfg2 = make_cfg(tmp_path, max_steps=30, resume=True)
    train(cfg2, MCFG)
    ckpt = torch.load(Path(cfg.ckpt_dir) / "ckpt.pt")
    assert ckpt["step"] == 30


def test_muon_training_and_resume(tmp_path):
    cfg = make_cfg(tmp_path, max_steps=20, optimizer="muon", lr=3e-3)
    _, val = train(cfg, MCFG)
    assert val < 0.5 * np.log(MCFG.vocab_size)
    ckpt = torch.load(Path(cfg.ckpt_dir) / "ckpt.pt")
    assert isinstance(ckpt["optimizer"], list) and len(ckpt["optimizer"]) == 2
    cfg2 = make_cfg(tmp_path, max_steps=25, optimizer="muon", lr=3e-3, resume=True)
    train(cfg2, MCFG)
    assert torch.load(Path(cfg.ckpt_dir) / "ckpt.pt")["step"] == 25


def test_adamw_checkpoint_stays_a_single_state(tmp_path):
    cfg = make_cfg(tmp_path, max_steps=10)
    train(cfg, MCFG)
    assert isinstance(torch.load(Path(cfg.ckpt_dir) / "ckpt.pt")["optimizer"], dict)


def test_mixed_source_follows_weights(tmp_path):
    from train import MixedSource

    np.full(5000, 1, dtype=np.uint16).tofile(tmp_path / "a.bin")
    np.full(5000, 2, dtype=np.uint16).tofile(tmp_path / "b.bin")
    mix = MixedSource(f"{tmp_path / 'a.bin'}=0.8,{tmp_path / 'b.bin'}=0.2")
    x, y = mix.batch(1000, 16, "cpu", torch.Generator().manual_seed(0))
    share_a = (x[:, 0] == 1).float().mean().item()
    assert 0.74 < share_a < 0.86
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_training_on_mixed_sources(tmp_path):
    pattern = np.tile(np.arange(16, dtype=np.uint16), 4000)
    pattern.tofile(tmp_path / "a.bin")
    pattern.tofile(tmp_path / "b.bin")
    spec = f"{tmp_path / 'a.bin'}=0.5,{tmp_path / 'b.bin'}=0.5"
    cfg = make_cfg(tmp_path, train_bin=spec, val_bin=spec)
    _, val = train(cfg, MCFG)
    assert val < 0.5 * np.log(MCFG.vocab_size)
