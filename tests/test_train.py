import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import GPTConfig  # noqa: E402
from train import TrainConfig, get_lr, train  # noqa: E402

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


def test_resume(tmp_path):
    cfg = make_cfg(tmp_path, max_steps=20)
    train(cfg, MCFG)
    ckpt = torch.load(Path(cfg.ckpt_dir) / "ckpt.pt")
    assert ckpt["step"] == 20
    cfg2 = make_cfg(tmp_path, max_steps=30, resume=True)
    train(cfg2, MCFG)
    ckpt = torch.load(Path(cfg.ckpt_dir) / "ckpt.pt")
    assert ckpt["step"] == 30
