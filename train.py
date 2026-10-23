import argparse
import csv
import math
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig

ROOT = Path(__file__).parent
DATA = ROOT / "data"


@dataclass
class TrainConfig:
    train_bin: str = str(DATA / "train.bin")
    val_bin: str = str(DATA / "validation.bin")
    ckpt_dir: str = str(ROOT / "checkpoints")
    device: str = "cuda"
    batch_size: int = 16
    accum_steps: int = 8
    max_steps: int = 10000
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 40
    save_interval: int = 500
    log_interval: int = 10
    seed: int = 1337
    resume: bool = False
    init_from: str = ""
    align_docs: bool = False


def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    min_lr = cfg.lr * cfg.min_lr_ratio
    return min_lr + 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))) * (cfg.lr - min_lr)


def doc_starts(data, block_size, eot_id=0):
    starts = np.flatnonzero(data == eot_id)
    return starts[starts < len(data) - block_size - 1]


def all_starts(data, block_size):
    return None


def get_batch(data, batch_size, block_size, device, gen, starts=None):
    if starts is None:
        ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen).tolist()
    else:
        ix = starts[torch.randint(len(starts), (batch_size,), generator=gen).numpy()].tolist()
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, data, cfg, block_size, autocast, starts):
    model.eval()
    gen = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = get_batch(data, cfg.batch_size, block_size, cfg.device, gen, starts)
        with autocast:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path, model, opt, step, mcfg, cfg):
    tmp = Path(str(path) + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "model_cfg": asdict(mcfg),
            "train_cfg": asdict(cfg),
        },
        tmp,
    )
    os.replace(tmp, path)


def train(cfg, mcfg):
    torch.manual_seed(cfg.seed)
    use_cuda = cfg.device.startswith("cuda")
    autocast = torch.autocast(cfg.device.split(":")[0], dtype=torch.bfloat16, enabled=use_cuda)
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ckpt.pt"

    train_data = np.memmap(cfg.train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(cfg.val_bin, dtype=np.uint16, mode="r")
    starts_fn = doc_starts if cfg.align_docs else all_starts
    train_starts = starts_fn(train_data, mcfg.block_size)
    val_starts = starts_fn(val_data, mcfg.block_size)

    model = GPT(mcfg).to(cfg.device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        fused=use_cuda,
    )
    step = 0
    if cfg.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
    elif cfg.init_from:
        model.load_state_dict(torch.load(cfg.init_from, map_location=cfg.device)["model"])

    log_path = ckpt_dir / "log.csv"
    new_log = not (cfg.resume and log_path.exists())
    log_file = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["step", "train_loss", "val_loss", "lr", "tokens_per_s"])

    tokens_per_step = cfg.batch_size * cfg.accum_steps * mcfg.block_size
    last_train, last_val = float("nan"), float("nan")
    t0 = time.time()
    tokens_since = 0
    model.train()
    while step < cfg.max_steps:
        lr = get_lr(step, cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        gen = torch.Generator().manual_seed(cfg.seed + step)
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(cfg.accum_steps):
            x, y = get_batch(train_data, cfg.batch_size, mcfg.block_size, cfg.device, gen, train_starts)
            with autocast:
                _, loss = model(x, y)
            (loss / cfg.accum_steps).backward()
            acc += loss.item() / cfg.accum_steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        step += 1
        tokens_since += tokens_per_step
        last_train = acc

        val = ""
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            last_val = evaluate(model, val_data, cfg, mcfg.block_size, autocast, val_starts)
            val = f"{last_val:.4f}"
        if step % cfg.log_interval == 0 or val:
            dt = time.time() - t0
            tps = tokens_since / dt if dt > 0 else 0.0
            t0, tokens_since = time.time(), 0
            log.writerow([step, f"{acc:.4f}", val, f"{lr:.6f}", f"{tps:.0f}"])
            log_file.flush()
            print(f"step {step} train {acc:.4f} val {val or '-'} lr {lr:.2e} {tps:.0f} tok/s", flush=True)
        if step % cfg.save_interval == 0 or step == cfg.max_steps:
            save_checkpoint(ckpt_path, model, opt, step, mcfg, cfg)

    log_file.close()
    return last_train, last_val


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    for f in fields(TrainConfig):
        name = "--" + f.name.replace("_", "-")
        if f.type is bool:
            p.add_argument(name, action="store_true")
        else:
            p.add_argument(name, type=f.type, default=f.default)
    return TrainConfig(**vars(p.parse_args(argv)))


if __name__ == "__main__":
    train(parse_args(), GPTConfig())
