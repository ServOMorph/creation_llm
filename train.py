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
    optimizer: str = "adamw"
    compile: bool = False


class MultiOptimizer:
    def __init__(self, opts):
        self.opts = opts

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        states = [o.state_dict() for o in self.opts]
        return states if len(states) > 1 else states[0]

    def load_state_dict(self, state):
        for o, s in zip(self.opts, state if isinstance(state, list) else [state]):
            o.load_state_dict(s)


def build_optimizer(model, cfg, use_cuda):
    adam_kw = dict(lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay, fused=use_cuda)
    if cfg.optimizer == "adamw":
        return MultiOptimizer([torch.optim.AdamW(model.parameters(), **adam_kw)])
    if cfg.optimizer != "muon":
        raise ValueError(f"optimiseur inconnu : {cfg.optimizer}")
    hidden = [p for n, p in model.named_parameters() if n.startswith("blocks.")]
    other = [p for n, p in model.named_parameters() if not n.startswith("blocks.")]
    return MultiOptimizer(
        [
            torch.optim.Muon(hidden, lr=cfg.lr, weight_decay=cfg.weight_decay, adjust_lr_fn="match_rms_adamw"),
            torch.optim.AdamW(other, **adam_kw),
        ]
    )


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


class MixedSource:
    def __init__(self, spec):
        parts = [p.partition("=") for p in spec.split(",")]
        self.sources = [np.memmap(path, dtype=np.uint16, mode="r") for path, _, _ in parts]
        self.weights = torch.tensor([float(w or 1) for _, _, w in parts])

    def batch(self, batch_size, block_size, device, gen):
        picks = torch.multinomial(self.weights, batch_size, replacement=True, generator=gen).tolist()
        rows = []
        for k in picks:
            src = self.sources[k]
            i = torch.randint(len(src) - block_size - 1, (1,), generator=gen).item()
            rows.append(torch.from_numpy(src[i : i + block_size + 1].astype(np.int64)))
        rows = torch.stack(rows)
        return rows[:, :-1].contiguous().to(device), rows[:, 1:].contiguous().to(device)


def load_data(spec):
    if "," in spec or "=" in spec:
        return MixedSource(spec)
    return np.memmap(spec, dtype=np.uint16, mode="r")


def get_batch(data, batch_size, block_size, device, gen, starts=None):
    if isinstance(data, MixedSource):
        return data.batch(batch_size, block_size, device, gen)
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

    train_data = load_data(cfg.train_bin)
    val_data = load_data(cfg.val_bin)
    starts_fn = doc_starts if cfg.align_docs else all_starts
    train_starts = None if isinstance(train_data, MixedSource) else starts_fn(train_data, mcfg.block_size)
    val_starts = None if isinstance(val_data, MixedSource) else starts_fn(val_data, mcfg.block_size)

    model = GPT(mcfg).to(cfg.device)
    opt = build_optimizer(model, cfg, use_cuda)
    step = 0
    if cfg.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
    elif cfg.init_from:
        model.load_state_dict(torch.load(cfg.init_from, map_location=cfg.device)["model"])

    raw_model = model
    if cfg.compile:
        model = torch.compile(model)

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
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
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
            save_checkpoint(ckpt_path, raw_model, opt, step, mcfg, cfg)

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
