import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from prepare_sft import load_split
from train import build_optimizer, save_checkpoint

ROOT = Path(__file__).parent
IGNORE = -100


@dataclass
class SFTConfig:
    data_dir: str = str(ROOT / "data" / "sft" / "prepared")
    init_from: str = str(ROOT / "checkpoints_v9" / "ckpt_chat.pt")
    ckpt_dir: str = str(ROOT / "checkpoints_sft")
    device: str = "cuda"
    batch_size: int = 2
    accum_steps: int = 32
    epochs: int = 2
    max_steps: int = 0
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_windows: int = 0
    save_interval: int = 250
    log_interval: int = 10
    seed: int = 1234
    resume: bool = False
    optimizer: str = "muon"
    compile: bool = False
    tokenizer: str = str(ROOT / "tokenizer_fr32k_chat.json")


def pack(lengths, cap, seed):
    lengths = np.asarray(lengths)
    if lengths.max(initial=0) > cap:
        raise ValueError(f"exemple de {lengths.max()} tokens > fenêtre de {cap}")
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.permutation(len(lengths)), -lengths))
    by_rem = [[] for _ in range(cap + 1)]
    bins = []
    for i in order.tolist():
        n = int(lengths[i])
        for r in range(n, cap + 1):
            if by_rem[r]:
                b = by_rem[r].pop()
                break
        else:
            b, r = len(bins), cap
            bins.append([])
        bins[b].append(i)
        by_rem[r - n].append(b)
    return [bins[k] for k in rng.permutation(len(bins))]


class PackedData:
    def __init__(self, data_dir, split, block_size):
        self.tokens, self.mask, self.offsets = load_split(Path(data_dir), split)
        self.lengths = np.diff(self.offsets)
        self.block_size = block_size
        self.cache = {}

    def windows(self, seed):
        if seed not in self.cache:
            self.cache[seed] = pack(self.lengths, self.block_size, seed)
        return self.cache[seed]

    def batch(self, windows, device):
        T = self.block_size
        n = len(windows)
        x = np.zeros((n, T), dtype=np.int64)
        y = np.full((n, T), IGNORE, dtype=np.int64)
        doc = np.full((n, T), -1, dtype=np.int64)
        for row, window in enumerate(windows):
            pos = 0
            for k, i in enumerate(window):
                s, e = self.offsets[i], self.offsets[i + 1]
                seq, m = self.tokens[s:e].astype(np.int64), self.mask[s:e]
                L = e - s
                x[row, pos : pos + L] = seq
                tgt = np.where(m[1:] == 1, seq[1:], IGNORE)
                y[row, pos : pos + L - 1] = tgt
                doc[row, pos : pos + L] = k
                pos += L
        return tuple(torch.from_numpy(a).to(device) for a in (x, y, doc))


def masked_loss_sum(model, x, y, doc):
    logits, _ = model(x, doc_ids=doc)
    loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=IGNORE, reduction="sum")
    return loss, int((y != IGNORE).sum())


class Schedule:
    def __init__(self, data, cfg):
        self.data = data
        self.cfg = cfg
        self.per_step = cfg.batch_size * cfg.accum_steps
        self.steps_per_epoch = [len(data.windows(cfg.seed + e)) // self.per_step for e in range(cfg.epochs)]
        self.total = self.lr_total = sum(self.steps_per_epoch)
        if cfg.max_steps:
            self.total = min(self.total, cfg.max_steps)

    def locate(self, step):
        for e, n in enumerate(self.steps_per_epoch):
            if step < n:
                return e, step
            step -= n
        raise IndexError(step)

    def micro_batches(self, step, device):
        e, local = self.locate(step)
        windows = self.data.windows(self.cfg.seed + e)
        start = local * self.per_step
        bs = self.cfg.batch_size
        for k in range(self.cfg.accum_steps):
            yield self.data.batch(windows[start + k * bs : start + (k + 1) * bs], device)


def get_lr(step, cfg, total):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    min_lr = cfg.lr * cfg.min_lr_ratio
    return min_lr + 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))) * (cfg.lr - min_lr)


@torch.no_grad()
def evaluate(model, val, cfg, autocast):
    model.eval()
    windows = val.windows(0)
    if cfg.eval_windows:
        windows = windows[: cfg.eval_windows]
    total, count = 0.0, 0
    for k in range(0, len(windows), cfg.batch_size):
        x, y, doc = val.batch(windows[k : k + cfg.batch_size], cfg.device)
        with autocast:
            loss, n = masked_loss_sum(model, x, y, doc)
        total += loss.item()
        count += n
    model.train()
    return total / max(1, count)


def train(cfg):
    torch.manual_seed(cfg.seed)
    use_cuda = cfg.device.startswith("cuda")
    autocast = torch.autocast(cfg.device.split(":")[0], dtype=torch.bfloat16, enabled=use_cuda)
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ckpt.pt"

    resuming = cfg.resume and ckpt_path.exists()
    src = torch.load(ckpt_path if resuming else cfg.init_from, map_location=cfg.device)
    mcfg = GPTConfig(**src["model_cfg"])
    model = GPT(mcfg).to(cfg.device)
    model.load_state_dict(src["model"])
    opt = build_optimizer(model, cfg, use_cuda)
    step = 0
    if resuming:
        opt.load_state_dict(src["optimizer"])
        step = src["step"]
    del src

    train_data = PackedData(cfg.data_dir, "train", mcfg.block_size)
    val_data = PackedData(cfg.data_dir, "val", mcfg.block_size)
    sched = Schedule(train_data, cfg)

    raw_model = model
    if cfg.compile:
        model = torch.compile(model)

    log_path = ckpt_dir / "log.csv"
    new_log = not (resuming and log_path.exists())
    log_file = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["step", "train_loss", "val_loss", "lr", "tokens_per_s"])
        log_file.flush()

    if new_log and step == 0:
        last_val = evaluate(model, val_data, cfg, autocast)
        log.writerow([0, "", f"{last_val:.4f}", "", ""])
        log_file.flush()
        print(f"step 0 val {last_val:.4f}", flush=True)

    last_train, last_val = float("nan"), float("nan")
    t0, tokens_since = time.time(), 0
    model.train()
    while step < sched.total:
        lr = get_lr(step, cfg, sched.lr_total)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        batches = list(sched.micro_batches(step, cfg.device))
        n_tgt = sum(int((y != IGNORE).sum()) for _, y, _ in batches)
        acc = 0.0
        for x, y, doc in batches:
            with autocast:
                loss, _ = masked_loss_sum(model, x, y, doc)
            (loss / n_tgt).backward()
            acc += loss.item() / n_tgt
            tokens_since += x.numel()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
        opt.step()
        step += 1
        last_train = acc

        val = ""
        if step % cfg.eval_interval == 0 or step == sched.total:
            last_val = evaluate(model, val_data, cfg, autocast)
            val = f"{last_val:.4f}"
        if step % cfg.log_interval == 0 or val:
            dt = time.time() - t0
            tps = tokens_since / dt if dt > 0 else 0.0
            t0, tokens_since = time.time(), 0
            log.writerow([step, f"{acc:.4f}", val, f"{lr:.6f}", f"{tps:.0f}"])
            log_file.flush()
            print(f"step {step}/{sched.total} train {acc:.4f} val {val or '-'} lr {lr:.2e} {tps:.0f} tok/s", flush=True)
        if step % cfg.save_interval == 0 or step == sched.total:
            save_checkpoint(ckpt_path, raw_model, opt, step, mcfg, cfg)

    log_file.close()
    return last_train, last_val


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    for f in fields(SFTConfig):
        name = "--" + f.name.replace("_", "-")
        if f.type is bool:
            p.add_argument(name, action="store_true")
        else:
            p.add_argument(name, type=f.type, default=f.default)
    return SFTConfig(**vars(p.parse_args(argv)))


if __name__ == "__main__":
    train(parse_args())
