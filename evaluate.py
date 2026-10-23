import math
from pathlib import Path

import numpy as np
import torch

from generate import complete, load_model, load_tokenizer
from train import TrainConfig, all_starts, evaluate

ROOT = Path(__file__).parent
PROMPTS = ROOT / "prompts_eval.txt"
REPORT = ROOT / "eval_report.md"
SEED = 1234


def distinct_n(tokens, n=3):
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def run_eval(ckpt_path, device, max_new_tokens=150, eval_batches=200, report=REPORT, temperature=0.7, prefix_eot=False):
    model, step = load_model(ckpt_path, device)
    tok = load_tokenizer()
    cfg = TrainConfig(device=device, eval_batches=eval_batches)
    val_data = np.memmap(cfg.val_bin, dtype=np.uint16, mode="r")
    autocast = torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda"))
    val_loss = evaluate(model, val_data, cfg, model.cfg.block_size, autocast, all_starts(val_data, model.cfg.block_size))

    prompts = [p for p in PROMPTS.read_text(encoding="utf-8").splitlines() if p.strip()]
    lines = [
        "# Rapport d'évaluation",
        "",
        f"- Checkpoint : pas {step}",
        f"- Loss de validation : {val_loss:.4f}",
        f"- Perplexité de validation : {math.exp(val_loss):.2f}",
        f"- Température : {temperature}",
        "",
    ]
    ratios = []
    for i, prompt in enumerate(prompts):
        text = complete(model, tok, prompt, prefix_eot=prefix_eot, max_new_tokens=max_new_tokens, temperature=temperature, top_k=50, top_p=0.95, seed=SEED + i)
        ratios.append(distinct_n(tok.encode(text).ids))
        lines += [f"## Prompt {i + 1}", "", text, ""]
    lines.insert(7,f"- Diversité moyenne (3-grammes distincts) : {sum(ratios) / len(ratios):.3f}")
    report.write_text("\n".join(lines), encoding="utf-8")
    return val_loss, ratios
