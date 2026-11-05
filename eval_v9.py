import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from generate import EOT_ID, complete, load_model, load_tokenizer

ROOT = Path(__file__).parent
MODELS = {"v8": ROOT / "checkpoints_v8" / "ckpt.pt", "v9": ROOT / "checkpoints_v9" / "ckpt.pt"}
DOMAINS = ["fw2hq", "wiki_fr", "vikidia", "lecarnet"]
STORIES = ROOT / "data" / "validation.jsonl"
PROMPTS = ROOT / "prompts_holdout3.txt"
SEEDS = [1234, 2000, 3000]
BLIND = ROOT / "eval_v9_blind.md"
KEY = ROOT / "eval_v9_key.json"
PPL = ROOT / "eval_v9_ppl.json"
DEVICE = "cuda"


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


@torch.no_grad()
def nll_sum(model, ids):
    x = torch.tensor([ids[:-1]], device=DEVICE)
    y = torch.tensor([ids[1:]], device=DEVICE)
    with autocast():
        logits, _ = model(x)
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="sum").item()


def domain_bpb(model, tok, name, windows=200):
    data = np.memmap(ROOT / "data" / "general" / f"{name}_val.bin", dtype=np.uint16, mode="r")
    block = model.cfg.block_size
    gen = np.random.default_rng(0)
    nll, n_tok, n_bytes = 0.0, 0, 0
    for i in gen.integers(0, len(data) - block - 1, windows):
        ids = data[i : i + block + 1].astype(np.int64).tolist()
        nll += nll_sum(model, ids)
        n_tok += block
        n_bytes += len(tok.decode(ids[1:]).encode("utf-8"))
    return {"loss": nll / n_tok, "ppl": math.exp(nll / n_tok), "bpb": nll / math.log(2) / n_bytes}


def story_texts(n=2000, max_v8_tokens=500):
    tok8 = load_tokenizer(ROOT / "tokenizer.json")
    out = []
    with open(STORIES, encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            if len(tok8.encode(text).ids) <= max_v8_tokens:
                out.append(text)
            if len(out) == n:
                break
    return out


def stories_bpb(model, tok, texts):
    nll, n_bytes = 0.0, 0
    for text in texts:
        nll += nll_sum(model, [EOT_ID] + tok.encode(text).ids)
        n_bytes += len(text.encode("utf-8"))
    return {"bpb": nll / math.log(2) / n_bytes}


def run_ppl():
    texts = story_texts()
    res = {}
    for name, path in MODELS.items():
        model, step = load_model(path, DEVICE)
        tok = load_tokenizer(model.tokenizer_path)
        res[name] = {"step": step, "stories_validation_jsonl": stories_bpb(model, tok, texts)}
        if name == "v9":
            for d in DOMAINS:
                res[name][d] = domain_bpb(model, tok, d)
        del model
        torch.cuda.empty_cache()
    res["n_stories"] = len(texts)
    PPL.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


def cut(text, n):
    if len(text) <= n:
        return text
    head = text[:n]
    return head[: head.rfind(" ")] if " " in head else head


def run_gen():
    prompts = [p for p in PROMPTS.read_text(encoding="utf-8").splitlines() if p.strip()]
    raw = {}
    for name, path in MODELS.items():
        model, _ = load_model(path, DEVICE)
        tok = load_tokenizer(model.tokenizer_path)
        raw[name] = {
            str(seed): [complete(model, tok, p, max_new_tokens=150, temperature=0.4, top_k=50, top_p=0.95, seed=seed + i) for i, p in enumerate(prompts)]
            for seed in SEEDS
        }
        del model
        torch.cuda.empty_cache()
    rng = random.Random(42)
    key, lines = [], ["# Évaluation à l'aveugle v9 (jeu prompts_holdout3.txt)", ""]
    for i, prompt in enumerate(prompts):
        for seed in SEEDS:
            a, b = raw["v8"][str(seed)][i], raw["v9"][str(seed)][i]
            n = min(len(a), len(b))
            pair = [("v8", cut(a, n)), ("v9", cut(b, n))]
            rng.shuffle(pair)
            item = f"P{i + 1}-s{seed}"
            key.append({"item": item, "A": pair[0][0], "B": pair[1][0]})
            lines += [f"## {item}", "", f"Prompt : {prompt}", "", f"### A", "", pair[0][1], "", f"### B", "", pair[1][1], ""]
    BLIND.write_text("\n".join(lines), encoding="utf-8")
    KEY.write_text(json.dumps({"key": key, "raw": raw}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("blind", len(key), "paires")


if __name__ == "__main__":
    {"ppl": run_ppl, "gen": run_gen}[sys.argv[1]]()
