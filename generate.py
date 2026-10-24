from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig

ROOT = Path(__file__).parent
EOT_ID = 0


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    train_cfg = ckpt.get("train_cfg", {})
    model.tokenizer_path = train_cfg.get("tokenizer", str(ROOT / "tokenizer.json"))
    model.val_bin = train_cfg.get("val_bin", str(ROOT / "data" / "validation.bin"))
    return model, ckpt["step"]


def load_tokenizer(path=ROOT / "tokenizer.json"):
    return Tokenizer.from_file(str(path))


def sample_next(logits, temperature, top_k, top_p, gen):
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        remove = probs.cumsum(dim=-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=gen)


def prefill(model, x):
    caches = [{} for _ in model.blocks]
    logits, _ = model(x, caches=caches)
    return logits[:, -1, :].float(), caches


@torch.no_grad()
def sample(model, ids, n=1, max_new_tokens=200, temperature=0.8, top_k=50, top_p=0.95, seed=None):
    device = next(model.parameters()).device
    block = model.cfg.block_size
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.tensor([ids] * n, dtype=torch.long, device=device)
    logits, caches = prefill(model, x[:, -block:])
    outs = [list(ids) for _ in range(n)]
    score = torch.zeros(n, device=device)
    length = torch.zeros(n, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        nxt = sample_next(logits, temperature, top_k, top_p, gen)
        active = ~done
        score += torch.log_softmax(logits, dim=-1).gather(1, nxt).squeeze(1) * active
        length += active
        for i in active.nonzero().flatten().tolist():
            outs[i].append(nxt[i].item())
        done |= nxt.squeeze(1) == EOT_ID
        if done.all():
            break
        x = torch.cat([x, nxt], dim=1)
        if caches[0]["k"].size(2) < block:
            out, _ = model(nxt, caches=caches)
            logits = out[:, -1, :].float()
        else:
            logits, caches = prefill(model, x[:, -block:])
    return outs, score / length.clamp(min=1)


def generate(model, ids, max_new_tokens=200, temperature=0.8, top_k=50, top_p=0.95, seed=None):
    return sample(model, ids, 1, max_new_tokens, temperature, top_k, top_p, seed)[0][0]


def complete(model, tok, prompt, prefix_eot=False, best_of=1, **kw):
    ids = ([EOT_ID] if prefix_eot else []) + tok.encode(prompt).ids
    outs, scores = sample(model, ids, n=best_of, **kw)
    out = outs[int(scores.argmax())]
    return tok.decode([i for i in out if i != EOT_ID])
