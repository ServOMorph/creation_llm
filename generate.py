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
    return model, ckpt["step"]


def load_tokenizer():
    return Tokenizer.from_file(str(ROOT / "tokenizer.json"))


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


@torch.no_grad()
def generate(model, ids, max_new_tokens=200, temperature=0.8, top_k=50, top_p=0.95, seed=None):
    device = next(model.parameters()).device
    block = model.cfg.block_size
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits, _ = model(x[:, -block:])
        nxt = sample_next(logits[:, -1, :].float(), temperature, top_k, top_p, gen)
        x = torch.cat([x, nxt], dim=1)
        if nxt.item() == EOT_ID:
            break
    return x[0].tolist()


def complete(model, tok, prompt, prefix_eot=False, **kw):
    ids = ([EOT_ID] if prefix_eot else []) + tok.encode(prompt).ids
    out = generate(model, ids, **kw)
    return tok.decode([i for i in out if i != EOT_ID])
