import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from model import GPTConfig

ROOT = Path(__file__).parent
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
EOT = "<|endoftext|>"
CHAT_TOKENS = (USER, ASSISTANT)
SRC_TOKENIZER = ROOT / "tokenizer_fr32k.json"
CHAT_TOKENIZER = ROOT / "tokenizer_fr32k_chat.json"
CHAT_VOCAB = 32832


def build_chat_tokenizer(src=SRC_TOKENIZER, dst=CHAT_TOKENIZER):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(src))
    base = tok.get_vocab_size()
    tok.add_special_tokens(list(CHAT_TOKENS))
    ids = [tok.token_to_id(t) for t in CHAT_TOKENS]
    if ids != list(range(base, base + len(CHAT_TOKENS))):
        raise ValueError(f"ids inattendus pour les balises : {ids}")
    tok.save(str(dst))
    return tok


def expand_vocab(state, new_vocab):
    new = dict(state)
    for key in ("wte.weight", "lm_head.weight"):
        w = state[key]
        if new_vocab < w.size(0):
            raise ValueError(f"{key} : vocab {w.size(0)} > {new_vocab}")
        extra = w.mean(dim=0, keepdim=True).expand(new_vocab - w.size(0), -1)
        new[key] = torch.cat([w, extra], dim=0)
    return new


def convert_checkpoint(src, dst, new_vocab=CHAT_VOCAB):
    ckpt = torch.load(src, map_location="cpu")
    cfg = GPTConfig(**ckpt["model_cfg"])
    state = expand_vocab(ckpt["model"], new_vocab)
    old_vocab, cfg.vocab_size = cfg.vocab_size, new_vocab
    torch.save({"model": state, "step": 0, "model_cfg": asdict(cfg), "train_cfg": {}, "source_step": ckpt["step"]}, dst)
    return old_vocab, new_vocab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(ROOT / "checkpoints_v9" / "ckpt.pt"))
    p.add_argument("--dst", default=str(ROOT / "checkpoints_v9" / "ckpt_chat.pt"))
    p.add_argument("--vocab", type=int, default=CHAT_VOCAB)
    args = p.parse_args()
    tok = build_chat_tokenizer()
    print("tokenizer", CHAT_TOKENIZER.name, tok.get_vocab_size(), {t: tok.token_to_id(t) for t in CHAT_TOKENS})
    old, new = convert_checkpoint(args.src, args.dst, args.vocab)
    print(f"checkpoint {old} -> {new} : {args.dst}")


if __name__ == "__main__":
    main()
