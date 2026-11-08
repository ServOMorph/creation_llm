import json
import random
import sys
from pathlib import Path

import torch

from chat import Chat
from generate import complete, load_model, load_tokenizer

ROOT = Path(__file__).parent
ASSISTANT_CKPT = ROOT / "checkpoints_sft" / "ckpt.pt"
BASE_CKPT = ROOT / "checkpoints_v9" / "ckpt.pt"
PROMPTS = ROOT / "prompts_assistant_holdout.txt"
SEEDS = [1234, 2000]
BLIND = ROOT / "eval_assistant_blind.md"
KEY = ROOT / "eval_assistant_key.json"
GEN = dict(max_new_tokens=300, temperature=0.5, top_k=50, top_p=0.95)
DEVICE = "cuda"


def base_prompt(question):
    return f"Question : {question}\nRéponse :"


def trim_base(text, prompt):
    answer = text[len(prompt):] if text.startswith(prompt) else text
    return answer.split("\nQuestion")[0].strip()


def run_assistant(prompts):
    model, _ = load_model(ASSISTANT_CKPT, DEVICE)
    chat = Chat(model, load_tokenizer(model.tokenizer_path), **GEN)
    out = {}
    for seed in SEEDS:
        answers = []
        for i, q in enumerate(prompts):
            chat.reset()
            answers.append(chat.reply(q, seed=seed + i))
        out[str(seed)] = answers
    del model
    torch.cuda.empty_cache()
    return out


def run_base(prompts):
    model, _ = load_model(BASE_CKPT, DEVICE)
    tok = load_tokenizer(model.tokenizer_path)
    out = {}
    for seed in SEEDS:
        out[str(seed)] = [trim_base(complete(model, tok, base_prompt(q), prefix_eot=True, seed=seed + i, **GEN), base_prompt(q)) for i, q in enumerate(prompts)]
    del model
    torch.cuda.empty_cache()
    return out


def write_blind(prompts, raw, rng_seed=42):
    rng = random.Random(rng_seed)
    key, lines = [], ["# Évaluation à l'aveugle assistant contre v9 de base (prompts_assistant_holdout.txt)", ""]
    for i, q in enumerate(prompts):
        for seed in SEEDS:
            pair = [(name, raw[name][str(seed)][i]) for name in ("assistant", "base")]
            rng.shuffle(pair)
            item = f"Q{i + 1}-s{seed}"
            key.append({"item": item, "A": pair[0][0], "B": pair[1][0]})
            lines += [f"## {item}", "", f"Question : {q}", "", "### A", "", pair[0][1], "", "### B", "", pair[1][1], ""]
    BLIND.write_text("\n".join(lines), encoding="utf-8")
    KEY.write_text(json.dumps({"key": key, "raw": raw, "gen": GEN, "base_prompt": base_prompt("{question}")}, indent=2, ensure_ascii=False), encoding="utf-8")
    return key


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    prompts = [p for p in PROMPTS.read_text(encoding="utf-8").splitlines() if p.strip()]
    raw = {"assistant": run_assistant(prompts), "base": run_base(prompts)}
    key = write_blind(prompts, raw)
    print("blind", len(key), "paires ->", BLIND.name)


if __name__ == "__main__":
    main()
