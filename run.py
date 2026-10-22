import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent
DEFAULT_CKPT = ROOT / "checkpoints" / "ckpt.pt"


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="LLM français from scratch")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train", help="entraînement (options de TrainConfig, ex. --max-steps 10000 --resume)", add_help=False)

    g = sub.add_parser("generate", help="complétion de texte")
    g.add_argument("prompt")
    g.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    g.add_argument("--device", default=default_device())
    g.add_argument("--max-new-tokens", type=int, default=200)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--seed", type=int, default=None)

    e = sub.add_parser("eval", help="perplexité + prompts verrouillés -> eval_report.md")
    e.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    e.add_argument("--device", default=default_device())

    args, rest = p.parse_known_args()
    if args.cmd == "train":
        from model import GPTConfig
        from train import parse_args, train

        cfg = parse_args(rest)
        train(cfg, GPTConfig())
    elif args.cmd == "generate":
        from generate import complete, load_model, load_tokenizer

        model, _ = load_model(args.ckpt, args.device)
        print(
            complete(
                model,
                load_tokenizer(),
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
            )
        )
    else:
        from evaluate import run_eval

        val_loss, _ = run_eval(args.ckpt, args.device)
        print(f"loss validation {val_loss:.4f} ; rapport : eval_report.md")


if __name__ == "__main__":
    sys.exit(main())
