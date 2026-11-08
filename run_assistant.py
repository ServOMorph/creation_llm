import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    os.chdir(ROOT)
    sys.stdout.reconfigure(encoding="utf-8")
    import torch

    from chat import Chat
    from generate import load_model, load_tokenizer

    p = argparse.ArgumentParser(description="Assistant français en ligne de commande")
    p.add_argument("--ckpt", default=str(ROOT / "checkpoints_sft" / "ckpt.pt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    model, step = load_model(args.ckpt, args.device)
    chat = Chat(model, load_tokenizer(model.tokenizer_path), args.max_new_tokens, args.temperature, args.top_k, args.top_p)
    print(f"Assistant (pas {step}). /reset pour effacer l'historique, /quit pour sortir.")
    turn = 0
    while True:
        try:
            text = input("\nVous : ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            chat.reset()
            print("Historique effacé.")
            continue
        seed = None if args.seed is None else args.seed + turn
        answer = chat.reply(text, seed=seed)
        turn += 1
        note = f" [{chat.dropped} tour(s) ancien(s) retiré(s) du contexte]" if chat.dropped else ""
        print(f"\nAssistant : {answer}{note}")


if __name__ == "__main__":
    main()
