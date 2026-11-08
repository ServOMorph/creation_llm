import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent

ARGS = [
    "--batch-size", "2", "--accum-steps", "32", "--epochs", "2",
    "--lr", "3e-4", "--warmup-steps", "100", "--min-lr-ratio", "0.1", "--optimizer", "muon", "--compile",
    "--log-interval", "10", "--eval-interval", "250", "--save-interval", "100",
    "--ckpt-dir", "checkpoints_sft", "--resume",
]

if __name__ == "__main__":
    os.chdir(ROOT)
    sys.stdout.reconfigure(encoding="utf-8")
    from train_sft import parse_args, train

    train(parse_args([*ARGS, *sys.argv[1:]]))
