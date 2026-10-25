import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
G = "data/general"
MIX = {"fw2hq": 0.58, "wiki_fr": 0.25, "lecarnet": 0.15, "vikidia": 0.02}


def spec(split):
    return ",".join(f"{G}/{name}_{split}.bin={w}" for name, w in MIX.items())


ARGS = [
    "--vocab-size", "32768", "--block-size", "1024", "--n-layer", "20", "--n-embd", "768", "--n-head", "12",
    "--batch-size", "2", "--accum-steps", "32", "--max-steps", "91500",
    "--lr", "3e-3", "--warmup-steps", "1000", "--min-lr-ratio", "0.1", "--optimizer", "muon", "--compile",
    "--log-interval", "100", "--eval-interval", "1000", "--eval-batches", "40", "--save-interval", "500",
    "--train-bin", spec("train"), "--val-bin", spec("val"),
    "--tokenizer", "tokenizer_fr32k.json", "--ckpt-dir", "checkpoints_v9", "--resume",
]

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, str(ROOT / "run.py"), "train", *ARGS, *sys.argv[1:]], cwd=ROOT))
