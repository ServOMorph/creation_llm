"""Pré-entraînement du modèle 369M (24 x 1024) sous jeton GPU VertIA.

python run_369m.py                                  pré-entraînement continu, reprise automatique
python run_369m.py --cycles 2 --token-hours 0.5     rodage : 2 cycles de 30 min avec pause et reprise
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from vram_guard import run_guarded

ROOT = Path(__file__).parent
G = "data/general"
MIX = {"fw2hq": 0.58, "wiki_fr": 0.25, "lecarnet": 0.15, "vikidia": 0.02}
SESSION = "creation_llm"
VRAM_GB = 6.5
PRIORITY = 9


def spec(split):
    return ",".join(f"{G}/{name}_{split}.bin={w}" for name, w in MIX.items())


ARGS = [
    "--vocab-size", "32768", "--block-size", "1024", "--n-layer", "24", "--n-embd", "1024", "--n-head", "16", "--tie-embeddings",
    "--batch-size", "2", "--accum-steps", "32", "--max-steps", "113000", "--vram-cap-gb", "6.1",
    "--lr", "3e-3", "--warmup-steps", "1000", "--min-lr-ratio", "0.1", "--optimizer", "muon", "--compile",
    "--grad-checkpoint",
    "--log-interval", "10", "--eval-interval", "250", "--eval-batches", "40",
    "--save-interval", "500", "--save-minutes", "15",
    "--vram-session", SESSION, "--pause-check-s", "60",
    "--train-bin", spec("train"), "--val-bin", spec("val"),
    "--tokenizer", "tokenizer_fr32k.json", "--ckpt-dir", "checkpoints_369m", "--resume",
]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=0)
    a, extra = p.parse_known_args(argv)
    os.chdir(ROOT)
    cmd = [sys.executable, str(ROOT / "run.py"), "train", *ARGS, *extra]
    return run_guarded(SESSION, VRAM_GB, 1440, cmd, priority=PRIORITY, max_cycles=a.cycles, run=subprocess.run)


if __name__ == "__main__":
    sys.exit(main())
