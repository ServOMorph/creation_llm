"""SFT du 369M, recette de la v2 en deux étapes. À lancer par gpu_task.py, une étape à la fois.
Pause VertIA : sortie en code PAUSE_CODE, gpu_task.py rend le jeton puis relance après un nouvel acquire.

Préalable (CPU) : python convert_chat.py --src checkpoints_369m/ckpt.pt --dst checkpoints_369m/ckpt_chat.pt
python gpu_task.py --gb 6.5 --minutes 1440 -- .venv/Scripts/python run_sft_369m.py --etape 1
python gpu_task.py --gb 6.5 --minutes 1440 -- .venv/Scripts/python run_sft_369m.py --etape 2
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
COMMON = [
    "--batch-size", "2", "--accum-steps", "32", "--min-lr-ratio", "0.1", "--optimizer", "muon", "--compile",
    "--grad-checkpoint", "--vram-cap-gb", "6.1",
    "--log-interval", "10", "--eval-interval", "250", "--save-interval", "100", "--resume",
    "--vram-session", "creation_llm", "--pause-check-s", "60",
]
ETAPES = {
    1: ["--data-dir", "data/sft/prepared", "--init-from", "checkpoints_369m/ckpt_chat.pt",
        "--ckpt-dir", "checkpoints_sft_369m_1", "--epochs", "2", "--lr", "3e-4", "--warmup-steps", "100"],
    2: ["--data-dir", "data/sft/prepared_v2", "--init-from", "checkpoints_sft_369m_1/ckpt.pt",
        "--ckpt-dir", "checkpoints_sft_369m", "--epochs", "1", "--lr", "1e-4", "--warmup-steps", "50"],
}


def build_args(etape, extra=()):
    return [*COMMON, *ETAPES[etape], *extra]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--etape", type=int, choices=sorted(ETAPES), required=True)
    a, extra = p.parse_known_args(argv)
    os.chdir(ROOT)
    sys.stdout.reconfigure(encoding="utf-8")
    from train_sft import parse_args, train
    from vram_guard import PAUSE_CODE, TrainingPaused

    try:
        train(parse_args(build_args(a.etape, extra)))
    except TrainingPaused:
        sys.exit(PAUSE_CODE)


if __name__ == "__main__":
    main()
