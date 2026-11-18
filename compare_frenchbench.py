"""Compare deux résultats FrenchBench et applique le gate d'une phase.

python compare_frenchbench.py v9 369m              gate phase 3 : grammaire et vocabulaire >= référence
Préalable, chaque éval par gpu_task.py :
python gpu_task.py --gb 3 --minutes 90 -- .venv/Scripts/python eval_frenchbench.py 369m
python gpu_task.py --gb 3 --minutes 30 -- .venv/Scripts/python run.py eval --ckpt checkpoints_369m/ckpt.pt --report eval_369m.md
python compare_frenchbench.py 369m sft_369m --marge 8   gate phase 4 : écart <= 8 points
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
GATE_TASKS = ("french_bench_grammar", "french_bench_vocab")


def load(name, root=ROOT / "resultats", fewshot=0):
    return json.loads((root / f"eval_frenchbench_{name}_{fewshot}shot.json").read_text(encoding="utf-8"))


def score(metrics):
    return 100 * metrics.get("acc_norm,none", metrics["acc,none"])


def compare(ref, cand, marge=0.0, gate=GATE_TASKS):
    rows = [(t, score(ref[t]), score(cand[t])) for t in ref if t in cand]
    failed = [t for t, r, c in rows if t in gate and c < r - marge]
    return rows, failed


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("ref")
    p.add_argument("cand")
    p.add_argument("--marge", type=float, default=0.0)
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    rows, failed = compare(load(a.ref), load(a.cand), a.marge)
    print(f"{'tâche':32} {a.ref:>8} {a.cand:>8}  écart")
    for t, r, c in rows:
        print(f"{t.removeprefix('french_bench_'):32} {r:8.1f} {c:8.1f} {c - r:+6.1f}{'  gate' if t in GATE_TASKS else ''}")
    print("gate : " + ("échec sur " + ", ".join(failed) if failed else "réussi"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
