import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from compare_frenchbench import compare, load, main  # noqa: E402

REF = {"french_bench_grammar": {"acc,none": 0.739}, "french_bench_vocab": {"acc,none": 0.689},
       "french_bench_hellaswag": {"acc,none": 0.29, "acc_norm,none": 0.317}}


def cand(grammar, vocab):
    return {"french_bench_grammar": {"acc,none": grammar}, "french_bench_vocab": {"acc,none": vocab},
            "french_bench_hellaswag": {"acc,none": 0.1, "acc_norm,none": 0.1}}


def test_gate_only_on_grammar_and_vocab():
    rows, failed = compare(REF, cand(0.74, 0.69))
    assert failed == []
    assert dict((t, round(r, 1)) for t, r, _ in rows)["french_bench_hellaswag"] == 31.7


def test_gate_fails_below_reference():
    assert compare(REF, cand(0.73, 0.70))[1] == ["french_bench_grammar"]


def test_margin():
    assert compare(REF, cand(0.66, 0.62), marge=8)[1] == []
    assert compare(REF, cand(0.65, 0.62), marge=8)[1] == ["french_bench_grammar"]


def test_real_v9_against_itself():
    if not (Path(__file__).parent.parent / "eval_frenchbench_v9_0shot.json").exists():
        return
    assert round(load("v9")["french_bench_grammar"]["acc,none"] * 100, 1) == 73.9
    assert main(["v9", "v9"]) == 0
