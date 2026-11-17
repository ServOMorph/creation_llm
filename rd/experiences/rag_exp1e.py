import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from bench_assistant import GEN, SEEDS, load_bench, score  # noqa: E402
from rag_exp1 import prompt  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "rag_exp1e_resultats.json"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from chat import Chat
    from generate import load_model, load_tokenizer

    hyb = json.loads((HERE / "rag_exp1d_resultats.json").read_text(encoding="utf-8"))["hybride"]["detail"]
    ref = {(r["id"], r["seed"]): r["correct"] for r in json.loads((HERE / "rag_exp1_resultats.json").read_text(encoding="utf-8"))["records"] if r["cond"] == "sans"}
    model, step = load_model(str(ROOT / "checkpoints_sft_v2" / "ckpt.pt"), "cuda")
    chat = Chat(model, load_tokenizer(model.tokenizer_path), **GEN)
    conds = {"hybride_1": lambda d: d["top3"][:1], "hybride_3": lambda d: d["top3"]}
    records = []
    for seed in SEEDS:
        for k, item in enumerate(load_bench()):
            if item["cat"] != "faits":
                continue
            for name, pick in conds.items():
                chat.reset()
                answer = chat.reply(prompt(pick(hyb[item["id"]]), item["turns"][-1]), seed=seed + 10 * k)
                records.append({"id": item["id"], "seed": seed, "cond": name, "reponse": answer, "tronque": chat.dropped, "sans": ref[(item["id"], seed)], **score(item, [answer])})
    summary = defaultdict(lambda: {"avec": 0, "sans": 0, "total": 0, "gagne": 0, "perd": 0})
    for r in records:
        s = summary[r["cond"]]
        s["avec"] += r["correct"]
        s["sans"] += r["sans"]
        s["total"] += 1
        s["gagne"] += r["correct"] and not r["sans"]
        s["perd"] += r["sans"] and not r["correct"]
    RESULTS.write_text(json.dumps({"pas": step, "gen": GEN, "seeds": list(SEEDS), "resume": summary, "records": records}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
