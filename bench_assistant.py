import argparse
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).parent
BENCH = ROOT / "bench_assistant.jsonl"
HOLDOUT = ROOT / "prompts_assistant_holdout.txt"
RESULTS = ROOT / "bench_assistant_results"
CATS = ("faits", "consignes", "transformations", "conversation", "multi_tours")
SEEDS = (1234, 2000, 3000)
GEN = dict(max_new_tokens=300, temperature=0.5, top_k=50, top_p=0.95, repetition_penalty=1.0)
FLAGS = re.I | re.S

REFUSAL = re.compile(
    r"je ne (peux|suis) pas (en mesure de )?(r[ée]pondre|vous aider|t'aider|aider|fournir|satisfaire)"
    r"|je suis d[ée]sol[ée],? mais je ne (peux|suis)",
    re.I,
)
REPEAT_THRESHOLD = 0.95


def load_bench(path=BENCH):
    items = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [it["id"] for it in items]
    if len(ids) != len(set(ids)):
        raise ValueError("identifiants en double")
    for it in items:
        if it["cat"] not in CATS:
            raise ValueError(f"{it['id']} : catégorie inconnue {it['cat']}")
        if not it["turns"] or not all(isinstance(t, str) and t.strip() for t in it["turns"]):
            raise ValueError(f"{it['id']} : tours vides")
        if it["cat"] == "multi_tours" and len(it["turns"]) < 2:
            raise ValueError(f"{it['id']} : un scénario multi-tours exige au moins 2 tours")
        if not it.get("ok") and not it.get("ok_all"):
            raise ValueError(f"{it['id']} : aucun critère de réussite")
        for key in ("ok", "ok_all", "ko"):
            for pattern in it.get(key, []):
                re.compile(pattern, FLAGS)
    return items


def normalize(text):
    text = re.sub(r"[«»\"“”'’.,;:!?()\[\]*-]", " ", text.lower())
    return " ".join(text.split())


def similarity(a, b):
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def quoted(text):
    m = re.findall(r"«\s*(.+?)\s*»", text, re.S)
    return m[-1] if m else text


def is_copy(src, answer):
    return normalize(src) in (normalize(answer), normalize(quoted(answer)))


def is_refusal(answer):
    return bool(REFUSAL.search(answer))


def sentences(answer):
    return [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]


def has_loop(answer):
    sents = [s.strip() for s in sentences(answer) if len(s.strip()) > 15]
    lines = [line.strip() for line in answer.splitlines() if len(line.strip()) > 15]
    return len(sents) - len(set(sents)) >= 2 or len(lines) - len(set(lines)) >= 2


def n_words(answer):
    return len(re.findall(r"\w+", answer))


def score(item, answers):
    answer = answers[-1].strip()
    reasons = []
    if not answer:
        reasons.append("vide")
    if item.get("ok") and not any(re.search(p, answer, FLAGS) for p in item["ok"]):
        reasons.append("critere_absent")
    if any(not re.search(p, answer, FLAGS) for p in item.get("ok_all", [])):
        reasons.append("critere_incomplet")
    if any(re.search(p, answer, FLAGS) for p in item.get("ko", [])):
        reasons.append("critere_interdit")
    if is_refusal(answer):
        reasons.append("refus")
    if has_loop(answer):
        reasons.append("boucle")
    if item.get("src") and is_copy(item["src"], answer):
        reasons.append("recopie")
    if len(answers) > 1 and answers[-2].strip() and similarity(answers[-2], answer) >= REPEAT_THRESHOLD:
        reasons.append("repete_tour")
    if "max_mots" in item and n_words(answer) > item["max_mots"]:
        reasons.append("trop_long")
    if "max_phrases" in item and len(sentences(answer)) > item["max_phrases"]:
        reasons.append("trop_de_phrases")
    return {"correct": not reasons, "raisons": reasons}


def summarize(records):
    by_cat = defaultdict(lambda: [0, 0])
    flags = defaultdict(int)
    for r in records:
        by_cat[r["cat"]][0] += r["correct"]
        by_cat[r["cat"]][1] += 1
        for reason in r["raisons"]:
            flags[reason] += 1
    total = [sum(v[0] for v in by_cat.values()), sum(v[1] for v in by_cat.values())]
    return {
        "par_categorie": {c: {"correct": by_cat[c][0], "total": by_cat[c][1], "taux": round(by_cat[c][0] / by_cat[c][1], 3)} for c in CATS if c in by_cat},
        "global": {"correct": total[0], "total": total[1], "taux": round(total[0] / total[1], 3) if total[1] else 0.0},
        "raisons": dict(sorted(flags.items())),
    }


def run(ckpt, gen, seeds, device="cuda", items=None):
    from chat import Chat
    from generate import load_model, load_tokenizer

    items = items or load_bench()
    model, step = load_model(ckpt, device)
    chat = Chat(model, load_tokenizer(model.tokenizer_path), **gen)
    records = []
    for seed in seeds:
        for k, item in enumerate(items):
            chat.reset()
            answers = [chat.reply(turn, seed=seed + 10 * k + t) for t, turn in enumerate(item["turns"])]
            records.append({"id": item["id"], "cat": item["cat"], "seed": seed, "reponses": answers, **score(item, answers)})
    return step, records


def rescore(records, items=None):
    by_id = {it["id"]: it for it in (items or load_bench())}
    return [{**r, **score(by_id[r["id"]], r["reponses"])} for r in records]


def check_overlap(items=None):
    from prepare_sft import load_conversations, source_files

    items = items or load_bench()
    keys = {normalize(t): it["id"] for it in items for t in it["turns"]}
    hits = []
    if HOLDOUT.exists():
        hits += [(keys[normalize(q)], "holdout") for q in HOLDOUT.read_text(encoding="utf-8").splitlines() if normalize(q) in keys]
    for name, messages in load_conversations(source_files()):
        for m in messages:
            if m["role"] == "user" and normalize(m["content"]) in keys:
                hits.append((keys[normalize(m["content"])], name))
    return sorted(set(hits))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Benchmark absolu de l'assistant")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ckpt", default=str(ROOT / "checkpoints_sft" / "ckpt.pt"))
    r.add_argument("--tag", required=True)
    r.add_argument("--device", default="cuda")
    r.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    for key, val in GEN.items():
        r.add_argument(f"--{key.replace('_', '-')}", type=type(val), default=val)
    rs = sub.add_parser("rescore")
    rs.add_argument("tags", nargs="+")
    sub.add_parser("check-overlap")
    args = p.parse_args()

    RESULTS.mkdir(exist_ok=True)
    if args.cmd == "check-overlap":
        hits = check_overlap()
        print("aucun recouvrement" if not hits else "\n".join(f"{i} <- {src}" for i, src in hits))
        return
    if args.cmd == "rescore":
        for tag in args.tags:
            path = RESULTS / f"{tag}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["records"] = rescore(data["records"])
            data["resume"] = summarize(data["records"])
            path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            print(tag, json.dumps(data["resume"], ensure_ascii=False))
        return
    gen = {k: getattr(args, k) for k in GEN}
    step, records = run(args.ckpt, gen, args.seeds, args.device)
    data = {"tag": args.tag, "ckpt": args.ckpt, "pas": step, "gen": gen, "seeds": args.seeds, "resume": summarize(records), "records": records}
    (RESULTS / f"{args.tag}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(args.tag, json.dumps(data["resume"], ensure_ascii=False))


if __name__ == "__main__":
    main()
