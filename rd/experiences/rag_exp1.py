import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench_assistant import FLAGS, GEN, SEEDS, load_bench, score  # noqa: E402

HERE = Path(__file__).parent
PASSAGES = HERE / "rag_exp1_passages.json"
RESULTS = HERE / "rag_exp1_resultats.json"
VIKIDIA = sorted((ROOT / "data" / "raw" / "vikidia").glob("*.parquet"))
WORDS = 120
TOP_K = 3
K1, B = 1.5, 0.75
STOP = set(
    "a au aux avec ce ces dans de des du elle en et eux il ils je la le les leur lui ma mais me mes moi mon ne nos notre nous "
    "on ou par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos votre vous y est sont a-t-il quel quelle "
    "quels quelles combien comment pourquoi quand où d l s n c j m t".split()
)


def words(text):
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOP]


def split_passages(text, size=WORDS):
    toks = text.split()
    return [" ".join(toks[i : i + size]) for i in range(0, len(toks), size) if len(toks[i : i + size]) >= 20]


def load_passages():
    import pyarrow.parquet as pq

    out = []
    for path in VIKIDIA:
        for text in pq.read_table(path).column("text").to_pylist():
            out.extend(split_passages(text))
    return out


class BM25:
    def __init__(self, docs):
        self.n = len(docs)
        self.lens = [0] * self.n
        self.post = defaultdict(list)
        for i, doc in enumerate(docs):
            tf = Counter(words(doc))
            self.lens[i] = sum(tf.values())
            for w, c in tf.items():
                self.post[w].append((i, c))
        self.avg = sum(self.lens) / self.n

    def scores(self, query):
        acc = defaultdict(float)
        for w in set(words(query)):
            plist = self.post.get(w)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, c in plist:
                acc[i] += idf * c * (K1 + 1) / (c + K1 * (1 - B + B * self.lens[i] / self.avg))
        return acc


def matches(item, text):
    if item.get("ok") and not any(re.search(p, text, FLAGS) for p in item["ok"]):
        return False
    return all(re.search(p, text, FLAGS) for p in item.get("ok_all", []))


def build_index():
    docs = load_passages()
    bm = BM25(docs)
    out = {}
    for item in [it for it in load_bench() if it["cat"] == "faits"]:
        question = item["turns"][-1]
        acc = bm.scores(question)
        ranked = sorted(acc, key=acc.get, reverse=True)
        top = ranked[:TOP_K]
        oracle = next((i for i in ranked if matches(item, docs[i])), None)
        out[item["id"]] = {
            "question": question,
            "bm25": [docs[i] for i in top],
            "bm25_ttt": [docs[i] for i in ranked[:20]],
            "bm25_contient_reponse": [matches(item, docs[i]) for i in top],
            "oracle": docs[oracle] if oracle is not None else None,
            "oracle_rang": ranked.index(oracle) + 1 if oracle is not None else None,
        }
    PASSAGES.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    n = len(out)
    print(f"passages {len(docs)} ; items faits {n}")
    print("couverture oracle", sum(v["oracle"] is not None for v in out.values()), "/", n)
    print("reponse dans le top", TOP_K, "BM25 :", sum(any(v["bm25_contient_reponse"]) for v in out.values()), "/", n)
    print("reponse au rang 1 BM25 :", sum(v["bm25_contient_reponse"][:1] == [True] for v in out.values()), "/", n)


def prompt(passages, question):
    if not passages:
        return question
    return "Voici un texte :\n" + "\n".join(passages) + "\n\nQuestion : " + question


def run(ckpt, device, seeds):
    from chat import Chat
    from generate import load_model, load_tokenizer

    data = json.loads(PASSAGES.read_text(encoding="utf-8"))
    bench = load_bench()
    model, step = load_model(ckpt, device)
    chat = Chat(model, load_tokenizer(model.tokenizer_path), **GEN)
    conds = {
        "sans": lambda d: [],
        "oracle": lambda d: [d["oracle"]] if d["oracle"] else [],
        "bm25": lambda d: d["bm25"],
    }
    records = []
    for seed in seeds:
        for k, item in enumerate(bench):
            if item["cat"] != "faits":
                continue
            d = data[item["id"]]
            for name, pick in conds.items():
                chat.reset()
                answer = chat.reply(prompt(pick(d), item["turns"][-1]), seed=seed + 10 * k)
                records.append({"id": item["id"], "seed": seed, "cond": name, "reponse": answer, "tronque": chat.dropped, **score(item, [answer])})
    summary = {c: sum(r["correct"] for r in records if r["cond"] == c) for c in conds}
    total = len(records) // len(conds)
    pairs = defaultdict(dict)
    for r in records:
        pairs[(r["id"], r["seed"])][r["cond"]] = r["correct"]
    paired = {
        c: {"gagne": sum(p[c] and not p["sans"] for p in pairs.values()), "perd": sum(p["sans"] and not p[c] for p in pairs.values())}
        for c in ("oracle", "bm25")
    }
    out = {"ckpt": str(ckpt), "pas": step, "gen": GEN, "seeds": list(seeds), "total": total, "correct": summary, "apparie_contre_sans": paired, "records": records}
    RESULTS.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("pas", "total", "correct", "apparie_contre_sans")}, ensure_ascii=False))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    r = sub.add_parser("run")
    r.add_argument("--ckpt", default=str(ROOT / "checkpoints_sft_v2" / "ckpt.pt"))
    r.add_argument("--device", default="cuda")
    r.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = p.parse_args()
    if args.cmd == "index":
        build_index()
    else:
        run(args.ckpt, args.device, args.seeds)


if __name__ == "__main__":
    main()
