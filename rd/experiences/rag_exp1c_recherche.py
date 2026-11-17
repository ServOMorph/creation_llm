import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from bench_assistant import load_bench  # noqa: E402
from rag_exp1 import STOP, VIKIDIA, WORDS, matches  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "rag_exp1c_resultats.json"
K1, B = 1.5, 0.75


def fold(w):
    w = unicodedata.normalize("NFD", w)
    return "".join(c for c in w if unicodedata.category(c) != "Mn")


def stem(w):
    w = fold(w)
    for suf in ("ement", "ments", "ment", "ations", "ation", "euses", "euse", "eurs", "eur", "ees", "ee", "es", "s", "x", "e"):
        if len(w) - len(suf) >= 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def toks(text, stemmed):
    out = [w for w in re.findall(r"\w+", text.lower()) if w not in STOP and len(w) > 1]
    return [stem(w) for w in out] if stemmed else out


def load():
    import pyarrow.parquet as pq

    docs, art, lead = [], [], []
    a = 0
    for path in VIKIDIA:
        for text in pq.read_table(path).column("text").to_pylist():
            words = text.split()
            for j, i in enumerate(range(0, len(words), WORDS)):
                chunk = words[i : i + WORDS]
                if len(chunk) < 20:
                    continue
                docs.append(" ".join(chunk))
                art.append(a)
                lead.append(j == 0)
            a += 1
    return docs, art, lead


class BM25:
    def __init__(self, texts, stemmed):
        self.stemmed = stemmed
        self.n = len(texts)
        self.lens = []
        self.post = defaultdict(list)
        for i, t in enumerate(texts):
            tf = Counter(toks(t, stemmed))
            self.lens.append(sum(tf.values()) or 1)
            for w, c in tf.items():
                self.post[w].append((i, c))
        self.avg = sum(self.lens) / self.n

    def scores(self, query):
        acc = defaultdict(float)
        for w in set(toks(query, self.stemmed)):
            plist = self.post.get(w)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, c in plist:
                acc[i] += idf * c * (K1 + 1) / (c + K1 * (1 - B + B * self.lens[i] / self.avg))
        return acc


def top(acc, k=3):
    return sorted(acc, key=acc.get, reverse=True)[:k]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    docs, art, lead = load()
    items = [it for it in load_bench() if it["cat"] == "faits"]
    lead_idx = [i for i in range(len(docs)) if lead[i]]
    art_lead = {art[i]: k for k, i in enumerate(lead_idx)}
    idx = {s: BM25(docs, s) for s in (False, True)}
    idx_lead = {s: BM25([docs[i] for i in lead_idx], s) for s in (False, True)}

    def passages(stemmed):
        return lambda q: top(idx[stemmed].scores(q))

    def lead_only(stemmed):
        return lambda q: [lead_idx[k] for k in top(idx_lead[stemmed].scores(q))]

    def combined(stemmed, w):
        def f(q):
            sp = idx[stemmed].scores(q)
            sl = idx_lead[stemmed].scores(q)
            cands = set(sp) | {lead_idx[k] for k in sl}
            score = {i: sp.get(i, 0.0) + w * sl.get(art_lead.get(art[i], -1), 0.0) + (w * 0.5 * sl.get(art_lead[art[i]], 0.0) if lead[i] else 0.0) for i in cands}
            return top(score)
        return f

    variants = {
        "A_bm25_passages": passages(False),
        "B_bm25_racines": passages(True),
        "C_debuts_articles": lead_only(False),
        "D_debuts_articles_racines": lead_only(True),
        "E_passage_plus_article_racines_w1": combined(True, 1.0),
        "F_passage_plus_article_racines_w2": combined(True, 2.0),
    }
    out = {}
    for name, fn in variants.items():
        rows = {}
        for it in items:
            ids = fn(it["turns"][-1])
            rows[it["id"]] = {"hits": [matches(it, docs[i]) for i in ids], "top1": docs[ids[0]] if ids else ""}
        t1 = sum(r["hits"][:1] == [True] for r in rows.values())
        t3 = sum(any(r["hits"]) for r in rows.values())
        out[name] = {"top1": t1, "top3": t3, "detail": rows}
        print(f"{name:38s} top1 {t1:2d}/30  top3 {t3:2d}/30")
    RESULTS.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
