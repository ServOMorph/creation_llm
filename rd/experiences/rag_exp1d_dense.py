import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from bench_assistant import load_bench  # noqa: E402
from rag_exp1 import matches  # noqa: E402
from rag_exp1c_recherche import BM25, load  # noqa: E402

HERE = Path(__file__).parent
MODEL = "intfloat/multilingual-e5-small"
PER_ARTICLE = 3
EMB = HERE / "rag_exp1d_emb.npy"
RESULTS = HERE / "rag_exp1d_resultats.json"


def encode(tok, model, texts, prefix, bs=64):
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            batch = tok([prefix + t for t in texts[i : i + bs]], padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = model(**batch).last_hidden_state
            m = batch["attention_mask"].unsqueeze(-1).float()
            out.append(F.normalize((h * m).sum(1) / m.sum(1), dim=-1).numpy().astype(np.float32))
    return np.concatenate(out)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(os.cpu_count() or 4)
    docs, art, _ = load()
    seen, keep = {}, []
    for i, a in enumerate(art):
        if seen.get(a, 0) < PER_ARTICLE:
            keep.append(i)
            seen[a] = seen.get(a, 0) + 1
    sub = [docs[i] for i in keep]
    print("passages retenus", len(sub), "sur", len(docs), flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    t0 = time.time()
    if EMB.exists() and np.load(EMB, mmap_mode="r").shape[0] == len(sub):
        emb = np.load(EMB)
    else:
        emb = encode(tok, model, sub, "passage: ")
        np.save(EMB, emb)
    print(f"encodage {time.time() - t0:.0f} s", flush=True)
    items = [it for it in load_bench() if it["cat"] == "faits"]
    qs = encode(tok, model, [it["turns"][-1] for it in items], "query: ")
    bm = BM25(sub, True)
    res = {}
    for name in ("bm25_meme_sous_ensemble", "dense", "hybride"):
        rows = {}
        for it, q in zip(items, qs):
            dense = emb @ q
            if name == "dense":
                ids = np.argsort(-dense)[:3].tolist()
            else:
                acc = bm.scores(it["turns"][-1])
                if name == "bm25_meme_sous_ensemble":
                    ids = sorted(acc, key=acc.get, reverse=True)[:3]
                else:
                    cand = set(sorted(acc, key=acc.get, reverse=True)[:100]) | set(np.argsort(-dense)[:100].tolist())
                    mx = max(acc.values()) if acc else 1.0
                    ids = sorted(cand, key=lambda i: dense[i] + 0.1 * acc.get(i, 0.0) / mx, reverse=True)[:3]
            rows[it["id"]] = {"hits": [matches(it, sub[i]) for i in ids], "top1": sub[ids[0]], "top3": [sub[i] for i in ids]}
        t1 = sum(r["hits"][:1] == [True] for r in rows.values())
        t3 = sum(any(r["hits"]) for r in rows.values())
        res[name] = {"top1": t1, "top3": t3, "detail": rows}
        print(f"{name:26s} top1 {t1:2d}/30  top3 {t3:2d}/30", flush=True)
    RESULTS.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
