import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfFileSystem, snapshot_download  # noqa: E402

FW2HQ = "datasets/epfml/FineWeb2-HQ/fra_Latn"


def fw2hq_file(path, out_dir):
    dst = out_dir / Path(path).name
    if dst.exists():
        return dst.name, "déjà présent"
    tmp = dst.with_suffix(".tmp")
    fs = HfFileSystem(skip_instance_cache=True)
    with fs.open(path, "rb", block_size=8 * 1024 * 1024) as f:
        pf = pq.ParquetFile(f)
        with pq.ParquetWriter(tmp, pa.schema([("text", pa.string())]), compression="zstd") as w:
            for i in range(pf.num_row_groups):
                w.write_table(pf.read_row_group(i, columns=["text"]))
    tmp.replace(dst)
    return dst.name, "ok"


def fetch_fw2hq(n_files, workers):
    fs = HfFileSystem()
    files = sorted(fs.ls(FW2HQ, detail=False))
    step = max(1, len(files) // n_files)
    chosen = files[::step][:n_files]
    out_dir = RAW / "fw2hq"
    out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        for name, status in ex.map(lambda p: fw2hq_file(p, out_dir), chosen):
            print(name, status, flush=True)


def fetch_wiki():
    from datasets import load_dataset

    out_dir = RAW / "wiki_fr"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train")
    ds = ds.map(lambda r: {"text": r["title"] + "\n\n" + r["text"]}, remove_columns=ds.column_names, num_proc=4)
    for i in range(16):
        ds.shard(16, i, contiguous=True).to_parquet(str(out_dir / f"wiki_{i:02d}.parquet"))
    print("wiki_fr", len(ds), "articles")


def clean_vikidia(text):
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_vikidia():
    local = snapshot_download("FrancophonIA/Vikidia-EnFr", repo_type="dataset", allow_patterns=["fr/*", "frsimple/*"])
    out_dir = RAW / "vikidia"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("fr", "frsimple"):
        texts = [clean_vikidia(p.read_text(encoding="utf-8", errors="ignore")) for p in sorted(Path(local, sub).glob("*.txt"))]
        texts = [t for t in texts if len(t) > 200]
        pq.write_table(pa.table({"text": texts}), out_dir / f"vikidia_{sub}.parquet", compression="zstd")
        print("vikidia", sub, len(texts), "articles")


EOT = "<|endoftext|>"
TOKENIZER = ROOT / "tokenizer_fr32k.json"
GENERAL = ROOT / "data" / "general"
SOURCES = ("fw2hq", "wiki_fr", "vikidia", "lecarnet")


def parquet_texts(paths, limit=None):
    n = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        for i in range(pf.num_row_groups):
            for text in pf.read_row_group(i, columns=["text"]).column("text").to_pylist():
                yield text
                n += 1
                if limit is not None and n >= limit:
                    return


def source_files():
    return {name: sorted((RAW / name).glob("*.parquet")) for name in SOURCES}


def export_lecarnet():
    out_dir = RAW / "lecarnet"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "data" / "train.jsonl", encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f]
    pq.write_table(pa.table({"text": texts}), out_dir / "lecarnet_train.parquet", compression="zstd")
    print("lecarnet", len(texts), "histoires")


def train_tokenizer(vocab_size=32768):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    files = source_files()

    def sample():
        yield from parquet_texts(files["fw2hq"][:3], 300_000)
        yield from parquet_texts(files["wiki_fr"], 150_000)
        yield from parquet_texts(files["vikidia"], 50_000)
        yield from parquet_texts(files["lecarnet"], 150_000)

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(sample(), trainer)
    tok.save(str(TOKENIZER))
    print("tokenizer", tok.get_vocab_size())


def write_batch(tok, texts, eot, seen, val_docs, f_train, f_val, counts):
    train_ids, val_ids = [], []
    for enc in tok.encode_batch(texts):
        target = val_ids if seen < val_docs else train_ids
        target.extend(enc.ids)
        target.append(eot)
        seen += 1
    for ids, f, split in ((train_ids, f_train, "train"), (val_ids, f_val, "val")):
        if ids:
            np.asarray(ids, dtype=np.uint16).tofile(f)
            counts[split] += len(ids)
    return seen


def encode_sources(val_docs=10_000, batch=2_000, only=None):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER))
    eot = tok.token_to_id(EOT)
    GENERAL.mkdir(parents=True, exist_ok=True)
    for name, paths in source_files().items():
        if only and name not in only:
            continue
        counts = {"train": 0, "val": 0}
        seen = 0
        with open(GENERAL / f"{name}_train.bin", "wb") as f_train, open(GENERAL / f"{name}_val.bin", "wb") as f_val:
            buf = []
            for text in parquet_texts(paths):
                buf.append(text)
                if len(buf) == batch:
                    seen = write_batch(tok, buf, eot, seen, val_docs, f_train, f_val, counts)
                    buf = []
            if buf:
                write_batch(tok, buf, eot, seen, val_docs, f_train, f_val, counts)
        print(name, counts, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["fw2hq", "wiki", "vikidia", "lecarnet", "tokenizer", "encode"])
    p.add_argument("--n-files", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", nargs="*")
    p.add_argument("--val-docs", type=int, default=10_000)
    args = p.parse_args()
    if args.step == "fw2hq":
        fetch_fw2hq(args.n_files, args.workers)
    elif args.step == "wiki":
        fetch_wiki()
    elif args.step == "vikidia":
        fetch_vikidia()
    elif args.step == "lecarnet":
        export_lecarnet()
    elif args.step == "tokenizer":
        train_tokenizer()
    else:
        encode_sources(val_docs=args.val_docs, only=args.only)
