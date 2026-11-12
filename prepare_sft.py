import argparse
import json
import random
import re
from pathlib import Path

import numpy as np

from convert_chat import ASSISTANT, CHAT_TOKENIZER, EOT, USER

ROOT = Path(__file__).parent
SFT_RAW = ROOT / "data" / "sft" / "luth-sft" / "data"
SFT_OUT = ROOT / "data" / "sft" / "prepared"
HOLDOUT = ROOT / "prompts_assistant_holdout.txt"
SOURCES = ("openhermes", "smoltalk2", "tulu3_persona_instruct", "croissantllm", "aya_dataset")
RESERVED = (EOT, USER, ASSISTANT)
MAX_LEN = 1024
VAL_SIZE = 2000
SEED = 1234


def source_files(raw=SFT_RAW, extra=()):
    files = {name: sorted(raw.glob(f"luth_{name}-*.parquet")) for name in SOURCES}
    for path in extra:
        files[Path(path).stem] = [Path(path)]
    return files


def read_messages(path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line)["messages"] for line in f if line.strip()]
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=["messages"]).column("messages").to_pylist()


def load_conversations(files):
    for name, paths in files.items():
        for path in paths:
            for messages in read_messages(Path(path)):
                yield name, [{"role": m["role"], "content": m["content"].strip()} for m in messages]


def is_valid(messages):
    if len(messages) < 2 or len(messages) % 2:
        return False
    for i, m in enumerate(messages):
        if m["role"] != ("user" if i % 2 == 0 else "assistant") or not m["content"]:
            return False
    return True


def has_reserved(messages):
    return any(r in m["content"] for m in messages for r in RESERVED)


REFUSAL = re.compile(
    r"(je suis d[ée]sol[ée],?|d[ée]sol[ée],?|malheureusement,?)\s+(mais\s+)?je ne (peux|suis) pas"
    r"|je ne (peux|suis) pas (en mesure de )?(r[ée]pondre|vous aider|t'aider|fournir|acc[ée]der)"
    r"|en tant que (mod[èe]le de langage|intelligence artificielle|ia)",
    re.I,
)


def is_refusal(messages, head=300):
    return any(REFUSAL.search(m["content"][:head]) for m in messages if m["role"] == "assistant")


def question_key(messages):
    return " ".join(messages[0]["content"].split())


def build_example(turn_ids, user_id, assistant_id, eot_id):
    ids, mask = [], []
    for i, content in enumerate(turn_ids):
        if i % 2 == 0:
            ids += [user_id, *content, assistant_id]
            mask += [0] * (len(content) + 2)
        else:
            ids += [*content, eot_id]
            mask += [1] * (len(content) + 1)
    return ids, mask


def fitting_turns(turn_ids, max_len):
    used = n = 0
    for i in range(0, len(turn_ids) - 1, 2):
        used += len(turn_ids[i]) + len(turn_ids[i + 1]) + 3
        if used > max_len:
            break
        n = i + 2
    return n


def load_holdout(path=HOLDOUT):
    if not path.exists():
        return set()
    return {" ".join(line.split()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def prepare(files, tok, max_len=MAX_LEN, val_size=VAL_SIZE, seed=SEED, holdout=frozenset(), batch=20_000, truncate=False, drop_refusals=False, repeat=None):
    user_id, assistant_id, eot_id = (tok.token_to_id(t) for t in (USER, ASSISTANT, EOT))
    stats = {name: {"lignes": 0, "invalides": 0, "balises": 0, "refus": 0, "trop_longs": 0, "tronqués": 0, "doublons": 0, "holdout": 0} for name in files}
    convs = []
    for name, messages in load_conversations(files):
        s = stats[name]
        s["lignes"] += 1
        if not is_valid(messages):
            s["invalides"] += 1
        elif has_reserved(messages):
            s["balises"] += 1
        elif drop_refusals and is_refusal(messages):
            s["refus"] += 1
        else:
            convs.append((name, messages))

    examples = []
    for start in range(0, len(convs), batch):
        chunk = convs[start : start + batch]
        texts = [m["content"] for _, messages in chunk for m in messages]
        encs = iter(tok.encode_batch(texts))
        for name, messages in chunk:
            turn_ids = [next(encs).ids for _ in messages]
            if truncate:
                n = fitting_turns(turn_ids, max_len)
                if 0 < n < len(turn_ids):
                    stats[name]["tronqués"] += 1
                    turn_ids, messages = turn_ids[:n], messages[:n]
            ids, mask = build_example(turn_ids, user_id, assistant_id, eot_id)
            if len(ids) > max_len:
                stats[name]["trop_longs"] += 1
            else:
                examples.append((name, question_key(messages), np.array(ids, dtype=np.uint16), np.array(mask, dtype=np.uint8)))

    seen, kept = set(), []
    for ex in examples:
        name, key = ex[0], ex[1]
        if key in holdout:
            stats[name]["holdout"] += 1
        elif key in seen:
            stats[name]["doublons"] += 1
        else:
            seen.add(key)
            kept.append(ex)

    random.Random(seed).shuffle(kept)
    splits = {"val": kept[:val_size], "train": kept[val_size:]}
    repeat = repeat or {}
    if repeat:
        train = [ex for ex in splits["train"] for _ in range(repeat.get(ex[0], 1))]
        random.Random(seed + 1).shuffle(train)
        splits["train"] = train
    for name in files:
        s = stats[name]
        s["gardés"] = sum(1 for ex in kept if ex[0] == name)
        s["répétition"] = repeat.get(name, 1)
        s["train"] = sum(1 for ex in splits["train"] if ex[0] == name)
    return splits, stats


def write_split(examples, out_dir, split):
    offsets = np.zeros(len(examples) + 1, dtype=np.int64)
    for i, ex in enumerate(examples):
        offsets[i + 1] = offsets[i] + len(ex[2])
    tokens = np.concatenate([ex[2] for ex in examples]) if examples else np.zeros(0, dtype=np.uint16)
    mask = np.concatenate([ex[3] for ex in examples]) if examples else np.zeros(0, dtype=np.uint8)
    tokens.tofile(out_dir / f"{split}_tokens.bin")
    mask.tofile(out_dir / f"{split}_mask.bin")
    np.save(out_dir / f"{split}_offsets.npy", offsets)
    lengths = np.diff(offsets)
    return {
        "exemples": len(examples),
        "tokens": int(offsets[-1]),
        "tokens_réponse": int(mask.sum()),
        "longueur_moyenne": round(float(lengths.mean()), 1) if len(examples) else 0,
        "longueur_max": int(lengths.max()) if len(examples) else 0,
    }


def load_split(out_dir, split):
    offsets = np.load(out_dir / f"{split}_offsets.npy")
    tokens = np.fromfile(out_dir / f"{split}_tokens.bin", dtype=np.uint16)
    mask = np.fromfile(out_dir / f"{split}_mask.bin", dtype=np.uint8)
    return tokens, mask, offsets


def main():
    from tokenizers import Tokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(SFT_OUT))
    p.add_argument("--max-len", type=int, default=MAX_LEN)
    p.add_argument("--val-size", type=int, default=VAL_SIZE)
    p.add_argument("--truncate", action="store_true", help="tronquer les conversations trop longues au dernier tour complet")
    p.add_argument("--drop-refusals", action="store_true", help="écarter les conversations dont une réponse commence par un refus")
    p.add_argument("--extra", nargs="*", default=[], help="sources supplémentaires .jsonl (champ messages)")
    p.add_argument("--repeat", nargs="*", default=[], help="suréchantillonnage du train, ex: transfo=4")
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(CHAT_TOKENIZER))
    splits, stats = prepare(source_files(extra=args.extra), tok, args.max_len, args.val_size, holdout=load_holdout(), truncate=args.truncate, drop_refusals=args.drop_refusals,
                           repeat={k: int(v) for k, v in (r.split("=") for r in args.repeat)})
    result = {"sources": stats, "max_len": args.max_len, "seed": SEED, "tronqué": args.truncate, "sans_refus": args.drop_refusals, "masque": "1 = token cible de la loss"}
    for split, examples in splits.items():
        result[split] = write_split(examples, out_dir, split)
    (out_dir / "stats.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
