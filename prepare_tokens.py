import json
import time
from itertools import islice
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).parent
DATA = ROOT / "data"
EOT = "<|endoftext|>"
CHUNK = 10_000


def encode_file(tok, src, dst, chunk=CHUNK):
    eot = tok.token_to_id(EOT)
    total = 0
    with open(src, encoding="utf-8") as f, open(dst, "wb") as out:
        while True:
            lines = list(islice(f, chunk))
            if not lines:
                break
            texts = [json.loads(line)["text"] for line in lines]
            ids = []
            for enc in tok.encode_batch(texts):
                ids.extend(enc.ids)
                ids.append(eot)
            np.asarray(ids, dtype=np.uint16).tofile(out)
            total += len(ids)
    return total


def main():
    tok = Tokenizer.from_file(str(ROOT / "tokenizer.json"))
    for split in ("validation", "test", "train"):
        t0 = time.time()
        n = encode_file(tok, DATA / f"{split}.jsonl", DATA / f"{split}.bin")
        print(f"{split}: {n} tokens en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
