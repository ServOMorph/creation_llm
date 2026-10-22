from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).parent / "data"


def main():
    OUT.mkdir(exist_ok=True)
    ds = load_dataset("MaxLSB/LeCarnet")
    for split, d in ds.items():
        d = d.filter(lambda r: len(r["text"].strip()) > 0)
        d.to_json(str(OUT / f"{split}.jsonl"), force_ascii=False)
        chars = sum(len(t) for t in d["text"])
        print(f"{split}: {len(d)} histoires, {chars / 1e6:.1f} M caracteres")


if __name__ == "__main__":
    main()
