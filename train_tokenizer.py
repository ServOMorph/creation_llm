import json
from itertools import islice
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "tokenizer.json"
VOCAB_SIZE = 8192
N_STORIES = 300_000
EOT = "<|endoftext|>"


def iter_texts(path, n):
    with open(path, encoding="utf-8") as f:
        for line in islice(f, n):
            yield json.loads(line)["text"]


def train(n_stories=N_STORIES, vocab_size=VOCAB_SIZE, out=OUT):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(iter_texts(DATA / "train.jsonl", n_stories), trainer)
    tok.save(str(out))
    return tok


if __name__ == "__main__":
    tok = train()
    print("vocab:", tok.get_vocab_size())
