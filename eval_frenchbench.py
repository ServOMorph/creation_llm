import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from lm_eval import simple_evaluate
from lm_eval.api.model import LM
from tqdm import tqdm

from generate import EOT_ID, load_model, load_tokenizer

ROOT = Path(__file__).parent
MODELS = {"v8": ROOT / "checkpoints_v8" / "ckpt.pt", "v9": ROOT / "checkpoints_v9" / "ckpt.pt", "sft": ROOT / "checkpoints_sft" / "ckpt.pt"}
MC_TASKS = [
    "french_bench_arc_challenge", "french_bench_hellaswag", "french_bench_grammar", "french_bench_vocab",
    "french_bench_reading_comp", "french_bench_boolqa", "french_bench_fquadv2_bool",
    "french_bench_topic_based_nli", "french_bench_xnli",
]


class LocalGPT(LM):
    def __init__(self, ckpt):
        super().__init__()
        self.model, _ = load_model(ckpt, "cuda")
        self.tok = load_tokenizer(self.model.tokenizer_path)
        self.block = self.model.cfg.block_size

    def encode(self, text):
        return self.tok.encode(text).ids if text else []

    @torch.no_grad()
    def logprobs(self, ids, n_target):
        ids = ids[-(self.block + 1):]
        x = torch.tensor([ids[:-1]], device="cuda")
        y = torch.tensor(ids[1:], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = self.model(x)
        logits = logits[0, -n_target:].float()
        lp = F.log_softmax(logits, dim=-1).gather(1, y[-n_target:, None]).sum().item()
        greedy = bool((logits.argmax(-1) == y[-n_target:]).all())
        return lp, greedy

    def loglikelihood(self, requests):
        out = []
        for req in tqdm(requests):
            ctx, cont = req.args
            ctx_ids = [EOT_ID] + self.encode(ctx)
            whole = [EOT_ID] + self.encode(ctx + cont)
            n = len(whole) - len(ctx_ids)
            if n <= 0 or whole[: len(ctx_ids)] != ctx_ids:
                cont_ids = self.encode(cont)
                whole, n = ctx_ids + cont_ids, len(cont_ids)
            out.append(self.logprobs(whole, min(n, self.block)))
        return out

    def loglikelihood_rolling(self, requests):
        out = []
        for req in tqdm(requests):
            ids = [EOT_ID] + self.encode(req.args[0])
            total, pos = 0.0, 1
            while pos < len(ids):
                end = min(pos + self.block, len(ids))
                start = max(0, end - self.block - 1)
                total += self.logprobs(ids[start:end], end - pos)[0]
                pos = end
            out.append(total)
        return out

    def generate_until(self, requests):
        raise NotImplementedError


def main(name, fewshot):
    lm = LocalGPT(MODELS[name])
    res = simple_evaluate(model=lm, tasks=MC_TASKS, num_fewshot=fewshot, bootstrap_iters=0)["results"]
    path = ROOT / f"eval_frenchbench_{name}_{fewshot}shot.json"
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for task, metrics in res.items():
        print(task, {k: v for k, v in metrics.items() if isinstance(v, float)})


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
