import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench_assistant import GEN, SEEDS, load_bench, score  # noqa: E402
from convert_chat import ASSISTANT, USER  # noqa: E402
from generate import EOT_ID, load_model, load_tokenizer, prefill, sample_next  # noqa: E402

RESULTS = Path(__file__).parent / "cfg_exp2_resultats.json"
GAMMAS = (1.0, 1.25, 1.5)
CATS = ("faits", "consignes")


@torch.no_grad()
def cfg_sample(model, cond_ids, unc_ids, gamma, seed, stop_ids, max_new_tokens, temperature, top_k, top_p, **_):
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    stop = set(stop_ids)
    lc, cc = prefill(model, torch.tensor([cond_ids], device=device))
    use_unc = gamma != 1.0
    if use_unc:
        lu, cu = prefill(model, torch.tensor([unc_ids], device=device))
    out = []
    for _ in range(max_new_tokens):
        logits = lu + gamma * (lc - lu) if use_unc else lc
        nxt = sample_next(logits, temperature, top_k, top_p, gen)
        tok = nxt.item()
        if tok in stop:
            break
        out.append(tok)
        lc = model(nxt, caches=cc)[0][:, -1, :].float()
        if use_unc:
            lu = model(nxt, caches=cu)[0][:, -1, :].float()
    return out


def run(ckpt, device, seeds, gammas, limit):
    bench = load_bench()
    model, step = load_model(ckpt, device)
    tok = load_tokenizer(model.tokenizer_path)
    user, assistant = tok.token_to_id(USER), tok.token_to_id(ASSISTANT)
    stop = (EOT_ID, user, assistant)
    unc = [user, assistant]
    records = []
    for seed in seeds:
        done = 0
        for k, item in enumerate(bench):
            if item["cat"] not in CATS or len(item["turns"]) != 1:
                continue
            if limit and done >= limit:
                break
            done += 1
            cond = [user, *tok.encode(item["turns"][0]).ids, assistant]
            for gamma in gammas:
                ids = cfg_sample(model, cond, unc, gamma, seed + 10 * k, stop, **GEN)
                answer = tok.decode(ids).strip()
                records.append({"id": item["id"], "cat": item["cat"], "seed": seed, "gamma": gamma, "reponse": answer, **score(item, [answer])})
    summary = {
        str(g): {c: [sum(r["correct"] for r in records if r["gamma"] == g and r["cat"] == c), sum(1 for r in records if r["gamma"] == g and r["cat"] == c)] for c in CATS}
        for g in gammas
    }
    pairs = defaultdict(dict)
    for r in records:
        pairs[(r["id"], r["seed"], r["cat"])][r["gamma"]] = r["correct"]
    paired = {
        str(g): {
            c: {
                "gagne": sum(p[g] and not p[1.0] for key, p in pairs.items() if key[2] == c),
                "perd": sum(p[1.0] and not p[g] for key, p in pairs.items() if key[2] == c),
            }
            for c in CATS
        }
        for g in gammas
        if g != 1.0 and 1.0 in gammas
    }
    reasons = {str(g): dict(sorted(Counter(x for r in records if r["gamma"] == g for x in r["raisons"]).items())) for g in gammas}
    out = {"ckpt": str(ckpt), "pas": step, "gen": GEN, "seeds": list(seeds), "correct_sur_total": summary, "apparie_contre_gamma_1": paired, "raisons": reasons, "records": records}
    RESULTS.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("pas", "correct_sur_total", "apparie_contre_gamma_1", "raisons")}, ensure_ascii=False))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "checkpoints_sft_v2" / "ckpt.pt"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--gammas", type=float, nargs="+", default=list(GAMMAS))
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    run(a.ckpt, a.device, a.seeds, a.gammas, a.limit)


if __name__ == "__main__":
    main()
