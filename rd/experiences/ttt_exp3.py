import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench_assistant import GEN, SEEDS, load_bench, score  # noqa: E402
from chat import Chat  # noqa: E402
from generate import load_model, load_tokenizer  # noqa: E402

HERE = Path(__file__).parent
PASSAGES = HERE / "rag_exp1_passages.json"
RESULTS = HERE / "ttt_exp3_resultats.json"
LRS = (0.0, 1e-5, 3e-5, 1e-4)


def adapt(model, tok, passages, lr, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    model.train()
    losses = []
    for text in passages:
        ids = [0, *tok.encode(text).ids][: model.cfg.block_size + 1]
        x = torch.tensor([ids[:-1]], device=device)
        y = torch.tensor([ids[1:]], device=device)
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    model.eval()
    return losses


def run(ckpt, device, seeds, lrs, limit):
    data = json.loads(PASSAGES.read_text(encoding="utf-8"))
    bench = load_bench()
    model, step = load_model(ckpt, device)
    tok = load_tokenizer(model.tokenizer_path)
    chat = Chat(model, tok, **GEN)
    base = {k: v.clone() for k, v in model.state_dict().items()}
    records = []
    done = 0
    for k, item in enumerate(bench):
        if item["cat"] != "faits":
            continue
        if limit and done >= limit:
            break
        done += 1
        for lr in lrs:
            losses = adapt(model, tok, data[item["id"]]["bm25_ttt"], lr, device) if lr > 0 else []
            for seed in seeds:
                chat.reset()
                answer = chat.reply(item["turns"][-1], seed=seed + 10 * k)
                records.append({
                    "id": item["id"], "seed": seed, "lr": lr, "reponse": answer,
                    "perte_debut": losses[0] if losses else None, "perte_fin": losses[-1] if losses else None,
                    **score(item, [answer]),
                })
            if lr > 0:
                model.load_state_dict(base)
    total = len(records) // len(lrs)
    summary = {str(lr): sum(r["correct"] for r in records if r["lr"] == lr) for lr in lrs}
    pairs = defaultdict(dict)
    for r in records:
        pairs[(r["id"], r["seed"])][r["lr"]] = r["correct"]
    paired = {
        str(lr): {"gagne": sum(p[lr] and not p[0.0] for p in pairs.values()), "perd": sum(p[0.0] and not p[lr] for p in pairs.values())}
        for lr in lrs
        if lr > 0 and 0.0 in lrs
    }
    flags = {str(lr): {x: sum(x in r["raisons"] for r in records if r["lr"] == lr) for x in ("boucle", "vide")} for lr in lrs}
    out = {"ckpt": str(ckpt), "pas": step, "gen": GEN, "seeds": list(seeds), "total": total, "correct": summary, "apparie_contre_lr_0": paired, "boucle_vide": flags, "records": records}
    RESULTS.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("pas", "total", "correct", "apparie_contre_lr_0", "boucle_vide")}, ensure_ascii=False))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "checkpoints_sft_v2" / "ckpt.pt"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--lrs", type=float, nargs="+", default=list(LRS))
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    run(a.ckpt, a.device, a.seeds, a.lrs, a.limit)


if __name__ == "__main__":
    main()
