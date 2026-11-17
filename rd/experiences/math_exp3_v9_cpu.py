"""Pole mathematiques, experience 3 : mesures sur la v9 (192M) en CPU, lecture seule.

A. Ou sont les bits : perte par position dans la fenetre, par classe de token, par source ;
   bruit d'etiquette V = 1 - somme(q^2) et calibration.
B. Echelle de bruit du gradient B* = tr(Sigma) / ||g||^2 par U-statistique entre groupes de sequences.

Usage : python math_exp3_v9_cpu.py A|B [n]
"""
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import GPT, GPTConfig

torch.set_num_threads(6)
G = ROOT / "data" / "general"
MIX = {"fw2hq": 0.58, "wiki_fr": 0.25, "lecarnet": 0.15, "vikidia": 0.02}
T = 1024


def load_v9():
    ck = torch.load(ROOT / "checkpoints_v9" / "ckpt.pt", map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["model_cfg"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    step = ck["step"]
    del ck
    return model, cfg, step


def seqs(split, name, n, rng):
    d = np.memmap(G / f"{name}_{split}.bin", dtype=np.uint16, mode="r")
    out = []
    for s in rng.integers(0, len(d) - T - 2, n):
        out.append(torch.from_numpy(np.asarray(d[s : s + T + 1]).astype(np.int64)))
    return torch.stack(out)


def part_a(n_per):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(ROOT / "tokenizer_fr32k.json"))
    vocab = [tok.id_to_token(i) or "" for i in range(32768)]
    cls = np.zeros(32768, dtype=np.int64)
    for i, t in enumerate(vocab):
        body = t.lstrip("ĠĊ")
        if i == 0 or body == "" or not any(ch.isalpha() for ch in body):
            cls[i] = 2 if any(ch.isdigit() for ch in body) else 3
        elif t.startswith("Ġ") or t.startswith("Ċ"):
            cls[i] = 0
        else:
            cls[i] = 1
    names = ["debut_de_mot", "suite_de_mot", "nombre", "ponctuation_et_autres"]
    cls_t = torch.from_numpy(cls)
    model, cfg, step = load_v9()
    model.eval()
    rng = np.random.default_rng(1)
    edges = [0, 4, 16, 64, 256, 512, 768, 1024]
    res = {"config": vars(cfg) | {"step": step, "params": model.num_params()}, "sources": {}}
    agg = {}
    t0 = time.time()
    for name, w in MIX.items():
        x = seqs("val", name, n_per[name], rng)
        nll, q_y, sumq2, ent = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(x), 2):
                logits, _ = model(x[i : i + 2, :-1])
                lp = F.log_softmax(logits.float(), -1)
                y = x[i : i + 2, 1:]
                l = -lp.gather(-1, y[..., None])[..., 0]
                p = lp.exp()
                nll.append(l)
                q_y.append((-l).exp())
                sumq2.append((p * p).sum(-1))
                ent.append(-(p * lp).sum(-1))
        nll, q_y, sumq2, ent = (torch.cat(v) for v in (nll, q_y, sumq2, ent))
        y = x[:, 1:]
        c = cls_t[y]
        r = {
            "n_sequences": len(x),
            "perte": nll.mean().item(),
            "erreur_type_entre_sequences": (nll.mean(1).std() / math.sqrt(len(x))).item(),
            "entropie_predite": ent.mean().item(),
            "V_bruit_etiquette_1-sum_q2": (1 - sumq2).mean().item(),
            "energie_gradient_logits_E|q-e_y|2": (1 - 2 * q_y + sumq2).mean().item(),
            "part_tokens_q_y>0.9": (q_y > 0.9).float().mean().item(),
            "part_perte_tokens_q_y>0.9": (nll[q_y > 0.9].sum() / nll.sum()).item(),
            "part_tokens_q_y<0.01": (q_y < 0.01).float().mean().item(),
            "part_perte_tokens_q_y<0.01": (nll[q_y < 0.01].sum() / nll.sum()).item(),
            "par_position": {f"{a}-{b}": nll[:, a:b].mean().item() for a, b in zip(edges[:-1], edges[1:])},
            "par_classe": {names[k]: {"part_tokens": (c == k).float().mean().item(), "perte_moyenne": nll[c == k].mean().item(),
                                     "part_perte": (nll[c == k].sum() / nll.sum()).item()} for k in range(4)},
        }
        res["sources"][name] = r
        for key in ("perte", "entropie_predite", "V_bruit_etiquette_1-sum_q2", "energie_gradient_logits_E|q-e_y|2",
                    "part_tokens_q_y>0.9", "part_tokens_q_y<0.01"):
            agg[key] = agg.get(key, 0) + w * r[key]
        for k, v in r["par_position"].items():
            agg.setdefault("par_position", {})
            agg["par_position"][k] = agg["par_position"].get(k, 0) + w * v
        for k in names:
            agg.setdefault("part_perte_par_classe", {})
            agg["part_perte_par_classe"][k] = agg["part_perte_par_classe"].get(k, 0) + w * r["par_classe"][k]["part_perte"] * r["perte"]
        print(name, round(time.time() - t0), "s", flush=True)
    tot = sum(agg["part_perte_par_classe"].values())
    agg["part_perte_par_classe"] = {k: v / tot for k, v in agg["part_perte_par_classe"].items()}
    res["melange"] = agg
    return res


def part_b(n_groups, per_group):
    model, cfg, step = load_v9()
    model.train()
    params = [(n, p) for n, p in model.named_parameters()]
    gen = torch.Generator().manual_seed(123)
    rng = np.random.default_rng(7)
    names = list(MIX)
    w = torch.tensor([MIX[n] for n in names])

    def kind(n):
        if not n.startswith("blocks."):
            return n.split(".")[0]
        return ".".join(n.split(".")[2:4])

    kinds = sorted({kind(n) for n, _ in params})
    sums = []
    sq = {k: 0.0 for k in kinds}
    losses = []
    t0 = time.time()
    for g in range(n_groups):
        acc = [torch.zeros_like(p) for _, p in params]
        for _ in range(per_group):
            src = names[torch.multinomial(w, 1, generator=gen).item()]
            x = seqs("train", src, 1, rng)
            model.zero_grad(set_to_none=True)
            _, loss = model(x[:, :-1], x[:, 1:])
            loss.backward()
            losses.append(loss.item())
            for a, (n, p) in zip(acc, params):
                a += p.grad
                sq[kind(n)] += p.grad.pow(2).sum().item()
            print(g, src, round(loss.item(), 3), round(time.time() - t0), "s", flush=True)
        sums.append(acc)
    n = n_groups * per_group
    res = {"n_sequences": n, "tokens_par_sequence": T, "perte_moyenne": float(np.mean(losses)), "par_groupe": {}}
    tot_cross, tot_sq = np.zeros((n_groups, n_groups)), 0.0
    for k in kinds:
        idx = [i for i, (nm, _) in enumerate(params) if kind(nm) == k]
        cross = np.zeros((n_groups, n_groups))
        for a in range(n_groups):
            for b in range(a + 1, n_groups):
                cross[a, b] = sum((sums[a][i] * sums[b][i]).sum().item() for i in idx) / per_group**2
        tot_cross += cross
        tot_sq += sq[k]
        res["par_groupe"][k] = summarize(cross, sq[k], n, n_groups)
    res["global"] = summarize(tot_cross, tot_sq, n, n_groups)
    return res


def summarize(cross, sq, n, n_groups):
    iu = np.triu_indices(n_groups, 1)
    vals = cross[iu]
    g2 = float(vals.mean())
    # jackknife par groupe sur l'estimation de ||g||^2
    jk = []
    for a in range(n_groups):
        keep = [(i, j) for i, j in zip(*iu) if i != a and j != a]
        jk.append(np.mean([cross[i, j] for i, j in keep]))
    se = float(np.sqrt((n_groups - 1) / n_groups * np.sum((np.array(jk) - np.mean(jk)) ** 2)))
    mean_sq = sq / n
    tr = mean_sq - g2
    out = {"norme2_gradient_vrai": g2, "erreur_type_jackknife": se, "moyenne_norme2_par_sequence": mean_sq}
    if g2 > 0:
        out["B_etoile_sequences"] = tr / g2
        out["B_etoile_tokens"] = tr / g2 * T
        lo, hi = g2 + 2 * se, g2 - 2 * se
        out["B_etoile_tokens_intervalle_2se"] = [(mean_sq - lo) / lo * T, (mean_sq - hi) / hi * T if hi > 0 else None]
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    out = Path(__file__).parent / f"math_exp3_v9_cpu_{mode}.json"
    if mode == "A":
        k = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        res = part_a({"fw2hq": 12 * k, "wiki_fr": 8 * k, "lecarnet": 4 * k, "vikidia": 4 * k})
    else:
        res = part_b(int(sys.argv[2]), int(sys.argv[3]))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
