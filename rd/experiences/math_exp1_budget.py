"""Pole mathematiques, experience 1 : budget d'information (CPU, lecture seule).

- entropies unigramme et bigramme du melange (reference sans reseau)
- octets par token
- longueur de code prequentielle des journaux v9 et 369M
- couverture du corpus par le tirage avec remise
"""
import csv
import json
import math
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "data" / "general"
MIX = {"fw2hq": 0.58, "wiki_fr": 0.25, "lecarnet": 0.15, "vikidia": 0.02}
V = 32768
LN2 = math.log(2)
rng = np.random.default_rng(0)


def sample(split, n_total, chunk=65536):
    out = []
    for name, w in MIX.items():
        d = np.memmap(G / f"{name}_{split}.bin", dtype=np.uint16, mode="r")
        n = int(n_total * w)
        k = max(1, n // chunk)
        starts = rng.integers(0, len(d) - chunk - 1, k)
        out.append((name, [np.asarray(d[s : s + chunk]).astype(np.int64) for s in starts]))
    return out


def ngram_reference():
    train = sample("train", 40_000_000)
    val = sample("val", 3_000_000)
    uni = np.zeros(V, dtype=np.int64)
    keys = []
    for _, chunks in train:
        for c in chunks:
            uni += np.bincount(c, minlength=V)
            keys.append(c[:-1] * V + c[1:])
    keys = np.concatenate(keys)
    uk, uc = np.unique(keys, return_counts=True)
    ctx = uk // V
    ctx_count = np.bincount(ctx, weights=uc, minlength=V)
    ctx_types = np.bincount(ctx, minlength=V)
    p_uni = (uni + 0.5) / (uni.sum() + 0.5 * V)
    D = 0.75
    res = {}
    tot_u = tot_b = tot_n = 0
    for name, chunks in val:
        hu = hb = n = 0
        for c in chunks:
            a, b = c[:-1], c[1:]
            k = a * V + b
            pos = np.searchsorted(uk, k)
            pos[pos >= len(uk)] = 0
            cnt = np.where(uk[pos] == k, uc[pos], 0)
            ca = ctx_count[a]
            lam = np.where(ca > 0, D * ctx_types[a] / np.maximum(ca, 1), 1.0)
            pb = np.maximum(cnt - D, 0) / np.maximum(ca, 1) + lam * p_uni[b]
            hu += -np.log(p_uni[b]).sum()
            hb += -np.log(pb).sum()
            n += len(b)
        res[name] = {"unigramme_nats": hu / n, "bigramme_nats": hb / n}
        w = MIX[name]
        tot_u += w * hu / n
        tot_b += w * hb / n
    res["melange"] = {"unigramme_nats": tot_u, "bigramme_nats": tot_b}
    return res


def bytes_per_token():
    tok = Tokenizer.from_file(str(ROOT / "tokenizer_fr32k.json"))
    res = {}
    mix = 0.0
    for name, w in MIX.items():
        d = np.memmap(G / f"{name}_val.bin", dtype=np.uint16, mode="r")
        nb = nt = 0
        for s in rng.integers(0, len(d) - 20000, 20):
            ids = np.asarray(d[s : s + 20000]).tolist()
            nb += len(tok.decode(ids, skip_special_tokens=False).encode("utf-8"))
            nt += len(ids)
        res[name] = nb / nt
        mix += w * nt / nb
    res["melange_harmonique"] = 1 / mix
    return res


def read_log(path):
    rows = list(csv.DictReader(open(path, newline="")))
    step = np.array([int(r["step"]) for r in rows])
    tr = np.array([float(r["train_loss"]) for r in rows])
    val = [(int(r["step"]), float(r["val_loss"])) for r in rows if r["val_loss"]]
    tps = np.array([float(r["tokens_per_s"]) for r in rows])
    return step, tr, val, tps


def prequential(path, n_params, tok_per_step=65536):
    step, tr, val, tps = read_log(path)
    full = np.interp(np.arange(1, step[-1] + 1), np.concatenate([[0], step]), np.concatenate([[math.log(V)], tr]))
    tokens = step[-1] * tok_per_step
    code_bits = full.sum() * tok_per_step / LN2
    final = val[-1][1]
    excess_bits = (full - final).sum() * tok_per_step / LN2
    return {
        "pas": int(step[-1]),
        "tokens": int(tokens),
        "val_finale_nats": final,
        "code_prequentiel_bits_par_token": code_bits / tokens,
        "exces_prequentiel_bits": excess_bits,
        "exces_bits_par_parametre": excess_bits / n_params,
        "exces_bits_par_token": excess_bits / tokens,
        "debit_median_tok_s": float(np.median(tps[5:])),
    }


def coverage(epochs_equiv, T=1024, L=2_000_000, trials=3):
    out = []
    for _ in range(trials):
        seen = np.zeros(L, dtype=np.int32)
        n = int(epochs_equiv * L / T)
        starts = rng.integers(0, L - T - 1, n)
        diff = np.zeros(L + 1, dtype=np.int32)
        np.add.at(diff, starts, 1)
        np.add.at(diff, starts + T, -1)
        seen = np.cumsum(diff)[:L]
        out.append([(seen == 0).mean(), (seen == 1).mean(), (seen >= 2).mean()])
    return np.mean(out, axis=0).tolist()


if __name__ == "__main__":
    res = {}
    res["octets_par_token"] = bytes_per_token()
    res["ngrammes"] = ngram_reference()
    res["v9"] = prequential(ROOT / "checkpoints_v9" / "log.csv", 20 * 12 * 768 * 768 + 2 * V * 768)
    res["369m"] = prequential(ROOT / "checkpoints_369m" / "log.csv", 24 * 12 * 1024 * 1024 + V * 1024)
    sizes = {"fw2hq": 4334618593}
    for name in MIX:
        sizes[name] = (G / f"{name}_train.bin").stat().st_size // 2
    cov = {}
    for name, w in MIX.items():
        e = w * 113000 * 65536 / sizes[name]
        c = coverage(e)
        cov[name] = {"tokens": sizes[name], "passes_equivalentes": e, "jamais_vu_simule": c[0], "vu_1_fois": c[1],
                     "vu_2_fois_ou_plus": c[2], "jamais_vu_theorique_exp(-e)": math.exp(-e)}
    res["couverture_369m"] = cov
    print(json.dumps(res, indent=1, ensure_ascii=False))
    (Path(__file__).parent / "math_exp1_budget.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
