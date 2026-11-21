import argparse
import ctypes
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import train  # noqa: E402
import train_sft  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
IGNORE = train_sft.IGNORE


def lower_priority():
    torch.set_num_threads(1)
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)


def small_model(seed=0, vocab=64):
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=vocab, block_size=128, n_layer=4, n_head=4, n_embd=128)
    model = GPT(cfg)
    for name, p in model.named_parameters():
        if name.endswith("proj.weight"):
            torch.nn.init.normal_(p, std=0.05)
    return model.eval(), cfg


def synthetic_packed(docs):
    p = train_sft.PackedData.__new__(train_sft.PackedData)
    tokens, mask, offsets = [], [], [0]
    for toks, m in docs:
        tokens += toks
        mask += m
        offsets.append(len(tokens))
    p.tokens = np.asarray(tokens, dtype=np.uint16)
    p.mask = np.asarray(mask, dtype=np.uint8)
    p.offsets = np.asarray(offsets)
    p.lengths = np.diff(p.offsets)
    p.block_size = 16
    p.cache = {}
    return p


def check_masks():
    model, cfg = small_model()
    T = cfg.block_size
    gen = torch.Generator().manual_seed(1)
    idx = torch.randint(1, cfg.vocab_size, (2, T), generator=gen)
    out = {}

    with torch.no_grad():
        base, _ = model(idx)
        changed = idx.clone()
        changed[:, 64] = (changed[:, 64] + 1) % cfg.vocab_size
        alt, _ = model(changed)
    out["causalite_passe_inchangee"] = bool(torch.allclose(base[:, :64], alt[:, :64], atol=1e-5))
    out["causalite_futur_sensible"] = bool(not torch.allclose(base[:, 64:], alt[:, 64:], atol=1e-5))

    doc = torch.cat([torch.zeros(60, dtype=torch.long), torch.ones(T - 60, dtype=torch.long)]).expand(2, T)
    with torch.no_grad():
        ref, _ = model(idx, doc_ids=doc)
        mod = idx.clone()
        mod[:, 10] = (mod[:, 10] + 1) % cfg.vocab_size
        iso, _ = model(mod, doc_ids=doc)
        open_ref, _ = model(idx)
        open_mod, _ = model(mod)
    out["isolation_documents"] = bool(torch.allclose(ref[:, 60:], iso[:, 60:], atol=1e-5))
    out["isolation_document_sensible"] = bool(not torch.allclose(ref[:, 11:60], iso[:, 11:60], atol=1e-5))
    out["controle_sans_masque_fuit"] = bool(not torch.allclose(open_ref[:, 60:], open_mod[:, 60:], atol=1e-5))

    docs = [([1, 2, 3, 4, 5], [0, 0, 1, 1, 1]), ([6, 7, 8, 9], [0, 1, 1, 0])]
    packed = synthetic_packed(docs)
    x, y, d = packed.batch([[0, 1]], "cpu")
    expected = [IGNORE, 3, 4, 5, IGNORE] + [7, 8, IGNORE, IGNORE]
    out["masque_reponses_et_decalage"] = y[0, :9].tolist() == expected
    out["frontiere_document_ignoree"] = int(y[0, 4]) == IGNORE and int(y[0, 8]) == IGNORE
    out["segments_documents"] = d[0, :9].tolist() == [0] * 5 + [1] * 4 and int(d[0, 9]) == -1

    model.train()
    xb = torch.randint(1, cfg.vocab_size, (2, T), generator=gen)
    yb = torch.randint(1, cfg.vocab_size, (2, T), generator=gen)
    yb[:, :40] = IGNORE
    loss, n = train_sft.masked_loss_sum(model, xb, yb, doc)
    loss.backward()
    out["gradients_finis"] = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    out["compte_cibles"] = n == int((yb != IGNORE).sum())
    model.zero_grad()
    empty = torch.full_like(yb, IGNORE)
    loss0, n0 = train_sft.masked_loss_sum(model, xb, empty, doc)
    loss0.backward()
    out["cas_sans_cible"] = n0 == 0 and loss0.item() == 0.0 and all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    return out


def run_steps(model, opt, cfg, start, stop, vocab):
    model.train()
    for step in range(start, stop):
        for g in opt.param_groups:
            g["lr"] = train.get_lr(step, cfg)
        gen = torch.Generator().manual_seed(cfg.seed + step)
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.accum_steps):
            x = torch.randint(0, vocab, (cfg.batch_size, 33), generator=gen)
            _, loss = model(x[:, :-1].contiguous(), x[:, 1:].contiguous())
            (loss / cfg.accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()


def check_resume(steps=8, cut=4, optimizer="muon"):
    cfg = train.TrainConfig(device="cpu", batch_size=2, accum_steps=2, max_steps=steps, lr=3e-3, warmup_steps=2, optimizer=optimizer, seed=11)
    straight, mcfg = small_model(seed=5)
    opt_a = train.build_optimizer(straight, cfg, False)
    run_steps(straight, opt_a, cfg, 0, steps, mcfg.vocab_size)

    first, _ = small_model(seed=5)
    opt_b = train.build_optimizer(first, cfg, False)
    run_steps(first, opt_b, cfg, 0, cut, mcfg.vocab_size)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        train.save_checkpoint(path, first, opt_b, cut, mcfg, cfg)
        ckpt = torch.load(path, map_location="cpu")
    resumed, _ = small_model(seed=99)
    resumed.load_state_dict(ckpt["model"])
    opt_c = train.build_optimizer(resumed, cfg, False)
    opt_c.load_state_dict(ckpt["optimizer"])
    run_steps(resumed, opt_c, cfg, ckpt["step"], steps, mcfg.vocab_size)
    diffs = [(a - b).abs().max().item() for a, b in zip(straight.state_dict().values(), resumed.state_dict().values())]
    return {"optimiseur": optimizer, "ecart_max_parametres": max(diffs), "reprise_identique": max(diffs) == 0.0, "pas_reprise": ckpt["step"]}


class FakeWindows:
    def __init__(self, n):
        self.n = n

    def windows(self, seed):
        return list(range(self.n))


def check_schedule():
    out = {}
    cfg = train.TrainConfig(lr=1e-3, warmup_steps=200, max_steps=1000, min_lr_ratio=0.1)
    lrs = [train.get_lr(s, cfg) for s in range(cfg.max_steps + 1)]
    out["pretrain_debut_warmup"] = abs(lrs[0] - cfg.lr / cfg.warmup_steps) < 1e-12
    out["pretrain_fin_plancher"] = abs(lrs[-1] - cfg.lr * cfg.min_lr_ratio) < 1e-9
    out["pretrain_decroissant_apres_warmup"] = all(a >= b - 1e-15 for a, b in zip(lrs[200:], lrs[201:]))
    longer = train.TrainConfig(lr=1e-3, warmup_steps=200, max_steps=1500, min_lr_ratio=0.1)
    out["pretrain_depend_de_max_steps"] = train.get_lr(500, cfg) != train.get_lr(500, longer)

    scfg = train_sft.SFTConfig(epochs=2, batch_size=2, accum_steps=4, warmup_steps=10, lr=3e-4)
    sched = train_sft.Schedule(FakeWindows(80), scfg)
    per = sched.steps_per_epoch
    out["sft_pas_par_epoque"] = per == [10, 10]
    out["sft_frontiere_epoque"] = sched.locate(9) == (0, 9) and sched.locate(10) == (1, 0)
    try:
        sched.locate(sched.total)
        out["sft_hors_borne_leve"] = False
    except IndexError:
        out["sft_hors_borne_leve"] = True
    slrs = [train_sft.get_lr(s, scfg, sched.lr_total) for s in range(sched.total)]
    out["sft_lr_borne"] = max(slrs) <= scfg.lr + 1e-12 and min(slrs) >= scfg.lr * scfg.min_lr_ratio - 1e-12
    return out


def simulate_sampling(n_windows=20000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for epochs in (1, 2, 3):
        draws = rng.integers(0, n_windows, n_windows * epochs)
        out[f"couverture_avec_remise_{epochs}x"] = round(len(np.unique(draws)) / n_windows, 4)
        out[f"couverture_permutation_{epochs}x"] = 1.0
    weights = np.array([0.7, 0.2, 0.1])
    worst = {}
    for n in (1000, 10000, 100000):
        picks = rng.choice(3, n, p=weights)
        worst[n] = round(float(np.abs(np.bincount(picks, minlength=3) / n - weights).max() * 100), 3)
    out["ecart_quota_max_points"] = worst
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Contrôles A3 du cycle 2 : masques, reprise, schedule, échantillonnage (CPU)")
    p.add_argument("--out", default=str(ROOT / "rd" / "experiences" / "c2_checks_resultats.json"))
    args = p.parse_args()
    lower_priority()
    results = {
        "masques": check_masks(),
        "reprise_muon": check_resume(optimizer="muon"),
        "reprise_adamw": check_resume(optimizer="adamw"),
        "schedule": check_schedule(),
        "echantillonnage": simulate_sampling(),
    }
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    failed = [f"{k}.{kk}" for k, v in results.items() for kk, vv in v.items() if vv is False]
    print(json.dumps(results, ensure_ascii=False))
    print("ECHECS :", failed or "aucun")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
