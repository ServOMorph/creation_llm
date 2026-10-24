import argparse
from dataclasses import asdict

import torch

from model import GPTConfig

ZEROED = ("attn.proj.weight", "mlp.proj.weight")


def block_params(state, i):
    prefix = f"blocks.{i}."
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def expand_state(state, n_layer, group):
    new = {k: v for k, v in state.items() if not k.startswith("blocks.")}
    j = 0
    for i in range(n_layer):
        params = block_params(state, i)
        for name, v in params.items():
            new[f"blocks.{j}.{name}"] = v
        j += 1
        if (i + 1) % group == 0:
            for name, v in params.items():
                new[f"blocks.{j}.{name}"] = torch.zeros_like(v) if name in ZEROED else v.clone()
            j += 1
    return new, j


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--group", type=int, default=2)
    args = p.parse_args()
    ckpt = torch.load(args.src, map_location="cpu")
    cfg = GPTConfig(**ckpt["model_cfg"])
    state, n_layer = expand_state(ckpt["model"], cfg.n_layer, args.group)
    cfg.n_layer = n_layer
    torch.save({"model": state, "step": 0, "model_cfg": asdict(cfg), "train_cfg": {}}, args.dst)
    print(f"{ckpt['model_cfg']['n_layer']} -> {n_layer} couches : {args.dst}")


if __name__ == "__main__":
    main()
