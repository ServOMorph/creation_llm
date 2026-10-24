import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import GPT, GPTConfig  # noqa: E402

TINY = GPTConfig(vocab_size=256, block_size=64, n_layer=2, n_head=4, n_embd=64)


def test_default_param_count_in_range():
    n = GPT(GPTConfig()).num_params()
    assert 20_000_000 <= n <= 50_000_000


def test_output_shapes():
    torch.manual_seed(0)
    cfg = GPTConfig()
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = m(idx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is None


def test_loss_is_scalar():
    torch.manual_seed(0)
    m = GPT(TINY)
    idx = torch.randint(0, TINY.vocab_size, (2, 16))
    _, loss = m(idx, idx)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_logit_cap():
    torch.manual_seed(0)
    m = GPT(TINY)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=1.0)
    idx = torch.randint(0, TINY.vocab_size, (2, 16))
    logits, _ = m(idx)
    assert logits.abs().max() <= TINY.logit_cap + 1e-5


def test_zero_init_residual_projections():
    m = GPT(TINY)
    for block in m.blocks:
        assert torch.all(block.attn.proj.weight == 0)
        assert torch.all(block.mlp.proj.weight == 0)


def test_causal_mask():
    torch.manual_seed(0)
    m = GPT(TINY)
    m.eval()
    a = torch.randint(0, TINY.vocab_size, (1, 20))
    b = a.clone()
    b[0, 10:] = torch.randint(0, TINY.vocab_size, (10,))
    with torch.no_grad():
        la, _ = m(a)
        lb, _ = m(b)
    assert torch.allclose(la[0, :10], lb[0, :10], atol=1e-5)
    assert not torch.allclose(la[0, 10:], lb[0, 10:], atol=1e-5)


def test_overfit_single_batch():
    torch.manual_seed(0)
    m = GPT(TINY)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, TINY.vocab_size, (4, 32))
    y = torch.randint(0, TINY.vocab_size, (4, 32))
    _, first = m(x, y)
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        _, loss = m(x, y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
    assert loss.item() < first.item() * 0.05


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    m = GPT(TINY)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.1)
    m.eval()
    x = torch.randint(0, TINY.vocab_size, (2, 12))
    with torch.no_grad():
        full, _ = m(x)
        caches = [{} for _ in m.blocks]
        steps = [m(x[:, :5], caches=caches)[0]]
        for t in range(5, 12):
            steps.append(m(x[:, t : t + 1], caches=caches)[0])
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5)


def test_kv_cache_rejects_multi_token_append():
    m = GPT(TINY).eval()
    caches = [{} for _ in m.blocks]
    x = torch.randint(0, TINY.vocab_size, (1, 6))
    with torch.no_grad():
        m(x[:, :3], caches=caches)
        with pytest.raises(ValueError):
            m(x[:, 3:], caches=caches)
