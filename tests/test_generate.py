import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from generate import EOT_ID, generate, sample, sample_next  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

CFG = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=4, n_embd=64)


def make_model():
    torch.manual_seed(0)
    m = GPT(CFG)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.2)
    return m.eval()


def test_length_and_range():
    out = generate(make_model(), [5, 6, 7], max_new_tokens=10, seed=1)
    assert 4 <= len(out) <= 13
    assert all(0 <= t < CFG.vocab_size for t in out)


def test_seed_determinism():
    m = make_model()
    a = generate(m, [5, 6], max_new_tokens=20, seed=3)
    b = generate(m, [5, 6], max_new_tokens=20, seed=3)
    assert a == b


def test_top_k_one_is_greedy():
    m = make_model()
    a = generate(m, [5, 6], max_new_tokens=15, temperature=1.0, top_k=1, seed=1)
    b = generate(m, [5, 6], max_new_tokens=15, temperature=0, seed=2)
    assert a == b


def test_context_longer_than_block():
    out = generate(make_model(), list(range(1, 31)), max_new_tokens=10, seed=1)
    assert len(out) >= 31


def test_top_p_restricts_support():
    logits = torch.tensor([[10.0, 9.0, 0.0, 0.0]])
    gen = torch.Generator().manual_seed(0)
    draws = {sample_next(logits, 1.0, 0, 0.5, gen).item() for _ in range(200)}
    assert draws == {0}


def test_stops_on_eot():
    m = make_model()
    with torch.no_grad():
        m.lm_head.weight.zero_()
        m.lm_head.weight[EOT_ID] = 5.0
        m.wte.weight.fill_(1.0)
    out = generate(m, [5, 6], max_new_tokens=20, temperature=0)
    assert out[-1] == EOT_ID
    assert len(out) < 22


def test_sample_returns_n_candidates_with_scores():
    m = make_model()
    outs, scores = sample(m, [5, 6], n=4, max_new_tokens=12, temperature=1.0, seed=0)
    assert len(outs) == 4 and scores.shape == (4,)
    assert all(o[:2] == [5, 6] for o in outs)
    assert torch.all(torch.isfinite(scores)) and torch.all(scores <= 0)


def test_cached_generation_matches_uncached_greedy():
    m = make_model()
    ids = [5, 6, 7]
    x = torch.tensor([ids])
    with torch.no_grad():
        for _ in range(10):
            nxt = m(x)[0][:, -1, :].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, nxt], dim=1)
            if nxt.item() == EOT_ID:
                break
    assert generate(m, ids, max_new_tokens=10, temperature=0) == x[0].tolist()
