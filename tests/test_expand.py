import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from expand_depth import expand_state  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

CFG = GPTConfig(vocab_size=64, block_size=32, n_layer=4, n_head=4, n_embd=64)


def make_model():
    torch.manual_seed(0)
    m = GPT(CFG)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.1)
    return m.eval()


def test_expanded_model_is_identical_at_init():
    m = make_model()
    state, n_layer = expand_state(m.state_dict(), CFG.n_layer, group=2)
    assert n_layer == 6
    big = GPT(replace(CFG, n_layer=n_layer)).eval()
    big.load_state_dict(state)
    x = torch.randint(0, CFG.vocab_size, (2, 20))
    with torch.no_grad():
        assert torch.allclose(m(x)[0], big(x)[0], atol=1e-5)


def test_new_blocks_copy_previous_with_zero_projections():
    m = make_model()
    state, _ = expand_state(m.state_dict(), CFG.n_layer, group=2)
    assert torch.equal(state["blocks.2.attn.qkv.weight"], state["blocks.1.attn.qkv.weight"])
    assert torch.equal(state["blocks.2.mlp.fc.weight"], state["blocks.1.mlp.fc.weight"])
    assert torch.all(state["blocks.2.attn.proj.weight"] == 0)
    assert torch.all(state["blocks.2.mlp.proj.weight"] == 0)
    assert torch.equal(state["blocks.3.attn.proj.weight"], m.state_dict()["blocks.2.attn.proj.weight"])
    assert torch.equal(state["blocks.5.mlp.fc.weight"], state["blocks.4.mlp.fc.weight"])


def test_expanded_block_is_trainable():
    m = make_model()
    state, n_layer = expand_state(m.state_dict(), CFG.n_layer, group=2)
    big = GPT(replace(CFG, n_layer=n_layer))
    big.load_state_dict(state)
    x = torch.randint(0, CFG.vocab_size, (2, 20))
    _, loss = big(x, x)
    loss.backward()
    assert big.blocks[2].attn.proj.weight.grad.abs().sum() > 0
    assert big.blocks[2].mlp.proj.weight.grad.abs().sum() > 0
