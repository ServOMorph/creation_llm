import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from chat import Chat, build_prompt  # noqa: E402
from convert_chat import ASSISTANT, CHAT_TOKENIZER, CHAT_VOCAB, USER  # noqa: E402
from eval_assistant import base_prompt, trim_base  # noqa: E402
from generate import EOT_ID, load_tokenizer, sample  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from prepare_sft import build_example  # noqa: E402

U, A, E = 900, 901, EOT_ID


def test_prompt_matches_training_format():
    history = [([5, 6], [7, 8, 9]), ([10], [11])]
    ids, dropped = build_prompt(history, [12, 13], U, A, E, budget=100)
    train_ids, _ = build_example([[5, 6], [7, 8, 9], [10], [11]], U, A, E)
    assert ids == train_ids + [U, 12, 13, A]
    assert dropped == 0


def test_oldest_turns_dropped_to_fit_budget():
    history = [([1] * 10, [2] * 10), ([3] * 5, [4] * 5), ([5], [6])]
    ids, dropped = build_prompt(history, [7, 7], U, A, E, budget=24)
    assert len(ids) <= 24
    assert dropped == 1
    assert ids[:2] == [U, 3] and ids[-4:] == [U, 7, 7, A]


def test_long_question_keeps_its_end():
    ids, dropped = build_prompt([([1], [2])], list(range(10, 60)), U, A, E, budget=12)
    assert ids == [U, *range(50, 60), A]
    assert dropped == 1


def test_sample_stops_on_any_stop_id():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=4, n_embd=64)).eval()
    with torch.no_grad():
        m.lm_head.weight.zero_()
        m.lm_head.weight[33] = 5.0
        m.wte.weight.fill_(1.0)
    out = sample(m, [5, 6], max_new_tokens=20, temperature=0, stop_ids=(EOT_ID, 33))[0][0]
    assert out == [5, 6, 33]


def test_chat_reply_strips_tags_and_keeps_history():
    tok = load_tokenizer(CHAT_TOKENIZER)
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=CHAT_VOCAB, block_size=64, n_layer=1, n_head=2, n_embd=32)).eval()
    user_id = tok.token_to_id(USER)
    with torch.no_grad():
        m.lm_head.weight.zero_()
        m.lm_head.weight[user_id] = 5.0
        m.wte.weight.fill_(1.0)
    chat = Chat(m, tok, max_new_tokens=16, temperature=0)
    assert chat.reply("Bonjour") == ""
    assert chat.history == [(tok.encode("Bonjour").ids, [])]
    assert chat.budget == 48
    chat.reset()
    assert chat.history == [] and tok.token_to_id(ASSISTANT) == chat.assistant_id


def test_base_answer_trimmed_at_next_question():
    p = base_prompt("Quelle heure est-il ?")
    assert trim_base(p + " Il est midi.\nQuestion : Et demain ?", p) == "Il est midi."
