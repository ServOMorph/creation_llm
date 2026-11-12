from convert_chat import ASSISTANT, USER
from generate import EOT_ID, sample


def build_prompt(history, question, user_id, assistant_id, eot_id, budget):
    current = [user_id, *question[-(budget - 2):], assistant_id] if budget > 2 else [user_id, assistant_id]
    turns = [[user_id, *q, assistant_id, *a, eot_id] for q, a in history]
    kept = []
    used = len(current)
    for turn in reversed(turns):
        if used + len(turn) > budget:
            break
        kept.insert(0, turn)
        used += len(turn)
    return [t for turn in kept for t in turn] + current, len(turns) - len(kept)


class Chat:
    def __init__(self, model, tok, max_new_tokens=300, temperature=0.5, top_k=50, top_p=0.95, repetition_penalty=1.0):
        self.model = model
        self.tok = tok
        self.user_id = tok.token_to_id(USER)
        self.assistant_id = tok.token_to_id(ASSISTANT)
        self.budget = model.cfg.block_size - max_new_tokens
        self.gen_kw = dict(max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
        self.history = []
        self.dropped = 0

    def reset(self):
        self.history = []
        self.dropped = 0

    def reply(self, text, seed=None):
        question = self.tok.encode(text).ids
        ids, self.dropped = build_prompt(self.history, question, self.user_id, self.assistant_id, EOT_ID, self.budget)
        out = sample(self.model, ids, seed=seed, stop_ids=(EOT_ID, self.user_id, self.assistant_id), **self.gen_kw)[0][0]
        answer = out[len(ids):]
        if answer and answer[-1] in (EOT_ID, self.user_id, self.assistant_id):
            answer = answer[:-1]
        self.history.append((question, answer))
        return self.tok.decode(answer).strip()
