import argparse
import ctypes
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rd" / "experiences" / "data_c2"
TOKENIZER = ROOT / "tokenizer_fr32k.json"
MAX_LEN = 1024
ABSTAIN = "Les documents fournis ne permettent pas de répondre à cette question."
VARIANTS = ("gold", "sans_preuve", "contrefactuel")
N_DISTRACTORS = 3
SYL_A = ("ba", "bro", "ca", "dro", "fa", "gri", "ha", "jo", "ka", "lo", "ma", "no", "pa", "ra", "sa", "ta", "vo", "zi")
SYL_B = ("ben", "dor", "lin", "mar", "nor", "ret", "sul", "tan", "vel", "zan")
SYL_C = ("a", "ia", "ie", "o", "on", "ur", "ac")
SYL_D = ("", "la", "ni", "te", "vi", "ro")
COLORS = ("vert", "rouge", "bleu", "jaune", "violet", "orange", "gris", "rose", "marron", "noir")
JOBS = ("boulanger", "forgeron", "médecin", "pilote", "tisserand", "jardinier", "horloger", "potier", "marin", "peintre")
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
TEMPLATES = (
    ("capitale", "Le pays de {E} a pour capitale {V}.", "Quelle est la capitale du pays de {E} ?", "La capitale du pays de {E} est {V}.", "name"),
    ("couleur", "Le fruit {E} est de couleur {V}.", "De quelle couleur est le fruit {E} ?", "Le fruit {E} est de couleur {V}.", "color"),
    ("habitants", "Le village de {E} compte {V} habitants.", "Combien d'habitants compte le village de {E} ?", "Le village de {E} compte {V} habitants.", "number"),
    ("fondateur", "La ville de {E} a été fondée par {V}.", "Qui a fondé la ville de {E} ?", "La ville de {E} a été fondée par {V}.", "name"),
    ("annee", "Le pont de {E} a été construit en {V}.", "En quelle année a été construit le pont de {E} ?", "Le pont de {E} a été construit en {V}.", "year"),
    ("metier", "{E} exerce le métier de {V}.", "Quel métier exerce {E} ?", "{E} exerce le métier de {V}.", "job"),
)


def lower_priority():
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)


def fake_name(rng):
    return rng.choice(SYL_A).capitalize() + rng.choice(SYL_B) + rng.choice(SYL_D) + rng.choice(SYL_C)


def unique_name(rng, taken):
    for _ in range(10000):
        name = fake_name(rng)
        if name not in taken:
            taken.add(name)
            return name
    raise RuntimeError("noms épuisés")


def draw_value(kind, rng, used):
    for _ in range(1000):
        if kind == "name":
            v = fake_name(rng)
        elif kind == "color":
            v = rng.choice(COLORS)
        elif kind == "job":
            v = rng.choice(JOBS)
        elif kind == "number":
            v = str(rng.randrange(120, 9800, 10))
        else:
            v = str(rng.randrange(1650, 2020))
        if v not in used:
            return v
    raise RuntimeError("valeurs épuisées")


def make_families(n, seed, taken=None):
    rng = random.Random(seed)
    taken = taken if taken is not None else set()
    families = []
    for k in range(n):
        tpl = TEMPLATES[k % len(TEMPLATES)]
        key, passage, question, answer, kind = tpl
        entity = unique_name(rng, taken)
        gold = draw_value(kind, rng, set())
        counter = draw_value(kind, rng, {gold})
        others = []
        for _ in range(N_DISTRACTORS + 1):
            other = unique_name(rng, taken)
            others.append({"entite": other, "valeur": draw_value(kind, rng, {gold, counter})})
        families.append(
            {
                "famille": f"s{seed}_{k:04d}",
                "gabarit": key,
                "entite": entity,
                "valeur": gold,
                "valeur_contrefactuelle": counter,
                "distracteurs": others,
                "passage": passage,
                "question": question,
                "reponse": answer,
            }
        )
    return families


def render(template, entity, value):
    return template.format(E=entity, V=value)


def build_example(family, variant, shuffle_seed):
    rng = random.Random(f"{family['famille']}-{variant}-{shuffle_seed}")
    docs = [render(family["passage"], d["entite"], d["valeur"]) for d in family["distracteurs"]]
    if variant == "sans_preuve":
        answer = ABSTAIN
        evidence = None
    else:
        value = family["valeur"] if variant == "gold" else family["valeur_contrefactuelle"]
        evidence = render(family["passage"], family["entite"], value)
        docs = docs[:N_DISTRACTORS] + [evidence]
        answer = render(family["reponse"], family["entite"], value)
    rng.shuffle(docs)
    context = "\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs))
    prompt = f"Documents :\n{context}\n\nQuestion : {render(family['question'], family['entite'], '')}"
    return {
        "famille": family["famille"],
        "variante": variant,
        "preuve": evidence,
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
    }


def split_by_family(families, counts, seed):
    if sum(counts.values()) > len(families):
        raise ValueError("pas assez de familles")
    order = list(families)
    random.Random(seed).shuffle(order)
    out, start = {}, 0
    for name, n in counts.items():
        out[name] = order[start : start + n]
        start += n
    return out


def validate_evidence(example, family, length_fn=None):
    errors = []
    user, answer = example["messages"][0]["content"], example["messages"][1]["content"]
    entity = family["entite"]
    if example["variante"] == "sans_preuve":
        if render(family["passage"], entity, family["valeur"]) in user or render(family["passage"], entity, family["valeur_contrefactuelle"]) in user:
            errors.append("preuve_presente_sans_preuve")
        if answer != ABSTAIN:
            errors.append("reponse_attendue_abstention")
    else:
        value = family["valeur"] if example["variante"] == "gold" else family["valeur_contrefactuelle"]
        other = family["valeur_contrefactuelle"] if example["variante"] == "gold" else family["valeur"]
        if example["preuve"] not in user:
            errors.append("preuve_absente")
        if value not in answer:
            errors.append("reponse_hors_preuve")
        if render(family["passage"], entity, other) in user:
            errors.append("contradiction")
        if user.count(example["preuve"]) != 1:
            errors.append("preuve_dupliquee")
    for d in family["distracteurs"]:
        if d["entite"] == entity:
            errors.append("distracteur_meme_entite")
    if length_fn is not None and length_fn(user) + length_fn(answer) > MAX_LEN:
        errors.append("trop_long")
    return errors


def check_disjoint(splits):
    seen, leaks = {}, []
    for name, fams in splits.items():
        for f in fams:
            names = [f["entite"]] + [d["entite"] for d in f["distracteurs"]]
            for n in names:
                if n in seen and seen[n] != name:
                    leaks.append(n)
                seen[n] = name
    return sorted(set(leaks))


def tokenizer_length():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER))
    return lambda text: len(tok.encode(text).ids)


def build_pilot(families, n_examples, seed=0):
    pairs = [(f, v) for f in families for v in VARIANTS]
    out = []
    for i in range(n_examples):
        f, v = pairs[i % len(pairs)]
        out.append((f, build_example(f, v, i // len(pairs) + seed)))
    return out


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Données A2 du cycle 2 : familles RAFT/H2 synthétiques (CPU)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--train", type=int, default=200)
    p.add_argument("--dev", type=int, default=40)
    p.add_argument("--test", type=int, default=300)
    p.add_argument("--pilot", type=int, default=2000)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    lower_priority()
    out = Path(args.out)
    families = make_families(args.train + args.dev + args.test, args.seed)
    splits = split_by_family(families, {"train": args.train, "dev": args.dev, "test": args.test}, args.seed)
    leaks = check_disjoint(splits)
    length_fn = tokenizer_length() if TOKENIZER.exists() else None
    pilot = build_pilot(splits["train"], args.pilot)
    errors = {}
    for f, ex in pilot:
        for e in validate_evidence(ex, f, length_fn):
            errors[e] = errors.get(e, 0) + 1
    test_examples = [build_example(f, v, 0) for f in splits["test"] for v in VARIANTS]
    dev_examples = [build_example(f, v, 0) for f in splits["dev"] for v in VARIANTS]
    for f in splits["test"]:
        for ex in [build_example(f, v, 0) for v in VARIANTS]:
            for e in validate_evidence(ex, f, length_fn):
                errors[e] = errors.get(e, 0) + 1
    write_jsonl(out / "familles.jsonl", [{"split": n, **f} for n, fams in splits.items() for f in fams])
    write_jsonl(out / "pilote_train.jsonl", [ex for _, ex in pilot])
    write_jsonl(out / "dev.jsonl", dev_examples)
    write_jsonl(out / "test.jsonl", test_examples)
    print(
        json.dumps(
            {
                "familles": {n: len(v) for n, v in splits.items()},
                "exemples": {"pilote_train": len(pilot), "dev": len(dev_examples), "test": len(test_examples)},
                "fuites_entites": leaks,
                "erreurs_validation": errors,
                "longueur_verifiee": length_fn is not None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
