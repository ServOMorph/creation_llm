import argparse
import json
import random
import re
import sys
import time
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "sft" / "transfo"
SENTENCES = OUT / "sentences.jsonl"
TASKS_FILE = OUT / "tasks.jsonl"
GENERATED = OUT / "generated.jsonl"
REJUDGED = OUT / "rejudged.jsonl"
FINAL = OUT / "transfo.jsonl"
URL = "http://localhost:11434/api"
MODEL = "gemma4:12b"
VAL_DOCS = 10_000
SEED = 1234
SOURCES = {"lecarnet": ["lecarnet"], "vikidia": ["vikidia"], "wiki_fr": ["wiki_fr"]}
SYSTEM = "Tu es un assistant expert de la langue française. Réponds uniquement en JSON."

TEMPLATES = {
    "politesse": ["Reformule cette phrase de manière plus polie", "Rends cette phrase plus polie", "Dis cette phrase plus poliment", "Peux-tu rendre cette demande plus aimable ?"],
    "formel": ["Reformule cette phrase dans un registre soutenu", "Réécris cette phrase dans un style plus formel", "Exprime cette phrase de façon plus soutenue"],
    "passe_compose": ["Mets cette phrase au passé composé", "Conjugue cette phrase au passé composé", "Réécris cette phrase au passé composé"],
    "imparfait": ["Mets cette phrase à l'imparfait", "Conjugue cette phrase à l'imparfait", "Réécris cette phrase à l'imparfait"],
    "futur": ["Mets cette phrase au futur simple", "Conjugue cette phrase au futur", "Réécris cette phrase au futur simple"],
    "present": ["Mets cette phrase au présent", "Conjugue cette phrase au présent de l'indicatif", "Réécris cette phrase au présent"],
    "pluriel": ["Mets cette phrase au pluriel", "Passe le sujet de cette phrase au pluriel", "Réécris cette phrase au pluriel"],
    "question": ["Transforme cette phrase en question", "Pose une question fermée à partir de cette phrase", "Fais de cette affirmation une question"],
    "negation": ["Mets cette phrase à la forme négative", "Transforme cette phrase en phrase négative", "Écris le contraire de cette phrase avec une négation"],
    "simplifier": ["Simplifie cette phrase pour un enfant", "Réécris cette phrase avec des mots simples", "Explique cette phrase plus simplement, en une phrase"],
    "resume": ["Résume ce texte en une phrase", "Fais un résumé de ce passage en une seule phrase", "Résume en une phrase"],
    "paraphrase": ["Reformule cette phrase avec d'autres mots", "Dis la même chose autrement", "Paraphrase cette phrase"],
    "correction": ["Corrige les fautes de cette phrase", "Corrige l'orthographe de cette phrase", "Cette phrase contient des fautes, corrige-la"],
}
GEN_PROMPT = {
    "politesse": "Reformule cette demande de manière plus polie, en une phrase",
    "formel": "Reformule cette phrase dans un registre soutenu, sans changer le sens",
    "passe_compose": "Mets tous les verbes conjugués de cette phrase au passé composé, sans rien changer d'autre",
    "imparfait": "Mets tous les verbes conjugués de cette phrase à l'imparfait, sans rien changer d'autre",
    "futur": "Mets tous les verbes conjugués de cette phrase au futur simple, sans rien changer d'autre",
    "present": "Mets tous les verbes conjugués de cette phrase au présent de l'indicatif, sans rien changer d'autre",
    "pluriel": "Mets le sujet de cette phrase au pluriel et fais tous les accords nécessaires",
    "question": "Transforme cette phrase affirmative en question fermée (réponse oui ou non), en une phrase terminée par « ? »",
    "negation": "Mets cette phrase à la forme négative (ne... pas, ne... plus, ne... jamais)",
    "simplifier": "Réécris cette phrase en une seule phrase courte avec des mots simples, compréhensible par un enfant de 8 ans",
    "resume": "Résume ce texte en une seule phrase",
    "paraphrase": "Reformule cette phrase avec d'autres mots, en gardant exactement le même sens",
}
INTROS = ["Voici la phrase {k} : « {o} »", "{O}", "{O}", "{O}", "Bien sûr : {o}", "{O}"]
INTRO_NAME = {"politesse": "plus polie", "formel": "reformulée", "passe_compose": "au passé composé", "imparfait": "à l'imparfait", "futur": "au futur",
              "present": "au présent", "pluriel": "au pluriel", "question": "transformée en question", "negation": "à la forme négative",
              "simplifier": "simplifiée", "resume": "résumée", "paraphrase": "reformulée", "correction": "corrigée"}

IMPERATIVE = re.compile(r"^(Viens|Va|Donne|Regarde|Fais|Prends|Arrête|Attends|Mange|Range|Ferme|Ouvre|Aide|Laisse|Écoute|Dis|Montre|Apporte|Passe|Rends|Mets|Lâche|Pose|Sors|Tiens|Reviens|Cours|Cherche|Lave|Va-t'en|Rentre|Monte|Descends)\b")
NEGATION = re.compile(r"\b(ne|n'|jamais|rien|personne|aucun|aucune|ni|non)\b", re.I)
PAST = re.compile(r"\b\w+(ait|aient|ais|èrent|irent|urent|âmes|âtes|a été)\b|\b(était|étaient|avait|avaient|fut|furent|eut)\b", re.I)
PERFECT = re.compile(r"\b(a|ai|as|avons|avez|ont|est|sont|suis|es|sommes|êtes|s'est|se sont)\s+\w+(é|ée|és|ées|i|is|it|u|ue|us|ert|ort)\b", re.I)
FUTURE = re.compile(r"\b\w+(rai|ras|ra|rons|rez|ront|rait|raient)\b", re.I)
SING_SUBJECT = re.compile(r"^(?:(?:Le|La|Un|Une|Mon|Ma|Ton|Ta|Son|Sa|Ce|Cet|Cette)\s|L')[a-zàâçéèêëîïôûùüÿœ]")
PRESENT_VERB = re.compile(r"\b(est|a|fait|va|vit|mange|joue|aime|regarde|habite|travaille|porte|donne|trouve|prend|met|dit|voit|peut|veut|doit|sont|ont|font|vont)\b")

SWAPS = [(r"\bà\b", "a"), (r"\ba\b", "à"), (r"\best\b", "et"), (r"\bet\b", "est"), (r"\boù\b", "ou"), (r"\bsont\b", "son"), (r"\bces\b", "ses"),
         (r"\bses\b", "ces"), (r"\bleurs\b", "leur"), (r"(\w{3,})és\b", r"\1é"), (r"(\w{3,})ées\b", r"\1é"), (r"\bles (\w{3,})s\b", r"les \1"),
         (r"(\w{3,})ent\b", r"\1e"), (r"é", "e"), (r"è", "e"), (r"ê", "e"), (r"(\w{4,})er\b", r"\1é"), (r"(\w{4,})é\b", r"\1er"), (r"\bc'est\b", "sais")]


def normalize(text):
    text = re.sub(r"[«»\"“”'’.,;:!?()\[\]*-]", " ", text.lower())
    return " ".join(text.split())


def clean(text):
    text = re.sub(r"\[[^\]]*\]", "", text)
    if text.count("«") != text.count("»"):
        text = text.replace("«", "").replace("»", "")
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text):
    return [clean(s) for s in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý«])", text) if s.strip()]


GLUED = re.compile(r"[a-zàâçéèêëîïôûùüÿœ]\s(?:(?:Il|Elle|Ils|Elles|Le|La|Les|Un|Une|Ce|Cette|En|Dans|Au|Son|Sa|Ses|Depuis|Selon)\s|L')")
INCISE = re.compile(r",\s*(dit|demanda|cria|répondit|s'écria|ajouta|murmura|lança|expliqua|déclara|pensa)\b", re.I)


def good_sentence(s, min_words=5, max_words=30):
    words = s.split()
    if not min_words <= len(words) <= max_words or not s[0].isupper() or s[-1] not in ".!":
        return False
    if re.search(r"[()\[\]{}:;/|=_#@<>\d«»\"]|https?|\.\.|  |\s[,.!]|'\s", s) or s.count(",") > 3 or INCISE.search(s):
        return False
    if len(words) > 1 and words[1][0].isupper() and not words[0].endswith(","):
        return False
    if GLUED.search(s) or re.search(r"\b[A-Z]\.$", s):
        return False
    return not any(a.lower() == b.lower() for a, b in zip(words, words[1:]))


def kinds_for(s, source):
    kinds = ["paraphrase", "question", "formel", "correction"] if s.endswith(".") else []
    if IMPERATIVE.match(s) and s.endswith(("!", ".")):
        return ["politesse", "paraphrase", "correction"]
    if not s.endswith("."):
        return kinds
    if not NEGATION.search(s):
        kinds.append("negation")
    past, perfect, future = PAST.search(s), PERFECT.search(s), FUTURE.search(s)
    if (past or perfect) and not future:
        kinds.append("present")
    if not (past or perfect or future) and PRESENT_VERB.search(s):
        kinds += ["passe_compose", "imparfait", "futur"]
    if SING_SUBJECT.match(s) and "," not in " ".join(s.split()[:4]) and not re.search(r"\b(les|des|ils|elles)\b", s.split(",")[0], re.I):
        kinds.append("pluriel")
    if source != "lecarnet" and len(s.split()) >= 15:
        kinds.append("simplifier")
    return kinds


def row_groups(paths, skip, rng, n):
    groups, start = [], 0
    import pyarrow.parquet as pq

    for path in paths:
        pf = pq.ParquetFile(path)
        for i in range(pf.num_row_groups):
            rows = pf.metadata.row_group(i).num_rows
            if start >= skip:
                groups.append((path, i))
            start += rows
    return rng.sample(groups, min(n, len(groups)))


def extract(n_per_source=4000, groups_per_source=40, seed=SEED, exclude=frozenset()):
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    out, seen = [], set(exclude)
    for source, dirs in SOURCES.items():
        paths = sorted(p for d in dirs for p in (RAW / d).glob("*.parquet"))
        singles, passages, imperatives = [], [], []
        for path, i in row_groups(paths, VAL_DOCS, rng, groups_per_source):
            for text in pq.ParquetFile(path).read_row_group(i, columns=["text"]).column("text").to_pylist():
                sents = split_sentences(text or "")
                quoted = [clean(q) for q in re.findall(r"[«\"“—]\s*([^«»\"“”\n]{8,160}?)\s*[»\"”\n]", text or "")]
                for q in quoted:
                    for s in split_sentences(q):
                        if IMPERATIVE.match(s) and s[-1] in ".!" and good_sentence(s, 3) and normalize(s) not in seen:
                            seen.add(normalize(s))
                            imperatives.append(s)
                for s in sents:
                    if good_sentence(s) and normalize(s) not in seen:
                        seen.add(normalize(s))
                        singles.append(s)
                for j in range(0, max(0, len(sents) - 3), 7):
                    block = sents[j : j + rng.randint(3, 5)]
                    passage = " ".join(block)
                    if all(good_sentence(b, 3, 40) for b in block) and 40 <= len(passage.split()) <= 130:
                        passages.append(passage)
        rng.shuffle(singles)
        rng.shuffle(passages)
        out += [{"source": source, "texte": s, "types": kinds_for(s, source)} for s in singles[:n_per_source]]
        out += [{"source": source, "texte": p, "types": ["resume"]} for p in passages[: n_per_source // 10]]
        out += [{"source": source, "texte": s, "types": ["politesse"]} for s in imperatives[: n_per_source // 10]]
    return out


def corrupt(sentence, rng, n_errors=None):
    n_errors = n_errors or rng.choice((1, 1, 2, 2, 3))
    out = sentence
    for pattern, repl in rng.sample(SWAPS, len(SWAPS)):
        if n_errors == 0:
            break
        matches = list(re.finditer(pattern, out))
        if not matches:
            continue
        m = rng.choice(matches)
        new = out[: m.start()] + m.expand(repl) + out[m.end() :]
        if new != out:
            out = new
            n_errors -= 1
    return out if out != sentence else None


def build_tasks(sentences, per_kind, seed=SEED):
    rng = random.Random(seed)
    pools = {k: [] for k in TEMPLATES}
    for i, s in enumerate(sentences):
        for k in s["types"]:
            pools[k].append(i)
    used, tasks = set(), []
    for kind, pool in pools.items():
        rng.shuffle(pool)
        n = 0
        for i in pool:
            if n >= per_kind.get(kind, 0):
                break
            if i in used:
                continue
            src = sentences[i]["texte"]
            task = {"id": f"{kind}-{i}", "type": kind, "source": sentences[i]["source"], "entree": src}
            if kind == "correction":
                bad = corrupt(src, rng)
                if not bad:
                    continue
                task.update(entree=bad, sortie=src)
            used.add(i)
            tasks.append(task)
            n += 1
    return tasks


def post(path, payload, timeout=600):
    req = urllib.request.Request(f"{URL}/{path}", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ask(prompt, model=MODEL, temperature=0.3, seed=0):
    d = post("chat", {"model": model, "stream": False, "think": False, "format": "json", "keep_alive": "10m",
                      "options": {"temperature": temperature, "num_ctx": 2048, "seed": seed},
                      "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]})
    return d["message"]["content"]


def gen_prompt(task):
    return f'{GEN_PROMPT[task["type"]]} : « {task["entree"]} »\nRéponds au format {{"sortie": "..."}}, avec uniquement le résultat.'


def judge_prompt(task, out):
    return (f'Consigne : {GEN_PROMPT[task["type"]]}.\nTexte d\'origine : « {task["entree"]} »\nRésultat proposé : « {out} »\n'
            "Le résultat applique-t-il correctement et entièrement la consigne, en français correct, sans erreur et sans changer le sens ? "
            'Réponds au format {"correct": true} ou {"correct": false}.')


LOOSE = ("politesse", "formel", "simplifier", "resume", "paraphrase")


def check(task, out):
    kind, src = task["type"], task["entree"]
    flags = []
    if not out or not isinstance(out, str):
        return ["vide"]
    if normalize(out) == normalize(src) or (kind in LOOSE and SequenceMatcher(None, normalize(src), normalize(out)).ratio() > 0.97):
        flags.append("recopie")
    if re.search(r"[«»\"]|\n", out):
        flags.append("guillemets")
    if re.search(r"\b(the|is|and|of|with)\b|[a-z]+ing\b", out):
        flags.append("anglais")
    if kind == "question" and not out.rstrip().endswith("?"):
        flags.append("pas_de_?")
    if kind == "negation" and not NEGATION.search(out):
        flags.append("pas_negation")
    if kind == "futur" and not FUTURE.search(out):
        flags.append("pas_futur")
    if kind == "passe_compose" and not PERFECT.search(out):
        flags.append("pas_passe_compose")
    if kind == "imparfait" and not re.search(r"\w(ait|aient|ais|ions|iez)\b", out):
        flags.append("pas_imparfait")
    if kind == "pluriel" and not re.search(r"\b(les|des|nos|vos|leurs|ses|ces|mes|tes|ils|elles)\b", out, re.I):
        flags.append("pas_pluriel")
    if kind in ("resume", "simplifier") and len(re.findall(r"[.!?](\s|$)", out)) > 1:
        flags.append("plusieurs_phrases")
    if kind in ("resume", "simplifier") and len(out.split()) >= len(src.split()):
        flags.append("pas_plus_court")
    ratio = len(out.split()) / max(1, len(src.split()))
    if kind not in ("resume", "simplifier") and not 0.5 <= ratio <= 2.2:
        flags.append("longueur")
    return flags


def parse(raw, key):
    try:
        return json.loads(raw).get(key)
    except (json.JSONDecodeError, AttributeError):
        return None


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generate(tasks, model=MODEL, limit=None, log_every=20):
    done = {r["id"] for r in read_jsonl(GENERATED)}
    todo = [t for t in tasks if t["type"] != "correction"]
    random.Random(SEED).shuffle(todo)
    todo = [t for t in todo if t["id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    t0 = time.time()
    with GENERATED.open("a", encoding="utf-8") as f:
        for n, task in enumerate(todo, 1):
            out = parse(ask(gen_prompt(task), model), "sortie")
            out = out.strip() if isinstance(out, str) else None
            flags = check(task, out)
            verdict = None
            if not flags:
                verdict = parse(ask(judge_prompt(task, out), model, temperature=0.0), "correct")
            f.write(json.dumps({**task, "sortie": out, "flags": flags, "juge": verdict}, ensure_ascii=False) + "\n")
            f.flush()
            if n % log_every == 0 or n == len(todo):
                print(f"{n}/{len(todo)} {60 * n / (time.time() - t0):.1f}/min", flush=True)


def render(task, rng):
    instr = rng.choice(TEMPLATES[task["type"]])
    src = task["entree"]
    q = instr[:-2] if instr.endswith(" ?") else instr
    forms = [f"{q} : « {src} »", f"{q} : {src}", f"{q} :\n{src}", f"« {src} »\n{instr}" if instr.endswith("?") else f"« {src} »\n{q}.", f'{q} : "{src}"']
    user = rng.choice(forms)
    out = task["sortie"]
    intro = rng.choice(INTROS)
    answer = intro.format(k=INTRO_NAME[task["type"]], o=out, O=out)
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]


MISSING = re.compile(r"\benviron (?:d'|de\b)|\bde (?:et|est|avec|ou)\b|\s,|^[^,]{0,25}, (?:est|sera|a|fut)\b"
                     r"|\b(?:du|en) (?:dans|comme|et|au|à|par|pour|sur)\b|\b(?:Notes et références|Liens externes|Bibliographie|Annexes)\b")
POLITE = re.compile(r"\?|s'il (?:te|vous) pla|\b(?:pourr|voudr|veuill|prie|merci|permet|aimable|gentil|invite|propose)", re.I)
POLITE_Q = re.compile(r"\b(?:pourr|voudr|serai|aurai)\w*-(?:tu|vous|nous)\b", re.I)
DOUBLE_SUBJECT = re.compile(r"^(?:Il|Elle|Ils|Elles|On)\s+\S+-(?:t-)?(?:il|elle|ils|elles|on)\b")
PLURAL_BLOCK = re.compile(r"^(?:Ce n'|C'est|L'ensemble|La plupart|La majorité|Le nombre|La large)", re.I)


def word_counts(texts):
    counts = {}
    for t in texts:
        for w in re.findall(r"[\wÀ-ÿ]+", t):
            if w.islower():
                counts[w] = counts.get(w, 0) + 1
    return counts


def glued(s, vocab, min_count=20):
    for prev, w in zip(s.split(), s.split()[1:]):
        if w[0].isupper() and prev[-1].isalpha() and prev.islower():
            head = re.split(r"['’-]", w)[0].lower()
            if vocab.get(head, 0) >= min_count:
                return True
    return False


def source_ok(task, vocab):
    src = task["sortie"] if task["type"] == "correction" else task["entree"]
    if MISSING.search(src) or (task["type"] != "resume" and glued(src, vocab)):
        return False
    if task["type"] in ("present", "futur", "passe_compose", "imparfait") and src.startswith("Il était une fois"):
        return False
    return not (task["type"] == "pluriel" and (len(src.split()) > 15 or PLURAL_BLOCK.match(src) or "s'appel" in src))


def output_ok(task, out):
    if task["type"] == "politesse":
        return bool(POLITE.search(out)) and not (POLITE_Q.search(out) and "?" not in out)
    if task["type"] == "question":
        return not DOUBLE_SUBJECT.match(out)
    return True


def faults_prompt(task, out):
    return (f"Un élève devait appliquer cette consigne : {GEN_PROMPT[task['type']]}.\nPhrase de départ : « {task['entree']} »\nRéponse de l'élève : « {out} »\n"
            "Tu es un correcteur sévère. Relève toutes les fautes de la réponse : verbe non transformé, accord manquant, mot qui n'existe pas, accent manquant, "
            "sens modifié, partie oubliée ou ajoutée, tournure non naturelle, consigne mal appliquée. "
            'Réponds au format {"fautes": ["..."]}, liste vide si la réponse est parfaite.')


def faults(task, out, model=MODEL):
    d = post("chat", {"model": model, "stream": False, "think": False, "format": "json", "keep_alive": "10m",
                      "options": {"temperature": 0.0, "num_ctx": 4096}, "messages": [{"role": "user", "content": faults_prompt(task, out)}]})
    found = parse(d["message"]["content"], "fautes")
    return found if isinstance(found, list) else None


def candidates(tasks):
    vocab = word_counts(t["entree"] for t in tasks)
    rows = [r for r in read_jsonl(GENERATED) if not r["flags"] and r["juge"] is True and output_ok(r, r["sortie"])]
    return [r for r in rows if source_ok(r, vocab)]


def rejudge(tasks, model=MODEL, log_every=50):
    done = {r["id"] for r in read_jsonl(REJUDGED)}
    todo = [r for r in candidates(tasks) if r["id"] not in done]
    t0 = time.time()
    with REJUDGED.open("a", encoding="utf-8") as f:
        for n, r in enumerate(todo, 1):
            f.write(json.dumps({"id": r["id"], "fautes": faults(r, r["sortie"], model)}, ensure_ascii=False) + "\n")
            f.flush()
            if n % log_every == 0 or n == len(todo):
                print(f"{n}/{len(todo)} {60 * n / (time.time() - t0):.1f}/min", flush=True)


def finalize(tasks, seed=SEED):
    rng = random.Random(seed)
    vocab = word_counts(t["entree"] for t in tasks)
    kept = [t for t in tasks if t["type"] == "correction"]
    kept += [r for r in read_jsonl(GENERATED) if not r["flags"] and r["juge"] is True and output_ok(r, r["sortie"])]
    bench = bench_sources()
    kept = [t for t in kept if source_ok(t, vocab) and not any(f" {b} " in f" {normalize(t['entree'] + ' ' + t['sortie'])} " for b in bench)]
    if REJUDGED.exists():
        clean_ids = {r["id"] for r in read_jsonl(REJUDGED) if r["fautes"] == []}
        kept = [t for t in kept if t["type"] == "correction" or t["id"] in clean_ids]
    rows = [{"id": t["id"], "type": t["type"], "messages": render(t, rng)} for t in kept]
    rng.shuffle(rows)
    return rows


def bench_sources():
    sys.path.insert(0, str(ROOT))
    from bench_assistant import load_bench

    return {normalize(it["src"]) for it in load_bench() if it.get("src")}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Transformations de phrases synthétiques (Ollama)")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--per-source", type=int, default=4000)
    t = sub.add_parser("tasks")
    t.add_argument("--per-kind", type=int, default=350)
    t.add_argument("--correction", type=int, default=600)
    g = sub.add_parser("generate")
    g.add_argument("--limit", type=int)
    g.add_argument("--model", default=MODEL)
    sub.add_parser("finalize")
    rj = sub.add_parser("rejudge")
    rj.add_argument("--model", default=MODEL)
    sub.add_parser("stats")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.cmd == "extract":
        rows = extract(args.per_source, exclude=bench_sources())
        SENTENCES.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        counts = {k: sum(k in r["types"] for r in rows) for k in TEMPLATES}
        print(len(rows), "textes", json.dumps(counts, ensure_ascii=False))
    elif args.cmd == "tasks":
        per_kind = {k: args.per_kind for k in TEMPLATES} | {"correction": args.correction}
        tasks = build_tasks(read_jsonl(SENTENCES), per_kind)
        TASKS_FILE.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in tasks), encoding="utf-8")
        print(len(tasks), "tâches", json.dumps({k: sum(t["type"] == k for t in tasks) for k in TEMPLATES}, ensure_ascii=False))
    elif args.cmd == "generate":
        generate(read_jsonl(TASKS_FILE), args.model, args.limit)
    elif args.cmd == "rejudge":
        rejudge(read_jsonl(TASKS_FILE), args.model)
    elif args.cmd == "finalize":
        rows = finalize(read_jsonl(TASKS_FILE))
        FINAL.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(len(rows), "exemples", json.dumps({k: sum(r["type"] == k for r in rows) for k in TEMPLATES}, ensure_ascii=False))
    elif args.cmd == "stats":
        gen = read_jsonl(GENERATED)
        flags = {}
        for r in gen:
            for fl in r["flags"]:
                flags[fl] = flags.get(fl, 0) + 1
        ok = sum(1 for r in gen if not r["flags"] and r["juge"] is True)
        print(json.dumps({"générés": len(gen), "gardés": ok, "rejet_juge": sum(1 for r in gen if not r["flags"] and r["juge"] is not True), "filtres": flags}, ensure_ascii=False))


if __name__ == "__main__":
    main()
