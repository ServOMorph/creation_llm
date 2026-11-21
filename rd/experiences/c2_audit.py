import argparse
import ctypes
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench_assistant import RESULTS, load_bench, score  # noqa: E402

FORM_DEFECTS = {"vide", "boucle", "recopie", "repete_tour", "refus"}
FORMAT_DEFECTS = {"trop_long", "trop_de_phrases"}
LABELS = (
    "motif_trop_strict_probable",
    "omission_ou_reponse_differente",
    "erreur_factuelle_probable",
    "defaut_forme",
    "contrainte_format",
)
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


def lower_priority():
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)


def fold(text):
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def literal_tokens(patterns):
    tokens = []
    for pattern in patterns:
        for alt in pattern.split("|"):
            token = re.sub(r"\\[bBdDwWsS]", " ", alt)
            token = re.sub(r"\\(.)", r"\1", token)
            token = re.sub(r"[^\w ]", " ", fold(token)).strip()
            if token:
                tokens.append(token)
    return tokens


def near_match(token, answer):
    words = re.findall(r"\w+", fold(answer))
    if " " in token:
        return token in " ".join(words)
    if len(token) < 4:
        return token in words
    return any(SequenceMatcher(None, token, w).ratio() >= 0.8 for w in words)


def classify_record(item, record):
    reasons = set(record["raisons"])
    if not reasons:
        return None
    answer = record["reponses"][-1]
    if "critere_interdit" in reasons:
        return "erreur_factuelle_probable"
    if reasons & FORM_DEFECTS:
        return "defaut_forme"
    if reasons & {"critere_absent", "critere_incomplet"}:
        patterns = item.get("ok", []) + item.get("ok_all", [])
        tokens = literal_tokens(patterns)
        if tokens and any(near_match(t, answer) for t in tokens):
            return "motif_trop_strict_probable"
        return "omission_ou_reponse_differente"
    if reasons & FORMAT_DEFECTS:
        return "contrainte_format"
    return "omission_ou_reponse_differente"


def load_tag(tag):
    return json.loads((RESULTS / f"{tag}.json").read_text(encoding="utf-8"))


def classify_failures(tag, items=None):
    by_id = {it["id"]: it for it in (items or load_bench())}
    data = load_tag(tag)
    rows = []
    for r in data["records"]:
        label = classify_record(by_id[r["id"]], r)
        if label:
            rows.append(
                {
                    "id": r["id"],
                    "cat": r["cat"],
                    "seed": r["seed"],
                    "raisons": r["raisons"],
                    "etiquette_auto": label,
                    "etiquette_humaine": "",
                    "attendu": by_id[r["id"]].get("ok", []) + by_id[r["id"]].get("ok_all", []),
                    "question": by_id[r["id"]]["turns"][-1],
                    "reponse": r["reponses"][-1],
                }
            )
    return rows


def check_family_splits(rows):
    seen = defaultdict(set)
    for family, split in rows:
        seen[family].add(split)
    return sorted(f for f, splits in seen.items() if len(splits) > 1)


def item_means(records):
    acc = defaultdict(list)
    for r in records:
        acc[r["id"]].append(1.0 if r["correct"] else 0.0)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def paired_report(tag_a, tag_b, n_boot=2000, seed=0):
    a = item_means(load_tag(tag_a)["records"])
    b = item_means(load_tag(tag_b)["records"])
    ids = sorted(set(a) & set(b))
    diffs = [b[i] - a[i] for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in ids) / len(ids) for _ in range(n_boot))
    return {
        "a": tag_a,
        "b": tag_b,
        "questions": len(ids),
        "taux_a": round(sum(a[i] for i in ids) / len(ids), 4),
        "taux_b": round(sum(b[i] for i in ids) / len(ids), 4),
        "ecart": round(sum(diffs) / len(diffs), 4),
        "ic95": [round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot) - 1], 4)],
        "gagnees": sum(d > 0 for d in diffs),
        "perdues": sum(d < 0 for d in diffs),
    }


def summary(rows):
    return {
        "echecs": len(rows),
        "par_etiquette": dict(Counter(r["etiquette_auto"] for r in rows)),
        "par_categorie": dict(Counter(r["cat"] for r in rows)),
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Audit A1 du cycle 2 : classement des échecs du benchmark assistant (CPU)")
    p.add_argument("--tag", default="v2")
    p.add_argument("--out", default=str(ROOT / "rd" / "experiences" / "c2_audit_v2.json"))
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = p.parse_args()
    lower_priority()
    items = load_bench()
    rescored = [score(next(it for it in items if it["id"] == r["id"]), r["reponses"])["correct"] == r["correct"] for r in load_tag(args.tag)["records"]]
    rows = classify_failures(args.tag, items)
    out = {
        "tag": args.tag,
        "score_historique_inchange": all(rescored),
        "resume": summary(rows),
        "echecs": rows,
    }
    if args.compare:
        out["comparaison"] = paired_report(*args.compare)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k != "echecs"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
