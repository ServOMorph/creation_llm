import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from bench_assistant import GEN, SEEDS, load_bench, score  # noqa: E402
from rag_exp1 import matches, prompt  # noqa: E402

HERE = Path(__file__).parent
PASSAGES = HERE / "rag_exp1b_passages.json"
RESULTS = HERE / "rag_exp1b_resultats.json"
GARDES = ("F01", "F04", "F06", "F08", "F09", "F14", "F16", "F17", "F18", "F21", "F22", "F28", "F29")
CANDIDATS = ("F02", "F03", "F10", "F27")
ECRITS = {
    "F05": "Pour mesurer le temps, on divise la journée en 24 heures. Chaque heure est elle-même divisée en 60 minutes, et chaque minute en 60 secondes. Ce découpage vient des Babyloniens, qui comptaient en base soixante.",
    "F07": "Le lion est un grand félin d'Afrique qui vit en groupe. Le mâle porte une crinière. Grâce à sa force et à son allure, le lion est surnommé le roi de la savane, ou parfois le roi des animaux.",
    "F11": "La photosynthèse permet aux plantes de fabriquer leur nourriture grâce à la lumière du Soleil. Pour cela, les feuilles absorbent un gaz présent dans l'air, le dioxyde de carbone, et puisent de l'eau dans le sol. Elles rejettent ensuite du dioxygène.",
    "F12": "Le Petit Prince est un conte publié en 1943. Il a été écrit et illustré par l'aviateur et écrivain français Antoine de Saint-Exupéry. Il raconte la rencontre entre un pilote tombé en panne dans le désert et un jeune garçon venu d'une autre planète.",
    "F13": "La France a longtemps utilisé le franc. Depuis 2002, la monnaie utilisée en France est l'euro, comme dans la plupart des pays de l'Union européenne. Les billets et les pièces en euros ont remplacé ceux en francs.",
    "F15": "L'eau peut être solide, liquide ou gazeuse. Au niveau de la mer, elle gèle à 0 degré Celsius et elle bout à 100 degrés Celsius. En altitude, la pression est plus faible et l'eau bout à une température plus basse.",
    "F19": "La Ve République est le régime politique de la France depuis 1958. Son premier président a été le général Charles de Gaulle, élu en décembre 1958. Georges Pompidou lui a succédé en 1969.",
    "F20": "Le football est un sport collectif qui se joue avec un ballon rond. Deux équipes s'affrontent sur un terrain rectangulaire. Chaque équipe a onze joueurs sur le terrain, dont un gardien de but, et des remplaçants attendent sur le banc.",
    "F23": "L'émeraude est une pierre précieuse très recherchée en bijouterie. Elle est de couleur verte, à cause du chrome qu'elle contient. Les plus belles émeraudes viennent de Colombie.",
    "F24": "Deux mots sont contraires quand ils ont des sens opposés. Par exemple, le contraire de grand est petit, le contraire de froid est chaud, et le contraire de rapide est lent. On appelle ces mots des antonymes.",
    "F25": "La table de multiplication de 7 se récite ainsi : 7 fois 6 font 42, 7 fois 7 font 49, 7 fois 8 font 56, 7 fois 9 font 63. Il faut la connaître par cœur pour calculer vite.",
    "F26": "Pour additionner 15 et 27, on ajoute d'abord les unités : 5 plus 7 font 12, on pose 2 et on retient 1. Puis les dizaines : 1 plus 2 plus 1 font 4. Le résultat de 15 plus 27 est donc 42.",
    "F30": "La Terre tourne sur elle-même en 24 heures. Pendant la journée, la partie de la Terre tournée vers le Soleil est éclairée par cet astre. La nuit, cette partie n'est plus tournée vers le Soleil et il fait sombre.",
}


def build():
    old = json.loads((HERE / "rag_exp1_passages.json").read_text(encoding="utf-8"))
    cand = json.loads((HERE / "rag_exp1b_candidats.json").read_text(encoding="utf-8"))
    out = {}
    for item in [it for it in load_bench() if it["cat"] == "faits"]:
        key = item["id"]
        if key in GARDES:
            text, src = old[key]["oracle"], "vikidia"
        elif key in CANDIDATS:
            text, src = cand[key][0], "vikidia"
        else:
            text, src = ECRITS[key], "ecrit"
        if not matches(item, text):
            raise SystemExit(f"{key} : le passage ne contient pas la réponse")
        out[key] = {"question": item["turns"][-1], "passage": text, "source": src}
    PASSAGES.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(out), "passages ;", sum(v["source"] == "vikidia" for v in out.values()), "vikidia,", sum(v["source"] == "ecrit" for v in out.values()), "écrits")


def run(ckpt, device, seeds):
    from chat import Chat
    from generate import load_model, load_tokenizer

    data = json.loads(PASSAGES.read_text(encoding="utf-8"))
    ref = {(r["id"], r["seed"]): r["correct"] for r in json.loads((HERE / "rag_exp1_resultats.json").read_text(encoding="utf-8"))["records"] if r["cond"] == "sans"}
    model, step = load_model(ckpt, device)
    chat = Chat(model, load_tokenizer(model.tokenizer_path), **GEN)
    records = []
    for seed in seeds:
        for k, item in enumerate(load_bench()):
            if item["cat"] != "faits":
                continue
            d = data[item["id"]]
            chat.reset()
            answer = chat.reply(prompt([d["passage"]], item["turns"][-1]), seed=seed + 10 * k)
            records.append({"id": item["id"], "seed": seed, "source": d["source"], "reponse": answer, "sans": ref[(item["id"], seed)], **score(item, [answer])})
    by = defaultdict(lambda: [0, 0, 0, 0, 0])
    for r in records:
        for key in (r["source"], "tout"):
            b = by[key]
            b[0] += r["correct"]
            b[1] += r["sans"]
            b[2] += 1
            b[3] += r["correct"] and not r["sans"]
            b[4] += r["sans"] and not r["correct"]
    summary = {k: {"avec_passage": v[0], "sans_passage": v[1], "total": v[2], "gagne": v[3], "perd": v[4]} for k, v in by.items()}
    RESULTS.write_text(json.dumps({"ckpt": str(ckpt), "pas": step, "gen": GEN, "seeds": list(seeds), "resume": summary, "records": records}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build", "run"])
    p.add_argument("--ckpt", default=str(ROOT / "checkpoints_sft_v2" / "ckpt.pt"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = p.parse_args()
    build() if a.cmd == "build" else run(a.ckpt, a.device, a.seeds)


if __name__ == "__main__":
    main()
