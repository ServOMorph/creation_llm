import json
import re
import sys
import time
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

URL = "http://localhost:11434/api"

TASKS = [
    ("politesse", "Reformule cette phrase de manière plus polie", "Ferme la fenêtre."),
    ("politesse", "Reformule cette phrase de manière plus polie", "Passe-moi ton stylo."),
    ("formel", "Reformule cette phrase dans un registre formel", "T'inquiète, je m'en occupe demain."),
    ("formel", "Reformule cette phrase dans un registre formel", "Le boss a dit que la réunion sautait."),
    ("passe_compose", "Mets cette phrase au passé composé", "Nous partons en vacances et nous visitons Rome."),
    ("passe_compose", "Mets cette phrase au passé composé", "Elle descend l'escalier et elle sort."),
    ("futur", "Mets cette phrase au futur simple", "Je vais au marché et j'achète des pommes."),
    ("pluriel", "Mets cette phrase au pluriel", "Le cheval blanc mange une pomme rouge."),
    ("pluriel", "Mets cette phrase au pluriel", "Mon voisin a un beau jardin."),
    ("question", "Transforme cette phrase en question", "Les élèves ont fini leurs devoirs."),
    ("question", "Transforme cette phrase en question", "Pierre travaille à la bibliothèque le mardi."),
    ("negation", "Mets cette phrase à la forme négative", "Il a encore faim."),
    ("simplifier", "Simplifie cette phrase pour un enfant de 7 ans", "Les précipitations abondantes ont provoqué une crue de la rivière."),
    ("simplifier", "Simplifie cette phrase pour un enfant de 7 ans", "Le médecin a prescrit un traitement antibiotique pendant une semaine."),
    ("resume", "Résume en une phrase", "Julie avait prévu un pique-nique au parc avec ses amis. Mais le matin, il s'est mis à pleuvoir très fort. Ils ont finalement mangé chez elle en regardant un film."),
    ("resume", "Résume en une phrase", "La mairie a décidé de fermer la piscine municipale pendant deux mois. Des travaux de rénovation sont nécessaires car le bassin fuit. Elle rouvrira en septembre."),
    ("paraphrase", "Dis la même chose avec d'autres mots", "Il pleut des cordes depuis ce matin."),
    ("paraphrase", "Dis la même chose avec d'autres mots", "Ce film m'a beaucoup ému."),
    ("correction", "Corrige les fautes de cette phrase", "Les fille on manger des gâteaux hier soir."),
    ("correction", "Corrige les fautes de cette phrase", "Je suis aller au cinéma avec mes ami."),
]

SYSTEM = "Tu es un assistant expert de la langue française. Réponds uniquement en JSON."

GEN_TYPES = ["mettre une phrase à l'imparfait", "rendre une phrase plus polie", "transformer une phrase affirmative en question"]


def post(path, payload, timeout=600):
    req = urllib.request.Request(f"{URL}/{path}", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(model, prompt, temperature=0.3):
    t = time.time()
    d = post("chat", {
        "model": model, "stream": False, "think": False, "format": "json",
        "options": {"temperature": temperature, "num_ctx": 4096, "seed": 42},
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    })
    toks = d.get("eval_count", 0)
    dur = d.get("eval_duration", 1) / 1e9
    return d["message"]["content"], toks / dur if dur else 0, time.time() - t


def check(kind, src, out):
    flags = []
    if not out:
        return ["vide"]
    sim = SequenceMatcher(None, src.lower(), out.lower()).ratio()
    if sim > 0.97:
        flags.append("recopie")
    if kind == "question" and not out.rstrip().endswith("?"):
        flags.append("pas_de_?")
    if kind == "negation" and not re.search(r"\bn['e ]|\bne\b|\bplus\b", out):
        flags.append("pas_negation")
    if kind == "resume" and len(re.findall(r"[.!?](\s|$)", out)) > 1:
        flags.append("plusieurs_phrases")
    if re.search(r"[a-zA-Z]+ing\b| the | is ", out):
        flags.append("anglais?")
    return flags


def main():
    models = sys.argv[1:]
    out_dir = Path(__file__).parent / "bench_results"
    out_dir.mkdir(exist_ok=True)
    for model in models:
        rows, speeds, invalid = [], [], 0
        try:
            post("generate", {"model": model, "prompt": "Bonjour", "stream": False, "options": {"num_ctx": 4096}, "think": False})
        except Exception as e:
            print(model, "ERREUR chargement", e)
            continue
        ps = post("ps", {}) if False else json.load(urllib.request.urlopen(f"{URL}/ps"))
        m = next((x for x in ps.get("models", []) if x["name"].startswith(model.split(":")[0])), {})
        vram = m.get("size_vram", 0) / 2**30
        total = m.get("size", 0) / 2**30
        for kind, instr, src in TASKS:
            prompt = f'{instr} : « {src} »\nRéponds au format {{"sortie": "..."}}.'
            raw, sp, _ = chat(model, prompt)
            speeds.append(sp)
            try:
                o = json.loads(raw).get("sortie", "").strip()
            except Exception:
                invalid += 1
                o = raw.strip()
            rows.append({"type": kind, "entree": src, "sortie": o, "flags": check(kind, src, o)})
        gens = []
        for g in GEN_TYPES:
            prompt = (f"Crée 5 exemples d'entraînement variés pour la tâche : {g}. Phrases du quotidien, sujets et vocabulaire différents. "
                      'Format : {"exemples": [{"consigne": "...", "entree": "...", "sortie": "..."}]}')
            raw, sp, el = chat(model, prompt, temperature=0.8)
            speeds.append(sp)
            try:
                ex = json.loads(raw).get("exemples", [])
            except Exception:
                invalid += 1
                ex = raw
            gens.append({"tache": g, "exemples": ex, "secondes": round(el, 1)})
        n_flag = sum(1 for r in rows if r["flags"])
        res = {"model": model, "vram_gio": round(vram, 2), "taille_gio": round(total, 2), "tok_s": round(sum(speeds) / len(speeds), 1),
               "json_invalides": invalid, "sorties_signalees": n_flag, "rows": rows, "gen": gens}
        (out_dir / f"{model.replace(':', '_').replace('/', '_')}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{model}: vram {vram:.2f}/{total:.2f} Gio, {res['tok_s']} tok/s, json invalides {invalid}, signalées {n_flag}/{len(rows)}", flush=True)
        post("generate", {"model": model, "keep_alive": 0})


if __name__ == "__main__":
    main()
