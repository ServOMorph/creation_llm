import argparse
import json
import random
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"

MASC = """Adam Adrien Alexandre Alexis Antoine Armand Arthur Augustin Axel Baptiste Basile Benjamin Benoît Bastien
Célestin Charles Clément Corentin Damien Daniel David Denis Diego Dimitri Édouard Élie Eliott Émile Enzo
Ethan Étienne Evan Fabien Félix Florian François Gabin Gabriel Gaël Gaspard Gaston Georges Gilles Grégoire Guillaume
Gustave Henri Hector Isaac Ivan Jacques Jean Jérémie Jonas Joseph Jules Julien Justin Kylian Lazare Léon Léandre
Liam Lorenzo Louis Luc Malo Marc Marcel Marius Martin Mathis Matteo Maxence Maxime Milo Nathanaël Noé Nolan Octave
Olivier Oscar Pablo Pascal Philippe Pierre Quentin Raphaël Robin Romain Romuald Samuel Simon Stéphane Sylvain
Théo Théodore Thibault Timothée Valentin Victor Vincent Xavier Yann Yanis Yves Zacharie
Léo Lucas Tom Paul Hugo Nathan Louis Jules Gabin Mathéo Timéo Lenny Nino Rayan Ibrahim Mohamed Youssef
Thomas Nicolas Sébastien Julien Mathieu Hugues Aymeric Marin Elias Tiago Kenzo Naël Ilyes Amir""".split()

FEM = """Adèle Agathe Alice Aline Amandine Amélie Anaïs Anna Annabelle Apolline Ariane Aurélie Aurore Axelle Bérénice
Blanche Capucine Carla Caroline Cécile Céleste Charlotte Clémence Colette Constance Coralie Daphné Delphine Diane
Éléonore Élise Élodie Éloïse Elsa Émilie Estelle Eva Fanny Faustine Flora Florence Gabrielle Garance Hélène
Hortense Inès Irène Iris Jade Jeanne Joséphine Judith Juliette Justine Laure Laurine Léonie Lina Lise Louise Lucie
Lucille Madeleine Maëlle Manon Margaux Marguerite Marie Marine Marion Mathilde Maya Mélanie Mélissa Mila Mona Nadia
Naomi Nina Noémie Océane Olivia Pauline Pénélope Rose Rosalie Salomé Sarah Solène Suzanne Sylvie Thaïs Valentine
Véronique Victoire Violette Yasmine Yvonne
Julie Emma Chloé Zoé Sophie Lili Mina Léa Jade Lola Ambre Romane Nora Rosa Lisa Clara Margot Agnès
Sofia Lya Aya Inaya Assia Léna Anna Ella Livia Héloïse Clémentine Bérangère Ophélie Perrine""".split()

NEUTRE = "Camille Alix Andrea Ange Charlie Dominique Eden Lou Noa Sacha Sasha Morgan".split()

EXCLUDED = set()

PATTERN = re.compile(r"\b(nommée?|appelée?|s'appelait|s’appelait) ([A-ZÀ-Ý][a-zà-ÿ]+)")


def pick(rng, intro):
    if intro.endswith("ée"):
        pool = FEM
    elif intro.endswith("é"):
        pool = MASC
    else:
        pool = NEUTRE
    return rng.choice([n for n in pool if n not in EXCLUDED])


def augment(text, rng, p=0.8):
    mapping = {}
    for intro, name in PATTERN.findall(text):
        if name in mapping:
            continue
        if rng.random() < p:
            mapping[name] = pick(rng, intro)
    for old, new in mapping.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text


INTRO = re.compile(r"\b(un|une) ((?:[a-zà-ÿ-]+ ){1,3})(nommée?|appelée?) ([A-ZÀ-Ý][a-zà-ÿ]+)")
INTRO_WEIGHTS = (0.4, 0.2, 0.2, 0.2)
VOWELS = "aeiouyhàâéèêëîïôûü"


def intro_variant(match, kind):
    art, words, _, name = match.groups()
    words = words.strip()
    if kind == 1:
        return f"{name}, {art} {words}"
    if kind == 2:
        return f"{art} {words} qui s'appelait {name}"
    if kind == 3:
        definite = "l'" if words[0] in VOWELS else ("le " if art == "un" else "la ")
        return f"{definite}{words} {name}"
    return match.group(0)


def vary_intro(text, rng, weights=INTRO_WEIGHTS):
    def repl(m):
        kind = rng.choices(range(4), weights=weights)[0]
        if kind == 2 and text[m.end() : m.end() + 4] == " qui":
            kind = 0
        return intro_variant(m, kind)

    return INTRO.sub(repl, text)


def main(src=DATA / "train.jsonl", dst=DATA / "train_aug.jsonl", seed=1, vary=False):
    rng = random.Random(seed)
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for line in f:
            text = augment(json.loads(line)["text"], rng)
            if vary:
                text = vary_intro(text, rng)
            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def build(seeds, out, vary=False):
    from tokenizers import Tokenizer

    from prepare_tokens import encode_file

    tok = Tokenizer.from_file(str(ROOT / "tokenizer.json"))
    total = 0
    with open(out, "wb") as dst:
        for seed in seeds:
            jsonl = DATA / f"train_aug_s{seed}.jsonl"
            part = jsonl.with_suffix(".bin")
            main(dst=jsonl, seed=seed, vary=vary)
            total += encode_file(tok, jsonl, part)
            with open(part, "rb") as src:
                shutil.copyfileobj(src, dst)
            part.unlink()
            jsonl.unlink()
    return total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--out", default=str(DATA / "train_aug.bin"))
    p.add_argument("--vary-intro", action="store_true")
    args = p.parse_args()
    print("tokens", build(args.seeds, Path(args.out), vary=args.vary_intro))
