import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"

MASC = """Adam Adrien Alexandre Alexis Antoine Armand Arthur Augustin Axel Baptiste Basile Benjamin Benoît Bastien
Camille Célestin Charles Clément Corentin Damien Daniel David Denis Diego Dimitri Édouard Élie Eliott Émile Enzo
Ethan Étienne Evan Fabien Félix Florian François Gabin Gabriel Gaël Gaspard Gaston Georges Gilles Grégoire Guillaume
Gustave Henri Hector Isaac Ivan Jacques Jean Jérémie Jonas Joseph Jules Julien Justin Kylian Lazare Léon Léandre
Liam Lorenzo Louis Luc Malo Marc Marcel Marius Martin Mathis Matteo Maxence Maxime Milo Nathanaël Noé Nolan Octave
Olivier Oscar Pablo Pascal Philippe Pierre Quentin Raphaël Robin Romain Romuald Samuel Simon Stéphane Sylvain
Théo Théodore Thibault Timothée Valentin Victor Vincent Xavier Yann Yanis Yves Zacharie""".split()

FEM = """Adèle Agathe Alice Aline Amandine Amélie Anaïs Anna Annabelle Apolline Ariane Aurélie Aurore Axelle Bérénice
Blanche Capucine Carla Caroline Cécile Céleste Charlotte Clémence Colette Constance Coralie Daphné Delphine Diane
Éléonore Élise Élodie Éloïse Elsa Émilie Estelle Eva Fanny Faustine Flora Florence Gabrielle Garance Hélène
Hortense Inès Irène Iris Jade Jeanne Joséphine Judith Juliette Justine Laure Laurine Léonie Lina Lise Louise Lucie
Lucille Madeleine Maëlle Manon Margaux Marguerite Marie Marine Marion Mathilde Maya Mélanie Mélissa Mila Mona Nadia
Naomi Nina Noémie Océane Olivia Pauline Pénélope Rose Rosalie Salomé Sarah Solène Suzanne Sylvie Thaïs Valentine
Véronique Victoire Violette Yasmine Yvonne""".split()

NEUTRE = "Alix Andrea Ange Charlie Dominique Eden Lou Noa Sacha Sasha Morgan".split()

EXCLUDED = {"Léo", "Mina", "Julie", "Lucas", "Tom", "Sophie", "Chloé", "Pompon", "Paul", "Bip", "Emma", "Hugo", "Lili", "Nathan", "Zoé"}

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


def main(src=DATA / "train.jsonl", dst=DATA / "train_aug.jsonl", seed=0):
    rng = random.Random(seed)
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for line in f:
            text = augment(json.loads(line)["text"], rng)
            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
