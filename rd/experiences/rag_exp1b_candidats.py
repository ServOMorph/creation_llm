import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rag_exp1 import load_passages  # noqa: E402

MOTIFS = {
    "F02": r"araignées?\b[^.]{0,80}\b(huit|8) pattes|(huit|8) pattes[^.]{0,80}araignée",
    "F03": r"loire[^.]{0,80}plus long fleuve (de|français)|plus long fleuve (de france|français)[^.]{0,60}loire",
    "F05": r"heure[^.]{0,60}\b(60|soixante) minutes|(60|soixante) minutes[^.]{0,40}heure",
    "F07": r"lion[^.]{0,120}roi de la savane|roi de la savane[^.]{0,120}lion",
    "F10": r"mercure[^.]{0,80}planète la plus proche du soleil|planète la plus proche du soleil[^.]{0,60}mercure",
    "F11": r"photosynthèse[^.]{0,200}(absorb|capt|utilis)[^.]{0,80}(dioxyde de carbone|gaz carbonique)",
    "F12": r"petit prince[^.]{0,80}(écrit|œuvre|conte|roman|livre)[^.]{0,80}saint[- ]exup|saint[- ]exup[^.]{0,120}(écrit|auteur)[^.]{0,80}petit prince",
    "F13": r"monnaie[^.]{0,60}(de la|en) france[^.]{0,60}euro|euro[^.]{0,80}monnaie[^.]{0,60}(de la france|en france)",
    "F15": r"eau[^.]{0,80}bout[^.]{0,40}100|ébullition[^.]{0,60}eau[^.]{0,60}100\s*°",
    "F19": r"de gaulle[^.]{0,120}premier président de la (ve|cinquième) république|premier président de la (ve|cinquième) république[^.]{0,80}de gaulle",
    "F20": r"football[^.]{0,200}(onze|11) joueurs|(onze|11) joueurs[^.]{0,120}football",
    "F23": r"émeraude[^.]{0,80}(couleur verte|pierre[^.]{0,30}verte|de couleur vert)",
    "F24": r"contraire d[eu] froid|chaud[^.]{0,40}contraire[^.]{0,40}froid|froid[^.]{0,40}contraire[^.]{0,40}chaud",
    "F25": r"7\s*[×x]\s*8\s*=\s*56|8\s*[×x]\s*7\s*=\s*56",
    "F26": r"15\s*\+\s*27\s*=\s*42|27\s*\+\s*15\s*=\s*42",
    "F27": r"féminin[^.]{0,60}acteur[^.]{0,40}actrice|acteur[^.]{0,30}(féminin|ou)[^.]{0,20}actrice|actrice[^.]{0,40}féminin d",
    "F30": r"soleil[^.]{0,80}(éclaire|lumière)[^.]{0,60}terre[^.]{0,60}(jour|journée)|le jour[^.]{0,80}soleil[^.]{0,40}éclaire",
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    docs = load_passages()
    out = {}
    for key, motif in MOTIFS.items():
        rx = re.compile(motif, re.I | re.S)
        hits = [(m.start(), d) for d in docs if (m := rx.search(d))]
        hits.sort(key=lambda h: h[0])
        out[key] = [d for _, d in hits[:3]]
        print(key, len(hits))
        for d in out[key][:2]:
            m = rx.search(d)
            print("   ", d[max(0, m.start() - 120) : m.end() + 120].replace("\n", " "))
    (Path(__file__).parent / "rag_exp1b_candidats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
