"""Instantané statique du tableau de bord (données et documents intégrés), pour l'artefact claude.ai.

python -m ui.snapshot   -> ui/snapshot.html
"""
import json
import sys
from pathlib import Path

from ui.server import ROOT, UI, MAX_VIEW, state

OUT = UI / "snapshot.html"
DOC_GLOBS = ["ameliorations.md", "roadmap_*.md", "rd/*.md", "rd/recherches/*/analyse.md", "rd/recherches/*/rapport.md",
             "rd/veille_actu/20*.md", "bilan_assistant_v2.md"]


def docs(root=ROOT):
    out = {}
    for g in DOC_GLOBS:
        for p in sorted(root.glob(g)):
            out[str(p.relative_to(root)).replace("\\", "/")] = p.read_text(encoding="utf-8", errors="replace")[-MAX_VIEW:]
    return out


def build(out=OUT):
    s = state()
    s["journal"] = s["journal"][:25]
    data = json.dumps(s, ensure_ascii=False).replace("</", "<\\/")
    d = json.dumps(docs(), ensure_ascii=False).replace("</", "<\\/")
    html = (UI / "index.html").read_text(encoding="utf-8")
    html = html.replace("<script>\n", f"<script>window.STATE={data};window.DOCS={d};</script>\n<script>\n", 1)
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    p = build()
    print(p, round(p.stat().st_size / 1024), "Ko", file=sys.stdout)
