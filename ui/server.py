"""Tableau de bord local du projet : serveur HTTP stdlib, lecture seule des fichiers du projet."""
import csv
import json
import re
import subprocess
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
UI = Path(__file__).resolve().parent
LOCK_STATE = Path(r"D:\ServOMorph\VertIA_v3\_orchestration\state\lock.json")
PROGRESS = UI / ".progress.json"
TOKENS_PER_STEP = 2 * 32 * 1024
MAX_STEPS_369M = 113000
VIEWABLE = {".md", ".txt", ".json", ".csv", ".log", ".py"}
MAX_VIEW = 400_000
_cache = {}


def cached(key, ttl, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    val = fn()
    _cache[key] = (time.time(), val)
    return val


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_log(path):
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append({
                        "step": int(r["step"]),
                        "train": float(r["train_loss"]) if r.get("train_loss") else None,
                        "val": float(r["val_loss"]) if r.get("val_loss") else None,
                        "tps": float(r["tokens_per_s"]) if r.get("tokens_per_s") else None,
                    })
                except (ValueError, KeyError):
                    continue
    except OSError:
        pass
    return rows


def record_progress(step, now=None, path=None):
    path = Path(path or PROGRESS)
    now = now or time.time()
    hist = read_json(path) or []
    if not hist or hist[-1][1] != step:
        hist.append([now, step])
    hist = [h for h in hist if now - h[0] < 3 * 86400]
    path.write_text(json.dumps(hist), encoding="utf-8")
    return hist


def real_rate(hist, window_h=24):
    if len(hist) < 2:
        return None
    end_t, end_s = hist[-1]
    older = [h for h in hist if end_t - h[0] <= window_h * 3600]
    t0, s0 = older[0]
    if end_t - t0 < 1800:
        return None
    return (end_s - s0) / ((end_t - t0) / 3600)


def training(ckpt_dir=None, max_steps=MAX_STEPS_369M, ref_log=None):
    ckpt_dir = Path(ckpt_dir or ROOT / "checkpoints_369m")
    etat = read_json(ckpt_dir / "etat.json") or {}
    rows = read_log(ckpt_dir / "log.csv")
    ref = read_log(ref_log or ROOT / "checkpoints_v9" / "log.csv")
    last = rows[-1] if rows else {}
    step = last.get("step", etat.get("step", 0))
    log_age = time.time() - (ckpt_dir / "log.csv").stat().st_mtime if (ckpt_dir / "log.csv").exists() else None
    tps = [r["tps"] for r in rows[-60:] if r["tps"]]
    tps_med = sorted(tps)[len(tps) // 2] if tps else None
    hist = record_progress(step) if step else []
    rate = real_rate(hist)
    vals = [(r["step"], r["val"]) for r in rows if r["val"] is not None]
    ref_vals = {r["step"]: r["val"] for r in ref if r["val"] is not None}
    same = [(s, v, ref_vals[s]) for s, v in vals if s in ref_vals]
    return {
        "step": step,
        "max_steps": max_steps,
        "pct": round(100 * step / max_steps, 2) if max_steps else None,
        "etat": etat,
        "actif": log_age is not None and log_age < 600,
        "log_age_s": round(log_age) if log_age is not None else None,
        "tps": tps_med,
        "rate_theorique": round(tps_med * 3600 / TOKENS_PER_STEP) if tps_med else None,
        "rate_reel": round(rate) if rate else None,
        "eta_jours": round((max_steps - step) / rate / 24, 1) if rate else None,
        "val": vals[-1][1] if vals else None,
        "train": last.get("train"),
        "comparaison_v9": same[-1] if same else None,
        "courbe": [[s, v] for s, v in vals],
        "courbe_v9": sorted([s, v] for s, v in ref_vals.items() if s <= max(step, 1) * 1.15),
    }


def gpu():
    def query():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
            used, total, util, temp = [float(x) for x in out.strip().split(",")]
            return {"used_mib": used, "total_mib": total, "util": util, "temp": temp}
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    return cached("gpu", 10, query)


def token():
    lock = read_json(LOCK_STATE)
    if not lock:
        return None
    valid = lock.get("expires", 0) > time.time()
    return {"holder": lock.get("holder") if valid else None, "gb": lock.get("gb"), "expires": lock.get("expires"),
            "note": lock.get("note", "")}


PHASE = re.compile(r"^##\s+(.+?)\s*\[(FAIT|EN COURS|TODO)\]", re.M)


def roadmaps(root=ROOT):
    out = []
    for p in sorted(root.glob("roadmap_*.md")):
        text = p.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text else p.stem
        phases = [{"titre": t, "statut": s} for t, s in PHASE.findall(text)]
        out.append({"fichier": p.name, "titre": title, "phases": phases})
    return out


def section(text, title):
    m = re.search(rf"^## {re.escape(title)}.*?$(.*?)(?=^## |\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else ""


def actions(path=None):
    path = Path(path or ROOT / "_contexte" / "signals.md")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {"maj": None, "actions": [], "contexte": []}
    maj = re.search(r"MAJ (\d{4}-\d{2}-\d{2})", text)
    items, cur = [], None
    for line in section(text, "Actions ouvertes").splitlines():
        m = re.match(r"- \[(P\d)\|(\w+)[^\]]*\]\s*(.*)", line)
        if m:
            cur = {"prio": m.group(1), "etat": m.group(2), "texte": m.group(3), "fait_quand": "", "ref": ""}
            items.append(cur)
        elif cur and "fait quand:" in line:
            cur["fait_quand"] = line.split("fait quand:", 1)[1].strip()
        elif cur and "réf:" in line:
            cur["ref"] = line.split("réf:", 1)[1].strip()
    ctx = [l[2:] for l in section(text, "Contexte chaud").splitlines() if l.startswith("- ")]
    return {"maj": maj.group(1) if maj else None, "actions": items, "contexte": ctx}


def journal(path=None, n=40):
    path = Path(path or ROOT / "_contexte" / "journal_session.md")
    try:
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    except OSError:
        return []
    out = []
    for l in lines[-n:]:
        parts = [p.strip() for p in l.split("|", 2)]
        out.append({"date": parts[0], "type": parts[1] if len(parts) > 1 else "", "texte": parts[2] if len(parts) > 2 else l})
    return out[::-1]


def index_lines(path):
    try:
        return [l[2:] for l in Path(path).read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
    except OSError:
        return []


def recherche(root=ROOT):
    rd = root / "rd"
    sujets = []
    for l in index_lines(rd / "recherches" / "INDEX.md"):
        p = [x.strip() for x in l.split("|")]
        if len(p) >= 5:
            verdict = p[3]
            tag = next((v for v in ("À INTÉGRER", "À TESTER", "À SURVEILLER", "ÉCARTÉ") if verdict.startswith(v)), "")
            sujets.append({"date": p[0], "slug": p[1], "sujet": p[2], "verdict": verdict, "tag": tag, "dossier": p[4]})
    file_ = [l for l in index_lines(rd / "sujets_a_traiter.md") if "|" in l]
    veilles = [{"ligne": l, "fichier": l.split("|")[-1].strip()} for l in index_lines(rd / "veille_actu" / "INDEX.md")]
    docs = sorted(str(p.relative_to(root)).replace("\\", "/") for p in rd.glob("*.md"))
    return {"sujets": sujets, "file": file_, "veilles": veilles, "documents": docs}


def benchmarks(root=ROOT):
    out = []
    for p in sorted((root / "resultats" / "bench_assistant_results").glob("*.json")):
        d = read_json(p) or {}
        r = d.get("resume", {})
        g = r.get("global", {})
        if g:
            out.append({"tag": d.get("tag", p.stem), "pas": d.get("pas"), "taux": g.get("taux"),
                        "categories": {k: v.get("taux") for k, v in r.get("par_categorie", {}).items()},
                        "raisons": r.get("raisons", {})})
    fb = []
    for p in sorted((root / "resultats").glob("eval_frenchbench_*_0shot.json")):
        d = read_json(p) or {}
        name = p.stem.removeprefix("eval_frenchbench_").removesuffix("_0shot")
        g = d.get("french_bench_grammar", {}).get("acc,none")
        v = d.get("french_bench_vocab", {}).get("acc,none")
        fb.append({"modele": name, "grammaire": round(100 * g, 1) if g else None, "vocabulaire": round(100 * v, 1) if v else None})
    return {"assistant": out, "frenchbench": fb}


def git_log(n=12):
    def run():
        try:
            out = subprocess.run(["git", "log", f"-{n}", "--format=%h|%ad|%s", "--date=format:%Y-%m-%d %H:%M"],
                                 cwd=ROOT, capture_output=True, text=True, timeout=5, encoding="utf-8").stdout
            status = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True,
                                    timeout=5, encoding="utf-8").stdout
            return {"commits": [dict(zip(("hash", "date", "message"), l.split("|", 2))) for l in out.splitlines()],
                    "non_commites": len([l for l in status.splitlines() if l.strip()])}
        except (OSError, subprocess.SubprocessError):
            return {"commits": [], "non_commites": None}
    return cached("git", 30, run)


def state():
    return {
        "maintenant": time.strftime("%Y-%m-%d %H:%M:%S"),
        "entrainement": training(),
        "gpu": gpu(),
        "jeton": token(),
        "roadmaps": roadmaps(),
        "signals": actions(),
        "journal": journal(),
        "recherche": recherche(),
        "benchmarks": benchmarks(),
        "git": git_log(),
        "ameliorations": (ROOT / "ameliorations.md").exists(),
    }


def safe_path(rel, root=ROOT):
    p = (root / rel).resolve()
    if root.resolve() not in p.parents or p.suffix not in VIEWABLE or not p.is_file():
        return None
    return p


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            return self.send(200, (UI / "index.html").read_bytes(), "text/html; charset=utf-8")
        if url.path == "/api/state":
            return self.send(200, json.dumps(state(), ensure_ascii=False), "application/json; charset=utf-8")
        if url.path == "/api/file":
            p = safe_path(parse_qs(url.query).get("path", [""])[0])
            if not p:
                return self.send(404, "fichier non autorisé", "text/plain; charset=utf-8")
            text = p.read_text(encoding="utf-8", errors="replace")[-MAX_VIEW:]
            return self.send(200, text, "text/plain; charset=utf-8")
        self.send(404, "introuvable", "text/plain; charset=utf-8")

    def log_message(self, *args):
        pass


def serve(port=8765, open_browser=True):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Tableau de bord : {url}  (Ctrl+C pour arrêter)", flush=True)
    if open_browser:
        threading.Timer(0.8, webbrowser.open, (url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
