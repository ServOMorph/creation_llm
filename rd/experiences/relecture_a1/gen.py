import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import bench_assistant as b

EXP = ROOT / "rd" / "experiences"
d = json.loads((EXP / "c2_audit_v2.json").read_text(encoding="utf-8"))
e = json.loads((EXP / "c2_audit_echantillon20.json").read_text(encoding="utf-8"))
items = {i["id"]: i for i in b.load_bench()}
data = json.loads((ROOT / "resultats" / "bench_assistant_results" / (d["tag"] + ".json")).read_text(encoding="utf-8"))
rec = {(x["id"], x["seed"]): x for x in data["records"]}
vus = {(z["id"], z["seed"]) for z in e}
r = [z for z in d["echecs"] if z["etiquette_auto"] == "omission_ou_reponse_differente" and (z["id"], z["seed"]) not in vus]
limite = {("T03", 1234), ("T06", 1234), ("V03", 1234), ("V04", 1234)}
cases = []
for i, z in enumerate(r):
    it = items[z["id"]]
    rr = rec[(z["id"], z["seed"])]
    assert len(it["turns"]) == len(rr["reponses"])
    assert it["turns"][-1] == z["question"] and rr["reponses"][-1] == z["reponse"]
    lab = ""
    if i < 20 and z["cat"] != "multi_tours":
        lab = "motif_trop_strict" if (z["id"], z["seed"]) in limite else "echec_reel"
    cases.append({
        "id": z["id"], "seed": z["seed"], "cat": z["cat"],
        "dlg": [{"u": u, "m": m} for u, m in zip(it["turns"], rr["reponses"])],
        "ok": it.get("ok", []), "okall": it.get("ok_all", []), "ko": it.get("ko", []),
        "lab": lab, "pre": bool(lab)})
html = (HERE / "tpl.html").read_text(encoding="utf-8").replace("__DATA__", json.dumps(cases, ensure_ascii=False))
(HERE / "relecture_a1.html").write_text(html, encoding="utf-8")
print(len(cases), sum(1 for c in cases if c["pre"]), sum(1 for c in cases if len(c["dlg"]) > 1), len(html))
