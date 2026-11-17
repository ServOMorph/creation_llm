"""Experience 2b : forme du plancher de bruit. L = a + b*s^-alpha + kappa*(lr/lr0)^p, p balaye.
Ajustement sur 60 % des pas de la v9, erreur sur la perte finale (2,5445)."""
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from math_exp2_plancher_lr import ROOT, lr_at, read_val

s, y = read_val(ROOT / "checkpoints_v9" / "log.csv")
k = s >= 2000
s, y = s[k], y[k]
T = 91500
res = {}
for frac in (0.6, 0.8, 1.0):
    m = s <= frac * T
    for p in (0.5, 1.0, 1.5, 2.0, 3.0):
        best = None
        for alpha in np.arange(0.05, 1.5, 0.01):
            X = np.stack([np.ones_like(s), s ** -alpha, (lr_at(s, T) / 3e-3) ** p], 1)
            c, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
            if c[1] < 0 or c[2] < 0:
                continue
            sse = ((X[m] @ c - y[m]) ** 2).sum()
            if best is None or sse < best[0]:
                best = (sse, alpha, c, X)
        if best is None:
            continue
        sse, alpha, c, X = best
        pred = X @ c
        res[f"frac{frac}_p{p}"] = {"alpha": round(float(alpha), 2), "kappa": round(float(c[2]), 3),
                                   "rmse_in": round(float(np.sqrt(sse / m.sum())), 4),
                                   "erreur_finale": round(float(pred[-1] - y[-1]), 4)}
print(json.dumps(res, indent=0))
(Path(__file__).parent / "math_exp2b_forme_plancher.json").write_text(json.dumps(res, indent=1))
