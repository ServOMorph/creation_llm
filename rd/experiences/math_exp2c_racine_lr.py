"""Experience 2c : loi L = a + b*x^-alpha + kappa*(lr/lr0)^p, x = pas (S) ou temps intrinseque (TAU).
Balayage fin de p ; test hors echantillon (ajustement 40/60/80 %) ; prediction pre-enregistree pour le 369M."""
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from math_exp2_plancher_lr import ROOT, lr_at, read_val


def xvar(kind, s, T):
    if kind == "S":
        return s
    grid = np.arange(1, T + 1)
    return np.interp(s, grid, np.cumsum(lr_at(grid - 1, T))) / 3e-3


def fit(kind, s, y, T, p, mask, alphas=np.arange(0.05, 2.0, 0.01)):
    best = None
    x = xvar(kind, s, T)
    f = (lr_at(s, T) / 3e-3) ** p
    for alpha in alphas:
        X = np.stack([np.ones_like(s), x ** -alpha, f], 1)
        c, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        if c[1] < 0 or c[2] < 0:
            continue
        sse = ((X[mask] @ c - y[mask]) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, alpha, c, X @ c)
    return best


s, y = read_val(ROOT / "checkpoints_v9" / "log.csv")
k = s >= 2000
s, y = s[k], y[k]
T = 91500
res = {"v9": {}}
for kind in ("S", "TAU"):
    for frac in (0.4, 0.6, 0.8, 1.0):
        m = s <= frac * T
        row = {}
        for p in (0.3, 0.4, 0.5, 0.6, 0.7, 1.0):
            b = fit(kind, s, y, T, p, m)
            if b:
                row[f"p{p}"] = {"alpha": round(float(b[1]), 2), "a": round(float(b[2][0]), 3), "kappa": round(float(b[2][2]), 3),
                                "rmse_in": round(float(np.sqrt(b[0] / m.sum())), 4), "err_finale": round(float(b[3][-1] - y[-1]), 4)}
        res["v9"][f"{kind}_frac{frac}"] = row

s3, y3 = read_val(ROOT / "checkpoints_369m" / "log.csv")
k = s3 >= 2000
s3, y3 = s3[k], y3[k]
T3 = 113000
res["369m"] = {}
for kind in ("S", "TAU"):
    b9 = fit(kind, s, y, T, 0.5, np.ones_like(s, dtype=bool))
    alpha9, kappa9 = b9[1], b9[2][2]
    for label, kap in (("kappa_v9", kappa9), ("kappa_v9_x0.7", 0.7 * kappa9), ("kappa_v9_x1.3", 1.3 * kappa9)):
        x = xvar(kind, s3, T3)
        f = np.sqrt(lr_at(s3, T3) / 3e-3)
        X = np.stack([np.ones_like(s3), x ** -alpha9], 1)
        ab, *_ = np.linalg.lstsq(X, y3 - kap * f, rcond=None)
        rmse = float(np.sqrt(((X @ ab + kap * f - y3) ** 2).mean()))
        xe = xvar(kind, np.array([float(T3)]), T3)[0]
        fin = ab[0] + ab[1] * xe ** -alpha9 + kap * np.sqrt(0.1)
        res["369m"][f"{kind}_{label}"] = {"alpha": round(float(alpha9), 2), "kappa": round(float(kap), 3), "a": round(float(ab[0]), 3),
                                          "rmse": round(rmse, 4), "prediction_finale": round(float(fin), 3),
                                          "part_plancher_finale": round(float(kap * np.sqrt(0.1)), 3)}
print(json.dumps(res))
(Path(__file__).parent / "math_exp2c_racine_lr.json").write_text(json.dumps(res, indent=1))
