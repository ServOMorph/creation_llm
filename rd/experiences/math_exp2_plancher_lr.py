"""Pole mathematiques, experience 2 : decomposition perte = progres + plancher de bruit du lr.

Modeles ajustes sur la perte de validation de la v9 (journal complet, cosinus) :
  M0 : L = a + b * s^-alpha                    (progres en pas seulement)
  M1 : L = a + b * s^-alpha + kappa * lr       (progres en pas + plancher lineaire en lr)
  M2 : L = a + b * tau^-alpha + kappa * lr     (progres en temps intrinseque tau = somme des lr)
Test hors echantillon : ajustement sur les 60 % premiers pas, prediction de la fin.
"""
import csv
import json
import math
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def lr_at(step, max_steps, lr=3e-3, warmup=1000, ratio=0.1):
    step = np.asarray(step, dtype=float)
    prog = np.clip((step - warmup) / (max_steps - warmup), 0, 1)
    cos = lr * ratio + 0.5 * (1 + np.cos(np.pi * prog)) * lr * (1 - ratio)
    return np.where(step < warmup, lr * (step + 1) / warmup, cos)


def read_val(path):
    rows = list(csv.DictReader(open(path, newline="")))
    v = [(int(r["step"]), float(r["val_loss"])) for r in rows if r["val_loss"]]
    return np.array([s for s, _ in v], dtype=float), np.array([l for _, l in v])


def design(model, s, max_steps, alpha):
    lr = lr_at(s, max_steps)
    if model == "M2":
        grid = np.arange(1, max_steps + 1)
        tau = np.cumsum(lr_at(grid - 1, max_steps))
        x = np.interp(s, grid, tau) / 3e-3
    else:
        x = s
    cols = [np.ones_like(s), x ** (-alpha)]
    if model != "M0":
        cols.append(lr / 3e-3)
    return np.stack(cols, axis=1)


def fit(model, s, y, max_steps, nonneg=True):
    best = None
    for alpha in np.arange(0.05, 1.5001, 0.01):
        X = design(model, s, max_steps, alpha)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        if nonneg and (coef[1] < 0 or (len(coef) > 2 and coef[2] < 0)):
            continue
        sse = float(((X @ coef - y) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, alpha, coef)
    return best


def predict(model, s, max_steps, alpha, coef):
    return design(model, np.asarray(s, dtype=float), max_steps, alpha) @ coef


if __name__ == "__main__":
    res = {}
    s9, y9 = read_val(ROOT / "checkpoints_v9" / "log.csv")
    keep = s9 >= 2000
    s9, y9 = s9[keep], y9[keep]
    T9 = 91500
    res["v9"] = {}
    for frac in (0.4, 0.6, 0.8, 1.0):
        m = s9 <= frac * T9
        for model in ("M0", "M1", "M2"):
            sse, alpha, coef = fit(model, s9[m], y9[m], T9)
            pred_end = float(predict(model, [T9], T9, alpha, coef)[0])
            out = ~m
            rmse_out = float(np.sqrt(((predict(model, s9[out], T9, alpha, coef) - y9[out]) ** 2).mean())) if out.any() else None
            res["v9"][f"{model}_ajuste_sur_{int(frac * 100)}pc"] = {
                "alpha": round(float(alpha), 2),
                "coef_a_b_kappa": [round(float(c), 4) for c in coef],
                "rmse_ajustement": round(math.sqrt(sse / m.sum()), 4),
                "rmse_hors_echantillon": None if rmse_out is None else round(rmse_out, 4),
                "prediction_finale": round(pred_end, 4),
                "erreur_finale": round(pred_end - 2.5445, 4),
            }

    s3, y3 = read_val(ROOT / "checkpoints_369m" / "log.csv")
    keep = s3 >= 2000
    s3, y3 = s3[keep], y3[keep]
    T3 = 113000
    res["369m"] = {"dernier_point": [float(s3[-1]), float(y3[-1])]}
    for model in ("M0", "M1", "M2"):
        sse, alpha, coef = fit(model, s3, y3, T3)
        res["369m"][f"{model}_libre"] = {
            "alpha": round(float(alpha), 2),
            "coef": [round(float(c), 4) for c in coef],
            "rmse": round(math.sqrt(sse / len(s3)), 4),
            "prediction_finale": round(float(predict(model, [T3], T3, alpha, coef)[0]), 4),
        }
    # transfert : alpha et kappa de la v9 (ajustement complet), a et b reajustes sur le 369M
    for model in ("M1", "M2"):
        _, alpha9, coef9 = fit(model, s9, y9, T9)
        X = design(model, s3, T3, alpha9)
        yy = y3 - coef9[2] * X[:, 2]
        ab, *_ = np.linalg.lstsq(X[:, :2], yy, rcond=None)
        coef = np.array([ab[0], ab[1], coef9[2]])
        rmse = float(np.sqrt(((X @ coef - y3) ** 2).mean()))
        res["369m"][f"{model}_alpha_kappa_v9"] = {
            "alpha": round(float(alpha9), 2),
            "coef": [round(float(c), 4) for c in coef],
            "rmse": round(rmse, 4),
            "prediction_finale_cosinus_0.1": round(float(predict(model, [T3], T3, alpha9, coef)[0]), 4),
            "gain_si_lr_final_nul": round(float(coef9[2] * 0.1), 4),
        }
    # bruit de mesure de la validation : ecarts successifs sur le plateau du 369M
    d = np.diff(y3[-20:])
    res["369m"]["ecart_type_differences_successives_val"] = round(float(d.std()), 4)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    (Path(__file__).parent / "math_exp2_plancher_lr.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
