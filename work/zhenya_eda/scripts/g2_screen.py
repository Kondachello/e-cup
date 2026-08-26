"""G2. Скрининг на остатке ДЕЙСТВУЮЩЕГО бленда (val_preds.parquet, 30 моделей).

Вопрос: объясняет ли дневная таксономия остаток бленда ЛУЧШЕ, чем контроль
равной ёмкости из старых величин на тех же окнах.

Проверка переноса между окнами сделана отдельно в f6_transfer.py:
  то же окно +0.000862 / перенос A1->A2 +0.000572 / обратно +0.000927 / плацебо -0.000123.
Положительный результат ЗДЕСЬ без того переноса не значил бы ничего.
"""
import numpy as np, polars as pl, json
from datetime import date
from pathlib import Path
from sklearn.linear_model import Ridge

CACHE = Path("../zhenya/cache")
VAL = date(2026, 1, 14)

v = pl.read_parquet("work/preds_pack/val_preds.parquet").sort("user_id")
X = pl.read_parquet(CACHE / f"a{VAL}.parquet")
assert np.array_equal(v["user_id"].to_numpy(), X["user_id"].to_numpy()), "порядок user_id разошёлся"

ly = np.log1p(np.clip(v["target"].to_numpy().astype(np.float64), 0, None))
blend = v["blend"].to_numpy().astype(np.float64)
resid = ly - blend
sb = float(np.sqrt(np.mean(resid ** 2)))
print(f"действующий бленд: val RMSLE={sb:.6f}  n={len(ly):,}  средний остаток {resid.mean():+.5f}")


def mat(prefs):
    cols = [c for c in X.columns if any(c.startswith(p) for p in prefs)]
    M = X.select(cols).to_numpy().astype(np.float64)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    return np.sign(M) * np.log1p(np.abs(M)), cols


rng = np.random.default_rng(0)
idx = rng.permutation(len(ly)); h = len(ly) // 2
TR, TE = idx[:h], idx[h:]


def r2(M, alpha=10.0):
    mu, sd = M[TR].mean(0), M[TR].std(0) + 1e-9
    m = Ridge(alpha=alpha).fit((M[TR] - mu) / sd, resid[TR])
    p = m.predict((M[TE] - mu) / sd)
    return 1 - np.sum((resid[TE] - p) ** 2) / np.sum((resid[TE] - resid[TR].mean()) ** 2)


GROUPS = {
    "BASE агрегаты (эталон: должен быть ~0)": ["b_"],
    "DT дневная таксономия": ["dt_"],
    "CTL равная ёмкость, старые величины": ["ct_"],
    "TRANS переходы типов дней": ["tr_"],
    "DT + TRANS": ["dt_", "tr_"],
}
print(f"\n{'представление':40s} {'k':>4} {'mdl_flint вне выборки':>15} {'плацебо':>11} {'выигрыш RMSLE':>14}")
out = {}
for name, prefs in GROUPS.items():
    M, cols = mat(prefs)
    if not cols:
        print(f"{name:40s}  — колонок нет"); continue
    a = r2(M)
    b = r2(rng.normal(size=M.shape))
    g = sb - sb * np.sqrt(max(1 - a, 0)) if a > 0 else 0.0
    out[name] = {"k": len(cols), "r2": a, "placebo": b, "gain": g}
    print(f"{name:40s} {len(cols):>4} {a:>15.6f} {b:>11.6f} {g:>14.6f}")

if "DT дневная таксономия" in out and "CTL равная ёмкость, старые величины" in out:
    d = out["DT дневная таксономия"]["r2"] - out["CTL равная ёмкость, старые величины"]["r2"]
    print(f"\nDT минус CTL при равной ёмкости: {d:+.6f} по mdl_flint")

print("\n=== какие именно колонки DT несут сигнал (одиночные mdl_flint) ===")
M, cols = mat(["dt_"])
single = []
for j, c in enumerate(cols):
    Mj = M[:, [j]]
    mu, sd = Mj[TR].mean(0), Mj[TR].std(0) + 1e-9
    m = Ridge(alpha=10.0).fit((Mj[TR] - mu) / sd, resid[TR])
    p = m.predict((Mj[TE] - mu) / sd)
    single.append((1 - np.sum((resid[TE] - p) ** 2) / np.sum((resid[TE] - resid[TR].mean()) ** 2), c))
single.sort(reverse=True)
for r, c in single[:12]:
    print(f"   {c:18s} {r:+.6f}")

Path("../zhenya/out").mkdir(exist_ok=True, parents=True)
Path("../zhenya/out/g2_screen.json").write_text(json.dumps(out, indent=1))
