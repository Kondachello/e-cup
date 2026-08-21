"""mdl_flint. Готовые направления зондов -Y4 (семейство «прошлогоднее праздничное окно»).

Каждое направление:
  1) строится из окна 2025 года, выровненного на ТЕСТОВОЕ окно 2026;
  2) резидуализуется по 365-дневной активности — чтобы не дублировать то,
     что модели и так знают;
  4) взаимно ортогонализуется (Y2/Y3/Y4 на непересекающихся подокнах);
  5) нормируется на q = mean(h²) = 0.0026 — правило 4x-шага из части D.

Применение: lp_проба = lp_база + step;  предсказание = expm1(lp_проба).
Восстановление: κ = c/q, c = (F0² − S² + q)/2, где S — публичный скор пробы.
"""
import os
import numpy as np, polars as pl
from datetime import date, timedelta
from sklearn.linear_model import Ridge

Q_TARGET = 0.0026
OUT = os.environ.get("ZH_OUT", "work/zhenya_eda/out")
df = pl.read_parquet("train.parquet", columns=["user_id", "event_date", "gmv", "to_ord"])
uid = np.sort(df["user_id"].unique().to_numpy())
t = pl.read_parquet("work/preds_pack/test_preds.parquet").sort("user_id")
assert np.array_equal(t["user_id"].to_numpy(), uid), "порядок user_id разошёлся"
lp_base = t["blend"].to_numpy().astype(np.float64)      # уже log1p
A = date(2026, 1, 14)


def wsum(s, e, col="gmv"):
    w = df.filter(pl.col("event_date").is_between(s, e)).group_by("user_id").agg(
        pl.col(col).sum().alias("y"))
    y = pl.DataFrame({"user_id": uid}).join(w, on="user_id", how="left")["y"].to_numpy().astype(float)
    return np.nan_to_num(y)


# контроль: 365-дневная активность на якоре (то, что модели видят)
R = np.column_stack([np.log1p(wsum(A - timedelta(days=364), A)),
                     np.log1p(wsum(A - timedelta(days=364), A, "to_ord")),
                     np.log1p(wsum(A - timedelta(days=29), A)),
                     np.log1p(wsum(A - timedelta(days=89), A))])
mu, sd = R.mean(0), R.std(0) + 1e-9
Rn = (R - mu) / sd

WINDOWS = {
    "Y1_ya_full":   (date(2025, 2, 14), date(2025, 3, 15)),   # весь прошлогодний аналог теста
    "Y2_pre8mar":   (date(2025, 3, 1),  date(2025, 3, 7)),    # пик перед 8 марта (03.03 x1.149)
    "Y3_feb23":     (date(2025, 2, 20), date(2025, 2, 24)),   # окрестность 23 февраля
    "Y4_post8mar":  (date(2025, 3, 8),  date(2025, 3, 15)),   # провал 08.03 и хвост
}

basis = [np.ones(len(uid)), lp_base - lp_base.mean()]        # уровень и масштаб уже пробиты


def orth(h, vecs):
    for v in vecs:
        vv = float(np.dot(v, v))
        if vv > 1e-12:
            h = h - v * float(np.dot(h, v)) / vv
    return h


print(f"{'направление':14s} {'окно 2025':>24} {'sd сырого':>10} {'после ортог.':>13} "
      f"{'корр с базой':>13} {'уйдёт в клип':>13}")
made, done = {}, list(basis)
for nm, (s, e) in WINDOWS.items():
    raw = np.log1p(wsum(s, e))
    resid = raw - Ridge(alpha=10.0).fit(Rn, raw).predict(Rn)   # снимаем известную активность
    h = orth(resid - resid.mean(), done)
    scale = np.sqrt(Q_TARGET / max(float(np.mean(h * h)), 1e-18))
    h = h * scale
    clip = int((lp_base + h < 0).sum())
    made[nm] = h
    done.append(h)
    print(f"{nm:14s} {str(s) + '..' + str(e):>24} {raw.std():>10.4f} {h.std():>13.5f} "
          f"{np.corrcoef(h, lp_base)[0, 1]:>13.4f} {clip:>13,}")

print(f"\nвзаимные корреляции направлений (должны быть ~0):")
ks = list(made)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        print(f"  {ks[i]:12s} x {ks[j]:12s} {np.corrcoef(made[ks[i]], made[ks[j]])[0,1]:+.2e}")

for nm, h in made.items():
    q = float(np.mean(h * h))
    pl.DataFrame({"user_id": uid, "step": h}).write_parquet(f"{OUT}/dir_{nm}.parquet")
    print(f"\n{nm}: q = {q:.6f}  (цель {Q_TARGET})")
    print(f"  σ_κ = {1.666395/np.sqrt(50000*q):.3f}   доза = {0.0416/(0.0416+1.666395**2/(50000*q)):.3f}")
    print(f"  цена пробного сабмита: +{q/(2*1.666395):.6f} к скору при κ=0")
    print(f"  ожидание при κ=0.45: выигрыш {0.45**2*q/(2*1.666395):.6f} "
          f"= {0.45**2*q/(2*1.666395)/0.000022:.1f} шума")
    print(f"  файл {OUT}/dir_{nm}.parquet")
