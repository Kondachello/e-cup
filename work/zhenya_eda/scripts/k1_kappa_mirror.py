"""K1. Зеркальный оценщик κ: сколько вал-оптимума оси доживает до следующего окна.

Геометрия копирует val->test: два якоря, целевые окна НЕ пересекаются и разнесены
примерно на 30 дней (в кэше шаг 14, поэтому берём разнос 28 — разница в λ^(28/30)
против λ^1 составляет 0.858 против 0.849, пренебрежимо).

Для каждого класса оси:
  1) строим h по рецепту класса НА ОКНЕ A1 (там, где рецепт требует подгонки — подгоняем);
  2) c1 = -<e1,h1>/||h1||^2  — «вал-оптимум»;
  3) c2 = -<e2,h2>/||h2||^2  — что реализовалось на следующем окне;
  4) κ̂ = c2/c1.
Никаких попыток лидерборда не тратится.
"""
import os
import numpy as np, polars as pl, lightgbm as lgb, json
from datetime import date, timedelta
from pathlib import Path
from sklearn.linear_model import Ridge

CACHE = Path(os.environ.get("ZH_CACHE", "work/zhenya_eda/cache"))
OUT = Path(os.environ.get("ZH_OUT", "work/zhenya_eda/out")); OUT.mkdir(parents=True, exist_ok=True)

# якоря берём ИЗ КЭША; пары с разносом 28 дней (ближайшее к геометрии val->test = 30).
# у пар с разносом 28 целевые окна перекрываются на 2 дня из 30 — это завышает κ
# примерно на 2/30 доли шумовой части; отмечено в отчёте, контроль парой с разносом 42.
PAIRS = [(date(2025, 11, 3), date(2025, 12, 1)),
         (date(2025, 10, 20), date(2025, 11, 17)),
         (date(2025, 11, 3), date(2025, 12, 15))]   # разнос 42, окна НЕ пересекаются
SEEDS = (42, 555, 1337)


def load(A):
    return pl.read_parquet(CACHE / f"a{A}.parquet")


def mat(X, prefs=("b_",)):
    cols = [c for c in X.columns if any(c.startswith(p) for p in prefs)]
    M = np.nan_to_num(X.select(cols).to_numpy().astype(np.float64), nan=-1.0,
                      posinf=1e9, neginf=-1e9)
    return M


def fit_base(train_anchors, seed=42):
    Xs, ys = [], []
    for a in train_anchors:
        X = load(a)
        Xs.append(mat(X)); ys.append(np.log1p(X["target"].to_numpy().astype(np.float64)))
    return lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.45,
                             learning_rate=.05, num_leaves=63, min_child_samples=100,
                             subsample=.8, colsample_bytree=.8, n_estimators=700,
                             verbose=-1, n_jobs=4, random_state=seed).fit(np.vstack(Xs),
                                                                          np.concatenate(ys))


def kappa(h1, e1, h2, e2):
    """c = -<e,h>/||h||^2 в обоих окнах; κ = c2/c1"""
    c1 = -float(np.dot(e1, h1) / max(np.dot(h1, h1), 1e-12))
    c2 = -float(np.dot(e2, h2) / max(np.dot(h2, h2), 1e-12))
    return c2 / c1 if abs(c1) > 1e-12 else np.nan, c1, c2


rows = []
for A1, A2 in PAIRS:
    # обучающие срезы: всё, что есть в кэше и чьё целевое окно кончилось до A1
    have = sorted(date.fromisoformat(p.stem[1:]) for p in CACHE.glob("a20*.parquet"))
    TR = [a for a in have if a <= A1 - timedelta(days=30)][-6:]
    if len(TR) < 3:
        print(f"пропуск пары {A1}/{A2}: мало обучающих срезов"); continue
    print(f"\n=== пара {A1} -> {A2} (обучение на {len(TR)} срезах, зазор 30) ===", flush=True)
    models = [fit_base(TR, s) for s in SEEDS]

    D = {}
    for tag, A in (("1", A1), ("2", A2)):
        X = load(A); M = mat(X)
        y = np.log1p(X["target"].to_numpy().astype(np.float64))
        preds = [np.clip(m.predict(M), 0, None) for m in models]
        base = np.mean(preds, axis=0)                    # «бленд» из 3 сидов, равные веса
        D[tag] = dict(X=X, M=M, y=y, preds=preds, base=base, e=base - y)
        print(f"  окно {tag}: n={len(y):,} RMSLE={np.sqrt(np.mean((base-y)**2)):.6f}")

    u1, u2 = D["1"]["X"]["user_id"].to_numpy(), D["2"]["X"]["user_id"].to_numpy()
    common = np.intersect1d(u1, u2)
    i1, i2 = np.searchsorted(u1, common), np.searchsorted(u2, common)
    for t, i in (("1", i1), ("2", i2)):
        for k in ("M", "y", "base", "e"):
            D[t][k] = D[t][k][i]
        D[t]["preds"] = [p[i] for p in D[t]["preds"]]
    e1, e2 = D["1"]["e"], D["2"]["e"]
    n = len(common)
    print(f"  общих юзеров {n:,}")

    # ---------- класс 1: УРОВЕНЬ (глобальный сдвиг) ----------
    h1 = np.ones(n); h2 = np.ones(n)
    k_, c1, c2 = kappa(h1, e1, h2, e2)
    rows.append(("уровень", str(A1), k_, c1, c2))
    print(f"  уровень                 κ̂={k_:+.3f}  (c1={c1:+.4f} c2={c2:+.4f})")

    # ---------- класс 2: СЕГМЕНТНАЯ СТУПЕНЬКА (поправка по децилям прогноза) ----------
    q = np.quantile(D["1"]["base"], np.linspace(0, 1, 11)); q[0], q[-1] = -np.inf, np.inf
    b1 = np.digitize(D["1"]["base"], q[1:-1]); b2 = np.digitize(D["2"]["base"], q[1:-1])
    sh = np.array([-e1[b1 == k].mean() if (b1 == k).sum() > 50 else 0.0 for k in range(10)])
    h1s, h2s = sh[b1], sh[b2]                      # ось = вектор посегментных поправок
    k_, c1, c2 = kappa(h1s, e1, h2s, e2)
    rows.append(("сегментная ступенька", str(A1), k_, c1, c2))
    print(f"  сегментная ступенька    κ̂={k_:+.3f}  (c1={c1:+.4f} c2={c2:+.4f})")

    # ---------- класс 3: СТЕК ПО ПРИЗНАКАМ (ridge остатка на признаки окна 1) ----------
    M1, M2 = np.sign(D["1"]["M"]) * np.log1p(np.abs(D["1"]["M"])), \
             np.sign(D["2"]["M"]) * np.log1p(np.abs(D["2"]["M"]))
    mu, sd = M1.mean(0), M1.std(0) + 1e-9
    rg = Ridge(alpha=10.0).fit((M1 - mu) / sd, -e1)
    h1f, h2f = rg.predict((M1 - mu) / sd), rg.predict((M2 - mu) / sd)
    k_, c1, c2 = kappa(h1f, e1, h2f, e2)
    rows.append(("стек по признакам", str(A1), k_, c1, c2))
    print(f"  стек по признакам       κ̂={k_:+.3f}  (c1={c1:+.4f} c2={c2:+.4f})")

    # ---------- класс 4: ДЕЛЬТА ПЕРЕСБОРКИ БЛЕНДА (веса переоптимизированы на окне 1) ----------
    mdl_amber = np.column_stack(D["1"]["preds"]); mdl_gabbro = np.column_stack(D["2"]["preds"])
    w_new, *_ = np.linalg.lstsq(mdl_amber, D["1"]["y"], rcond=None)   # переподбор весов НА ОКНЕ 1
    h1b, h2b = mdl_amber @ w_new - D["1"]["base"], mdl_gabbro @ w_new - D["2"]["base"]
    k_, c1, c2 = kappa(h1b, e1, h2b, e2)
    rows.append(("дельта пересборки бленда", str(A1), k_, c1, c2))
    print(f"  дельта пересборки бленда κ̂={k_:+.3f}  (c1={c1:+.4f} c2={c2:+.4f})")

    # ---------- класс 5: ЮЗЕРСКОЕ СМЕЩЕНИЕ (эталон β=1: ось из ЧУЖОГО окна) ----------
    # ось строится по остатку ОБУЧАЮЩЕГО среза, не по окну 1 -> подгонки под окно нет
    Xt = load(TR[0]); Mt = mat(Xt)
    yt = np.log1p(Xt["target"].to_numpy().astype(np.float64))
    pt = np.mean([np.clip(m.predict(Mt), 0, None) for m in models], axis=0)
    ut = Xt["user_id"].to_numpy()
    ct = np.intersect1d(ut, common)
    j = np.searchsorted(ut, ct); jj = np.searchsorted(common, ct)
    et = (pt - yt)[j]
    h1r = np.zeros(n); h2r = np.zeros(n)
    h1r[jj] = -et; h2r[jj] = -et                  # одна и та же ось в обоих окнах
    k_, c1, c2 = kappa(h1r, e1, h2r, e2)
    rows.append(("юзерское смещение (эталон)", str(A1), k_, c1, c2))
    print(f"  юзерское смещение       κ̂={k_:+.3f}  (c1={c1:+.4f} c2={c2:+.4f})")

print("\n\n=== СВОДКА κ̂ по классам (среднее по парам) ===")
import collections
agg = collections.defaultdict(list)
for nm, a, k_, c1, c2 in rows:
    if np.isfinite(k_):
        agg[nm].append(k_)
REPORTED = {"уровень": 0.20, "сегментная ступенька": 0.05, "стек по признакам": 0.15,
            "дельта пересборки бленда": 0.565, "юзерское смещение (эталон)": None}
print(f"{'класс':28s} {'κ̂ зеркало':>12} {'κ замерено LB':>15} {'|разница|':>11}")
for nm, v in agg.items():
    r = REPORTED.get(nm)
    d = f"{abs(np.mean(v)-r):.3f}" if r is not None else "—"
    rs = f"{r:.3f}" if r is not None else "—"
    print(f"{nm:28s} {np.mean(v):>12.3f} {rs:>15} {d:>11}")
(OUT / "k1_kappa_mirror.json").write_text(json.dumps(
    [{"class": nm, "anchor": a, "kappa": k_, "c1": c1, "c2": c2} for nm, a, k_, c1, c2 in rows], indent=1))
print(f"\nзаписано {OUT}/k1_kappa_mirror.json")
