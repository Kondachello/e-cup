"""Направление НОВОЙ оси из НЕСКОЛЬКИХ моделей сразу — для make_axis_probe.py --dir.

Обобщение work/scripts/seq/add_direction.py на N моделей. Логика та же и по тем же
причинам:

  на валидации подбираем МНК-веса ДВАЖДЫ — без новых моделей и с ними; разница этих
  двух блендов НА ТЕСТЕ и есть вклад новых моделей, очищенный от всего остального.

Смешивать напрямую нельзя: база несёт глобальную и сегментные поправки, замеренные
зондами лидерборда, и смесь размыла бы их пропорционально весу. Добавка их не трогает.

Зачем N моделей, а не одна. Бестабличная ось живёт в ДВУХ моделях сразу
(`tfm4_tabless` и `kevf_tl_gru`), их корреляция ошибок 0.9842 — они разъехались и с
блендом, и между собой. По joint_gain набор даёт +0.000583 против +0.000427 у лучшей
поодиночке: ось не «занята», на ней есть место под второго. Гнать их двумя отдельными
осями значит потратить две посылки вместо одной.

Пишет направление в parquet (колонка `d`, центрированный шаг в log1p) — формат
`work/reports/gls_draft/directions/*.parquet`, который читает make_axis_probe.py.

Печатает то, что нужно ДО траты посылки: честный перенос по фолдам, q = mean(d²),
σ_κ и остаточную новизну q_ост после ортогонализации против уже применённых осей.

Запуск:
  .venv/bin/python work/scripts/make_tabless_dir.py \
      --model tfm4_tabless --model kevf_tl_gru \
      --base submissions/F12_ebint.csv --out work/data/dir_N2_tabless.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
N_PUB = 50_000
NOISE = 2.2e-5

L1 = lambda x: np.log1p(np.clip(np.asarray(x, np.float64), 0, None))
rm = lambda l, y: float(np.sqrt(np.mean((l - y) ** 2)))


def fit_shifts(lp, yy, idx=None, bins=24):
    """Бин-калибровка: сдвиги в log1p по квантилям прогноза (как в add_direction.py)."""
    i = slice(None) if idx is None else idx
    qs = np.quantile(lp[i], np.linspace(0, 1, bins + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    c, s = [], []
    for k in range(bins):
        m = (lp[i] > qs[k]) & (lp[i] <= qs[k + 1])
        if m.sum() < 500:
            continue
        c.append(lp[i][m].mean())
        s.append(yy[i][m].mean() - lp[i][m].mean())
    return np.array(c), np.array(s)


ap = lambda lp, c, s: np.clip(lp + np.interp(lp, c, s), 0, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True,
                   help="имя в work/preds (ждёт NAME_val.parquet и NAME_test.parquet); можно повторять")
    p.add_argument("--pack", default=str(ROOT / "work" / "preds_pack"))
    p.add_argument("--base", required=True, help="замеренная база, например submissions/F12_ebint.csv")
    p.add_argument("--out", required=True, help="кудаписать направление (parquet, колонка d)")
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()

    outp = Path(a.out)
    if outp.exists():
        raise SystemExit(f"СТОП: {outp} уже есть — направления не перезаписываются")
    outp.parent.mkdir(parents=True, exist_ok=True)

    v = pl.read_parquet(f"{a.pack}/val_preds.parquet").sort("user_id")
    t = pl.read_parquet(f"{a.pack}/test_preds.parquet").sort("user_id")
    y = L1(v["target"].to_numpy())

    # Колонки пакета УЖЕ в log1p (README пакета). Двойной log стоил бы 160 шумов.
    probe = v[[c for c in v.columns if c not in ("user_id", "target")][0]].to_numpy()
    space = "log" if float(np.nanmax(probe)) < 50 else "raw"
    conv = (lambda x: np.asarray(x, np.float64)) if space == "log" else L1
    print(f"пространство колонок пакета: {space}")

    his = [c for c in v.columns if c not in ("user_id", "target") and c in t.columns]
    Xv = np.column_stack([conv(v[c].to_numpy()) for c in his] + [np.ones(len(y))])
    Xt = np.column_stack([conv(t[c].to_numpy()) for c in his] + [np.ones(t.height)])
    print(f"колонок пакета в бленде: {len(his)}")

    def load(name, side):
        d = pl.read_parquet(ROOT / "work" / "preds" / f"{name}_{side}.parquet").sort("user_id")
        col = "pred" if "pred" in d.columns else "predict"
        return d["user_id"].to_numpy(), L1(d[col].to_numpy())

    lv, lt = [], []
    for m in a.model:
        uid_v, x = load(m, "val")
        uid_t, xt = load(m, "test")
        assert np.array_equal(uid_v, v["user_id"].to_numpy()), f"{m}: другой user_id на вале"
        assert np.array_equal(uid_t, t["user_id"].to_numpy()), f"{m}: другой user_id на тесте"
        lv.append(x)
        lt.append(xt)
    print(f"моделей в оси: {len(a.model)} — {', '.join(a.model)}")

    ins = lambda X, cols: np.column_stack([X[:, :-1]] + cols + [X[:, -1]])

    # --- честный перенос: веса И калибровка учатся только на train-фолдах
    gains = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        f = rng.permutation(len(y)) % a.folds
        b_all, w_all = np.zeros(len(y)), np.zeros(len(y))
        for k in range(a.folds):
            tr, te = f != k, f == k
            cols = [ap(x, *fit_shifts(x, y, tr)) for x in lv]
            X2 = ins(Xv, cols)
            w0 = np.linalg.lstsq(Xv[tr], y[tr], rcond=None)[0]
            w1 = np.linalg.lstsq(X2[tr], y[tr], rcond=None)[0]
            b_all[te], w_all[te] = Xv[te] @ w0, X2[te] @ w1
        gains.append(rm(w_all, y) - rm(b_all, y))
    g = np.array(gains)
    print(f"\nчестный перенос (5 разбиений x {a.folds} фолдов):")
    print(f"  прирост {g.mean():+.6f}  разброс {g.std():.6f}  = {abs(g.mean())/NOISE:.1f} шумовых единиц")
    if g.mean() >= 0:
        print("  ВНИМАНИЕ: на валидации ось НЕ помогает — дальше идти не стоит")

    # --- рабочие веса на всей валидации
    shifts = [fit_shifts(x, y) for x in lv]
    Xv2 = ins(Xv, [ap(x, c, s) for x, (c, s) in zip(lv, shifts)])
    Xt2 = ins(Xt, [ap(x, c, s) for x, (c, s) in zip(lt, shifts)])
    w0 = np.linalg.lstsq(Xv, y, rcond=None)[0]
    w1 = np.linalg.lstsq(Xv2, y, rcond=None)[0]
    print(f"\nвалидация без оси {rm(Xv @ w0, y):.6f}")
    print(f"валидация с осью  {rm(Xv2 @ w1, y):.6f}")
    for m, w in zip(a.model, w1[-len(a.model) - 1:-1]):
        print(f"  вес {m:16s} {w:+.4f}")

    d = Xt2 @ w1 - Xt @ w0
    m_raw = float(d.mean())
    d = d - m_raw                      # среднесохраняющий шаг: уровень базы не трогаем
    q = float(np.mean(d * d))

    base = pl.read_csv(a.base).sort("user_id")
    assert np.array_equal(base["user_id"].to_numpy(), t["user_id"].to_numpy()), "база: другой user_id"
    sigma_k = 1.6456762 / np.sqrt(N_PUB * q)

    print(f"\nнаправление: среднее снято {m_raw:+.6f}; q = mean(d²) = {q:.6e}")
    print(f"  |d|: медиана {np.median(np.abs(d)):.5f}  p99 {np.quantile(np.abs(d), .99):.5f}  max {np.abs(d).max():.4f}")
    print(f"  σ_κ при дозе 1.0 = F0/√(50000·q) = {sigma_k:.4f}")

    # --- новизна: сколько остаётся после ортогонализации против применённых осей
    dd = ROOT / "work" / "reports" / "gls_draft" / "directions"
    axes = sorted(dd.glob("*.parquet"))
    if axes:
        A = []
        for f_ in axes:
            z = pl.read_parquet(f_).sort("user_id")
            col = "d" if "d" in z.columns else "step"
            A.append(z[col].to_numpy().astype(np.float64))
        A = np.column_stack(A)
        coef, *_ = np.linalg.lstsq(A, d, rcond=None)
        resid = d - A @ coef
        q_res = float(np.mean(resid * resid))
        print(f"\nновизна против {A.shape[1]} применённых осей:")
        print(f"  q_ост = {q_res:.6e}   novelty = q_ост/q = {q_res/q:.3f}")
        print(f"  гейт скрининга: q_ост ≥ 0.015 И novelty ≥ 0.5 -> "
              f"{'ПРОЙДЕН' if (q_res >= 0.015 and q_res/q >= 0.5) else 'НЕ ПРОЙДЕН'}")

    pl.DataFrame({"user_id": base["user_id"], "d": d}).write_parquet(outp)
    print(f"\nсохранено {outp}")


if __name__ == "__main__":
    main()
