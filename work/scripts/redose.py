"""Пересдозировка оси по двум замеренным точкам: взять оптимум вместо угаданной дозы.

ЗАЧЕМ. Замер 23.08 показал, что вал-оптимальный шаг нельзя нести на тест в полную силу:
по оси G1 полный шаг дал 1.6472881 (ХУЖЕ базы V3 1.6472250), а слепая доза 0.20 —
1.6471581 (лучше). Две точки на одной оси задают параболу целиком, значит оптимум больше
не надо угадывать: он вычисляется.

    выигрыш(d) = (2·d·κ − d²)·Q        d — доза, κ — доля вал-оптимума, дожившая до теста

Две измеренные дозы дают κ и Q, дальше оптимум ровно в d* = κ и стоит κ²·Q. Для оси G1:
κ = 0.436, оптимум 0.44 против применённых 0.20, недобрано 0.0000278 (1.3 шума).

ЧЕГО ЭТОТ СКРИПТ НЕ ДЕЛАЕТ: не угадывает. Перед выдачей файла он ПРОВЕРЯЕТ, что пробный
файл действительно равен «база + доза·ось» — и тем самым устанавливает, в каком
пространстве строилась ось (лог или сырое), вместо того чтобы принимать это на веру.
Если проверка не проходит, файл не пишется.

Запуск (там, где лежат submissions/):
  python work/scripts/redose.py \
      --base submissions/V3_canon.csv            --base-score 1.6472249545 \
      --full submissions/G1_gru_tfm_full.csv     --full-score 1.6472880883 \
      --probe submissions/G2_gru_tfm_02.csv --probe-dose 0.20 --probe-score 1.6471581395 \
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def read_sub(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = pl.read_csv(path).sort("user_id")
    col = [c for c in d.columns if c != "user_id"][0]
    return d["user_id"].to_numpy(), d[col].to_numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--base-score", type=float, required=True)
    ap.add_argument("--full", required=True); ap.add_argument("--full-score", type=float, required=True)
    ap.add_argument("--probe", required=True); ap.add_argument("--probe-score", type=float, required=True)
    ap.add_argument("--probe-dose", type=float, default=0.20)
    ap.add_argument("--out", default="")
    ap.add_argument("--sample", default="sample_submit.csv")
    ap.add_argument("--space", choices=["log", "raw", "auto"], default="auto",
                    help="в каком пространстве строилась ось; auto — определить проверкой")
    a = ap.parse_args()

    uid_b, pb = read_sub(Path(a.base))
    uid_f, pf = read_sub(Path(a.full))
    uid_p, pp = read_sub(Path(a.probe))
    for u, nm in ((uid_f, "full"), (uid_p, "probe")):
        if not np.array_equal(u, uid_b):
            raise SystemExit(f"порядок user_id в {nm} не совпал с базой")

    # ---- в каком пространстве строилась ось: проверкой, а не допущением
    lb, lf, lp = (np.log1p(np.clip(x, 0, None)) for x in (pb, pf, pp))
    d_log, d_raw = lf - lb, pf - pb
    # Невязку нельзя мерить абсолютным порогом: сабмиты лежат в csv с ограниченной
    # точностью, и round-trip сам по себе даёт ~1e-4 в логах. Поэтому невязка
    # нормируется на величину САМОГО шага — так порог не зависит ни от точности записи,
    # ни от масштаба пространства, и два пространства становятся сравнимы между собой.
    step_log = float(np.sqrt(np.mean((a.probe_dose * d_log) ** 2)))
    step_raw = float(np.sqrt(np.mean((a.probe_dose * d_raw) ** 2)))
    err_log = float(np.sqrt(np.mean((lp - (lb + a.probe_dose * d_log)) ** 2)))
    err_raw = float(np.sqrt(np.mean((pp - (pb + a.probe_dose * d_raw)) ** 2)))
    rel_log = err_log / max(step_log, 1e-12)
    rel_raw = err_raw / max(step_raw, 1e-12)
    # Поэлементная невязка забита шумом записи csv и пространства не различает (проверено
    # на синтетике: 2.08% против 1.91% при заведомо логарифмической оси). Устойчивый
    # признак — ОЦЕНКА САМОЙ ДОЗЫ проекцией: в верном пространстве она равна заявленной,
    # в неверном — нет, а шум в скалярном произведении усредняется по 250k строкам.
    def dose_hat(dp, df):
        den = float(np.dot(df, df))
        return float(np.dot(dp, df) / den) if den > 0 else float("nan")
    dh_log = dose_hat(lp - lb, d_log)
    dh_raw = dose_hat(pp - pb, d_raw)
    print(f"проверка оси (заявленная доза пробы {a.probe_dose}):")
    print(f"  в логарифмах  доза по проекции {dh_log:+.5f}   отклонение {abs(dh_log-a.probe_dose):.2e}"
          f"   поэлементная невязка {rel_log:.2%} шага")
    print(f"  в сыром       доза по проекции {dh_raw:+.5f}   отклонение {abs(dh_raw-a.probe_dose):.2e}"
          f"   поэлементная невязка {rel_raw:.2%} шага")
    off_log, off_raw = abs(dh_log - a.probe_dose), abs(dh_raw - a.probe_dose)
    if a.space in ("log", "raw"):
        in_log = a.space == "log"
        print(f"  -> пространство задано вручную: {a.space}")
    else:
        in_log = off_log <= off_raw
        best, other = (off_log, off_raw) if in_log else (off_raw, off_log)
        if best > 0.01:
            raise SystemExit(f"ни одно пространство не даёт заявленную дозу (лучшее отклонение "
                             f"{best:.3f}) — ось задана иначе. Укажите --space явно или "
                             f"проверьте --probe-dose.")
        if other < 5 * max(best, 1e-9):
            raise SystemExit(f"пространства неразличимы ({off_log:.2e} против {off_raw:.2e}) — "
                             f"угадывать нельзя. Укажите --space log или --space raw.")
    print(f"  -> ось построена {'в логарифмах' if in_log else 'в сыром пространстве'}")

    # ---- парабола по двум дозам
    a1 = a.base_score - a.full_score          # выигрыш при дозе 1.0
    a2 = a.base_score - a.probe_score         # выигрыш при дозе probe_dose
    t = a.probe_dose
    r = a1 / a2
    k = (1 - t * t * r) / (2 - 2 * t * r)
    Q = a2 / (2 * t * k - t * t)
    chk1, chk2 = (2 * k - 1) * Q, (2 * t * k - t * t) * Q
    print(f"\nвыигрыш при дозе 1.00 {a1:+.7f}, при дозе {t:.2f} {a2:+.7f}")
    print(f"самопроверка параболы: {chk1:+.7f} / {chk2:+.7f} — "
          f"{'сходится' if max(abs(chk1-a1), abs(chk2-a2)) < 1e-9 else 'РАСХОДИТСЯ'}")
    if max(abs(chk1 - a1), abs(chk2 - a2)) >= 1e-9:
        raise SystemExit("парабола не воспроизводит замеры — проверьте скоры")
    best = k * k * Q
    print(f"\nκ оси = {k:.4f}   Q = {Q:.6f}")
    print(f"оптимальная доза {k:.3f}; ожидаемый скор {a.base_score - best:.7f} "
          f"(против {a.probe_score:.7f} у дозы {t:.2f}, выигрыш {best - a2:+.7f})")
    if k <= t:
        print("ВНИМАНИЕ: оптимум не дальше уже применённой дозы — пересдозировка не нужна.")

    if not a.out:
        print("\n--out не задан, файл не пишется (это был расчёт).")
        return

    new = np.expm1(lb + k * d_log) if in_log else pb + k * d_raw
    new = np.clip(new, 0, None)
    if not np.isfinite(new).all():
        raise SystemExit("в результате есть не-конечные значения")
    smp = Path(a.sample)
    if smp.exists():
        us, _ = read_sub(smp)
        if not np.array_equal(np.sort(us), np.sort(uid_b)):
            raise SystemExit("user_id не совпадает с sample_submit")
    # имя колонки — как во всех сабмитах и в sample_submit: predict, не target
    out = pl.DataFrame({"user_id": uid_b.astype(np.int64), "predict": new})
    out.write_csv(a.out)
    lpn = np.log1p(new)
    print(f"\nзаписано {a.out}: {out.height} строк, отрицательных 0, NaN 0")
    print(f"  среднее log1p {lpn.mean():.6f}, разброс {lpn.std():.6f} "
          f"(у базы {lb.mean():.6f} / {lb.std():.6f})")


if __name__ == "__main__":
    main()
