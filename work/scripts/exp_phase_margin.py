"""Решающий эксперимент Жени: у кого запас на юзерах с волатильной фазой.

Тезис отчёта задачи №2: весь остаток бленда сидит в оценке ТЕКУЩЕЙ ФАЗЫ юзера
(медленная компонента λ=0.79), и секвенсные модели берут её лучше табличного
бустинга, потому что работают с дневным разрешением. Проверяемое следствие:
у секвенсных моделей запас должен быть ВЫШЕ на юзерах с быстро меняющейся фазой
(высокая дисперсия дневной активности внутри окна) и одинаковый с бустингом на
стабильных. Если это так — секвенсный трек стоит продолжать; если нет — тезис
«весь остаток в фазе» рушится.

Дизайн:
- волатильность = std недельных счётчиков активных дней за 12 недель до якоря
  (матрица act_real_cumsum, дни 294..377, вал-окно не задето);
- уровень активности — конфаундер (у пустых юзеров дисперсия нулевая тривиально),
  поэтому волатильные/стабильные берутся ВНУТРИ квинтилей суммарной активности
  (терцили std внутри каждого квинтиля уровня; группы level-matched по построению);
- запас группы = sb_g/sm_g − rho_g на калиброванных прогнозах (калибровка честная,
  на всём вале — как в margin.py); модели секвенсного класса против бустинга;
- решающая статистика: diff-in-diff Δ = (запас_seq_вол − запас_seq_стаб) −
  (запас_boost_вол − запас_boost_стаб), доверительный интервал — бутстрап юзеров.

Запуск: .venv/bin/python work/scripts/exp_phase_margin.py
Артефакт: work/reports/exp_phase_margin.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, REPORTS_DIR, ROOT
from margin import calibrate_honest, score

SEQ = ["seq2tr_f", "gseq_small_s42", "gseq_big_s42"]
BOOST = ["c_ts2_s42", "c_xtw_s42", "twl_v7", "c_dirlgb_s42"]

ANCHOR_DAY = 378          # 2026-01-14 при DAY0=2025-01-01
WIN_WEEKS = 12


def group_margin(lp: np.ndarray, lb: np.ndarray, ly: np.ndarray, g: np.ndarray) -> float:
    e, eb = lp[g] - ly[g], lb[g] - ly[g]
    sm, sb = float(np.sqrt(np.mean(e * e))), float(np.sqrt(np.mean(eb * eb)))
    rho = float(np.mean(e * eb) / (sm * sb))
    return sb / sm - rho


def main():
    pack = pl.read_parquet(ROOT / "work" / "preds_pack" / "val_preds.parquet").sort("user_id")
    uid = pack["user_id"].to_numpy()
    ly = np.log1p(np.clip(pack["target"].to_numpy().astype(np.float64), 0, None))
    lb = pack["blend"].to_numpy().astype(np.float64)

    C = np.load(ROOT / "work" / "data" / "act_real_cumsum.npy", mmap_mode="r")
    assert C.shape[0] == len(uid), "юниверс матрицы активности не совпал с пакетом"
    days = np.diff(C[:, ANCHOR_DAY - WIN_WEEKS * 7: ANCHOR_DAY + 1].astype(np.int32), axis=1)
    weekly = days.reshape(len(uid), WIN_WEEKS, 7).sum(axis=2)          # (U, 12) активных дней в неделю
    vol = weekly.std(axis=1)
    level = weekly.sum(axis=1)

    # терцили волатильности внутри квинтилей уровня: группы совпадают по уровню активности
    q = np.searchsorted(np.quantile(level, [0.2, 0.4, 0.6, 0.8]), level, side="right")
    volat = np.zeros(len(uid), bool)
    stab = np.zeros(len(uid), bool)
    for b in range(5):
        m = q == b
        lo, hi = np.quantile(vol[m], [1 / 3, 2 / 3])
        stab[m & (vol <= lo)] = True
        volat[m & (vol >= hi)] = True
    print(f"группы: волатильные {volat.sum()}, стабильные {stab.sum()}; "
          f"уровень (акт.дней/12нед) вол {level[volat].mean():.1f} против стаб {level[stab].mean():.1f}")

    cand: dict[str, np.ndarray] = {}
    for n in SEQ + BOOST:
        p = PREDS_DIR / f"{n}_val.parquet"
        d = pl.read_parquet(p).sort("user_id")
        assert np.array_equal(d["user_id"].to_numpy(), uid), f"порядок user_id: {p}"
        raw = np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))
        cand[n] = calibrate_honest(raw, ly, 24, 0)

    rows = []
    print(f"\n{'модель':<18}{'класс':>7}{'скор':>10}{'зап.вол':>10}{'зап.стаб':>10}{'дельта':>10}")
    for n in SEQ + BOOST:
        mv = group_margin(cand[n], lb, ly, volat)
        ms = group_margin(cand[n], lb, ly, stab)
        cls = "seq" if n in SEQ else "boost"
        rows.append({"model": n, "cls": cls, "score": round(score(cand[n], ly), 6),
                     "margin_volatile": round(mv, 6), "margin_stable": round(ms, 6),
                     "delta": round(mv - ms, 6)})
        print(f"{n:<18}{cls:>7}{rows[-1]['score']:>10.4f}{mv:>+10.5f}{ms:>+10.5f}{mv - ms:>+10.5f}")

    def did(volat_m, stab_m):
        dseq = np.mean([group_margin(cand[n], lb, ly, volat_m) -
                        group_margin(cand[n], lb, ly, stab_m) for n in SEQ])
        dboost = np.mean([group_margin(cand[n], lb, ly, volat_m) -
                          group_margin(cand[n], lb, ly, stab_m) for n in BOOST])
        return dseq, dboost, dseq - dboost

    dseq, dboost, delta = did(volat, stab)
    print(f"\nсредняя дельта (вол−стаб): секвенсы {dseq:+.5f}, бустинг {dboost:+.5f}, "
          f"diff-in-diff {delta:+.5f}")

    # бутстрап юзеров внутри групп: устойчив ли знак diff-in-diff
    rng = np.random.default_rng(7)
    vi, si = np.where(volat)[0], np.where(stab)[0]
    boots = []
    for _ in range(200):
        bv = np.zeros(len(uid), bool); bv[rng.choice(vi, len(vi))] = True
        bs = np.zeros(len(uid), bool); bs[rng.choice(si, len(si))] = True
        boots.append(did(bv, bs)[2])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"бутстрап diff-in-diff: 95% [{lo:+.5f}, {hi:+.5f}] "
          f"({'ЗНАК УСТОЙЧИВ' if lo * hi > 0 else 'знак не установлен'})")

    out = {
        "design": "терцили std недельной активности (12 нед до якоря) внутри квинтилей уровня",
        "n_volatile": int(volat.sum()), "n_stable": int(stab.sum()),
        "models": rows,
        "delta_seq": round(float(dseq), 6), "delta_boost": round(float(dboost), 6),
        "diff_in_diff": round(float(delta), 6),
        "bootstrap_95": [round(float(lo), 6), round(float(hi), 6)],
    }
    (REPORTS_DIR / "exp_phase_margin.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("\nJSON: work/reports/exp_phase_margin.json")


if __name__ == "__main__":
    main()
