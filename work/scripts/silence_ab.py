"""ЗАДАЧА А: две метки нулевого GMV с ПРОТИВОПОЛОЖНЫМИ зазорами.

Правило отбора считает СТРОКИ. Поэтому нулевой GMV в тестовом окне даёт две разные
группы, и отбор двигает их в РАЗНЫЕ стороны:

  norows  «ноль строк»            вал 0.0000%  чистый якорь 2.0178%   ->  вниз
  empty   «строки есть, все нули» вал 1.8503%  чистый якорь 1.1170%   ->  ВВЕРХ

Применённая поправка построена только по первой. Но признаки, предсказывающие
первую, предсказывают и вторую, поэтому часть приложенной силы неизбежно давит вниз
тех, кого надо поднимать. Скрипт обучает ОБЕ вероятности на одном и том же
населении и признаках, выдаёт их на чистом якоре (там метки видны) и на тестовом
(там строится направление), а вся арифметика направлений — в silence_ab_apply.py.

    .venv/bin/python work/scripts/silence_ab.py --threads 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("USE_V2", "1")
os.environ.setdefault("USE_V3", "1")
os.environ.setdefault("USE_V4", "1")
from common import ROOT, feature_cols, load_anchor                    # noqa: E402
from silence_model import (EVAL_ANCHOR, EVAL_TRAIN, TEST_ANCHOR, auc,  # noqa: E402
                           fit_gbm, fit_lr, old_table_apply, old_table_fit,
                           pick_cols, platt, rank_inplace, score, sel_mask)
from silence_split import build_real_cumsum, labels                    # noqa: E402
from silence_target import build_cumsum                                # noqa: E402

OUT = ROOT / "work" / "reports"
NPZ = ROOT / "work" / "data" / "silence_ab.npz"


def load_X(anchors, cols, C, mode):
    Xs, ai = [], []
    for k, a in enumerate(anchors):
        d = load_anchor(a)
        A = np.ascontiguousarray(d.select(cols).to_numpy().astype(np.float32)[sel_mask(C, a)])
        del d
        if mode == "rank":
            rank_inplace(A)
        Xs.append(A)
        ai.append(np.full(A.shape[0], k, dtype=np.int16))
        print(f"  якорь {a}: {A.shape[0]} строк", flush=True)
    return np.concatenate(Xs), np.concatenate(ai)


def load_y(anchors, C, R):
    yn, ye = [], []
    for a in anchors:
        m = sel_mask(C, a)
        a_n, a_e = labels(C, R, a)
        yn.append(a_n[m]); ye.append(a_e[m])
    return np.concatenate(yn).astype(np.int8), np.concatenate(ye).astype(np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=5)
    args = ap.parse_args()

    C = build_cumsum()
    R = build_real_cumsum()
    fit_anchors, cal_anchor = EVAL_TRAIN[:-1], EVAL_TRAIN[-1]
    cols = feature_cols(load_anchor(fit_anchors[0]))

    print(f"обучение {[str(a) for a in fit_anchors]}")
    Xtr, atr = load_X(fit_anchors, cols, C, "raw")
    yn_tr, ye_tr = load_y(fit_anchors, C, R)
    print(f"  доли: ноль строк {yn_tr.mean():.5f}, пустые {ye_tr.mean():.5f}")

    keep, drop, rdrift = pick_cols(Xtr, atr, cols)
    keep = np.array([j for j in keep if rdrift[j] <= 0.30])
    print(f"признаков {len(keep)} из {len(cols)} (выброшено {len(drop)} + дрейф>0.30)")
    for k in range(atr.max() + 1):
        sub = np.ascontiguousarray(Xtr[atr == k]); rank_inplace(sub); Xtr[atr == k] = sub
    A = np.ascontiguousarray(Xtr[:, keep]); del Xtr

    print(f"калибровка {cal_anchor}")
    Xca, _ = load_X([cal_anchor], cols, C, "rank")
    Bc = np.ascontiguousarray(Xca[:, keep]); del Xca
    yn_ca, ye_ca = load_y([cal_anchor], C, R)

    print(f"чистый якорь {EVAL_ANCHOR}")
    Xev, _ = load_X([EVAL_ANCHOR], cols, C, "rank")
    Be = np.ascontiguousarray(Xev[:, keep]); del Xev
    ev_mask = sel_mask(C, EVAL_ANCHOR)
    yn_ev, ye_ev = load_y([EVAL_ANCHOR], C, R)

    print(f"тестовый якорь {TEST_ANCHOR}")
    assert sel_mask(C, TEST_ANCHOR).all(), "тестовое население не совпало с отбором"
    dte = load_anchor(TEST_ANCHOR)
    uid_te = dte["user_id"].to_numpy()
    Bt = np.ascontiguousarray(dte.select(cols).to_numpy().astype(np.float32)[:, keep])
    del dte
    rank_inplace(Bt)

    out = {"uid_te": uid_te, "ev_mask": ev_mask, "yn_ev": yn_ev, "ye_ev": ye_ev}
    res = {}
    for lab, ytr, yca, yev in (("norows", yn_tr, yn_ca, yn_ev), ("empty", ye_tr, ye_ca, ye_ev)):
        pe, pt = {}, {}
        for mn in ("lr", "gbm"):
            mdl = (fit_gbm(A, ytr, atr, atr.max() + 1, threads=args.threads) if mn == "gbm"
                   else fit_lr(A, ytr, atr, atr.max() + 1))
            cal = platt(score(mdl, Bc), yca)
            pe[mn] = 1 / (1 + np.exp(-(cal[0] * score(mdl, Be) + cal[1])))
            pt[mn] = 1 / (1 + np.exp(-(cal[0] * score(mdl, Bt) + cal[1])))
            a_ = auc(yev, pe[mn])
            res[f"{lab}_{mn}_auc"] = a_
            print(f"  {lab}/{mn}: AUC на чистом якоре {a_:.5f}, среднее p на тесте "
                  f"{pt[mn].mean():.5f}", flush=True)
        out[f"p_ev_{lab}"] = 0.5 * pe["lr"] + 0.5 * pe["gbm"]
        out[f"p_te_{lab}"] = 0.5 * pt["lr"] + 0.5 * pt["gbm"]
        res[f"{lab}_mix_auc"] = auc(yev, out[f"p_ev_{lab}"])
        print(f"  {lab}/смесь: AUC {res[f'{lab}_mix_auc']:.5f}")
    del A, Bc, Be, Bt

    # старая таблица — тот же эталон, что в silence_model
    tab = old_table_fit(C, fit_anchors)
    out["p_ev_tab"] = old_table_apply(tab, EVAL_ANCHOR)[ev_mask]
    out["p_te_tab"] = old_table_apply(tab, TEST_ANCHOR)
    res["tab_auc"] = auc(yn_ev, out["p_ev_tab"])
    print(f"  таблица/norows: AUC {res['tab_auc']:.5f}")
    res["cross"] = {"corr_p_te": float(np.corrcoef(out["p_te_norows"], out["p_te_empty"])[0, 1]),
                    "rate_ev_norows": float(yn_ev.mean()), "rate_ev_empty": float(ye_ev.mean())}

    NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NPZ, **out)
    (OUT / "silence_ab_fit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"записан {NPZ}\nзаписан {OUT / 'silence_ab_fit.json'}")


if __name__ == "__main__":
    main()
