"""Тиры v2 (.extra) и v3 тестового якоря с MAX_BACK=409. Пишет в .mb409.* — ничего не трогает.

Оба билдера несут ту же константу MAX_BACK=379, и модели блока их используют
(USE_V2=1, USE_V3=1 во всех meta). Реально затронуты: в v2 — dec_gmv_h120,
gmvday_q50/q90, gmv_concentration, dec_orddays_h60; в v3 — gmv_daymed_full.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import build_features_v2 as b2
import build_features_v3 as b3
from common import FEATURES_DIR, TEST_ANCHOR, TRAIN_PARQUET, user_universe

MB = 409
A = TEST_ANCHOR


def main() -> None:
    b2.MAX_BACK = MB
    b3.MAX_BACK = MB
    uni = user_universe()
    lf = pl.scan_parquet(TRAIN_PARQUET)

    p2 = FEATURES_DIR / f"anchor={A.isoformat()}.mb{MB}.extra.parquet"
    if not p2.exists():
        t0 = time.time()
        hist = lf.filter((pl.col("event_date") <= A)
                         & (pl.col("event_date") >= A - pl.duration(days=MB)))
        feats = hist.group_by("user_id").agg(b2.extra_exprs(A)).collect(engine="streaming")
        out = uni.join(feats, on="user_id", how="left")
        zero = [c for c in out.columns
                if c.startswith(("dec_", "hv", "cart_minus", "s2o_cnt", "c2o_cnt"))]
        out = out.with_columns([pl.col(c).fill_null(0) for c in zero])
        casts = [pl.col(c).cast(pl.Float32) for c, dt in zip(out.columns, out.dtypes)
                 if dt == pl.Float64]
        out.with_columns(casts).write_parquet(p2)
        print(f"v2: {out.shape} за {time.time()-t0:.1f}s -> {p2.name}", flush=True)
    else:
        print("v2 уже есть", flush=True)

    # v3: своя функция build(); она и ПИШЕТ в .v3.parquet, и ЧИТАЕТ базовый якорь
    # ради ранговых столбцов (rk_*, burstiness — а burstiness считается из
    # ord_gap_std/ord_gap_mean, оба затронуты). Поэтому подменяем FEATURES_DIR на
    # временный каталог и кладём туда ссылку на ИСПРАВЛЕННЫЙ базовый файл.
    p3 = FEATURES_DIR / f"anchor={A.isoformat()}.mb{MB}.v3.parquet"
    if not p3.exists():
        t0 = time.time()
        tmp = FEATURES_DIR / "_mbfix_tmp"
        tmp.mkdir(exist_ok=True)
        link = tmp / f"anchor={A.isoformat()}.parquet"
        if not link.exists():
            link.symlink_to(FEATURES_DIR / f"anchor={A.isoformat()}.mb{MB}.parquet")
        b3.FEATURES_DIR = tmp
        b3.build(A, uni, lf)
        (tmp / f"anchor={A.isoformat()}.v3.parquet").rename(p3)
        link.unlink()
        tmp.rmdir()
        print(f"v3: за {time.time()-t0:.1f}s -> {p3.name}", flush=True)
    else:
        print("v3 уже есть", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
