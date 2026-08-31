"""Сверка воспроизведённых файлов с каноническими work_kostya/preds/*.parquet.
PASS: user_id идентичны и max |Δ log1p(pred)| < 1e-6 (бит-повтор при KTHREADS=2 и
lightgbm==4.7.0; иной конфиг даёт float-дрейф ~1e-7).

Сверяется только kostya46 — единственный член бленда из этого каталога. Побочные
kostya46shade_* reproduce.py тоже пишет, но в состав они не входят и канонической
копии для них в репозитории нет."""
import numpy as np, polars as pl, sys
from pathlib import Path
from paths import WORK

repro = WORK.parent / "preds_repro"
canon = WORK.parent / "preds"
ok = True
for f in ["kostya46_val", "kostya46_test"]:
    a = pl.read_parquet(str(canon / f"{f}.parquet")).sort("user_id")
    b = pl.read_parquet(str(repro / f"{f}.parquet")).sort("user_id")
    same_ids = (a["user_id"] == b["user_id"]).all()
    d = np.abs(np.log1p(a["pred"].to_numpy()) - np.log1p(b["pred"].to_numpy())).max()
    status = "PASS" if same_ids and d < 1e-6 else "FAIL"
    ok &= status == "PASS"
    print(f"{f}: ids={'ok' if same_ids else 'MISMATCH'}  max|dlog|={d:.2e}  {status}")
sys.exit(0 if ok else 1)
