"""Что именно поменялось в признаках тестового якоря при MAX_BACK 379 -> 409."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES_DIR, TEST_ANCHOR

A = TEST_ANCHOR.isoformat()
old = pl.read_parquet(FEATURES_DIR / f"anchor={A}.parquet").sort("user_id")
new = pl.read_parquet(FEATURES_DIR / f"anchor={A}.mb409.parquet").sort("user_id")
assert old["user_id"].to_list() == new["user_id"].to_list()
assert old.columns == new.columns, (set(old.columns) ^ set(new.columns))
print(f"строк {old.height}, столбцов {len(old.columns)}\n")

rows = []
for c in old.columns:
    if c in ("user_id", "anchor_date"):
        continue
    a = old[c].to_numpy().astype(np.float64)
    b = new[c].to_numpy().astype(np.float64)
    na, nb = np.isnan(a), np.isnan(b)
    nullflip = int((na ^ nb).sum())
    m = ~(na | nb)
    if m.sum() == 0:
        if nullflip:
            rows.append((c, 1.0, nullflip, np.nan, np.nan, np.nan))
        continue
    d = b[m] - a[m]
    frac = float((np.abs(d) > 1e-6).mean())
    if frac == 0 and nullflip == 0:
        continue
    rows.append((c, frac, nullflip, float(a[m].mean()), float(b[m].mean()),
                 float(np.abs(d).max())))

rows.sort(key=lambda r: -r[1])
print(f"{'столбец':24s} {'доля изм':>9s} {'null-флип':>9s} {'старое ср':>12s} {'новое ср':>12s} {'max|d|':>10s}")
for c, frac, nf, ma, mb_, mx in rows:
    print(f"{c:24s} {frac:9.4f} {nf:9d} {ma:12.4f} {mb_:12.4f} {mx:10.3f}")
print(f"\nЗАТРОНУТО СТОЛБЦОВ: {len(rows)} из {len(old.columns) - 2}")

# ключевой столбец: сдвиг в логарифме
for c in ("gmv_sum_full", "log_gmv_sum_full", "tenure", "active_days_full"):
    a = old[c].to_numpy().astype(np.float64)
    b = new[c].to_numpy().astype(np.float64)
    if c.startswith("log_"):
        d = b - a
    else:
        d = np.log1p(np.clip(b, 0, None)) - np.log1p(np.clip(a, 0, None))
    print(f"\n{c}: изменилось у {float((np.abs(b-a)>1e-6).mean()):.4f} юзеров; "
          f"сдвиг log1p ср {d.mean():.5f} sd {d.std():.5f} p99 {np.quantile(d,0.99):.3f}; "
          f"spearman {np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0,1]:.5f}")

# юзеры, невидимые при 379 и видимые при 409
gone = np.isnan(old["tenure"].to_numpy().astype(np.float64))
back = ~np.isnan(new["tenure"].to_numpy().astype(np.float64))
print(f"\nюзеров без единого события в окне 379д: {int(gone.sum())}; "
      f"из них появляются при 409д: {int((gone & back).sum())}")
