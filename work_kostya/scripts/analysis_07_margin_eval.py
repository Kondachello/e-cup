"""№3: запас и честный вклад kostya46 в бленд по протоколу команды.
Требует preds_pack (val_preds.parquet) и файлы m1/m2 предсказаний. Печатает числа §6."""
import numpy as np, polars as pl
from scipy.optimize import nnls

PACK = "/mnt/user-data/uploads/ozon_cup/work/preds_pack/val_preds.parquet"  # поправь путь при запуске в репо
d = pl.read_parquet(PACK).sort("user_id")
y = np.log1p(d["target"].to_numpy().astype(np.float64))
b = d["blend"].to_numpy().astype(np.float64)
cols = [c for c in d.columns if c not in ("user_id", "target", "blend")]
M = np.stack([d[c].to_numpy().astype(np.float64) for c in cols], axis=1)

def fit_shifts(lp, ly, bins=24):
    qs = np.quantile(lp, np.linspace(0, 1, bins + 1)); qs[0] -= 1e-9; qs[-1] += 1e-9
    cs, ss = [], []
    for i in range(bins):
        m = (lp > qs[i]) & (lp <= qs[i + 1])
        if m.sum() < 500: continue
        cs.append(lp[m].mean()); ss.append(ly[m].mean() - lp[m].mean())
    return np.array(cs), np.array(ss)

def cal_honest(lp, ly, seed=0):
    r = np.random.default_rng(seed); half = r.permutation(len(ly)) < len(ly) // 2
    out = np.empty_like(lp)
    for m in (half, ~half):
        c, s = fit_shifts(lp[m], ly[m]); out[~m] = np.clip(lp[~m] + np.interp(lp[~m], c, s), 0, None)
    return out

mine_raw = (0.25 * np.load("val_pz.npy").astype(np.float64) + 0.25 * np.load("val_two.npy").astype(np.float64)
            + 0.2 * np.load("m2_val_pz.npy").astype(np.float64) + 0.3 * np.load("m2_val_two.npy").astype(np.float64))
mine = cal_honest(mine_raw, y)

sb = np.sqrt(((y - b) ** 2).mean()); sm = np.sqrt(((y - mine) ** 2).mean())
rho = np.corrcoef(y - b, y - mine)[0, 1]
print(f"скор {sm:.6f}  corr {rho:.5f}  ЗАПАС {sb/sm - rho:+.5f}  (бленд {sb:.6f})")

def honest(mat, seed=0):
    r = np.random.default_rng(seed); half = r.permutation(len(y)) < len(y) // 2
    out = np.empty_like(y); ws = []
    for m in (half, ~half):
        w, _ = nnls(mat[~m], y[~m]); ws.append(w)
        out[m] = mat[m] @ w
    return float(np.sqrt(np.mean((out - y) ** 2))), ws

base, _ = honest(M)
plus, ws = honest(np.column_stack([M, mine]))
print(f"NNLS 30 моделей пака: {base:.6f};  + kostya46: {plus:.6f} "
      f"(gain {base - plus:+.6f}, вес {[round(w[-1], 4) for w in ws]})")
rng = np.random.default_rng(0)
gains = []
for t in range(5):
    idx = rng.choice(len(cols), 4, replace=False)
    s, _ = honest(np.column_stack([M, M[:, idx]]))
    gains.append(round(base - s, 6))
print("плацебо (4 существующих колонки повторно):", gains)
