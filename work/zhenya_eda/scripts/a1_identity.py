"""Проверка тождества и константы 7.1, на которых построен весь критерий приёмки.

Утверждение команды (TEAM_PLAN.md, NEXT_STEPS.md, KNOWLEDGE.md):
    вклад = 7.1 * запас^2,  запас = s_b/s_m - rho
    -> для прироста 0.0003 нужен запас 0.0065

Проверяем: (1) выполняется ли тождество rho = s_b/s_m для моделей в оболочке;
           (2) действительно ли коэффициент перед запас^2 равен 7.1 для ВСЕХ моделей.
"""
import numpy as np, polars as pl
from itertools import combinations

v = pl.read_parquet("preds/preds_pack/val_preds.parquet").sort("user_id")
y = v["target"].to_numpy().astype(np.float64)
ly = np.log1p(np.clip(y, 0, None))
cols = [c for c in v.columns if c not in ("user_id", "target")]
L = {c: np.log1p(np.clip(v[c].to_numpy().astype(np.float64), 0, None)) for c in cols}
n = len(ly)

def rmsle_lp(lp): return float(np.sqrt(np.mean((lp - ly) ** 2)))

print(f"n={n}  models={len(cols)}")
print("\n--- сольные скоры ---")
for c in sorted(cols, key=lambda c: rmsle_lp(L[c])):
    print(f"  {c:16s} {rmsle_lp(L[c]):.6f}")

# Честный бленд: NNLS в log-пространстве на половине юзеров, замер на другой.
# Для проверки ТОЖДЕСТВА нужен просто оптимальный бленд; берём его на всех юзерах,
# потому что тождество - геометрия, а не обобщение.
from scipy.optimize import nnls
POOL = ["mlpziln_cal", "mlpbin_cal", "mlp2_big_cal", "mlp2_final_cal", "twdeep",
        "twl_repair_ab", "twl_seqoof", "c_ts2_s42", "c_ts2_s1337", "c_twlog_s42",
        "c_twlog_s1337", "c_dirlgb_s42", "c_dirlgb_s1337", "c_xtw_s42",
        "seq2tr_f", "gru_final", "febspec"]
A = np.column_stack([L[c] for c in POOL])
w, _ = nnls(A, ly)
lb = A @ w
eb = lb - ly
sb = float(np.sqrt(np.mean(eb ** 2)))
print(f"\nбленд NNLS ({len(POOL)} моделей) s_b={sb:.6f}  sum(w)={w.sum():.4f}  mean(e_b)={eb.mean():+.6f}")
print("  веса:", {c: round(float(x), 3) for c, x in zip(POOL, w) if x > 1e-4})
