"""M1. Что именно меряет SE(κ). Три РАЗНЫЕ величины, которые я склеил в одну:
  sigma_arith  — ошибка восстановления k из арифметики скоров (скоры точные!)
  sigma_sample — расхождение k, померенной на 50k публики, и истинной k окна
  tau_class    — истинный разброс κ между осями одного класса
Меряем sigma_sample ПРЯМО: режем валидацию на 50k и 200k и сравниваем k."""
import os, numpy as np, polars as pl
from pathlib import Path
from sklearn.linear_model import Ridge
CACHE = Path(os.environ["ZH_CACHE"])
v = pl.read_parquet("work/preds_pack/val_preds.parquet").sort("user_id")
ly = np.log1p(np.clip(v["target"].to_numpy().astype(np.float64),0,None))
e  = v["blend"].to_numpy().astype(np.float64) - ly
X = pl.read_parquet(CACHE/"a2026-01-14.parquet")
B = X.select([c for c in X.columns if c.startswith("b_")]).to_numpy().astype(np.float64)
B = np.sign(np.nan_to_num(B))*np.log1p(np.abs(np.nan_to_num(B)))
rng = np.random.default_rng(0)

def k_of(h, idx):
    """k = -<e,h>/||h||^2 на подвыборке idx"""
    hh = float(np.dot(h[idx],h[idx]))
    return -float(np.dot(e[idx],h[idx]))/max(hh,1e-12)

AX = {}
AX["уровень"] = np.ones(len(e))
q = np.quantile(v["blend"].to_numpy(), np.linspace(0,1,11)); q[0],q[-1]=-np.inf,np.inf
b = np.digitize(v["blend"].to_numpy(), q[1:-1])
AX["сегментная ступенька"] = np.array([-e[b==k].mean() for k in range(10)])[b]
mu,sd = B.mean(0), B.std(0)+1e-9
AX["стек по признакам"] = Ridge(alpha=10.).fit((B-mu)/sd, -e).predict((B-mu)/sd)
AX["дельта модели"] = v["mlpbin_cal"].to_numpy().astype(np.float64) - v["blend"].to_numpy().astype(np.float64)
AX["дельта fusion"] = v["fusion_v3_avg_cal"].to_numpy().astype(np.float64) - v["blend"].to_numpy().astype(np.float64)
AX["крошка (шум)"] = rng.normal(size=len(e))*0.01

n = len(e); NPUB = 50_000
print(f"n={n:,}, публика {NPUB:,}\n")
print(f"{'ось':24s} {'k(все 250k)':>12} {'sd k по 50k':>12} {'sd(k50 - k200)':>15} {'|c_val|':>9}")
res={}
for nm,h in AX.items():
    k_all = k_of(h, np.arange(n))
    ks, kd = [], []
    for _ in range(200):
        p = rng.permutation(n); pub, priv = p[:NPUB], p[NPUB:]
        a_, b_ = k_of(h,pub), k_of(h,priv)
        ks.append(a_); kd.append(a_-b_)
    res[nm]=(k_all, np.std(ks,ddof=1), np.std(kd,ddof=1))
    print(f"{nm:24s} {k_all:>12.4f} {np.std(ks,ddof=1):>12.4f} {np.std(kd,ddof=1):>15.4f} {abs(k_all):>9.4f}")

print(f"\n=== ЧТО ЭТО ЗНАЧИТ ===")
print("sd k по 50k — это и есть sigma_sample: насколько публика врёт про истинную k окна.")
print("Она РАЗНАЯ у разных осей и определяется геометрией, а не общей константой.")
print("\nМоя ошибка: я взял верхнюю оценку (мелкие пробы) и распространил на все классы.")
print("Твоя параболическая 0.042-0.055 — это, судя по величине, оси с БОЛЬШИМ |c_val|.")
for nm,(k_all,s50,sd_) in res.items():
    rel = s50/max(abs(k_all),1e-9)
    print(f"  {nm:24s} sigma_sample={s50:.4f}  относительная {rel:.3f}")
