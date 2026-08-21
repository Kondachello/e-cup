"""mdl_wulfen. Чья σ_κ верна: моя (сэмплинг c на 50k) или параболическая (шум скора 0.000022)?
Решается прямым замером: строим оси с заданным q, режем 50k, смотрим разброс κ.

Их параметризация (kappa_registry): S(b)² = F0² − 2bc + b²q, κ = c/q.
Отсюда c = -mean_P(e·h), q = mean_P(h²) при h = направление.
Их σ:  σ = шум_LB·(F0+S)/(2q),  шум_LB = 0.000022
Моя σ: σ = sd(e·h)/(√n · q)
"""
import os, numpy as np, polars as pl
from pathlib import Path
CACHE = Path(os.environ["ZH_CACHE"])
v = pl.read_parquet("work/preds_pack/val_preds.parquet").sort("user_id")
ly = np.log1p(np.clip(v["target"].to_numpy().astype(np.float64),0,None))
e  = v["blend"].to_numpy().astype(np.float64) - ly
F0 = float(np.sqrt(np.mean(e**2))); n = len(e); NPUB = 50_000
rng = np.random.default_rng(0)
NOISE_LB = 0.000022
print(f"F0 = {F0:.6f},  n = {n:,},  публика = {NPUB:,}\n")

# оси разной величины: масштабируем случайное направление так, чтобы получить нужный q
base = rng.normal(size=n)
base = base/np.sqrt(np.mean(base**2))
real = v["fusion_v3_avg_cal"].to_numpy().astype(np.float64) - v["blend"].to_numpy().astype(np.float64)
real = real/np.sqrt(np.mean(real**2))

print(f"{'ось':16s} {'q':>10} {'κ':>8} {'σ эмпир.':>10} {'σ моя':>9} {'σ параб.':>9} {'парабола/эмп':>13}")
for tag, d in (("случайная", base), ("настоящая", real)):
    for q in (0.0006592, 0.0022942, 0.0028048, 0.02):
        h = d*np.sqrt(q)                      # mean(h²) = q
        c = -float(np.mean(e*h)); kap = c/q
        ks = []
        for _ in range(300):
            p = rng.permutation(n)[:NPUB]
            ks.append(-float(np.mean(e[p]*h[p]))/float(np.mean(h[p]*h[p])))
        emp = float(np.std(ks, ddof=1))
        mine = float(np.std(e*h))/(np.sqrt(NPUB)*q)
        S = float(np.sqrt(max(F0**2 - 2*c + q, 1e-12)))
        par = NOISE_LB*(F0+S)/(2*q)
        print(f"{tag:16s} {q:>10.7f} {kap:>8.3f} {emp:>10.4f} {mine:>9.4f} {par:>9.4f} {par/emp:>13.3f}")

print(f"\n=== ВЕРДИКТ ===")
print("если «σ моя» ложится на «σ эмпир.», а параболическая систематически ниже —")
print("значит параболическая недооценивает неопределённость κ, и кучность 0.601/0.529")
print("не является доказательством малого разброса.")
