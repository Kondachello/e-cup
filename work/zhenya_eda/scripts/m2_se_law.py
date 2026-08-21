"""M2. ЗАКОН для sigma_kappa. Вывод:
   kappa = c_test/c_val,  sigma(c) = sd(e*h)/(sqrt(n)*mean(h^2))
   валидационный выигрыш оси в MSE:  G = c_val^2 * mean(h^2)
   =>  sigma_kappa = sigma(c)/|c_val| = sd(e*h)/(sqrt(n)*mean(h^2)*|c_val|)
   при слабой зависимости e и h:  sd(e*h) ~ sd(e)*sqrt(mean(h^2))
   =>  sigma_kappa ~ sd(e)/sqrt(n*G)
Это ОДНА формула, объясняющая и мои 0.33, и параболические 0.042-0.055."""
import os, numpy as np, polars as pl
from pathlib import Path
from sklearn.linear_model import Ridge
CACHE = Path(os.environ["ZH_CACHE"])
v = pl.read_parquet("work/preds_pack/val_preds.parquet").sort("user_id")
ly = np.log1p(np.clip(v["target"].to_numpy().astype(np.float64),0,None))
e  = v["blend"].to_numpy().astype(np.float64) - ly
sb = float(np.sqrt(np.mean(e**2)))
X = pl.read_parquet(CACHE/"a2026-01-14.parquet")
B = X.select([c for c in X.columns if c.startswith("b_")]).to_numpy().astype(np.float64)
B = np.sign(np.nan_to_num(B))*np.log1p(np.abs(np.nan_to_num(B)))
rng = np.random.default_rng(0); n=len(e); NPUB=50_000

AX={}
AX["уровень"]=np.ones(n)
q=np.quantile(v["blend"].to_numpy(),np.linspace(0,1,11)); q[0],q[-1]=-np.inf,np.inf
b=np.digitize(v["blend"].to_numpy(),q[1:-1])
AX["сегментная ступенька"]=np.array([-e[b==k].mean() for k in range(10)])[b]
mu,sd_=B.mean(0),B.std(0)+1e-9
AX["стек по признакам"]=Ridge(alpha=10.).fit((B-mu)/sd_,-e).predict((B-mu)/sd_)
AX["дельта модели"]=v["mlpbin_cal"].to_numpy().astype(np.float64)-v["blend"].to_numpy().astype(np.float64)
AX["дельта fusion"]=v["fusion_v3_avg_cal"].to_numpy().astype(np.float64)-v["blend"].to_numpy().astype(np.float64)
AX["крошка"]=rng.normal(size=n)*0.01

print(f"sd(e) = {sb:.4f}, публика n = {NPUB:,}\n")
print(f"{'ось':22s} {'G_val(RMSLE)':>13} {'G_mse':>10} {'σ эмпир.':>10} {'σ по закону':>12} {'отн.':>6}")
for nm,h in AX.items():
    hh=float(np.mean(h*h)); c=-float(np.dot(e,h)/np.dot(h,h))
    G_mse = c*c*hh                                   # выигрыш при применении оптимального c
    G_rmsle = G_mse/(2*sb)
    ks=[]
    for _ in range(200):
        p=rng.permutation(n)[:NPUB]
        ks.append(-float(np.dot(e[p],h[p])/np.dot(h[p],h[p])))
    emp=float(np.std(ks,ddof=1))
    law = sb/np.sqrt(NPUB*max(G_mse,1e-18))
    print(f"{nm:22s} {G_rmsle:>13.6f} {G_mse:>10.6f} {emp:>10.4f} {law:>12.4f} {law/max(emp,1e-9):>6.2f}")

print(f"\n=== ЗАКОН: sigma_kappa = sd(e)/sqrt(n*G_mse),  G_mse = 2*sd(e)*G_rmsle ===")
print(f"{'валид. выигрыш оси':>20} {'σ_kappa':>10}   кто это")
for g,who in [(0.0001,"мелкая проба probes_5"),(0.0003,"порог приёмки"),
              (0.001,"заметная ось"),(0.005,"крупная ось"),
              (0.0116,"глобальный сдвиг +0.1163 -> его выигрыш"),(0.02,"очень крупная")]:
    gm = 2*sb*g
    print(f"{g:>20.4f} {sb/np.sqrt(NPUB*gm):>10.4f}   {who}")
print(f"\nМОИ 0.33 отвечают оси с выигрышем {sb**2/(NPUB*0.33**2)/(2*sb):.5f} — это мелкие пробы.")
print(f"ТВОИ 0.042-0.055 отвечают выигрышу {sb**2/(NPUB*0.055**2)/(2*sb):.4f}..{sb**2/(NPUB*0.042**2)/(2*sb):.4f} — крупные оси.")
print("СПОР СНЯТ: обе цифры верны, просто относятся к осям разной величины.")
