# -*- coding: utf-8 -*-
"""/. Математика под точечный процесс: оценщик цели, разрыв Йенсена,
идентифицируемость фазы (реальные числа из train.parquet)."""
import math, sys, json
import numpy as np, polars as pl
from datetime import date, timedelta
sys.stdout.reconfigure(encoding="utf-8")
rng=np.random.default_rng(11)

print("=== L1.1 РАЗРЫВ ЙЕНСЕНА: НАСКОЛЬКО ВЕЛИКА ОШИБКА log(E[S]) ===")
print("Метрика — RMSLE на S30, значит оптимальный прогноз = expm1(E[log1p(S30)]).")
print("Соблазн: посчитать E[S30] и взять log1p. Это ЗАВЫШАЕТ (log1p вогнута).\n")
print(f"{'сценарий':34s}{'E[S]':>10}{'log1p(E[S])':>13}{'E[log1p(S)]':>13}{'разрыв':>9}")
def compound(lam, mu, sig, n=400_000):
    N=rng.poisson(lam,n)
    S=np.array([rng.lognormal(mu,sig,k).sum() if k else 0.0 for k in N]) if n<=0 else None
    # быстрее: сумма N лог-нормалей через накопление
    S=np.zeros(n); mx=N.max()
    for k in range(1,mx+1):
        m=N>=k
        S[m]+=rng.lognormal(mu,sig,m.sum())
    return S
for lam,mu,sig,lab in [(0.5,7.0,1.0,"редкий, чек ~1100"),(2.0,7.0,1.0,"средний"),
                       (2.0,7.0,1.8,"средний, чек тяжелее"),(8.0,6.0,1.0,"частый")]:
    S=compound(lam,mu,sig)
    a=math.log1p(S.mean()); b=float(np.log1p(S).mean())
    print(f"{lab:34s}{S.mean():10.0f}{a:13.4f}{b:13.4f}{a-b:9.4f}")
print("\n  Разрыв 0.5-2.5 в логах при нашем F0=1.647 — то есть путь через E[S]")
print("  даёт ошибку ПОРЯДКА самой метрики. Это не поправка, это другой ответ.")

print("\n=== L1.1 РЕЦЕПТ: точный E[log1p(S30)] за 3 шага ===")
print("  S30 = сумма N чеков, N — счётчик процесса за 30 дней.")
print("  E[log1p(S)] = P(N=0)*0 + sum_{n>=1} P(N=n) * E[log1p(S_n)]")
print("  1) P(N=n) — из модели (для Пуассона Poisson(Lambda), Lambda=int lambda dt);")
print("  2) S_n = сумма n чеков ~ логнормаль по Фентону-Уилкинсону (моменты складываются);")
print("  3) E[log1p(LN(m,s))] — одномерная квадратура Гаусса-Эрмита, 20 узлов.")
print("  Обрезать n на квантиле 0.999 распределения N.")

# точность квадратуры против MC
def E_log1p_LN(m,s,nodes=20):
    x,w=np.polynomial.hermite_e.hermegauss(nodes)
    return float((w*np.log1p(np.exp(m+s*x))).sum()/w.sum())
print(f"\n  проверка квадратуры (одна логнормаль):")
print(f"  {'m':>5}{'s':>5}{'квадратура':>13}{'MC 2e6':>11}{'ошибка':>10}")
for m,s in [(7.0,1.0),(7.0,1.8),(5.0,2.5),(9.0,0.7)]:
    q=E_log1p_LN(m,s); mc=float(np.log1p(rng.lognormal(m,s,2_000_000)).mean())
    print(f"  {m:5.1f}{s:5.1f}{q:13.5f}{mc:11.5f}{q-mc:+10.5f}")

df=pl.read_parquet("../repo2/train.parquet",columns=["user_id","event_date","to_ord"])
buys=df.filter(pl.col("to_ord")>0).select(["user_id","event_date"]).sort(["user_id","event_date"])
g=buys.group_by("user_id").agg([pl.col("event_date").diff().dt.total_days().alias("d"),
                                pl.len().alias("k")])
g=g.with_columns([pl.col("d").list.mean().alias("mT"), pl.col("d").list.std().alias("sT")])
g=g.filter(pl.col("k")>=3).drop_nulls(["mT","sT"])
mT=g["mT"].to_numpy(); sT=g["sT"].to_numpy(); k=g["k"].to_numpy()
tau_pop=float(np.std(mT[(k>=8)]))            # разброс ЛИЧНЫХ средних интервалов
s_within=float(np.median(sT[(k>=8)]))        # типичный внутриюзерный разброс
print(f"  юзеров с >=3 покупками: {len(mT):,}")
print(f"  tau_pop (разброс личных циклов, k>=8) = {tau_pop:.1f} дней")
print(f"  s_within (медианный внутриюзерный sd) = {s_within:.1f} дней")
print(f"\n  ВЕС ЛИЧНОГО ЦИКЛА: w = tau_pop^2*(k-1) / (tau_pop^2*(k-1) + s_within^2)")
print(f"  {'покупок k':>10}{'w личного':>12}{'что значит':>28}")
for kk in (2,3,4,5,6,8,12,20):
    w=tau_pop**2*(kk-1)/(tau_pop**2*(kk-1)+s_within**2)
    lab="популяционный" if w<0.35 else ("смесь" if w<0.65 else "личный")
    print(f"  {kk:10d}{w:12.3f}{lab:>28}")
kstar=1+ (s_within/tau_pop)**2
print(f"\n  ПОРОГ w=0.5: k > 1 + (s/tau)^2 = {kstar:.1f}, то есть k >= {math.ceil(kstar)} покупок")
json.dump(dict(tau_pop=tau_pop,s_within=s_within,kstar=kstar),
          open("out/l1_tpp.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
