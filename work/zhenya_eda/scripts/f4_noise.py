"""F4. Форензика GMV: есть ли квантование и подмешан ли шум.
Если в цену добавлен шум, у RMSLE есть неустранимый пол, и команде важно знать его."""
import polars as pl, numpy as np
df = pl.read_parquet("train.parquet", columns=["user_id","event_date","to_ord","gmv"])
p = df.filter(pl.col("to_ord")==1)["gmv"].to_numpy()
print(f"наблюдений: {len(p):,}")

print("\n=== 1. МИКРОСТРУКТУРА в узкой полосе (ищем пики каталога) ===")
band = p[(p>=0.030)&(p<=0.050)]
print(f"  в полосе [0.030, 0.050]: {len(band):,} наблюдений")
h, edges = np.histogram(band, bins=800)
print(f"  гистограмма 800 корзин: max={h.max()} median={np.median(h):.0f} "
      f"отношение max/median={h.max()/max(np.median(h),1):.2f}")
sp = np.abs(np.fft.rfft(h - h.mean()))
k = np.argmax(sp[3:]) + 3
print(f"  сильнейшая периодика: период {len(h)/k:.1f} корзин = {(0.050-0.030)/(len(h)/k)*1e4:.2f}e-4 по цене")
print(f"  мощность пика / средней: {sp[k]/sp[3:].mean():.2f}  (>5 = реальная сетка)")

print("\n=== 2. ЛОГАРИФМИЧЕСКАЯ СЕТКА (частая при анонимизации) ===")
lp = np.log(p[p>0])
for name, arr in (("log(price)", lp), ("price", p)):
    d = np.diff(np.sort(np.unique(np.round(arr, 6))))
    d = d[d>0]
    print(f"  {name:11s}: медианный зазор между уникальными {np.median(d):.3e}")

print("\n=== 3. ЕСТЬ ЛИ ОБЩИЙ ДЕЛИТЕЛЬ (тест Фурье по кандидатам) ===")
sub = p[:400000]
best=[]
for unit in np.concatenate([np.linspace(1e-4, 5e-3, 200), np.linspace(5e-3, 5e-2, 100)]):
    s = np.abs(np.mean(np.exp(2j*np.pi*sub/unit)))
    best.append((s, unit))
best.sort(reverse=True)
for s, unit in best[:5]:
    print(f"  единица {unit:.5f}: сила сетки {s:.4f}   (>0.3 = настоящая сетка, ~0.00 = непрерывно)")

print("\n=== 4. ПОВТОРЫ ЗНАЧЕНИЙ: каталог или шум ===")
vals, cnt = np.unique(np.round(p,4), return_counts=True)
print(f"  уникальных {len(vals):,}, кратность: mean={cnt.mean():.2f} max={cnt.max()}")
print(f"  доля значений, встреченных 1 раз: {100*(cnt==1).mean():.1f}%")
exp_uniq = len(p) * (1 - np.exp(-1))          # если бы значения были из ~len(p) равновероятных
print(f"  при чисто непрерывных ценах ожидалось бы ~100% одиночных")
print(f"  ВЫВОД: {'дискретный каталог со сглаживанием' if (cnt==1).mean()<0.9 else 'непрерывные значения'}")

print("\n=== 5. НЕУСТРАНИМЫЙ ПОЛ: разброс дневного gmv при ОДНОМ товаре ===")
print("  (если цена одного и того же товара дрожит, сумма за 30 дней тоже дрожит)")
q = np.quantile(p, [.1,.25,.5,.75,.9])
print(f"  цены: p10={q[0]:.3f} p25={q[1]:.3f} p50={q[2]:.3f} p75={q[3]:.3f} p90={q[4]:.3f}")
print(f"  sd(log price) = {np.std(np.log(p[p>0])):.4f}  -> разброс цены товара {100*np.std(np.log(p[p>0])):.0f}% в логарифме")
