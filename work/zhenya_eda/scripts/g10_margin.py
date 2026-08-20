"""G10. Проверка критерия приёмки «вклад = 7.1 * запас^2».

Вывод формулы. Двухчленная смесь бленда b и модели m в log-пространстве:
  s_new^2 = s_b^2 - (s_b^2 - rho*s_m*s_b)^2 / (s_m^2 - 2*rho*s_m*s_b + s_b^2)
Обозначим r = s_b/s_m (<=1, бленд лучше) и запас d = r - rho. Тогда
  выигрыш ~ (s_b^2 - s_new^2)/(2 s_b) = s_b * d^2 / (2*(1 - r^2 + 2*r*d))
Коэффициент перед d^2 равен s_b/(2*(1-r^2)) и ЗАВИСИТ ОТ СОЛЬНОГО СКОРА МОДЕЛИ.
7.1 соответствует 1-r^2 = s_b/14.2 = 0.117, то есть r=0.940, s_m=1.774 —
это класс февральского специалиста. Для сильной модели коэффициент во много раз больше.
Проверяем на 30 моделях действующего пакета.
"""
import numpy as np, polars as pl
v = pl.read_parquet("work/preds_pack/val_preds.parquet").sort("user_id")
ly = np.log1p(np.clip(v["target"].to_numpy().astype(np.float64),0,None))
b = v["blend"].to_numpy().astype(np.float64)
eb = b - ly; sb = float(np.sqrt(np.mean(eb**2)))
cols = [c for c in v.columns if c not in ("user_id","target","blend")]
print(f"бленд s_b={sb:.6f}, моделей {len(cols)}\n")
print(f"{'модель':20s} {'s_m':>9} {'rho':>8} {'запас':>10} {'ФАКТ выигрыш':>13} {'7.1*d^2':>11} {'моя формула':>12}")
rows=[]
for c in cols:
    m = v[c].to_numpy().astype(np.float64)
    em = m - ly; sm = float(np.sqrt(np.mean(em**2)))
    rho = float(np.corrcoef(em, eb)[0,1])
    r = sb/sm; d = r - rho
    dd = em - eb
    w = float(-np.dot(eb, dd)/max(np.dot(dd,dd),1e-12))
    snew = float(np.sqrt(np.mean(((1-w)*b + w*m - ly)**2)))
    fact = sb - snew
    f_team = 7.1*d*d
    f_mine = sb*d*d/(2*(1 - r*r + 2*r*d)) if (1-r*r+2*r*d) > 0 else np.nan
    rows.append((c, sm, rho, d, fact, f_team, f_mine, w))
rows.sort(key=lambda x: -x[4])
for c,sm,rho,d,fact,ft,fm,w in rows:
    print(f"{c:20s} {sm:>9.5f} {rho:>8.5f} {d:>+10.5f} {fact:>13.6f} {ft:>11.6f} {fm:>12.6f}")

f = np.array([r[4] for r in rows]); t = np.array([r[5] for r in rows]); mm = np.array([r[6] for r in rows])
ok = np.isfinite(mm) & (f > 1e-9)
print(f"\nсредняя |ошибка| предсказания вклада:")
print(f"  формула команды 7.1*запас^2 : {np.mean(np.abs(t[ok]-f[ok])):.6f}")
print(f"  формула с учётом сольного скора: {np.mean(np.abs(mm[ok]-f[ok])):.6f}")
print(f"\n=== ЧТО ЭТО МЕНЯЕТ ДЛЯ ПОРОГА ПРИЁМКИ ===")
print(f"{'сольный скор модели':>22} {'r':>8} {'коэффициент':>13} {'запас для +0.0003':>19}")
for s_m in (1.6700, 1.6800, 1.7000, 1.7500, 1.8266):
    r = sb/s_m; k = sb/(2*(1-r*r))
    need = np.sqrt(0.0003/k)
    print(f"{s_m:>22.4f} {r:>8.4f} {k:>13.1f} {need:>19.5f}")
print(f"\nкоманда говорит команде: «нужен запас 0.0065, рекорд проекта 0.00193».")
print(f"это верно только для СЛАБОЙ модели (~1.77). Для модели уровня 1.68 хватает вчетверо меньшего.")
