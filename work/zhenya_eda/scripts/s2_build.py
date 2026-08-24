"""S2. Что я МОГУ построить из имеющегося, и чего это стоит по прогнозу."""
import numpy as np, polars as pl
exec(open('../zhenya/scripts/s1_lbpred.py',encoding='utf-8').read().split('print(f"\n=== ПРОВЕРКА')[0])

t = pl.read_parquet('work/preds_pack/test_preds.parquet').sort('user_id')
lp_blend = t['blend'].to_numpy().astype(float)
carry, mdl_tektit = z['carry_lp'], z['dir_old']
REF_MEAN, REF_SD, STEP, A_OLD = 2.324718457996938, 1.632001151855992, 0.469, 0.894

b = REF_SD/lp_blend.std()
matched = (REF_MEAN - b*lp_blend.mean()) + b*lp_blend
full = matched + carry
base = ref + STEP*(full - ref)
lp_pipe = base + A_OLD*mdl_tektit                      # без нового silence-направления (нет модели)

print(f"{'кандидат':40s} {'прогноз':>11} {'против ref':>12} {'против 1.64637':>15}")
BEST_THEIRS = 1.6463720678387952
for nm, lp in (("бленд, приведённый к моментам", matched),
               ("+ цепочка (full)", full),
               ("+ шаг к опоре (base)", base),
               ("+ старое silence-направление", lp_pipe)):
    p = predict(lp)
    print(f"{nm:40s} {p:>11.7f} {p-1.6489446:>+12.7f} {p-BEST_THEIRS:>+15.7f}")

print(f"\n=== ДОБАВЛЯЕМ МОИ НАПРАВЛЕНИЯ -Y4 ===")
import os
for nm in ("Y1_ya_full","Y2_pre8mar","Y3_feb23","Y4_post8mar"):
    d = pl.read_parquet(f"../zhenya/out/dir_{nm}.parquet").sort('user_id')['step'].to_numpy()
    best=(None,9); 
    for c in np.arange(-1.0, 1.01, 0.1):
        p = predict(lp_pipe + c*d)
        if p<best[1]: best=(c,p)
    print(f"  {nm:14s} лучший коэффициент {best[0]:+.1f} -> прогноз {best[1]:.7f} "
          f"({best[1]-predict(lp_pipe):+.7f} к базе)")
print(f"\nОГОВОРКА: базис предиктора всего 2 файла, поэтому почти всё новое")
print(f"направление попадает в ОРТОГОНАЛЬНЫЙ остаток, который оценивается")
print(f"ковариацией с ВАЛИДАЦИОННЫМ таргетом. Их же документация: для")
print(f"принципиально новых направлений MAE ~0.0015 даже при базисе из 50 файлов.")
