"""
Починка уже собранного тензора: пересобирает только gmv в float32.
В float16 не помещается максимальный дневной GMV (73830 при потолке 65504),
переполнение даёт inf и NaN в лоссе. Полная пересборка не нужна —
seq.f16 и ord.f16 корректны.

  python fix_gmv.py --src train.parquet --out tensor
"""
import numpy as np, pyarrow.parquet as pq, argparse, os
from datetime import date
START = date(2025,1,1)

p = argparse.ArgumentParser()
p.add_argument('--src', default='train.parquet'); p.add_argument('--out', default='tensor')
a = p.parse_args()

m = np.load(f'{a.out}/meta.npz')
uids, n_u, n_d = m['user_ids'], int(m['n_users']), int(m['n_days'])
gmv = np.memmap(f'{a.out}/gmv.f32', np.float32, 'w+', shape=(n_u, n_d)); gmv[:] = 0
done = 0
for b in pq.ParquetFile(a.src).iter_batches(batch_size=2_000_000,
                                            columns=['user_id','event_date','gmv']):
    u = np.searchsorted(uids, b.column('user_id').to_numpy(zero_copy_only=False))
    d = b.column('event_date').to_numpy(zero_copy_only=False)
    d = (d - np.datetime64(START)).astype('timedelta64[D]').astype(np.int32)
    ok = (d >= 0) & (d < n_d)
    gmv[u[ok], d[ok]] = b.column('gmv').to_numpy(zero_copy_only=False)[ok].astype(np.float32)
    done += int(ok.sum()); print(f'  {done:,} строк', flush=True)
gmv.flush()
print('максимум дневного GMV:', float(np.asarray(gmv).max()))
print('inf/nan:', int(np.isinf(gmv).sum()), int(np.isnan(gmv).sum()))
old = f'{a.out}/gmv.f16'
if os.path.exists(old): os.remove(old); print('удалён старый gmv.f16')
print('готово')
