"""
Шаг 1: превращаем прореженный parquet в плотный дневной тензор.

На выходе (папка --out):
  seq.f16     memmap [n_users, n_days, 10]  — признаки дня, уже под log1p
  gmv.f32     memmap [n_users, n_days]      — сырой дневной GMV (float32: в fp16 не влезает)
  ord.f16     memmap [n_users, n_days]      — дневное число заказов (вспом. таргет)
  meta.npz    user_ids, valid_anchor, calendar

Память: пишем через np.memmap, в RAM единовременно один батч parquet.
Итоговый размер ~2.5 ГБ на диске.
"""
import numpy as np, pyarrow.parquet as pq, argparse, os
from datetime import date

SRC_COLS = ['search','cat','search_to_cart','search_to_ord',
            'cat_to_cart','cat_to_ord','gmv_search','gmv_cat','searches']
N_CH = len(SRC_COLS) + 1                      # +1: флаг «строка за этот день есть»
START = date(2025,1,1)

def main(src, out, batch_rows=2_000_000):
    os.makedirs(out, exist_ok=True)
    pf = pq.ParquetFile(src)
    n_days = (date(2026,2,13) - START).days + 1          # 409

    # --- сначала список юзеров (данные отсортированы по user_id, поэтому дёшево)
    uids = np.unique(np.concatenate([
        b.column('user_id').to_numpy(zero_copy_only=False)
        for b in pf.iter_batches(batch_size=batch_rows, columns=['user_id'])]))
    n_users = len(uids)
    print(f'users={n_users} days={n_days} channels={N_CH}')

    seq = np.memmap(f'{out}/seq.f16', np.float16, 'w+', shape=(n_users, n_days, N_CH))
    gmv = np.memmap(f'{out}/gmv.f32', np.float32, 'w+', shape=(n_users, n_days))
    ordr= np.memmap(f'{out}/ord.f16', np.float16, 'w+', shape=(n_users, n_days))
    seq[:] = 0; gmv[:] = 0; ordr[:] = 0

    cols = ['user_id','event_date'] + SRC_COLS + ['gmv','to_ord']
    done = 0
    for b in pf.iter_batches(batch_size=batch_rows, columns=cols):
        u = np.searchsorted(uids, b.column('user_id').to_numpy(zero_copy_only=False))
        d = b.column('event_date').to_numpy(zero_copy_only=False)
        d = (d - np.datetime64(START)).astype('timedelta64[D]').astype(np.int32)
        ok = (d >= 0) & (d < n_days)
        u, d = u[ok], d[ok]
        for j, c in enumerate(SRC_COLS):
            v = b.column(c).to_numpy(zero_copy_only=False)[ok].astype(np.float32)
            seq[u, d, j] = np.log1p(v).astype(np.float16)
        seq[u, d, N_CH-1] = 1.0                                     # флаг присутствия
        gmv[u, d]  = b.column('gmv').to_numpy(zero_copy_only=False)[ok].astype(np.float32)
        ordr[u, d] = b.column('to_ord').to_numpy(zero_copy_only=False)[ok].astype(np.float16)
        done += len(u); print(f'  {done:,} строк', flush=True)
    seq.flush(); gmv.flush(); ordr.flush()

    # --- календарь: сезонность и всплески распродаж
    days = np.arange(n_days)
    dates = np.array([np.datetime64(START) + np.timedelta64(int(i),'D') for i in days])
    dow  = ((dates - np.datetime64('2025-01-06')).astype('timedelta64[D]').astype(int)) % 7
    doy  = np.array([int(str(x)[5:7])*31 + int(str(x)[8:10]) for x in dates], float)
    md   = np.array([str(x)[5:10] for x in dates])
    peak = np.isin(md, ['02-14','02-20','02-21','02-22','02-23','03-06','03-07','03-08',
                        '11-11','11-24','12-30','12-31','01-01']).astype(float)
    pre  = np.isin(md, ['02-18','02-19','03-04','03-05','11-09','11-10']).astype(float)
    cal = np.stack([
        np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
        (dow >= 5).astype(float),
        np.sin(2*np.pi*doy/372), np.cos(2*np.pi*doy/372),
        peak, pre, days/n_days,
    ], 1).astype(np.float32)

    # --- какие якоря допустимы: у юзера есть строка за последние 30 дней (как в тестовой когорте)
    present = np.asarray(seq[:, :, N_CH-1], np.float32)
    cum = np.concatenate([np.zeros((n_users,1),np.float32), np.cumsum(present,1)], 1)
    valid = np.empty((n_users, n_days), bool)
    for a in range(n_days):
        lo = max(0, a-29)
        valid[:, a] = (cum[:, a+1] - cum[:, lo]) > 0
    np.savez(f'{out}/meta.npz', user_ids=uids, calendar=cal, valid_anchor=valid,
             n_users=n_users, n_days=n_days, n_ch=N_CH)
    print('готово:', out)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='train.parquet')
    p.add_argument('--out', default='tensor')
    a = p.parse_args(); main(a.src, a.out)
