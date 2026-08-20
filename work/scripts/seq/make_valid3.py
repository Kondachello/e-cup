"""
Пересчёт маски допустимых якорей под правило отбора юзеров.

Юниверс соревнования отобран как активный в КАЖДОМ из трёх 30-дневных блоков
перед тестовым якорем — проверено, все 250000 этому удовлетворяют. Наш прежний
фильтр требовал активности только в одном блоке и пускал в обучение на 10-13%
больше слабых юзеров, среди которых вдвое больше молчащих. Модель на такой
популяции занижает базовый уровень.

Дописывает в meta.npz поле valid_anchor3, тензор не трогает.

  python make_valid3.py --data tensor
"""
import numpy as np, argparse

p = argparse.ArgumentParser(); p.add_argument('--data', default='tensor')
a = p.parse_args()

m = dict(np.load(f'{a.data}/meta.npz'))
n_u, n_d, n_ch = int(m['n_users']), int(m['n_days']), int(m['n_ch'])
seq = np.memmap(f'{a.data}/seq.f16', np.float16, 'r', shape=(n_u, n_d, n_ch))

print('читаю флаг присутствия...')
present = np.asarray(seq[:, :, n_ch-1], np.float32)
cum = np.concatenate([np.zeros((n_u,1), np.float32), np.cumsum(present, 1)], 1)
def act(lo, hi):                       # активен ли в [lo, hi] включительно
    lo = max(lo, 0)
    return (cum[:, hi+1] - cum[:, lo]) > 0

v3 = np.zeros((n_u, n_d), bool)
for A in range(n_d):
    ok = act(A-29, A)
    if A-30 >= 0: ok &= act(A-59, A-30)
    if A-60 >= 0: ok &= act(A-89, A-60)
    v3[:, A] = ok
m['valid_anchor3'] = v3
np.savez(f'{a.data}/meta.npz', **m)
print(f'valid_anchor  (1 блок):  доля на якоре 378 = {m["valid_anchor"][:,378].mean():.4f}')
print(f'valid_anchor3 (3 блока): доля на якоре 378 = {v3[:,378].mean():.4f}')
for A in (258, 318, 348, 378, 408):
    print(f'  якорь {A}: 1 блок {m["valid_anchor"][:,A].mean():.4f}  3 блока {v3[:,A].mean():.4f}')
