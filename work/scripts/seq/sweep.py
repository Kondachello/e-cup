"""
Подбор гиперпараметров. Рассчитан на длинный прогон (по умолчанию 24 часа).

  python sweep.py --data tensor --hours 24

Две фазы:
  1. Разведка — много коротких испытаний (по умолчанию 12 минут каждое) со
     случайными конфигурациями. OneCycle сам подстраивает расписание под
     отведённое время, поэтому короткий прогон — честный, хоть и шумный,
     прокси полного.
  2. Финал — лучшие конфигурации переобучаются на полное время (55 минут),
     потому что короткие прогоны систематически недооценивают крупные модели.

Тензор грузится в память один раз на все испытания. Результаты пишутся в
момент — уже сделанное не потеряется.

Что перебирается: learning rate, доля шагов на разогрев (включая полное его
отсутствие), число слоёв и каналов, dropout, weight decay, вес вспомогательных
голов, размер батча, коэффициент EMA, самый ранний обучающий якорь.
"""
import numpy as np, torch, argparse, time, csv, os, json, copy, gc
import train_tcn as T

SPACE = {
    'lr':         ('logu', 2e-4, 3e-3),
    'pct_start':  ('cat', [0.0, 0.05, 0.15, 0.30]),
    'channels':   ('cat', [96, 128, 192, 256]),
    'layers':     ('cat', [3, 4, 6, 8]),
    'heads':      ('cat', [4, 8]),
    'dropout':    ('cat', [0.0, 0.05, 0.1, 0.2]),
    'wd':         ('logu', 1e-4, 1e-1),
    'aux':        ('cat', [0.0, 0.05, 0.1, 0.25]),
    'batch':      ('cat', [256, 384, 512]),
    'ema':        ('cat', [0.995, 0.999, 0.9995]),
    'min_anchor': ('cat', [30, 60, 120, 200]),
}

def sample(rng):
    out = {}
    for k, spec in SPACE.items():
        if spec[0] == 'logu':
            out[k] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        else:
            out[k] = spec[1][rng.integers(len(spec[1]))]
    if out['channels'] % out['heads']: out['heads'] = 4      # размерность должна делиться
    return out

def make_args(base, cfg, minutes, tag):
    a = copy.deepcopy(base)
    for k, v in cfg.items(): setattr(a, k, v)
    a.minutes = minutes; a.tag = tag; a.ckpt = f'sweep_{tag}.pt'
    a.predict = ''; a.no_plots = True; a.eval_only = False   # history_<tag>.csv пишется всегда
    a.steps = 200000
    return a

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='tensor')
    p.add_argument('--arch', default='transformer', choices=['tcn','transformer'])
    p.add_argument('--hours', type=float, default=24)
    p.add_argument('--trial-minutes', type=float, default=12)
    p.add_argument('--final-minutes', type=float, default=55)
    p.add_argument('--n-final', type=int, default=6)
    p.add_argument('--val-users', type=int, default=25000)
    p.add_argument('--eval-every', type=int, default=1500)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    s = p.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    st = T.Store(s.data, pin=(dev=='cuda'), abs_time=False); st.to_device(dev)

    base = argparse.Namespace(
        data=s.data, arch=s.arch, batch=384, channels=128, blocks=8, layers=4, heads=4,
        dropout=0.1, lr=6e-4, wd=1e-2, steps=200000, minutes=s.trial_minutes,
        eval_every=s.eval_every, workers=s.workers, val_users=s.val_users, ema=0.999,
        aux=0.1, calib=0.95, pct_start=0.15, min_anchor=60, abs_time=False,
        seed=s.seed, tag='x', ckpt='x.pt', predict='', eval_only=False, no_plots=True,
        es_metric='cal', val_anchor=378, export='')

    done = []
    if os.path.exists(s.out):
        with open(s.out) as f: done = list(csv.DictReader(f))
        print(f'найдено прошлых испытаний: {len(done)}')
    else:
        with open(s.out, 'w', newline='') as f:
            csv.writer(f).writerow(['phase','trial','rmsle','minutes'] + list(SPACE))

    rng = np.random.default_rng(s.seed)
    t0 = time.time(); budget = s.hours*3600
    reserve = s.n_final * s.final_minutes * 60 * 1.15      # запас на вторую фазу
    trial = len(done)

    # ---------------- фаза 1: разведка
    while time.time()-t0 < budget - reserve:
        cfg = sample(rng); trial += 1
        print(f'\n=== испытание {trial} === {json.dumps(cfg, default=str)}', flush=True)
        try:
            r = T.main(make_args(base, cfg, s.trial_minutes, f't{trial}'), st=st)
        except torch.cuda.OutOfMemoryError:
            print('не хватило видеопамяти, пропускаю'); r = float('nan')
        except Exception as e:
            print('ошибка:', e); r = float('nan')
        with open(s.out, 'a', newline='') as f:
            csv.writer(f).writerow(['explore', trial, r, s.trial_minutes] + [cfg[k] for k in SPACE])
        torch.cuda.empty_cache(); gc.collect()
        el = (time.time()-t0)/3600
        print(f'--- испытание {trial}: RMSLE {r:.4f} | прошло {el:.1f} ч из {s.hours}', flush=True)

    # ---------------- фаза 2: лучшие на полное время
    with open(s.out) as f:
        rows = [r for r in csv.DictReader(f) if r['phase']=='explore' and r['rmsle'] not in ('','nan')]
    rows.sort(key=lambda r: float(r['rmsle']))
    print(f'\n=== фаза 2: {min(s.n_final,len(rows))} лучших на {s.final_minutes} мин ===', flush=True)
    for i, row in enumerate(rows[:s.n_final]):
        cfg = {}
        for k in SPACE:
            v = row[k]
            cfg[k] = float(v) if ('.' in v or 'e' in v.lower()) else int(v)
        tag = f'final{i+1}'
        print(f'\n=== {tag} (разведка дала {float(row["rmsle"]):.4f}) === {json.dumps(cfg)}', flush=True)
        try:
            r = T.main(make_args(base, cfg, s.final_minutes, tag), st=st)
        except Exception as e:
            print('ошибка:', e); r = float('nan')
        with open(s.out, 'a', newline='') as f:
            csv.writer(f).writerow(['final', tag, r, s.final_minutes] + [cfg[k] for k in SPACE])
        torch.cuda.empty_cache(); gc.collect()

    with open(s.out) as f:
        rows = [r for r in csv.DictReader(f) if r['rmsle'] not in ('','nan')]
    rows.sort(key=lambda r: float(r['rmsle']))
    print('\n===== ИТОГ =====')
    for r in rows[:10]:
        print(f'{r["rmsle"]:>8}  {r["phase"]:8} {r["trial"]:8}  ' +
              ' '.join(f'{k}={r[k]}' for k in SPACE))
    print('\nполные результаты:', s.out)
    print('лучшую конфигурацию перенесите во флаги train_tcn.py и обучите с --predict')

if __name__ == '__main__':
    main()
