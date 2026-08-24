"""Самопроверка tfm4 на синтетике. Ничего не требует, кроме torch/numpy.

Проверяем ровно то, что дорого проверять на GPU-машине:
  1. правила отбрасывания колонок срабатывают на подложенных патологиях,
     в том числе на «жива на обучении, мертва на валидации» (семейство ya_wide);
  2. нулевая инициализация: tfm4 со свежей табличной веткой даёт РОВНО те же
     числа, что голый трансформер, до последнего бита;
  3. --tab-off не создаёт ни одного лишнего параметра (контроль равноёмкий);
  4. обучение идёт, обе метрики (с таблицей и без) считаются, выгрузка пишется;
  5. NaN и выбросы в таблице не превращаются в NaN на выходе сети.
"""
from __future__ import annotations

import json, os, shutil, sys, tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch


# Windows: при перенаправлении вывода в файл Python берёт кодировку локали
# (cp866/cp1251), и первая же кириллица роняет скрипт UnicodeEncodeError.
# Тот же приём, что в run_all.py.
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

N_U, N_D, N_CH = 1200, 450, 6
FAIL: list[str] = []
try:
    import polars
    HAVE_POLARS = polars is not None
except ImportError:
    HAVE_POLARS = False


def check(cond, msg):
    print(('  ок   ' if cond else '  ПЛОХО ') + msg, flush=True)
    if not cond: FAIL.append(msg)


def make_tensor(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    seq = np.abs(rng.normal(0, 1, (N_U, N_D, N_CH))).astype(np.float16)
    gmv = (np.abs(rng.normal(0, 300, (N_U, N_D))) * (rng.random((N_U, N_D)) < .05)).astype(np.float32)
    orr = (rng.random((N_U, N_D)) < .05).astype(np.float16)
    seq.tofile(root / 'seq.f16'); gmv.tofile(root / 'gmv.f32'); orr.tofile(root / 'ord.f16')
    d = np.arange(N_D)
    cal = np.stack([np.sin(2*np.pi*d/7), np.cos(2*np.pi*d/7), np.sin(2*np.pi*d/365),
                    d / N_D], 1).astype(np.float32)
    valid = np.ones((N_U, N_D), bool)
    np.savez(root / 'meta.npz', n_users=N_U, n_days=N_D, n_ch=N_CH,
             user_ids=np.arange(N_U, dtype=np.int64), valid_anchor=valid,
             valid_anchor3=valid, calendar=cal)


def make_npz(root: Path, dates: list[str], n_train: int):
    """Синтетика с теми же патологиями, что в настоящей выгрузке."""
    root.mkdir(parents=True, exist_ok=True)
    C = 40
    cols = [f'f{i}' for i in range(C)]
    cols[10] = 'gmv_sum_ya_tgt'      # мертва на раннем обучении, жива на val/test
    cols[11] = 'gmv_sum_ya_wide'     # мертва на обучении И на val, жива на тесте
    cols[12] = 'history_days'        # почти константа внутри якоря, ползёт между
    cols[13] = 'tenure'              # тоже ползёт, но с настоящей разницей по юзерам
    cols[14] = 'dead_const'          # константа везде
    cols[15] = 'live_train_dead_val'  # жива на обучении, мертва на валидации
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, (N_U, C)).astype(np.float32)
    for i, d in enumerate(dates):
        f = (base + rng.normal(0, .1, (N_U, C))).astype(np.float32)
        f[:, 20] = np.where(rng.random(N_U) < (.5 if i < n_train // 2 else .2), np.nan, f[:, 20])
        f[:, 10] = np.nan if i < n_train - 4 else f[:, 10]
        f[:, 11] = np.nan if i < len(dates) - 1 else f[:, 11]
        f[:, 12] = 5.0 + i * 0.05 + rng.normal(0, .002, N_U)
        f[:, 13] = 5.0 + i * 0.05 + rng.normal(0, .5, N_U)
        f[:, 14] = 3.0
        if d == dates[-2]: f[:, 15] = np.nan
        np.savez_compressed(root / f'tabf16_{d}.npz', feats=f.astype(np.float16),
                            cols=np.array(cols), user_id=np.arange(N_U, dtype=np.int64),
                            anchor=d, transform='signed_log1p', tiers='synthetic')
    (root / 'tabf16_meta.json').write_text(json.dumps({
        'grid': 'wed', 'anchors': dates, 'n_train': n_train, 'val': dates[-2],
        'test': dates[-1], 'n_features': C, 'cols': cols, 'transform': 'signed_log1p',
        'degenerate_on_train_slices': ['gmv_sum_ya_tgt', 'gmv_sum_ya_wide',
                                       'history_days', 'dead_const']}, ensure_ascii=False), encoding='utf-8')
    return cols


def run() -> int:
    import tfm4
    tmp = Path(tempfile.mkdtemp(prefix='tfm4_selftest_'))
    cwd = os.getcwd(); os.chdir(tmp)
    try:
        print(f'песочница: {tmp}\n', flush=True)
        d0 = date(2025, 1, 1)
        dates = [(d0 + timedelta(days=182 + 7 * k)).isoformat()
                 for k in range(8)] + ['2026-01-14', '2026-02-13']
        make_tensor(tmp / 'tensor')
        make_npz(tmp / 'npz', dates, n_train=8)

        print('1. сборка кэша и правила отбрасывания')
        meta = tfm4.prep_tab(tmp / 'npz', tmp / 'tab_raw.f16', None)
        dropped = meta['dropped']
        check('gmv_sum_ya_wide' in dropped, 'ya_wide (мертва на валидации, жива на тесте) отброшена')
        check('live_train_dead_val' in dropped,
              'колонка, живая на обучении и мёртвая на валидации, отброшена')
        check('gmv_sum_ya_tgt' in dropped, 'ya_tgt (мертва на части якорей обучения) отброшена')
        check('history_days' in dropped, 'history_days (метка якоря) отброшена')
        check('dead_const' in dropped, 'константа отброшена')
        check('tenure' not in dropped, 'tenure сохранена — у неё есть разброс между юзерами')
        check('f20' not in dropped, 'колонка с частичными NaN сохранена')
        check(len(meta['keep_idx']) == 40 - len(dropped), 'счёт колонок сходится')
        print('   отброшено:', list(dropped))

        print('\n2. нулевая инициализация и равная ёмкость')
        from train_tcn import Transformer
        torch.manual_seed(7)
        plain = Transformer(N_CH + 3, 32, 1, 0.0, 2).eval()
        torch.manual_seed(7)
        base = Transformer(N_CH + 3, 32, 1, 0.0, 2)
        tp = tfm4.TabPack(tmp / 'tab_raw.f16', meta, uids=np.arange(N_U, dtype=np.int64))
        tp.fit_stats([tfm4.day_index(s) for s in dates[:8]], verbose=False)
        tp.to_device('cpu')
        fused = tfm4.TabFusion(base, tp.n_tab, 16).eval()
        torch.manual_seed(7)
        ctrl = tfm4.TabFusion(Transformer(N_CH + 3, 32, 1, 0.0, 2), 0, 0).eval()
        x = torch.randn(24, N_CH + 3, 365)
        u = torch.arange(24); an = torch.full((24,), tfm4.day_index(dates[-2]))
        tb = tp.to_dev(tp.gather(u, an), 'cpu')
        with torch.no_grad():
            a_, b_, c_ = plain(x)['y30'], fused(x, tb)['y30'], fused(x, None)['y30']
            d_ = ctrl(x, None)['y30']
        check(torch.equal(a_, b_), f'tfm4 с таблицей == голый трансформер, max|Δ| = '
                                   f'{(a_-b_).abs().max().item():.3e}')
        check(torch.equal(a_, c_), 'tfm4 без таблицы == голый трансформер')
        check(torch.equal(a_, d_), '--tab-off даёт те же числа')
        n_plain = sum(p.numel() for p in plain.parameters())
        n_ctrl = sum(p.numel() for p in ctrl.parameters())
        check(n_plain == n_ctrl, f'--tab-off не добавил параметров ({n_plain} == {n_ctrl})')
        check(sum(p.numel() for p in fused.parameters()) > n_plain, 'с таблицей параметров больше')
        check(torch.isfinite(b_).all(), 'выход конечен при NaN во входной таблице')

        print('\n3. таблица после нормировки')
        check(torch.isfinite(tb).all(), 'нет NaN/inf после to_dev')
        check(float(tb.abs().max()) <= 10.0 + 1e-6, 'клип по ±10 работает')
        big = tp.gather(u, an).clone(); big[:] = 1e4
        check(torch.isfinite(tp.to_dev(big, 'cpu')).all(), 'выбросы 1e4 не дают inf')

        print('\n4. градиент доходит до табличной ветки')
        fused.train()
        out = fused(x, tb)['y30'].sum(); out.backward()
        gmix = fused.mix.weight.grad
        check(gmix is not None and float(gmix.abs().sum()) > 0, 'градиент в mix ненулевой')

        print('\n5. полный прогон обучения (фаза A)')
        p = tfm4.build_parser()
        args = p.parse_args(['--data', str(tmp / 'tensor'), '--tab-cache', str(tmp / 'tab_raw.f16'),
                             '--phase', 'A', '--tag', 'st', '--channels', '32', '--layers', '1',
                             '--heads', '2', '--batch', '64', '--steps', '30', '--minutes', '0',
                             '--eval-every', '15', '--tab-dim', '16', '--workers', '1',
                             '--no-plots', '--val-users', '300', '--no-val-all']
                            + (['--export', str(tmp / 'preds')] if HAVE_POLARS else []))
        args.ckpt = str(tmp / 'ckpt_st.pt')
        r = tfm4.main(args)
        check(np.isfinite(r), f'фаза A досчиталась, val RMSLE {r:.4f}')
        if HAVE_POLARS:
            check((tmp / 'preds' / 'st_val.parquet').exists(), 'выгрузка val записана')
            check((tmp / 'preds' / 'st_tabless_val.parquet').exists(),
                  'ствол без таблицы выгружен отдельным файлом')
            import polars as _pl
            _a = _pl.read_parquet(tmp / 'preds' / 'st_val.parquet')['pred'].to_numpy()
            _b = _pl.read_parquet(tmp / 'preds' / 'st_tabless_val.parquet')['pred'].to_numpy()
            check(not np.allclose(_a, _b), 'выгрузки совместной модели и ствола различаются')
        else:
            print('  ~     polars нет: проверка выгрузки parquet пропущена')
        _r = json.loads((tmp / 'result_st.json').read_text(encoding='utf-8'))
        check('cal_tabless' in _r, 'усадка ствола записана в result json')
        check((tmp / 'val_logpred_st_tabless.npy').exists(), 'сохранён прогноз без таблицы')

        print('\n6. контроль равной ёмкости (--tab-off) на той же сетке')
        args2 = p.parse_args(['--data', str(tmp / 'tensor'), '--tab-cache', str(tmp / 'tab_raw.f16'),
                              '--phase', 'A', '--tag', 'stoff', '--channels', '32', '--layers', '1',
                              '--heads', '2', '--batch', '64', '--steps', '30', '--minutes', '0',
                              '--eval-every', '15', '--workers', '1', '--no-plots', '--tab-off',
                              '--val-users', '300', '--no-val-all'])
        args2.ckpt = str(tmp / 'ckpt_stoff.pt')
        r2 = tfm4.main(args2)
        check(np.isfinite(r2), f'контроль досчитался, val RMSLE {r2:.4f}')

        print('\n7. тёплый старт из чекпоинта голого трансформера')
        ck = tmp / 'model_plain.pt'
        torch.save(plain.state_dict(), ck)
        args3 = p.parse_args(['--data', str(tmp / 'tensor'), '--tab-cache', str(tmp / 'tab_raw.f16'),
                              '--phase', 'A', '--tag', 'stw', '--channels', '32', '--layers', '1',
                              '--heads', '2', '--batch', '64', '--steps', '0', '--minutes', '0',
                              '--eval-every', '1000', '--tab-dim', '16', '--workers', '1',
                              '--no-plots', '--init-from', str(ck), '--check-init-only',
                              '--val-users', '100', '--no-val-all'])
        args3.ckpt = str(tmp / 'model_stw.pt')
        try:
            tfm4.main(args3); check(True, 'тёплый старт принят, нулевая инициализация выдержала')
        except SystemExit as e:
            check(False, f'тёплый старт сорвался: {e}')

        print('\n7b. чекпоинт от другой архитектуры должен быть отвергнут')
        torch.save(Transformer(N_CH + 3, 16, 1, 0.0, 2).state_dict(), tmp / 'model_wrong.pt')
        args4 = p.parse_args(['--data', str(tmp / 'tensor'), '--tab-cache', str(tmp / 'tab_raw.f16'),
                              '--phase', 'A', '--tag', 'stx', '--channels', '32', '--layers', '1',
                              '--heads', '2', '--batch', '64', '--steps', '0', '--minutes', '0',
                              '--workers', '1', '--no-plots', '--init-from', str(tmp / 'model_wrong.pt'),
                              '--check-init-only', '--val-users', '100', '--no-val-all'])
        args4.ckpt = str(tmp / 'model_stx.pt')
        try:
            tfm4.main(args4); check(False, 'чужой чекпоинт проглочен молча')
        except SystemExit:
            check(True, 'чекпоинт от другой архитектуры отвергнут')

        print('\n7c. отсев колонок по сдвигу распределения')
        tp2 = tfm4.TabPack(tmp / 'tab_raw.f16', meta, uids=np.arange(N_U, dtype=np.int64))
        tp2.fit_stats([tfm4.day_index(s) for s in dates[:8]], verbose=False)
        n0 = tp2.n_tab
        gone = tp2.drop_by_drift(0.3, {'валидация': tfm4.day_index(dates[-2])})
        check('tenure' in gone, 'ползущая tenure поймана отсевом по сдвигу')
        check(tp2.n_tab == n0 - len(gone) and len(tp2.cols) == tp2.n_tab, 'счёт после отсева сходится')
        tp2.to_device('cpu')
        tb2 = tp2.to_dev(tp2.gather(torch.arange(8), torch.full((8,), tfm4.day_index(dates[-1]))), 'cpu')
        check(tuple(tb2.shape) == (8, tp2.n_tab), 'выборка после отсева нужной ширины')

        print('\n7c2. каждый a.<флаг> в tfm4.py существует в разборщике аргументов')
        # Ловит целый класс: обращение к флагу, которого нет. На CPU такие строки
        # часто не выполняются (спрятаны за `dev == "cuda"`), и опечатка всплывает
        # только на GPU-машине, посреди прогона.
        import ast as _ast
        src = (Path(__file__).resolve().parent / 'tfm4.py').read_text(encoding='utf-8')
        used = {n.attr for n in _ast.walk(_ast.parse(src))
                if isinstance(n, _ast.Attribute) and isinstance(n.value, _ast.Name)
                and n.value.id == 'a'}
        known = set(vars(tfm4.build_parser().parse_args([])))
        gap = sorted(used - known)
        check(not gap, f'нет несуществующих флагов (лишние: {gap})' if gap
              else f'все {len(used)} обращений a.<флаг> покрыты разборщиком')

        print('\n7d. регрессии на баги, найденные ревью')
        # --steps 0 с --check-init-only должен выходить и при --tab-off
        for tail in (['--tab-off'], ['--no-check-init']):
            ar = p.parse_args(['--data', str(tmp / 'tensor'), '--tab-cache', str(tmp / 'tab_raw.f16'),
                               '--phase', 'A', '--tag', 'stz', '--channels', '32', '--layers', '1',
                               '--heads', '2', '--batch', '64', '--steps', '0', '--minutes', '0',
                               '--workers', '1', '--no-plots', '--check-init-only',
                               '--val-users', '50', '--no-val-all'] + tail)
            ar.ckpt = str(tmp / 'model_stz.pt')
            try:
                tfm4.main(ar); check(True, f'--check-init-only --steps 0 {tail[0]}: вышел чисто')
            except Exception as e:
                check(False, f'--check-init-only --steps 0 {tail[0]}: {type(e).__name__}: {e}')
        # фаза B с таблицей, но без усадки ствола
        at = p.parse_args(['--data', str(tmp / 'tensor'), '--tab-cache', str(tmp / 'tab_raw.f16'),
                           '--phase', 'B', '--tag', 'stt', '--fixed-steps', '10',
                           '--cal-fixed', '0.93', '--channels', '32', '--layers', '1',
                           '--heads', '2', '--batch', '64', '--minutes', '0', '--workers', '1',
                           '--no-plots', '--val-users', '50', '--no-val-all', '--tab-dim', '16'])
        at.ckpt = str(tmp / 'model_stt.pt')
        try:
            tfm4.main(at); check(False, 'фаза B без --cal-fixed-tabless пропущена')
        except SystemExit:
            check(True, 'фаза B без --cal-fixed-tabless отклонена')
        # фаза B без --cal-fixed подбирала бы усадку на протекшем якоре
        ac = p.parse_args(['--data', str(tmp / 'tensor'), '--tab-cache', str(tmp / 'tab_raw.f16'),
                           '--phase', 'B', '--tag', 'stc', '--fixed-steps', '10'])
        try:
            tfm4.main(ac); check(False, 'фаза B без --cal-fixed пропущена')
        except SystemExit:
            check(True, 'фаза B без --cal-fixed отклонена')
        # чужая сетка дат не должна молча стать другой задачей
        av = p.parse_args(['--data', str(tmp / 'tensor'), '--tab-cache', str(tmp / 'tab_raw.f16'),
                           '--phase', 'A', '--tag', 'stv', '--expect-val', '999'])
        try:
            tfm4.main(av); check(False, 'несовпадение якорей проглочено')
        except SystemExit:
            check(True, 'несовпадение дат валидации отклонено')
        # переставленные колонки в одном якоре
        bad = tmp / 'npz_bad'; bad.mkdir()
        for f in (tmp / 'npz').glob('*'): shutil.copy(f, bad / f.name)
        z = dict(np.load(bad / f'tabf16_{dates[1]}.npz', allow_pickle=True))
        c = list(np.asarray(z['cols']).astype(str)); c[0], c[1] = c[1], c[0]
        z['cols'] = np.array(c); np.savez_compressed(bad / f'tabf16_{dates[1]}.npz', **z)
        try:
            tfm4.prep_tab(bad, tmp / 'bad.f16', None)
            check(False, 'переставленные колонки проглочены')
        except SystemExit:
            check(True, 'переставленный порядок колонок пойман')

        print('\n8. фаза B без --fixed-steps должна отказаться идти')
        argsb = p.parse_args(['--data', str(tmp / 'tensor'), '--tab-cache', str(tmp / 'tab_raw.f16'),
                              '--phase', 'B', '--tag', 'stb'])
        try:
            tfm4.main(argsb); check(False, 'фаза B без --fixed-steps сорвалась в прогон')
        except SystemExit:
            check(True, 'фаза B без --fixed-steps отклонена')
        print('\n9. без polars --export обязан падать СРАЗУ, до обучения')
        import subprocess
        stub = tmp / 'nopolars'; stub.mkdir()
        # содержимое строго ASCII: .py читается как UTF-8, а write_text на Windows
        # по умолчанию пишет в кодировке локали — кириллица тут даёт SyntaxError,
        # и проверка ловила бы не то падение
        (stub / 'polars.py').write_text('raise ImportError("stub")', encoding='ascii')
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / 'tfm4.py'),
             '--data', str(tmp / 'tensor'), '--tab-cache', str(tmp / 'tab_raw.f16'),
             '--phase', 'A', '--tag', 'stp', '--export', str(tmp / 'preds2'),
             '--steps', '5', '--minutes', '0', '--no-plots'],
            capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=tmp,
            env=dict(os.environ, PYTHONPATH=str(stub), PYTHONIOENCODING='utf-8'))
        out = r.stdout + r.stderr
        ok_msg = 'pip install polars' in out
        ok_early = 'n_channels' not in out and 'Traceback' not in out.split('SystemExit')[-1]
        if not (ok_msg and ok_early):
            print('   вывод дочернего процесса:\n' + '\n'.join(out.strip().splitlines()[-8:]))
        check(ok_msg, 'сказано, что ставить')
        check('users=' not in out, 'упало ДО загрузки тензора, а не после обучения')

    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)

    print('\n' + '=' * 60)
    if FAIL:
        print(f'ПРОВАЛЕНО {len(FAIL)}:')
        for f in FAIL: print('  -', f)
        return 1
    print('всё прошло')
    return 0


if __name__ == '__main__':
    raise SystemExit(run())
