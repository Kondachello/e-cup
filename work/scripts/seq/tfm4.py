"""tfm4 — совместное слияние трансформера с табличными признаками.

Схема из брифа координатора, буквально:

    tab_emb = Linear(n_tab -> d)                 обычная инициализация
    pool'   = concat([pool, tab_emb])            перед головами
    z       = mlp(pool')                         = mlp_pool(pool) + mdl_onyx(tab_emb)

равна tfm3b: проверяется флагом --check-init, печатает max|Δ| и падает, если он
не ровно 0.

Отличие от tfm3b, которое надо помнить: якоря обучения берутся не из всех дней
подряд, а только из сетки, для которой выгружена таблица (24 среды). Это само по
себе меняет распределение обучения, поэтому прогон с --tab-off на той же сетке
обязателен — иначе прирост tfm4 неотличим от эффекта смены сетки.

Фаза A: обучение на 24 средах (последняя 2025-12-10 = день 343), валидация 378.
        Зазор 35 дней — больше штатных 30, то есть оценка консервативная.
Фаза B: те же 24 среды + якорь 378, предсказание на 408. Зазор ровно 30.
        Валидация при этом подсматривает, поэтому фаза B обязана идти с
        --fixed-steps и --cal-fixed из фазы A, как и в train_tcn.py.

  python tfm4.py --prep-tab --tab-npz from_gpu/kaggle_tabfeats_wed_v1
  python tfm4.py --selftest
  python tfm4.py --phase A --init-from ckpt_tfm3b_a_s1.pt --tag tfm4_a_s1 ...
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time, warnings
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Windows: при перенаправлении вывода в файл Python берёт кодировку локали
# (cp866/cp1251), и первая же кириллица роняет скрипт UnicodeEncodeError.
# Тот же приём, что в run_all.py.
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_tcn import (Store, Transformer, TCN, EMA, HEADS, HORIZON,
                       cal_rmsle, rmsle_from_log, save_history, make_plots)

DAY0 = date(2025, 1, 1)          # день 0 в индексации проекта
VAL_ANCHOR_DEF, TEST_ANCHOR_DEF = 378, 408


def day_index(s: str) -> int:
    y, m, d = (int(v) for v in s.split('-'))
    return (date(y, m, d) - DAY0).days


# ------------------------------------------------------------------ таблица
class TabPack:
    """Сырые табличные признаки на сетке якорей + статистики обучающего среза.

    В кэш кладём СЫРУЮ матрицу (NaN сохранены), а нормировку применяем на лету.
    Причина: у фаз A и B разные обучающие срезы, значит разные mu/sd, а держать
    две копии по 2.3 ГБ незачем.
    """

    def __init__(self, cache: Path, meta: dict, uids: np.ndarray | None = None, ram=False):
        self.meta = meta
        self.all_cols = meta['all_cols']
        self.keep_idx = np.asarray(meta['keep_idx'], np.int64)
        self.cols = [self.all_cols[i] for i in self.keep_idx]
        self.n_all = len(self.all_cols)
        self.n_tab = len(self.cols)
        self.anchors = [day_index(s) for s in meta['anchor_dates']]
        self.slot = {d: i for i, d in enumerate(self.anchors)}
        self.n_u = meta['n_users']
        self.mat = np.memmap(cache, np.float16, 'r',
                             shape=(len(self.anchors), self.n_u, self.n_all))
        if ram:
            print('  поднимаю таблицу в RAM...', flush=True)
            self.mat = np.asarray(self.mat)
        if uids is not None:
            got = hashlib.sha1(np.asarray(uids).astype(np.int64).tobytes()).hexdigest()[:12]
            if got != meta['uids_sha1']:
                raise SystemExit(
                    f'user_id тензора не совпадают с таблицей ({got} против '
                    f"{meta['uids_sha1']}): выравнивание строк нарушено, всё дальнейшее — мусор")
        self.mu = self.sd = None
        self.mu_d = self.sd_d = None

    # --- статистики обучающего среза -------------------------------------
    def fit_stats(self, train_days: list[int], verbose=True):
        n = np.zeros(self.n_tab, np.float64)
        s1 = np.zeros(self.n_tab, np.float64)
        s2 = np.zeros(self.n_tab, np.float64)
        for d in train_days:
            F32 = np.asarray(self.mat[self.slot[d]], np.float32)[:, self.keep_idx]
            m = ~np.isnan(F32)
            n += m.sum(0)
            s1 += np.where(m, F32, 0).sum(0, dtype=np.float64)
            s2 += np.where(m, F32.astype(np.float64) ** 2, 0).sum(0)
        n = np.maximum(n, 1)
        mu = s1 / n
        sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0))
        dead = sd < 1e-6
        if dead.any() and verbose:
            print(f'  вырожденных на обучающем срезе (sd<1e-6): {int(dead.sum())} — '
                  f'{[self.cols[i] for i in np.where(dead)[0]][:8]}', flush=True)
        self.mu, self.sd = mu.astype(np.float32), np.maximum(sd, 1e-3).astype(np.float32)
        self.dead = dead
        if verbose:
            print(f'  статистики посчитаны по {len(train_days)} якорям обучения, '
                  f'{self.n_tab} колонок', flush=True)
        return {'mu': self.mu.tolist(), 'sd': self.sd.tolist(), 'cols': self.cols,
                'train_days': train_days}

    def load_stats(self, st: dict):
        if st['cols'] != self.cols:
            raise SystemExit('состав колонок в статистиках не совпадает с текущим')
        self.mu = np.asarray(st['mu'], np.float32)
        self.sd = np.asarray(st['sd'], np.float32)
        self.dead = self.sd < 1e-3 + 1e-9

    def to_device(self, dev):
        self.mu_d = torch.from_numpy(self.mu).to(dev)
        self.sd_d = torch.from_numpy(self.sd).to(dev)

    # --- отчёт о сдвиге распределения ------------------------------------
    def drift_report(self, days: dict[str, int], top=12):
        """Насколько среднее на якоре отъехало от обучающего в единицах sd.

        Это не украшение: колонка, у которой val/test лежат в двух sd от
        обучающего среза, — та самая, на которой линейная ветка экстраполирует.
        """
        print('\nсдвиг распределения относительно обучающего среза (в sd):', flush=True)
        for name, d in days.items():
            F32 = np.asarray(self.mat[self.slot[d]], np.float32)[:, self.keep_idx]
            with np.errstate(all='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mu = np.nanmean(F32, 0)
            z = np.abs(mu - self.mu) / self.sd
            z[self.dead] = 0
            order = np.argsort(-z)[:top]
            nanr = np.isnan(F32).mean()
            print(f'  {name} (день {d}): NaN {nanr*100:.2f}%, худшие колонки:', flush=True)
            for i in order[:6]:
                print(f'      {self.cols[i]:26s} |Δmu|/sd = {z[i]:6.2f}', flush=True)
            print(f'      колонок с |Δmu|/sd > 1: {int((z > 1).sum())}, > 2: {int((z > 2).sum())}',
                  flush=True)

    def drift(self, day: int) -> np.ndarray:
        F32 = np.asarray(self.mat[self.slot[day]], np.float32)[:, self.keep_idx]
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mu = np.nanmean(F32, 0)
        z = np.abs(mu - self.mu) / self.sd
        z[self.dead] = 0
        return np.nan_to_num(z, nan=0.0, posinf=0.0)

    def drop_by_drift(self, z_max: float, days: dict[str, int]):
        """Убрать колонки, у которых среднее на валидации или тесте отъехало от
        обучающего среза больше чем на z_max сигм. mu/sd считаются поколоночно,
        поэтому пересчитывать их после отсева не нужно — достаточно подрезать."""
        z = np.zeros(self.n_tab)
        for d in days.values():
            z = np.maximum(z, self.drift(d))
        bad = z > z_max
        if not bad.any():
            print(f'  отсев по сдвигу (> {z_max} sd): ничего не отброшено', flush=True)
            return []
        names = [self.cols[i] for i in np.where(bad)[0]]
        print(f'  отсев по сдвигу (> {z_max} sd): убрано {len(names)} колонок', flush=True)
        for i in np.where(bad)[0]:
            print(f'      - {self.cols[i]:26s} |Δmu|/sd = {z[i]:.2f}', flush=True)
        good = ~bad
        self.keep_idx = self.keep_idx[good]
        self.cols = [c for c, k in zip(self.cols, good) if k]
        self.n_tab = len(self.cols)
        self.mu, self.sd, self.dead = self.mu[good], self.sd[good], self.dead[good]
        return names

    # --- выборка ----------------------------------------------------------
    def gather(self, users: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        rows = np.fromiter((self.slot[int(a)] for a in anchors), np.int64, len(anchors))
        u = users.numpy().astype(np.int64)
        return torch.from_numpy(np.asarray(self.mat[rows, u][:, self.keep_idx], np.float16))

    def to_dev(self, t: torch.Tensor, dev, clamp=10.0) -> torch.Tensor:
        x = t.to(dev, non_blocking=True).float()
        x = (x - self.mu_d) / self.sd_d
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x.clamp_(-clamp, clamp)


# ------------------------------------------------------- сборка кэша таблицы
def prep_tab(npz_dir: Path, cache: Path, uids: np.ndarray | None,
             keep_degenerate=False, keep_all=False) -> dict:
    src = json.loads((npz_dir / 'tabf16_meta.json').read_text(encoding='utf-8'))
    cols = src['cols']; C = len(cols)
    dates = src['anchors']; A = len(dates)
    tr_dates = set(dates[:src['n_train']])
    print(f'готовлю кэш: {A} якорей x {C} колонок', flush=True)

    n_u = None
    NF = np.zeros((A, C)); SD = np.zeros((A, C)); MU = np.zeros((A, C))
    cache.parent.mkdir(parents=True, exist_ok=True)
    mm = None
    for i, d in enumerate(dates):
        z = np.load(npz_dir / f'tabf16_{d}.npz')
        f = z['feats']; u = z['user_id']
        if n_u is None:
            n_u = f.shape[0]
            uids_ref = np.asarray(u).astype(np.int64)
            mm = np.memmap(cache, np.float16, 'w+', shape=(A, n_u, C))
        elif not np.array_equal(np.asarray(u).astype(np.int64), uids_ref):
            raise SystemExit(f'{d}: другой порядок user_id — выравнивание строк нарушено')
        if f.shape != (n_u, C):
            raise SystemExit(f'{d}: форма {f.shape}, ожидалась {(n_u, C)}')
        if 'cols' in z and list(np.asarray(z['cols']).astype(str)) != cols:
            raise SystemExit(f'{d}: порядок колонок отличается от tabf16_meta.json — '
                             f'сверка строк такое не ловит, а статистики отравит')
        mm[i] = f
        g = np.asarray(f, np.float32)
        NF[i] = np.isnan(g).mean(0)
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')      # колонка целиком из NaN — это ожидаемо
            SD[i] = np.nanstd(g, 0); MU[i] = np.nanmean(g, 0)
        print(f'  [{i+1:2d}/{A}] {d}  NaN {NF[i].mean()*100:5.2f}%', flush=True)
    mm.flush(); del mm

    if uids is not None and not np.array_equal(np.asarray(uids).astype(np.int64), uids_ref):
        raise SystemExit('user_id таблицы не совпадают с user_id тензора')

    tr = np.array([d in tr_dates for d in dates])
    drop: dict[str, str] = {}
    deg = set(src.get('degenerate_on_train_slices', []))
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sd_max_in = np.nanmax(SD, 0)                  # максимум по якорям от sd ВНУТРИ якоря
        mu_rng = np.nanmax(MU, 0) - np.nanmin(MU, 0)  # разброс средних МЕЖДУ якорями
    for j, c in enumerate(cols):
        if (NF[tr, j] > 0.999).all():
            drop[c] = 'весь обучающий срез — NaN'
        elif (NF[~tr, j] > 0.999).any():
            drop[c] = 'полностью NaN на валидации или тесте, а на обучении жива'
        elif (NF[tr, j] > 0.999).any():
            drop[c] = f'NaN на {int((NF[tr, j] > 0.999).sum())} из {int(tr.sum())} якорей обучения'
        elif np.nanmax(SD[tr, j]) < 1e-6:
            drop[c] = 'константа на всём обучающем срезе'
        elif np.isfinite(sd_max_in[j]) and sd_max_in[j] * 5 < mu_rng[j]:
            drop[c] = (f'почти константа внутри якоря (sd<={sd_max_in[j]:.4g}), '
                       f'но среднее ездит на {mu_rng[j]:.4g} между якорями — метка якоря')
        elif c in deg and not keep_degenerate:
            drop[c] = 'помечена degenerate_on_train_slices в выгрузке'
    if keep_all:
        drop = {}
    keep_idx = [j for j, c in enumerate(cols) if c not in drop]

    print(f'\nотброшено {len(drop)} колонок, остаётся {len(keep_idx)}:', flush=True)
    for c, why in drop.items():
        print(f'  - {c:26s} {why}', flush=True)
    only_auto = [c for c in drop if c not in deg]
    if only_auto:
        print(f'\n  из них НЕ помеченных выгрузкой: {only_auto}', flush=True)

    meta = {'all_cols': cols, 'keep_idx': keep_idx, 'anchor_dates': dates,
            'n_users': int(n_u), 'n_train': int(src['n_train']),
            'val_date': src['val'], 'test_date': src['test'],
            'uids_sha1': hashlib.sha1(uids_ref.tobytes()).hexdigest()[:12],
            'cols_sha1': src.get('cols_sha1', ''),
            'dropped': drop,
            'nan_frac': NF.tolist()}
    cache.with_suffix('.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
    print(f'\nкэш: {cache} ({cache.stat().st_size/2**30:.2f} ГБ), описание рядом в .json', flush=True)
    return meta


# ------------------------------------------------------------------ модель
class TabFusion(nn.Module):
    """Трансформер + линейная табличная ветка, вливающаяся в пулинг.

    concat([pool, tab_emb]) -> Linear  тождественно равно
    mlp0(pool) + mdl_onyx(tab_emb), и второй член мы обнуляем на старте. Поэтому
    свежая tfm4 с весами tfm3b даёт РОВНО те же числа, что tfm3b.
    """

    def __init__(self, base: nn.Module, n_tab: int, d: int = 128, gelu=False):
        super().__init__()
        self.base = base
        if n_tab > 0:
            self.emb = nn.Linear(n_tab, d)
            self.act = nn.GELU() if gelu else nn.Identity()
            self.mix = nn.Linear(d, base.neck.mlp[0].out_features, bias=False)
            nn.init.zeros_(self.mix.weight)
        else:
            # --tab-off: ни одного лишнего параметра, иначе контроль не равноёмкий
            self.emb = self.mix = None
            self.act = nn.Identity()

    def pool(self, x):
        h = self.base.encode(x) if hasattr(self.base, 'encode') else self._tcn_encode(x)
        return torch.cat([h[:, :, -1], h.mean(-1), h.amax(-1)], 1)

    def _tcn_encode(self, x):
        h = self.base.stem(x)
        for b in self.base.blocks: h = b(h)
        return h

    def _heads(self, z):
        nk = self.base.neck
        for m in nk.mlp[1:]:
            z = m(z)
        return {k: nk.out[k](z).squeeze(-1) for k in HEADS}

    def forward(self, x, tab=None):
        z = self.base.neck.mlp[0](self.pool(x))
        if tab is not None and self.emb is not None:
            z = z + self.mix(self.act(self.emb(tab)))
        return self._heads(z)

    def forward_both(self, x, tab):
        """Оба ответа — с таблицей и без — за ОДИН проход ствола.

        Ствол здесь самый дорогой, а отличаются ветки только слагаемым в
        пулинге. Два отдельных вызова forward удваивали бы стоимость оценки,
        а она идёт по всем 250000 юзеров."""
        z0 = self.base.neck.mlp[0](self.pool(x))
        if tab is None or self.emb is None:
            o = self._heads(z0)
            return o, o
        return self._heads(z0 + self.mix(self.act(self.emb(tab)))), self._heads(z0)

    def tab_params(self):
        if self.emb is None: return []
        return list(self.emb.parameters()) + list(self.mix.parameters())


# --------------------------------------------------------------- prefetcher
class GridPrefetcher:
    """Как Prefetcher из train_tcn.py, но якорь берётся только из сетки, для
    которой есть таблица, и вместе с батчем едут табличные строки."""

    def __init__(self, store: Store, tab: TabPack | None, grid: list[int],
                 batch: int, depth=6, workers=2):
        import threading, queue
        self.q = queue.Queue(maxsize=depth)
        self.st, self.tab, self.b = store, tab, batch
        self.grid = torch.tensor(sorted(grid))
        self.stop = False
        self.th = [threading.Thread(target=self._run, daemon=True) for _ in range(workers)]
        for t in self.th: t.start()

    def _run(self):
        st, B, G = self.st, self.b, self.grid
        while not self.stop:
            u = torch.randint(0, st.n_u, (B,))
            an = G[torch.randint(0, len(G), (B,))]
            for _ in range(4):
                bad = ~st.valid_tr[u, an]
                if not bad.any(): break
                an[bad] = G[torch.randint(0, len(G), (int(bad.sum()),))]
            keep = st.valid_tr[u, an]
            u, an = u[keep], an[keep]
            out = st.cpu_batch(u, an)
            if self.tab is not None:
                out['tab'] = self.tab.gather(u, an)
            self.q.put(out)

    def get(self):
        return self.q.get()


# ------------------------------------------------------------------ обучение
def _eval(model, st, tab, users, anchor, dev, batch):
    """Возвращает (с таблицей, без таблицы, таргеты) за один проход.
    Второй элемент — это score_tabzero из брифа; при --tab-off он равен первому."""
    ps, ps0, ts = [], [], []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(dev == 'cuda')):
        for i in range(0, len(users), batch):
            u = users[i:i + batch]
            anc = torch.full((len(u),), anchor)
            xb, yb = st.batch(u, anc, dev)
            tb = tab.to_dev(tab.gather(u, anc), dev) if tab is not None else None
            o, o0 = model.forward_both(xb, tb)
            ps.append(o['y30'].float().cpu())
            ps0.append(o0['y30'].float().cpu())
            ts.append(yb['y30'].float().cpu())
    return torch.cat(ps), torch.cat(ps0), torch.cat(ts)


def main(a):
    if not a.ckpt: a.ckpt = f'model_{a.tag}.pt'
    if a.phase == 'B' and not a.tab_off and not a.cal_fixed_tabless:
        raise SystemExit('в фазе B усадку ствола без таблицы подбирать не на чем (валидация '
                         'входит в обучение): нужен --cal-fixed-tabless C из фазы A, поле '
                         'cal_tabless в result_tfm4_a_s<сид>.json. Проверяю до обучения.')
    if a.phase == 'B' and not (a.fixed_steps and a.cal_fixed):
        raise SystemExit('фаза B обучается на якоре 378, а валидация тоже на 378: метрика '
                         'подсматривает. Обязательны ОБА флага из фазы A — --fixed-steps N и '
                         '--cal-fixed C. Без --cal-fixed усадка подберётся по запомненному '
                         'якорю и умножит все 250000 тестовых предсказаний.')
    if a.export:
        # Проверка на пять секунд вместо падения на последней строке после часа
        # обучения: parquet пишется polars, и без него --export бесполезен.
        try:
            import polars  # noqa: F401
        except ImportError:
            raise SystemExit('для --export нужен polars, а его нет: pip install polars\n'
                             'Проверяю до обучения, чтобы не потерять час и упасть на '
                             'последней строке. Без выгрузки можно считать так: убрать '
                             '--export, тогда прогноз останется в val_logpred_<тег>.npy.')
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    if dev == 'cuda':
        free, total = torch.cuda.mem_get_info()
        print(f'карта: {torch.cuda.get_device_name(0)}, свободно '
              f'{free/2**30:.2f} из {total/2**30:.2f} ГБ', flush=True)
        if free < 3.0 * 2**30:
            print('  ВНИМАНИЕ: меньше 3 ГБ свободно на карте — проверь nvidia-smi.', flush=True)
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f'память хоста: свободно {vm.available/2**30:.1f} из {vm.total/2**30:.1f} ГБ'
              + ('  <- МАЛО. Закрепление тензора (~2.9 ГБ) может не пройти, пробуй --no-pin'
                 if vm.available < 6 * 2**30 else ''), flush=True)
    except ImportError:
        pass
    st = Store(a.data, pin=(dev == 'cuda' and not a.no_pin),
               abs_time=a.abs_time, cohort3=not a.cohort1)
    st.to_device(dev)

    tab = None
    if not a.tab_off:
        cache = Path(a.tab_cache)
        meta = json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))
        tab = TabPack(cache, meta, uids=st.uids, ram=a.tab_ram)
        print(f'таблица: {tab.n_tab} колонок из {tab.n_all}, {len(tab.anchors)} якорей', flush=True)

    # --- сетка якорей -----------------------------------------------------
    src_meta = json.loads(Path(a.tab_cache).with_suffix('.json').read_text(encoding='utf-8'))
    all_days = [day_index(s) for s in src_meta['anchor_dates']]
    n_tr = src_meta['n_train']
    grid = [d for d in all_days[:n_tr] if d >= a.min_anchor_day]
    if len(grid) < n_tr:
        print(f'--min-anchor-day {a.min_anchor_day}: из {n_tr} якорей обучения оставлено '
              f'{len(grid)}', flush=True)
    if not grid:
        raise SystemExit('--min-anchor-day отсёк все якоря обучения')
    VAL = day_index(src_meta['val_date']); TEST = day_index(src_meta['test_date'])
    if (VAL, TEST) != (a.expect_val, a.expect_test):
        raise SystemExit(f'в описании таблицы валидация {VAL} и тест {TEST}, а задача — '
                         f'{a.expect_val} и {a.expect_test}. Либо это другая выгрузка, либо '
                         f'сместились даты; проверь, прежде чем считать.')
    if a.phase == 'B':
        grid = grid + [VAL]
    gap = VAL - max(grid)
    print(f'фаза {a.phase}: якорей обучения {len(grid)} ({min(grid)}..{max(grid)}), '
          f'валидация {VAL}, тест {TEST}, зазор {gap} дней', flush=True)
    if a.phase == 'A' and gap < HORIZON:
        raise SystemExit(f'зазор {gap} < {HORIZON}: таргет обучения залезает в валидацию')

    if tab is not None:
        tab.fit_stats(grid)
        if a.tab_drop_drift > 0:
            days = {'валидация': VAL} | ({'тест': TEST} if a.tab_drift_use_test else {})
            tab.drop_by_drift(a.tab_drop_drift, days)
        Path(f'tabstats_{a.tag}.json').write_text(json.dumps(
            {'mu': tab.mu.tolist(), 'sd': tab.sd.tolist(), 'cols': tab.cols,
             'train_days': grid}, ensure_ascii=False), encoding='utf-8')
        tab.to_device(dev)
        if not a.no_drift:
            tab.drift_report({'валидация': VAL, 'тест': TEST})

    # --- модель -----------------------------------------------------------
    base = (Transformer(st.n_in, a.channels, a.layers, a.dropout, a.heads)
            if a.arch == 'transformer' else TCN(st.n_in, a.channels, a.blocks, a.dropout))
    model = TabFusion(base, tab.n_tab if tab is not None else 0, a.tab_dim, a.tab_gelu)
    model = model.to(dev)
    if a.init_from:
        sd = torch.load(a.init_from, map_location=dev)
        # чекпоинт tfm3b лежит без префикса base. — принимаем оба написания
        sd = {(k if k.startswith(('base.', 'emb.', 'mix.')) else 'base.' + k): v
              for k, v in sd.items()}
        own = model.state_dict()
        take = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(take, strict=False)
        bad_shape = [k for k in sd if k in own and own[k].shape != sd[k].shape]
        if bad_shape:
            raise SystemExit(f'формы не совпали у {bad_shape}: чекпоинт от другого состава '
                             f'колонок таблицы. Тёплый старт был бы частичным и незаметным.')
        miss = [k for k in sd if k not in take]
        print(f'тёплый старт из {a.init_from}: перенесено {len(take)} тензоров'
              + (f', НЕ перенесено {len(miss)}: {miss[:6]}' if miss else ''), flush=True)
        print('  табличная ветка: '
              + ('перенесена из чекпоинта' if any(k.startswith(('emb.', 'mix.')) for k in take)
                 else 'инициализирована заново (нулевая) — так и задумано при старте из tfm3b'),
              flush=True)
        want = len([k for k in own if k.startswith('base.')])
        if len(take) < want:
            raise SystemExit(f'из чекпоинта перенеслось {len(take)} тензоров, а ствол требует '
                             f'{want}: архитектуры не совпали, тёплый старт был бы фиктивным')

    print('параметров:', sum(p.numel() for p in model.parameters()) / 1e6, 'млн', flush=True)
    if tab is not None:
        print('  из них табличная ветка:',
              sum(p.numel() for p in model.tab_params()) / 1e6, 'млн', flush=True)

    # --- проверка нулевой инициализации ----------------------------------
    if a.check_init and tab is not None:
        model.eval()
        u = torch.arange(min(512, st.n_u))
        xb, _ = st.batch(u, torch.full((len(u),), VAL), dev)
        tb = tab.to_dev(tab.gather(u, torch.full((len(u),), VAL)), dev)
        with torch.no_grad():
            d = (model(xb, tb)['y30'] - model(xb, None)['y30']).abs().max().item()
        print(f'проверка нулевой инициализации: max|Δ| = {d:.3e}', flush=True)
        if d != 0.0:
            raise SystemExit('табличная ветка на старте НЕ нулевая — сравнение с tfm3b нечестно')
        if float(model.mix.weight.detach().abs().max()) != 0.0:
            raise SystemExit('веса mix не нулевые')
    if a.check_init_only:
        # ранний выход обязан работать и при --tab-off, и при --no-check-init,
        # иначе стадия check падает на OneCycleLR(total_steps=0)
        print('проверка завершена, обучение не запускалось', flush=True)
        return 0.0

    # --- валидационная выборка -------------------------------------------
    val_u = (torch.arange(st.n_u) if a.val_all
             else torch.arange(st.n_u)[st.valid[:, VAL]])
    if a.val_all and a.val_users:
        print(f'--val-all: подвыборка {a.val_users} отключена', flush=True)
        a.val_users = 0
    if a.val_users and a.val_users < len(val_u):
        g = torch.Generator().manual_seed(42)
        val_u = val_u[torch.randperm(len(val_u), generator=g)[:a.val_users]]
    np.save(f'val_user_ids_{a.tag}.npy', st.uids[val_u.numpy()])
    print(f'валидация: {len(val_u)} юзеров', flush=True)

    g_base = [p for n, p in model.named_parameters() if n.startswith('base.')]
    g_tab = [p for n, p in model.named_parameters() if not n.startswith('base.')]
    groups = [{'params': g_base, 'lr': a.lr}]
    max_lr = [a.lr]
    if g_tab:
        groups.append({'params': g_tab, 'lr': a.lr * a.tab_lr_mult})
        max_lr.append(a.lr * a.tab_lr_mult)
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.wd)
    scaler = torch.amp.GradScaler(enabled=(dev == 'cuda'))
    W = {'y30': 1.0, 'y7': a.aux, 'y14': a.aux, 'ord30': a.aux, 'act30': a.aux, 'buy': a.aux}
    if a.fixed_steps:
        a.steps, a.minutes = a.fixed_steps, 0.0
        print(f'режим переобучения: ровно {a.fixed_steps} шагов, ранней остановки нет', flush=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr, total_steps=max(a.steps, 8), pct_start=max(a.pct_start, 1e-3))
    ema = EMA(model, a.ema)
    pf = GridPrefetcher(st, tab, grid, a.batch, workers=a.workers)

    t0 = time.time(); best = (9e9, None); step = 0; retimed = False; hist = []
    base0 = None
    while step < a.steps:
        model.train()
        raw = pf.get()
        x, y = st.to_dev(raw, dev)
        tb = tab.to_dev(raw['tab'], dev) if tab is not None else None
        if tb is not None and step < a.tab_warmup:
            tb = None
        try:
            with torch.amp.autocast('cuda', enabled=(dev == 'cuda')):
                p = model(x, tb)
                loss = sum(W[k] * (F.binary_cross_entropy_with_logits(p[k], y[k]) if k == 'buy'
                                   else F.mse_loss(p[k], y[k])) for k in W)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            _sc = scaler.get_scale(); scaler.step(opt); scaler.update()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if 'out of memory' not in str(e).lower() and 'unknown error' not in str(e).lower():
                raise
            fr = torch.cuda.mem_get_info()[0] / 2**30 if dev == 'cuda' else 0
            raise SystemExit(
                f'отказ выделения на шаге {step} (свободно на карте {fr:.2f} ГБ).\n'
                f'Если свободного много, а не дали десятки мегабайт — это НЕ нехватка '
                f'видеопамяти, а отказ драйвера. На Windows это обычно память хоста: '
                f'тензор закрепляется (~2.9 ГБ), и драйверу нечем подпереть выделения карты.\n'
                f'  1. --no-pin       не закреплять тензор  <- пробовать первым\n'
                f'  2. закрыть Chrome и прочее, что ест RAM\n'
                f'  3. --workers 1    меньше закреплённых буферов батчей\n'
                f'  4. --batch 256    вдвое меньше памяти карты (тогда --fixed-steps вдвое '
                f'больше, чтобы число просмотренных примеров совпало)') from e
        if scaler.get_scale() >= _sc and sched.last_epoch < sched.total_steps - 1: sched.step()
        step += 1; ema.update(model, step)


        if step % a.eval_every == 0 or step == a.steps:
            lp, lp0, lt = _eval(ema.model, st, tab, val_u, VAL, dev, a.batch)
            r_raw = rmsle_from_log(lp, lt)
            r = cal_rmsle(lp.numpy(), lt.numpy()) if a.es_metric == 'cal' else r_raw
            extra = ''
            if tab is not None:
                r0 = rmsle_from_log(lp0, lt)
                if base0 is None: base0 = r0
                extra = f'  без таблицы {r0:.4f} (Δ {r_raw - r0:+.4f})'
                if r0 > base0 + a.collapse_tol:
                    extra += f'  <- СТВОЛ ПРОСЕЛ на {r0 - base0:.4f}, ветка перетянула'
            el = time.time() - t0
            print(f'step {step:6d}  loss {loss.item():.4f}  val RMSLE {r:.4f} '
                  f'(сырой {r_raw:.4f}){extra}  {el/60:.1f} мин', flush=True)
            hist.append((step, float(loss.item()), r, float(opt.param_groups[0]['lr']), el / 60))
            if a.fixed_steps:
                best = (r, step)
                # Пишем на каждой оценке, а не только после цикла: иначе падение
                # на предпоследнем шаге стоит всего прогона.
                torch.save(ema.model.state_dict(), a.ckpt)
            elif r < best[0]:
                best = (r, step); torch.save(ema.model.state_dict(), a.ckpt)
            # Скорость меряем ПОСЛЕ первой оценки: оценка идёт по всем 250000
            # юзеров и стоит заметную долю шага. Замер до неё (как в train_tcn)
            # завышал бы темп, и отжиг learning rate не успевал бы закончиться.
            if not retimed and a.minutes and step >= 300:
                rate = step / el
                plan = max(step + 100, min(a.steps, int(rate * a.minutes * 60 * 0.97)))
                if abs(plan - a.steps) > 500:
                    print(f'скорость {rate:.2f} шаг/с (с учётом оценок) -> '
                          f'расписание на {plan} шагов', flush=True)
                    a.steps = plan
                    sched = torch.optim.lr_scheduler.OneCycleLR(
                        opt, max_lr, total_steps=plan, pct_start=max(a.pct_start, 1e-3),
                        last_epoch=-1)
                    for _ in range(step): sched.step()
                retimed = True
            if a.minutes and el > a.minutes * 60:
                print('лимит времени'); break
    pf.stop = True
    if a.fixed_steps:
        torch.save(ema.model.state_dict(), a.ckpt)
        print(f'сохранены веса EMA на шаге {step} (переобучение, без отбора)')
    else:
        print(f'ЛУЧШЕЕ val RMSLE {best[0]:.4f} на шаге {best[1]}')

    # --- финал: калибровка, выгрузка, предсказание -------------------------
    model.load_state_dict(torch.load(a.ckpt, map_location=dev)); model.eval()
    lp, lp0, lt = _eval(model, st, tab, val_u, VAL, dev, a.batch)
    np.save(f'val_logpred_{a.tag}.npy', lp.numpy())
    np.save(f'val_logtrue_{a.tag}.npy', lt.numpy())
    if tab is not None:
        print(f'итог: с таблицей {rmsle_from_log(lp, lt):.4f}, '
              f'без таблицы {rmsle_from_log(lp0, lt):.4f}, '
              f'corr предсказаний {np.corrcoef(lp.numpy(), lp0.numpy())[0,1]:.5f}', flush=True)

    def pick_shrink(v, fixed, what):
        if fixed:
            r = rmsle_from_log(fixed * v, lt)
            print(f'усадка {what} задана флагом: {fixed} -> val RMSLE {r:.4f} '
                  f'(при нарушенном зазоре это число НЕ показатель)')
            return fixed, r
        b = (9e9, 1.0)
        for c in [1.0, 0.97, 0.95, 0.93, 0.9, 0.87, 0.85, 0.82, 0.8]:
            r = rmsle_from_log(c * v, lt); b = min(b, (r, c))
            print(f'  усадка {what} {c}: {r:.4f}')
        print(f'выбрана усадка {what} {b[1]} -> val RMSLE {b[0]:.4f}')
        return b[1], b[0]

    cal, _r = pick_shrink(lp, a.cal_fixed, 'совместной модели')
    bestc = (_r, cal)

    cal0 = r0_cal = None
    if tab is not None:
        cal0, r0_cal = pick_shrink(lp0, a.cal_fixed_tabless, 'ствола')

    res = {'tag': a.tag, 'phase': a.phase, 'seed': a.seed, 'cal': float(cal),
           'rmsle_cal': float(bestc[0]), 'rmsle_raw': float(rmsle_from_log(lp, lt)),
           'best_step': int(best[1] or step), 'steps_run': int(step),
           'n_tab': (tab.n_tab if tab is not None else 0),
           'grid': grid, 'val_anchor': VAL, 'test_anchor': TEST,
           'tab_off': bool(a.tab_off), 'init_from': a.init_from}
    if tab is not None:
        res['cal_tabless'] = float(cal0)
        res['rmsle_tabless_cal'] = float(r0_cal)
        res['rmsle_tabless'] = float(rmsle_from_log(lp0, lt))
        res['corr_with_tabless'] = float(np.corrcoef(lp.numpy(), lp0.numpy())[0, 1])
    Path(f'result_{a.tag}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'записано result_{a.tag}.json', flush=True)
    if a.export:
        import polars as _pl
        os.makedirs(a.export, exist_ok=True)
        def _dump(v, c, name):
            vp = np.expm1(np.clip(c * v.numpy(), 0, None))
            _pl.DataFrame({'user_id': st.uids[val_u.numpy()].astype(np.int64),
                           'pred': vp.astype(np.float64)}).sort('user_id').write_parquet(
                f'{a.export}/{name}_val.parquet')
            print(f'выгружено {a.export}/{name}_val.parquet ({len(vp)} юзеров)', flush=True)
        _dump(lp, cal, a.tag)
        if tab is not None:
            # Ствол без таблицы — отдельный член бленда, а не диагностика:
            # он слабее по скору, но заметно менее скоррелирован с паком.
            pass
    save_history(a.tag, hist)
    if not a.no_plots:
        make_plots(a.tag, hist, lp.numpy(), lt.numpy(), cal)

    if a.predict:
        out = np.zeros(st.n_u, np.float32); out0 = np.zeros(st.n_u, np.float32)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(dev == 'cuda')):
            for i in range(0, st.n_u, a.batch):
                u = torch.arange(i, min(i + a.batch, st.n_u))
                anc = torch.full((len(u),), TEST)
                x, _ = st.batch(u, anc, dev, with_target=False)
                tb = tab.to_dev(tab.gather(u, anc), dev) if tab is not None else None
                o, o0 = model.forward_both(x, tb)
                out[i:i + len(u)] = o['y30'].float().cpu().numpy()
                out0[i:i + len(u)] = o0['y30'].float().cpu().numpy()
        np.save(a.predict + '.userids.npy', st.uids)

        def _sub(v, c, path, tag):
            pred = np.expm1(np.clip(c * v, 0, None))
            np.save(path + '.logpred.npy', v)
            with open(path, 'w', encoding='utf-8', newline='') as f:
                f.write('user_id,predict\n')
                for uid, x_ in zip(st.uids, pred): f.write(f'{int(uid)},{x_:.6f}\n')
            print('сохранено', path, 'среднее', float(pred.mean()), flush=True)
            if a.export:
                import polars as _pl
                _pl.DataFrame({'user_id': st.uids.astype(np.int64),
                               'pred': pred.astype(np.float64)}).sort('user_id').write_parquet(
                    f'{a.export}/{tag}_test.parquet')
                print(f'выгружено {a.export}/{tag}_test.parquet', flush=True)

        _sub(out, cal, a.predict, a.tag)
        if tab is not None:
            pass
    return bestc[0]


# ------------------------------------------------------------------- аргументы
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default='tensor')
    p.add_argument('--phase', choices=['A', 'B'], default='A')
    p.add_argument('--expect-val', type=int, default=VAL_ANCHOR_DEF)
    p.add_argument('--expect-test', type=int, default=TEST_ANCHOR_DEF)
    p.add_argument('--tag', default='tfm4')
    p.add_argument('--ckpt', default='')
    p.add_argument('--arch', default='transformer', choices=['transformer', 'tcn'])
    # таблица
    p.add_argument('--prep-tab', action='store_true', help='собрать кэш и выйти')
    p.add_argument('--tab-npz', default='', help='каталог с tabf16_*.npz для --prep-tab')
    p.add_argument('--tab-cache', default='tab_raw.f16')
    p.add_argument('--tab-off', action='store_true',
                   help='КОНТРОЛЬ РАВНОЙ ЁМКОСТИ: та же сетка якорей, но без таблицы')
    p.add_argument('--tab-dim', type=int, default=128)
    p.add_argument('--tab-gelu', action='store_true')
    p.add_argument('--tab-lr-mult', type=float, default=1.0)
    p.add_argument('--tab-warmup', type=int, default=0)
    p.add_argument('--tab-keep-degenerate', action='store_true')
    p.add_argument('--tab-keep-all', action='store_true')
    p.add_argument('--tab-ram', action='store_true', help='держать таблицу в RAM (+2.5 ГБ), а не в memmap')
    p.add_argument('--no-drift', action='store_true')
    p.add_argument('--tab-drift-use-test', action='store_true',
                   help='учитывать тестовый якорь при отсеве по сдвигу (по умолчанию только валидация)')
    p.add_argument('--tab-drop-drift', type=float, default=0.0,
                   help='выбросить колонки, чьё среднее на val/test отъехало больше чем на N sd')
    p.add_argument('--min-anchor-day', type=int, default=0,
                   help='не брать якоря обучения раньше этого дня (у ранних мало истории)')
    p.add_argument('--collapse-tol', type=float, default=0.02)
    p.add_argument('--check-init', action='store_true', default=True)
    p.add_argument('--no-check-init', dest='check_init', action='store_false')
    p.add_argument('--check-init-only', action='store_true')
    # обучение — те же значения, что у tfm3b
    p.add_argument('--minutes', type=float, default=55)
    p.add_argument('--steps', type=int, default=200000)
    p.add_argument('--fixed-steps', type=int, default=0)
    p.add_argument('--lr', type=float, default=7e-4)
    p.add_argument('--pct-start', type=float, default=0.1)
    p.add_argument('--channels', type=int, default=192)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--blocks', type=int, default=8)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--wd', type=float, default=0.0122)
    p.add_argument('--aux', type=float, default=0.25)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--ema', type=float, default=0.995)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--eval-every', type=int, default=2000)
    p.add_argument('--es-metric', default='cal', choices=['cal', 'raw'])
    p.add_argument('--cal-fixed', type=float, default=0.0)
    p.add_argument('--cal-fixed-tabless', type=float, default=0.0,
                   help='усадка ствола без таблицы, из фазы A (поле cal_tabless)')
    p.add_argument('--val-users', type=int, default=0)
    p.add_argument('--val-all', action='store_true', default=True)
    p.add_argument('--no-val-all', dest='val_all', action='store_false')
    p.add_argument('--init-from', default='')
    p.add_argument('--export', default='')
    p.add_argument('--predict', default='')
    p.add_argument('--abs-time', action='store_true')
    p.add_argument('--cohort1', action='store_true')
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--selftest', action='store_true')
    return p


if __name__ == '__main__':
    a = build_parser().parse_args()
    if not a.ckpt: a.ckpt = f'model_{a.tag}.pt'   # как в train_tcn.py
    if a.selftest:
        from tfm4_selftest import run
        raise SystemExit(run())
    if a.prep_tab:
        if not a.tab_npz:
            raise SystemExit('--prep-tab требует --tab-npz с каталогом tabf16_*.npz')
        prep_tab(Path(a.tab_npz), Path(a.tab_cache), None,
                 a.tab_keep_degenerate, a.tab_keep_all)
        raise SystemExit(0)
    raise SystemExit(0 if main(a) is not None else 1)
