"""
Шаг 2: TCN на дневных последовательностях. Заточено под RTX 2060 6 ГБ / 16 ГБ RAM.

Ключевые решения по памяти:
  * тензор данных живёт в pinned RAM (~2.5 ГБ), в VRAM едут только батчи (~4 МБ);
  * fp16 через AMP + GradScaler (на Turing нет bf16 и нет TF32, fp16 — единственный рычаг);
  * дилатации 1..128 дают рецептивное поле 365 дней за 8 слоёв, внимания нет вообще.

Обучение: якорь сэмплируется случайно для каждого юзера на каждом шаге, поэтому
эффективный датасет — сотни миллионов пар (юзер, дата), а не 250k строк.

  python train_tcn.py --data tensor --minutes 55
  python train_tcn.py --data tensor --minutes 55 --predict sub_nn.csv
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import argparse, time, math, os

SEQ_LEN      = 365
HORIZON      = 30
VAL_ANCHOR   = 378      # 2026-01-14; переопределяется флагом --val-anchor
                        # 364 = 2025-12-31 (второе окно для отбора конфигураций)
TEST_ANCHOR  = 408      # 2026-02-13
MAX_TR_ANCHOR= 348      # = VAL_ANCHOR-30, пересчитывается в main()
MIN_TR_ANCHOR= 200      # переопределяется флагом --min-anchor

# ---------------------------------------------------------------- данные
class Store:
    def __init__(self, path, pin=True, abs_time=False, cohort3=True):
        m = np.load(f'{path}/meta.npz')
        self.n_u, self.n_d, self.n_ch = int(m['n_users']), int(m['n_days']), int(m['n_ch'])
        self.uids  = m['user_ids']
        # valid — для валидации и выгрузки (обязаны быть все 250к юзеров)
        # valid_tr — для сэмплирования обучающих якорей: правило трёх блоков,
        # то самое, по которому организаторы отбирали юниверс
        self.valid = torch.from_numpy(m['valid_anchor'])
        if cohort3 and 'valid_anchor3' in m:
            self.valid_tr = torch.from_numpy(m['valid_anchor3']); print('когорта обучения: 3 блока')
        else:
            self.valid_tr = self.valid
            if cohort3: print('valid_anchor3 нет в meta.npz — запустите make_valid3.py')
            else: print('когорта обучения: 1 блок (старое поведение)')
        cal = torch.from_numpy(m['calendar'])
        seq = np.memmap(f'{path}/seq.f16', np.float16, 'r', shape=(self.n_u,self.n_d,self.n_ch))
        gmv = np.memmap(f'{path}/gmv.f32', np.float32, 'r', shape=(self.n_u,self.n_d))
        orr = np.memmap(f'{path}/ord.f16', np.float16, 'r', shape=(self.n_u,self.n_d))
        print('загружаю тензор в RAM...', flush=True)
        self.seq = torch.from_numpy(np.array(seq))     # ~2 ГБ в RAM
        self.gmv = torch.from_numpy(np.array(gmv))
        self.ord = torch.from_numpy(np.array(orr))
        self.cal = cal if abs_time else cal[:, :-1]   # последний столбец — days/n_days
        if pin:
            for t in ('seq','gmv','ord'):
                try: setattr(self, t, getattr(self, t).pin_memory())
                except RuntimeError: pass
        # нормировка каналов по глобальной статистике (считаем на подвыборке юзеров)
        sub = self.seq[::37].float()
        self.mu = sub.mean((0,1)); self.sd = sub.std((0,1)).clamp_min(1e-3)
        del sub
        self.n_in = self.n_ch + self.cal.shape[1]
        self.mu_d = self.sd_d = self.cal_d = None
        self.do_pin = torch.cuda.is_available()

    def to_device(self, dev):
        self.mu_d, self.sd_d, self.cal_d = self.mu.to(dev), self.sd.to(dev), self.cal.to(dev)
        print(f'users={self.n_u} days={self.n_d} in_channels={self.n_in}', flush=True)

    def cpu_batch(self, users, anchors, with_target=True):
        """Тяжёлая часть: гатер по 250k юзеров. Выполняется в фоновом потоке."""
        idx = anchors.view(-1,1) - torch.arange(SEQ_LEN-1, -1, -1)
        pad = idx < 0
        idx = idx.clamp_min(0)
        p = (lambda t: t.pin_memory()) if self.do_pin else (lambda t: t)
        out = {'x': p(self.seq[users.view(-1,1), idx]), 'pad': p(pad), 'idx': p(idx.int())}
        if with_target:
            t = anchors.view(-1,1) + torch.arange(1, HORIZON+1)
            out['g'] = p(self.gmv[users.view(-1,1), t])
            out['o'] = p(self.ord[users.view(-1,1), t])
        return out

    def to_dev(self, b, dev, with_target=True):
        """Лёгкая часть: перенос и арифметика уже на карте."""
        x = b['x'].to(dev, non_blocking=True).float()
        x = (x - self.mu_d) / self.sd_d
        x = x * (~b['pad'].to(dev, non_blocking=True)).unsqueeze(-1)
        c = self.cal_d[b['idx'].to(dev, non_blocking=True).long()]
        x = torch.cat([x, c], -1).transpose(1,2)
        if not with_target: return x, None
        g = b['g'].to(dev, non_blocking=True).float()
        o = b['o'].to(dev, non_blocking=True).float()
        y = {
            'y30' : torch.log1p(g.sum(1)),
            'y7'  : torch.log1p(g[:, :7].sum(1)),
            'y14' : torch.log1p(g[:, :14].sum(1)),
            'ord30': torch.log1p(o.sum(1)),
            'act30': (g > 0).float().sum(1) / HORIZON,
            'buy' : (g.sum(1) > 0).float(),
        }
        return x, y

    def batch(self, users, anchors, dev, with_target=True):
        return self.to_dev(self.cpu_batch(users, anchors, with_target), dev, with_target)

class Prefetcher:
    """Собирает батчи целиком в фоновых потоках. Гатер 384 случайных юзеров из
    тензора на 250k строк — это ~3 МБ копирования, и если делать его в главном
    потоке, карта простаивает большую часть шага."""
    def __init__(self, store, batch, depth=6, workers=2, lo=MIN_TR_ANCHOR):
        import threading, queue
        self.q = queue.Queue(maxsize=depth); self.st = store; self.b = batch; self.lo = lo
        self.stop = False
        self.th = [threading.Thread(target=self._run, daemon=True) for _ in range(workers)]
        for t in self.th: t.start()
    def _run(self):
        st, B = self.st, self.b
        while not self.stop:
            u = torch.randint(0, st.n_u, (B,))
            an = torch.randint(self.lo, MAX_TR_ANCHOR+1, (B,))
            for _ in range(4):
                bad = ~st.valid_tr[u, an]
                if not bad.any(): break
                an[bad] = torch.randint(self.lo, MAX_TR_ANCHOR+1, (int(bad.sum()),))
            keep = st.valid_tr[u, an]
            self.q.put(st.cpu_batch(u[keep], an[keep]))
    def get(self): return self.q.get()

# ---------------------------------------------------------------- модель
class Block(nn.Module):
    def __init__(self, ch, dil, drop):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        self.n1, self.n2 = nn.GroupNorm(8, ch), nn.GroupNorm(8, ch)
        self.do = nn.Dropout(drop)
    def forward(self, x):
        h = self.do(F.gelu(self.n1(self.c1(x))))
        h = self.do(F.gelu(self.n2(self.c2(h))))
        return x + h

HEADS = {'y30':1, 'y7':1, 'y14':1, 'ord30':1, 'act30':1, 'buy':1}

class Neck(nn.Module):
    """Общая для обеих архитектур часть: пулинг по времени + мультизадачные головы."""
    def __init__(self, ch, drop):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(ch*3, 256), nn.GELU(), nn.Dropout(drop))
        self.out = nn.ModuleDict({k: nn.Linear(256, v) for k,v in HEADS.items()})
    def forward(self, h):                       # h: [B, ch, T]
        # последний шаг + среднее + максимум: «что сейчас» и профиль целиком
        z = self.mlp(torch.cat([h[:,:,-1], h.mean(-1), h.amax(-1)], 1))
        return {k: self.out[k](z).squeeze(-1) for k in HEADS}

class TCN(nn.Module):
    def __init__(self, n_in, ch=128, n_blocks=8, drop=0.1):
        super().__init__()
        self.stem = nn.Conv1d(n_in, ch, 1)
        self.blocks = nn.ModuleList([Block(ch, 2**i, drop) for i in range(n_blocks)])
        self.neck = Neck(ch, drop)
    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks: h = b(h)
        return self.neck(h)

class Transformer(nn.Module):
    """Мультиразрешение: последние 84 дня подённо + предыдущие 280 дней
    недельными агрегатами. 124 токена вместо 365 — внимание дешевеет в 8.6 раза,
    а разрешение теряется только там, где оно и не нужно."""
    RECENT, OLD_W = 84, 7
    def __init__(self, n_in, ch=128, n_layers=4, drop=0.1, heads=4):
        super().__init__()
        self.n_old = (SEQ_LEN - self.RECENT) // self.OLD_W
        n_tok = self.n_old + self.RECENT
        self.proj = nn.Linear(n_in, ch)
        self.pos  = nn.Parameter(torch.randn(1, n_tok, ch) * 0.02)
        self.res  = nn.Parameter(torch.randn(1, 2, ch) * 0.02)     # метка разрешения токена
        layer = nn.TransformerEncoderLayer(ch, heads, ch*2, drop, 'gelu',
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.neck = Neck(ch, drop)
    def forward(self, x):                                  # x: [B, C, L]
        cut = self.n_old * self.OLD_W
        old = x[:, :, -SEQ_LEN:-self.RECENT][:, :, -cut:]
        old = old.reshape(x.shape[0], x.shape[1], self.n_old, self.OLD_W).mean(-1)
        tok = torch.cat([old, x[:, :, -self.RECENT:]], -1).transpose(1, 2)
        h = self.proj(tok) + self.pos
        h = h + torch.cat([self.res[:, :1].expand(-1, self.n_old, -1),
                           self.res[:, 1:].expand(-1, self.RECENT, -1)], 1)
        h = self.enc(h).transpose(1, 2)                     # [B, ch, T]
        return self.neck(h)

def build_model(arch, n_in, a):
    if arch == 'tcn':         return TCN(n_in, a.channels, a.blocks, a.dropout)
    if arch == 'transformer': return Transformer(n_in, a.channels, a.layers, a.dropout, a.heads)
    raise ValueError(arch)

class EMA:
    """Скользящее среднее весов. При шумном таргете одиночный чекпоинт скачет
    на ±0.02 RMSLE от шага к шагу; усреднённые веса дают стабильно лучший
    результат и почти ничего не стоят."""
    def __init__(self, model, decay):
        import copy
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model, step):
        # прогрев: с decay=0.999 без него первые тысячи шагов EMA держит инициализацию
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        for e, m in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point: e.mul_(d).add_(m.detach(), alpha=1-d)
            else: e.copy_(m)

# ---------------------------------------------------------------- обучение
def save_history(tag, hist):
    import csv
    if not hist: return
    with open(f'history_{tag}.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['step','loss','val_rmsle','lr','minutes']); w.writerows(hist)

def make_plots(tag, hist, lp, lt, cal):
    """Кривые обучения и диагностика предсказаний на валидации."""
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib не установлен, графики пропущены (pip install matplotlib)'); return

    if hist:
        st_,ls,rm,lr,_ = map(np.array, zip(*hist))
        fig,ax = plt.subplots(1,3, figsize=(15,4))
        ax[0].plot(st_, ls); ax[0].set_title('лосс на батче'); ax[0].set_xlabel('шаг')
        ax[1].plot(st_, rm, marker='o', ms=3); ax[1].axhline(rm.min(), ls='--', c='gray')
        ax[1].set_title(f'val RMSLE (лучшее {rm.min():.4f})'); ax[1].set_xlabel('шаг')
        ax[2].plot(st_, lr); ax[2].set_title('learning rate'); ax[2].set_xlabel('шаг')
        for x in ax: x.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f'curves_{tag}.png', dpi=110); plt.close(fig)

    p = np.clip(cal*lp, 0, None); e2 = (p-lt)**2
    fig,ax = plt.subplots(1,3, figsize=(15,4))
    # калибровка: идеальная модель ложится на диагональ
    q = np.quantile(p, np.linspace(0,1,21)); q[-1] += 1e-6
    b = np.clip(np.digitize(p, q)-1, 0, 19)
    mp = np.array([p[b==i].mean() for i in range(20)])
    mt = np.array([lt[b==i].mean() for i in range(20)])
    ax[0].plot(mp, mt, marker='o'); ax[0].plot([0,mp.max()],[0,mp.max()],'--',c='gray')
    ax[0].set_xlabel('предсказано, log1p'); ax[0].set_ylabel('факт, log1p'); ax[0].set_title('калибровка')
    # откуда берётся ошибка
    contrib = np.array([e2[b==i].sum() for i in range(20)]); contrib /= contrib.sum()
    ax[1].bar(range(20), 100*contrib)
    ax[1].set_xlabel('вентиль предсказания'); ax[1].set_ylabel('% суммарной MSLE')
    ax[1].set_title('вклад в ошибку по группам')
    # распределения
    ax[2].hist(lt, bins=60, alpha=.5, label='факт', density=True)
    ax[2].hist(p,  bins=60, alpha=.5, label='предсказание', density=True)
    ax[2].legend(); ax[2].set_title('распределение log1p'); ax[2].set_xlabel('log1p(GMV)')
    for x in ax: x.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f'diagnostics_{tag}.png', dpi=110); plt.close(fig)

    z = lt == 0
    print(f'\nдоля истинных нулей {z.mean():.3f}, их вклад в ошибку {e2[z].sum()/e2.sum():.1%}')
    print(f'медиана предсказания на истинных нулях: {np.expm1(p[z]).mean():.2f}')
    print(f'графики: curves_{tag}.png, diagnostics_{tag}.png, history_{tag}.csv')

def cal_rmsle(lp, lt, bins=24, folds=2, seed=0):
    """RMSLE после биновой калибровки, честно — со скрещиванием по фолдам.

    Перед блендом прогнозы проходят калибровку, которая переписывает уровень.
    Значит и раннюю остановку надо вести по калиброванному скору: сырая метрика
    почти целиком отражает схождение уровня, то есть оптимизирует то, что и так
    будет переписано, ценой ранжирования, которое калибровка сохраняет.
    """
    lp = np.asarray(lp, np.float64); lt = np.asarray(lt, np.float64)
    n = len(lp); rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, n)
    out = lp.copy()
    edges = np.quantile(lp, np.linspace(0, 1, bins + 1)); edges[-1] += 1e-9
    b = np.clip(np.digitize(lp, edges) - 1, 0, bins - 1)
    for f in range(folds):
        tr, te = fold != f, fold == f
        for k in range(bins):
            mtr, mte = tr & (b == k), te & (b == k)
            if mtr.sum() >= 50 and mte.any():
                out[mte] = lp[mte] + (lt[mtr] - lp[mtr]).mean()
    return float(np.sqrt(np.mean((np.clip(out, 0, None) - lt) ** 2)))

def rmsle_from_log(pred_log, true_log):
    return float(torch.sqrt(((pred_log.clamp_min(0) - true_log)**2).mean()))

def main(a, st=None):
    global VAL_ANCHOR, MAX_TR_ANCHOR, TEST_ANCHOR
    VAL_ANCHOR = a.val_anchor
    if a.test_anchor: TEST_ANCHOR = a.test_anchor
    # По умолчанию зазор 30 дней: таргет обучения не залезает в валидацию.
    # --max-tr-anchor отвязывает границу — нужно для переобучения на train+val
    # перед предсказанием теста (контракт exp_lib). Зазор при этом нарушается
    # СОЗНАТЕЛЬНО, поэтому такой прогон обязан идти с --fixed-steps: валидационная
    # метрика там подсматривает и ранней остановке доверять нельзя.
    MAX_TR_ANCHOR = a.max_tr_anchor if a.max_tr_anchor else VAL_ANCHOR - HORIZON
    _gap = VAL_ANCHOR - MAX_TR_ANCHOR
    if _gap < HORIZON:
        print(f'ВНИМАНИЕ: зазор {_gap} дней вместо {HORIZON} — валидационный скор '
              f'завышен и для сравнения моделей НЕ ГОДИТСЯ', flush=True)
        if not a.fixed_steps:
            raise SystemExit('при нарушенном зазоре обязателен --fixed-steps N '
                             '(число шагов берите из парного прогона с зазором)')
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    if st is None:
        st = Store(a.data, pin=(dev=='cuda'), abs_time=a.abs_time, cohort3=not a.cohort1); st.to_device(dev)
    model = build_model(a.arch, st.n_in, a).to(dev)
    print('архитектура:', a.arch, flush=True)
    if dev == 'cuda': model = model.to(memory_format=torch.contiguous_format)
    print('параметров:', sum(p.numel() for p in model.parameters())/1e6, 'млн', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    scaler = torch.amp.GradScaler(enabled=(dev=='cuda'))
    W = {'y30':1.0, 'y7':a.aux, 'y14':a.aux, 'ord30':a.aux, 'act30':a.aux, 'buy':a.aux}

    # валидация фиксирована: те же юзеры, что и в тесте
    # Маска valid_anchor на раннем якоре отсеивает часть юзеров (на 2025-09-15
    # проходило 0.9086). Для выгрузки в бленд нужны все 250000 строк, иначе
    # координатору нечего джойнить, поэтому --val-all маску обходит.
    val_u = (torch.arange(st.n_u) if a.val_all
             else torch.arange(st.n_u)[st.valid[:, VAL_ANCHOR]])
    if a.val_all:
        # подвыборка убила бы смысл флага: по умолчанию --val-users 40000
        if a.val_users:
            print(f'--val-all: подвыборка {a.val_users} отключена', flush=True)
            a.val_users = 0
        print(f'--val-all: маска якоря обойдена, берём всех {len(val_u)} юзеров', flush=True)
    if a.val_users and a.val_users < len(val_u):        # сид фиксирован: иначе прогоны несравнимы
        g = torch.Generator().manual_seed(42)
        val_u = val_u[torch.randperm(len(val_u), generator=g)[:a.val_users]]
    val_a = torch.full((len(val_u),), VAL_ANCHOR)
    np.save(f'val_user_ids_{a.tag}.npy', st.uids[val_u.numpy()])  # для джойна при блендинге
    print(f'валидация: {len(val_u)} юзеров', flush=True)

    ema = EMA(model, a.ema)
    if a.fixed_steps:
        a.steps, a.minutes = a.fixed_steps, 0.0
        print(f'режим переобучения: ровно {a.fixed_steps} шагов, ранней остановки нет', flush=True)
    # расписание строится ПОСЛЕ подмены a.steps: иначе фаза переобучения (--fixed-steps)
    # обрывает отжиг на середине (LR≈5e-4 из пика 7e-4) и тестовые модели систематически
    # хуже парных валидационных
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=max(a.pct_start,1e-3))
    t0 = time.time(); best = (9e9, None); step = 0; retimed = False
    hist = []   # (шаг, лосс, val RMSLE, lr, минуты)
    if a.eval_only:
        model.load_state_dict(torch.load(a.ckpt, map_location=dev))
        ema.model.load_state_dict(model.state_dict())
        print('режим выгрузки: обучение пропущено', flush=True)
        a.steps = 0
    pf = Prefetcher(st, a.batch, workers=a.workers, lo=a.min_anchor)
    print(f'валидационный якорь {VAL_ANCHOR}, обучающие {a.min_anchor}..{MAX_TR_ANCHOR} '
          f'({MAX_TR_ANCHOR-a.min_anchor+1} шт, зазор {HORIZON} дней)', flush=True)
    t_wait = 0.0
    while step < a.steps:
        model.train()
        _tw = time.time(); raw = pf.get(); t_wait += time.time() - _tw
        x, y = st.to_dev(raw, dev)
        with torch.amp.autocast('cuda', enabled=(dev=='cuda')):
            p = model(x)
            loss = sum(W[k]*(F.binary_cross_entropy_with_logits(p[k], y[k]) if k=='buy'
                             else F.mse_loss(p[k], y[k])) for k in W)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        _sc = scaler.get_scale(); scaler.step(opt); scaler.update()
        if scaler.get_scale() >= _sc and sched.last_epoch < sched.total_steps-1: sched.step()
        step += 1; ema.update(model, step)

        # после разгона пересчитываем длину расписания под фактическую скорость,
        # чтобы learning rate успел отжечься ровно к концу отведённого времени
        if not retimed and step == 300 and a.minutes:
            rate = step / (time.time()-t0)
            plan = max(step+100, min(a.steps, int(rate * a.minutes * 60 * 0.93)))
            if abs(plan - a.steps) > 500:
                print(f'скорость {rate:.1f} шаг/с -> расписание на {plan} шагов', flush=True)
                a.steps = plan
                sched = torch.optim.lr_scheduler.OneCycleLR(
                    opt, a.lr, total_steps=plan, pct_start=max(a.pct_start,1e-3), last_epoch=-1)
                for _ in range(step): sched.step()
            retimed = True

        if step % a.eval_every == 0 or step == a.steps:
            preds=[]; trues=[]
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(dev=='cuda')):
                for i in range(0, len(val_u), a.batch):
                    xb, yb = st.batch(val_u[i:i+a.batch], val_a[i:i+a.batch], dev)
                    preds.append(ema.model(xb)['y30'].float().cpu()); trues.append(yb['y30'].float().cpu())
            _p, _t = torch.cat(preds).numpy(), torch.cat(trues).numpy()
            r_raw = rmsle_from_log(torch.cat(preds), torch.cat(trues))
            r = cal_rmsle(_p, _t) if a.es_metric == 'cal' else r_raw
            el = time.time()-t0
            print(f'step {step:6d}  loss {loss.item():.4f}  val RMSLE {r:.4f} (сырой {r_raw:.4f})  '
                  f'{el/60:.1f} мин  ({step/el:.1f} шаг/с, простой на данных '
                  f'{100*t_wait/el:.0f}%)', flush=True)
            hist.append((step, float(loss.item()), r, float(opt.param_groups[0]['lr']), el/60))
            if a.fixed_steps:
                best = (r, step)          # чекпоинт пишется один раз, после цикла
            elif r < best[0]:
                best = (r, step); torch.save(ema.model.state_dict(), a.ckpt)
            if a.minutes and el > a.minutes*60:
                print('лимит времени'); break
    if a.fixed_steps and not a.eval_only:
        torch.save(ema.model.state_dict(), a.ckpt)
        print(f'сохранены веса EMA на шаге {step} (переобучение, без отбора)')
    elif not a.eval_only:
        print(f'ЛУЧШЕЕ val RMSLE {best[0]:.4f} на шаге {best[1]}')

    model.load_state_dict(torch.load(a.ckpt, map_location=dev)); model.eval()
    preds=[]; trues=[]
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(dev=='cuda')):
        for i in range(0, len(val_u), a.batch):
            xb, yb = st.batch(val_u[i:i+a.batch], val_a[i:i+a.batch], dev)
            preds.append(model(xb)['y30'].float().cpu()); trues.append(yb['y30'].float().cpu())
    lp, lt = torch.cat(preds), torch.cat(trues)
    np.save(f'val_logpred_{a.tag}.npy', lp.numpy()); np.save(f'val_logtrue_{a.tag}.npy', lt.numpy())
    if a.cal_fixed:
        # при нарушенном зазоре валидация подсматривает, подбирать усадку на ней нельзя
        cal = a.cal_fixed
        # bestc нужен для return в конце main(): sweep.py читает это значение
        bestc = (rmsle_from_log(cal * lp, lt), cal)
        print(f'усадка задана флагом: {cal} -> val RMSLE {bestc[0]:.4f} '
              f'(подбор отключён; при нарушенном зазоре это число НЕ показатель)')
    else:
        cal = a.calib; bestc = (9e9, 1.0)
        for c in [1.0, 0.97, 0.95, 0.93, 0.9, 0.87, 0.85]:
            r = rmsle_from_log(c*lp, lt); bestc = min(bestc, (r, c))
            print(f'  усадка {c}: {r:.4f}')
        cal = bestc[1]; print(f'выбрана усадка {cal} -> val RMSLE {bestc[0]:.4f}')
    if a.export:
        import polars as _pl, os as _os
        _os.makedirs(a.export, exist_ok=True)
        vp = np.expm1(np.clip(cal*lp.numpy(), 0, None))
        _pl.DataFrame({'user_id': st.uids[val_u.numpy()].astype(np.int64),
                       'pred': vp.astype(np.float64)}).sort('user_id').write_parquet(
            f'{a.export}/{a.tag}_val.parquet')
        print(f'выгружено {a.export}/{a.tag}_val.parquet ({len(vp)} юзеров)', flush=True)
    save_history(a.tag, hist)
    if not a.no_plots: make_plots(a.tag, hist, lp.numpy(), lt.numpy(), cal)

    if a.predict:
        out = np.zeros(st.n_u, np.float32)
        anc = torch.full((a.batch,), TEST_ANCHOR)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(dev=='cuda')):
            for i in range(0, st.n_u, a.batch):
                u = torch.arange(i, min(i+a.batch, st.n_u))
                x, _ = st.batch(u, anc[:len(u)], dev, with_target=False)
                out[i:i+len(u)] = model(x)['y30'].float().cpu().numpy()
        pred = np.expm1(np.clip(cal*out, 0, None))
        np.save(a.predict + '.logpred.npy', out)          # сырые логиты для ансамбля
        np.save(a.predict + '.userids.npy', st.uids)
        with open(a.predict, 'w') as f:
            f.write('user_id,predict\n')
            for uid, v in zip(st.uids, pred): f.write(f'{int(uid)},{v:.6f}\n')
        print('сохранено', a.predict, 'среднее', float(pred.mean()))
        if a.export:
            import polars as _pl
            _pl.DataFrame({'user_id': st.uids.astype(np.int64),
                           'pred': pred.astype(np.float64)}).sort('user_id').write_parquet(
                f'{a.export}/{a.tag}_test.parquet')
            print(f'выгружено {a.export}/{a.tag}_test.parquet', flush=True)
    # не bestc[0]: при --cal-fixed ветка подбора не выполняется и bestc не существует —
    # NameError падал уже ПОСЛЕ записи артефактов, run_all считал прогон упавшим и не
    # переименовывал негодный val-parquet
    return float(rmsle_from_log(cal * lp, lt))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='tensor')
    p.add_argument('--batch', type=int, default=384)
    p.add_argument('--channels', type=int, default=128)
    p.add_argument('--arch', default='tcn', choices=['tcn','transformer'])
    p.add_argument('--blocks', type=int, default=8)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=6e-4)
    p.add_argument('--wd', type=float, default=1e-2)
    p.add_argument('--steps', type=int, default=60000)
    p.add_argument('--minutes', type=float, default=55)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--val-users', type=int, default=40000)
    p.add_argument('--ema', type=float, default=0.999)
    p.add_argument('--aux', type=float, default=0.1)
    p.add_argument('--calib', type=float, default=0.95)
    p.add_argument('--val-anchor', type=int, default=378,
                   help='378 = 2026-01-14, 364 = 2025-12-31')
    p.add_argument('--export', default='',
                   help='папка для выгрузки предсказаний в формате work/preds тиммейта')
    p.add_argument('--min-anchor', type=int, default=60,
                   help='самый ранний обучающий якорь в днях от 2025-01-01')
    p.add_argument('--abs-time', action='store_true',
                   help='вернуть в календарь признак абсолютного времени')
    p.add_argument('--pct-start', type=float, default=0.15,
                   help='доля шагов на разогрев; 0 = без разогрева')
    p.add_argument('--es-metric', default='cal', choices=['raw','cal'],
                   help='по чему выбирать чекпоинт: cal = после биновой калибровки')
    p.add_argument('--cohort1', action='store_true',
                   help='старый фильтр: активность в одном блоке вместо трёх')
    p.add_argument('--max-tr-anchor', type=int, default=0,
                   help='верхняя граница обучающих якорей; 0 = val_anchor-30 (зазор). '
                        'Ставьте 378 для переобучения на train+val перед тестом; '
                        'требует --fixed-steps')
    p.add_argument('--fixed-steps', type=int, default=0,
                   help='обучать ровно столько шагов, без ранней остановки; '
                        'число берите из парного прогона с зазором')
    p.add_argument('--cal-fixed', type=float, default=0.0,
                   help='усадка из парного прогона с зазором; 0 = подбирать на валидации')
    p.add_argument('--test-anchor', type=int, default=0,
                   help='якорь для --predict; 0 = 408 (2026-02-13). Поставьте 380 для '
                        'инференса с застоялостью 28 дней')
    p.add_argument('--val-all', action='store_true',
                   help='игнорировать маску valid_anchor и брать все 250000 юзеров; '
                        'нужно при выгрузке с ранних якорей')
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tag', default='', help='метка прогона: имена чекпоинта и val-файлов')
    p.add_argument('--ckpt', default='')
    p.add_argument('--predict', default='')
    p.add_argument('--eval-only', action='store_true',
                   help='не обучать, только выгрузить val/test предсказания из --ckpt')
    args = p.parse_args()
    if not args.tag:  args.tag  = f'{args.arch}_s{args.seed}'
    if not args.ckpt: args.ckpt = f'model_{args.tag}.pt'
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    main(args)
