"""Самообучение энкодера на неразмеченных последовательностях (трек №5, план).

    python work/scripts/seq/pretrain.py --data tensor --minutes 60 --out pre378.pt

Задача: восстановить замаскированные токены истории по остальным. Разметка не
нужна вовсе, GMV-таргет не используется. Затем ствол переносится в train_tcn.py
флагом --init-from и дообучается на GMV.

Зачем. План просит искать представление, которое НЕ является функцией их 203
табличных признаков. Энкодер, обученный восстанавливать поведение, оптимизирован
не под целевую метрику вообще — это ровно тот класс представлений.

═══ ГЛАВНОЕ ПРО УТЕЧКУ: читать до запуска ═══

План говорит «в предобучении можно использовать февраль 2026, входные данные
теста есть, утечки нет, таргет не трогаем». Это верно лишь наполовину.

Таргет действительно не используется. Но входное окно якоря A — это дни
A-364..A, и если предобучать на якорях выше 378, энкодер увидит дни 379-408 как
ВХОД. А 379-408 — это в точности окно, на котором меряется валидация. Веса,
знающие статистику окна оценки, дают на валидации завышенный скор, которого на
тесте не будет: там аналогичных данных (409-438) не существует.

Правило: **якорь предобучения не выше начала окна оценки минус один день.**

    для валидации (окно 379-408):  --max-pre-anchor 378   <- по умолчанию
    для теста     (окно 409-438):  --max-pre-anchor 408

То есть нужны ДВА предобучения, симметрично фазам A и B в run_all.py. Февраль
законно входит только во второе, из которого получаются тестовые предсказания.
Скрипт печатает это соответствие при запуске и отказывается работать при
якоре выше 408.

═══ Что именно восстанавливается ═══

Трансформер режет 365 дней на 124 токена: последние 84 дня подённо плюс 40
недельных агрегатов. Маскируются токены, а не сырые дни — модель всё равно
живёт на этой сетке.

Восстанавливаются только ПОВЕДЕНЧЕСКИЕ каналы (первые n_ch). Календарные
каналы — детерминированная функция даты, восстанавливать их бессмысленно:
модель выучила бы календарь и получила бы низкий лосс ни за что.
"""
from __future__ import annotations
import argparse, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_tcn as T


class Recon(nn.Module):
    """Ствол трансформера + голова восстановления. Neck не участвует."""
    def __init__(self, trunk: "T.Transformer", ch: int, n_out: int, drop: float):
        super().__init__()
        self.trunk = trunk
        self.mask_tok = nn.Parameter(torch.randn(1, 1, ch) * 0.02)
        self.head = nn.Sequential(nn.LayerNorm(ch), nn.Linear(ch, ch), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(ch, n_out))

    def forward(self, x, mask):
        tok = self.trunk.tokenize(x)                 # [B, T, C]
        h = self.trunk.embed(tok)                    # [B, T, ch], позиция уже добавлена
        h = torch.where(mask.unsqueeze(-1), self.mask_tok.to(h.dtype), h)
        h = self.trunk.enc(h)
        return self.head(h), tok


class Sampler:
    """Батчи без таргета: любой юзер, любой якорь до потолка."""
    def __init__(self, st, batch, lo, hi, depth=6, workers=2):
        import threading, queue
        self.q = queue.Queue(maxsize=depth)
        self.st, self.b, self.lo, self.hi = st, batch, lo, hi
        self.th = [threading.Thread(target=self._run, daemon=True) for _ in range(workers)]
        for t in self.th: t.start()

    def _run(self):
        st, B = self.st, self.b
        while True:
            u = torch.randint(0, st.n_u, (B,))
            an = torch.randint(self.lo, self.hi + 1, (B,))
            keep = st.valid[u, an]
            if keep.sum() < 8: continue
            self.q.put(st.cpu_batch(u[keep], an[keep], with_target=False))

    def get(self): return self.q.get()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='tensor')
    p.add_argument('--out', default='', help='куда сохранить веса ствола')
    p.add_argument('--max-pre-anchor', type=int, default=378,
                   help='378 для валидационной ветки, 408 для тестовой. См. шапку файла')
    p.add_argument('--min-anchor', type=int, default=30)
    p.add_argument('--mask-frac', type=float, default=0.3)
    p.add_argument('--minutes', type=float, default=60)
    p.add_argument('--steps', type=int, default=200000)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--lr', type=float, default=7e-4)
    p.add_argument('--wd', type=float, default=0.0122)
    p.add_argument('--pct-start', type=float, default=0.15)
    p.add_argument('--channels', type=int, default=192)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--holdout-users', type=int, default=20000,
                   help='юзеры, исключённые из предобучения, для честного замера восстановления')
    p.add_argument('--cohort1', action='store_true')
    a = p.parse_args()

    if a.max_pre_anchor > 408:
        raise SystemExit(f'--max-pre-anchor {a.max_pre_anchor} выше дня 408: тензор кончается там')
    win = 'валидации 379-408' if a.max_pre_anchor <= 378 else 'теста 409-438'
    if a.max_pre_anchor > 378:
        print(f'ВНИМАНИЕ: якорь до {a.max_pre_anchor} — эти веса годятся ТОЛЬКО для тестовой '
              f'ветки. Для валидации нужен отдельный прогон с --max-pre-anchor 378', flush=True)
    print(f'потолок якоря {a.max_pre_anchor} -> веса законны для окна {win}', flush=True)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    st = T.Store(a.data, pin=(dev == 'cuda'), cohort3=not a.cohort1); st.to_device(dev)
    if a.max_pre_anchor >= st.n_d:
        raise SystemExit(f'потолок {a.max_pre_anchor} вне тензора ({st.n_d} дней)')

    trunk = T.Transformer(st.n_in, a.channels, a.layers, a.dropout, a.heads)
    net = Recon(trunk, a.channels, st.n_ch, a.dropout).to(dev)
    print(f'параметров: {sum(q.numel() for q in net.parameters())/1e6:.3f} млн '
          f'(из них ствол {sum(q.numel() for q in trunk.parameters())/1e6:.3f})', flush=True)
    print(f'восстанавливаем {st.n_ch} поведенческих каналов из {st.n_in}; '
          f'календарные не восстанавливаем — это функция даты', flush=True)

    # отложенные юзеры: на них не учимся, по ним меряем
    g = torch.Generator().manual_seed(12345)
    perm = torch.randperm(st.n_u, generator=g)
    hold = perm[:a.holdout_users]
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=max(a.pct_start, 1e-3))
    scaler = torch.amp.GradScaler(enabled=(dev == 'cuda'))
    pf = Sampler(st, a.batch, a.min_anchor, a.max_pre_anchor, workers=a.workers)

    def make_mask(B, Tn):
        k = max(1, int(round(Tn * a.mask_frac)))
        idx = torch.rand(B, Tn, device=dev).argsort(-1)[:, :k]
        m = torch.zeros(B, Tn, dtype=torch.bool, device=dev)
        return m.scatter_(1, idx, True)

    @torch.no_grad()
    def evaluate(n_batch=6):
        net.eval(); tot = cnt = 0.0
        for _ in range(n_batch):
            u = hold[torch.randint(0, len(hold), (a.batch,))]
            an = torch.randint(a.min_anchor, a.max_pre_anchor + 1, (a.batch,))
            keep = st.valid[u, an]
            if keep.sum() < 8: continue
            x, _ = st.to_dev(st.cpu_batch(u[keep], an[keep], with_target=False), dev, False)
            mk = make_mask(x.shape[0], trunk.n_old + trunk.RECENT)
            with torch.amp.autocast('cuda', enabled=(dev == 'cuda')):
                pr, tk = net(x, mk)
            e = ((pr[..., :st.n_ch] - tk[..., :st.n_ch]) ** 2).mean(-1)[mk]
            tot += e.float().sum().item(); cnt += e.numel()
        net.train(); return tot / max(cnt, 1)

    print(f'отложено {len(hold)} юзеров для честного замера', flush=True)
    t0 = time.time(); step = 0; retimed = False; hist = []
    base = None
    while step < a.steps:
        x, _ = st.to_dev(pf.get(), dev, with_target=False)
        mk = make_mask(x.shape[0], trunk.n_old + trunk.RECENT)
        with torch.amp.autocast('cuda', enabled=(dev == 'cuda')):
            pr, tk = net(x, mk)
            d = (pr[..., :st.n_ch] - tk[..., :st.n_ch]) ** 2
            loss = d.mean(-1)[mk].mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        sc = scaler.get_scale(); scaler.step(opt); scaler.update()
        if scaler.get_scale() >= sc and sched.last_epoch < sched.total_steps - 1: sched.step()
        step += 1

        if not retimed and step == 300 and a.minutes:
            rate = step / (time.time() - t0)
            plan = max(step + 100, min(a.steps, int(rate * a.minutes * 60 * 0.93)))
            print(f'скорость {rate:.1f} шаг/с -> расписание на {plan} шагов', flush=True)
            a.steps = plan
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=plan,
                                                        pct_start=max(a.pct_start, 1e-3))
            for _ in range(step): sched.step()
            retimed = True

        if step % a.eval_every == 0 or step == a.steps:
            h = evaluate(); el = time.time() - t0
            if base is None: base = h
            print(f'step {step:6d}  train {loss.item():.4f}  holdout {h:.4f}  '
                  f'({h/base:.3f} от первого замера)  {el/60:.1f} мин  '
                  f'({step/el:.1f} шаг/с)', flush=True)
            hist.append((step, float(loss.item()), h))
            if a.minutes and el > a.minutes * 60:
                print('лимит времени'); break

    out = a.out or f'pretrain_a{a.max_pre_anchor}.pt'
    torch.save(trunk.state_dict(), out)
    import csv
    with open(f'pretrain_history_a{a.max_pre_anchor}.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['step', 'train_mse', 'holdout_mse']); w.writerows(hist)
    print(f'\nсохранён ствол: {out}  ({os.path.getsize(out)/2**20:.1f} МБ)')
    if hist:
        print(f'восстановление на отложенных: {hist[0][2]:.4f} -> {hist[-1][2]:.4f} '
              f'({hist[-1][2]/hist[0][2]:.3f} от старта)')
    print(f'\nдальше: train_tcn.py ... --init-from {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
