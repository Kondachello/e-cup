"""Минимальная проверка карты: где именно ломается выделение памяти.

Тензор не грузит, данные не читает — только модель на случайных числах.
Секунды вместо минут, поэтому можно перебирать варианты быстро.

  python work/scripts/seq/gpucheck.py
  python work/scripts/seq/gpucheck.py --batch 256
"""
import argparse, sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_tcn import Transformer, SEQ_LEN, HEADS
from tfm4 import TabFusion


def gb(x): return x / 2**30


def free_now():
    return torch.cuda.mem_get_info()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, nargs='+', default=[512, 384, 256, 128])
    ap.add_argument('--channels', type=int, default=192)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--n-in', type=int, default=17)
    ap.add_argument('--n-tab', type=int, default=176)
    ap.add_argument('--load-tensor', default='',
                    help='сначала загрузить настоящий тензор (как в обучении), потом мерить')
    ap.add_argument('--pin', action='store_true', help='закреплять тензор (по умолчанию нет)')
    a = ap.parse_args()

    print('=' * 66)
    print(f'torch          : {torch.__version__}')
    print(f'собран под CUDA: {torch.version.cuda}')
    print(f'карта видна    : {torch.cuda.is_available()}')
    if not torch.cuda.is_available():
        print('CUDA недоступна — дальше проверять нечего'); return 1
    cap = torch.cuda.get_device_capability(0)
    print(f'устройство     : {torch.cuda.get_device_name(0)}, compute {cap[0]}.{cap[1]}')
    archs = torch.cuda.get_arch_list()
    mine = f'sm_{cap[0]}{cap[1]}'
    print(f'скомпилировано под: {archs}')
    if mine not in archs:
        print(f'  !! {mine} НЕТ в списке. Ядра будут собираться драйвером на лету (JIT),')
        print(f'     это ест память хоста и даёт странные отказы. Нужна сборка torch с {mine}.')
    f, t = torch.cuda.mem_get_info()
    print(f'память карты   : свободно {gb(f):.2f} из {gb(t):.2f} ГБ')
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f'память хоста   : свободно {gb(vm.available):.1f} из {gb(vm.total):.1f} ГБ')
    except ImportError:
        print('память хоста   : psutil не установлен (pip install psutil)')
    print('=' * 66)

    if a.load_tensor:
        print('\n0. загружаю настоящий тензор — как в обучении')
        from train_tcn import Store
        st = Store(a.load_tensor, pin=a.pin, cohort3=True)
        st.to_device('cuda')
        a.n_in = st.n_in
        print(f'  тензор в памяти, закрепление={a.pin}, каналов на входе {st.n_in}')
        try:
            import psutil
            vm = psutil.virtual_memory()
            print(f'  память хоста теперь: свободно {gb(vm.available):.1f} ГБ')
        except ImportError:
            pass
        print(f'  память карты теперь: свободно {gb(free_now()):.2f} ГБ')

    print('\n1. просто выделить память кусками по 256 МБ')
    held, n = [], 0
    try:
        while n < 16:
            held.append(torch.empty(256 * 2**20 // 2, dtype=torch.float16, device='cuda'))
            n += 1
        print(f'  выделено {n * 256} МБ без отказа (дальше не пробовал)')
    except Exception as e:
        print(f'  отказ после {n * 256} МБ: {type(e).__name__}: {str(e)[:120]}')
    del held; torch.cuda.empty_cache()
    print(f'  свободно после освобождения: {gb(free_now()):.2f} ГБ')

    print('\n2. прямой и обратный проход настоящей модели на случайных данных')
    W = {'y30': 1.0, 'y7': .25, 'y14': .25, 'ord30': .25, 'act30': .25, 'buy': .25}
    for B in a.batch:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        f0 = free_now()
        try:
            base = Transformer(a.n_in, a.channels, a.layers, 0.0, a.heads)
            m = TabFusion(base, a.n_tab, 128).cuda()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler()
            x = torch.randn(B, a.n_in, SEQ_LEN, device='cuda')
            tb = torch.randn(B, a.n_tab, device='cuda')
            y = {k: torch.rand(B, device='cuda') for k in HEADS}
            peak = 0
            for _ in range(3):
                with torch.amp.autocast('cuda'):
                    p = m(x, tb)
                    loss = sum(W[k] * (F.binary_cross_entropy_with_logits(p[k], y[k])
                                       if k == 'buy' else F.mse_loss(p[k], y[k])) for k in W)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                peak = max(peak, torch.cuda.max_memory_allocated())
            print(f'  батч {B:4d}: ОК, пик выделения {gb(peak):.2f} ГБ, '
                  f'свободно было {gb(f0):.2f} ГБ')
        except Exception as e:
            print(f'  батч {B:4d}: ОТКАЗ  {type(e).__name__}: {str(e)[:150]}')
        finally:
            for v in ('m', 'base', 'opt', 'scaler', 'x', 'tb', 'y', 'p', 'loss'):
                locals().pop(v, None)
            torch.cuda.empty_cache()
    print('\nЕсли крупные батчи отказывают, а мелкие проходят — это предел процесса,')
    print('а не карты: под Windows система выдаёт процессу бюджет видеопамяти, и он')
    print('меньше свободного объёма. Тогда рабочий путь — меньший батч.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
