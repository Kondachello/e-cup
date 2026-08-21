"""Part A: intra-window GMV shape from daily calendar (410 days)."""
import polars as pl
import numpy as np
import json

dates = cal['event_date'].to_list()
g = cal['gmv_sum'].to_numpy()
buyers = cal['gmv_nnz'].to_numpy()
import datetime as dt
d0 = dt.date(2025, 1, 1)
day_idx = np.array([(d - d0).days for d in dates])
assert (day_idx == np.arange(len(day_idx))).all(), 'gaps in calendar'
N = len(g)  # 409? 410

def win_stats(anchor_date):
    """window = anchor+1 .. anchor+30; returns front/tail sums etc."""
    a = (anchor_date - d0).days
    idx = np.arange(a + 1, a + 31)
    if idx[-1] >= N:
        return None
    w = g[idx]
    b = buyers[idx]
    front, tail = w[:15].sum(), w[15:].sum()
    return dict(anchor=str(anchor_date), total=float(w.sum()),
                front=float(front), tail=float(tail),
                tail_share=float(tail / (front + tail)),
                buyers_front=int(b[:15].sum()), buyers_tail=int(b[15:].sum()),
                peak_day=str(dates[idx[np.argmax(w)]]), peak_val=float(w.max()),
                daily=[float(x) for x in w])

# rolling tail-share over all anchors
rows = []
for a in range(0, N - 30):
    w = g[a + 1:a + 31]
    rows.append((str(dates[a]), w[15:].sum() / w.sum(), w.sum()))
ts = np.array([r[1] for r in rows])
print(f'tail_share over {len(rows)} anchors: mean {ts.mean():.4f}  std {ts.std():.4f}  p5 {np.percentile(ts,5):.4f}  p95 {np.percentile(ts,95):.4f}')

key_anchors = {
    'val_actual_2026-01-14': dt.date(2026, 1, 14),
    'val_analog_2025-01-14': dt.date(2025, 1, 14),
    'test_analog_2025-02-13': dt.date(2025, 2, 13),
    'ctrl_2025-04-30': dt.date(2025, 4, 30),
    'ctrl_2025-06-30': dt.date(2025, 6, 30),
    'ctrl_2025-09-30': dt.date(2025, 9, 30),
}
out = {}
for k, ad in key_anchors.items():
    s = win_stats(ad)
    out[k] = s
    if s:
        print(f"{k}: total {s['total']/1e6:.2f}M  tail_share {s['tail_share']:.4f}  "
              f"buyersF {s['buyers_front']}  buyersT {s['buyers_tail']}  peak {s['peak_day']} ({s['peak_val']/1e3:.0f}k)")

# where does test-analog tail_share sit in the anchor distribution?
for k in ['val_actual_2026-01-14', 'test_analog_2025-02-13']:
    v = out[k]['tail_share']
    pct = (ts < v).mean() * 100
    print(f'{k}: tail_share {v:.4f} = percentile {pct:.1f} of all anchors')

# daily shape of test-analog window: normalized daily share, mark holidays
s = out['test_analog_2025-02-13']
daily = np.array(s['daily'])
shares = daily / daily.sum()
wd = [dates[(dt.date(2025,2,13)-d0).days + 1 + i] for i in range(30)]
print('\ntest-analog 2025 window daily GMV share (%):')
for i, (d, sh) in enumerate(zip(wd, shares)):
    mark = ''
    if str(d) == '2025-02-14': mark = ' <-- 14 Feb'
    if str(d) == '2025-02-23': mark = ' <-- 23 Feb'
    if str(d) in ('2025-03-05','2025-03-06','2025-03-07'): mark = ' <-- pre-8Mar'
    if str(d) == '2025-03-08': mark = ' <-- 8 Mar'
    print(f'  day {i+1:2d} {d} {sh*100:5.2f}%{mark}')

# same for val actual window
s = out['val_actual_2026-01-14']
daily = np.array(s['daily'])
print(f"\nval actual window: day-share max {daily.max()/daily.sum()*100:.2f}%  min {daily.min()/daily.sum()*100:.2f}%")

# per-buyer average check: is the tail-share driven by participation or basket?
json.dump(out, open('/Users/alexanderkondakov/ozon-cup/work/reports/eda3_window_shape.json', 'w'), indent=1)
print('\nsaved work/reports/eda3_window_shape.json')
