"""One streaming pass: per-user F/T targets + 30d pre-anchor features at 5 anchors.
Val anchor additionally: richer features, 10-day slices, first/last gmv day in window."""
import polars as pl
import datetime as dt

D = lambda s: pl.lit(dt.date.fromisoformat(s))
ed = pl.col('event_date')

anchors = {
    'ta': '2025-02-13',   # test-analog
    'c4': '2025-04-30',
    'c6': '2025-06-30',
    'c9': '2025-09-30',
    'va': '2026-01-14',   # val (real residual available)
}

aggs = []
for tag, a in anchors.items():
    ad = dt.date.fromisoformat(a)
    f0, f1 = ad + dt.timedelta(1), ad + dt.timedelta(15)     # front 15d
    t0, t1 = ad + dt.timedelta(16), ad + dt.timedelta(30)    # tail 15d
    h0 = ad - dt.timedelta(29)                                # 30d history
    inF = ed.is_between(pl.lit(f0), pl.lit(f1))
    inT = ed.is_between(pl.lit(t0), pl.lit(t1))
    inH = ed.is_between(pl.lit(h0), pl.lit(ad))
    aggs += [
        pl.col('gmv').filter(inF).sum().alias(f'{tag}_F'),
        pl.col('gmv').filter(inT).sum().alias(f'{tag}_T'),
        pl.col('gmv').filter(inH).sum().alias(f'{tag}_gmv30'),
        pl.col('to_ord').filter(inH).sum().alias(f'{tag}_ord30'),
        pl.col('to_cart').filter(inH).sum().alias(f'{tag}_cart30'),
        pl.col('searches').filter(inH).sum().alias(f'{tag}_srch30'),
        inH.sum().alias(f'{tag}_act30'),
        ed.filter(ed <= pl.lit(ad)).max().alias(f'{tag}_lastact'),
        ed.filter((ed <= pl.lit(ad)) & (pl.col('gmv') > 0)).max().alias(f'{tag}_lastgmv'),
    ]

# val-anchor extras
va = dt.date.fromisoformat(anchors['va'])
w0 = va + dt.timedelta(1)
inW = ed.is_between(pl.lit(w0), pl.lit(va + dt.timedelta(30)))
h90 = ed.is_between(pl.lit(va - dt.timedelta(89)), pl.lit(va))
aggs += [
    pl.col('gmv').filter(h90).sum().alias('va_gmv90'),
    pl.col('to_ord').filter(h90).sum().alias('va_ord90'),
    pl.col('gmv').filter(ed <= pl.lit(va)).sum().alias('va_gmv365'),
    pl.col('to_ord').filter(ed <= pl.lit(va)).sum().alias('va_ord365'),
    # window micro-structure (future info, diagnostics only)
    ed.filter(inW & (pl.col('gmv') > 0)).min().alias('va_first_gmv_date'),
    ed.filter(inW & (pl.col('gmv') > 0)).max().alias('va_last_gmv_date'),
    (inW & (pl.col('gmv') > 0)).sum().alias('va_n_gmv_days'),
    pl.col('gmv').filter(ed.is_between(pl.lit(w0), pl.lit(va + dt.timedelta(10)))).sum().alias('va_S1'),
    pl.col('gmv').filter(ed.is_between(pl.lit(va + dt.timedelta(11)), pl.lit(va + dt.timedelta(20)))).sum().alias('va_S2'),
    pl.col('gmv').filter(ed.is_between(pl.lit(va + dt.timedelta(21)), pl.lit(va + dt.timedelta(30)))).sum().alias('va_S3'),
    # ta-anchor slices too (holiday in tail there)
]
ta = dt.date.fromisoformat(anchors['ta'])
aggs += [
    pl.col('gmv').filter(ed.is_between(pl.lit(ta + dt.timedelta(1)), pl.lit(ta + dt.timedelta(10)))).sum().alias('ta_S1'),
    pl.col('gmv').filter(ed.is_between(pl.lit(ta + dt.timedelta(11)), pl.lit(ta + dt.timedelta(20)))).sum().alias('ta_S2'),
    pl.col('gmv').filter(ed.is_between(pl.lit(ta + dt.timedelta(21)), pl.lit(ta + dt.timedelta(30)))).sum().alias('ta_S3'),
]

lf = pl.scan_parquet('/Users/alexanderkondakov/ozon-cup/train.parquet')
df = lf.group_by('user_id').agg(aggs).collect(engine='streaming')
print(df.shape)
df.write_parquet('/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad/user_windows.parquet')
print('done')
