import polars as pl
R='/Users/alexanderkondakov/ozon-cup/'; SP='/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad/'
lf=pl.scan_parquet(R+'train.parquet').with_columns(pl.col('event_date').dt.day().alias('dom'), pl.col('event_date').dt.month().alias('mo'), pl.col('event_date').dt.year().alias('yr'))
# half A = odd months, half B = even months, both restricted to history < 2026-01-15
h=lf.filter(pl.col('event_date')<pl.date(2026,1,15))
for nm,cond in [('A',(pl.col('mo')%2==1)),('B',(pl.col('mo')%2==0))]:
    (h.filter(cond).group_by(['user_id','dom']).agg(pl.col('gmv').sum().alias('s_gmv'), pl.len().alias('n_act'))
      ).sink_parquet(SP+f'user_dom_{nm}.parquet')
# windows for 2025 analog test: Q=2025-01-15..02-13 (val-analog), P=2025-02-14..03-15 (test-analog)
for nm,a,b in [('Q','2025-01-15','2025-02-13'),('P','2025-02-14','2025-03-15')]:
    (lf.filter((pl.col('event_date')>=pl.lit(a).str.to_date())&(pl.col('event_date')<=pl.lit(b).str.to_date()))
       .group_by('user_id').agg(pl.col('gmv').sum().alias('gmv'), pl.len().alias('act'))
    ).sink_parquet(SP+f'win_{nm}.parquet')
# profile source for the 2025-analog mechanism test: 2025-03-16..2026-01-14 (disjoint from P and Q)
(lf.filter((pl.col('event_date')>=pl.date(2025,3,16))&(pl.col('event_date')<pl.date(2026,1,15)))
   .group_by(['user_id','dom']).agg(pl.col('gmv').sum().alias('s_gmv'), pl.len().alias('n_act'))
).sink_parquet(SP+'user_dom_L.parquet')
print('ok')
