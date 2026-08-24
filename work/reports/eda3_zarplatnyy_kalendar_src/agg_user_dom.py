import polars as pl, time
t0=time.time()
lf = pl.scan_parquet('/Users/alexanderkondakov/ozon-cup/train.parquet')
VAL0 = pl.date(2026,1,15)
hist = lf.filter(pl.col('event_date') < VAL0)
out = (hist.with_columns(pl.col('event_date').dt.day().alias('dom'))
          .group_by(['user_id','dom'])
          .agg(pl.col('gmv').sum().alias('s_gmv'),
               pl.col('to_ord').sum().alias('s_ord'),
               pl.len().alias('n_act')))
out.sink_parquet('/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad/user_dom_hist.parquet')
print('agg done', round(time.time()-t0,1))
# per-user first active date in history + totals
f = (hist.group_by('user_id').agg(pl.col('event_date').min().alias('first_dt'),
                                  pl.col('event_date').max().alias('last_dt'),
                                  pl.col('gmv').sum().alias('gmv_tot'),
                                  pl.len().alias('act_days'),
                                  pl.col('to_ord').sum().alias('ord_tot')))
f.sink_parquet('/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad/user_first.parquet')
print('first done', round(time.time()-t0,1))
