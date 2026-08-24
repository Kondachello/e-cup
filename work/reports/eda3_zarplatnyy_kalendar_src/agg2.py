import polars as pl
R='/Users/alexanderkondakov/ozon-cup/'; SP='/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad/'
lf=(pl.scan_parquet(R+'train.parquet').filter(pl.col('event_date')<pl.date(2026,1,1))   # whole months 2025 only
      .with_columns(pl.col('event_date').dt.day().alias('dom'),
                    pl.col('event_date').dt.month().alias('mo'),
                    pl.col('event_date').dt.weekday().alias('dw')))
for nm,cond in [('A2',(pl.col('mo')%2==1)),('',(pl.col('mo')%2==0))]:
    lf.filter(cond).group_by(['user_id','dom']).agg(pl.col('gmv').sum().alias('s_gmv'), pl.len().alias('n_act')).sink_parquet(SP+f'ud_{nm}.parquet')
    lf.filter(cond).group_by(['user_id','dw']).agg(pl.col('gmv').sum().alias('s_gmv'), pl.len().alias('n_act')).sink_parquet(SP+f'uw_{nm}.parquet')
    lf.filter(cond).group_by('user_id').agg(pl.col('gmv').sum().alias('g'), pl.len().alias('n'), pl.col('mo').n_unique().alias('nmo')).sink_parquet(SP+f'ut_{nm}.parquet')
# months active over 2025 for the long-lived filter
lf.group_by('user_id').agg(pl.col('mo').n_unique().alias('nmo25'), pl.col('gmv').sum().alias('g25'), pl.len().alias('n25')).sink_parquet(SP+'umo25.parquet')
print('ok')
