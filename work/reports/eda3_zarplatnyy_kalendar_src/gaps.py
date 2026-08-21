import polars as pl, numpy as np
R='/Users/alexanderkondakov/ozon-cup/'
lf=(pl.scan_parquet(R+'train.parquet')
    .filter((pl.col('to_ord')>0)&(pl.col('event_date')<pl.date(2026,1,15)))
    .select(['user_id','event_date'])
    .filter(pl.col('user_id')%5==0)                       # 20% sample of users
    .sort(['user_id','event_date']))
df=lf.collect(engine='streaming')
print('order-days rows (20% users):',df.height,'users',df['user_id'].n_unique())
g=(df.with_columns(pl.col('event_date').diff().over('user_id').dt.total_days().alias('gap'),
                   pl.col('event_date').dt.day().alias('dom'))
     .drop_nulls('gap').filter(pl.col('gap')>0))
h=g.group_by('gap').agg(pl.len().alias('n')).sort('gap')
arr=np.zeros(70); 
for gp,nn in zip(h['gap'].to_list(),h['n'].to_list()):
    if 1<=gp<70: arr[gp]=nn
sm=np.convolve(arr,np.ones(9)/9,mode='same')
print('gap  count  excess_vs_local_smooth')
for gp in list(range(1,10))+[13,14,15,20,21,22,27,28,29,30,31,32,33,34,35,42,49,56,60,61,62]:
    print('  %3d %8d  %+.3f'%(gp,int(arr[gp]), arr[gp]/max(sm[gp],1)-1))
# calendar-locked monthly check: among gaps 28..32, share landing on the SAME dom as previous order
g2=g.filter((pl.col('gap')>=27)&(pl.col('gap')<=34)).with_columns(
    (pl.col('event_date')-pl.duration(days=pl.col('gap'))).dt.day().alias('prev_dom'))
tab=g2.group_by(pl.col('dom')==pl.col('prev_dom')).agg(pl.len())
print('gaps 27-34: same-dom share =', dict(zip(tab['dom'].to_list(),tab['len'].to_list())))
tot=g2.height; same=g2.filter(pl.col('dom')==pl.col('prev_dom')).height
print('  same-dom %d/%d = %.4f (chance ~ share of gap in {30,31} weighted; random dom match ~1/8=%.3f over 8 gap values)'%(same,tot,same/tot,1/8))
