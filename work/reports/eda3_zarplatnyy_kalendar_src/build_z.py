import numpy as np, polars as pl, datetime as dt, json
SP='/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad/'
R='/Users/alexanderkondakov/ozon-cup/'
dates = d['event_date'].to_list(); G_all = d['gmv'].to_numpy().astype(float)
idx = {x:i for i,x in enumerate(dates)}
HIST_END = dt.date(2026,1,14); VAL0=dt.date(2026,1,15); VAL1=dt.date(2026,2,13)
hi_end = idx[HIST_END]
hdates = dates[:hi_end+1]; Gh = G_all[:hi_end+1]
hdom = np.array([x.day for x in hdates])
nH = len(hdates); print('history days', nH, hdates[0], hdates[-1])

# cumulative exposure per dom over history
M = np.zeros((nH,31)); M[np.arange(nH), hdom-1] = Gh
cumM = np.cumsum(M, axis=0)                       # (nH,31)
Mu = np.zeros((nH,31)); Mu[np.arange(nH), hdom-1] = 1.0
cumMu = np.cumsum(Mu, axis=0)

# users, aligned to val_preds order
vp = pl.read_parquet(R+'work/preds_pack/val_preds.parquet', columns=['user_id','target','blend'])
uids = vp['user_id'].to_numpy(); U=len(uids); pos = {int(u):i for i,u in enumerate(uids)}
ud = pl.read_parquet(SP+'user_dom_hist.parquet')
ui = np.array([pos[int(x)] for x in ud['user_id'].to_numpy()])
S = np.zeros((U,31)); S[ui, ud['dom'].to_numpy()-1] = ud['s_gmv'].to_numpy()
NA = np.zeros((U,31)); NA[ui, ud['dom'].to_numpy()-1] = ud['n_act'].to_numpy()
OR = np.zeros((U,31)); OR[ui, ud['dom'].to_numpy()-1] = ud['s_ord'].to_numpy()
uf = pl.read_parquet(SP+'user_first.parquet')
fpos = np.array([pos[int(x)] for x in uf['user_id'].to_numpy()])
fidx = np.zeros(U, dtype=int); fidx[fpos] = np.array([idx[x] for x in uf['first_dt'].to_list()])
gmv_tot = np.zeros(U); gmv_tot[fpos] = uf['gmv_tot'].to_numpy()
act_days = np.zeros(U); act_days[fpos] = uf['act_days'].to_numpy()
ord_tot = np.zeros(U); ord_tot[fpos] = uf['ord_tot'].to_numpy()
last_idx = np.zeros(U,dtype=int); last_idx[fpos] = np.array([idx[x] for x in uf['last_dt'].to_list()])
print('users', U, 'with hist rows', len(np.unique(ui)))

def exposure(cum, f):
    hi = cum[-1][None,:]
    lo = np.where(f[:,None]>0, cum[np.maximum(f-1,0)], 0.0)
    return hi-lo
E  = exposure(cumM, fidx)      # gmv-weighted exposure per dom
Eu = exposure(cumMu, fidx)     # day-count exposure per dom
np.save(SP+'z_S.npy',S); np.save(SP+'z_E.npy',E); np.save(SP+'z_Eu.npy',Eu)
np.save(SP+'z_meta.npy', np.c_[gmv_tot, act_days, ord_tot, fidx, last_idx])

def window_days(a,b):
    ds=[]; x=a
    while x<=b: ds.append(x); x+=dt.timedelta(days=1)
    return ds
val_days = window_days(VAL0,VAL1)
Gval = np.array([G_all[idx[x]] for x in val_days]); vdom=np.array([x.day for x in val_days])
print('val days',len(val_days),'dom coverage', np.bincount(vdom,minlength=32)[1:].tolist())
test_days = window_days(dt.date(2026,2,14), dt.date(2026,3,15))
tdom = np.array([x.day for x in test_days])
print('test days',len(test_days),'dom coverage', np.bincount(tdom,minlength=32)[1:].tolist())
json.dump({'val_dom':np.bincount(vdom,minlength=32)[1:].tolist(),'test_dom':np.bincount(tdom,minlength=32)[1:].tolist()}, open(SP+'wincov.json','w'))
