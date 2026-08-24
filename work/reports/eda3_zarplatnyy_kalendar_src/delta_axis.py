import numpy as np, polars as pl, datetime as dt, json
SP='/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad/'
R='/Users/alexanderkondakov/ozon-cup/'
rng=np.random.default_rng(1)
S=np.load(SP+'z_S.npy'); E=np.load(SP+'z_E.npy'); U=S.shape[0]
dg=json.load(open(SP+'dom_global.json')); domf=np.exp(np.array(dg['dom_mean_log'])); wkf=np.exp(np.array(dg['weekday']))
dates=d['event_date'].to_list(); G_all=d['gmv'].to_numpy().astype(float); idx={x:i for i,x in enumerate(dates)}
def wdays(a,b):
    o=[];x=a
    while x<=b: o.append(x); x+=dt.timedelta(days=1)
    return o
def wvec(days, mode='model'):
    w=np.zeros(31)
    for x in days:
        g = G_all[idx[x]] if (mode=='actual' and x in idx) else domf[x.day-1]*wkf[x.weekday()]
        w[x.day-1]+=g
    return w/w.sum()
val_days=wdays(dt.date(2026,1,15),dt.date(2026,2,13)); test_days=wdays(dt.date(2026,2,14),dt.date(2026,3,15))
wv=wvec(val_days); wt=wvec(test_days)
# aggregate (population-level) coverage effect
print('AGG coverage: test/val level ratio from dom composition = %.5f (%.3f%%)'%( (wt*domf).sum()/(wv*domf).sum()*0+ np.dot(wt,domf)/np.dot(wv,domf), 100*(np.dot(wt,domf)/np.dot(wv,domf)-1)))
uni_v=np.bincount([x.day for x in val_days],minlength=32)[1:]/30.0
uni_t=np.bincount([x.day for x in test_days],minlength=32)[1:]/30.0
print('AGG coverage (counts x global dom factor): %.5f'%(np.dot(uni_t,domf)/np.dot(uni_v,domf)))

def zvec(S,E,w):
    tot=S.sum(1); Et=E.sum(1)
    abar=np.where(tot>0, tot/np.maximum(Et,1e-9),0.0)
    a=np.where(E>0,S/np.maximum(E,1e-9),abar[:,None])
    num=(a*w[None,:]).sum(1); den=abar*w.sum()
    return np.where((tot>0)&(den>0), np.log(np.maximum(num,1e-12)/np.maximum(den,1e-12)),0.0)
zv=zvec(S,E,wv); zt=zvec(S,E,wt); delta=zt-zv
print('delta=z_test-z_val: mean %+.4f sd %.4f  q[1,5,25,50,75,95,99]=%s'%(delta.mean(),delta.std(),np.round(np.quantile(delta,[.01,.05,.25,.5,.75,.95,.99]),4)))
buy=S.sum(1)>0
print('  on gmv>0 (n=%d): mean %+.4f sd %.4f'%(buy.sum(),delta[buy].mean(),delta[buy].std()))
hi=S.sum(1)>np.quantile(S.sum(1)[buy],0.5)
print('  on top-50%% spenders: sd %.4f ; on bottom: sd %.4f'%(delta[hi].std(),delta[buy&~hi].std()))

# ---- persistence: profile from disjoint halves of history ----
ud=pl.read_parquet(SP+'user_dom_hist.parquet')  # full-history; need split -> recompute
