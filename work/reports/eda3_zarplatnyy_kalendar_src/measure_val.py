import numpy as np, polars as pl, datetime as dt, json
SP='/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad/'
R='/Users/alexanderkondakov/ozon-cup/'
rng=np.random.default_rng(0)
S=np.load(SP+'z_S.npy'); E=np.load(SP+'z_E.npy'); Eu=np.load(SP+'z_Eu.npy')
meta=np.load(SP+'z_meta.npy'); gmv_tot,act_days,ord_tot,fidx,last_idx = [meta[:,i] for i in range(5)]
dates=d['event_date'].to_list(); G_all=d['gmv'].to_numpy().astype(float); idx={x:i for i,x in enumerate(dates)}
vp = pl.read_parquet(R+'work/preds_pack/val_preds.parquet', columns=['user_id','target','blend'])
y=np.log1p(vp['target'].to_numpy()); b=vp['blend'].to_numpy().astype(float); e=y-b; U=len(y)
print('base RMSLE %.6f  users %d  gmv_tot>0: %d (%.1f%%)'%(np.sqrt((e**2).mean()),U,(gmv_tot>0).sum(),100*(gmv_tot>0).mean()))

def wdays(a,b):
    o=[];x=a
    while x<=b: o.append(x); x+=dt.timedelta(days=1)
    return o
val_days=wdays(dt.date(2026,1,15),dt.date(2026,2,13))
Gval=np.array([G_all[idx[x]] for x in val_days]); vdom=np.array([x.day for x in val_days])
wv=np.zeros(31)
for g,dd in zip(Gval,vdom): wv[dd-1]+=g
wv/=wv.sum()

def zscore(S,E,w):
    tot=S.sum(1); Et=E.sum(1)
    abar=np.where(tot>0, tot/np.maximum(Et,1e-9), 0.0)
    a=np.where(E>0, S/np.maximum(E,1e-9), abar[:,None])
    num=(a*w[None,:]).sum(1); den=abar*w.sum()
    z=np.where((tot>0)&(den>0), np.log(np.maximum(num,1e-12)/np.maximum(den,1e-12)), 0.0)
    return z, a, abar
z,a,abar = zscore(S,E,wv)
print('z_val: sd %.4f  mean %.4f  q[1,5,50,95,99] %s'%(z.std(),z.mean(),np.round(np.quantile(z,[.01,.05,.5,.95,.99]),4)))
active = gmv_tot>0
print('z_val on gmv_tot>0 (n=%d): sd %.4f'%(active.sum(), z[active].std()))

def report(name, x, e, mask=None):
    m = np.ones(len(x),bool) if mask is None else mask
    xx=x[m]-x[m].mean(); ee=e[m]
    if xx.std()<1e-12: print(name,'degenerate'); return
    r=float(np.corrcoef(xx,ee)[0,1]); sl=float((xx*ee).sum()/(xx*xx).sum())
    gain = sl*sl*(xx*xx).sum()/len(e)   # MSE reduction over FULL population
    base=np.sqrt((e**2).mean()); new=np.sqrt(max((e**2).mean()-gain,0))
    print('%-22s n=%6d corr=%+.5f slope=%+.4f  dMSE=%.6f  RMSLE %.6f->%.6f (%+.6f)'%(name,m.sum(),r,sl,gain,base,new,new-base))
    return sl

sl=report('e ~ z_val', z, e)
sl=report('e ~ z_val (gmv>0)', z, e, active)
# orders-based profile
z_o,_,_ = zscore(np.load(SP+'z_S.npy')*0+0, E, wv) # placeholder
import numpy as _np
ud = pl.read_parquet(SP+'user_dom_hist.parquet')
uids=vp['user_id'].to_numpy(); pos={int(u):i for i,u in enumerate(uids)}
ui=np.array([pos[int(x)] for x in ud['user_id'].to_numpy()])
ORD=np.zeros_like(S); ORD[ui, ud['dom'].to_numpy()-1]=ud['s_ord'].to_numpy()
ACT=np.zeros_like(S); ACT[ui, ud['dom'].to_numpy()-1]=ud['n_act'].to_numpy()
z_ord,_,_=zscore(ORD,E,wv); z_act,_,_=zscore(ACT,Eu,wv)
report('e ~ z_val(orders)', z_ord, e)
report('e ~ z_val(activity)', z_act, e)
# PLACEBO: permute dom labels within each user
Sp=S.copy()
for k in range(0,U,50000):
    blk=Sp[k:k+50000]
    p=np.argsort(rng.random(blk.shape),axis=1)
    Sp[k:k+50000]=np.take_along_axis(blk,p,axis=1)
z_p,_,_=zscore(Sp,E,wv)
report('e ~ z_PLACEBO(perm)', z_p, e)
np.save(SP+'zval.npy', np.c_[z,z_ord,z_act,z_p])
json.dump({'wv':wv.tolist()}, open(SP+'wv.json','w'))
