import numpy as np, polars as pl, datetime as dt, json
SP='/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad/'
R='/Users/alexanderkondakov/ozon-cup/'
rng=np.random.default_rng(2)
dg=json.load(open(SP+'dom_global.json')); domf=np.exp(np.array(dg['dom_mean_log'])); wkf=np.exp(np.array(dg['weekday']))
dates=d['event_date'].to_list(); idx={x:i for i,x in enumerate(dates)}; G_all=d['gmv'].to_numpy().astype(float)
vp=pl.read_parquet(R+'work/preds_pack/val_preds.parquet', columns=['user_id','target','blend'])
uids=vp['user_id'].to_numpy(); U=len(uids); pos={int(u):i for i,u in enumerate(uids)}
def wdays(a,b):
    o=[];x=a
    while x<=b: o.append(x); x+=dt.timedelta(days=1)
    return o
def wvec(days):
    w=np.zeros(31)
    for x in days: w[x.day-1]+=domf[x.day-1]*wkf[x.weekday()]
    return w/w.sum()
wv26=wvec(wdays(dt.date(2026,1,15),dt.date(2026,2,13))); wt26=wvec(wdays(dt.date(2026,2,14),dt.date(2026,3,15)))
wv25=wvec(wdays(dt.date(2025,1,15),dt.date(2025,2,13))); wt25=wvec(wdays(dt.date(2025,2,14),dt.date(2025,3,15)))
print('coverage check 2025 pair == 2026 pair:', np.allclose(np.bincount([x.day for x in wdays(dt.date(2025,2,14),dt.date(2025,3,15))],minlength=32)[1:], np.bincount([x.day for x in wdays(dt.date(2026,2,14),dt.date(2026,3,15))],minlength=32)[1:]))

def load_mat(path, col='s_gmv'):
    t=pl.read_parquet(path)
    M=np.zeros((U,31)); ui=np.array([pos[int(x)] for x in t['user_id'].to_numpy()])
    M[ui, t['dom'].to_numpy()-1]=t[col].to_numpy(); return M
def expo(days_range, per_user_start=None):
    # global day-weighted exposure per dom over a fixed calendar range (same for all users)
    w=np.zeros(31)
    for x in days_range: w[x.day-1]+=domf[x.day-1]*wkf[x.weekday()]
    return w
def zvec(S,Ev,w):
    tot=S.sum(1); Et=(Ev[None,:]*np.ones((1,31))).sum()
    abar=np.where(tot>0, tot/Et,0.0)
    a=np.where(Ev[None,:]>0, S/np.maximum(Ev[None,:],1e-9), abar[:,None])
    num=(a*w[None,:]).sum(1); den=abar*w.sum()
    return np.where((tot>0)&(den>0), np.log(np.maximum(num,1e-12)/np.maximum(den,1e-12)),0.0)

# ---------- 1. persistence of the delta axis across disjoint halves ----------
SA=load_mat(SP+'user_dom_A.parquet'); SB=load_mat(SP+'user_dom_B.parquet')
daysA=[x for x in dates if x<dt.date(2026,1,15) and x.month%2==1]
daysB=[x for x in dates if x<dt.date(2026,1,15) and x.month%2==0]
EA=expo(daysA); EB=expo(daysB)
dA=zvec(SA,EA,wt26)-zvec(SA,EA,wv26); dB=zvec(SB,EB,wt26)-zvec(SB,EB,wv26)
both=(SA.sum(1)>0)&(SB.sum(1)>0)
def cc(x,y,m): 
    x=x[m]-x[m].mean(); y=y[m]-y[m].mean(); return float((x*y).sum()/np.sqrt((x*x).sum()*(y*y).sum()))
print('PERSISTENCE delta(A) vs delta(B): n=%d corr=%+.4f  sdA=%.3f sdB=%.3f'%(both.sum(),cc(dA,dB,both),dA[both].std(),dB[both].std()))
# placebo: permute doms within user in half B
SBp=SB.copy()
for k in range(0,U,50000):
    blk=SBp[k:k+50000]; p=np.argsort(rng.random(blk.shape),axis=1); SBp[k:k+50000]=np.take_along_axis(blk,p,axis=1)
dBp=zvec(SBp,EB,wt26)-zvec(SBp,EB,wv26)
print('PLACEBO   delta(A) vs delta(Bperm): corr=%+.4f'%cc(dA,dBp,both))
# by spend decile
tot=SA.sum(1)+SB.sum(1)
for lo,hi,nm in [(0.0,0.5,'bottom50%'),(0.5,0.9,'50-90%'),(0.9,0.99,'90-99%'),(0.99,1.0,'top1%')]:
    q1,q2=np.quantile(tot[both],[lo,hi]); m=both&(tot>=q1)&(tot<=q2)
    print('   %-10s n=%6d corr(dA,dB)=%+.4f  placebo=%+.4f  sd(dA)=%.3f'%(nm,m.sum(),cc(dA,dB,m),cc(dA,dBp,m),dA[m].std()))

# ---------- 2. 2025 analog: realized window pair vs profile from disjoint later history ----------
SL=load_mat(SP+'user_dom_L.parquet')
daysL=[x for x in dates if dt.date(2025,3,16)<=x<dt.date(2026,1,15)]
EL=expo(daysL)
dL=zvec(SL,EL,wt25)-zvec(SL,EL,wv25)
wq=pl.read_parquet(SP+'win_Q.parquet'); wp=pl.read_parquet(SP+'win_P.parquet')
gQ=np.zeros(U); gP=np.zeros(U)
gQ[[pos[int(x)] for x in wq['user_id'].to_numpy()]]=wq['gmv'].to_numpy()
gP[[pos[int(x)] for x in wp['user_id'].to_numpy()]]=wp['gmv'].to_numpy()
D=np.log1p(gP)-np.log1p(gQ)
m=(SL.sum(1)>0)&((gP>0)|(gQ>0))
print('\n2025 ANALOG PAIR (same dom coverage as val->test). n=%d  mean D=%+.4f sd D=%.4f'%(m.sum(),D[m].mean(),D[m].std()))
print('  sd(dL)=%.4f on mask'%dL[m].std())
x=dL[m]-dL[m].mean(); yv=D[m]-D[m].mean()
sl=float((x*yv).sum()/(x*x).sum()); r=cc(dL,D,m)
print('  regression D ~ delta_hist: slope=%+.4f corr=%+.5f  mdl_flint=%.6f'%(sl,r,r*r))
# placebo profile
SLp=SL.copy()
for k in range(0,U,50000):
    blk=SLp[k:k+50000]; p=np.argsort(rng.random(blk.shape),axis=1); SLp[k:k+50000]=np.take_along_axis(blk,p,axis=1)
dLp=zvec(SLp,EL,wt25)-zvec(SLp,EL,wv25)
xp=dLp[m]-dLp[m].mean(); slp=float((xp*yv).sum()/(xp*xp).sum())
print('  PLACEBO slope=%+.4f corr=%+.5f'%(slp,cc(dLp,D,m)))
# restrict to buyers in both windows (less zero-inflation noise)
m2=m&(gP>0)&(gQ>0)
x2=dL[m2]-dL[m2].mean(); y2=D[m2]-D[m2].mean()
print('  buyers-in-both n=%d: slope=%+.4f corr=%+.5f'%(m2.sum(), float((x2*y2).sum()/(x2*x2).sum()), cc(dL,D,m2)))
np.save(SP+'delta_axes.npy', np.c_[dA,dB,dL,dLp,D,gQ,gP])
