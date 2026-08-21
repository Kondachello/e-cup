import numpy as np, polars as pl, datetime as dt, json
SP='/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad/'
R='/Users/alexanderkondakov/ozon-cup/'
rng=np.random.default_rng(7)
dg=json.load(open(SP+'dom_global.json')); domf=np.exp(np.array(dg['dom_mean_log'])); wkf=np.exp(np.array(dg['weekday']))
vp=pl.read_parquet(R+'work/preds_pack/val_preds.parquet'); uids=vp['user_id'].to_numpy(); U=len(uids); pos={int(u):i for i,u in enumerate(uids)}
def mat(path,key,K,col):
    t=pl.read_parquet(path); M=np.zeros((U,K)); ui=np.array([pos[int(x)] for x in t['user_id'].to_numpy()]); M[ui,t[key].to_numpy()-1]=t[col].to_numpy(); return M
def vec(path,col):
    t=pl.read_parquet(path); v=np.zeros(U); v[[pos[int(x)] for x in t['user_id'].to_numpy()]]=t[col].to_numpy(); return v
daysA=[x for x in dates if x.year==2025 and x.month%2==1]; daysB=[x for x in dates if x.year==2025 and x.month%2==0]
def Cmat(days):
    C=np.zeros((31,7))
    for x in days: C[x.day-1, x.weekday()]+=1
    return C
CA,CB=Cmat(daysA),Cmat(daysB)
NA=mat(SP+'ud_A2.parquet','dom',31,'n_act'); NB=mat(SP+'ud_B2.parquet','dom',31,'n_act')
DNA=mat(SP+'uw_A2.parquet','dw',7,'n_act'); DNB=mat(SP+'uw_B2.parquet','dw',7,'n_act')
nmoA=vec(SP+'ut_A2.parquet','nmo'); nmoB=vec(SP+'ut_B2.parquet','nmo')
LL=(nmoA>=6)&(nmoB>=6)
def own_expo(DN,C):
    w=DN/np.maximum(DN.sum(1,keepdims=True),1e-9)      # user's own weekday shares
    return domf[None,:]*(w@C.T)                        # U x 31 expected exposure
def prof(S,E):
    p=S/np.maximum(E,1e-9); s=p.sum(1,keepdims=True); return np.where(s>0,p/np.maximum(s,1e-12),0.)
ang=2*np.pi*np.arange(1,32)/30.44
def cc(x,y,m):
    x=x[m]-x[m].mean(); y=y[m]-y[m].mean(); s=np.sqrt((x*x).sum()*(y*y).sum()); return float((x*y).sum()/s) if s>0 else 0.
def perm(S):
    Sp=S.copy()
    for k in range(0,U,50000):
        b=Sp[k:k+50000]; q=np.argsort(rng.random(b.shape),axis=1); Sp[k:k+50000]=np.take_along_axis(b,q,axis=1)
    return Sp
PAY=np.zeros(31,bool); PAY[[0,1,2,9,10,11,14,15,16,24,25,26]]=True
print('=== cos30 persistence: global-weekday exposure vs USER-OWN-weekday exposure (activity profile, long-lived) ===')
for tag,(EAx,EBx) in {'global weekday exposure':(np.broadcast_to(domf*CA.sum(1),(U,31)), np.broadcast_to(domf*CB.sum(1),(U,31))),
                      'own weekday exposure':(own_expo(DNA,CA), own_expo(DNB,CB))}.items():
    pA=prof(NA,EAx); pB=prof(NB,EBx); pBp=prof(perm(NB),EBx)
    for nm,f in [('cos30',lambda p:(p*np.cos(ang)).sum(1)),('sin30',lambda p:(p*np.sin(ang)).sum(1)),
                 ('payday_share',lambda p:p[:,PAY].sum(1)),('d29_31',lambda p:p[:,28:31].sum(1)),('d14_15',lambda p:p[:,13:15].sum(1))]:
        c=cc(f(pA),f(pB),LL); pl_=cc(f(pA),f(pBp),LL)
        print('  %-24s %-13s corr=%+.4f placebo=%+.4f net=%+.4f'%(tag,nm,c,pl_,c-pl_))

# ===== decile analysis: 2025 analog (exact test coverage) =====
da=np.load(SP+'delta_axes.npy'); dA,dB,dL,dLp,D,gQ,gP=[da[:,i] for i in range(7)]
S=np.load(SP+'z_S.npy')
m=(dL!=0)&((gP>0)|(gQ>0))
print('\n=== 2025 ANALOG decile test: mean realized log-shift D by decile of history delta (n=%d) ==='%m.sum())
q=np.quantile(dL[m],np.linspace(0,1,11)); q[0]-=1e-9
lab=np.digitize(dL[m],q[1:-1])
Dm=D[m]
for k in range(10):
    s=lab==k
    print('  dec%2d  delta=[%+.3f..%+.3f] mean_delta=%+.4f  mean_D=%+.4f  SE=%.4f  n=%d'%(k+1,q[k],q[k+1],dL[m][s].mean(),Dm[s].mean(),Dm[s].std()/np.sqrt(s.sum()),s.sum()))
gr=np.corrcoef(dL[m],Dm)[0,1]; print('  overall corr=%+.5f  SE=%.5f  -> |corr| 2SE bound %.5f'%(gr,1/np.sqrt(m.sum()),2/np.sqrt(m.sum())))
print('  MSE-gain ceiling from 2SE bound: %.6f -> RMSLE %.6f'%( (2/np.sqrt(m.sum()))**2*Dm.var(), (2/np.sqrt(m.sum()))**2*Dm.var()/(2*1.6656)))

# ===== val residual: honest 5-fold OOF with dom-coverage feature(s) =====
y=np.log1p(vp['target'].to_numpy()); b=vp['blend'].to_numpy().astype(float); e=y-b
zz=np.load(SP+'zval.npy'); z=zz[:,0]; z_ord=zz[:,1]; z_act=zz[:,2]; z_pl=zz[:,3]
# bounded features on full pre-val history
Sh=np.load(SP+'z_S.npy'); Eh=np.load(SP+'z_E.npy')
ph=prof(Sh,Eh)
feats={'z_val':z,'z_val_ord':z_ord,'z_val_act':z_act,'payday_share':ph[:,PAY].sum(1),'cos30':(ph*np.cos(ang)).sum(1),'d29_31':ph[:,28:31].sum(1),'d14_15':ph[:,13:15].sum(1),'PLACEBO_perm':z_pl}
fold=rng.integers(0,5,U)
base=np.sqrt((e**2).mean())
print('\n=== HONEST 5-fold OOF (by user) on val residual, single feature + intercept ===')
for nm,x in feats.items():
    pred=np.zeros(U)
    for f in range(5):
        tr=fold!=f; te=~tr
        X=np.c_[np.ones(tr.sum()),x[tr]]; beta=np.linalg.lstsq(X,e[tr],rcond=None)[0]
        pred[te]=beta[0]+beta[1]*x[te]
    new=np.sqrt(((e-pred)**2).mean())
    # whale concentration of the gain
    contrib=e**2-(e-pred)**2
    o=np.argsort(-np.abs(contrib))
    tot=contrib.sum()
    top1=contrib[o[:U//100]].sum(); top01=contrib[o[:U//1000]].sum()
    print('  %-14s OOF %.6f (%+.6f)  gain-share top1%%=%.2f top0.1%%=%.2f  corr=%+.5f'%(nm,new,new-base, top1/tot if tot!=0 else np.nan, top01/tot if tot!=0 else np.nan, np.corrcoef(x,e)[0,1]))
