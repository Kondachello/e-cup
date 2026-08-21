import numpy as np, polars as pl, datetime as dt, json
SP='/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad/'
R='/Users/alexanderkondakov/ozon-cup/'
rng=np.random.default_rng(5)
dg=json.load(open(SP+'dom_global.json')); domf=np.exp(np.array(dg['dom_mean_log'])); wkf=np.exp(np.array(dg['weekday']))
vp=pl.read_parquet(R+'work/preds_pack/val_preds.parquet', columns=['user_id']); uids=vp['user_id'].to_numpy(); U=len(uids)
pos={int(u):i for i,u in enumerate(uids)}
def mat(path,keycol,K,col):
    t=pl.read_parquet(path); M=np.zeros((U,K)); ui=np.array([pos[int(x)] for x in t['user_id'].to_numpy()])
    M[ui,t[keycol].to_numpy()-1]=t[col].to_numpy(); return M
def vec(path,col):
    t=pl.read_parquet(path); v=np.zeros(U); v[[pos[int(x)] for x in t['user_id'].to_numpy()]]=t[col].to_numpy(); return v
daysA=[x for x in dates if x.year==2025 and x.month%2==1]; daysB=[x for x in dates if x.year==2025 and x.month%2==0]
def expo_dom(days):
    w=np.zeros(31)
    for x in days: w[x.day-1]+=domf[x.day-1]*wkf[x.weekday()]
    return w
def expo_dw(days):
    w=np.zeros(7)
    for x in days: w[x.weekday()]+=domf[x.day-1]*wkf[x.weekday()]
    return w
EA,EB=expo_dom(daysA),expo_dom(daysB); WA,WB=expo_dw(daysA),expo_dw(daysB)
SA=mat(SP+'ud_A2.parquet','dom',31,'s_gmv'); SB=mat(SP+'ud_B2.parquet','dom',31,'s_gmv')
NA=mat(SP+'ud_A2.parquet','dom',31,'n_act'); NB=mat(SP+'ud_B2.parquet','dom',31,'n_act')
DA=mat(SP+'uw_A2.parquet','dw',7,'s_gmv');  DB=mat(SP+'uw_B2.parquet','dw',7,'s_gmv')
DNA=mat(SP+'uw_A2.parquet','dw',7,'n_act'); DNB=mat(SP+'uw_B2.parquet','dw',7,'n_act')
gA=vec(SP+'ut_A2.parquet','g'); gB=vec(SP+'ut_B2.parquet','g')
nmoA=vec(SP+'ut_A2.parquet','nmo'); nmoB=vec(SP+'ut_B2.parquet','nmo')
nA=vec(SP+'ut_A2.parquet','n'); nB=vec(SP+'ut_B2.parquet','n')
cntA=np.bincount([x.day for x in daysA],minlength=32)[1:].astype(float); cntB=np.bincount([x.day for x in daysB],minlength=32)[1:].astype(float)
cwA=np.bincount([x.weekday() for x in daysA],minlength=7).astype(float); cwB=np.bincount([x.weekday() for x in daysB],minlength=7).astype(float)
def prof(S,Ev):
    p=S/np.maximum(Ev[None,:],1e-9); s=p.sum(1,keepdims=True); return np.where(s>0,p/np.maximum(s,1e-12),0.0)
def perm(S):
    Sp=S.copy()
    for k in range(0,U,50000):
        blk=Sp[k:k+50000]; q=np.argsort(rng.random(blk.shape),axis=1); Sp[k:k+50000]=np.take_along_axis(blk,q,axis=1)
    return Sp
def cc(x,y,m):
    x=x[m]-x[m].mean(); y=y[m]-y[m].mean(); s=np.sqrt((x*x).sum()*(y*y).sum()); return float((x*y).sum()/s) if s>0 else 0.
PAY=np.zeros(31,bool); PAY[[0,1,2,9,10,11,14,15,16,24,25,26]]=True
ang=2*np.pi*np.arange(1,32)/30.44
def dom_stats(p): return {'payday_share':p[:,PAY].sum(1),'cos30':(p*np.cos(ang)).sum(1),'sin30':(p*np.sin(ang)).sum(1),'d29_31':p[:,28:31].sum(1),'d14_15':p[:,13:15].sum(1)}
angw=2*np.pi*np.arange(7)/7.
def dw_stats(p): return {'weekend_share':p[:,5:7].sum(1),'cos7':(p*np.cos(angw)).sum(1),'sin7':(p*np.sin(angw)).sum(1)}
LL=(nmoA>=6)&(nmoB>=6)                                  # active in all 6 odd and all 6 even months
print('long-lived (active in all 12 months of 2025): n=%d'%LL.sum())
print('POSITIVE CONTROL corr(log1p gmv A, B) on long-lived: %+.4f ; on all-both: %+.4f'%(cc(np.log1p(gA),np.log1p(gB),LL), cc(np.log1p(gA),np.log1p(gB),(gA>0)&(gB>0))))
print('POSITIVE CONTROL corr(log1p act-days A, B) long-lived: %+.4f'%cc(np.log1p(nA),np.log1p(nB),LL))
for lbl,(PA,PB,fA,fB,statfn) in {
   'DOM gmv-profile':(prof(SA,EA),prof(SB,EB),SA,SB,dom_stats),
   'DOM activity-profile':(prof(NA,cntA),prof(NB,cntB),NA,NB,dom_stats),
   'WEEKDAY gmv-profile':(prof(DA,WA),prof(DB,WB),DA,DB,dw_stats),
   'WEEKDAY activity-profile':(prof(DNA,cwA),prof(DNB,cwB),DNA,DNB,dw_stats)}.items():
    sa=statfn(PA); sb=statfn(PB); sbp=statfn(prof(perm(fB), EB if 'DOM' in lbl else WB) if 'gmv' in lbl else prof(perm(fB), cntB if 'DOM' in lbl else cwB))
    m=LL&(fA.sum(1)>0)&(fB.sum(1)>0)
    print('-- %s  n=%d'%(lbl,m.sum()))
    for k in sa:
        c=cc(sa[k],sb[k],m); p=cc(sa[k],sbp[k],m)
        print('   %-14s corr=%+.4f placebo=%+.4f net=%+.4f  sd=%.4f  (SE~%.4f)'%(k,c,p,c-p,sa[k][m].std(),1/np.sqrt(m.sum())))
