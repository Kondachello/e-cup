import numpy as np, polars as pl, datetime as dt, json
SP='/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad/'
R='/Users/alexanderkondakov/ozon-cup/'
dates=d['event_date'].to_list(); n=len(dates)
dom=np.array([x.day for x in dates]); dw=np.array([x.weekday() for x in dates])
t=np.arange(n)
def fit(y, mask, ndf=26):
    # log y ~ cubic-ish smooth trend (piecewise linear basis, knots every ~16 days) + weekday + dom
    knots=np.linspace(0,n-1,ndf)
    B=[np.ones(n), t/n]
    for k in knots[1:-1]: B.append(np.maximum(t-k,0)/n)
    for w in range(1,7): B.append((dw==w).astype(float))
    for dd in range(2,32): B.append((dom==dd).astype(float))
    X=np.array(B).T
    ly=np.log(y)
    A=X[mask]; bfit=np.linalg.lstsq(A,ly[mask],rcond=None)[0]
    ndomstart=len(B)-30
    domc=np.zeros(31); domc[1:]=bfit[ndomstart:]
    domc-=domc.mean()
    dwc=np.zeros(7); dwc[1:]=bfit[len(B)-30-6:len(B)-30]; dwc-=dwc.mean()
    resid=ly[mask]-A@bfit
    return domc, dwc, float(resid.std())
NY=np.array([(x>=dt.date(2025,12,20) and x<=dt.date(2026,1,10)) or (x<=dt.date(2025,1,10)) for x in dates])
allm=np.ones(n,bool)
for nm,y in [('gmv',d['gmv'].to_numpy()),('orders',d['orders'].to_numpy().astype(float)),('buyers',d['buyers'].to_numpy().astype(float))]:
    for mk,mask in [('full',allm),('no-NY',~NY)]:
        dc,wc,rs=fit(y,mask)
        print('%-7s %-6s  dom p2p=%.4f (x%.3f)  sd(dom)=%.4f  weekday p2p=%.4f  resid sd=%.4f'%(nm,mk,dc.max()-dc.min(),np.exp(dc.max()-dc.min()),dc.std(),wc.max()-wc.min(),rs))
        if nm=='gmv' and mk=='no-NY':
            DOM=dc.copy(); DW=wc.copy()
print('\nrobust dom factors (gmv, no-NY, log dev):')
print(' '.join('%d:%+.3f'%(i+1,DOM[i]) for i in range(31)))
def wdays(a,b):
    o=[];x=a
    while x<=b: o.append(x); x+=dt.timedelta(days=1)
    return o
def lvl(days, dcm, dwc, use_dw=True):
    return np.mean([np.exp(dcm[x.day-1]+(dwc[x.weekday()] if use_dw else 0)) for x in days])
V=wdays(dt.date(2026,1,15),dt.date(2026,2,13)); T=wdays(dt.date(2026,2,14),dt.date(2026,3,15))
print('\nAGGREGATE coverage effect (test/val), dom only: %.5f ; dom+weekday: %.5f'%(lvl(T,DOM,DW,False)/lvl(V,DOM,DW,False), lvl(T,DOM,DW)/lvl(V,DOM,DW)))
V5=wdays(dt.date(2025,1,15),dt.date(2025,2,13)); T5=wdays(dt.date(2025,2,14),dt.date(2025,3,15))
print('  same for 2025 analog windows: dom only %.5f ; dom+weekday %.5f'%(lvl(T5,DOM,DW,False)/lvl(V5,DOM,DW,False), lvl(T5,DOM,DW)/lvl(V5,DOM,DW)))
# stability of dom pattern across halves of the data
h1=np.array([x<dt.date(2025,7,15) for x in dates])&~NY; h2=np.array([x>=dt.date(2025,7,15) for x in dates])&~NY
d1,_,_=fit(d['gmv'].to_numpy(),h1,ndf=13); d2,_,_=fit(d['gmv'].to_numpy(),h2,ndf=13)
print('\ndom-pattern stability H1 vs H2 2025: corr=%.3f  sd1=%.4f sd2=%.4f'%(np.corrcoef(d1,d2)[0,1],d1.std(),d2.std()))
noNov=(~NY)&np.array([x.month!=11 for x in dates])
dn,_,_=fit(d['gmv'].to_numpy(),noNov)
print('dom pattern excluding November (11.11 sale): corr with main %.3f ; p2p %.4f ; coverage ratio %.5f'%(np.corrcoef(dn,DOM)[0,1],dn.max()-dn.min(), lvl(T,dn,DW,False)/lvl(V,dn,DW,False)))
json.dump({'DOM':DOM.tolist(),'DW':DW.tolist()},open(SP+'dom_robust.json','w'))
