import numpy as np, polars as pl, datetime as dt, json
SP='/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad/'
R='/Users/alexanderkondakov/ozon-cup/'
J=json.load(open(SP+'dom_robust.json')); DOM=np.array(J['DOM']); DW=np.array(J['DW'])
J2=json.load(open(SP+'dom_global.json')); DW_ma=np.array(J2['weekday'])
print('robust weekday (Mon..Sun):', np.round(DW,4), ' p2p %.4f'%(DW.max()-DW.min()))
print('MA     weekday (Mon..Sun):', np.round(DW_ma,4), ' p2p %.4f'%(DW_ma.max()-DW_ma.min()))
dates=d['event_date'].to_list(); G={x:g for x,g in zip(dates,d['gmv'].to_numpy())}
def wd(a,b):
    o=[];x=a
    while x<=b: o.append(x); x+=dt.timedelta(days=1)
    return o
W1=wd(dt.date(2025,1,15),dt.date(2025,2,13)); mdl_onyx=wd(dt.date(2025,2,14),dt.date(2025,3,15))
V=wd(dt.date(2026,1,15),dt.date(2026,2,13)); T=wd(dt.date(2026,2,14),dt.date(2026,3,15))
for nm,w in [('W1_2025',W1),('W2_2025',mdl_onyx),('VAL_2026',V),('TEST_2026',T)]:
    print('%-10s start=%s  weekday counts=%s  mean wk factor: robust %.5f  MA %.5f'%(nm,w[0].strftime('%a %Y-%m-%d'),
        np.bincount([x.weekday() for x in w],minlength=7).tolist(),
        np.mean([np.exp(DW[x.weekday()]) for x in w]), np.mean([np.exp(DW_ma[x.weekday()]) for x in w])))
def ratio(A,B,adj):
    fa=np.mean([G[x]/ (np.exp(adj[x.weekday()]) if adj is not None else 1) for x in A])
    fb=np.mean([G[x]/ (np.exp(adj[x.weekday()]) if adj is not None else 1) for x in B])
    return fb/fa
print('\nR_season 2025 (/) raw               = %.4f'%ratio(W1,mdl_onyx,None))
print('R_season 2025 weekday-adjusted (robust) = %.4f'%ratio(W1,mdl_onyx,DW))
print('R_season 2025 weekday-adjusted (MA)     = %.4f'%ratio(W1,mdl_onyx,DW_ma))
for nm,adj in [('robust',DW),('MA',DW_ma)]:
    c25=np.mean([np.exp(adj[x.weekday()]) for x in mdl_onyx])/np.mean([np.exp(adj[x.weekday()]) for x in W1])
    c26=np.mean([np.exp(adj[x.weekday()]) for x in T])/np.mean([np.exp(adj[x.weekday()]) for x in V])
    print('weekday-composition ratio: 2025 pair %.5f, 2026 pair %.5f -> carry correction x%.5f (%s)'%(c25,c26,c26/c25,nm))
