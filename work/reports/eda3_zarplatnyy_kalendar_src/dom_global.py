import numpy as np, polars as pl, json
dt = d['event_date'].to_numpy()
g = d['gmv'].to_numpy().astype(float)
o = d['orders'].to_numpy().astype(float)
n = len(g)
dom = np.array([x.astype('datetime64[D]').item().day for x in dt])
dow = np.array([x.astype('datetime64[D]').item().weekday() for x in dt])
print('n days', n, dt[0], dt[-1])

def detrend(x, w=29):
    # centered rolling mean, edges by shorter window
    lg = np.log(x)
    tr = np.empty(n)
    h = w//2
    for i in range(n):
        a,b = max(0,i-h), min(n,i+h+1)
        tr[i] = lg[a:b].mean()
    return lg - tr, tr

r, tr = detrend(g)
# remove weekday effect
wk = np.array([r[dow==k].mean() for k in range(7)])
r2 = r - wk[dow]
print('weekday effect (log):', np.round(wk,4), 'peak-to-peak', round(wk.max()-wk.min(),4), 'ratio', round(float(np.exp(wk.max()-wk.min())),4))
print('resid sd after trend:', round(float(r.std()),4), 'after weekday:', round(float(r2.std()),4))

# dom means
rows=[]
for dd in range(1,32):
    m = dom==dd
    rows.append((dd, int(m.sum()), float(r2[m].mean()), float(r2[m].std(ddof=1)/np.sqrt(m.sum()))))
print('dom | k | mean_log_resid | se')
for dd,k,mu,se in rows:
    print(f'{dd:2d} {k:3d} {mu:+.4f} {se:.4f}  {"*" if abs(mu)>2*se else ""}')
dm = np.array([x[2] for x in rows])
print('dom peak-to-peak (log):', round(float(dm.max()-dm.min()),4), 'ratio', round(float(np.exp(dm.max()-dm.min())),4), 'sd', round(float(dm.std()),4))

# F-test dom effect
ss_tot = float((r2**2).sum())
pred = np.array([rows[dd-1][2] for dd in dom])
ss_res = float(((r2-pred)**2).sum())
print('mdl_flint of dom on detrended+deweekdayed daily gmv:', round(1-ss_res/ss_tot,4))

# periodogram of r2 (after removing weekday) — power at periods 2..60 days
freqs = np.arange(1, n//2)
P = np.abs(np.fft.rfft(r2 - r2.mean()))**2
per = n/np.arange(1,len(P))
top = np.argsort(P[1:])[::-1][:12]+1
print('top periodogram peaks (period days, power):', [(round(float(n/k),2), round(float(P[k]/P[1:].sum()*100),2)) for k in top])
# power in 28-32d band
band = [k for k in range(1,len(P)) if 27.5 <= n/k <= 33.5]
print('share of power in 27.5-33.5d band: %.2f%%' % (100*P[band].sum()/P[1:].sum()))
band7 = [k for k in range(1,len(P)) if 6.7 <= n/k <= 7.3]
print('share of power in ~7d band (weekday already removed): %.2f%%' % (100*P[band7].sum()/P[1:].sum()))
json.dump({'dom_mean_log':[x[2] for x in rows],'weekday':wk.tolist()}, open('/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad/dom_global.json','w'))
