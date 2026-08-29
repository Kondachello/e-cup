# -*- coding: utf-8 -*-
"""C3: полный перебор законных конструкций второго файла + оптимизация P(топ-3) пары."""
import json, os, sys, itertools
import numpy as np, scipy.linalg as sla
sys.path.insert(0, "/Users/alexanderkondakov/ozon-cup/work/scripts")
from p_top3 import Objective, MU_US, SIGMA_US, NOISE

SCR = os.path.dirname(os.path.abspath(__file__))
st = np.load("/Users/alexanderkondakov/ozon-cup/work/reports/lineA/gls_state_eb.npz", allow_pickle=True)
names = [str(x) for x in st["names"]]
Q, cQ, V, Lam, q, bF7 = st["Q"], st["cQ"], st["mdl_vivian"], st["Lam"], st["q"], st["doses_F7"]
FS = float(st["F_SCALE"]); n = len(names); RG = 1e-9*np.trace(Q)/n*np.eye(n); SAMP = 0.0011
def solve(g=0.1, lam=1.0, keep=None):
    A = Q + lam*Lam + RG + g*np.diag(np.diag(Q))
    if keep is None: return np.linalg.solve(A, cQ)
    d = np.zeros(n); k = np.array(sorted(keep)); d[k] = np.linalg.solve(A[np.ix_(k,k)], cQ[k]); return d
def gain(d): return float((2*d@cQ - d@Q@d)/(2*FS))
def gsd(d): return float(np.sqrt(max(1.25**2*d@V@d,0))/FS)
def rms_lp(dd): return float(np.sqrt(max(dd@Q@dd,0)))
vert = bF7 + cQ/q
def diag_b(d):
    b = bF7+d; r = b/vert
    cost = q*((b-vert)**2 - vert**2)/(2*FS*NOISE)
    viol = (r<0)|(r>2)
    return float(np.abs(r).max()), int(viol.sum()), int((viol&(cost>1)).sum()), float(cost.max())

d01 = solve(0.1)
GAIN_F8 = gain(d01) - 0.43*NOISE
SIG_SHARED = float(np.sqrt(SIGMA_US**2 - gsd(d01)**2))
SD_F8_EXTRA = 1.52*NOISE          # старый приор F8 против нового — независимая добавка (сверено в C2.0)
SD_F8 = float(np.hypot(gsd(d01), SD_F8_EXTRA))

obj = Objective(ns=400_000)
rng = np.random.default_rng(20260829)
z_sh = obj.z                       # ОБЩАЯ неопределённость (одна на оба файла)
e_a  = obj.w                       # независимые компоненты переноса
e_b  = rng.standard_normal(obj.ns)
e_c  = rng.standard_normal(obj.ns)

def mu_of(d): return MU_US - (gain(d) - GAIN_F8)

def draw(d, extra=0.0, eps=None):
    """приват файла: общая часть + перенос доз (коррелирован через V) + добавка."""
    return mu_of(d) + SIG_SHARED*z_sh + eps

def P_single(d, extra=0.0):
    s = float(np.hypot(gsd(d), extra))
    g = mu_of(d) + SIG_SHARED*z_sh + s*e_a
    return float((g < obj.c3).mean())

def P_pair(d1, d2, ex1=0.0, ex2=0.0):
    """общая z_sh + двумерный перенос с ковариацией из V + независимые добавки + семплинг."""
    s1 = gsd(d1); s2 = gsd(d2); c = float(1.25**2*d1@V@d2/FS**2)
    ssamp = SAMP*rms_lp(d1-d2)
    C = np.array([[s1**2+ex1**2, c],[c, s2**2+ex2**2+ssamp**2]])
    w_,U = np.linalg.eigh(C); w_=np.clip(w_,0,None); A = U@np.diag(np.sqrt(w_))
    t1 = A[0,0]*e_a + A[0,1]*e_b; t2 = A[1,0]*e_a + A[1,1]*e_b
    g1 = mu_of(d1)+SIG_SHARED*z_sh+t1; g2 = mu_of(d2)+SIG_SHARED*z_sh+t2
    return float((np.minimum(g1,g2) < obj.c3).mean())

P_F8 = P_single(d01, SD_F8_EXTRA)
print(f"БАЗА: одиночный F8  P(топ-3) = {P_F8*100:.2f} %   (OBJECTIVE: 20.92 %)\n")

# ---------------- семейства кандидатов
cands = {}
for g in [0.0,0.005,0.01,0.02,0.03,0.05,0.0765,0.08,0.1,0.12,0.2,0.5,1.0,3.0,10.0,100.0]:
    cands[f"g{g:g}"] = solve(g)
for lam in [0.0,0.25,0.5,2.0,4.0]:
    cands[f"g0_L{lam:g}"] = solve(0.0, lam)
    cands[f"g0.08_L{lam:g}"] = solve(0.08, lam)
cands["F7(нулевые дозы)"] = np.zeros(n)
# подмножества осей
P_ax = [i for i,k in enumerate(names) if k.startswith("P")]
Z_ax = [i for i,k in enumerate(names) if k.startswith("Z")]
oth  = [i for i,k in enumerate(names) if not k.startswith(("P","Z"))]
allx = list(range(n))
cands["только P*"] = solve(0.0, 1.0, P_ax)
cands["только Z*"] = solve(0.0, 1.0, Z_ax)
cands["только прочие"] = solve(0.0, 1.0, oth)
cands["без 8 нарушителей"] = solve(0.0, 1.0, [i for i in allx if i not in mat])
cands["только 8 нарушителей"] = solve(0.0, 1.0, mat)
o = np.argsort(-np.abs(cQ))
for k in (5,10,20,30):
    cands[f"топ-{k} осей по |cQ|"] = solve(0.0,1.0,list(o[:k]))
    cands[f"хвост-{n-k} осей"] = solve(0.0,1.0,list(o[k:]))
rs = np.random.default_rng(7)
for j in range(6):
    sub = list(rs.choice(n, n//2, replace=False))
    cands[f"случайная половина #{j+1}"] = solve(0.0,1.0,sub)
# направления максимальной развязки
M = 1.25**2*V/FS**2 + SAMP**2*Q
w_,Vg = sla.eigh(M,Q)
for j in range(3):
    v = Vg[:,-1-j]; v = v/np.sqrt(v@Q@v)
    for t in [-0.06,-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03]:
        cands[f"g0 + {t:+.2f}*v{j+1}"] = solve(0.0) + t*v

rows=[]
for nm,d in cands.items():
    mr,nv,nm_,cst = diag_b(d)
    ps = P_single(d)
    pp = P_pair(d, d01, 0.0, SD_F8_EXTRA)
    rows.append(dict(name=nm, gain_n=(gain(d)-GAIN_F8)/NOISE, sd_n=gsd(d)/NOISE,
                     sd_diff_n=float(np.hypot(np.sqrt(max(1.25**2*(d-d01)@V@(d-d01),0))/FS, SD_F8_EXTRA))/NOISE,
                     rms=rms_lp(d-d01), p_single=ps, p_pair=pp,
                     add=(pp-ps)*100, vs_f8=(pp-P_F8)*100,
                     maxr=mr, nviol=nv, nmat=nm_, cost=cst))
rows.sort(key=lambda r:-r["p_pair"])
print(f"{'конструкция':26s}{'vsF8,ш':>8}{'sd,ш':>7}{'sdразн,ш':>10}{'P(один)':>9}"
      f"{'P(пара+F8)':>12}{'2-й даёт':>10}{'max|r|':>8}{'мат':>5}{'цена,ш':>8}")
print("-"*115)
for r in rows[:34]:
    print(f"{r['name']:26s}{r['gain_n']:+8.2f}{r['sd_n']:7.2f}{r['sd_diff_n']:10.2f}"
          f"{r['p_single']*100:8.2f}%{r['p_pair']*100:11.2f}%{r['add']:+10.2f}"
          f"{r['maxr']:8.2f}{r['nmat']:5d}{r['cost']:8.1f}")
print("...")
for nm in ["g0","g0.08","g0.1","g0.2","F7(нулевые дозы)","только P*","только Z*","g0_L0","g0_L0.5"]:
    r=[x for x in rows if x["name"]==nm][0]
    print(f"{r['name']:26s}{r['gain_n']:+8.2f}{r['sd_n']:7.2f}{r['sd_diff_n']:10.2f}"
          f"{r['p_single']*100:8.2f}%{r['p_pair']*100:11.2f}%{r['add']:+10.2f}"
          f"{r['maxr']:8.2f}{r['nmat']:5d}{r['cost']:8.1f}")
json.dump(dict(P_F8=P_F8, rows=rows), open(os.path.join(SCR,"partC3.json"),"w"),
          ensure_ascii=False, indent=1)
