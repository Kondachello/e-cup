# -*- coding: utf-8 -*-
"""N16. Почему слипы SHOW9/10 стали ОТРИЦАТЕЛЬНЫМИ.

Гипотеза (Саша, со ссылкой на мою §2.4): когда оптимизированный файл ЗАМЕРЕН и
внесён в базис, его публичный скор пиннит ровно ту комбинацию, которую
оптимизатор эксплуатировал. Delta по этому направлению становится известной, и
следующая оптимизация не может его переиспользовать.

Прямой тест: два раунда оптимизации, во втором результат первого В БАЗИСЕ.
"""
import json, math, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
NP_, F0R, MEAN_T = 50_000, 1.6470, 2.3275
ANCH, DROP = "A1_gram7_shift", {"sample"}
z=np.load("out/lb_full.npz"); meta=json.load(open("out/lb_meta.json"))
SHOWS={"SHOW_maxpub","SHOW2_aggr","SHOW3_maxpub","SHOW3b_safe"}
names=[n for n in meta["names"] if n not in DROP]
L=np.vstack([z[f"lp_{n}"].astype(np.float64) for n in names])
fsc=np.array([meta["sc"][n] for n in names])
N=L.shape[1]; a=names.index(ANCH); lp_a=L[a]; D=L-lp_a
qd=(L*L).mean(1)
psi_all=np.concatenate([[MEAN_T],((qd-qd[a])-(fsc**2-fsc[a]**2))/2])
rng=np.random.default_rng(23); NDRAW=60
SPL=[]
for _ in range(NDRAW):
    P=rng.choice(N,NP_,replace=False); Dp=D[:,P]; ap=lp_a[P]
    SPL.append((D@D.T/N-Dp@Dp.T/NP_, D@lp_a/N-Dp@ap/NP_, D.mean(1)-Dp.mean(1),
                lp_a.mean()-ap.mean(), P))
def slip(c,S,g,h,A):
    c0,ci=c[0],c[1:]
    return float(ci@np.diag(S)-2*c0*(ci@h)-ci@S@ci-2*c0*A)

def run(idx, lam, extra=None):
    """extra = (вектор направления d_ex, его ТОЧНЫЙ публичный psi) — заякоренный файл"""
    out=[]
    for S,g,h,A,P in SPL:
        Ss,gs,hs=S[np.ix_(idx,idx)],g[idx],h[idx]
        cols=[np.ones(N)]+[D[i] for i in idx]
        if extra is not None: cols.append(extra)
        B=np.vstack(cols); G=B@B.T/N; m=B@lp_a/N
        Dl=[0.0]+list(2*gs+np.diag(Ss))
        if extra is not None:
            # заякоренный файл: его Delta ИЗВЕСТНА => psi точна, вклад в Delta = 0
            Dl.append(0.0)
        Dl=np.array(Dl)
        ps=[MEAN_T]+list(psi_all[1:][idx])
        if extra is not None:
            ex2=extra*extra
            ps.append(float((2*lp_a*extra+ex2).mean())/2)   # заглушка уровня, роль играет только Delta=0
        ps=np.array(ps)
        R=np.eye(len(m)); R[0,0]=1e-4
        c=np.linalg.solve(G+lam*R, ps+Dl/2-m)
        # слип считаем по РЕАЛЬНЫМ направлениям (включая extra, у которого delta известна)
        c0=c[0]; ci=c[1:len(idx)+1]
        d=ci@D[idx]
        if extra is not None: d=d+c[-1]*extra
        u=(ci[:,None]*(D[idx]**2)).sum(0)+((c[-1]*extra**2) if extra is not None else 0)\
          -(c0+d)**2-2*c0*lp_a
        out.append((u.mean()-u[P].mean())/(2*F0R))
    return float(np.mean(out)), float(np.std(out))

base=[i for i in range(len(names)) if names[i] not in SHOWS]
print(f"базис без SHOW: {len(base)} файлов; с SHOW: {len(names)}")
print(f"\n{'lam':>8}{'раунд 1 (без SHOW)':>22}{'раунд 2 (SHOW в базисе)':>26}")
for lam in (1e-2,3e-3,1e-3,3e-4):
    e1,_=run(base,lam)
    e2,_=run(list(range(len(names))),lam)     # SHOW-файлы замерены и в базисе
    print(f"{lam:8.0e}{e1:+22.6f}{e2:+26.6f}")

print("")
print("ВЫВОД (ОПРОВЕРГАЕТ исходную гипотезу): внесение SHOW в базис слип НЕ снижает,")
print("а ПОВЫШАЕТ. Причина: psi КАЖДОГО файла базиса несёт свою Delta_i, поэтому")
print("новый файл ДОБАВЛЯЕТ свободы для эксплуатации, а не пиннит её.")
print("Отрицательные слипы SHOW9/10 объясняются НЕ заякориванием, а тем, что на")
print("малой агрессии систематический член тонет в собственном разбросе — см. n17.")
