"""mdl_silica. Переносится ли смещение сегментов gc_nocat и search_int между окнами?
Родовой класс «сегментная ступенька» имеет κ=0.05 (не переносится). Но эти два
сегмента заданы СТРУКТУРНЫМ свойством данных, а не квантилем прогноза — проверяем."""
import os, numpy as np, polars as pl, lightgbm as lgb
from datetime import date, timedelta
from pathlib import Path
CACHE=Path(os.environ["ZH_CACHE"])
df=pl.read_parquet("train.parquet")
def mat(X): 
    c=[x for x in X.columns if x.startswith("b_")]
    return np.nan_to_num(X.select(c).to_numpy().astype(np.float64),nan=-1.,posinf=1e9,neginf=-1e9)
have=sorted(date.fromisoformat(p.stem[1:]) for p in CACHE.glob("a20*.parquet"))
A1,A2=date(2025,10,20),date(2025,12,1)
TR=[a for a in have if a<=A1-timedelta(days=30)][-5:]
print(f"обучение на {len(TR)} срезах {TR[0]}..{TR[-1]}; окна {A1} и {A2} (разнос 42д)")
Xs=[];ys=[]
for a in TR:
    X=pl.read_parquet(CACHE/f"a{a}.parquet"); Xs.append(mat(X))
    ys.append(np.log1p(X["target"].to_numpy().astype(float)))
m=lgb.LGBMRegressor(objective="tweedie",tweedie_variance_power=1.45,learning_rate=.05,
    num_leaves=63,min_child_samples=100,subsample=.8,colsample_bytree=.8,
    n_estimators=700,verbose=-1,n_jobs=4,random_state=42).fit(np.vstack(Xs),np.concatenate(ys))

def segs(A,uid):
    h=df.filter(pl.col("event_date").is_between(A-timedelta(days=364),A)).group_by("user_id").agg(
        ((pl.col("gmv_cat")>0)&(pl.col("cat")==0)).sum().alias("gcn"))
    h90=df.filter(pl.col("event_date").is_between(A-timedelta(days=89),A)).group_by("user_id").agg(
        [(pl.col("search")>0).sum().alias("sd"),pl.len().alias("ad")])
    b=pl.DataFrame({"user_id":uid}).join(h,on="user_id",how="left").join(h90,on="user_id",how="left").fill_null(0)
    gcn=b["gcn"].to_numpy(); sd=b["sd"].to_numpy().astype(float); ad=b["ad"].to_numpy().astype(float)
    return gcn>0, (np.divide(sd,np.maximum(ad,1))>=0.95)&(ad>0)

R={}
for tag,A in (("A1",A1),("A2",A2)):
    X=pl.read_parquet(CACHE/f"a{A}.parquet"); uid=X["user_id"].to_numpy()
    p=np.clip(m.predict(mat(X)),0,None); y=np.log1p(X["target"].to_numpy().astype(float))
    e=p-y; e=e-e.mean()                        # уровень окна снимается глобальным сдвигом
    g,s=segs(A,uid); R[tag]=(uid,e,g,s)
    print(f"  {tag}: n={len(uid):,}  gc_nocat {100*g.mean():.1f}%  search_int {100*s.mean():.1f}%")
print(f"\n{'сегмент':14s} {'смещение A1':>13} {'смещение A2':>13} {'перенос':>9}")
u1,e1,g1,s1=R["A1"]; u2,e2,g2,s2=R["A2"]
c=np.intersect1d(u1,u2); i1=np.searchsorted(u1,c); i2=np.searchsorted(u2,c)
for nm,m1,m2 in (("gc_nocat",g1[i1],g2[i2]),("search_int",s1[i1],s2[i2])):
    b1=e1[i1][m1].mean()-e1[i1][~m1].mean(); b2=e2[i2][m2].mean()-e2[i2][~m2].mean()
    print(f"{nm:14s} {b1:>+13.5f} {b2:>+13.5f} {b2/b1 if abs(b1)>1e-9 else float('nan'):>9.3f}")
print(f"\nперенос ~1.0 = смещение структурное (переносится); ~0 = свойство окна")
