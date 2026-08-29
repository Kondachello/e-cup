# -*- coding: utf-8 -*-
"""Локальная репродукция M3: собираем всех кандидатов слота-2 (lp + замеренный паблик)."""
import json, os, re, sys
import numpy as np, polars as pl
sys.path.insert(0, "/Users/alexanderkondakov/ozon-cup/work")
from doctrine import transfer as T

REP = "/Users/alexanderkondakov/ozon-cup/"
SCR = os.path.dirname(os.path.abspath(__file__))
src = open(REP + "work/scripts/predict_lb.py", encoding="utf-8").read()
pairs = re.findall(r'\("([^"]+)",\s*"([^"]+)",\s*([\d.]+)\)', src)
SC = {m[0]: float(m[2]) for m in pairs}
FN = {m[0]: m[1] for m in pairs}
order = list(FN.keys()); pos = {n: i for i, n in enumerate(order)}

have = []
for n, f in FN.items():
    p = REP + "submissions/" + f
    if os.path.exists(p) and n in SC:
        have.append((n, p, SC[n]))
print(f"замеренных всего: {len(SC)};  с локальным csv: {len(have)}", file=sys.stderr)

uid = None; L = {}
for n, p, s in have:
    d = pl.read_csv(p, schema_overrides={"user_id": pl.Int64}).sort("user_id")
    c = "predict" if "predict" in d.columns else d.columns[1]
    v = d[c].to_numpy().astype(np.float64)
    lp = np.log1p(np.clip(v, 0, None))
    if uid is None:
        uid = d["user_id"].to_numpy()
    elif len(lp) != len(uid):
        print(f"  пропуск {n}: длина {len(lp)}", file=sys.stderr); continue
    L[n] = lp
print(f"загружено lp: {len(L)}, юзеров {len(uid)}", file=sys.stderr)
np.savez_compressed(os.path.join(SCR, "cands_lp.npz"), uid=uid,
                    names=np.array(list(L.keys())),
                    lp=np.stack([L[k] for k in L]),
                    scores=np.array([SC[k] for k in L]),
                    posidx=np.array([pos[k] for k in L]))
print("saved", file=sys.stderr)
