import numpy as np, polars as pl
from pathlib import Path
SP = Path("/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/0b55ab9f-3777-4ebc-bd91-937895c0e355/scratchpad")
t = pl.read_parquet(SP / "p62_agg.parquet")
rec = t["last_di"].to_numpy().astype(np.float64); rec = np.where(np.isnan(rec), 1e9, rec)
never = t["nbuyd"].to_numpy() == 0
act30 = t["act30"].to_numpy(); act30s = t["act30s"].to_numpy(); browse30 = t["browse30"].to_numpy()
act90 = t["act90"].to_numpy(); browse90 = t["browse90"].to_numpy(); s30 = t["searches30"].to_numpy()
nrows = t["nrows"].to_numpy()

print("=== массы кандидатов «спящих» ===")
defs = {
 "S1 rec91-365":        (rec>=91)&(rec<=365),
 "S2 rec>=91 (покупали)":(rec>=91)&(~never),
 "S3 rec>=91 или NEVER": ((rec>=91)|never),
 "S4 rec>=60 покупали":  (rec>=60)&(~never),
 "S5 rec>=46 покупали":  (rec>=46)&(~never),
 "S6 rec>=31 покупали":  (rec>=31)&(~never),
 "S7 rec>=60 или NEVER": ((rec>=60)|never),
 "S8 rec>=91|NEVER & есть строки": ((rec>=91)|never)&(nrows>0),
}
for k,v in defs.items():
    print(f"  {k:32s} m={v.mean():.5f}  q_контраст(0.30)={0.09*v.mean():.5f}")

print("\n=== распределение browsing внутри S2 (rec>=91, покупали), m=%.4f ===" % defs["S2 rec>=91 (покупали)"].mean())
for nm, x in [("act30",act30),("act30s",act30s),("browse30",browse30),("act90",act90),("browse90",browse90),("searches30",s30)]:
    for key in ["S1 rec91-365","S2 rec>=91 (покупали)","S3 rec>=91 или NEVER"]:
        m = defs[key]; xs = x[m]
        qs = np.percentile(xs,[25,50,75,90])
        print(f"  {nm:10s} [{key:22s}] med={np.median(xs):6.1f} p25/50/75/90={qs} доля>0={float((xs>0).mean()):.4f}")
    print()
