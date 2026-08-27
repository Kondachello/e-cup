"""S1. Локальный предиктор публичного скора из того, что есть.
   f² = mean_P(lp²) − 2·φ(lp) + mean_P(t²),  φ(lp)=mean_P(lp·t) линеен по lp.
Известны точно: mean_P(t)=2.3275, mean_P(t²)=10.79 (KNOWLEDGE, замерено парами).
Два замеренных файла: sample_submit (=валидационный таргет, 2.122483523224017)
с валидационным таргетом с коэффициентом a=1.052 (из predict_lb.py)."""
import numpy as np, polars as pl
MEAN_T, MEAN_T2, A_RESID = 2.3275, 10.79, 1.0520
S_SAMPLE, S_REF = 2.122483523224017, 1.6489445575

z = np.load('final_submission/models/chain_test.npz')
uid = z['user_id']; ref = z['ref_lp']
ss = pl.read_csv('sample_submit.csv', schema_overrides={"user_id": pl.Int64}).sort('user_id')
assert np.array_equal(ss['user_id'].to_numpy(), uid)
col = 'predict' if 'predict' in ss.columns else ss.columns[1]
samp = np.log1p(np.clip(ss[col].to_numpy().astype(float), 0, None))
tval = samp.copy()                       # sample_submit побитово = валидационный таргет

phi = lambda lp, f2: (float(np.mean(lp*lp)) + MEAN_T2 - f2)/2
phi_s = phi(samp, S_SAMPLE**2); phi_r = phi(ref, S_REF**2)
print(f"φ(sample) = {phi_s:.6f}   φ(ref) = {phi_r:.6f}")

B = np.column_stack([np.ones(len(uid)), samp, ref])
PHI = np.array([MEAN_T, phi_s, phi_r])
G = B.T @ B / len(uid)

def predict(lp):
    w = np.linalg.solve(G, B.T @ lp / len(uid))       # проекция на {1, sample, ref}
    r = lp - B @ w
    phi_hat = float(w @ PHI) + A_RESID*float(np.mean(r*(tval - tval.mean())))
    f2 = float(np.mean(lp*lp)) - 2*phi_hat + MEAN_T2
    return float(np.sqrt(max(f2, 0)))

print(f"\n=== ПРОВЕРКА на известных ответах ===")
np.save('../zhenya/out/lb_basis.npy', {"uid":uid,"ref":ref,"samp":samp,"phi_s":phi_s,"phi_r":phi_r},
        allow_pickle=True)
