"""S4. Что подогнано под паблик, а что принесло информацию.
Тест: для каждого файла в хронологии считаем ЛУЧШИЙ ДОПУСТИМЫЙ микс его
предшественников (в подпространстве первых k главных компонент — базис почти
двумерен, больше брать нельзя, иначе подгоняется шум) и сравниваем со скором файла.
  скор файла < оптимума миксов предшественников  ->  файл ПРИНЁС информацию
  скор файла >= оптимума                          ->  файл выразим миксом, т.е. подгонка
"""
import json, numpy as np
z=np.load("../zhenya/out/lb_full.npz"); M=json.load(open("../zhenya/out/lb_meta.json"))
SC=M["sc"]; PHI=M["phi"]; uid=z["uid"]; n=len(uid); MEAN_T,MEAN_T2=2.3275,10.79
ORDER=["sample","sub_blend_w1a","lbmix2","lbmix4_3way","sub_c_cand","A1_gram7_shift",
 "A2_probe_s1_gmv","F4_applied","F5_probe_hmmsim","G1_probe_zeropush","H1_applied",
 "H2_edge_p1","H2_tfm","H2_tfm_centered","H2_tfm_half","Q1_probes5","R2_newblend",
 "R3_ridge","S1_segwall","R5_shade","V1_tfm3b_pre","V2_tfm3b_opt","V3_canon",
 "SHOW_maxpub","SHOW2_aggr","SHOW3b_safe","SHOW3_maxpub","G1_gru_tfm_full",
 "F1_trio_full","G2_gru_tfm_02","T1_tfm4_orth_090","T2_tfm4_orth_045"]
ORDER=[o for o in ORDER if o in M["names"]]

def best_mix(prev, k):
    """оптимум по подпространству первых k главных компонент предшественников"""
    X=np.column_stack([z[f"lp_{p}"] for p in prev])
    mu=X.mean(0); Xc=X-mu
    U,S,Vt=np.linalg.svd(Xc,full_matrices=False)
    k=min(k,len(S))
    Bk=np.column_stack([np.ones(n), Xc@Vt[:k].T])          # 1 + k компонент
    # phi для компонент: phi линейна, phi(Xc@v) = sum_j v_j*(phi_j - mean-часть)
    ph_files=np.array([PHI[p] for p in prev])
    ph_c=ph_files - mu*MEAN_T                               # phi(x - mean(x)*1)
    PH=np.concatenate([[MEAN_T], Vt[:k]@ph_c])
    G=Bk.T@Bk/n
    w=np.linalg.solve(G+1e-10*np.eye(len(PH)), PH)
    lp=Bk@w
    if (lp<0).sum()>0: lp=np.clip(lp,0,None)               # честно: платформа клипует
    f2=float(np.mean(lp*lp))-2*float(w@PH)+MEAN_T2
    return float(np.sqrt(max(f2,0))), int((Bk@w<0).sum())

print(f"{'файл':22s} {'скор':>11} {'оптимум миксов':>15} {'дельта':>10}  вердикт")
for i,nm in enumerate(ORDER):
    if i<4: continue
    opt,neg=best_mix(ORDER[:i], k=3)
    d=SC[nm]-opt
    verd="ПРИНЁС информацию" if d<-0.00005 else ("подгонка/микс" if d>-0.00005 else "")
    print(f"{nm:22s} {SC[nm]:>11.7f} {opt:>15.7f} {d:>+10.7f}  {verd}")
