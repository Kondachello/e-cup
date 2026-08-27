"""Единая точка воспроизведения kostya46 v2 из чистого клона.

    python work_kostya/reproduce.py            # полный прогон: train.parquet -> preds/*.parquet
    KSMOKE=1 python work_kostya/reproduce.py   # смоук: только проверка, что код работает

Канон: сиды m2=1..4, twlog=1..2, m1=1; KTHREADS=2 (бит-повтор shipped-файлов при
lightgbm==4.7.0; при другом числе потоков — float-дрейф ~1e-7, проверять check_reproduce.py
с допуском). Правило команды: обучающие срезы с зазором >=35 дней от вал-окна.
"""
import numpy as np, polars as pl, lightgbm as lgb, json, gc, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import wp, SMOKE, SEED_OFF, NUM_THREADS, WORK
import prep_data
from cube import build_cube
from features import build_features
from datetime import date, timedelta

DAY0 = date(2025, 1, 1)
R = (lambda n: max(8, n // 100)) if SMOKE else (lambda n: n)
M2_SEEDS = [1] if SMOKE else [1, 2, 3, 4]
TW_SEEDS = [1] if SMOKE else [1, 2]

BASE_M2 = dict(learning_rate=0.05, num_leaves=160, min_data_in_leaf=400,
               feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
               max_bin=127, verbose=-1, num_threads=NUM_THREADS)
BASE_M1 = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=300,
               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
               verbose=-1, num_threads=NUM_THREADS)

def stage_prep():
    if not os.path.exists(wp("act.parquet")):
        prep_data.main()
    for b, out in [(379, "cube_val.npy"), (409, "cube_test.npy")]:
        if not os.path.exists(wp(out)):
            build_cube(b, wp(out))

def stage_targets():
    buys = pl.read_parquet(wp("buys.parquet"))
    users = pl.read_parquet(wp("users_order.parquet"))["user_id"]
    uid = users.to_numpy()
    bd = buys.with_columns(((pl.col("event_date") - pl.lit(DAY0)).dt.total_days()).alias("day"))
    u_idx = np.searchsorted(uid, bd["user_id"].to_numpy())
    day = bd["day"].to_numpy().astype(np.int32); gmv = bd["gmv"].to_numpy()
    def gmat(anchors):
        G = np.zeros((len(uid), len(anchors)))
        for j, T in enumerate(anchors):
            m = (day >= T) & (day < T + 30)
            np.add.at(G[:, j], u_idx[m], gmv[m])
        return G
    val_anchors = [36 + 7 * k for k in range(50)]
    np.save(wp("gmv_mat.npy"), gmat(val_anchors))
    np.save(wp("anchor_days.npy"), np.array(val_anchors))
    test_anchors = list(range(311, 375, 7))
    np.save(wp("gmv_mat_testgrid.npy"), gmat(test_anchors))
    json.dump(test_anchors, open(wp("testgrid_days.json"), "w"))
    # appearance targets (natural anchors, test grid) for P_app
    act = pl.read_parquet(wp("act.parquet")).with_columns(
        (pl.col("event_date") - pl.lit(DAY0)).dt.total_days().alias("day"))
    ud = np.searchsorted(uid, act["user_id"].to_numpy()); dd = act["day"].to_numpy().astype(np.int32)
    APP = [248, 255, 262, 269, 276, 283]
    A = np.zeros((len(uid), len(APP)), bool)
    for j, T in enumerate(APP):
        m = (dd >= T) & (dd < T + 30)
        A[np.unique(ud[m]), j] = True
    np.save(wp("app_mat_natural.npy"), A)
    json.dump(APP, open(wp("app_anchor_days.json"), "w"))
    print("targets done", flush=True)

def build_train(cube, anchors, boundary, gcol, ncols=None):
    Xs, ys, zs, ws = [], [], [], []
    for d in anchors:
        X, names = build_features(d, cube, boundary)
        if ncols: X = X[:, :ncols]
        Xs.append(X)
        z = np.log1p(gcol(d)).astype(np.float32)
        zs.append(z); ys.append(gcol(d) > 0)
        ws.append(np.full(X.shape[0], 0.5 ** (((boundary - d) / 7.0) / 26.0), dtype=np.float32))
        print("  built", d, flush=True)
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(zs),
            np.concatenate(ws), names[:ncols] if ncols else names)

def heads(tag, Xtr, ytr, ztr, wtr, names, Xpred, base, seeds, rounds, use_w=True):
    for seed in seeds:
        prm = dict(base, seed=seed + SEED_OFF)
        kw = dict(weight=wtr) if use_w else {}
        for hname, obj, rr, sub in [("pz", "l2", rounds[0], None), ("p", "binary", rounds[1], None),
                                     ("s", "l2", rounds[2], "buy")]:
            out = wp(f"{tag}_{hname}_s{seed}.npy")
            if os.path.exists(out): continue
            if sub == "buy":
                mb = ytr
                ds = lgb.Dataset(Xtr[mb], ztr[mb], feature_name=names, **({"weight": wtr[mb]} if use_w else {}))
            else:
                tgt = ztr if obj == "l2" else ytr
                ds = lgb.Dataset(Xtr, tgt, feature_name=names, **kw)
            mdl = lgb.train(dict(objective=obj, **prm), ds, num_boost_round=R(rr))
            np.save(out, mdl.predict(Xpred).astype(np.float32))
            mdl.save_model(wp(f"{tag}_{hname}_s{seed}.txt")); del mdl; gc.collect()
            print(f"  {tag} {hname} s{seed} done", flush=True)

LADDER_T = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.5]

def ladder_heads(tag, Xtr, ztr, wtr, names, Xpred):
    for t in LADDER_T:
        out = wp(f"{tag}_lad_t{t}.npy")
        if os.path.exists(out): continue
        prm = dict(BASE_M2, seed=1 + SEED_OFF)
        mdl = lgb.train(dict(objective="binary", **prm),
                        lgb.Dataset(Xtr, (ztr > t), weight=wtr, feature_name=names), num_boost_round=R(350))
        np.save(out, mdl.predict(Xpred).astype(np.float32))
        mdl.save_model(wp(f"{tag}_lad_t{t}.txt")); del mdl; gc.collect()
        print(f"  {tag} ladder t={t} done", flush=True)

def tw_heads(tag, Xtr, ztr, wtr, names, Xpred, seeds):
    for seed in seeds:
        out = wp(f"{tag}_twlog_s{seed}.npy")
        if os.path.exists(out): continue
        prm = dict(BASE_M2, seed=seed + SEED_OFF)
        mdl = lgb.train(dict(objective="tweedie", tweedie_variance_power=1.3, **prm),
                        lgb.Dataset(Xtr, ztr, weight=wtr, feature_name=names), num_boost_round=R(800))
        np.save(out, mdl.predict(Xpred).astype(np.float32))
        mdl.save_model(wp(f"{tag}_twlog_s{seed}.txt")); del mdl; gc.collect()
        print(f"  {tag} twlog s{seed} done", flush=True)

def main():
    stage_prep()
    if not os.path.exists(wp("gmv_mat.npy")):
        stage_targets()
    G = np.load(wp("gmv_mat.npy")); anchor_days = np.load(wp("anchor_days.npy"))
    d2c = {int(d): i for i, d in enumerate(anchor_days)}
    cube_v = np.load(wp("cube_val.npy"), mmap_mode="r")
    cube_t = np.load(wp("cube_test.npy"), mmap_mode="r")
    Gt = np.load(wp("gmv_mat_testgrid.npy")); TG = json.load(open(wp("testgrid_days.json")))
    tg2c = {d: i for i, d in enumerate(TG)}

    # ---- val grid: m2 recipe (10 slices) + twlog ----
    an_m2 = [281 + 7 * k for k in range(10)][-2:] if SMOKE else [281 + 7 * k for k in range(10)]
    Xtr, ytr, ztr, wtr, names = build_train(cube_v, an_m2, 379, lambda d: G[:, d2c[d]])
    Xv, _ = build_features(379, cube_v, 379)
    np.save(wp("Xval_379.npy"), Xv)
    heads("v_m2", Xtr, ytr, ztr, wtr, names, Xv, BASE_M2, M2_SEEDS, (800, 550, 650))
    tw_heads("v_m2", Xtr, ztr, wtr, names, Xv, TW_SEEDS)
    # лестница обучается на 8 последних срезах (295..344)
    m8 = np.isin(np.repeat(an_m2, 1), an_m2[-8:]).repeat(250000) if len(an_m2) > 8 else np.ones(len(ztr), bool)
    ladder_heads("v", Xtr[m8], ztr[m8], wtr[m8], names, Xv)
    del Xtr; gc.collect()

    # ---- val grid: m1 recipe (7 slices, первые 121 признак, без весов, max_bin=255) ----
    an_m1 = [302 + 7 * k for k in range(7)][-2:] if SMOKE else [302 + 7 * k for k in range(7)]
    Xtr, ytr, ztr, wtr, _n121 = build_train(cube_v, an_m1, 379, lambda d: G[:, d2c[d]], ncols=121)
    heads("v_m1", Xtr, ytr, ztr, wtr, _n121, Xv[:, :121], BASE_M1, [1], (700, 500, 500), use_w=False)
    del Xtr; gc.collect()

    # ---- test grid: m2 + twlog ----
    an_t = TG[-2:] if SMOKE else TG
    Xtr, ytr, ztr, wtr, names = build_train(cube_t, an_t, 409, lambda d: Gt[:, tg2c[d]])
    Xte, _ = build_features(409, cube_t, 409)
    np.save(wp("Xtest_409.npy"), Xte)
    heads("t_m2", Xtr, ytr, ztr, wtr, names, Xte, BASE_M2, M2_SEEDS, (800, 550, 650))
    tw_heads("t_m2", Xtr, ztr, wtr, names, Xte, TW_SEEDS)
    m8 = np.isin(np.repeat(an_t, 1), an_t[-8:]).repeat(250000) if len(an_t) > 8 else np.ones(len(ztr), bool)
    ladder_heads("t", Xtr[m8], ztr[m8], wtr[m8], names, Xte)
    # P_app head (natural anchors)
    if not os.path.exists(wp("papp_409.npy")):
        A = np.load(wp("app_mat_natural.npy")); APP = json.load(open(wp("app_anchor_days.json")))
        an_a = APP[-2:] if SMOKE else APP
        Xs, ys = [], []
        for j, d in enumerate(APP):
            if d not in an_a: continue
            X, _ = build_features(d, cube_t, 409)
            Xs.append(X); ys.append(A[:, j])
        app = lgb.train(dict(objective="binary", **dict(BASE_M2, seed=1 + SEED_OFF)),
                        lgb.Dataset(np.concatenate(Xs), np.concatenate(ys), feature_name=names),
                        num_boost_round=R(400))
        np.save(wp("papp_409.npy"), app.predict(Xte).astype(np.float32))
        app.save_model(wp("t_app.txt"))
    del Xtr; gc.collect()

    # ---- m1-slot на тесте: val-обученные m1-модели на тестовом якоре (121 признак) ----
    for h in ["pz", "p", "s"]:
        out = wp(f"t_m1_{h}.npy")
        if not os.path.exists(out):
            mdl = lgb.Booster(model_file=wp(f"v_m1_{h}_s1.txt"))
            np.save(out, mdl.predict(Xte[:, :121]).astype(np.float32))

    # ---- сборка v2 ----
    LV = lambda f: np.load(wp(f)).astype(np.float64)
    def fam(tag, seeds):
        pz = np.mean([LV(f"{tag}_pz_s{s}.npy") for s in seeds], axis=0)
        p = np.mean([LV(f"{tag}_p_s{s}.npy") for s in seeds], axis=0)
        sz = np.mean([LV(f"{tag}_s_s{s}.npy") for s in seeds], axis=0)
        return pz, p * sz
    vpz, vtwo = fam("v_m2", M2_SEEDS)
    vtw = np.mean([LV(f"v_m2_twlog_s{s}.npy") for s in TW_SEEDS], axis=0)
    m1pz, m1two = LV("v_m1_pz_s1.npy"), LV("v_m1_p_s1.npy") * LV("v_m1_s_s1.npy")
    v2_val = 0.55 * (0.4 * vpz + 0.6 * vtwo) + 0.25 * vtw + 0.2 * (0.5 * m1pz + 0.5 * m1two)
    vp = np.mean([LV(f"v_m2_p_s{s}.npy") for s in M2_SEEDS], axis=0)
    G_ = np.load(wp("gmv_mat.npy")); zh = np.log1p(G_[:, :-1]).ravel()
    TAIL = float((zh[zh > 8.5] - 8.5).mean())
    ts_ = np.array([0.0] + LADDER_T)
    def quad(cols_):
        S = np.minimum.accumulate(np.column_stack(cols_), axis=1)
        return ((S[:, :-1] + S[:, 1:]) * 0.5 * np.diff(ts_)).sum(axis=1) + S[:, -1] * TAIL
    lad_v = quad([vp] + [LV(f"v_lad_t{t}.npy") for t in LADDER_T])
    val_mix = 0.6 * v2_val + 0.4 * lad_v
    tpz, ttwo = fam("t_m2", M2_SEEDS)
    ttw = np.mean([LV(f"t_m2_twlog_s{s}.npy") for s in TW_SEEDS], axis=0)
    tm1 = 0.5 * LV("t_m1_pz.npy") + 0.5 * (LV("t_m1_p.npy") * LV("t_m1_s.npy"))
    v2_test = 0.55 * (0.4 * tpz + 0.6 * ttwo) + 0.25 * ttw + 0.2 * tm1
    tp = np.mean([LV(f"t_m2_p_s{s}.npy") for s in M2_SEEDS], axis=0)
    lad_t = quad([tp] + [LV(f"t_lad_t{t}.npy") for t in LADDER_T])
    test_mix = 0.6 * v2_test + 0.4 * lad_t
    papp = LV("papp_409.npy")

    users = pl.read_parquet(wp("users_order.parquet"))["user_id"].cast(pl.Int64)
    outdir = WORK.parent / "preds_repro"; outdir.mkdir(exist_ok=True)
    for nm, arr in [("kostya46_val", val_mix), ("kostya46_test", test_mix),
                    ("kostya46shade_val", val_mix), ("kostya46shade_test", test_mix * papp)]:
        pl.DataFrame({"user_id": users, "pred": np.expm1(np.maximum(arr, 0))}
                     ).write_parquet(str(outdir / f"{nm}.parquet"))
    print("DONE ->", outdir, flush=True)

if __name__ == "__main__":
    main()
