"""Секвенсная модель на GPU Colab. Самодостаточный файл: ничего из репозитория не импортирует.

ЗАЧЕМ. Локально мы упёрлись в железо: обучаем на 112 днях истории из 409 доступных,
двумя слоями, на восьми срезах из двадцати трёх, и строго в один поток, потому что
ноутбук падал. Единственное направление с потенциалом порядка 0.003 — секвенсная модель
с настоящей ёмкостью. Здесь она и обучается.

ОПЫТ ПОСТРОЕН КАК ДВЕ РУКИ, а не как один прогон:
    small : L=112, d=96,  слоёв 2  — наша нынешняя конфигурация
    big   : L=364, d=256, слоёв 6  — то, ради чего берётся GPU
Один и тот же код, одни и те же сиды и срезы. Разница между руками и есть ответ на
вопрос «покупает ли ёмкость что-нибудь», без сравнения с локальными прогонами, где
отличается ещё десяток мелочей.

ЧТО ПЕРЕНЕСЕНО ИЗ ОСНОВНОГО ПРОЕКТА И ПОЧЕМУ ЭТО ВАЖНО:
  * зазор 30 дней: обучающий срез берётся, только если его целевое окно кончается
    не позже валидационного якоря. Без этого валидационный скор завышен на 0.05-0.10;
  * ранняя остановка ПО КАЛИБРОВАННОМУ скору, а не по сырому. На секвенсах это стоило
    0.0028 — сырой скор выбирает не ту точку, потому что калибровка переписывает уровень;
  * усреднение поздних контрольных точек: кривая калиброванного скора не монотонна,
    ранняя остановка берёт максимум шума, а не его среднее.

ЗАПУСК НА COLAB (файл читает /content/train.parquet и /content/sample_submit.csv):
    python gpu_seq.py build --arm big
    python gpu_seq.py train --arm big --seed 42 --name gseq_big_s42
Выход: /content/out/ИМЯ_val.parquet и ИМЯ_test.parquet, колонки user_id + pred в
исходном масштабе GMV — ровно то, что читает наш пул моделей.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(os.environ.get("GPU_SEQ_ROOT", "/content"))
SEQ_BASE = ROOT / "seq"
OUT = ROOT / "out"
TRAIN_PARQUET = ROOT / "train.parquet"
SAMPLE_SUBMIT = ROOT / "sample_submit.csv"

VAL_ANCHOR = date(2026, 1, 14)     # цель 2026-01-15..2026-02-13, наблюдаема
TEST_ANCHOR = date(2026, 2, 13)    # цель 2026-02-14..2026-03-15, её и сдаём
DATA_END = date(2026, 2, 13)
N_USERS = 250_000
C = 12

# распаковка: значение = сохранённый_uint8 * SCALES[c]
SCALES = [0.05, 0.05, 1.0, 1.0, 0.05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

ARMS = {
    # L, d_model, слоёв, голов, ширина полносвязного, срезов
    "small": dict(L=112, d=96, layers=2, heads=4, ff=192, n_train=8),
    # срезов 8, а не 10, по двум причинам. Первая: 10 срезов это 10.9 ГБ обучающих
    # тензоров при 12 ГБ памяти — не помещается, и каждый шаг уходил на диск (замер
    # 20.08: загрузка GPU 6%, процессы по 21% CPU, то есть ожидание ввода-вывода).
    # При 8 срезах это 8.7 ГБ, всё лежит в кэше. Вторая: у малой руки тоже 8, поэтому
    # между руками остаётся ровно два отличия — длина истории и размер модели.
    "big": dict(L=364, d=256, layers=6, heads=8, ff=512, n_train=8),
}


def arm_cfg(name: str) -> dict:
    if name not in ARMS:
        sys.exit(f"неизвестная рука {name}, есть: {list(ARMS)}")
    return ARMS[name]


def seq_dir(arm: str) -> Path:
    """У каждой руки СВОЯ папка. Иначе вторая сборка видит файл с тем же именем,
    пропускает его как готовый и молча берёт чужую длину истории."""
    return SEQ_BASE / arm


def clean_anchors(n: int, stride: int = 14) -> list[date]:
    """Срезы с зазором 30 дней: целевое окно кончается не позже валидационного якоря."""
    out = []
    i = 1
    while len(out) < n and i < 60:
        a = VAL_ANCHOR - timedelta(days=stride * i)
        if a + timedelta(days=30) <= VAL_ANCHOR:
            out.append(a)
        i += 1
    return sorted(out)


def universe() -> pl.DataFrame:
    return (pl.read_csv(SAMPLE_SUBMIT, schema_overrides={"user_id": pl.Int64})
            .select("user_id").sort("user_id"))


# ---------------- сборка тензоров ----------------

def qlog(x: np.ndarray) -> np.ndarray:
    """канал-логарифм -> uint8 с шагом 1/20 (максимум log1p в данных 11.2 -> 224)."""
    return np.clip(np.rint(np.log1p(x) * 20.0), 0, 255).astype(np.uint8)


def qcnt(x: np.ndarray, cap: int) -> np.ndarray:
    return np.minimum(x, cap).astype(np.uint8)


def build_anchor(lf, uni, row_of, anchor: date, L: int, sd: Path) -> None:
    t0 = time.time()
    p = sd / f"a{anchor.isoformat()}.npy"
    if p.exists():
        print(f"  {anchor}: уже есть, пропуск", flush=True)
        return
    win = (lf.filter((pl.col("event_date") <= anchor)
                     & (pl.col("event_date") > anchor - timedelta(days=L)))
           .select("user_id", "event_date", "gmv_search", "gmv_cat", "to_ord", "to_cart",
                   "searches", "search", "cat", "has_search_to_ord", "has_cat_to_ord",
                   "search_to_cart", "search_to_ord", "cat_to_cart", "cat_to_ord"))
    df = win.collect(engine="streaming")
    uidx = df["user_id"].replace_strict(row_of, return_dtype=pl.Int32).to_numpy()
    days_ago = df.select((pl.lit(anchor) - pl.col("event_date"))
                         .dt.total_days().alias("d"))["d"].to_numpy()
    didx = (L - 1) - days_ago
    assert didx.min() >= 0 and didx.max() < L, f"{anchor}: день вне диапазона"

    arr = np.zeros((N_USERS, L, C), dtype=np.uint8)
    arr[uidx, didx, 0] = qlog(df["gmv_search"].to_numpy())
    arr[uidx, didx, 1] = qlog(df["gmv_cat"].to_numpy())
    arr[uidx, didx, 2] = qcnt(df["to_ord"].to_numpy(), 10)
    arr[uidx, didx, 3] = qcnt(df["to_cart"].to_numpy(), 20)
    arr[uidx, didx, 4] = qlog(df["searches"].to_numpy())
    arr[uidx, didx, 5] = df["search"].to_numpy().astype(np.uint8)
    arr[uidx, didx, 6] = df["cat"].to_numpy().astype(np.uint8)
    arr[uidx, didx, 7] = ((df["has_search_to_ord"].to_numpy()
                           + df["has_cat_to_ord"].to_numpy()) > 0).astype(np.uint8)
    arr[uidx, didx, 8] = qcnt(df["search_to_cart"].to_numpy(), 20)
    arr[uidx, didx, 9] = qcnt(df["search_to_ord"].to_numpy(), 10)
    arr[uidx, didx, 10] = qcnt(df["cat_to_cart"].to_numpy(), 20)
    arr[uidx, didx, 11] = qcnt(df["cat_to_ord"].to_numpy(), 10)
    np.save(p, arr)
    del arr, df

    # цель: суммарный GMV за 30/7/14 дней после якоря; для теста её нет
    if anchor + timedelta(days=30) <= DATA_END:
        tgts = []
        for h in (30, 7, 14):
            t = (lf.filter(pl.col("event_date").is_between(
                    anchor + timedelta(days=1), anchor + timedelta(days=h)))
                 .group_by("user_id").agg(pl.col("gmv").sum().alias("t"))
                 .collect(engine="streaming"))
            v = uni.join(t, on="user_id", how="left").with_columns(pl.col("t").fill_null(0.0))
            tgts.append(v["t"].to_numpy().astype(np.float32))
        np.save(sd / f"a{anchor.isoformat()}.target.npy", np.stack(tgts, axis=1))
    print(f"  {anchor}: готово за {time.time()-t0:.1f}с", flush=True)


def cmd_build(args) -> None:
    cfg = arm_cfg(args.arm)
    L = cfg["L"]
    sd = seq_dir(args.arm)
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "meta.json").write_text(json.dumps({"L": L, "C": C, "arm": args.arm,
                                              "scales": SCALES}, indent=1))
    train = clean_anchors(cfg["n_train"])
    anchors = [TEST_ANCHOR, VAL_ANCHOR] + train
    gb = N_USERS * L * C / 1e9
    print(f"рука {args.arm}: L={L} C={C} -> {gb:.2f} ГБ на срез, срезов {len(anchors)}, "
          f"итого {gb*len(anchors):.1f} ГБ", flush=True)
    print(f"обучающие срезы (зазор 30 дней): {[a.isoformat() for a in train]}", flush=True)

    uni = universe()
    row_of = {u: i for i, u in enumerate(uni["user_id"].to_list())}
    lf = pl.scan_parquet(TRAIN_PARQUET)
    for a in anchors:
        build_anchor(lf, uni, row_of, a, L, sd)
    print("сборка закончена", flush=True)


# ---------------- калибровка (побитово как в calibrate.py) ----------------

def fit_shifts(lp: np.ndarray, ly: np.ndarray, bins: int):
    qs = np.quantile(lp, np.linspace(0, 1, bins + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    centers, shifts = [], []
    for i in range(bins):
        m = (lp > qs[i]) & (lp <= qs[i + 1])
        if m.sum() < 500:          # пустые корзины пропускаются — иначе расхождение 0.0017
            continue
        centers.append(lp[m].mean())
        shifts.append(ly[m].mean() - lp[m].mean())
    return np.array(centers), np.array(shifts)


def apply_shifts(lp: np.ndarray, centers: np.ndarray, shifts: np.ndarray):
    return np.clip(lp + np.interp(lp, centers, shifts), 0, None)


def rmsle(y_true, y_pred) -> float:
    lt = np.log1p(np.clip(np.asarray(y_true, dtype=np.float64), 0, None))
    lp = np.log1p(np.clip(np.asarray(y_pred, dtype=np.float64), 0, None))
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


def cal_rmsle_2fold(pred_log, ly, y_raw, half, bins=24) -> float:
    """Честный калиброванный скор: половина A настраивает сдвиги для B и наоборот,
    поэтому ни одна строка не калибруется сама собой."""
    lp = np.clip(np.asarray(pred_log, dtype=np.float64), 0, None)
    out = np.empty_like(lp)
    c_a, s_a = fit_shifts(lp[half], ly[half], bins)
    out[~half] = apply_shifts(lp[~half], c_a, s_a)
    c_b, s_b = fit_shifts(lp[~half], ly[~half], bins)
    out[half] = apply_shifts(lp[half], c_b, s_b)
    return rmsle(y_raw, np.expm1(out))


# ---------------- модель ----------------

def build_model(cfg: dict, dropout: float, device):
    import torch
    import torch.nn as nn

    L, d = cfg["L"], cfg["d"]
    n_tok = L // 7

    class SeqNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(C, d, kernel_size=7, stride=7)   # день -> недельный токен
            self.pos = nn.Parameter(torch.zeros(1, n_tok, d))
            nn.init.trunc_normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=cfg["heads"], dim_feedforward=cfg["ff"], dropout=0.1,
                activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=cfg["layers"],
                                             norm=nn.LayerNorm(d),
                                             enable_nested_tensor=False)
            blocks, prev = [], 2 * d
            for h in (384, 256):
                blocks += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(dropout)]
                prev = h
            self.trunk = nn.Sequential(*blocks)
            self.head_logit = nn.Linear(prev, 1)   # вероятность того, что GMV > 0
            self.head_mu = nn.Linear(prev, 1)      # ожидание log1p(GMV) при GMV > 0
            self.head_aux = nn.Linear(prev, 2)     # горизонты 7 и 14 дней

        def forward(self, xseq):                   # [B, L, C]
            h = self.conv(xseq.transpose(1, 2)).transpose(1, 2) + self.pos
            h = self.enc(h)
            z = self.trunk(torch.cat([h.mean(dim=1), h[:, -1]], dim=1))
            return (self.head_logit(z).squeeze(1), self.head_mu(z).squeeze(1),
                    self.head_aux(z))

    m = SeqNet().to(device)
    print(f"модель: L={L} d={d} слоёв={cfg['layers']} токенов={n_tok} "
          f"параметров={sum(p.numel() for p in m.parameters()):,}", flush=True)
    return m


def take_seq(mm, rows: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Строки из memmap uint8 -> float32 в исходном масштабе."""
    return mm[rows].astype(np.float32) * scales


def predict_log(model, mm, idx, device, scales, batch: int) -> np.ndarray:
    import torch
    model.eval()
    out = np.empty(len(idx), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            rows = idx[s:s + batch]
            xb = torch.from_numpy(take_seq(mm, rows, scales)).to(device)
            logit, mu, _ = model(xb)
            out[s:s + batch] = (torch.sigmoid(logit)
                                * torch.clamp(mu, min=0)).float().cpu().numpy()
    model.train()
    return out


class Prefetcher:
    """Читает батчи в фоновом потоке, пока карта считает предыдущий.

    ЗАЧЕМ. Замер 20.08: загрузка GPU 6% при средней нагрузке 3.4 на двух ядрах —
    карта простаивала 94% времени. Тензор большой руки 13 ГБ при 12 ГБ памяти, в
    кэш не помещается, и каждый шаг уходил на диск за 512 случайными строками
    СИНХРОННО: сначала ждём диск, потом считаем, потом снова ждём. Более мощная
    карта такого не лечит вообще — узкое место не в ней.

    Поток отдаёт готовые numpy-массивы через очередь ограниченной глубины: пока
    считается батч N, читается N+1. Глубина маленькая намеренно — каждый батч
    большой руки это 8.9 МБ после распаковки, и большая очередь съест память,
    которой и так не хватает.
    """

    def __init__(self, mm, tg, order, perm, batch, scales, depth=3):
        import queue
        import threading
        self.q = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self._args = (mm, tg, order, perm, batch, scales)
        self.t = threading.Thread(target=self._work, daemon=True)
        self.t.start()

    def _work(self):
        mm, tg, order, perm, batch, scales = self._args
        for a, s_ in order:
            if self.stop.is_set():
                break
            rows = np.sort(perm[a][s_:s_ + batch])
            if len(rows) < 8:      # хвостовой огрызок среза: пропускаем, а не кладём
                continue           # пустышку, иначе цикл обучения упадёт на распаковке
            self.q.put((take_seq(mm[a], rows, scales), tg[a][rows]))
        self.q.put("КОНЕЦ")

    def __iter__(self):
        while True:
            item = self.q.get()
            if isinstance(item, str):
                return
            yield item

    def close(self):
        self.stop.set()
        try:
            while True:
                self.q.get_nowait()
        except Exception:
            pass


def epoch_order(seed: int, ep: int, anchors: list, batch: int):
    """Порядок ОДНОЙ эпохи, однозначно определяемый её номером.

    Зачем именно так. Обучение эпохами лучше случайной выборки с возвратом: каждая
    строка попадает ровно один раз за эпоху, тогда как при выборке с возвратом около
    5% пар не попадают ни разу, а перемешивание без возврата вдобавок сходится не хуже.
    Но эпохи мешают продолжению с середины — если не знать порядка. Здесь генератор
    засеян номером эпохи, поэтому по номеру шага восстанавливаются и эпоха, и смещение
    внутри неё, и порядок получается тот же самый. Продолжение точное, качество эпох
    сохранено.
    """
    r = np.random.default_rng(seed * 1_000_003 + ep)
    perm = {a: r.permutation(N_USERS) for a in anchors}
    order = [(a, s) for a in anchors for s in range(0, N_USERS, batch)]
    r.shuffle(order)
    return perm, order


def dump(model, mm, tg_idx, device, scales, args, ck_preds_val, ck_scores,
         cfg, step, total, y_val, ly_val, half, opt, sched, t0):
    """Записать предсказания и полное состояние НА ДИСК МАШИНЫ.

    Зачем это есть. Прогон 19.08 шёл 2ч40м и был потерян целиком, когда Colab забрал
    машину: всё жило только в её памяти. Теперь каждые --dump-every шагов на диск
    ложатся и предсказания, и состояние для продолжения, а локальный опросчик их
    забирает. Потеря машины стоит одного промежутка, а не всего прогона.
    """
    import torch
    pv = np.mean(np.stack(ck_preds_val), axis=0)
    sc = cal_rmsle_2fold(pv, ly_val, y_val, half)
    pt = predict_log(model, mm[TEST_ANCHOR], tg_idx, device, scales, args.eval_batch)
    uid = universe()["user_id"].to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"user_id": uid, "pred": np.expm1(np.clip(pv, 0, None))}
                 ).write_parquet(OUT / f"{args.name}_val.parquet")
    pl.DataFrame({"user_id": uid, "pred": np.expm1(np.clip(pt, 0, None))}
                 ).write_parquet(OUT / f"{args.name}_test.parquet")
    (OUT / f"{args.name}.json").write_text(json.dumps({
        "arm": args.arm, "seed": args.seed, "step": step, "total": total,
        "cal_rmsle_last": ck_scores[-1], "cal_rmsle_best": min(ck_scores),
        "cal_rmsle_ckptavg": sc, "curve": ck_scores, "cfg": cfg,
        "epochs": args.epochs, "lr": args.lr, "done": step >= total,
        "minutes": (time.time() - t0) / 60}, indent=1))
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "step": step, "scores": ck_scores,
                "ck_preds_val": np.stack(ck_preds_val).astype(np.float32)},
               ROOT / f"{args.name}.ckpt")
    print(f"  ВЫГРУЗКА на шаге {step}: усреднённый скор {sc:.6f}, файлы записаны",
          flush=True)


def cmd_train(args) -> None:
    import torch
    import torch.nn.functional as F

    cfg = arm_cfg(args.arm)
    L = cfg["L"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ВНИМАНИЕ: GPU не виден, обучение на процессоре будет очень долгим", flush=True)
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    scales = np.array(SCALES, dtype=np.float32)

    sd = seq_dir(args.arm)
    train_anchors = clean_anchors(cfg["n_train"])
    mm = {a: np.load(sd / f"a{a.isoformat()}.npy", mmap_mode="r")
          for a in train_anchors + [VAL_ANCHOR, TEST_ANCHOR]}
    tg = {a: np.load(sd / f"a{a.isoformat()}.target.npy")
          for a in train_anchors + [VAL_ANCHOR]}
    for a in mm:
        assert mm[a].shape == (N_USERS, L, C), f"{a}: форма {mm[a].shape}, ожидалась L={L}"

    y_val = tg[VAL_ANCHOR][:, 0].astype(np.float64)
    ly_val = np.log1p(y_val)
    rng = np.random.default_rng(0)
    half = rng.random(N_USERS) < 0.5          # деление на половины фиксировано сидом 0
    all_idx = np.arange(N_USERS)

    model = build_model(cfg, args.dropout, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    _, _order0 = epoch_order(args.seed, 0, train_anchors, args.batch)
    steps_per_epoch = len(_order0)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    ck_preds_val: list[np.ndarray] = []
    ck_scores: list[float] = []
    step = 0
    ck_path = ROOT / f"{args.name}.ckpt"
    if args.resume and ck_path.exists():
        st = torch.load(ck_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); step = int(st["step"])
        ck_scores = list(st["scores"])
        ck_preds_val = [a for a in np.asarray(st["ck_preds_val"])]
        print(f"продолжаю с шага {step}/{total}, скоров в истории {len(ck_scores)}",
              flush=True)

    print(f"срезов {len(train_anchors)}, шагов за эпоху {steps_per_epoch}, всего {total}; "
          f"замер каждые {args.eval_every}, выгрузка каждые {args.dump_every}", flush=True)

    t0 = time.time()
    ep, off = divmod(step, steps_per_epoch)
    while step < total:
        perm, order = epoch_order(args.seed, ep, train_anchors, args.batch)
        pf = Prefetcher(mm, tg, order[off:], perm, args.batch, scales)
        for xs, yt in pf:
            xb = torch.from_numpy(xs).to(device, non_blocking=True)
            yb = torch.from_numpy(np.log1p(yt.astype(np.float32))).to(device)
            bb = (yb[:, 0] > 0).float()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logit, mu, aux = model(xb)
                bce = F.binary_cross_entropy_with_logits(logit, bb)
                pos = bb > 0
                mse_pos = (F.mse_loss(mu[pos], yb[pos, 0]) if pos.any()
                           else torch.zeros((), device=device))
                l7 = F.mse_loss(aux[:, 0], yb[:, 1])
                l14 = F.mse_loss(aux[:, 1], yb[:, 2])
                loss = args.bce_w * bce + mse_pos + args.aux_w * (l7 + l14)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1

            if step % args.eval_every == 0 or step == total:
                pv = predict_log(model, mm[VAL_ANCHOR], all_idx, device, scales,
                                 args.eval_batch)
                sc = cal_rmsle_2fold(pv, ly_val, y_val, half)
                ck_scores.append(sc)
                # держим последние K точек: усреднение снимает шум выбора одной точки
                ck_preds_val.append(pv.astype(np.float32))
                if len(ck_preds_val) > args.ckpt_avg:
                    ck_preds_val.pop(0)
                print(f"шаг {step}/{total} эпоха {ep} калиброванный скор {sc:.6f} "
                      f"({time.time()-t0:.0f}с)", flush=True)

            if step % args.dump_every == 0 or step == total:
                dump(model, mm, all_idx, device, scales, args, ck_preds_val, ck_scores,
                     cfg, step, total, y_val, ly_val, half, opt, sched, t0)
            if step >= total:
                break
        pf.close()
        off = 0
        ep += 1

    print(f"последняя точка {ck_scores[-1]:.6f}, лучшая {min(ck_scores):.6f}", flush=True)
    print(f"готово, файлы в {OUT}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="собрать тензоры из train.parquet")
    b.add_argument("--arm", default="big", choices=list(ARMS))
    b.set_defaults(fn=cmd_build)

    t = sub.add_parser("train", help="обучить и записать предсказания")
    t.add_argument("--arm", default="big", choices=list(ARMS))
    t.add_argument("--name", required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--eval-batch", type=int, default=2048)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--wd", type=float, default=0.01)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--bce-w", type=float, default=0.7)
    t.add_argument("--aux-w", type=float, default=0.3)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--ckpt-avg", type=int, default=5,
                   help="сколько последних контрольных точек усреднять")
    t.add_argument("--dump-every", type=int, default=1500,
                   help="каждые сколько шагов выкладывать предсказания и состояние на "
                        "диск машины: потеря машины тогда стоит одного промежутка")
    t.add_argument("--resume", action="store_true",
                   help="продолжить с сохранённого состояния, если оно есть")
    t.set_defaults(fn=cmd_train)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
