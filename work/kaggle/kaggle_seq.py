"""Событийный трансформер на Kaggle. Самодостаточный файл: из репозитория ничего не импортирует.

ЗАЧЕМ ИМЕННО ЭТО ПРЕДСТАВЛЕНИЕ. Доказанное тождество (KNOWLEDGE.md) оставляет один класс
с ненулевой перспективой: модель, чей прогноз не является функцией 203 табличных агрегатов.
На Colab уже обучается недельно-токенный трансформер (день -> Conv1d(7,7) -> 52 токена).
Здесь обучается ДРУГОЕ представление тех же данных: токен = активный день пользователя
(строка train.parquet) со всеми 15 счётчиками, явным интервалом до предыдущего события,
расстоянием до якоря и календарной фазой. Недельная свёртка уничтожает внутринедельные
интервалы и смежность событий; агрегаты уничтожают их порядок. Событийная последовательность
сохраняет и то и другое — это и есть ответ на вопрос «чем прогноз принципиально не является
функцией наших признаков».

ЧЕМ ЭТО РЕШАЕТ ЗАМЕРЕННУЮ ПРОБЛЕМУ ВВОДА-ВЫВОДА. На Colab тензоры дублируют календарь по
срезам: 10 срезов x 364 дня x 12 каналов = 13 ГБ при 12 ГБ памяти, GPU был занят на 6%.
Здесь хранилище ОДНО на все срезы: 30.6 млн событий x (15 каналов uint8 + день int16) =
0.6 ГБ. Оно целиком лежит в памяти GPU, окно любого якоря — это пара индексов
(start, end) на пользователя, сборка батча происходит на карте. Ввода-вывода в шаге
обучения нет вообще, число срезов бесплатно: 24 недельных вместо 8 двухнедельных.

ПРОТОКОЛ ПЕРЕНЕСЁН ИЗ ПРОЕКТА ПОЛНОСТЬЮ:
  * зазор 30 дней: целевое окно каждого обучающего среза кончается не позже 2026-01-14;
  * сравнение и ранняя диагностика ТОЛЬКО по калиброванному скору (2-fold, побитово как
    calibrate.py); сырой скор обманул проект девять раз;
  * валидационные И тестовые предсказания пишутся с ОДНИХ весов на ОДНОЙ сетке выгрузок
    и одинаково усредняются по последним K выгрузкам — ловушка «одинаковый val, разный
    test» исключена по построению;
  * контрольные точки на диск: потеря сессии стоит одного промежутка, не прогона.

ЗАПУСК НА KAGGLE (T4 x2; данные — Kaggle Dataset с train.parquet и sample_submit.csv):
    python kaggle_seq.py build
    python kaggle_seq.py train --name kevf_s42 --seed 42 --device cuda:0
    # параллельно на второй карте: --name kevf_s1337 --seed 1337 --device cuda:1
Выход: <ROOT>/out/ИМЯ_val.parquet и ИМЯ_test.parquet (user_id, pred в исходном GMV) —
ровно то, что принимает work/colab/ingest.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

# ---------------- пути: Kaggle или локальный смоук ----------------

def _default_root() -> Path:
    p = Path("/kaggle/working")
    return p if p.exists() else Path(__file__).resolve().parent / "run"


def _find_input(name: str) -> Path:
    """Файл данных: сначала KSEQ_DATA, затем /kaggle/input на глубину до 3 уровней
    (датасет может хранить файлы во вложенной папке), затем корень репозитория
    (локальный смоук). Если не нашли — падаем ГРОМКО и показываем, что смонтировано:
    молчаливый фолбэк один раз уже превратился в невнятный '/sample_submit.csv'."""
    cand: list[Path] = []
    env = os.environ.get("KSEQ_DATA")
    if env:
        cand.append(Path(env) / name)
    for depth in ("", "*/", "*/*/", "*/*/*/"):
        cand += [Path(p) for p in sorted(glob.glob(f"/kaggle/input/{depth}{name}"))]
    here = Path(__file__).resolve()
    if len(here.parents) >= 3:
        cand.append(here.parents[2] / name)
    for p in cand:
        if p.exists():
            return p
    ki = Path("/kaggle/input")
    listing = ("\n".join(f"  {p}" for p in sorted(ki.glob("*/**"))[:40]) or "  (пусто)"
               ) if ki.exists() else "  (нет /kaggle/input — это не Kaggle)"
    sys.exit(f"не найден {name}. Прикрепи датасет с train.parquet и sample_submit.csv "
             f"через Add Input и перезапусти ячейку. Сейчас в /kaggle/input:\n{listing}")

ROOT = Path(os.environ.get("KSEQ_ROOT", str(_default_root())))
STORE = ROOT / "store"
OUT = ROOT / "out"

DAY0 = date(2025, 1, 1)            # день 0; данные покрывают дни 0..408
VAL_ANCHOR = date(2026, 1, 14)     # цель 2026-01-15..2026-02-13, наблюдаема
TEST_ANCHOR = date(2026, 2, 13)    # цель 2026-02-14..2026-03-15, её и сдаём
N_USERS = 250_000
WINDOW = 364                       # окно истории (a-364, a], как у большой руки Colab
C = 15

# канал -> (колонка, кодирование). qlog: log1p*20 в uint8 (максимум log1p в данных 11.2).
# Распаковка при обучении: uint8 * SCALE. Для qlog это даёт ровно log1p(x) — сеть любит.
CHANNELS = [
    ("gmv_search", "qlog"), ("gmv_cat", "qlog"), ("searches", "qlog"),
    ("to_ord", "cap10"), ("to_cart", "cap20"),
    ("search", "cap255"), ("cat", "cap255"),
    ("has_search_to_cart", "flag"), ("has_search_to_ord", "flag"),
    ("has_cat_to_cart", "flag"), ("has_cat_to_ord", "flag"),
    ("search_to_cart", "cap20"), ("search_to_ord", "cap10"),
    ("cat_to_cart", "cap20"), ("cat_to_ord", "cap10"),
]
SCALES = {"qlog": 0.05, "cap10": 0.2, "cap20": 0.1, "cap255": 0.05, "flag": 1.0}


def d2i(d: date) -> int:
    return (d - DAY0).days


def train_anchors(n: int, stride: int) -> list[date]:
    """Срезы с зазором 30: целевое окно кончается не позже валидационного якоря."""
    last = VAL_ANCHOR - timedelta(days=30)          # 2025-12-15, самый поздний допустимый
    out = [last - timedelta(days=stride * k) for k in range(n)]
    assert all(a + timedelta(days=30) <= VAL_ANCHOR for a in out)
    assert min(out) >= date(2025, 5, 1), "якорь слишком ранний: истории почти нет"
    return sorted(out)


def universe_ids() -> np.ndarray:
    import polars as pl
    sub = pl.read_csv(_find_input("sample_submit.csv"),
                      schema_overrides={"user_id": pl.Int64})
    return np.sort(sub["user_id"].to_numpy())


# ---------------- сборка хранилища событий ----------------

def encode(col: np.ndarray, kind: str) -> np.ndarray:
    if kind == "qlog":
        return np.clip(np.rint(np.log1p(col) * 20.0), 0, 255).astype(np.uint8)
    cap = {"cap10": 10, "cap20": 20, "cap255": 255, "flag": 1}[kind]
    return np.clip(col, 0, cap).astype(np.uint8)


def cmd_build(args) -> None:
    import polars as pl
    t0 = time.time()
    STORE.mkdir(parents=True, exist_ok=True)
    uid = universe_ids()
    row_of = {int(u): i for i, u in enumerate(uid)}

    cols = ["user_id", "event_date", "gmv"] + [c for c, _ in CHANNELS]
    df = (pl.scan_parquet(_find_input("train.parquet")).select(cols)
          .collect(engine="streaming"))
    print(f"прочитано {df.height:,} строк за {time.time()-t0:.0f}с", flush=True)

    uidx = df["user_id"].replace_strict(row_of, return_dtype=pl.Int32).to_numpy()
    didx = df.select((pl.col("event_date") - pl.lit(DAY0)).dt.total_days()
                     .cast(pl.Int32).alias("d"))["d"].to_numpy()
    assert didx.min() >= 0 and didx.max() <= d2i(TEST_ANCHOR), "день вне диапазона данных"
    gmv = df["gmv"].to_numpy().astype(np.float64)
    ch_raw = [df[c].to_numpy() for c, _ in CHANNELS]
    del df

    # порядок (пользователь, день): окно якоря = непрерывный отрезок событий пользователя
    order = np.lexsort((didx, uidx))
    uidx, didx, gmv = uidx[order], didx[order], gmv[order]
    ch = np.stack([encode(c[order], kind) for c, (_, kind) in zip(ch_raw, CHANNELS)],
                  axis=1)
    del ch_raw, order
    n_ev = len(uidx)

    # границы окна (a-WINDOW, a] на пользователя: searchsorted по ключу uidx*512+didx
    key = uidx.astype(np.int64) * 512 + didx
    anchors = train_anchors(args.n_anchors, args.stride) + [VAL_ANCHOR, TEST_ANCHOR]
    u = np.arange(N_USERS, dtype=np.int64) * 512
    bounds = np.empty((len(anchors), 2, N_USERS), dtype=np.int64)
    for i, a in enumerate(anchors):
        ad = d2i(a)
        bounds[i, 0] = np.searchsorted(key, u + max(ad - WINDOW, 0) + 1)   # первый в окне
        bounds[i, 1] = np.searchsorted(key, u + ad + 1)                    # за последним
    print(f"события {n_ev:,}, границы посчитаны ({time.time()-t0:.0f}с)", flush=True)

    # цели: точный GMV из сырых данных (не из квантованных каналов) за 30/7/14 дней
    dsort = np.argsort(didx, kind="stable")
    dd, du, dg = didx[dsort], uidx[dsort], gmv[dsort]
    tgt = np.zeros((len(anchors) - 1, N_USERS, 3), dtype=np.float32)   # без теста
    for i, a in enumerate(anchors[:-1]):
        ad = d2i(a)
        for k, h in enumerate((30, 7, 14)):
            lo, hi = np.searchsorted(dd, ad + 1), np.searchsorted(dd, ad + h + 1)
            tgt[i, :, k] = np.bincount(du[lo:hi], weights=dg[lo:hi],
                                       minlength=N_USERS).astype(np.float32)
    del dsort, dd, du, dg

    np.save(STORE / "events_day.npy", didx.astype(np.int16))
    np.save(STORE / "events_ch.npy", ch)
    np.save(STORE / "bounds.npy", bounds)
    np.save(STORE / "targets.npy", tgt)
    np.save(STORE / "user_ids.npy", uid)
    (STORE / "meta.json").write_text(json.dumps({
        "n_events": int(n_ev), "window": WINDOW, "channels": [c for c, _ in CHANNELS],
        "scales": [SCALES[k] for _, k in CHANNELS],
        "anchors": [a.isoformat() for a in anchors],
        "n_train": args.n_anchors, "stride": args.stride}, indent=1))
    gb = (didx.nbytes // 2 + ch.nbytes) / 1e9
    print(f"хранилище готово: {gb:.2f} ГБ событий, {len(anchors)-2} обучающих срезов "
          f"({anchors[0]}..{anchors[-3]}), всё за {time.time()-t0:.0f}с", flush=True)


# ---------------- калибровка (побитово как calibrate.py) ----------------

def fit_shifts(lp, ly, bins):
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
    if not centers:   # крошечная выборка (смоук): все бины ниже порога -> глобальный сдвиг
        return np.array([lp.mean()]), np.array([ly.mean() - lp.mean()])
    return np.array(centers), np.array(shifts)


def apply_shifts(lp, centers, shifts):
    return np.clip(lp + np.interp(lp, centers, shifts), 0, None)


def rmsle(y_true, y_pred) -> float:
    lt = np.log1p(np.clip(np.asarray(y_true, dtype=np.float64), 0, None))
    lp = np.log1p(np.clip(np.asarray(y_pred, dtype=np.float64), 0, None))
    return float(np.sqrt(np.mean((lt - lp) ** 2)))


def cal_rmsle_2fold(pred_log, ly, y_raw, half, bins=24) -> float:
    """Половина A настраивает сдвиги для B и наоборот: ни одна строка не калибруется собой."""
    lp = np.clip(np.asarray(pred_log, dtype=np.float64), 0, None)
    out = np.empty_like(lp)
    c_a, s_a = fit_shifts(lp[half], ly[half], bins)
    out[~half] = apply_shifts(lp[~half], c_a, s_a)
    c_b, s_b = fit_shifts(lp[~half], ly[~half], bins)
    out[half] = apply_shifts(lp[half], c_b, s_b)
    return rmsle(y_raw, np.expm1(out))


# ---------------- данные на карте ----------------

class GpuStore:
    """Всё хранилище событий в памяти устройства; сборка окна любого якоря — на карте.

    Токены отдаются В ОБРАТНОМ ПОРЯДКЕ (позиция 0 = самое свежее событие): маска валидности
    тогда просто j < len, а позиционная эмбеддинг-таблица означает «сколько событий назад»,
    что одинаково осмысленно на всех якорях.
    """

    def __init__(self, device, users: int):
        import torch
        meta = json.loads((STORE / "meta.json").read_text())
        self.meta = meta
        self.anchors = [date.fromisoformat(a) for a in meta["anchors"]]
        self.n_train = int(meta["n_train"])
        self.a_days = torch.tensor([d2i(a) for a in self.anchors], device=device)
        self.day = torch.from_numpy(np.load(STORE / "events_day.npy")).to(device)
        self.ch = torch.from_numpy(np.load(STORE / "events_ch.npy")).to(device)
        b = np.load(STORE / "bounds.npy")[:, :, :users]
        self.starts = torch.from_numpy(b[:, 0]).to(device)
        self.ends = torch.from_numpy(b[:, 1]).to(device)
        self.tgt = np.load(STORE / "targets.npy")[:, :users]
        self.scale = torch.tensor(meta["scales"], device=device)
        self.device = device
        self.users = users

    def batch(self, a_ids, u_ids, lmax: int):
        """a_ids, u_ids: int64-тензоры [B] на устройстве -> (feats [B,L,F], valid [B,L])."""
        import torch
        e = self.ends[a_ids, u_ids]
        s = self.starts[a_ids, u_ids]
        ln = (e - s).clamp_(max=lmax)                       # длиннее lmax -> старые отпадают
        j = torch.arange(lmax, device=self.device)
        valid = j.unsqueeze(0) < ln.unsqueeze(1)            # [B, L]
        src = (e - 1).unsqueeze(1) - j                      # обратный порядок
        src = src.masked_fill(~valid, 0)
        day = self.day[src].to(torch.float32)               # [B, L]
        ch = self.ch[src].to(torch.float32) * self.scale    # [B, L, C]

        a_day = self.a_days[a_ids].to(torch.float32).unsqueeze(1)
        wstart = a_day - WINDOW
        prev = torch.cat([day[:, 1:], day[:, :1]], dim=1)   # j+1 = предыдущее по времени
        has_prev = (j + 1).unsqueeze(0) < ln.unsqueeze(1)
        dt_prev = torch.where(has_prev, day - prev, day - wstart).clamp_(min=0)
        dt_anchor = (a_day - day).clamp_(min=0)
        wd = ((day + 2) % 7).long()                         # 2025-01-01 — среда
        wd1h = torch.nn.functional.one_hot(wd, 7).to(torch.float32)
        phase = day * (2 * np.pi / 365.0)
        feats = torch.cat([
            ch,
            torch.log1p(dt_prev).unsqueeze(2) / 6.0,
            torch.log1p(dt_anchor).unsqueeze(2) / 6.0,
            (dt_anchor / WINDOW).unsqueeze(2),
            wd1h,
            torch.sin(phase).unsqueeze(2), torch.cos(phase).unsqueeze(2),
        ], dim=2)
        return feats * valid.unsqueeze(2), valid

    F_DIM = C + 3 + 7 + 2


# ---------------- модель ----------------

def build_model(args, device):
    import torch
    import torch.nn as nn

    d, lmax = args.d, args.lmax

    class EventNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.Linear(GpuStore.F_DIM, d), nn.LayerNorm(d))
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.pos = nn.Parameter(torch.zeros(1, lmax + 1, d))
            nn.init.trunc_normal_(self.cls, std=0.02)
            nn.init.trunc_normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=args.heads, dim_feedforward=args.ff, dropout=args.dropout,
                activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=args.layers,
                                             norm=nn.LayerNorm(d),
                                             enable_nested_tensor=False)
            blocks, prev = [], 3 * d
            for hdim in (384, 256):
                blocks += [nn.Linear(prev, hdim), nn.GELU(), nn.LayerNorm(hdim),
                           nn.Dropout(args.dropout)]
                prev = hdim
            self.trunk = nn.Sequential(*blocks)
            self.head_logit = nn.Linear(prev, 1)   # вероятность GMV > 0
            self.head_mu = nn.Linear(prev, 1)      # ожидание log1p(GMV) при GMV > 0
            self.head_aux = nn.Linear(prev, 2)     # горизонты 7 и 14 дней
            self.head_bins = nn.Linear(prev, 32)   # распределение log1p по корзинам

        def forward(self, feats, valid):
            B = feats.shape[0]
            h = self.proj(feats)
            h = torch.cat([self.cls.expand(B, 1, -1), h], dim=1) + self.pos
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=feats.device),
                             ~valid], dim=1)
            h = self.enc(h, src_key_padding_mask=pad)
            ev = h[:, 1:]
            cnt = valid.sum(dim=1, keepdim=True).clamp(min=1)
            hmean = (ev * valid.unsqueeze(2)).sum(dim=1) / cnt
            hlast = ev[:, 0] * (valid[:, :1]).to(ev.dtype)          # свежайшее событие
            z = self.trunk(torch.cat([h[:, 0], hmean, hlast], dim=1))
            return (self.head_logit(z).squeeze(1), self.head_mu(z).squeeze(1),
                    self.head_aux(z), self.head_bins(z))

    m = EventNet().to(device)
    print(f"модель: Lmax={lmax} d={d} слоёв={args.layers} "
          f"параметров={sum(p.numel() for p in m.parameters()):,}", flush=True)
    return m


def predict_log(model, store, a_id: int, device, batch: int) -> np.ndarray:
    import torch
    model.eval()
    n = store.users
    out = np.empty(n, dtype=np.float32)
    a = torch.full((batch,), a_id, dtype=torch.long, device=device)
    with torch.no_grad():
        for s_ in range(0, n, batch):
            u = torch.arange(s_, min(s_ + batch, n), device=device)
            feats, valid = store.batch(a[:len(u)], u, model.pos.shape[1] - 1)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logit, mu, _, _ = model(feats, valid)
            out[s_:s_ + len(u)] = (torch.sigmoid(logit.float())
                                   * torch.clamp(mu.float(), min=0)).cpu().numpy()
    model.train()
    return out


def epoch_order(seed: int, ep: int, n_anchors: int, users: int) -> np.ndarray:
    """Перестановка всех пар (якорь, пользователь), однозначная по номеру эпохи:
    продолжение с середины воспроизводит тот же порядок."""
    r = np.random.default_rng(seed * 1_000_003 + ep)
    return r.permutation(n_anchors * users)


def dump(model, store, args, state, device):
    """Val И test с ОДНИХ текущих весов, оба в скользящее среднее последних K выгрузок.
    Записываются усреднённые файлы + контрольная точка для продолжения."""
    import torch
    pv = predict_log(model, store, len(store.anchors) - 2, device, args.eval_batch)
    pt = predict_log(model, store, len(store.anchors) - 1, device, args.eval_batch)
    state["pv_hist"] = (state["pv_hist"] + [pv])[-args.pred_avg:]
    state["pt_hist"] = (state["pt_hist"] + [pt])[-args.pred_avg:]
    pv_avg = np.mean(np.stack(state["pv_hist"]), axis=0)
    pt_avg = np.mean(np.stack(state["pt_hist"]), axis=0)
    sc_file = cal_rmsle_2fold(pv_avg, state["ly_val"], state["y_val"], state["half"])

    import polars as pl
    OUT.mkdir(parents=True, exist_ok=True)
    uid = np.load(STORE / "user_ids.npy")[:store.users]
    pl.DataFrame({"user_id": uid, "pred": np.expm1(np.clip(pv_avg, 0, None))}
                 ).write_parquet(OUT / f"{args.name}_val.parquet")
    pl.DataFrame({"user_id": uid, "pred": np.expm1(np.clip(pt_avg, 0, None))}
                 ).write_parquet(OUT / f"{args.name}_test.parquet")
    (OUT / f"{args.name}.json").write_text(json.dumps({
        "arm": "event", "seed": args.seed, "step": state["step"], "total": state["total"],
        "cal_rmsle_last": state["curve"][-1] if state["curve"] else None,
        "cal_rmsle_best": min(state["curve"]) if state["curve"] else None,
        "cal_rmsle_file": sc_file, "curve": state["curve"],
        "cfg": {k: getattr(args, k) for k in
                ("d", "layers", "heads", "ff", "lmax", "batch", "lr", "epochs")},
        "n_anchors": store.n_train, "done": state["step"] >= state["total"],
        "minutes": (time.time() - state["t0"]) / 60}, indent=1))
    torch.save({"model": model.state_dict(), "opt": state["opt"].state_dict(),
                "sched": state["sched"].state_dict(),
                "scaler": state["scaler"].state_dict(), "step": state["step"],
                "curve": state["curve"],
                "pv_hist": np.stack(state["pv_hist"]).astype(np.float32),
                "pt_hist": np.stack(state["pt_hist"]).astype(np.float32)},
               ROOT / f"{args.name}.ckpt")
    print(f"  ВЫГРУЗКА на шаге {state['step']}: скор файла (среднее {len(state['pv_hist'])}"
          f" выгрузок) {sc_file:.6f}", flush=True)


def cmd_train(args) -> None:
    import torch
    import torch.nn.functional as F

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device
                          else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
        print(f"ВНИМАНИЕ: обучение на процессоре ({args.threads} потоков) — только смоук",
              flush=True)
    else:
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    users = args.users or N_USERS
    store = GpuStore(device, users)
    n_tr = store.n_train
    val_id, test_id = len(store.anchors) - 2, len(store.anchors) - 1
    assert store.anchors[val_id] == VAL_ANCHOR and store.anchors[test_id] == TEST_ANCHOR

    tg = torch.from_numpy(np.log1p(store.tgt[:n_tr])).to(device)   # [A, U, 3] в log1p
    assert store.tgt.shape[0] == n_tr + 1, "в targets.npy ожидались срезы + валидация"
    y_val = store.tgt[n_tr][:, 0].astype(np.float64)
    ly_val = np.log1p(y_val)
    half = np.random.default_rng(0).random(users) < 0.5   # то же деление, что в ingest.py
    bin_edges = torch.linspace(1e-6, 11.5, 31, device=device)

    model = build_model(args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = (n_tr * users + args.batch - 1) // args.batch
    total = steps_per_epoch * args.epochs
    if args.max_steps:
        total = min(total, args.max_steps)
    warm = min(args.warmup, max(total // 20, 1))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [torch.optim.lr_scheduler.LinearLR(opt, 0.05, 1.0, warm),
              torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total - warm, 1))],
        milestones=[warm])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    state = dict(step=0, total=total, curve=[], pv_hist=[], pt_hist=[], opt=opt,
                 sched=sched, scaler=scaler, y_val=y_val, ly_val=ly_val, half=half,
                 t0=time.time())
    ck_path = ROOT / f"{args.name}.ckpt"
    if args.resume and ck_path.exists():
        st = torch.load(ck_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        state["step"] = int(st["step"]); state["curve"] = list(st["curve"])
        state["pv_hist"] = [a for a in np.asarray(st["pv_hist"])]
        state["pt_hist"] = [a for a in np.asarray(st["pt_hist"])]
        print(f"продолжаю с шага {state['step']}/{total}", flush=True)

    print(f"срезов {n_tr}, пользователей {users:,}, шагов за эпоху {steps_per_epoch}, "
          f"всего {total}; замер каждые {args.eval_every}, выгрузка каждые "
          f"{args.dump_every}", flush=True)

    ep, off = divmod(state["step"], steps_per_epoch)
    while state["step"] < total:
        order = epoch_order(args.seed, ep, n_tr, users)
        for s_ in range(off * args.batch, n_tr * users, args.batch):
            idx = torch.from_numpy(order[s_:s_ + args.batch].astype(np.int64)).to(device)
            if len(idx) < 8:
                continue
            a_ids, u_ids = idx // users, idx % users
            feats, valid = store.batch(a_ids, u_ids, args.lmax)
            yb = tg[a_ids, u_ids]                              # [B, 3] log1p
            bb = (yb[:, 0] > 0).float()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logit, mu, aux, bins = model(feats, valid)
                bce = F.binary_cross_entropy_with_logits(logit, bb)
                pos = bb > 0
                mse_pos = (F.mse_loss(mu[pos], yb[pos, 0]) if pos.any()
                           else torch.zeros((), device=device))
                l_aux = F.mse_loss(aux[:, 0], yb[:, 1]) + F.mse_loss(aux[:, 1], yb[:, 2])
                l_bin = F.cross_entropy(bins, torch.bucketize(yb[:, 0].contiguous(),
                                                              bin_edges))
                loss = args.bce_w * bce + mse_pos + args.aux_w * l_aux + args.bin_w * l_bin
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            state["step"] += 1

            if state["step"] % args.eval_every == 0 or state["step"] == total:
                pv = predict_log(model, store, val_id, device, args.eval_batch)
                sc = cal_rmsle_2fold(pv, ly_val, y_val, half)
                state["curve"].append(sc)
                print(f"шаг {state['step']}/{total} эпоха {ep} калиброванный скор "
                      f"{sc:.6f} ({time.time()-state['t0']:.0f}с)", flush=True)
            if state["step"] % args.dump_every == 0 or state["step"] == total:
                dump(model, store, args, state, device)
            if state["step"] >= total:
                break
        off = 0
        ep += 1

    print(f"готово: последняя точка {state['curve'][-1]:.6f}, "
          f"лучшая {min(state['curve']):.6f}, файлы в {OUT}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="train.parquet -> хранилище событий + границы + цели")
    b.add_argument("--n-anchors", type=int, default=24)
    b.add_argument("--stride", type=int, default=7)
    b.set_defaults(fn=cmd_build)

    t = sub.add_parser("train", help="обучить и записать предсказания")
    t.add_argument("--name", required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--threads", type=int, default=2, help="потоков CPU для смоука")
    t.add_argument("--users", type=int, default=0, help="0 = все 250к; меньше — смоук")
    t.add_argument("--d", type=int, default=256)
    t.add_argument("--layers", type=int, default=6)
    t.add_argument("--heads", type=int, default=8)
    t.add_argument("--ff", type=int, default=512)
    t.add_argument("--lmax", type=int, default=320)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--eval-batch", type=int, default=1024)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--wd", type=float, default=0.01)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--bce-w", type=float, default=0.7)
    t.add_argument("--aux-w", type=float, default=0.3)
    t.add_argument("--bin-w", type=float, default=0.25)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--dump-every", type=int, default=1500)
    t.add_argument("--pred-avg", type=int, default=5,
                   help="сколько последних выгрузок усреднять в файлах val и test")
    t.add_argument("--max-steps", type=int, default=0, help="обрезать бюджет (смоук)")
    t.add_argument("--resume", action="store_true")
    t.set_defaults(fn=cmd_train)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
