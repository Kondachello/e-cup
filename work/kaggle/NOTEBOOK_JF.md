# Kaggle-сессия 3: joint fusion (kevf3) — событийный трансформер + таблица, T4×2

Цель: со-обучение событийного kevf с 196 табличными признаками по спеке
`work/reports/eve2_joint_fusion_design.md`. Тёплый старт с чекпойнтов сессии 2
(kevf_s42/kevf_s1337), два сида параллельно, ~5.5 ч из 9 доступных.
Гейт запуска (см. спеку): запас kevf > 0 уже подтверждён — сессия разрешена.

## Подготовка (один раз, локально уже сделано агентом)

1. `bash work/kaggle/make_bundle3.sh` → `/tmp/jf_bundle.zip` (kaggle_seq v3 +
   этот файл). Залить **новой версией** приватного Dataset **ozon-code**
   (Kaggle → Datasets → ozon-code → New Version; старый tfm3b_bundle.zip можно
   оставить рядом — ячейка 1 ищет именно jf_bundle.zip).
2. Табличные матрицы: `work/data/kaggle_session3/kaggle_tabfeats_wed_v1.zip`
   (~1.25 ГБ; собирает очередь, job `tabexport_wed`) → залить как НОВЫЙ приватный
   Dataset, например **ozon-tabfeats-wed** (New Dataset → Upload → этот zip;
   Kaggle сам распакует — внутри 26 `tabf16_*.npz` + `tabf16_meta.json`).
3. Датасет данных **ozon-ecup-t3** (train.parquet + sample_submit.csv) — уже есть.
4. Новый Notebook → Accelerator **GPU T4 x2** → Add Input, ЧЕТЫРЕ источника:
   - Datasets: **ozon-ecup-t3**, **ozon-code**, **ozon-tabfeats-wed**;
   - **Notebooks → Your Work → ноутбук сессии 2 → его Output** (там лежат
     `kevf_s42.ckpt`, `kevf_s1337.ckpt` и `out/kevf_s42.json` с cfg).
   Internet: можно Off (если polars уже в образе; иначе On для ячейки 1).

## Ячейка 1 — код и проверка входов

```python
import glob, os, shutil, subprocess, sys
os.chdir("/kaggle/working")

def find(pat):
    # раскладка /kaggle/input меняется — ищем рекурсивно на любой глубине
    return sorted(glob.glob(f"/kaggle/input/**/{pat}", recursive=True))

z = find("jf_bundle.zip")
tree = find("work/kaggle/kaggle_seq.py")
if z:
    subprocess.run(["unzip", "-oq", z[0], "-d", "/kaggle/working"], check=True)
    shutil.copy("work/kaggle/kaggle_seq.py", "kaggle_seq.py")
elif tree:
    shutil.copy(tree[0], "kaggle_seq.py")
else:
    for r, d, f in os.walk("/kaggle/input"):
        print(r, "->", f[:5])
        if r.count("/") - 2 >= 4: d[:] = []
    raise SystemExit("kaggle_seq.py не найден — смотри дерево выше (ozon-code подключён?)")
try:
    import polars
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "polars"], check=True)
assert find("train.parquet"), "нет train.parquet — Add Input: ozon-ecup-t3"
assert find("tabf16_meta.json"), "нет tabf16_meta.json — Add Input: ozon-tabfeats-wed"
n_npz = len(find("tabf16_*.npz"))
assert n_npz == 26, f"tabf16-файлов {n_npz}, а не 26 — датасет tabfeats неполный"
cks = find("kevf_s42.ckpt") + find("kevf_s1337.ckpt")
print("код: ок; таблица: 26 npz; тёплые чекпойнты:",
      cks or "НЕТ — сработает фолбэк с нуля (см. ячейку 3)")
```

## Ячейка 2 — сборка хранилища НА СЕТКЕ СРЕД (~2-5 мин, дождаться «хранилище готово»)

`--gap 35` обязателен: последний обучающий якорь 2025-12-10 (среда), ровно та
сетка, на которой существуют табличные экспорты. Без него TabStore упадёт с
«якоря таблицы != якоря хранилища».

```python
import subprocess, sys
subprocess.run([sys.executable, "kaggle_seq.py", "build", "--gap", "35"], check=True)
```

## Ячейка 3 — оба сида параллельно (Popen: `!cmd &` на Kaggle запрещён)

Арх. флаги сессии-источника подтягиваются автоматически из `<имя>.json` рядом с
чекпойнтом (автоподтяжка — страховка от дрейфа). ВАЖНО, 27.08: v1-чекпойнты
`kevf_s42/kevf_s1337` УТРАЧЕНЫ — их нет ни в Kaggle Output'ах, ни в локальных
архивах, ни в git (проверено полным поиском; kevf_s1337 существует только в
спеке). Тёплые базы теперь — чекпойнты сессии 2 из пакета `ozon_kevf_ckpt.zip`
(корень репо, 165 МБ): `kevf_v2.ckpt` (tfm + v2-флаги, кривая ~1.674) и
`kevf_gru.ckpt` (encoder=gru, кривая ~1.670); их json лежат рядом, arch_flags
ниже пробрасывает и `--encoder`. Для новой базы замените в launch() строку
`ck = find1(f"kevf_s{seed}.ckpt")` на явную базу (`find1("kevf_v2.ckpt")`).
Фолбэк: чекпойнт не найден или `WARM=False` → тот же запуск с нуля (lr 3e-4
вместо 2e-4).

```python
import glob, json, os, subprocess, sys

def find1(pat):
    r = sorted(glob.glob(f"/kaggle/input/**/{pat}", recursive=True))
    return r[0] if r else None

WARM = True          # False = принудительно с нуля (фолбэк --no-warm)
# РЕЖИМ ПО КВОТЕ (26.08: у Саши осталось 3 ч GPU/нед, сброс в субботу 00:00 UTC):
#   4500  — «короткая нога», ~2-2.5 ч сессии, влезает в 3 ч; тёплый старт делает
#           даже 4500 шагов полезными; продолжение потом через --resume;
#   12000 — полный прогон, ~5-5.5 ч — только при квоте >= 6 ч (чужой аккаунт
#           с расшаренными датасетами или после субботнего сброса).
MAX_STEPS = 4500

def arch_flags(ck):
    """Арх. флаги сессии-источника ДОСЛОВНО — из json рядом с чекпойнтом (спека 3.4)."""
    j = find1(os.path.basename(ck).replace(".ckpt", ".json"))
    if not j:
        print("ВНИМАНИЕ: json с cfg не найден — надеюсь на дефолты v1")
        return []
    meta = json.load(open(j))
    cfg, v2, fl = meta.get("cfg", {}), meta.get("v2", {}), []
    for k in ("d", "layers", "heads", "ff", "lmax"):
        if k in cfg: fl += [f"--{k}", str(cfg[k])]
    if cfg.get("encoder") and cfg["encoder"] != "tfm":
        fl += ["--encoder", cfg["encoder"]]   # kevf_gru: без этого тензоры не совпадут
    if v2.get("time_bias"): fl += ["--time-bias", str(v2["time_bias"])]   # формы!
    if v2.get("time2vec"):  fl += ["--time2vec", str(v2["time2vec"])]
    if v2.get("aux_dt"):    fl += ["--aux-dt", str(v2["aux_dt"])]
    print(os.path.basename(ck), "cfg:", cfg, ("v2: " + str(v2)) if v2 else "")
    return fl

def launch(name, seed, dev):
    ck = find1(f"kevf_s{seed}.ckpt") if WARM else None
    cmd = [sys.executable, "kaggle_seq.py", "train", "--name", name,
           "--seed", str(seed), "--device", dev,
           "--tab", "", "--tab-mode", "concat", "--tab-dropout", "0.15",
           "--max-steps", str(MAX_STEPS)]
    if ck:
        cmd += ["--warm-from", ck, "--lr", "2e-4"] + arch_flags(ck)
        print(f"{name}: тёплый старт <- {ck}")
    else:
        cmd += ["--lr", "3e-4"]
        print(f"{name}: С НУЛЯ (чекпойнта нет или WARM=False)")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    return subprocess.Popen(cmd, stdout=open(f"{name}.log", "w"),
                            stderr=subprocess.STDOUT, env=env, start_new_session=True)

procs = {"kevf3_s42": launch("kevf3_s42", 42, "cuda:0"),
         "kevf3_s1337": launch("kevf3_s1337", 1337, "cuda:1")}
print({k: p.pid for k, p in procs.items()})
```

Через ~5 минут в логах обязаны появиться строки (порядок): «таблица: (26, 250000,
176) f16 …», «модель: … таблица 176->256 (concat, drop 0.15)», «--warm-from:
скопировано … тензоров; свежая инициализация: ['tab_branch…']», «warm-старт OK:
прогноз шага 0 с таблицей == без неё». Если вместо этого падение про якоря или
формы — см. ячейку 2 и cfg выше, НЕ продолжать вслепую.

## Ячейка 4 — мониторинг (перезапускать по вкусу)

```python
!tail -n 4 kevf3_s42.log kevf3_s1337.log; ls -la out/ 2>/dev/null | tail -8
```

Каждый eval печатает ПАРУ `калиброванный скор X | tabzero Y` (спека §4.1):
- tabzero держится около кривой v2 (~1.67) → событийный путь жив, всё штатно;
- tabzero разваливается (≫ v2), а скор ≈ уровню табличных моделей → коллапс в
  таблицу: убить процесс, перезапустить тем же `--warm-from` под НОВЫМ именем
  (например kevf3_s42d04) с `--tab-dropout 0.4`.

## Ячейка 5 — сторож + упаковка (ОБЯЗАТЕЛЬНА последней при Save & Run All)

```python
import subprocess, time
while any(p.poll() is None for p in procs.values()):
    time.sleep(600)
    subprocess.run(["tail", "-n", "2", "kevf3_s42.log", "kevf3_s1337.log"])
print("коды выхода:", {k: p.returncode for k, p in procs.items()})
subprocess.run(["zip", "-q", "-r", "results_jf.zip", "out",
                "kevf3_s42.ckpt", "kevf3_s1337.ckpt",
                "kevf3_s42.log", "kevf3_s1337.log"], check=True)
print("СКАЧАТЬ: /kaggle/working/results_jf.zip")
```

В интерактиве ячейку можно запустить и уйти; в батче (Save Version → Save & Run
All) без неё прогон завершится сразу после запуска процессов и убьёт их.

## Бюджет и что где лежит

- сборка ~2-5 мин + загрузка таблицы ~2-4 мин на процесс + 12000 шагов ≈ 5-5.5 ч
  (шаг ~1.5 с; eval подорожал в 2 раза из-за tabzero — учтено);
- выгрузки каждые 1500 шагов → годные файлы с первого часа (тёплый старт);
- `/kaggle/working/out/`: `kevf3_s42_{val,test}.parquet`, `kevf3_s42.json`
  (кривая + curve_tabzero) и то же для s1337; чекпойнты `kevf3_*.ckpt` в корне —
  забрать ДЛЯ СЕССИИ 4;
- OOM → `--batch 384`; сессия оборвалась → тот же launch + `"--resume"` в cmd
  (build пересобрать; --resume имеет приоритет над --warm-from и продолжит с
  полного состояния).

## После сессии (локально)

`results_jf.zip` → `out/kevf3_*` (6 файлов: parquet×4 + json×2) в
`work/colab/out/`, чекпойнты — в `work/kaggle/run/` (или куда скажет план
сессии 4). Затем по одному имени:

```
POLARS_MAX_THREADS=3 .venv/bin/python work/colab/ingest.py --name kevf3_s42
POLARS_MAX_THREADS=3 .venv/bin/python work/colab/ingest.py --name kevf3_s1337
.venv/bin/python work/reports/eve2_collapse_check.py kevf_s42 kevf3_s42
.venv/bin/python work/scripts/joint_gain.py --each kevf_s42 kevf3_s42
```

Вердикты collapse_check (зашиты в скрипт): КОЛЛАПС / запас<0 — не принимать;
РАЗМЫВАНИЕ — поднять tab-dropout или сузить tab_dim; ЗДОРОВ — нести в
joint_gain. Приговор набору выносит ТОЛЬКО joint_gain (порог пары 2×шум
0.000044): fusion обязан добавлять к паре с чистым kevf, иначе он пересказ.

## Чеклист владельца аккаунта — РЕЖИМ ЭКОНОМИИ КВОТЫ (руками, сверху вниз)

Квота считает время сессии с включённым ускорителем, поэтому проверку входов
делаем на Accelerator=None (бесплатно), а GPU включаем только на сам батч.

| # | шаг | время |
|---|-----|-------|
| 1 | `bash work/kaggle/make_bundle3.sh`; ozon-code → New Version → `/tmp/jf_bundle.zip` | ~2 мин |
| 2 | New Dataset **ozon-tabfeats-wed** ← `work/data/kaggle_session3/kaggle_tabfeats_wed_v1.zip` | ~10-30 мин (аплоад 1.25 ГБ) |
| 3 | Notebook: Add Input ×4 (ecup-t3, ozon-code, tabfeats-wed, Output сессии 2); **Accelerator = None** | ~5 мин |
| 4 | вставить ячейки 1-5; прогнать ТОЛЬКО ячейку 1: «26 npz» и оба kevf-чекпойнта найдены | ~3 мин, GPU 0 |
| 5 | если чекпойнтов НЕТ — стоп, чинить Input сессии 2 (с нуля на 4500 шагах не стартуем) | — |
| 6 | MAX_STEPS=4500 в ячейке 3; Accelerator = **GPU T4 x2**; сразу **Save Version → Save & Run All** (интерактивно ничего не гонять) | батч ~2-2.5 ч |
| 7 | после батча: Output → скачать results_jf.zip; чекпойнты kevf3_*.ckpt — тоже (для продолжения) | ~5 мин |
| 8 | продолжение до 12000: после сброса квоты (суббота) тот же ноутбук, `"--resume"` в cmd ячейки 3, MAX_STEPS=12000; или полный прогон на аккаунте Саши-2 (расшарить оба датасета + Output сессии 2) | позже |
| 8 | скачать `results_jf.zip` из Output, разложить (см. «После сессии»), запустить ingest/collapse/joint_gain | ~20 мин |
