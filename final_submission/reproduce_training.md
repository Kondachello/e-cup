# Воспроизведение: от `train.parquet` до отправленного файла

Документ описывает **фактический** пайплайн действующего решения — того самого файла,
что уходит в финальную сдачу;

Всё запускается из корня репозитория, окружение — `final_submission/requirements.txt`
(Python 3.10.17). Референсное железо: Apple M1 Pro, 10 ядер, 16 GB RAM (нейросети — на
MPS; на CPU те же команды, дольше). Времена ниже — **фактические**, из
`work/queue/done/*.json`, поле `seconds`.

> **Про параллельность.** На 16 GB тяжёлые обучения запускать строго
> ПОСЛЕДОВАТЕЛЬНО. Для этого есть очередь: `work/scripts/enqueue.py` кладёт задание,
> `work/scripts/queue_runner.py` берёт по одному. Параллельные запуски дважды роняли
> машину.

---

## 0. Шесть шагов решения

Порядок обязателен. Каждый шаг — отдельный раздел ниже и отдельная стадия
`final_submission/inference.py --stage ...`.

| # | шаг | чем делается | стадия инференса |
|---|---|---|---|
| 1 | обучение базовых моделей пула, у нейросетевых — с `--es-metric cal` | `work/scripts/train_*.py` | `predict` (грузит веса) |
| 2 | поквантильная калибровка каждого члена, 24 бина | `calibrate.py` | `ensemble` (применяет `NAME_cal.npz`) |
| 3 | линейное смешивание 14 членов в log1p | `blend_reopt.py` | `ensemble` |
| 4 | приведение к оценённым моментам целевого окна | `make_candidate.py` | `moments` |
| 5 | перенос накопленной цепочки поправок + сила шага | `make_candidate.py --carry-from` | `moments` |
| 6 | **среднесохраняющая поправка на молчащих** | `silence_model.py` | `silence` |

Шаг 6 даёт **0.00084 из 0.00098** всего дневного прироста 19 августа. Без него
воспроизводится не отправленный файл, а файл на 0.00084 хуже — вчетверо больше, чем
дали за те же сутки все улучшения самих моделей. Ему посвящён раздел 7.

## 1. Общий протокол (контракт `exp_lib`)

Обучающие примеры нарезаны по cutoff-датам («якорям»): признаки считаются по данным до
якоря включительно, таргет — суммарный GMV за следующие 30 дней.

| якорь | роль |
|---|---|
| `2026-02-13` | **тест**: его 30-дневное окно (14.02–15.03.2026) и есть предсказываемое |
| `2026-01-14` | **валидация**: таргет полностью наблюдаем |
| `<= 2025-12-10` | **обучение** при отборе (`--gap-days 30`) |
| `2025-12-17 … 2026-01-07` | «gap»-якоря: в отборе НЕ участвуют, добавляются в retrain |

`--gap-days 30` обязателен: без зазора таргет-окна обучающих срезов пересекаются с
валидационным, и val-скор завышается до +0.10 RMSLE.

Каждый трейнер делает две фазы:

1. **Отбор**: обучение на чистых якорях, ранняя остановка по якорю 2026-01-14,
   `work/preds/NAME_val.parquet` + строка в `work/reports/scores.tsv`.
2. **Retrain**: дообучение на train + gap + val (число итераций/эпох масштабируется по
   росту числа строк, `iter_mult = 1 + 0.7*(row_ratio-1)`), прогноз тестового якоря →
   `work/preds/NAME_test.parquet`.

**Веса сохраняются только на шаге retrain** (`work/scripts/model_io.py`): бустинги через
`Booster.save_model()`, torch-модели через `torch.save(state_dict)`, плюс
`work/models/NAME_meta.json` с порядком колонок признаков и конфигом архитектуры. Отсюда
важное следствие: **прогон с `--no-test` (или без `--final` у сеточных) весов не
оставляет вообще.** Именно эти файлы читает `final_submission/inference.py`.

### 1.1 Критерий ранней остановки `--es-metric cal`

Находка 19 августа. Ранняя остановка по СЫРОМУ валидационному RMSLE обрывает обучение
вчетверо раньше нужного, потому что сырой скор в основном отражает общий сдвиг уровня, а
не форму прогноза — а уровень всё равно снимается калибровкой (шаг 2). Остановка по
КАЛИБРОВАННОМУ скору даёт до 0.0028 на модель.

| семейство | флаг | замер |
|---|---|---|
| `train_mlpziln.py` | `--es-metric cal` | средняя дельта −0.000612 на сид, интервал [−0.000909, −0.000296]; сырой останавливал на эпохе 1–4 вместо 6–7 |
| `train_fusion3.py` | `--es-metric cal` | +0.001654 на сиде 555 (1.668676 против 1.670330) |
| `train_gbdt.py` | `--es-metric cal` есть, но **не использован**: парный замер на `c_ts2` и `twl` дал 1.692342 против 1.693077 и 1.694155 против 1.694288, то есть у бустинга выигрыша нет |
| `train_seq2.py` | флага нет вовсе (модель обучена до находки) |

В командах раздела 2 флаг стоит **ровно там, где он был при обучении**. Ставить его
дополнительно к бустингам не надо — это будет другая модель, а не воспроизведение.

## 2. Признаки и тензоры (один раз)

```bash
python3.10 -m venv .venv && .venv/bin/pip install -r final_submission/requirements.txt
# train.parquet и sample_submit.csv — в корень репозитория (или задать OZON_ROOT)

A="2026-02-13,2026-01-14,2025-07-02,2025-07-16,2025-07-30,2025-08-13,2025-08-27,2025-09-10"

.venv/bin/python work/scripts/build_features.py    --preset all      # base -> anchor=DATE.parquet
.venv/bin/python work/scripts/build_features_v2.py --anchors "$A"    # -> .extra.parquet (USE_V2)
.venv/bin/python work/scripts/build_features_v3.py                   # -> .v3.parquet    (USE_V3)
.venv/bin/python work/scripts/build_features_v4.py --anchors "$A"    # -> .v4.parquet    (USE_V4), BTYD
.venv/bin/python work/scripts/build_features_v7.py --anchors 2026-02-13 \
    --states 4 --sims 300 --win 120 --em-cap 15000 --seed 42         # -> .v7.parquet    (USE_V7)
.venv/bin/python work/scripts/build_count_targets.py                 # -> .cnttgt.parquet (countaov)

POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/build_seq3.py --max-train 8   # 3.4 ГБ
POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/build_seq2.py                 # ~11 ГБ
```

Шесть якорей 2025-07-02 … 2025-09-10 нужны **только для модели молчания** (раздел 7):
это те якоря, чьё 30-дневное целевое окно кончается раньше 2025-11-16.

Наборы подключаются переменными окружения — их читает `load_anchor()` в
`work/scripts/common.py`, и **от них зависит порядок колонок**, поэтому каждая модель
обучается со своим набором флагов и тот же набор записан в её
`work/models/NAME_meta.json`.

| набор | флаг | что в нём | нужен для |
|---|---|---|---|
| base | — | суммы/счётчики за окна 1–365 дней, recency, интервалы, тренды, «то же окно год назад» | всех |
| v2 | `USE_V2` | экспоненциальные затухания, концентрация трат, дни с крупными покупками | всех табличных |
| v3 | `USE_V3` | percentile-ранги внутри среза, детализация прошлогоднего окна, burstiness | всех табличных |
| v4 | `USE_V4` | BTYD: BG/NBD (P(alive), ожидаемое число покупок) + Gamma-Gamma (чек) | всех, кроме `c_ts2_*` и `c_xtw_s42` |
| v7 | `USE_V7` | 3 колонки HMM-симулятора (`hmm_elog`, `hmm_p_zero`, `hmm_sim_std`) | только `twl_v7` |
| seq3 | — | uint8 [250k × 112 дней × 12 каналов] на якорь | `fusion_v3*` |
| seq2 | — | float16 [250k × 196 дней × 8 каналов] на якорь | `seq2tr_f` |

Контрольные числа колонок: **203** при V2+V3+V4, **206** при +V7, **194** при V2+V3,
**85** у `behavonly` (правило отбрасывания «денежных» колонок).

> Наборы v5/v6/v8/v10 и `build_seq.py` собраны, измерены и **отвергнуты** — в решение не
> входят, строить их не нужно. `train_wklin.py --emit-tier` попутно пишет `.v5s.parquet`;
> сам бленд его не использует, но флаг в команде оставлен, потому что модель обучалась
> именно так.

## 3. Шаг 1: базовые модели

Команды ниже воспроизводят обучение каждой модели и исторически точны. Но **не все
описанные модели входят в действующий бленд** — часть получила при переоптимизации вес
ровно ноль. Ориентир, какие подразделы нужны для сборки решения:

| подраздел | модель | вес в бленде |
|---|---|---|
| 3.1 | `fusion_v3c*`, `fusion_v3ctl`, `fusion_f` | **0.346** |
| 3.4 | `c_ts2_s42` | **0.033** (`c_xtw_s42`, `twl_v7` — вес 0) |
| 3.5 | `behavonly` (3 сида) | **0.041** |
| 3.7 | `weak_an_d`, `weak_ft_recency` | **0.089** (`weak_ft_counts`, `weak_ft_long90` — вес 0) |
| 3.8 | `wklin`, `wklin_wk` | **0.074** |
| 3.9 | `febspec2` | **0.009** |
| 3.2 | `seq2tr_f` | 0 — не входит |
| 3.3 | `mlpziln_c*` | 0 — не входит |
| 3.6 | `countaov_s7` | 0 — не входит |
| 3.10 | `hmmsim` | 0 — не входит |

Не описаны здесь и обучаются на машинах своих треков: `kostya46` (вес **0.246**, трек №3,
`work_kostya/`), `gseq_small_s42` и `gseq_big_s42` (вес **0.133**, трек №5,
`work/colab/gpu_seq.py`), `lagd28` (вес **0.035**, `work/scripts/lag_tta.py`).

Полная таблица весов — §5.

**Про набор обучающих якорей.** Трейнеры, бравшие его из содержимого каталога признаков,
переведены на протокол: `--anchor-source protocol` (умолчание) задаёт набор через
`train_anchors(N)` и не зависит от того, какие ещё файлы лежат в `work/features`.
Историческое поведение доступно как `--anchor-source disk` и нужно для воспроизведения
артефактов, собранных до 25.08. Почему это важно, с измерениями — `../README.md` §7.

### 3.1 Секвенсные (тензоры seq3), `train_fusion3.py`

```bash
E="USE_V2=1 USE_V3=1 USE_V4=1 OMP_NUM_THREADS=4 POLARS_MAX_THREADS=3"

# три сида «калиброванного критерия» -> усредняются в fusion_v3c_avg
env $E .venv/bin/python work/scripts/train_fusion3.py --name fusion_v3c42  --final \
  --epochs 3 --batch 2048 --eval-batch 1024 --lr 1e-3 --seeds 42  --threads 4 \
  --eval-every 492 --n-ch 12 --es-metric cal                       # 1050 с
env $E .venv/bin/python work/scripts/train_fusion3.py --name fusion_v3c555 --final \
  --epochs 3 --batch 2048 --eval-batch 1024 --lr 1e-3 --seeds 555 --threads 4 \
  --eval-every 492 --n-ch 12 --es-metric cal                       # 952 с
env $E .venv/bin/python work/scripts/train_fusion3.py --name fusion_v3c7   --final \
  --epochs 3 --batch 2048 --eval-batch 1024 --lr 1e-3 --seeds 7   --threads 4 \
  --eval-every 492 --n-ch 12 --es-metric cal                       # 918 с

# 12 каналов и строгий контроль на 8 каналах (сырой критерий остановки)
env $E .venv/bin/python work/scripts/train_fusion3.py --name fusion_v3    --final \
  --epochs 3 --batch 2048 --eval-batch 1024 --lr 1e-3 --seeds 42 --threads 4 \
  --eval-every 984 --n-ch 12                                       # 921 с, val 1.689871
env $E .venv/bin/python work/scripts/train_fusion3.py --name fusion_v3ctl --final \
  --epochs 3 --batch 2048 --eval-batch 1024 --lr 1e-3 --seeds 42 --threads 4 \
  --eval-every 984 --n-ch 8                                        # 875 с, val 1.691090
```

`--n-ch` просто обрезает тензор до первых N каналов: 0–7 совпадают с набором seq2,
8–11 — дневные счётчики воронки (`search_to_cart`, `search_to_ord`, `cat_to_cart`,
`cat_to_ord`). `fusion_v3ctl` — строгий контроль к `fusion_v3`: те же тензоры, якоря, сид,
L=112 и квантование, отличие ровно в этих четырёх каналах.

> **Веса: НЕ СОХРАНЯЮТСЯ.** В `train_fusion3.py` нет ни одного вызова `model_io`; он
> пишет только `work/models/NAME_stats.npz`. Артефактом этих пяти моделей является сам
> прогноз `work/preds/NAME_test.parquet`, и другого способа получить его, кроме
> повторного обучения, нет. Это самая крупная дыра воспроизводимости пакета: суммарный
> вес пяти моделей в бленде — 0.397 из 1.005.

### 3.2 Секвенсная на тензорах seq2, `train_seq2.py`

```bash
OMP_NUM_THREADS=4 .venv/bin/python work/scripts/train_seq2.py --name seq2tr_f \
  --arch tr --final --epochs 3 --batch 2048 --lr 1e-3 --seeds 42,1337 --threads 4
# 19485 с = 5.4 ЧАСА — самая долгая модель проекта. val 1.710203.
# Табличных признаков не видит вовсе (USE_* не задаются), поэтому её ошибки меньше
# всего скоррелированы с остальными. Веса: work/models/seq2tr_f_seed{42,1337}.pt
# ТЕНЗОРЫ work/seq2 УДАЛЕНЫ (освобождали диск); сначала build_seq2.py, ~11 ГБ.
```

### 3.3 Сеточная табличная, `train_mlpziln.py` (три сида, `--es-metric cal`)

```bash
E="USE_V2=1 USE_V3=1 USE_V4=1 OMP_NUM_THREADS=4 POLARS_MAX_THREADS=3"
env $E .venv/bin/python work/scripts/train_mlpziln.py --name mlpziln_c42 \
  --n-anchors 14 --gap-days 30 --seeds 42   --es-metric cal    # 248 с, val 1.684804
env $E .venv/bin/python work/scripts/train_mlpziln.py --name mlpziln_c1337 \
  --n-anchors 14 --gap-days 30 --seeds 1337 --es-metric cal    # 188 с, val 1.682801
env $E .venv/bin/python work/scripts/train_mlpziln.py --name mlpziln_c7 \
  --n-anchors 14 --gap-days 30 --seeds 7    --es-metric cal    # 112 с, val 1.680350
# Zero-inflated lognormal (p, mu, sigma); E[log1p] берётся 20-точечной квадратурой
# Гаусса-Эрмита. Остальное — умолчания: --epochs 30 --patience 4 --batch 8192
# --lr 1e-3 --wd 1e-4 --dropout 0.15 --hidden 512,256 --feat-prep clip99.
# Веса: mlpziln_c{42,1337,7}_seed{42,1337,7}.pt + *_stats.npz
```

### 3.4 Бустинги, `train_gbdt.py` и обёртки над ним

```bash
# двухстадийный LightGBM: P(y>0) x E[log1p|y>0]. БЕЗ USE_V4 — 194 признака.
E="USE_V2=1 USE_V3=1 OMP_NUM_THREADS=6"
for S in 42 7; do
env $E .venv/bin/python work/scripts/train_gbdt.py --name c_ts2_s$S \
  --threads 6 --gap-days 30 --model lgb --objective two_stage --n-anchors 14 --seed $S \
  --params  '{"num_leaves":127,"min_data_in_leaf":500,"n_estimators":5000}' \
  --params2 '{"num_leaves":255,"min_data_in_leaf":100,"n_estimators":5000}'
done            # 303 с и 471 с; val 1.693138 и 1.692363
                # веса: c_ts2_sS__stage1.txt, c_ts2_sS__stage2.txt

# XGBoost tweedie, тоже без USE_V4
env $E .venv/bin/python work/scripts/train_gbdt.py --name c_xtw_s42 \
  --threads 6 --gap-days 30 --model xgb --objective log_mse --n-anchors 10 --seed 42 \
  --params '{"objective":"reg:tweedie","tweedie_variance_power":1.2,"max_leaves":511,
             "min_child_weight":100,"learning_rate":0.05,"colsample_bytree":0.8,
             "n_estimators":6000}'
                # 246 с, val 1.697925; веса: c_xtw_s42.xgb.json

# LightGBM tweedie + HMM-признаки. ЕДИНСТВЕННАЯ причина строить build_features_v7.py.
USE_V2=1 USE_V3=1 USE_V4=1 USE_V7=1 OMP_NUM_THREADS=6 POLARS_MAX_THREADS=3 \
.venv/bin/python work/scripts/train_gbdt.py --name twl_v7 \
  --threads 6 --gap-days 30 --model lgb --objective log_mse --n-anchors 8 --seed 42 \
  --params '{"objective":"tweedie","tweedie_variance_power":1.45,"n_estimators":6000}'
                # 187 с, val 1.694155, 206 признаков, 8 якорей (по покрытию v7)
                # веса: twl_v7.txt
```

### 3.5 Модель без единого «денежного» признака, `train_behavonly.py`

```bash
E="USE_V2=1 USE_V3=1 USE_V4=1"
env $E OMP_NUM_THREADS=6 .venv/bin/python work/scripts/train_behavonly.py \
  --name behavonly --n-anchors 14 --threads 6 --seed 42               # 214 с
env $E OMP_NUM_THREADS=4 POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/train_behavonly.py \
  --name behavonly_s1337 --seed 1337 --threads 4                      # 427 с
env $E OMP_NUM_THREADS=4 POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/train_behavonly.py \
  --name behavonly_s7    --seed 7    --threads 4                      # 442 с
# Скрипт по ПРАВИЛАМ выбрасывает всё, что несёт деньги, остаётся 85 поведенческих
# признаков, дальше делегирует в train_gbdt.main() — поэтому веса сохраняются им, а
# meta["script"] у этих моделей = train_gbdt.py.
# ВНИМАНИЕ: у двух дополнительных сидов НЕТ --n-anchors, поэтому они обучены на всех
# 27 доступных якорях, а базовый сид 42 — на 14. Так и было; это не описка.
# Веса: behavonly.txt, behavonly_s1337.txt, behavonly_s7.txt.  val_avg 1.710680
```

### 3.6 Разложение количество × средний чек, `train_countaov.py`

```bash
USE_V2=1 USE_V3=1 USE_V4=1 OMP_NUM_THREADS=4 POLARS_MAX_THREADS=3 \
.venv/bin/python work/scripts/train_countaov.py --name countaov_s7 \
  --threads 4 --n-anchors 14 --gap-days 30 --seed 7
# 489 с, val 1.693744. Две LightGBM-головы: count (log1p) и AOV (режим uplift).
# Нужен build_count_targets.py. Веса: countaov_s7__count.txt, countaov_s7__aov.txt
```

### 3.7 Слабые специализированные модели, `train_weak.py`

```bash
E="USE_V2=1 USE_V3=1 USE_V4=1 OMP_NUM_THREADS=4 POLARS_MAX_THREADS=3"
P='{"objective":"tweedie","tweedie_variance_power":1.45,"n_estimators":6000}'

env $E .venv/bin/python work/scripts/train_weak.py --name weak_an_d --threads 4 \
  --mech anchors --k-anchors 4 --sel-seed 77 --anchor-pool 0 \
  --model lgb --objective log_mse --params "$P"        # 123 с, val 1.683269

for F in recency counts long90; do
env $E .venv/bin/python work/scripts/train_weak.py --name weak_ft_$F --threads 4 \
  --mech ftype --ftype $F --n-anchors 14 \
  --model lgb --objective log_mse --params "$P"        # ~190 с каждая
done   # val 1.729126 / 1.702208 / 1.716552
# --gap-days 30 скрипт добавляет сам, если флага нет. Делегирует в train_gbdt.main(),
# веса: weak_an_d.txt, weak_ft_{recency,counts,long90}.txt
```

### 3.8 Линейная модель на недельных колонках, `train_wklin.py`

```bash
USE_V2=1 USE_V3=1 USE_V4=1 THREADS=4 POLARS_MAX_THREADS=3 \
.venv/bin/python work/scripts/train_wklin.py --name wklin --emit-tier
# 91 с. Гребневая регрессия на 180 недельных колонках, привязанных к якорю
# (акт/корзины/заказы/поиски/GMV x 36 недель, в signed log) + 203 базовых признака.
# alpha подбирается на ОТЛОЖЕННОМ обучающем якоре, никогда не на валидационном.
# ОДИН прогон пишет СРАЗУ три набора: wklin_base (только база), wklin (недели+база,
# val 1.684188) и wklin_wk (только недели, val 1.731511). Отдельно wklin_wk не получить.
# Сида нет: решение закрытой формы детерминировано.
# Веса: НЕ СОХРАНЯЮТСЯ (в скрипте нет model_io) -> артефакт это прогноз.
```

### 3.9 Февральский специалист, `train_febspec2.py`

```bash
OMP_NUM_THREADS=3 POLARS_MAX_THREADS=3 THREADS=3 \
.venv/bin/python work/scripts/train_febspec2.py --name febspec2 \
  --config auto --cohort 0.20 --threads 3
# 429 с, val 1.785259. Короткоисторический тир (93 признака, funnel=False),
# 39 недельных якорей от когорты 2025-02-17, 20% пользователей.
# Свой набор признаков (build_features_short.py), USE_* не использует; при
# отсутствии work/features_short пересобирает его сам (уже входит в 429 с).
# Веса: НЕ СОХРАНЯЮТСЯ -> артефакт это прогноз.
```

### 3.10 Порождающий симулятор, `train_hmm_sim.py`

```bash
THREADS=6 .venv/bin/python work/scripts/train_hmm_sim.py --name hmmsim \
  --states 4 --sims 500 --win 120 --em-cap 25000 --splits val,test
# 366 с, val 1.823787. Обучаемых весов НЕТ по построению: скрытая марковская модель
# покупательской активности оценивается EM по собственной истории каждого юзера,
# таргет не видит вовсе, дальше 500 симуляций вперёд на 30 дней. Воспроизводится
# повторным запуском с тем же сидом; hmmsim_meta.json это фиксирует ("stateless": true).
# --splits ЗДЕСЬ ОБЯЗАТЕЛЕН: умолчание "val", без "test" прогноза теста не будет.
```

## 4. Шаг 2: калибровка и усреднение сидов

Порядок жёсткий: **сначала усреднение сидов, потом калибровка усреднённого**. Так эти
члены и собирались; калибровать по отдельности, а потом усреднять — другая операция.

```bash
# усреднение сидов в log1p (равные веса)
.venv/bin/python work/scripts/avg_log1p.py --out fusion_v3c_avg \
  --preds fusion_v3c555,fusion_v3c42,fusion_v3c7
.venv/bin/python work/scripts/avg_log1p.py --out mlpziln_cal_avg \
  --preds mlpziln_c42,mlpziln_c1337,mlpziln_c7
.venv/bin/python work/scripts/avg_log1p.py --out behavonly_avg \
  --preds behavonly,behavonly_s1337,behavonly_s7

# калибровка: 24 квантильных бина в log1p
for m in fusion_v3c_avg fusion_v3ctl c_ts2_s7 mlpziln_cal_avg c_ts2_s42 behavonly_avg \
         seq2tr_f weak_an_d weak_ft_recency countaov_s7 weak_ft_counts hmmsim \
         fusion_v3 twl_v7 febspec2 weak_ft_long90 ; do
  .venv/bin/python work/scripts/calibrate.py --pred $m --bins 24
done
```

В каждом бине считается сдвиг `mean(log1p(факт)) − mean(log1p(прогноз))`, между центрами
бинов он интерполируется. Честность контролируется внутри скрипта: таблица подгоняется на
половине пользователей, проверяется на другой. Эффект +0.010…0.012 RMSLE на модель.

Таблица сдвигов замораживается в `work/models/NAME_cal.npz` (ключи `centers`, `shifts`) —
инференс её ПРИМЕНЯЕТ, а не переподбирает. **24 бина и `clip99` в подготовке признаков
проверены парными замерами и стоят в оптимуме**: 15/63/127 бинов и варианты
`signlog`/`rank`/`clip999`/`noclip` измерены и хуже.

Четыре члена входят в бленд **сырыми**, без калибровки — так их выбрал оптимизатор:
`wklin`, `wklin_wk`, `hmmsim` (он же входит и калиброванным, с другим весом) и
`c_xtw_s42`.

## 5. Шаг 3: бленд

```bash
.venv/bin/python work/scripts/blend_reopt.py --save --boot 50
# метод ridge_free; честный OOF по 5 фолдам по ПОЛЬЗОВАТЕЛЯМ 1.667450 (in-sample 1.667428)
# -> work/preds/blend_opt_{val,test}.parquet + work/reports/blend_reopt.json
```

Смешивание — взвешенная сумма в log1p: `lp_blend = sum_i w_i * lp_i`. Скор итоговой смеси
на валидационном якоре **1.665647**; это же число служит эталоном во всех измерениях
приёмки (колонка `blend` пакета `work/preds_pack/val_preds.parquet`).

| член | вес | | член | вес |
|---|---|---|---|---|
| `kostya46_cal` | 0.246021 | | `behavonly_avg_cal` | 0.041373 |
| `fusion_v3c_avg_cal` | 0.228925 | | `lagd28` | 0.035049 |
| `gseq_small_s42_cal` | 0.108701 | | `c_ts2_s42_cal` | 0.033278 |
| `fusion_v3ctl_cal` | 0.106124 | | `gseq_big_s42_cal` | 0.024431 |
| `wklin` | 0.070987 | | `fusion_f_cal` | 0.010602 |
| `weak_an_d_cal` | 0.045493 | | `febspec2_cal` | 0.008916 |
| `weak_ft_recency_cal` | 0.043429 | | `wklin_wk_cal` | 0.002788 |
| | | | **сумма** | **1.0061** |

Веса неотрицательные, нормировка не навязывается. Все члены калиброваны, кроме `wklin` и
`lagd28`, которые входят сырыми.

Эти же 14 весов **зафиксированы константой `BLEND_WEIGHTS` в `inference.py`**: отчёт
`blend_reopt.json` переписывается при каждом перезапуске оптимизатора, а пакет обязан
собирать один и тот же файл. Сверено: константа и отчёт совпадают до 1e-6. Проверить:

```bash
.venv/bin/python final_submission/inference.py --stage check --verify-blend
```

## 6. Шаги 4–5: моменты и накопленная цепочка

```bash
.venv/bin/python work/scripts/make_candidate.py --pred blend_opt --name FILE.csv \
  --carry-from blend_cal --strength 0.469
```

**Шаг 4 — приведение к моментам.** Оценено (KNOWLEDGE.md, факт Ф18): пара
сабмитов, отличающихся на известную константу в log-пространстве, даёт точное среднее из
разности квадратов скоров. среднее log1p **2.3247** и разброс **1.6320**.

Это свойство ТЕСТОВОГО ОКНА, а не конкретного бленда, поэтому при улучшении бленда
числа не переподбираются — новый бленд приводится к тем же двум моментам, и улучшение
сохраняется, а проверенная сезонная настройка не теряется. Физическая причина сдвига:
тестовое окно 14.02–15.03 содержит неделю перед 8 марта, а модели обучены на
осенне-зимних окнах. Независимая проверка на исторических данных: на прошлогодних аналогах окон
среднее log1p растёт 1.5396 → 1.7154 (+0.1759), наш подъём +0.1694.

**Шаг 5а — перенос цепочки** (`--carry-from blend_cal`). над приведённым к тем же
моментам старым блендом от неё остаётся вектор с разбросом 0.121 (7.4% от 1.632). Без
переноса эта накопленная работа теряется.

**Шаг 5б — сила шага** (`--strength 0.469`). улучшение бленда,
посчитанное на валидации, переносится на тест лишь на 39%, и применение полной силы
перелетает через оптимум. 1.6488027376 при прогнозе 1.6488044, попадание
1.7e-6).

Оба вектора (опора и цепочка) — **разности отправленных файлов, а не выход модели**,
пересчитать их из весов нельзя. В пакете они лежат замороженными в
`final_submission/models/chain_test.npz`; пересобрать:

```bash
.venv/bin/python final_submission/inference.py --stage freeze
```

## 7. Шаг 6: поправка на молчащих

**Это самый ценный шаг решения: 0.00084 из 0.00098 дневного прироста.**

### 7.1 Почему она нужна

Организаторы отобрали 250000 пользователей как активных в КАЖДОМ из трёх 30-дневных
блоков перед тестовым окном (16.11–15.12, 16.12–14.01, 15.01–13.02). Наше валидационное
окно 15.01–13.02 — **один из этих блоков**. Значит:

* на валидации молчащих (ноль событий за 30 дней) нет **по построению**, это не свойство
  данных, а следствие правила отбора;
* в тестовом окне 14.02–15.03 никакого отбора уже нет, и молчащие там будут;
* для молчащего верный ответ — ноль, а любая наша модель, обученная и проверенная там,
  где молчащих нет, даёт ему обычный положительный прогноз.

Ни одна локальная валидация этого увидеть не может: нужный сигнал вырезан из
валидационного окна условием отбора. Поэтому поправка строится на «чистых» якорях —
тех, чьё 30-дневное целевое окно кончается **раньше 2025-11-16**, начала первого блока
отбора. Последний такой якорь — **2025-10-15**.

### 7.2 Форма поправки

Среднесохраняющая: `delta_i = -(p_i * m_i - mean(p*m))`, где `m = log1p(прогноза)`,
`p` — вероятность молчания. Работает в ней только РАЗБРОС `p` между людьми: общий уровень
`p` ничем локальным не определён (доля молчащих на чистых якорях сама падает 0.037 → 0.020
по мере приближения к блокам отбора — это артефакт отбора, а не тренд) и лишь
перепараметризует силу. Отсюда протокол: направление приводится к фиксированному размеру
`q = mean(d²) = 0.0027149`, и тогда коэффициент силы означает одно и то же физическое
количество поправки независимо от выбранного уровня `p`.

### 7.3 Как обучается модель молчания

```bash
.venv/bin/python work/scripts/silence_model.py --stage eval    # честный замер
.venv/bin/python work/scripts/silence_model.py --stage final   # рабочая модель
```

* **Обучающие якоря:** 2025-07-02, 07-16, 07-30, 08-13, 08-27 (пять). Якорь 2025-09-10
  отложен под калибровку наклона Платта, 2025-10-15 — под честную проверку.
* **Цель:** ноль событий в (якорь, якорь+30].
* **Модель:** смесь 0.5/0.5 логистической регрессии и LightGBM. AUC 0.901 на честном
  якоре 2025-10-15.

Три защиты от артефакта отбора, без которых модель выучила бы правило отбора вместо
молчания:

1. **Население подогнано под отбор.** На каждом обучающем якоре берутся только те, кто
   активен в каждом из трёх предшествующих 30-дневных блоков — ровно то условие, которому
   все 250000 удовлетворяют на тестовом якоре. Инференс это проверяет утверждением
   `assert sel_mask(C, TEST_ANCHOR).all()`, а не полагается на слово.
2. **Свой свободный член на якорь** (у логрегрессии фиктивные переменные, у бустинга
   `init_score`): уровень якоря поглощается и не участвует в обучении формы.
3. **Признаки, живущие на уровне якоря, выбрасываются.** `history_days`,
   `seasonal_index`, `ya_cov_*` постоянны внутри якоря (доля межъякорной дисперсии 1.0), а
   `ya_cov_*` к тому же равны 0 на всех обучающих якорях и 1 на тестовом — модель на них
   экстраполировала бы вслепую. Отсев по доле межъякорной дисперсии > 0.30 оставляет 180
   признаков из 203; все оставшиеся заменяются **внутриякорными процентильными рангами**
   (связки получают средний ранг — обязательно, потому что у большинства признаков крупная
   масса точных нулей).

### 7.4 Как применяется

Отправленный файл раскладывается точно:

```
M1 = FILE.csv + 0.894 * mdl_tektit + 0.65 * (d_new - proj_{mdl_tektit} d_new)
```

* `mdl_tektit` — **старое** направление, построенное по грубой двумерной таблице «активных
  дней за 90 x давность последней активности».
* `d_new` — направление **модели** из 7.3, приведённое к тому же `q`. Её преимущество
  измерено на честном якоре 2025-10-15 отношением выигрышей `c²/q`, где `c = cov(p*m, y*m)`:
  **1.155**, бутстрап по пользователям [1.096, 1.223]. Эта величина не зависит от масштаба
  `p`, поэтому она чистая мера выравнивания ФОРМЫ.
* Ортогональная новизна `e = d_new − proj_{mdl_tektit} d_new` — та часть, которой в опорном
  направлении нет. Её новизна относительно всего измеренного базиса 0.902, наивысшая за проект,
  но собственного замера у неё пока нет, поэтому шаг усажен до **0.65**, а не применён
  целиком.
* Корреляция нового направления со старым 0.839; `q` обоих 0.0027149.

Проверка алгебры (выполняется за секунду и должна давать машинный ноль):

```bash
# FILE.csv + 0.894*mdl_tektit должно совпасть с FILE.csv до 1e-16
.venv/bin/python - <<'PY'
import sys, numpy as np; sys.path.insert(0,'work/scripts')
from subs import lp
z = np.load('final_submission/models/chain_test.npz')
print(np.abs((lp('FILE.csv')[1] + 0.894*z['dir_old']) - lp('FILE.csv')[1]).max())
PY
```

В инференсе вероятность `p` для 250000 пользователей кэшируется в
`final_submission/models/silence_p_test.npz`; если файла нет, стадия `silence` обучает
модель на месте (~10 минут), импортируя примитивы прямо из `work/scripts/silence_model.py`.

## 8. Что придётся переобучить, и сколько это стоит

Состояние проверяется одной командой — она же печатает команду восстановления для
каждого недостающего артефакта:

```bash
.venv/bin/python final_submission/inference.py --stage check
```

На 19 августа 20:00 картина такая.

**Веса есть, инференс их грузит (13 моделей):** `mlpziln_c42`, `mlpziln_c1337`,
`mlpziln_c7`, `behavonly_s1337`, `behavonly_s7`, `weak_an_d`, `weak_ft_recency`,
`weak_ft_counts`, `weak_ft_long90`, `countaov_s7` (и таблицы калибровки к ним).

**Переобучить обязательно — весов нет и получить их неоткуда:**

| модель | почему | время |
|---|---|---|
| `seq2tr_f` | трейнер сохраняет веса, но прогон был до `model_io`; **плюс тензоры `work/seq2` удалены**, их пересборка ~11 ГБ | **5.4 ч** + сборка seq2 |
| `c_ts2_s7` | прогон был до `model_io` | 8 мин |
| `c_ts2_s42` | то же | 5 мин |
| `twl_v7` | то же (есть только сиды s7/s1337) | 3 мин |
| `c_xtw_s42` | то же | 4 мин |
| `behavonly` (сид 42) | то же (есть только s1337 и s7) | 4 мин |
| **итого** | | **5.8 ч** |

**Трейнер не сохраняет веса вообще — артефактом является прогноз** (сейчас прогнозы лежат
в `work/preds`, но в отгружаемом пакете их не будет):

| модель | скрипт | время |
|---|---|---|
| `fusion_v3c42`, `fusion_v3c555`, `fusion_v3c7`, `fusion_v3`, `fusion_v3ctl` | `train_fusion3.py` — нет ни одного вызова `model_io` | 15–18 мин каждая, **1.3 ч** |
| `wklin` (+`wklin_wk` тем же прогоном) | `train_wklin.py` | 1.5 мин |
| `febspec2` | `train_febspec2.py` | 7 мин |
| `hmmsim` | весов нет по построению, пересчёт симулятора | 6 мин |
| **итого** | | **1.6 ч** |

**Плюс к этому:** сборка тензоров `seq2` (~11 ГБ, `work/seq3` 3.4 ГБ на месте), четыре
недостающие таблицы калибровки (`c_ts2_s42`, `seq2tr_f`, `hmmsim`, `twl_v7`, по ~1 мин),
обучение модели молчания (~10 мин).

**Итог: 7.4 часа последовательного обучения на чистой машине** (5.8 ч того, у чего веса
должны были быть, но нет + 1.6 ч того, у чего весов не бывает), плюс сборка признаков и
тензоров. Из них 5.4 часа — одна модель `seq2tr_f` с весом 0.049.

> **Самое дешёвое улучшение воспроизводимости**, если появится время: добавить вызовы
> `model_io.save_torch` / `save_meta` в `train_fusion3.py`. Это снимет 1.3 ч из 1.6 ч
> второй таблицы и, главное, закроет дыру на 0.397 веса бленда — почти сорок процентов
> решения сейчас нельзя проверить, не переобучив.

## 9. Что НЕ входит в решение

Чтобы не тратить время на воспроизведение отвергнутого: наборы признаков v5/v6/v8/v10,
`build_seq.py`; модели `train_mlp.py`, `train_gru.py`, `train_bagged.py`, `train_gls.py`,
`train_pseudo.py`, `train_quantint.py`, `train_rank.py`, `train_whale.py`,
`train_horizon.py`, `train_hjit.py`, `train_channel.py`, `train_mlpbin.py`,
`train_feb_specialist.py`, `train_febspec3.py`, `train_fusion.py` (предшественник
`train_fusion3.py`, работал на тензорах seq2), `train_xtw.py` (более поздний клон
`train_gbdt.py`, отправленный `c_xtw_s42` обучен НЕ им).

Модели эпохи до `--gap-days 30` (`lgblog_final`, `xgblog_final`, `cblog_final`,
`mlp_final`, `gru_final`, `hjit37`, `hjit44`) исключены из библиотеки бленда как
`OLD_ERA`: их val-скоры завышены пересечением таргет-окон.
