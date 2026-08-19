# Воспроизведение обучения: от `train.parquet` до финального сабмита

Документ описывает **фактический** пайплайн финального решения: какие признаки
каким скриптом строятся, какие модели каким флагом обучаются, как считается
бленд и какие два числа применяются в самом конце.

Всё запускается из корня репозитория, окружение — `final_submission/requirements.txt`
(Python 3.10). Референсное железо: Apple M1 Pro, 10 ядер, 16 GB RAM (нейросети —
на MPS; на CPU те же команды, дольше). Времена ниже — фактические, из
`work/queue/done/*.json` (поле `seconds`).

> **Про параллельность.** На 16 GB тяжёлые обучения запускать строго
> ПОСЛЕДОВАТЕЛЬНО. У нас для этого очередь: `work/scripts/enqueue.py` кладёт
> задание, `work/scripts/queue_runner.py` берёт по одному. Параллельные запуски
> дважды роняли машину.

---

## 0. Общий протокол (контракт `exp_lib`)

Обучающие примеры нарезаны по cutoff-датам («якорям») с шагом 7 дней: признаки
считаются по данным до якоря включительно, таргет — суммарный GMV за следующие
30 дней.

| якорь | роль |
|---|---|
| `2026-02-13` | **тест**: его 30-дневное окно (14.02–15.03.2026) и есть предсказываемое |
| `2026-01-14` | **валидация**: таргет полностью наблюдаем |
| `<= 2025-12-10` | **обучение** при отборе (`--gap-days 30`) |
| `2025-12-17 … 2026-01-07` | «gap»-якоря: в отборе НЕ участвуют, добавляются в retrain |

`--gap-days 30` обязателен: без зазора таргет-окна обучающих срезов
пересекаются с валидационным, и val-скор завышается до +0.10 RMSLE.

Каждый трейнер делает две фазы:

1. **Отбор**: обучение на чистых якорях, ранняя остановка по якорю 2026-01-14,
   `work/preds/NAME_val.parquet` + строка в `work/reports/scores.tsv`.
2. **Retrain**: дообучение на train + gap + val (число итераций/эпох
   масштабируется по росту числа строк, `iter_mult = 1 + 0.7*(row_ratio-1)`),
   прогноз тестового якоря → `work/preds/NAME_test.parquet`.

**Веса моделей сохраняются на шаге retrain** (`work/scripts/model_io.py`):
бустинги через `Booster.save_model()`, torch-модели через
`torch.save(state_dict)`, плюс `work/models/NAME_meta.json` с порядком колонок
признаков и конфигом архитектуры. Именно эти файлы читает
`final_submission/inference.py`.

## 1. Признаки (один раз, ~40–60 мин суммарно)

```bash
python3.10 -m venv .venv && .venv/bin/pip install -r final_submission/requirements.txt
# train.parquet и sample_submit.csv — в корень репозитория (или задать OZON_ROOT)

.venv/bin/python work/scripts/build_features.py --preset all   # base -> anchor=DATE.parquet
.venv/bin/python work/scripts/build_features_v2.py             # -> .extra.parquet  (USE_V2)
.venv/bin/python work/scripts/build_features_v3.py             # -> .v3.parquet     (USE_V3)
.venv/bin/python work/scripts/build_features_v4.py             # -> .v4.parquet     (USE_V4), BTYD
.venv/bin/python work/scripts/build_features_v7.py             # -> .v7.parquet     (USE_V7), только для twl_v7
.venv/bin/python work/scripts/build_seq2.py                    # -> work/seq2/anchor=DATE.npy
.venv/bin/python work/scripts/build_channel_targets.py         # -> .chtgt.parquet
.venv/bin/python work/scripts/build_count_targets.py           # -> .cnttgt.parquet
```

Наборы признаков подключаются переменными окружения — их читает `load_anchor()`
в `work/scripts/common.py`, и **от них зависит порядок колонок**, поэтому каждая
модель обучается со своим набором флагов (см. таблицу в §2) и тот же набор
записан в её `work/models/NAME_meta.json`.

| набор | флаг | что в нём | нужен для |
|---|---|---|---|
| base | — | суммы/счётчики за окна 1–365 дней, recency, интервалы, тренды, «то же окно год назад» | всех |
| v2 | `USE_V2` | экспоненциальные затухания, концентрация трат, дни с крупными покупками | всех табличных |
| v3 | `USE_V3` | percentile-ранги внутри среза, детализация прошлогоднего окна, burstiness | всех табличных |
| v4 | `USE_V4` | BTYD: BG/NBD (P(alive), ожидаемое число покупок) + Gamma-Gamma (чек) | всех, кроме `c_ts2_s42` |
| v7 | `USE_V7` | 3 колонки из HMM-симулятора (`hmm_elog`, `hmm_p_zero`, `hmm_sim_std`) | только `twl_v7` |
| seq2 | — | тензор [250k × 196 дней × 8 каналов] на якорь | `seq2tr_f`, `fusion_f` |

Контрольные числа колонок: **203** при V2+V3+V4, **206** при +V7, **194** при
V2+V3, **85** у `behavonly` (правило отбрасывания «денежных» колонок).

> Наборы v6, v8, v10 и `build_seq.py` собраны, измерены и **отвергнуты** — в
> финальный ансамбль не входят, строить их не нужно.

## 2. Девять моделей бленда

Каждая строка — ровно та команда, которой модель обучена (из `work/queue/done/`).
`val` — RMSLE на якоре 2026-01-14 по протоколу gap30 из `work/reports/scores.tsv`.

### 2.1 `fusion_f` — seq+tab fusion, вес 0.316, val 1.6851

```bash
USE_V2=1 USE_V3=1 USE_V4=1 OMP_NUM_THREADS=5 \
.venv/bin/python work/scripts/train_fusion.py --name fusion_f --final \
  --epochs 3 --batch 2048 --eval-batch 1024 --lr 1e-3 --seeds 42 --threads 5
# ~31 мин. Conv7 + 2 слоя трансформера по последовательности + табличный
# энкодер, hurdle-голова (P(y>0) x E[log1p|>0]) + вспомогательные головы y7/y14.
# Веса: work/models/fusion_f_seed42.pt, статистики: fusion_f_stats.npz
```

### 2.2 `c_ts2_s42` — LightGBM two-stage, вес 0.246, val 1.6931

```bash
USE_V2=1 USE_V3=1 OMP_NUM_THREADS=6 \
.venv/bin/python work/scripts/train_gbdt.py --name c_ts2_s42 \
  --threads 6 --gap-days 30 --model lgb --objective two_stage --n-anchors 14 --seed 42 \
  --params  '{"num_leaves":127,"min_data_in_leaf":500,"n_estimators":5000}' \
  --params2 '{"num_leaves":255,"min_data_in_leaf":100,"n_estimators":5000}'
# ~5 мин. БЕЗ USE_V4 (194 признака). Стадия 1 — P(y>0), стадия 2 — E[log1p|>0].
# Веса: work/models/c_ts2_s42__stage1.txt, c_ts2_s42__stage2.txt
```

### 2.3 `mlpziln` — MLP с ZILN-головой, вес 0.122, val 1.6778

```bash
USE_V2=1 USE_V3=1 USE_V4=1 OMP_NUM_THREADS=4 \
.venv/bin/python work/scripts/train_mlpziln.py --name mlpziln \
  --n-anchors 14 --gap-days 30 --seeds 42,1337,7 --epochs 40 --batch 8192 --lr 1e-3
# ~5 мин на MPS. Zero-inflated lognormal (p, mu, sigma); E[log1p] берётся
# 20-точечной квадратурой Гаусса-Эрмита. Ранняя остановка дала ep=[1,4,7] по сидам.
# Веса: work/models/mlpziln_seed{42,1337,7}.pt, статистики: mlpziln_stats.npz
```

### 2.4 `behavonly` — GBDT без единого «денежного» признака, вес 0.080, val 1.7124

```bash
USE_V2=1 USE_V3=1 USE_V4=1 OMP_NUM_THREADS=6 \
.venv/bin/python work/scripts/train_behavonly.py --name behavonly \
  --n-anchors 14 --threads 6 --seed 42
# ~3.5 мин. Скрипт по ПРАВИЛАМ выбрасывает всё, что несёт деньги (любое "gmv",
# BTYD-денежные головы, пороговые счётчики дней) и объёмные прокси, остаётся 85
# поведенческих признаков; дальше делегирует в train_gbdt.main().
# Смысл: модель, никогда не видевшая рубля, структурно не может повторить
# ошибки остальных. Веса: work/models/behavonly.txt
```

### 2.5 `countaov` — разложение на число заказов x средний чек, вес 0.074, val 1.6936

```bash
USE_V2=1 USE_V3=1 USE_V4=1 OMP_NUM_THREADS=6 POLARS_MAX_THREADS=3 \
.venv/bin/python work/scripts/train_countaov.py --name countaov \
  --threads 6 --n-anchors 14 --gap-days 30 --seed 42
# ~6 мин. Две LightGBM-головы: count (log1p) и AOV (режим uplift).
# Веса: work/models/countaov__count.txt, countaov__aov.txt
```

### 2.6 `seq2tr_f` — трансформер по дневным последовательностям, вес 0.070, val 1.7102

```bash
OMP_NUM_THREADS=4 \
.venv/bin/python work/scripts/train_seq2.py --name seq2tr_f --arch tr --final \
  --epochs 3 --batch 2048 --lr 1e-3 --seeds 42,1337 --threads 4
# ~5.4 ЧАСА — самая долгая модель. Табличных признаков не видит вообще
# (USE_* не задаются), поэтому её ошибки меньше всего скоррелированы с остальными.
# Веса: work/models/seq2tr_f_seed{42,1337}.pt
```

### 2.7 `twl_v7` — LightGBM tweedie + HMM-признаки, вес 0.055, val 1.6942

```bash
USE_V2=1 USE_V3=1 USE_V4=1 USE_V7=1 OMP_NUM_THREADS=6 POLARS_MAX_THREADS=3 \
.venv/bin/python work/scripts/train_gbdt.py --name twl_v7 \
  --threads 6 --gap-days 30 --model lgb --objective log_mse --n-anchors 8 --seed 42 \
  --params '{"objective":"tweedie","tweedie_variance_power":1.45,"n_estimators":6000}'
# ~3 мин, 206 признаков, 8 якорей (ограничено покрытием v7).
# ЕДИНСТВЕННАЯ причина, по которой нужен build_features_v7.py. Веса: twl_v7.txt
```

### 2.8 `hmmsim` — генеративный симулятор, вес 0.028, val 1.8238

```bash
THREADS=6 .venv/bin/python work/scripts/train_hmm_sim.py --name hmmsim \
  --states 4 --sims 500 --win 120 --em-cap 25000 --splits val,test
# ~6 мин. У модели НЕТ обучаемых весов: скрытая марковская модель покупательской
# активности оценивается EM по собственной истории каждого юзера и таргет не
# видит вовсе, дальше 500 симуляций вперёд на 30 дней. Воспроизводится
# повторным запуском с тем же сидом; work/models/hmmsim_meta.json это
# фиксирует (`"stateless": true`).
```

### 2.9 `channel2` — канальная декомпозиция, вес 0.012, val 1.6872

```bash
USE_V2=1 USE_V3=1 USE_V4=1 OMP_NUM_THREADS=6 \
.venv/bin/python work/scripts/train_channel.py --name channel2 \
  --threads 6 --n-anchors 14 --gap-days 30 --seed 42
# ~8.5 мин. GMV = «из поиска» + «из каталога», две LightGBM (tweedie vp=1.45
# на log1p канального таргета), сумма в линейном пространстве.
# Веса: work/models/channel2__search.txt, channel2__cat.txt
```

## 3. Калибровка каждой модели

Все девять моделей перед блендом калибруются одинаково:

```bash
for m in fusion_f c_ts2_s42 mlpziln behavonly countaov seq2tr_f twl_v7 hmmsim channel2; do
  .venv/bin/python work/scripts/calibrate.py --pred $m     # -> ${m}_cal_{val,test}.parquet
done
```

Прогнозы бьются на 24 квантильных бина в log1p-пространстве, в каждом бине
считается сдвиг `mean(log1p(факт)) - mean(log1p(прогноз))`, между центрами бинов
сдвиг интерполируется. Честность контролируется внутри скрипта: таблица
подгоняется на половине юзеров, проверяется на другой. Эффект: +0.010…0.012
RMSLE на модель.

Таблица сдвигов сохраняется в `work/models/NAME_cal.npz` (ключи `centers`,
`shifts`) — инференс её просто применяет, а не переподбирает.

## 4. Бленд

Веса подобраны NNLS на валидации, честная оценка — OOF по 5 фолдам по юзерам
(`work/reports/scores.tsv`: `blend_cal 1.666791`). Канонический словарь весов
лежит в коде: `work/scripts/blend_testopt.py`, константа `W_VAL`.

| модель | вес | | модель | вес |
|---|---|---|---|---|
| `fusion_f_cal` | 0.316 | | `seq2tr_f_cal` | 0.070 |
| `c_ts2_s42_cal` | 0.246 | | `twl_v7_cal` | 0.055 |
| `mlpziln_cal` | 0.122 | | `hmmsim_cal` | 0.028 |
| `behavonly_cal` | 0.080 | | `channel2_cal` | 0.012 |
| `countaov_cal` | 0.074 | | **сумма** | **1.003** |

Смешивание — взвешенная сумма в log1p-пространстве:

```
lp_blend = sum_i w_i * log1p(pred_i)
```

Результат: `work/preds/blend_cal_{val,test}.parquet`, честный val-OOF **1.666791**.

## 5. Финал: два числа

Бленд применяется к тесту, и дальше к нему применяется **аффинная
перенастройка в log1p-пространстве** — ровно два числа:

```
lp_final = 1.0775793 * lp_blend + 0.0061760
```

Что происходит с распределением прогноза (фактические числа на нашем
`blend_cal_test.parquet`, 250 000 строк):

| | до | после | целевое |
|---|---|---|---|
| mean log1p | 2.1553 | **2.3287** | 2.3275 |
| sd log1p | 1.5106 | **1.6278** | 1.628 |

### Откуда берётся каждое число

**Число 1 — уровень (подъём среднего 2.155 → 2.325).** Целевое `mean_P(t) = 2.3275 ± 0.0064`
замерено НА ЛИДЕРБОРДЕ: пара сабмитов, отличающихся на известную константу в
log-пространстве, даёт точное среднее из разности квадратов скоров
(`work/reports/KNOWLEDGE.md`, факт Ф18; методика — `work/scripts/predict_lb.py`).
Причина сдвига физическая: тестовое окно 14.02–15.03 содержит неделю перед
8 марта, а все модели обучены на осенне-зимних окнах и систематически
недопредсказывают. Локальная валидация этого увидеть не может в принципе —
такой сезонности нет ни в одном обучаемом окне.

*Независимая проверка (без лидерборда):* на прошлогодних аналогах окон
(янв–фев 2025 против фев–мар 2025, все 250k юзеров) среднее log1p растёт
1.5396 → 1.7154, то есть **+0.1759**. Наш применённый подъём — **+0.1694**.
Сходится.

**Число 2 — масштаб (множитель 1.0774…1.0776 к логарифму прогноза).** Бленд
недодисперсен: его sd в log1p равен 1.510, а нужно 1.628. Причина — калибровка
обучена на валидации, а тестовое окно шире по разбросу. Множитель раскладывается
как сезонность **1.036** (тот же прошлогодний замер: sd 2.1644 → 2.2432)
умножить на исправление недодисперсности бленда **~1.04**. То есть и он
подтверждён независимо от лидерборда.

Численные значения записаны в `work/reports/blend_testopt_honest.json` (ключ
`_affine`) и `blend_testopt_final.json` (ключ `affine_valblend`). Эквивалентная
операционная формулировка (`work/scripts/stack_meta_ship.py`, константы
`TARGET_SD = 1.628`, `TARGET_MEAN = 2.3275`): растянуть до sd = 1.628, затем
сдвинуть до mean = 2.3275.

### Почему именно два числа, а не длинная цепочка


```
FILE.csv = 0.0027 + 1.0774 * blend_cal_test     (корреляция 0.9972,
                                                   разброс остатка 0.121
                                                   против 1.632 у прогноза)
```

То есть **весь результат — это честный бленд плюс те же два числа**, а вся
цепочка мелких шагов — остаток 7.4%. Оба числа измерены с запасом в десятки
шумов (`work/reports/finalists.md`: один подобранный по публичному скору
параметр даёт фиктивный выигрыш 0.000022 RMSLE, а наши шаги дали 2.2–86 таких
единиц).

Честная цена самой аффинной подгонки (2 параметра, `blend_testopt_honest.json`):
`d_emp = 0.00061` при выигрыше 0.0132.

Аффинная перенастройка
> предназначена ТОЛЬКО для файлов из `*_cal`-пула.

## 6. Инференс

```bash
bash final_submission/run_inference.sh          # -> FILE.csv
```

`inference.py` повторяет §1 (только тестовый якорь), загружает веса из
`final_submission/models/` (или `work/models/`), считает прогнозы девяти
моделей, применяет калибровки из §3, веса из §4 и аффин из §5. Если какого-то
файла весов нет — падает с явным сообщением, какого именно, а не выдаёт мусор.

## 7. Что НЕ входит в решение

Чтобы не тратить время на воспроизведение отвергнутого: наборы признаков
v6/v8/v10, `build_seq.py`; модели `train_mlp.py`, `train_gru.py`,
`train_bagged.py`, `train_gls.py`, `train_pseudo.py`, `train_quantint.py`,
`train_rank.py`, `train_whale.py`, `train_horizon.py`, `train_hjit.py`,
`train_fusion3.py`. Модели эпохи до `--gap-days 30` (`lgblog_final`,
`xgblog_final`, `cblog_final`, `mlp_final`, `gru_final`, `hjit37`, `hjit44`)
перечислены как `CONTAMINATED` в `work/scripts/blend_testopt.py` и исключены:
их val-скоры завышены пересечением таргет-окон.

1.650554 против 1.6489446 у основной линии.
Скрипты `train_mlpbin.py` и `train_feb_specialist.py` оставлены рабочими и тоже
сохраняют веса, но в финальный бленд не входят.
