# Воспроизведение обучения моделей финального ансамбля

Все команды запускаются из корня репозитория; окружение — `requirements.txt`
(Python 3.10). Референсное железо: Apple M1 Pro, 10 ядер, 16 GB RAM (нейросети
обучались на MPS; на CPU — те же команды, дольше). Времена ниже — фактические
с этой машины. Сиды зафиксированы в командах; результаты бустингов при тех же
версиях библиотек детерминированы, у torch-моделей возможен минимальный дрейф
между устройствами (MPS/CPU/CUDA).

На 16 GB RAM тяжёлые обучения запускать ПОСЛЕДОВАТЕЛЬНО (у нас — очередь
`work/scripts/queue_runner.py` + `enqueue.py`; параллельные запуски дважды
роняли машину).

Общий для всех протокол (`exp_lib` контракт): обучение на срезах с зазором
30 дней до валидации (`--gap-days 30`), ранняя остановка по срезу 2026-01-14,
затем переобучение с добавлением поздних срезов (итерации/эпохи масштабируются)
и прогноз тестового среза 2026-02-13. Прогнозы каждой модели сохраняются в
`work/preds/ИМЯ_{val,test}.parquet`, скор — строкой в `work/reports/scores.tsv`.

## 0. Признаки (один раз, ~15–25 мин суммарно)

```bash
python3.10 -m venv .venv && .venv/bin/pip install -r final_submission/requirements.txt
# train.parquet и sample_submit.csv — в корень репозитория (или задать OZON_ROOT)
.venv/bin/python work/scripts/build_features.py --preset all   # base, 16 срезов, ~3 мин
.venv/bin/python work/scripts/build_features_v2.py             # ~3 мин
.venv/bin/python work/scripts/build_features_v3.py             # ~4 мин
.venv/bin/python work/scripts/build_features_v4.py             # BTYD, ~5-10 мин
.venv/bin/python work/scripts/build_seq.py                     # тензоры для GRU, ~5 мин
.venv/bin/python work/scripts/build_channel_targets.py         # канальные таргеты, ~2 мин
```

## 1. MLP с ZILN-головой → `mlpziln`, калибровка → `mlpziln_cal` (val 1.6687)

```bash
USE_V2=1 USE_V3=1 USE_V4=1 OMP_NUM_THREADS=4 \
.venv/bin/python work/scripts/train_mlpziln.py --name mlpziln \
  --n-anchors 14 --gap-days 30 --seeds 42,1337,7 --epochs 40 --batch 8192 --lr 1e-3
# ~5 мин на MPS; val RMSLE 1.6778 (scores.tsv: ep=[1,4,7] по сидам)

.venv/bin/python work/scripts/calibrate.py --pred mlpziln
# -> mlpziln_cal, val RMSLE 1.6687 (honesty-контроль на половине юзеров внутри скрипта)
```

## 2. MLP-классификация по бинам → `mlpbin`, калибровка → `mlpbin_cal` (val 1.6688)

```bash
USE_V2=1 USE_V3=1 USE_V4=1 OMP_NUM_THREADS=4 \
.venv/bin/python work/scripts/train_mlpbin.py --name mlpbin \
  --n-anchors 14 --gap-days 30 --seeds 42,1337,7 --epochs 40 --batch 8192 --lr 1e-3
# ~3.5 мин на MPS; val RMSLE 1.6764 (ep=[1,3,7] по сидам)

.venv/bin/python work/scripts/calibrate.py --pred mlpbin
# -> mlpbin_cal, val RMSLE 1.6688
```

## 3. Канальная декомпозиция → `channel2` (val 1.6872)

```bash
USE_V2=1 USE_V3=1 USE_V4=1 OMP_NUM_THREADS=6 \
.venv/bin/python work/scripts/train_channel.py --name channel2 \
  --threads 6 --n-anchors 14 --gap-days 30 --seed 42
# ~8.5 мин; два LightGBM tweedie(vp=1.45) на log1p канальных таргетов,
# nl255 mdl300 lr0.05 ff0.75; val total 1.6872 (search 1.6794, cat 0.8670)
```

## 4. XGBoost tweedie-on-log1p → `c_xtw_s42` (val 1.6979)

```bash
USE_V2=1 USE_V3=1 OMP_NUM_THREADS=6 \
.venv/bin/python work/scripts/train_gbdt.py --name c_xtw_s42 \
  --threads 6 --gap-days 30 --model xgb --objective log_mse --n-anchors 10 --seed 42 \
  --params '{"objective":"reg:tweedie","tweedie_variance_power":1.2,"max_leaves":511,"min_child_weight":100,"learning_rate":0.05,"colsample_bytree":0.8,"n_estimators":6000}'
# ~4 мин; 194 признака (base+v2+v3); val RMSLE 1.6979
# при нехватке RAM: экономный клон .venv/bin/python work/scripts/train_xtw.py (те же аргументы)
```

## 5. GRU по дневным последовательностям → `gru_final` (val 1.6988)

```bash
.venv/bin/python work/scripts/train_gru.py --name gru_final
# дефолты скрипта = финальная конфигурация: hidden 96 x 2 слоя, dropout 0.1,
# batch 4096, lr 2e-3, 8 срезов; ~20-30 мин на MPS; val RMSLE 1.6988
```

## 6. Сборка ансамбля

```bash
# 6.1 веса бленда координатным спуском на валидации (log1p-пространство)
.venv/bin/python work/scripts/assemble_final.py --name final_blend
# (простой вариант: work/scripts/blend.py --include mlpziln_cal,mlpbin_cal,channel2,c_xtw_s42,gru_final)

# 6.2 при необходимости — калибровка итогового бленда
.venv/bin/python work/scripts/calibrate.py --pred final_blend

# 6.3 объединение с файлами с известным public-скором: веса из ковариационной
#     алгебры (скор смеси предсказывается ДО отправки, точность ~0.0005)
.venv/bin/python work/scripts/lb_blend.py --a submissions/FILE.csv --fa <скор A> \
                                          --b submissions/FILE.csv --fb <скор B>

# 6.4 сезонная и сегментные поправки, замеренные по публичному ЛБ, с усадкой
#     под private -> финальные кандидаты (консервативный и основной)
.venv/bin/python work/scripts/make_finalists.py
```

Величины поправок и веса фиксируются в `final_submission/models/blend_config.json`
и `lb_corrections.json` (см. `models/README.md`).

## 7. Экспорт артефактов в `final_submission/models/` (шаг freeze)

Трейнеры пишут прогнозы и `*_stats.npz`, но не сохраняют веса моделей
(модель живёт в памяти процесса). Перед финальной фиксацией:

1. В retrain-фазу каждого трейнера добавить сохранение модели:
   `torch.save(model.state_dict(), ...)` (mlpziln / mlpbin / gru),
   `booster.save_model(...)` (train_gbdt xgb, train_channel lgb).
2. Прогнать команды п.1–5 заново (сиды те же — скоры воспроизводятся).
3. Скопировать веса + `work/models/{mlpziln,mlpbin}_stats.npz` + таблицы
   калибровки + конфиги в `final_submission/models/` по контракту из
   `models/README.md`.
4.
