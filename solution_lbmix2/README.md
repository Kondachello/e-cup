# lbmix2 — решение E-CUP 2026 Задача 3 (public LB 1.65896)

> **ИСТОРИЧЕСКИЙ ПАКЕТ, НЕ ДЕЙСТВУЮЩЕЕ РЕШЕНИЕ.** Он воспроизводит сабмит
> `lbmix2.csv` от 18.08 (public 1.65896). Действующее решение — в
> `final_submission/`; на 23.08 лучший честный замер около 1.6472.
> `solution_lbmix2/scripts/` — ЗАМОРОЖЕННЫЕ КОПИИ пяти скриптов
> (`build_features.py`, `build_features_v2.py`, `common.py`, `exp_lib.py`,
> `train_gbdt.py`), которые с тех пор разошлись с `work/scripts/` (train_gbdt
> на ~315 строк, common на ~144). Так и задумано: пакет обязан собирать ровно
> тот файл, скор которого замерен. Править их вместе с `work/scripts/` НЕ надо,
> и брать из них код для новой работы — тоже.


## Состав

```
solution_lbmix2/
├── README.md            # этот файл
├── requirements.txt     # пиненые версии (Python 3.10)
├── run_all.sh           # end-to-end воспроизведение
├── scripts/             # весь код пайплайна
└── artifacts/           # готовые артефакты для проверки без тренировки:
    ├── sub_blend_w1a.csv   # наша компонента A (public 1.6754553658578413)
    └── lbmix2.csv          # финальный сабмит (public 1.65896)
```

## Входные данные (положить в корень пакета или задать OZON_ROOT)

кладите рядом или укажите путь)

## Окружение

```bash
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
export OZON_ROOT=$(pwd)   # корень с train.parquet
```

Железо референса: Apple M1 Pro, 10 ядер, 16GB RAM. При меньшей RAM у XGBoost
использовать `--n-anchors 8` (мы так и делали), у LightGBM можно `--n-anchors 10`.

## Пайплайн (или просто `bash run_all.sh`)

### 1. Фичи: 149 признаков × 16 временных якорей (~3 мин)

```bash
python scripts/build_features.py --preset all
```

Якоря: TEST=2026-02-13, VAL=2026-01-14, 14 тренировочных назад с шагом 14 дней
(2025-07-02…2025-12-31). На каждый якорь: окна 1/3/7/14/30/60/90/180/365 дней
(gmv/заказы/корзины/поиски/активные дни), диз-джойнт полосы для трендов, окна
«тот же период год назад» (ya_tgt = [A-364, A-335] — для теста это ровно
14 фев–15 мар 2025, ловит сезонность 8 марта), recency по всем типам событий,
статистики межпокупочных интервалов, производные конверсии/шеры/тренды.
Таргет: сумма gmv юзера за [A+1, A+30], отсутствующие = 0.

### 2. Две GBDT-модели (контракт: train<VAL, early stop на VAL; ретрейн train+VAL с best_iter×1.07 → тест-предсказания)

```bash
# LightGBM (val RMSLE 1.6334): якорное взвешивание по свежести tau=150
python scripts/train_gbdt.py --name lgblog_final --model lgb --objective log_mse \
  --weight-tau 150 --drop-cols seasonal_index \
  --params '{"num_leaves":255,"min_data_in_leaf":300,"learning_rate":0.05,"feature_fraction":0.75,"n_estimators":5000}'

# XGBoost (val RMSLE 1.6304): глубокие деревья, 8 последних якорей (RAM)
python scripts/train_gbdt.py --name xgblog_final --model xgb --objective log_mse \
  --n-anchors 8 \
  --params '{"max_leaves":511,"min_child_weight":100,"learning_rate":0.05,"colsample_bytree":0.8}'
```

Обе — регрессия MSE на log1p(target), т.е. прямая оптимизация RMSLE. Сиды
зафиксированы (42), результат детерминирован при тех же версиях библиотек.

### 3. Бленд двух моделей (val RMSLE 1.6233)

```bash
python scripts/blend.py --include lgblog_final,xgblog_final,base_best --name blend_w1a --scale-grid
python scripts/make_submission.py --pred blend_w1a --out sub_blend_w1a
```

Жадный hill-climb в log1p-пространстве → веса 50/50 + глобальный масштаб 1.005
(подобран сеткой на валидации). (`base_best` — эвристика, hill-climb даёт ей вес 0;
можно опустить `--include` вовсе.)

### 4. Финальный микс с компонентой тиммейта

```bash
python scripts/mix_lbmix2.py --a sub_blend_w1a.csv --b FILE.csv --out lbmix2.csv
```

Вес подбирается **математически без траты сабмитов** из двух известных
public-скоров и локального расхождения предсказаний. Для ошибок в log1p-простр.
a = lpA − ly, b = lpB − ly на public-подмножестве:

- fa² = E[a²], fb² = E[b²] — известны с ЛБ (1.67546², 1.66218²)
- D² = E[(a−b)²] = E[(lpA−lpB)²] — считается локально (одинаково на любом
  подмножестве юзеров при n=50k)
- cov = (fa²+fb²−D²)/2, оптимальный вес w_B = (fa²−cov)/(fa²+fb²−2cov) ≈ 0.702
- прогноз качества смеси: sqrt(fa² − (fa²−cov)²/(fa²+fb²−2cov)) = 1.65924
  (факт на ЛБ: **1.65896** — сошлось с точностью 0.0003)

## Проверка без тренировки

`artifacts/lbmix2.csv` — тот самый файл.

## Валидация и известные числа

| Компонент | val RMSLE (окно 15 янв–13 фев) | public LB (окно 14 фев–15 мар) |
|---|---|---|
| lgblog_final | 1.6334 | — |
| xgblog_final | 1.6304 | — |
| blend_w1a (A) | 1.6233 | 1.67546 |
| **lbmix2** | — | **1.65896** |

Разрыв val↔LB — разная сложность окон (тестовое окно с 8 марта тяжелее);
ранжирование моделей валидация передаёт хорошо, корреляция ошибок A и B = 0.975.
