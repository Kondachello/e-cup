# models/ — обученные артефакты финального ансамбля

Сюда кладутся веса девяти моделей бленда. `../inference.py` читает отсюда всё,
что нужно, и ничего не скачивает. Как обучить каждую модель — `../reproduce_training.md`.

**Файлы создают сами трейнеры.** Каждый производственный трейнер на retrain-фазе
вызывает `work/scripts/model_io.py` и пишет артефакты в `work/models/`.
Инференс ищет файл сначала в этом каталоге, потом в `work/models/`, поэтому
после переобучения ничего копировать вручную не обязательно — копирование нужно
только чтобы собрать самодостаточный пакет сдачи.

Что уже есть, а чего не хватает, показывает:

```bash
python final_submission/inference.py --stage check
```

## Контракт именования

`NAME` — значение `--name` у трейнера (`mlpziln`, `fusion_f`, `c_ts2_s42`, …).

| файл | что это | кто пишет |
|---|---|---|
| `NAME_meta.json` | порядок колонок признаков, конфиг архитектуры, сиды, флаги `USE_V*`, список файлов весов | `model_io.save_meta()` |
| `NAME_stats.npz` | препроцессинг нейросетей: `med/lo/hi/mean/std` (+ `edges`/`centers` у биновой) | трейнер (как и раньше) |
| `NAME_seed{S}.pt` | `state_dict` torch-модели retrain-фазы, по одному на сид, тензоры на CPU | `model_io.save_torch()` |
| `NAME.txt` | бустер LightGBM (`Booster.save_model`) | `model_io.save_lgb()` |
| `NAME__TAG.txt` | бустер LightGBM подмодели TAG (`__search`/`__cat`, `__count`/`__aov`, `__stage1`/`__stage2`) | `model_io.save_lgb()` |
| `NAME.xgb.json` | бустер XGBoost | `model_io.save_xgb()` |
| `NAME.cbm` | модель CatBoost (в финальном ансамбле не используется) | `model_io.save_cb()` |
| `NAME_cal.npz` | поквантильная калибровка: `centers`, `shifts` | `calibrate.py` |
| `NAME_chcal.npz` | канальная калибровка `channel2`: `k`, `{search,cat}_{centers,shifts}` | `train_channel.py` |
| `preds_test/` | кэш стадии `predict` (создаётся инференсом) | `inference.py` |

## Что ожидается для каждой модели бленда

| модель | вес | веса модели | калибровка |
|---|---|---|---|
| `fusion_f` | 0.316 | `fusion_f_seed42.pt` + `fusion_f_stats.npz` | `fusion_f_cal.npz` |
| `c_ts2_s42` | 0.246 | `c_ts2_s42__stage1.txt`, `c_ts2_s42__stage2.txt` | `c_ts2_s42_cal.npz` |
| `mlpziln` | 0.122 | `mlpziln_seed{42,1337,7}.pt` + `mlpziln_stats.npz` | `mlpziln_cal.npz` |
| `behavonly` | 0.080 | `behavonly.txt` | `behavonly_cal.npz` |
| `countaov` | 0.074 | `countaov__count.txt`, `countaov__aov.txt` | `countaov_cal.npz` |
| `seq2tr_f` | 0.070 | `seq2tr_f_seed{42,1337}.pt` | `seq2tr_f_cal.npz` |
| `twl_v7` | 0.055 | `twl_v7.txt` | `twl_v7_cal.npz` |
| `hmmsim` | 0.028 | **весов нет по построению** (см. ниже) | `hmmsim_cal.npz` |
| `channel2` | 0.012 | `channel2__search.txt`, `channel2__cat.txt` | `channel2_cal.npz` |

Плюс `NAME_meta.json` для каждой из девяти.

`hmmsim` — генеративный симулятор: скрытая марковская модель оценивается EM по
собственной истории каждого пользователя и таргет не видит вообще, поэтому
обучаемых весов у неё нет. Её `hmmsim_meta.json` помечен `"stateless": true` и
хранит гиперпараметры; инференс при отсутствии готового прогноза просто
пересчитывает симулятор с тем же сидом (~6 мин).

## Веса бленда и финальные два числа

Эти константы зашиты в `../inference.py` (а не в файл конфига), чтобы сабмит
нельзя было испортить подменой json:

* `BLEND_WEIGHTS` — NNLS-веса с валидации, канонический источник
  `work/scripts/blend_testopt.py`, константа `W_VAL`;
* `AFFINE_SLOPE = 1.0775792958468002`, `AFFINE_SHIFT = 0.006176042172469855` —
  аффинная перенастройка в log1p, источник
  `work/reports/blend_testopt_honest.json`, ключ `_affine`.

Что означает каждое число и откуда взято — `../reproduce_training.md` §5.
