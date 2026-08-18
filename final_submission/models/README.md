# models/ — обученные артефакты финального ансамбля

Каталог заполняется при финальной фиксации решения (freeze). Инференс
(`../inference.py`) читает отсюда всё, что нужно, и ничего не скачивает.
Как обучить каждую модель — `../reproduce_training.md`.

## Контракт: какие файлы здесь лежат

| Файл | Что это | Откуда берётся |
|---|---|---|
| `blend_config.json` | веса бленда в log1p + списки колонок признаков | фиксация весов после отбора на валидации (`assemble_final.py`), см. пример ниже |
| `lb_corrections.json` | глобальный сезонный сдвиг и сегментные поправки (уже с усадкой под private) | замеры по публичному ЛБ, README §5.3–5.4 |
| `xgb_tweedie_log.json` | бустер XGBoost (tweedie на log1p) | `Booster.save_model()` после retrain-фазы `train_gbdt.py --model xgb` |
| `channel_search.txt`, `channel_cat.txt` | два бустера LightGBM канальной модели | `Booster.save_model()` после retrain-фазы `train_channel.py` |
| `mlp_ziln_seed{42,1337,7}.pt` | state_dict ZILN-MLP по сидам (retrain-веса) | `torch.save()` в `train_mlpziln.py` |
| `mlp_ziln_stats.npz` | препроцессинг: med/lo/hi/mean/std (ключи как в `work/models/mlpziln_stats.npz`) | пишется трейнером |
| `mlp_bin_seed{42,1337,7}.pt` | state_dict биновой MLP по сидам | `torch.save()` в `train_mlpbin.py` |
| `mlp_bin_stats.npz` | препроцессинг + `edges`/`centers` бинов (как в `work/models/mlpbin_stats.npz`) | пишется трейнером |
| `gru_seed42.pt` | state_dict GRU (retrain-веса) | `torch.save()` в `train_gru.py` |
| `calibration_mlp_ziln.npz`, `calibration_mlp_bin.npz` | поквантильные сдвиги: массивы `centers`, `shifts` | `calibrate.py` (fit на валидации) |
| `calibration_blend.npz` | то же для итогового бленда (опционально) | `calibrate.py` |
| `preds_test/` | создаётся инференсом: log1p-прогнозы моделей (кэш стадии predict) | — |

ВНИМАНИЕ: тренировочные скрипты сейчас сохраняют только `*_stats.npz`
(препроцессинг), но не веса моделей — они держат модель в памяти и сразу пишут
прогнозы. Перед freeze в каждый трейнер добавляется сохранение артефактов
retrain-фазы (`torch.save(state_dict)` / `booster.save_model()`) — помечено
в `reproduce_training.md`, шаг «Экспорт артефактов».

## Пример `blend_config.json`

Веса — иллюстративные; финальные фиксируются отбором на валидации перед сдачей.

```json
{
  "space": "log1p",
  "weights": {
    "mlp_ziln": 0.40,
    "mlp_bin": 0.30,
    "channels": 0.15,
    "xgb_tweedie_log": 0.10,
    "gru": 0.05
  },
  "feature_columns": {
    "tabular": ["...203 имени колонок base+v2+v3+v4 в порядке обучения..."],
    "xgb": ["...194 имени колонок base+v2+v3 (модель обучена без v4)..."]
  }
}
```

## Пример `lb_corrections.json`

Все сдвиги — в log1p-пространстве; величины замерены по публичному лидерборду
валидными сабмитами (методика в README §5.3) и уже умножены на консервативные
shrinkage-коэффициенты переноса на private (README §5.4).

```json
{
  "global_log_shift": 0.1163,
  "comment": "сезонный недопрогноз тестового окна (8 марта); замер парой сабмитов, sigma~0.001",
  "segments": [
    {"name": "example_high_activity", "column": "gmv_sum_90d", "op": ">=",
     "threshold": 1000.0, "log_shift": 0.0, "comment": "заполняется при freeze"}
  ]
}
```
