# models/ — артефакты финального решения

Сюда кладётся всё, что читает `../inference.py`, и ничего не скачивается.
Как получить каждый артефакт — `../reproduce_training.md`.

**Файлы весов создают сами трейнеры.** Производственный трейнер на retrain-фазе вызывает
`work/scripts/model_io.py` и пишет артефакты в `work/models/`. Инференс ищет файл сначала
в этом каталоге, потом в `work/models/`, поэтому после переобучения ничего копировать
вручную не обязательно — копирование нужно только чтобы собрать самодостаточный пакет
сдачи (команда — в `../README.md`, раздел «Упаковка»).

Что уже есть, а чего не хватает, показывает:

```bash
python final_submission/inference.py --stage check
```

## Контракт именования

`NAME` — значение `--name` у трейнера (`mlpziln_c42`, `c_ts2_s7`, `weak_an_d`, …).

| файл | что это | кто пишет |
|---|---|---|
| `NAME_meta.json` | порядок колонок признаков, конфиг архитектуры, сиды, флаги `USE_V*`, список файлов весов | `model_io.save_meta()` |
| `NAME_stats.npz` | препроцессинг нейросетей: `med/lo/hi/mean/std` | трейнер |
| `NAME_seed{S}.pt` | `state_dict` torch-модели retrain-фазы, по одному на сид, тензоры на CPU | `model_io.save_torch()` |
| `NAME.txt` | бустер LightGBM (`Booster.save_model`) | `model_io.save_lgb()` |
| `NAME__TAG.txt` | бустер LightGBM подмодели TAG (`__stage1`/`__stage2`, `__count`/`__aov`) | `model_io.save_lgb()` |
| `NAME.xgb.json` | бустер XGBoost | `model_io.save_xgb()` |
| `NAME_cal.npz` | поквантильная калибровка: `centers`, `shifts` | `calibrate.py` |
| `chain_test.npz` | замороженная цепочка (см. ниже) | `inference.py --stage freeze` |
| `silence_p_test.npz` | вероятность молчания для 250 000 (кэш шага 6) | `inference.py --stage silence` |
| `preds_test/` | кэш стадии `predict` | `inference.py` |

## Что ожидается для 25 базовых моделей

25 базовых моделей складываются в 20 членов бленда (три семейства усредняются по сидам).
Полная таблица весов — `../reproduce_training.md` §5.

| базовая модель | веса | воспроизводимость |
|---|---|---|
| `mlpziln_c42`, `mlpziln_c1337`, `mlpziln_c7` | `NAME_seed{S}.pt` + `NAME_stats.npz` | веса сохраняются |
| `c_ts2_s42`, `c_ts2_s7` | `NAME__stage1.txt`, `NAME__stage2.txt` | веса сохраняются |
| `behavonly`, `behavonly_s1337`, `behavonly_s7` | `NAME.txt` | веса сохраняются |
| `weak_an_d`, `weak_ft_recency`, `weak_ft_counts`, `weak_ft_long90` | `NAME.txt` | веса сохраняются |
| `twl_v7` | `twl_v7.txt` | веса сохраняются |
| `countaov_s7` | `countaov_s7__count.txt`, `countaov_s7__aov.txt` | веса сохраняются |
| `c_xtw_s42` | `c_xtw_s42.xgb.json` | веса сохраняются |
| `seq2tr_f` | `seq2tr_f_seed{42,1337}.pt` | веса сохраняются, но тензоры `work/seq2` удалены |
| `fusion_v3c42`, `fusion_v3c555`, `fusion_v3c7`, `fusion_v3`, `fusion_v3ctl` | — | **весов нет: `train_fusion3.py` не вызывает `model_io`** |
| `wklin`, `wklin_wk` | — | **весов нет: `train_wklin.py` не вызывает `model_io`** |
| `febspec2` | — | **весов нет: `train_febspec2.py` не вызывает `model_io`** |
| `hmmsim` | — | **весов нет по построению** (см. ниже) |

Плюс `NAME_meta.json` у каждой модели, которая сохраняет веса, и `NAME_cal.npz` у каждого
калиброванного члена бленда.

Для четырёх последних групп артефактом является сам прогноз
`work/preds/NAME_test.parquet`, и другого способа получить его, кроме повторного
обучения, нет.

`hmmsim` — генеративный симулятор: скрытая марковская модель оценивается EM по
собственной истории каждого пользователя и таргет не видит вообще, поэтому обучаемых
весов у неё нет. `hmmsim_meta.json` помечен `"stateless": true` и хранит гиперпараметры;
инференс при отсутствии готового прогноза пересчитывает симулятор с тем же сидом (~6 мин).

## chain_test.npz — измеренная цепочка

Три вектора по 250 000. Каждый — **разность отправленных файлов, а не выход модели**,
поэтому пересчитать их из весов нельзя и они входят в пакет как данные.

| ключ | что это |
|---|---|
| `user_id` | порядок строк (отсортирован) |
| `carry_lp` | накопленная цепочка LB-поправок: `ref_lp` минус приведённый к тем же моментам старый бленд `blend_cal`. Ровно то, что делает `make_candidate.py --carry-from blend_cal` |


`../reproduce_training.md` §7.4).

## Константы, зашитые в inference.py

Зашиты в код, а не в json, чтобы сабмит нельзя было испортить подменой конфига:

* `BLEND_WEIGHTS` — 20 весов, источник `work/reports/blend_reopt.json`, ключ `winner`
  (библиотека `B_plus_cal`, метод `ridge_free`, `alpha_rel` 1e-4). Сверка с текущим
  отчётом: `inference.py --stage check --verify-blend`;
* `REF_MEAN = 2.324718…`, `REF_SD = 1.632001…` — моменты log1p, замеренные на лидерборде;
* `STEP = 0.469` — сила шага от опорного файла к новому кандидату;
* `Q_REF = 0.0027149`, `P_LEVEL = 0.030843`, `A_OLD = 0.894`, `A_NEW = 0.65` — поправка на
  молчащих.

Что означает каждое число и откуда взято — `../reproduce_training.md` §6 и §7.
