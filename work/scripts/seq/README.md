# Секвенсная ветка (задача №5)

Восемь скриптов. Кладутся в `work/scripts/seq/`, запускаются из корня репозитория.

## Порядок

```powershell
python work\scripts\seq\build_tensor.py --src train.parquet --out tensor
python work\scripts\seq\make_valid3.py --data tensor
python work\scripts\seq\train_tcn.py --data tensor --arch transformer --minutes 55 `
  --lr 7e-4 --channels 192 --layers 4 --heads 4 --dropout 0.0 --wd 0.0122 `
  --aux 0.25 --batch 512 --ema 0.995 --min-anchor 30 `
  --val-users 0 --val-anchor 378 --seed 1 --tag tfm_s1 `
  --export work\preds --predict FILE.csv
```

Три сида (`--seed 1/2/3`, `--tag tfm_s1/s2/s3`), затем:

```powershell
python work\scripts\seq\avg_seeds.py --out work\preds\tfm3_val.parquet  work\preds\tfm_s1_val.parquet work\preds\tfm_s2_val.parquet work\preds\tfm_s3_val.parquet
python work\scripts\seq\avg_seeds.py --out work\preds\tfm3_test.parquet work\preds\tfm_s1_test.parquet work\preds\tfm_s2_test.parquet work\preds\tfm_s3_test.parquet

python work\scripts\calibrate.py --pred tfm3 --bins 24
python work\scripts\err_corr.py tfm3_cal
```

## Файлы

| файл | назначение |
|---|---|
| `build_tensor.py` | parquet → тензор `[250000, 409, 10]` fp16, календарь, маска якорей |
| `fix_gmv.py` | пересобирает gmv в fp32; нужен только если тензор строился старой версией |
| `make_valid3.py` | маска якорей по правилу трёх блоков, дописывает в `meta.npz` |
| `train_tcn.py` | обучение: `--arch tcn` или `transformer` |
| `avg_seeds.py` | усреднение сидов в лог-пространстве |
| `sweep.py` | подбор гиперпараметров, две фазы |
| `analyze_sweep.py` | разбор свипа: сводка и графики |
| `add_direction.py` | добавляет модель в готовый сабмит, не размывая поправки |
| `run_all.py` | **обе фазы прогона одной командой**, резюмируемый |
| `margin_vs_pack.py` | ЗАПАС против пакета, в обход неработающего `err_corr.py` |
| `preflight_tfm3.py` | проверка входов до запуска цепочки калибровки |

`CONTEXT.md` — состояние проекта, что уже закрыто измерениями, правила.
`TASK5.md` — задача целиком с обоснованиями.
`TRACK5_REPORT.md` (в корне) — что сделано, измерено и исправлено; читать первым.
`RUN_TFM.md` — как делался прогон от 18.08. `RUN_TFM2.md` — как делать следующий.
`HANDOFF_tfm3.md` — записка треку №1 со всеми числами.

## Прогон одной командой

```powershell
python work\scripts\seq\run_all.py --data C:\ozon\tensor
```

Сам поставит маску когорты, прогонит фазу A (три сида, зазор 30, `--es-metric cal`),
вытащит из результатов число шагов и усадку, прогонит фазу B (переобучение по
якорь 378 для теста) и сложит всё в `_to_kosta/`. Прерывать можно в любой момент —
повторный запуск пропускает сделанное. `--dry-run` показывает план.

## Включено по умолчанию

* `--es-metric cal` — ранняя остановка по калиброванному скору
* когорта обучения по трём блокам (нужен `make_valid3.py`), откат `--cohort1`
* зазор 30 дней: `--val-anchor` сам ставит границу обучающих на `якорь − 30`

## Требования

`torch` со сборкой под CUDA (`--index-url https://download.pytorch.org/whl/cu128`),
`polars`, `pyarrow`, `numpy`, `pandas`, `matplotlib`.
