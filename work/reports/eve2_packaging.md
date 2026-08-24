# eve2: упаковка final_submission + проверка стадии ensemble (21.08, вечер)

Задача: выполнить шаг «Упаковка» из `final_submission/README.md` §7 (веса из
`work/models` -> `final_submission/models`) и проверить стадию `ensemble` после смены
состава 21.08. Стадии features/predict не запускались (тяжёлые). Ничего не коммичено;
все новые файлы в `final_submission/models/` видны в `git status` как untracked.

## 1. Что скопировано (по MEMBER_PARTS/BASES текущего состава из inference.py)

Текущий состав: 14 членов, 18 базовых моделей. Артефакты в `final_submission/models/`
нужны только weights-persist базам (6 шт) и калиброванным членам (12 таблиц);
для persist="preds" баз артефакт — сам `work/preds/NAME_test.parquet`, в models/ им
лежать нечему.

Скопировано **20 файлов**, все сверены sha256 с `work/models` (байт-в-байт):

| группа | файлы | статус |
|---|---|---|
| калибровки 12 членов | kostya46, fusion_v3c_avg, gseq_small_s42, fusion_v3ctl, weak_an_d, weak_ft_recency, behavonly_avg, c_ts2_s42, gseq_big_s42, fusion_f, febspec2, wklin_wk — все `*_cal.npz` | **12/12 найдены и скопированы** |
| бустеры + меты weights-persist | behavonly_s1337{.txt,_meta.json}, behavonly_s7{...}, weak_an_d{...}, weak_ft_recency{...} | **4/6 моделей закрыты** (8 файлов) |
| уже были в пакете | chain_test.npz (шаги 4–5), README.md | на месте, ключи читаются |

**Не нашлось в work/models (2 из 6 weights-persist моделей):**

| модель | вес в бленде | чего нет | почему | закрытие |
|---|---|---|---|---|
| `c_ts2_s42` | 0.0333 | `c_ts2_s42__stage1.txt`, `__stage2.txt`, `_meta.json` | обучена до появления model_io | ретрейн ~5 мин: `USE_V2=1 USE_V3=1 OMP_NUM_THREADS=6 .venv/bin/python work/scripts/train_gbdt.py --name c_ts2_s42 --threads 6 --gap-days 30 --model lgb --objective two_stage --n-anchors 14 --seed 42 --params '{"num_leaves":127,"min_data_in_leaf":500,"n_estimators":5000}' --params2 '{"num_leaves":255,"min_data_in_leaf":100,"n_estimators":5000}'` (>5 мин с учётом фич — через очередь 991+) |
| `behavonly` (сид 42) | 0.0138 | `behavonly.txt`, `behavonly_meta.json` | обучена до появления model_io | ретрейн ~4 мин: `USE_V2=1 USE_V3=1 USE_V4=1 OMP_NUM_THREADS=6 .venv/bin/python work/scripts/train_behavonly.py --name behavonly --n-anchors 14 --threads 6 --seed 42` |

Обе модели сейчас работают через кэш `work/preds` (inference так и делает при
отсутствии меты, с предупреждением) — предикт-стадия на этой машине не сломана,
но на чистой машине им нужен ретрейн. `--stage check` после копирования:
**готово 16/18 баз, суммарный вклад 0.959 из 1.006; сверка `--verify-blend` с
`work/reports/blend_reopt.json` — «совпадает полностью»**; `silence_p_test.npz`
отсутствует (стадия silence обучит на месте, ~10 мин — вне охвата этой проверки).

Не копировались: `fusion_v3c*__p{1,2}_seed*.pt` + stats (появились ночью от
save_torch) — inference.py их не читает: в BASES fusion-модели остаются
persist="preds", мет для них нет. Follow-up ниже.

## 2. Проверка стадии ensemble (шаги 2–3), лёгкая

Механика: кэш стадии predict собран НЕ предиктом, а напрямую из готовых
`work/preds/*_test.parquet` (18 баз, 250000 юзеров) в scratch-каталог, затем запущен
штатный `--stage ensemble` (CACHE_DIR=scratch). Стадия отработала < 1 c: 14 членов,
сумма весов 1.0061, сборка прошла. Эталон сравнения — колонка `blend`
`work/preds_pack/test_preds.parquet` (log1p, float32; закоммичен в pack-new).

Три результата:

**[C] Константы пакета верны.** Бленд из замороженных колонок пака с весами
`BLEND_WEIGHTS` из inference.py: max|Δ| = **3.89e-07** (тест) и 3.83e-07 (вал);
val RMSLE пересборки **1.665647** — эталон воспроизведён точно. Состав/веса/
MEMBER_PARTS после смены 21.08 в пакете корректны.

**[A] На ТЕКУЩИХ work/preds стадия ensemble эталон НЕ воспроизводит:
max|Δ| = 4.64e-01, mean|Δ| = 1.50e-02 (лог-пространство).** Причина известна и
задокументирована (`work/reports/night_repro_inventory.md`): ночные джобы 980–985
перезаписали пять баз (fusion_v3c42/555/7, fusion_v3ctl, wklin) протокольными, но не
побитовыми ретрейнами, и перефитили их калибровки (таблицы от 02:50 отличаются от
замороженных до 0.063 по shifts). 11/14 членов совпадают с паком до float32
(< 5e-07); расходятся ровно fusion_v3c_avg_cal, fusion_v3ctl_cal, wklin. Val-скор
текущей сборки — **1.665770**, цифра в цифру с ночным замером дрейфа (+0.000123 к
эталону 1.665647).

**[E] Замороженный бленд ВОССТАНОВИМ, цель < 1e-6 достигнута.** Базы пяти
перезаписанных моделей взяты из `work/preds/backup_pre_night/`, две fusion-калибровки
перефичены в памяти по backup-валу (calibrate.py детерминирован), остальное — из
`final_submission/models/`. Тестовый бленд против колонки пака:
**max|Δ| = 2.44e-07**, mean|Δ| = 4.7e-08 — в пределах половины ulp float32-хранения
пака (~4.6e-07), т.е. совпадение машинное. Члены-реконструкции: fusion_v3c_avg_cal
2.38e-07, fusion_v3ctl_cal 2.38e-07, wklin 4.68e-07.

Восстановленные замороженные таблицы сохранены (нигде больше на диске их нет):
`work/reports/eve2_frozen_fusion_v3c_avg_cal.npz`,
`work/reports/eve2_frozen_fusion_v3ctl_cal.npz` (ключи centers/shifts, формат
inference-совместимый).

## 3. Развилка винтажей — решение за Сашей

В пакете сейчас смешаны два самосогласованных состояния:

- **Замороженный винтаж** (на нём подобраны BLEND_WEIGHTS, val 1.665647): базы живут
  только в колонках git-пака и в локальном `backup_pre_night/`; его калибровки — в
  `work/reports/eve2_frozen_*.npz` (восстановлены сегодня).
- **Ночной винтаж** (воспроизводим из весов: .pt в work/models, val 1.665770 на тех же
  весах): его базы — текущие work/preds, его калибровки — те, что скопированы в пакет.

Полный predict->ensemble на чистой машине даст ночной винтаж (веса .pt — ночные),
поэтому скопированные ночные калибровки ему консистентны. Но эталонный файл собран на
замороженном. Перед отгрузкой надо выбрать: (а) объявить действующим ночной винтаж и
перемерить/пересобрать цепочку шагов 4–6 на нём, либо (б) заморозить пакет на
пак-колонках (бленд собирается из них с точностью 4e-07 вообще без весов). До выбора —
ничего в пакете не перетирал.

## 4. Хвосты упаковки (рекомендации, не сделано)

1. Очередь 991+: ретрейны `c_ts2_s42` и `behavonly` (команды в §1) — закроют последние
   2 weights-дыры состава.
2. Прошить fusion в inference.py: BASES persist="preds" -> "weights" + мета/загрузчик
   для ночных `__p1/__p2_seed*.pt` (train_fusion3 мет не пишет) — закроет 39.7% веса
   бленда на чистой машине.
3. `final_submission/README.md` §7 «пакет сейчас неполон» после ревью можно смягчить:
   калибровки и 4/6 весов уже в пакете.
4. `silence_p_test.npz` в пакете нет — либо прогнать `--stage silence` один раз
   (~10 мин, через очередь), либо оставить обучение на месте.

## 5. Что где лежит

- пакет: `final_submission/models/` (+20 файлов, untracked, sha256 = work/models)
- scratch-кэш проверки: `/private/tmp/claude-501/-Users-alexanderkondakov-ozon-cup/b994bee8-2354-4524-9a2b-530595cd5feb/scratchpad/preds_test_check/` (вне репо)
- этот отчёт: `work/reports/eve2_packaging.md`; замороженные таблицы: `work/reports/eve2_frozen_*.npz`
- work/preds, submissions/, kaggle_seq.py — не тронуты; в git ничего не коммичено
