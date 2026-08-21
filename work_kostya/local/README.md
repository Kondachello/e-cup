# work_kostya/local — адаптация пайплайна kostya46 под эту машину (сиды 3–5)

Механические копии `../scripts/*` для прогона сид-вариантов kostya46 на M1.
Оригиналы не тронуты. Подготовлено 21.08, исполняется через `work/queue`
(задания 986–988). Ничего не пишет в `work/preds` и `work/preds_pack`.

## Дельта против ../scripts (полный список)

1. **Пути**: `/root/work` → `<repo>/work_kostya/work` (через `Path(__file__)`, от cwd не
   зависит); чтение сырья — `<repo>/train.parquet`. Затронуты: cube.py, features.py,
   train_model.py, train_model2.py, train_test_model.py, train_test_m1.py.
2. **Сид через env `KSEED`** (unset → в точности оригинальные значения):
   - train_model.py (конфиг 1, val): `common.seed = KSEED` (ориг. 1);
   - train_test_m1.py (конфиг 1, test): `prm.seed = KSEED` (ориг. 1);
   - train_model2.py / train_test_model.py (конфиг 2, «свои» 2 сида [1, 2]):
     `SEEDS = [1+1000*KSEED, 2+1000*KSEED]` — детерминированный сдвиг, 2-сидовое
     усреднение внутри конфига сохранено. Файлы моделей получают суффиксы
     `_s3001/_s3002` и т.п. — коллизий нет.
   - Вспомогательная голова P(appear) в train_test_model.py остаётся на seed=1
     (она не входит в kostya46, только в kostya46shade); в сид-заданиях
     пропускается через env `KOSTYA_SKIP_APP=1` (default — выполняется, как в
     оригинале). Это единственный добавленный флаг.
3. **prep_data.py — НОВЫЙ файл.** Оригинальная подготовка `/root/work/*` в репо не
   попала; реконструирована по использованию в скриптах и сверена с артефактами
   команды (см. «Проверки» ниже). Создаёт: act.parquet, users_order.parquet,
   buy_mat.npy (bool), gmv_mat.npy, anchor_days.npy (36..379 шаг 7),
   gmv_mat_testgrid.npy, testgrid_days.json (311..374), app_mat_natural.npy,
   app_anchor_days.json (248..283).
4. **assemble_kostya46.py — НОВЫЙ файл.** Сборка смеси по формуле README
   («Состав kostya46»): `0.25·direct₁ + 0.25·(p×size)₁ + 0.2·direct₂ + 0.3·(p×size)₂`
   в лог-пространстве, `pred = expm1(max(mix, 0))`; пишет
   `work_kostya/preds/kostya46_s{KSEED}_{val,test}.parquet` (контракт как у
   kostya46_val.parquet: 250000 строк, user_id i64 сорт., pred f64 сырой GMV).
   Печатает corr(log) с существующим kostya46 (s1) и сырой val RMSLE.

Другой логики не менялось: параметры LightGBM, num_threads=2, число раундов, срезы,
признаки — байт-в-байт как в оригинале.

## Проверки реконструкции prep (выполнены до постановки в очередь)

- train.parquet уже отсортирован по (user_id, event_date) → act.parquet = байтовая
  копия; инвариант interval_feats («day sorted within user») выполняется.
- 250 000 уникальных user_id; сортированный порядок == порядку preds_pack ==
  порядку kostya46_val.parquet.
- Таргет: сумма gmv за [anchor, anchor+30) дней; при anchor=379 совпадает с
  `preds_pack.target` поюзерно (max |Δ| = 7e-12), доля нулей 0.45934 (REPORT: 0.4593).
- buy = (gmv окна > 0): в вал-окне множество совпало с (to_ord > 0) в точности
  (135165 юзеров); сам train_test_model.py использует G>0.
- app_mat (только для вспомогательной головы): «появился» = есть активность в
  [anchor, anchor+30) — реконструированная догадка, на kostya46 не влияет.

## Известные оговорки

- REPORT §6 говорит «125 признаков», README «Состав» — «конфиг 1: 121 признак».
  Текст скриптов: train_model.py (val, конфиг 1) обучается на полном наборе
  features.py (сегодня это 125), train_test_m1.py (test, конфиг 1) срезает `[:121]`.
  Копии воспроизводят текст скриптов как есть (механическая адаптация); если
  оригинальный val-прогон конфига 1 делался до добавления extra_feats (121), у
  s{N} конфиг-1-val будет на 4 признака больше, чем у s1. На пригодность к
  сид-усреднению не влияет (внутрисидовая согласованность та же, что в скриптах).
- Версии venv: lightgbm 4.7.0 и polars 1.43.2 совпадают с requirements.txt Кости;
  numpy 2.2.6 (у него 2.4.4) и sklearn 1.7.2 (1.8.0) — на головы LightGBM не влияют
  (sklearn только в диагностических печатях).
- Файлы моделей конфига 1 (reg_z.txt, clf2.txt, size_z.txt, mt1_*.txt) имеют
  фиксированные имена и перезаписываются следующим сид-прогоном; головы *.npy тоже.
  Поэтому сборка стоит в той же цепочке `&&`, что и обучение. Один сид = одно
  задание очереди, параллельно не запускать.

## Порядок запуска (= задания очереди)

```bash
# 986: prep + куб val        (факт 21.08: ~1 мин; все проверки prep прошли в задании)
.venv/bin/python work_kostya/local/prep_data.py && .venv/bin/python work_kostya/local/cube.py 379 work_kostya/work/cube_val.npy
# 987: куб test              (факт: 7 с)
.venv/bin/python work_kostya/local/cube.py 409 work_kostya/work/cube_test.npy
# 988: сид 3, полная цепочка (оценка ~1–2 ч на FILE.csv, доминирует LightGBM при num_threads=2),
#      env: KSEED=3 KOSTYA_SKIP_APP=1 POLARS_MAX_THREADS=3
.venv/bin/python work_kostya/local/train_model.py && .venv/bin/python work_kostya/local/train_model2.py && \
.venv/bin/python work_kostya/local/train_test_model.py && .venv/bin/python work_kostya/local/train_test_m1.py && \
.venv/bin/python work_kostya/local/assemble_kostya46.py
```

Сиды 4–5 НЕ ставить, пока не замерен вклад s3 (решение по замеру — за №1);
для сида N скопировать 988-е задание с KSEED=N.
Диск: cube_val 0.55 ГБ + cube_test 0.59 ГБ + матрицы/модели ≈ 2 ГБ в
work_kostya/work (каталог добавлен в .gitignore).
