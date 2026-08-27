# work_kostya — задача №3 (пул «не купит»), Костя

Отчёт с выводами: `REPORT_KOSTYA.md`. Здесь — воспроизведение.

## Артефакты для команды

- `preds/kostya46_val.parquet`, `preds/kostya46_test.parquet` — сырые предсказания
  (user_id, pred в GMV), 250 000 строк, порядок user_id как в паке. Дальше стандартно:
  `calibrate.py --pred kostya46 --bins 24`, затем `err_corr.py kostya46_cal`.
- `preds/kostya46shade_test.parquet` — тот же тест, лог-прогноз умножен на
  P(явится|признаки) (обучена на доотборных окнах). На val непроверяемо по построению
  (стена отбора, §4 отчёта) — кандидат на LB-пробу, решает №1.

## Пайплайн (полное воспроизведение с нуля)

```bash
# данные: train.parquet в /root/data (или поправить пути в скриптах)
python3 cube.py 379 cube_val.npy      # недельный куб для вал-сетки якорей
python3 cube.py 409 cube_test.npy     # для тест-сетки
python3 train_clf.py                  # §1: классификатор, пул, AUC в пуле
python3 persistence.py                # скоринг 10 непересекающихся окон
python3 analysis_02_persistence_ceiling.py   # §2-§3: персистентность, потолок, λ-оракул
python3 analysis_03_selection_wall.py        # §4: правило отбора, явка
python3 train_model.py && python3 train_model2.py   # модель kostya46 (val)
python3 train_test_model.py                          # модель kostya46 (test) + P(явка)
python3 analysis_07_margin_eval.py           # §6: запас, NNLS-вклад, плацебо
```

Зазор: обучающие якоря ≤ 2025-12-11 для вал-модели (35 дней до вал-окна) и
≤ 2026-01-09 для тест-модели. Никаких внешних данных. Сиды фиксированы.

Окружение: python3.11, polars, numpy, scipy, scikit-learn, lightgbm (см. версии в
requirements_kostya.txt).

## Главные числа (сверены с preds_pack)

| что | значение |
|---|---|
| пул (нижние 30% p), доля покупателей | 16.4% (у команды 16.5%) |
| AUC в пуле: модель / оракул-λ | 0.669 / 0.731–0.762 |
| прошлые покупки внутри бинов p | AUC 0.4995 (ноль) |
| цена полного λ-оракула | ~0.014 RMSLE |
| kostya46 val (честная калибровка) | 1.669376 (v3: 0.6·v2 + 0.4·лестница) |
| ЗАПАС / NNLS-вклад (плацебо 0) | +0.00127 / +0.000335 |
| явка пула за стеной отбора | 94.2–94.7% (val: 100% принудительно) |
