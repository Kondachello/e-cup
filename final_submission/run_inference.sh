#!/usr/bin/env bash
# Полный инференс финального решения E-CUP 2026, задача 3.
#   bash final_submission/run_inference.sh                # всё с нуля
#   OZON_ROOT=/path/to/data bash final_submission/run_inference.sh
#   SKIP_FEATURES=1 bash final_submission/run_inference.sh # признаки уже собраны
#
# Требования: python3.10 + окружение из final_submission/requirements.txt;
#   train.parquet и sample_submit.csv в $OZON_ROOT (по умолчанию корень репозитория);
#   обученные артефакты в final_submission/models/ (см. models/README.md).
# Сетевых вызовов нет.
#
# ВРЕМЯ. Стадия признаков ТЯЖЁЛАЯ: помимо табличных наборов она собирает тензоры
# seq3 (3.4 ГБ) — это десятки минут и много диска. Тензоры seq2 не собираются:
# в действующем составе бленда нет ни одного члена, который на них обучается.
# Если признаки уже собраны, ставьте SKIP_FEATURES=1; остальные стадии вместе
# укладываются в 5-15 минут на CPU (плюс ~10 минут, если модель молчания ещё не
# обучена и её нет в models/silence_p_test.npz).
set -euo pipefail


# Интерпретатор: .venv рядом с корнем, иначе системный python3.10/python3
if [[ -x "$OZON_ROOT/.venv/bin/python" ]]; then
  PY="$OZON_ROOT/.venv/bin/python"
else
  PY="$(command -v python3.10 || command -v python3)"
fi

# Потоки. torch и lightgbm тянут разные сборки libomp, и на macOS их совместная
# работа в одном процессе ломается тем скорее, чем больше потоков. Отсюда
# умеренное умолчание 4 и KMP_DUPLICATE_LIB_OK ниже; при OMP: Error #179 —
# понижать до 1. Переопределяемо через окружение.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-4}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

echo "== E-CUP 2026 Задача 3: инференс =="
echo "python:     $PY"
echo "OZON_ROOT:  $OZON_ROOT"
echo "MODELS_DIR: $MODELS_DIR"
echo "OUT:        $OUT"

for f in "$OZON_ROOT/train.parquet" "$OZON_ROOT/sample_submit.csv"; do
  [[ -f "$f" ]] || { echo "ОШИБКА: нет входного файла $f"; exit 1; }
done

# Ранняя проверка: чего не хватает и какой командой оно получается. Печатает
# таблицу и останавливается ДО того, как будут потрачены десятки минут.
#
# ВАЖНО про коды возврата (см. inference.py, stage_check):
#   0 — всё на месте;
#   1 — не хватает ТОЛЬКО того, что соберут стадии features/silence этого же
#       прогона (признаки, тензоры, модель молчания) — это нормальный старт с
#       чистого клона, продолжаем;
#   2 — не хватает того, что прогон произвести не может (веса базовых моделей,
#       таблицы калибровки, замороженная цепочка, входные файлы) — только тогда стоп.
# Раньше здесь стояло `if ! ... check`, то есть выход при ЛЮБОЙ нехватке, включая
# признаки, которые следующая же строка и собрала бы: «инференс одной командой»
# с чистого клона не мог отработать в принципе.
set +e
"$PY" "$HERE/inference.py" --stage check
CHECK_RC=$?
set -e
if [[ "$CHECK_RC" -ge 2 ]]; then
  echo
  echo "ОШИБКА: не хватает артефактов, которые этот прогон собрать не может"
  echo "(таблица выше — там же команды восстановления)."
  echo "Полная последовательность: $HERE/reproduce_training.md"
  exit 1
fi
if [[ "$CHECK_RC" -eq 1 ]]; then
  echo
  echo "Недостающее соберут стадии features/silence этого прогона — продолжаю."
fi

if [[ -z "${SKIP_FEATURES:-}" ]]; then
  time "$PY" "$HERE/inference.py" --stage features
fi
for s in predict ensemble moments silence submission; do
  time "$PY" "$HERE/inference.py" --stage "$s"
done
echo "== Готово: $OUT =="
