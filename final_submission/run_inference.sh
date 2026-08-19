#   bash final_submission/run_inference.sh                # всё с нуля
#   OZON_ROOT=/path/to/data bash final_submission/run_inference.sh
# Требования: python3.10 + окружение из final_submission/requirements.txt;
#   train.parquet и sample_submit.csv в $OZON_ROOT (по умолчанию корень репозитория);
#   обученные артефакты в final_submission/models/ (см. models/README.md).
# Сетевых вызовов нет. Ориентир полного прогона: 20-40 мин CPU (см. README §6).
set -euo pipefail


# Интерпретатор: .venv рядом с корнем, иначе системный python3.10/python3
if [[ -x "$OZON_ROOT/.venv/bin/python" ]]; then
  PY="$OZON_ROOT/.venv/bin/python"
else
  PY="$(command -v python3.10 || command -v python3)"
fi

# Ограничиваем потоки BLAS/OpenMP разумным значением (переопределяемо)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "== E-CUP 2026 Задача 3: инференс =="
echo "python:     $PY"
echo "OZON_ROOT:  $OZON_ROOT"
echo "MODELS_DIR: $MODELS_DIR"
echo "OUT:        $OUT"

for f in "$OZON_ROOT/train.parquet" "$OZON_ROOT/sample_submit.csv"; do
  [[ -f "$f" ]] || { echo "ОШИБКА: нет входного файла $f"; exit 1; }
done

# Ранняя проверка: есть ли все веса. Печатает таблицу «модель -> чего не хватает»
# и останавливается до того, как будут потрачены минуты на построение признаков.
if ! "$PY" "$HERE/inference.py" --stage check; then
  echo
  echo "ОШИБКА: не хватает обученных артефактов (таблица выше)."
  echo "Команды переобучения: $HERE/reproduce_training.md, раздел 2."
  exit 1
fi

time "$PY" "$HERE/inference.py" --stage all
echo "== Готово: $OUT =="
