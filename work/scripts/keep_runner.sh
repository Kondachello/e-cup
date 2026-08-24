#!/usr/bin/env bash
# Держит очередь живой: если queue_runner упал, поднимает заново.
# Запуск: nohup work/scripts/keep_runner.sh > /dev/null 2>&1 & disown
# Корень: OZON_ROOT, иначе два уровня вверх от этого скрипта.
cd "${OZON_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
while true; do
  if ! pgrep -f "queue_runner.py" > /dev/null; then
    # выходим, если очередь пуста и стоит флаг остановки
    if [ -f work/queue/STOP ] && [ -z "$(ls work/queue/*.json 2>/dev/null)" ]; then
      echo "[keeper] очередь пуста + STOP, выходим" >> work/reports/queue.log
      exit 0
    fi
    echo "[keeper] $(date +%H:%M:%S) раннер не найден, поднимаю" >> work/reports/queue.log
    nohup nice -n 5 .venv/bin/python work/scripts/queue_runner.py >> work/reports/queue_runner.out 2>&1 &
  fi
  sleep 30
done
