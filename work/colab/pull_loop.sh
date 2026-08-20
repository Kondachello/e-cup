#!/usr/bin/env bash
# Забирать выгрузки обучения с машины Colab, пока оно идёт.
#
# ЗАЧЕМ. Прогон 19.08 шёл 2ч40м и пропал целиком: всё жило в памяти машины, а машину
# забрали за простой. Теперь обучение каждые --dump-every шагов кладёт предсказания и
# состояние на её диск, а этот цикл переносит их сюда. Потеря машины стоит одного
# промежутка между выгрузками.
#
# Запуск:  nohup bash work/colab/pull_loop.sh gseq_big_s42 > work/colab/pull.log 2>&1 &
set -uo pipefail

NAME="${1:-gseq_big_s42}"
SESSION="${COLAB_SESSION:-ozon}"
EVERY="${PULL_EVERY:-300}"
ROOT="/Users/alexanderkondakov/ozon-cup"
OUT="$ROOT/work/colab/out"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OUT"

last=""
while true; do
  ok=1
  for f in "${NAME}.json" "${NAME}_val.parquet" "${NAME}_test.parquet"; do
    timeout 600 colab download -s "$SESSION" "/content/out/$f" "$OUT/$f.tmp" 2>/dev/null \
      | grep -q Downloaded && mv -f "$OUT/$f.tmp" "$OUT/$f" || ok=0
  done
  if [ "$ok" = 1 ] && [ -f "$OUT/${NAME}.json" ]; then
    line=$(python3 -c "
import json
d=json.load(open('$OUT/${NAME}.json'))
print(f\"шаг {d['step']}/{d['total']} усреднённый {d['cal_rmsle_ckptavg']:.6f} \"
      f\"лучшая точка {d['cal_rmsle_best']:.6f} минут {d['minutes']:.0f} \"
      f\"{'ГОТОВО' if d.get('done') else ''}\")
" 2>/dev/null)
    # печатаем только когда что-то изменилось, иначе лог заплывает
    if [ "$line" != "$last" ]; then
      echo "$(date +%H:%M:%S) $line"
      last="$line"
      # снимок на случай, если поздние выгрузки окажутся хуже ранних
      cp -f "$OUT/${NAME}_val.parquet"  "$OUT/hist_${NAME}_$(date +%H%M)_val.parquet"  2>/dev/null
      cp -f "$OUT/${NAME}_test.parquet" "$OUT/hist_${NAME}_$(date +%H%M)_test.parquet" 2>/dev/null
    fi
    case "$line" in *ГОТОВО*) echo "обучение закончено, цикл выходит"; exit 0;; esac
  else
    echo "$(date +%H:%M:%S) выгрузки ещё нет"
  fi
  sleep "$EVERY"
done
