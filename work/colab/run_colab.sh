#!/usr/bin/env bash
# Полный прогон секвенсной модели на GPU Colab: от загрузки данных до предсказаний.
#
# ПРЕДВАРИТЕЛЬНО (один раз, вручную — нужен браузер):
#     colab new -s ozon --gpu T4
# Если ответит "Precondition Failed" / TooManyAssignments — значит на бесплатном
# тарифе уже занята единственная машина. Закрыть её в браузере: Colab -> Среда
# выполнения -> Управление сеансами -> закрыть сеанс, и повторить команду.
#
# Запуск:  bash work/colab/run_colab.sh big gseq_big_s42 42
#
# ПОЧЕМУ ЗАДАНИЯ УХОДЯТ В ФОН, а не гоняются через colab exec напрямую.
# У colab exec таймаут по умолчанию 30 секунд, а сборка тензоров идёт минуты и
# обучение десятки минут. Даже с большим таймаутом обрыв туннеля убил бы прогон.
# Поэтому exec только СТАРТУЕТ процесс через nohup и сразу возвращается, а мы
# опрашиваем лог. Тогда обрыв связи стоит нам опроса, а не всей работы.
set -euo pipefail

ARM="${1:-big}"
NAME="${2:-gseq_${ARM}_s42}"
SEED="${3:-42}"
SESSION="${COLAB_SESSION:-ozon}"
ROOT="/Users/alexanderkondakov/ozon-cup"
export PATH="$HOME/.local/bin:$PATH"

say() { printf '\n=== %s ===\n' "$1"; }

# Запустить команду на машине в фоне и дождаться маркера завершения в логе.
# $1 — метка (имя лога), $2 — команда, $3 — предельное ожидание в секундах.
run_bg() {
  local tag="$1" cmd="$2" limit="$3" waited=0
  colab exec -s "$SESSION" --timeout 60 >/dev/null <<PYEOF
import subprocess
subprocess.run(
    "nohup bash -c '$cmd; echo ЗАВЕРШЕНО_\$? >> /content/$tag.log' "
    "> /content/$tag.out 2>&1 &", shell=True)
print("запущено")
PYEOF
  echo "[$tag] запущено, жду (предел ${limit}с)"
  while [ "$waited" -lt "$limit" ]; do
    sleep 30
    waited=$((waited + 30))
    local tail_out
    tail_out=$(colab exec -s "$SESSION" --timeout 60 2>/dev/null <<PYEOF || true
print(open("/content/$tag.log").read()[-1500:] if __import__("os").path.exists("/content/$tag.log") else "")
PYEOF
)
    printf '%s' "$tail_out" | tail -3
    if printf '%s' "$tail_out" | grep -q 'ЗАВЕРШЕНО_0'; then
      echo "[$tag] готово за ${waited}с"; return 0
    fi
    if printf '%s' "$tail_out" | grep -q 'ЗАВЕРШЕНО_'; then
      echo "[$tag] УПАЛО, смотри /content/$tag.out"; return 1
    fi
    echo "[$tag] ${waited}с..."
  done
  echo "[$tag] не уложилось в ${limit}с"; return 1
}

say "сессия"
colab status -s "$SESSION"

# Данные заливаются один раз и живут на машине до её остановки. 180 МБ + 4 МБ.
if ! colab ls -s "$SESSION" /content 2>/dev/null | grep -q 'train.parquet'; then
  say "загрузка данных (180 МБ, самая долгая часть)"
  colab upload -s "$SESSION" "$ROOT/train.parquet"      /content/train.parquet
  colab upload -s "$SESSION" "$ROOT/sample_submit.csv"  /content/sample_submit.csv
else
  say "данные уже на машине, загрузка пропущена"
fi

say "код"
colab upload -s "$SESSION" "$ROOT/work/colab/gpu_seq.py" /content/gpu_seq.py

say "зависимости"
colab install -s "$SESSION" polars

say "сборка тензоров (рука $ARM)"
run_bg "build_$ARM" "python /content/gpu_seq.py build --arm $ARM" 2400

say "обучение $NAME"
run_bg "train_$NAME" \
  "python /content/gpu_seq.py train --arm $ARM --name $NAME --seed $SEED --ckpt-dir /content" \
  5400

say "выгрузка предсказаний"
mkdir -p "$ROOT/work/colab/out"
for f in "${NAME}_val.parquet" "${NAME}_test.parquet" "${NAME}.json"; do
  colab download -s "$SESSION" "/content/out/$f" "$ROOT/work/colab/out/$f"
done

say "готово"
echo "предсказания в $ROOT/work/colab/out/"
echo "машина НЕ остановлена — чтобы освободить слот: colab stop -s $SESSION"
