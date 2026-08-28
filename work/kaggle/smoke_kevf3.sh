#!/usr/bin/env bash
# kevf3, третий сид s2024 — локально на RTX 2060 (4.9 ГБ свободно).
# Задача скрипта: НЕ обучать, а выяснить два числа — какой батч влезает и с
# какой скоростью идёт. По ним решаем, влезают ли 12000 шагов в ночь.
#
# Запуск из корня репозитория:  bash work/kaggle/smoke_kevf3.sh
set -u
cd "$(dirname "$0")/../.." || exit 1
export KSEQ_ROOT="$PWD/kseq"
export KSEQ_DATA="$PWD"
K=work/kaggle/kaggle_seq.py
TAB=kaggle_tabfeats_wed_v1
ARCH="--d 256 --layers 6 --heads 8 --ff 512 --lmax 320 --time-bias alibi --time2vec 8"
# ^ ровно то, что arch_flags собрал бы из kevf_v2.json; batch оттуда НЕ берётся
say(){ echo; echo "=================== $(date +%H:%M:%S)  $*"; echo; }

say "0/3 проверки"
for f in train.parquet sample_submit.csv kevf_v2.ckpt kevf_v2.json $K $TAB/tabf16_meta.json; do
  [ -e "$f" ] || { echo "НЕТ $f — стоп"; exit 1; }
done
FREE_KB=$(df -k . | awk 'NR==2{print $4}')
echo "свободно на диске: $((FREE_KB/1024/1024)) ГБ (хранилище займёт ~4-5 ГБ)"
[ "$FREE_KB" -lt 7340032 ] && { echo "меньше 7 ГБ — хранилище может не влезть, стоп"; exit 1; }

say "1/3 сборка хранилища (--gap 35, один раз; при повторе пропустится)"
if [ -d "$KSEQ_ROOT/store" ]; then
  echo "$KSEQ_ROOT/store уже есть, пропускаю"
else
  python $K build --gap 35 || { echo "сборка упала"; exit 1; }
fi

say "2/3 лесенка батчей: ищем максимальный, который влезает"
BEST=0
for B in 512 384 256 128 64; do
  echo "--- пробую батч $B ---"
  if python $K train --name smk$B --seed 2024 --device cuda:0 \
       --tab $TAB --tab-mode concat --tab-dropout 0.15 \
       --warm-from kevf_v2.ckpt --lr 2e-4 $ARCH \
       --batch $B --eval-batch $B --max-steps 4 --eval-every 2 \
       > runlogs/smk$B.log 2>&1; then
    echo "батч $B: ВЛЕЗ"; BEST=$B; break
  else
    if grep -qiE "out of memory|CUDA error" runlogs/smk$B.log; then
      echo "батч $B: не хватило видеопамяти, пробую меньше"
    else
      echo "батч $B: упал НЕ по памяти — смотри runlogs/smk$B.log, вот хвост:"
      tail -15 runlogs/smk$B.log; exit 1
    fi
  fi
done
[ "$BEST" = 0 ] && { echo "не влез даже батч 64 — сообщи мне"; exit 1; }

say "3/3 замер скорости на батче $BEST (200 шагов)"
python $K train --name smkspeed --seed 2024 --device cuda:0 \
  --tab $TAB --tab-mode concat --tab-dropout 0.15 \
  --warm-from kevf_v2.ckpt --lr 2e-4 $ARCH \
  --batch $BEST --eval-batch $BEST --max-steps 200 --eval-every 100 \
  > runlogs/smkspeed.log 2>&1
echo "хвост лога:"; tail -25 runlogs/smkspeed.log
echo
echo "=== ИТОГ ==="
echo "максимальный батч: $BEST  (на Kaggle идёт 512)"
grep -iE "шаг|step|s/it|it/s|мин" runlogs/smkspeed.log | tail -5
echo
echo "Пришли этот вывод целиком — посчитаю, влезают ли 12000 шагов в ночь."
