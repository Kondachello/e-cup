#!/usr/bin/env bash
# Добивка ночи 25.08. Фаза A конфигов 17 и 23 уже готова — они пропустятся сами.
# ПОРЯДОК ИЗМЕНЁН: фазы B готовых конфигов идут ПЕРВЫМИ. В прошлый раз падение
# конфига 777 стояло перед ними и унесло весь остаток ночи.
#
# Запуск (см. NIGHT_RUN.md): нужно ОТДЕЛЬНОЕ консольное окно, иначе закрытие
# терминала шлёт CTRL_CLOSE_EVENT всей группе и убивает прогон вместе с bash.
set -u
cd "$(dirname "$0")/../../.." || exit 1          # корень репозитория, откуда бы ни звали
R=work/scripts/seq/run_tfm4.py
C="--tensor-mmap"
A17="--tab-dim 256 --heads 8 --dropout 0.05 --tab-lr-mult 2.0"
A23="--lr 4e-4 --wd 0.025 --aux 0.45 --ema 0.999 --pct-start 0.2"
A777="--heads 12 --tab-gelu --tab-warmup 2000 --dropout 0.10"
say(){ echo; echo "=================== $(date +%H:%M:%S)  $*"; echo; }

# --- ворота -----------------------------------------------------------------
FREE_KB=$(df -k . | awk 'NR==2{print $4}')
echo "свободно на диске: $((FREE_KB/1024/1024)) ГБ"
if [ "$FREE_KB" -lt 3145728 ]; then
  echo "МЕНЬШЕ 3 ГБ СВОБОДНО — выгрузки и чекпоинты писать некуда, не начинаю."
  echo "Освободи место и запусти снова."; exit 1
fi
say "0/6 самопроверка"
python work/scripts/seq/tfm4_selftest.py || { echo "САМОПРОВЕРКА УПАЛА"; exit 1; }

# --- сначала то, что уже оплачено обучением ---------------------------------
say "1/6 фаза B, конфиг 17"
python $R --stage phaseB --minutes 55 --seeds 17  --init-a model_tfm2_s1.pt --init-b model_tfm2_s1_rt.pt $C $A17
say "2/6 фаза B, конфиг 23"
python $R --stage phaseB --minutes 55 --seeds 23  --init-a model_tfm2_s2.pt --init-b model_tfm2_s2_rt.pt $C $A23

# --- и только потом недоделанный 777 ----------------------------------------
say "3/6 фаза A, конфиг 777"
python $R --stage seeds --no-ctrl --minutes 55 --seeds 777 --init-a model_tfm2_s3.pt $C $A777
say "4/6 фаза B, конфиг 777"
python $R --stage phaseB --minutes 55 --seeds 777 --init-a model_tfm2_s3.pt --init-b model_tfm2_s3_rt.pt $C $A777

say "5/6 отчёт"
python $R --stage report
say "6/6 сборка посылки"
python $R --stage collect --seeds 17 23 777
say "готово. Пришли _to_kosta/tfm4_s17_s23_s777.zip и runlogs/night3.log"
