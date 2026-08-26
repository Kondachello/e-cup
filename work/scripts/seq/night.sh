#!/usr/bin/env bash
# Ночь 24->25.08. ТРИ РАЗНЫЕ КОНФИГУРАЦИИ tfm4, а не три одинаковых сида.
# Почему не сиды: корреляция ошибок сид-к-сиду 0.9985, весь предел усреднения
# по сидам (n -> бесконечность) = 1 шумовая единица. Разные конфиги дают
# корреляцию 0.984 и стоят вчетверо больше за тот же час GPU. Числа в NIGHT_RUN.md.
#
# Запуск из корня репозитория на GPU-машине:   bash work/scripts/seq/night.sh
# Лог целиком:                                 bash work/scripts/seq/night.sh 2>&1 | tee runlogs/night.log
set -u
R=work/scripts/seq/run_tfm4.py
C="--tensor-mmap"                       # без него фаза B падала на WDDM при 3.5 ГБ свободной VRAM
A17="--tab-dim 256 --heads 8 --dropout 0.05 --tab-lr-mult 2.0"
A23="--lr 4e-4 --wd 0.025 --aux 0.45 --ema 0.999 --pct-start 0.2"
A777="--heads 12 --tab-gelu --tab-warmup 2000 --dropout 0.10"
say(){ echo; echo "=================== $(date +%H:%M:%S)  $*"; echo; }

# --- ворота. Всё здесь бесплатное и ловит ошибку ДО того, как сгорит ночь ----
say "0/8 самопроверка"
python work/scripts/seq/tfm4_selftest.py || { echo "САМОПРОВЕРКА УПАЛА - на GPU идти незачем"; exit 1; }
say "0/8 тёплый старт всех трёх конфигов (должно быть max|d| = 0.000e+00)"
python $R --stage check --seeds 17  --init-a model_tfm2_s1.pt $C $A17  || { echo "конфиг 17 не встал";  exit 1; }
python $R --stage check --seeds 23  --init-a model_tfm2_s2.pt $C $A23  || { echo "конфиг 23 не встал";  exit 1; }
python $R --stage check --seeds 777 --init-a model_tfm2_s3.pt $C $A777 || { echo "конфиг 777 не встал"; exit 1; }

# --- фаза A. --no-ctrl: контроль tab-off не гоняем, вопрос закрыт зондом сида 1
say "1/8 фаза A, конфиг 17 (таблица сильнее)"
python $R --stage seeds --no-ctrl --minutes 55 --seeds 17  --init-a model_tfm2_s1.pt $C $A17
say "2/8 фаза A, конфиг 23 (другая оптимизация)"
python $R --stage seeds --no-ctrl --minutes 55 --seeds 23  --init-a model_tfm2_s2.pt $C $A23
say "3/8 фаза A, конфиг 777 (другая динамика)"
python $R --stage seeds --no-ctrl --minutes 55 --seeds 777 --init-a model_tfm2_s3.pt $C $A777

# --- фаза B. ФЛАГИ ТЕ ЖЕ САМЫЕ. Иначе фаза B соберёт другую сеть под шаги фазы A
say "4/8 фаза B, конфиг 17"
python $R --stage phaseB --minutes 55 --seeds 17  --init-a model_tfm2_s1.pt --init-b model_tfm2_s1_rt.pt $C $A17
say "5/8 фаза B, конфиг 23"
python $R --stage phaseB --minutes 55 --seeds 23  --init-a model_tfm2_s2.pt --init-b model_tfm2_s2_rt.pt $C $A23
say "6/8 фаза B, конфиг 777"
python $R --stage phaseB --minutes 55 --seeds 777 --init-a model_tfm2_s3.pt --init-b model_tfm2_s3_rt.pt $C $A777

say "7/8 отчёт"
python $R --stage report
say "8/8 сборка посылки"
python $R --stage collect --seeds 17 23 777
say "готово. Пришли _to_kosta/tfm4_s17_s23_s777.zip и runlogs/"
