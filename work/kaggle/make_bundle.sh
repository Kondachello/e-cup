#!/bin/zsh
# Собирает tfm3b_bundle.zip для Kaggle-ноутбука (см. NOTEBOOK_TFM3B.md).
# Внутри — структура work/scripts/seq/... как в репо: run_all.py требует запуска
# из корня с таким расположением.
set -e
cd "$(dirname "$0")/../.."
OUT=/tmp/tfm3b_bundle.zip
rm -f "$OUT"
zip -q "$OUT" \
  work/scripts/seq/build_tensor.py \
  work/scripts/seq/make_valid3.py \
  work/scripts/seq/train_tcn.py \
  work/scripts/seq/run_all.py \
  work/scripts/seq/avg_seeds.py \
  work/kaggle/kaggle_seq.py \
  work/kaggle/NOTEBOOK_TFM3B.md
echo "готово: $OUT ($(du -h $OUT | cut -f1))"
echo "залить как приватный Kaggle Dataset (например ozon-code), вторая ячейка ноутбука его распакует"
