#!/bin/zsh
# Собирает jf_bundle.zip для Kaggle-ноутбука сессии 3 (joint fusion, NOTEBOOK_JF.md).
# Внутри — репозиторные пути work/kaggle/..., ячейка 1 ноутбука копирует
# kaggle_seq.py в /kaggle/working. Табличные матрицы едут ОТДЕЛЬНЫМ датасетом
# (work/data/kaggle_session3/kaggle_tabfeats_wed_v1.zip, собирает очередь).
set -e
cd "$(dirname "$0")/../.."
OUT=/tmp/jf_bundle.zip
rm -f "$OUT"
zip -q "$OUT" \
  work/kaggle/kaggle_seq.py \
  work/kaggle/NOTEBOOK_JF.md
echo "готово: $OUT ($(du -h $OUT | cut -f1))"
echo "залить НОВОЙ ВЕРСИЕЙ приватного Kaggle Dataset ozon-code (ячейка 1 ищет jf_bundle.zip)"
