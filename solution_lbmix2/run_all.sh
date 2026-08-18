#!/usr/bin/env bash
# End-to-end reproduction of lbmix2.csv.
set -euo pipefail
cd "$(dirname "$0")"
export OZON_ROOT="${OZON_ROOT:-$(pwd)}"
PY="${PY:-.venv/bin/python}"

$PY scripts/build_features.py --preset all

$PY scripts/train_gbdt.py --name lgblog_final --model lgb --objective log_mse \
  --weight-tau 150 --drop-cols seasonal_index \
  --params '{"num_leaves":255,"min_data_in_leaf":300,"learning_rate":0.05,"feature_fraction":0.75,"n_estimators":5000}'

$PY scripts/train_gbdt.py --name xgblog_final --model xgb --objective log_mse \
  --n-anchors 8 \
  --params '{"max_leaves":511,"min_child_weight":100,"learning_rate":0.05,"colsample_bytree":0.8}'

$PY scripts/blend.py --include lgblog_final,xgblog_final --name blend_w1a --scale-grid
$PY scripts/make_submission.py --pred blend_w1a --out sub_blend_w1a

$OZON_ROOT/lbmix2.csv"
