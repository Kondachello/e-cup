#!/usr/bin/env python3
"""Инференс финального решения E-CUP 2026, задача 3 (прогноз GMV за 30 дней).

Собирает кандидата из train.parquet + sample_submit.csv. Сетевых вызовов нет.

ШЕСТЬ ШАГОВ РЕШЕНИЯ (порядок обязателен, каждый шаг — отдельная стадия ниже):

  1. Обучение членов пула. У нейросетевых — с флагом `--es-metric cal`
     (ранняя остановка по КАЛИБРОВАННОМУ валидационному скору, до 0.0028 на модель).
     Здесь не выполняется: инференс читает сохранённые веса. Команды — в
     reproduce_training.md §3.
  2. Поквантильная калибровка каждого члена (24 бина, `work/scripts/calibrate.py`).
     Таблица сдвигов заморожена в `NAME_cal.npz`, инференс её ПРИМЕНЯЕТ, а не подбирает.
  3. Линейное смешивание калиброванных членов в log1p-пространстве с весами
     BLEND_WEIGHTS (зафиксированы константой в этом файле).
  4. Приведение к целевым моментам log1p: среднее REF_MEAN 2.3247,
     разброс REF_SD 1.6320.
  5. Перенос накопленной цепочки поправок (CHAIN['carry']) и шаг силы STEP от опорного
     файла к новому кандидату (в make_candidate.py это `--carry-from` и `--strength`).
  6. СРЕДНЕСОХРАНЯЮЩАЯ ПОПРАВКА НА МОЛЧАЩИХ. Даёт 0.00084 из 0.00098 дневного прироста.
     Юниверс из 250000 отобран как активный в КАЖДОМ из трёх 30-дневных блоков перед
     тестом, а наше валидационное окно — один из этих блоков, поэтому молчащих на
     валидации нет ПО ПОСТРОЕНИЮ, а в тесте они будут, и верный ответ для них ноль.
     Модель молчания — `work/scripts/silence_model.py`, обучается на якорях, чьё
     целевое окно кончается раньше 2025-11-16.

Стадии (`--stage`):
  check       что есть, чего не хватает и какой командой каждое получается (ничего не считает)
  features    признаки тестового среза + якорей модели молчания, тензоры seq2/seq3
  predict     прогнозы базовых моделей из сохранённых весов -> кэш .npy
  ensemble    усреднение сидов + калибровка + бленд                        (шаги 2-3)
  moments     приведение к моментам + перенос цепочки + шаг силы           (шаги 4-5)
  silence     поправка на молчащих                                         (шаг 6)
  all         всё подряд (по умолчанию)
  freeze      пересобрать models/chain_test.npz из измеренных сабмитов (служебная)

Переменные окружения:
  OZON_ROOT   корень с train.parquet / sample_submit.csv (по умолчанию — родитель этого каталога)
  MODELS_DIR  каталог с весами (по умолчанию final_submission/models; если файла там нет,
              ищем в work/models — туда пишут трейнеры)
  CACHE_DIR   каталог промежуточных прогнозов (по умолчанию MODELS_DIR/preds_test)

Архитектуры, функции предсказания, таблица калибровки и примитивы модели молчания НЕ
дублируются: они импортируются из тех же work/scripts/*.py, которыми модели обучены.
Это единственный способ гарантировать, что инференс и обучение не разъедутся.

Если какого-то артефакта нет — скрипт падает с явным сообщением, какого именно и какой
командой он создаётся, а не выдаёт мусор.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("OZON_ROOT", str(HERE.parent))).resolve()
os.environ["OZON_ROOT"] = str(ROOT)          # common.py читает это при импорте

# torch и lightgbm тянут ДВЕ разные сборки libomp, и на macOS второй импорт падает
# «OMP: Error #179: Function pthread_mutex_init failed». Ловилось на этом скрипте:
# mlpziln (torch) отрабатывал, а следующий за ним бустинг умирал. Ставим до импортов.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SCRIPTS = ROOT / "work" / "scripts"
WORK_MODELS = ROOT / "work" / "models"
WORK_PREDS = ROOT / "work" / "preds"
FEATURES = ROOT / "work" / "features"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(HERE / "models"))).resolve()
CACHE_DIR = Path(os.environ.get("CACHE_DIR", str(MODELS_DIR / "preds_test")))

sys.path.insert(0, str(SCRIPTS))

import numpy as np  # noqa: E402

TEST_ANCHOR_ISO = "2026-02-13"
VAL_ANCHOR_ISO = "2026-01-14"

# ============================================================================ #
#  ШАГ 3. СОСТАВ И ВЕСА БЛЕНДА                                                 #
# ============================================================================ #
# Источник: work/reports/blend_reopt.json, ключ "winner" — библиотека B_plus_cal
# (203 модели), метод ridge_free, alpha_rel 1e-4. val 1.666302, честный OOF по
# 5 фолдам по ПОЛЬЗОВАТЕЛЯМ 1.666419.
#
# ЗАФИКСИРОВАНО ЗДЕСЬ НАМЕРЕННО. blend_reopt.json переписывается при каждом
# перезапуске оптимизатора (в том числе пока идёт разработка), а пакет обязан
# собирать один и тот же файл. Сверить состав с текущим отчётом:
#     python final_submission/inference.py --stage check --verify-blend
BLEND_WEIGHTS = {
    "fusion_v3c_avg_cal":  0.230705,
    "fusion_v3ctl_cal":    0.151006,
    "c_ts2_s7_cal":        0.090592,
    "mlpziln_cal_avg_cal": 0.077215,
    "c_ts2_s42_cal":       0.075589,
    "wklin":               0.067217,
    "behavonly_avg_cal":   0.057488,
    "seq2tr_f_cal":        0.049046,
    "weak_an_d_cal":       0.043814,
    "weak_ft_recency_cal": 0.023857,
    "countaov_s7_cal":     0.023260,
    "wklin_wk":            0.019400,
    "weak_ft_counts_cal":  0.017269,
    "hmmsim_cal":          0.016734,
    "fusion_v3_cal":       0.015661,
    "twl_v7_cal":          0.014492,
    "febspec2_cal":        0.012494,
    "hmmsim":              0.007364,
    "weak_ft_long90_cal":  0.007295,
    "c_xtw_s42":           0.004399,
}

# Член бленда -> (базовые прогнозы, усредняемые в log1p; имя таблицы калибровки).
# Усреднение по сидам — ровно то, что делает work/scripts/avg_log1p.py: среднее
# log1p прогнозов с равными весами. Калибровка (`NAME_cal.npz`, ключи centers/shifts)
# применяется ПОСЛЕ усреднения — так эти члены и собирались.
# cal=None означает, что член входит в бленд сырым (так его выбрал оптимизатор).
MEMBER_PARTS: dict[str, tuple[list[str], str | None]] = {
    "fusion_v3c_avg_cal":  (["fusion_v3c42", "fusion_v3c555", "fusion_v3c7"], "fusion_v3c_avg_cal"),
    "fusion_v3ctl_cal":    (["fusion_v3ctl"], "fusion_v3ctl_cal"),
    "c_ts2_s7_cal":        (["c_ts2_s7"], "c_ts2_s7_cal"),
    "mlpziln_cal_avg_cal": (["mlpziln_c42", "mlpziln_c1337", "mlpziln_c7"], "mlpziln_cal_avg_cal"),
    "c_ts2_s42_cal":       (["c_ts2_s42"], "c_ts2_s42_cal"),
    "wklin":               (["wklin"], None),
    "behavonly_avg_cal":   (["behavonly", "behavonly_s1337", "behavonly_s7"], "behavonly_avg_cal"),
    "seq2tr_f_cal":        (["seq2tr_f"], "seq2tr_f_cal"),
    "weak_an_d_cal":       (["weak_an_d"], "weak_an_d_cal"),
    "weak_ft_recency_cal": (["weak_ft_recency"], "weak_ft_recency_cal"),
    "countaov_s7_cal":     (["countaov_s7"], "countaov_s7_cal"),
    "wklin_wk":            (["wklin_wk"], None),
    "weak_ft_counts_cal":  (["weak_ft_counts"], "weak_ft_counts_cal"),
    "hmmsim_cal":          (["hmmsim"], "hmmsim_cal"),
    "fusion_v3_cal":       (["fusion_v3"], "fusion_v3_cal"),
    "twl_v7_cal":          (["twl_v7"], "twl_v7_cal"),
    "febspec2_cal":        (["febspec2"], "febspec2_cal"),
    "hmmsim":              (["hmmsim"], None),
    "weak_ft_long90_cal":  (["weak_ft_long90"], "weak_ft_long90_cal"),
    "c_xtw_s42":           (["c_xtw_s42"], None),
}

# ---------------------------------------------------------------------------- #
#  Базовые модели: чем обучены, сколько это заняло, где лежат веса              #
# ---------------------------------------------------------------------------- #
# persist:
#   "weights" — трейнер вызывает work/scripts/model_io.py и кладёт веса + NAME_meta.json
#               в work/models/. Инференс ГРУЖАЕТ веса и пересчитывает прогноз.
#   "preds"   — трейнер НЕ сохраняет веса (в нём нет вызовов model_io). Артефактом
#               является сам прогноз work/preds/NAME_test.parquet; воспроизвести можно
#               только повторным запуском трейнера.
#   "stateless" — обучаемых весов нет по построению (генеративный симулятор):
#               воспроизводится повторным запуском с тем же сидом.
# secs — фактическое время из work/queue/done/*.json (Apple M1 Pro, 10 ядер, 16 GB).
BASES: dict[str, dict] = {
    "fusion_v3c42": dict(
        persist="preds", secs=1050, tier="seq3",
        env={"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "OMP_NUM_THREADS": "4", "POLARS_MAX_THREADS": "3"},
        cmd="work/scripts/train_fusion3.py --name fusion_v3c42 --final --epochs 3 --batch 2048 "
            "--eval-batch 1024 --lr 1e-3 --seeds 42 --threads 4 --eval-every 492 --n-ch 12 --es-metric cal"),
    "fusion_v3c555": dict(
        persist="preds", secs=952, tier="seq3",
        env={"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "OMP_NUM_THREADS": "4", "POLARS_MAX_THREADS": "3"},
        cmd="work/scripts/train_fusion3.py --name fusion_v3c555 --final --epochs 3 --batch 2048 "
            "--eval-batch 1024 --lr 1e-3 --seeds 555 --threads 4 --eval-every 492 --n-ch 12 --es-metric cal"),
    "fusion_v3c7": dict(
        persist="preds", secs=918, tier="seq3",
        env={"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "OMP_NUM_THREADS": "4", "POLARS_MAX_THREADS": "3"},
        cmd="work/scripts/train_fusion3.py --name fusion_v3c7 --final --epochs 3 --batch 2048 "
            "--eval-batch 1024 --lr 1e-3 --seeds 7 --threads 4 --eval-every 492 --n-ch 12 --es-metric cal"),
    "fusion_v3ctl": dict(
        persist="preds", secs=875, tier="seq3",
        env={"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "OMP_NUM_THREADS": "4", "POLARS_MAX_THREADS": "3"},
        cmd="work/scripts/train_fusion3.py --name fusion_v3ctl --final --epochs 3 --batch 2048 "
            "--eval-batch 1024 --lr 1e-3 --seeds 42 --threads 4 --eval-every 984 --n-ch 8"),
    "fusion_v3": dict(
        persist="preds", secs=921, tier="seq3",
        env={"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "OMP_NUM_THREADS": "4", "POLARS_MAX_THREADS": "3"},
        cmd="work/scripts/train_fusion3.py --name fusion_v3 --final --epochs 3 --batch 2048 "
            "--eval-batch 1024 --lr 1e-3 --seeds 42 --threads 4 --eval-every 984 --n-ch 12"),
    "c_ts2_s7": dict(
        persist="weights", secs=471,
        env={"USE_V2": "1", "USE_V3": "1", "OMP_NUM_THREADS": "6"},
        cmd="work/scripts/train_gbdt.py --name c_ts2_s7 --threads 6 --gap-days 30 --model lgb "
            "--objective two_stage --n-anchors 14 --seed 7 "
            "--params '{\"num_leaves\":127,\"min_data_in_leaf\":500,\"n_estimators\":5000}' "
            "--params2 '{\"num_leaves\":255,\"min_data_in_leaf\":100,\"n_estimators\":5000}'"),
    "c_ts2_s42": dict(
        persist="weights", secs=303,
        env={"USE_V2": "1", "USE_V3": "1", "OMP_NUM_THREADS": "6"},
        cmd="work/scripts/train_gbdt.py --name c_ts2_s42 --threads 6 --gap-days 30 --model lgb "
            "--objective two_stage --n-anchors 14 --seed 42 "
            "--params '{\"num_leaves\":127,\"min_data_in_leaf\":500,\"n_estimators\":5000}' "
            "--params2 '{\"num_leaves\":255,\"min_data_in_leaf\":100,\"n_estimators\":5000}'"),
    "mlpziln_c42": dict(
        persist="weights", secs=248,
        env={"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "OMP_NUM_THREADS": "4"},
        cmd="work/scripts/train_mlpziln.py --name mlpziln_c42 --n-anchors 14 --gap-days 30 "
            "--seeds 42 --es-metric cal"),
    "mlpziln_c1337": dict(
        persist="weights", secs=188,
        env={"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "OMP_NUM_THREADS": "4"},
        cmd="work/scripts/train_mlpziln.py --name mlpziln_c1337 --n-anchors 14 --gap-days 30 "
            "--seeds 1337 --es-metric cal"),
    "mlpziln_c7": dict(
        persist="weights", secs=112,
        env={"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "OMP_NUM_THREADS": "4"},
        cmd="work/scripts/train_mlpziln.py --name mlpziln_c7 --n-anchors 14 --gap-days 30 "
            "--seeds 7 --es-metric cal"),
    # ОДИН запуск train_wklin.py пишет СРАЗУ три набора: wklin_base, wklin, wklin_wk.
    # Отдельно wklin_wk получить нельзя. Сида нет: гребневая регрессия детерминирована.
    "wklin": dict(
        persist="preds", secs=91, group="wklin",
        env={"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "THREADS": "4", "POLARS_MAX_THREADS": "3"},
        cmd="work/scripts/train_wklin.py --name wklin --emit-tier"),
    "wklin_wk": dict(
        persist="preds", secs=91, group="wklin",
        env={"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "THREADS": "4", "POLARS_MAX_THREADS": "3"},
        cmd="work/scripts/train_wklin.py --name wklin --emit-tier   (тот же запуск, что и wklin)"),
    # у behavonly (сид 42) --n-anchors 14, у двух дополнительных сидов флага НЕТ,
    # поэтому они обучены на всех 27 доступных якорях — так и было, это не описка
    "behavonly": dict(
        persist="weights", secs=214,
        env={"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "OMP_NUM_THREADS": "6"},
        cmd="work/scripts/train_behavonly.py --name behavonly --n-anchors 14 --threads 6 --seed 42"),
    "behavonly_s1337": dict(
        persist="weights", secs=427,
        env={"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "OMP_NUM_THREADS": "4", "POLARS_MAX_THREADS": "3"},
        cmd="work/scripts/train_behavonly.py --name behavonly_s1337 --seed 1337 --threads 4"),
    "behavonly_s7": dict(
        persist="weights", secs=442,
        env={"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "OMP_NUM_THREADS": "4", "POLARS_MAX_THREADS": "3"},
        cmd="work/scripts/train_behavonly.py --name behavonly_s7 --seed 7 --threads 4"),
    "seq2tr_f": dict(
        persist="weights", secs=19485, tier="seq2",
        env={"OMP_NUM_THREADS": "4"},
        cmd="work/scripts/train_seq2.py --name seq2tr_f --arch tr --final --epochs 3 "
            "--batch 2048 --lr 1e-3 --seeds 42,1337 --threads 4"),
    "weak_an_d": dict(
        persist="weights", secs=123,
        env={"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "OMP_NUM_THREADS": "4"},
        cmd="work/scripts/train_weak.py --name weak_an_d --threads 4 --mech anchors --k-anchors 4 "
            "--sel-seed 77 --anchor-pool 0 --model lgb --objective log_mse "
            "--params '{\"objective\":\"tweedie\",\"tweedie_variance_power\":1.45,\"n_estimators\":6000}'"),
    "weak_ft_recency": dict(
        persist="weights", secs=190, approx_secs=True,
        env={"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "OMP_NUM_THREADS": "4"},
        cmd="work/scripts/train_weak.py --name weak_ft_recency --threads 4 --mech ftype "
            "--ftype recency --n-anchors 14 --model lgb --objective log_mse "
            "--params '{\"objective\":\"tweedie\",\"tweedie_variance_power\":1.45,\"n_estimators\":6000}'"),
    "weak_ft_counts": dict(
        persist="weights", secs=190, approx_secs=True,
        env={"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "OMP_NUM_THREADS": "4"},
        cmd="work/scripts/train_weak.py --name weak_ft_counts --threads 4 --mech ftype "
            "--ftype counts --n-anchors 14 --model lgb --objective log_mse "
            "--params '{\"objective\":\"tweedie\",\"tweedie_variance_power\":1.45,\"n_estimators\":6000}'"),
    "weak_ft_long90": dict(
        persist="weights", secs=190, approx_secs=True,
        env={"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "OMP_NUM_THREADS": "4"},
        cmd="work/scripts/train_weak.py --name weak_ft_long90 --threads 4 --mech ftype "
            "--ftype long90 --n-anchors 14 --model lgb --objective log_mse "
            "--params '{\"objective\":\"tweedie\",\"tweedie_variance_power\":1.45,\"n_estimators\":6000}'"),
    "countaov_s7": dict(
        persist="weights", secs=489,
        env={"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "OMP_NUM_THREADS": "4", "POLARS_MAX_THREADS": "3"},
        cmd="work/scripts/train_countaov.py --name countaov_s7 --threads 4 --n-anchors 14 "
            "--gap-days 30 --seed 7"),
    "hmmsim": dict(
        persist="stateless", secs=366,
        env={"THREADS": "6"},
        cmd="work/scripts/train_hmm_sim.py --name hmmsim --states 4 --sims 500 --win 120 "
            "--em-cap 25000 --splits val,test"),
    "twl_v7": dict(
        persist="weights", secs=187, tier="v7",
        env={"USE_V2": "1", "USE_V3": "1", "USE_V4": "1", "USE_V7": "1",
             "OMP_NUM_THREADS": "6", "POLARS_MAX_THREADS": "3"},
        cmd="work/scripts/train_gbdt.py --name twl_v7 --threads 6 --gap-days 30 --model lgb "
            "--objective log_mse --n-anchors 8 --seed 42 "
            "--params '{\"objective\":\"tweedie\",\"tweedie_variance_power\":1.45,\"n_estimators\":6000}'"),
    # febspec2 не использует табличные тиры вовсе: у него свой короткоисторический
    # набор (build_features_short.py), который он при необходимости пересобирает сам
    "febspec2": dict(
        persist="preds", secs=429,
        env={"OMP_NUM_THREADS": "3", "POLARS_MAX_THREADS": "3", "THREADS": "3"},
        cmd="work/scripts/train_febspec2.py --name febspec2 --config auto --cohort 0.20 --threads 3"),
    # c_xtw_s42 обучен train_gbdt.py --model xgb, БЕЗ USE_V4 (194 признака), как и c_ts2_*
    "c_xtw_s42": dict(
        persist="weights", secs=246,
        env={"USE_V2": "1", "USE_V3": "1", "OMP_NUM_THREADS": "6"},
        cmd="work/scripts/train_gbdt.py --name c_xtw_s42 --threads 6 --gap-days 30 --model xgb "
            "--objective log_mse --n-anchors 10 --seed 42 "
            "--params '{\"objective\":\"reg:tweedie\",\"tweedie_variance_power\":1.2,\"max_leaves\":511,"
            "\"min_child_weight\":100,\"learning_rate\":0.05,\"colsample_bytree\":0.8,\"n_estimators\":6000}'"),
}

# ============================================================================ #
#  ШАГИ 4-5. МОМЕНТЫ И НАКОПЛЕННАЯ ЦЕПОЧКА                                     #
# ============================================================================ #
# Моменты log1p прогноза, ЗАМЕРЕННЫЕ НА ЛИДЕРБОРДЕ (KNOWLEDGE.md, факт Ф18):
# пара сабмитов, отличающихся на известную константу в log-пространстве, даёт
# среднее из разности квадратов скоров. Это свойство ТЕСТОВОГО ОКНА, а не бленда,
# поэтому при улучшении бленда числа не переподбираются, а новый бленд приводится
REF_MEAN = 2.324718457996938
REF_SD = 1.632001151855992
# Сила шага от опорного файла к новому кандидату (make_candidate.py --strength).
STEP = 0.469

# ============================================================================ #
#  ШАГ 6. ПОПРАВКА НА МОЛЧАЩИХ                                                 #
# ============================================================================ #
# Направление среднесохраняющее:  d = -(p*m - mean(p*m)),  m = log1p(прогноза).
# Работает в нём только РАЗБРОС p между людьми, поэтому общий уровень p ничем
# локальным не определён — он лишь перепараметризует силу. Отсюда протокол:
# оба направления (старое табличное и новое модельное) приводятся к ОДНОМУ размеру
# q = Q_REF, и тогда коэффициенты означают одно и то же физическое количество.
Q_REF = 0.0027148975861072634        # work/reports/silence_final.json, "q_old"
# Разложение отправленного файла (KNOWLEDGE.md):

# новой части (её новизна 0.902 — наивысшая за проект, но замера у неё ещё нет).
A_OLD = 0.894
A_NEW = 0.65

# Замороженные векторы измеренной цепочки: models/chain_test.npz (см. --stage freeze).
CHAIN_NPZ = "chain_test.npz"

# Контрольные значения итогового файла (250000 строк).
EXPECT_MEAN = 2.3247
EXPECT_SD = 1.6470

# ---------------------------------------------------------------------------- #
FEATURE_TIERS = {
    "base": "anchor={a}.parquet",
    "v2":   "anchor={a}.extra.parquet",
    "v3":   "anchor={a}.v3.parquet",
    "v4":   "anchor={a}.v4.parquet",
    "v7":   "anchor={a}.v7.parquet",
}
# Якоря, на которых обучается модель молчания (см. silence_model.EVAL_TRAIN):
# пять обучающих + один для калибровки наклона Платта. Все «чистые»: их 30-дневное
# целевое окно кончается раньше 2025-11-16, начала блоков отбора юниверса.
SILENCE_ANCHORS = ["2025-07-02", "2025-07-16", "2025-07-30",
                   "2025-08-13", "2025-08-27", "2025-09-10"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class MissingArtifact(RuntimeError):
    """Нет файла, без которого прогноз был бы мусором. Падаем громко."""


def howto(base: str) -> str:
    b = BASES.get(base)
    if not b:
        return ""
    env = " ".join(f"{k}={v}" for k, v in b["env"].items())
    return f"{env} .venv/bin/python {b['cmd']}"


def find(fname: str, what: str, base: str | None = None) -> Path:
    """Ищет артефакт в MODELS_DIR, затем в work/models. Иначе — внятная ошибка."""
    for d in (MODELS_DIR, WORK_MODELS):
        p = d / fname
        if p.exists():
            return p
    hint = f"\n  создаётся: {howto(base)}" if base in BASES else ""
    raise MissingArtifact(
        f"НЕ ХВАТАЕТ ФАЙЛА: {fname}  ({what})\n"
        f"  искали в: {MODELS_DIR}\n"
        f"            {WORK_MODELS}{hint}\n"
        f"  без него прогноз посчитать нельзя — прекращаю, чтобы не выдать мусор.")


def have(fname: str) -> bool:
    return any((d / fname).exists() for d in (MODELS_DIR, WORK_MODELS))


def load_meta(base: str) -> dict:
    return json.loads(find(f"{base}_meta.json",
                           f"конфиг модели {base} (порядок признаков, архитектура)",
                           base).read_text())


def base_state(base: str) -> tuple[str, str]:
    """(состояние, пояснение) для одной базовой модели."""
    spec = BASES[base]
    cached = (WORK_PREDS / f"{base}_test.parquet").exists()
    if spec["persist"] == "stateless":
        return ("готово" if have(f"{base}_meta.json") or cached else "пересчёт",
                "весов нет по построению, пересчитывается за "
                f"{spec['secs'] // 60} мин" if not cached else "прогноз в work/preds")
    if spec["persist"] == "preds":
        return ("готово" if cached else "НЕТ",
                "прогноз-артефакт (трейнер не сохраняет веса)")
    if not have(f"{base}_meta.json"):
        return ("кэш" if cached else "НЕТ", "нет meta" + (", есть кэш прогноза" if cached else ""))
    need = load_meta(base).get("weights") or []
    got = [f for f in need if have(f)]
    if len(got) == len(need):
        return "готово", f"{len(got)}/{len(need)} файлов весов"
    return ("кэш" if cached else "НЕТ",
            f"весов {len(got)}/{len(need)}" + (", есть кэш прогноза" if cached else ""))


def run_script(script: str, *args: str, env: dict | None = None) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    e = dict(os.environ)
    e.update(env or {})
    log(f"$ {' '.join(cmd[1:])}" + (f"   env={env}" if env else ""))
    subprocess.run(cmd, check=True, env=e)


# ============================== СТАДИЯ: ПРИЗНАКИ ============================= #

def missing_features() -> list[tuple[str, str, str]]:
    """[(что, путь-образец, команда)] — чего не хватает из признаков и тензоров."""
    out = []
    anchors = [TEST_ANCHOR_ISO, VAL_ANCHOR_ISO] + SILENCE_ANCHORS
    for tier, pat in FEATURE_TIERS.items():
        need = [TEST_ANCHOR_ISO] if tier == "v7" else anchors
        miss = [a for a in need if not (FEATURES / pat.format(a=a)).exists()]
        if not miss:
            continue
        script = {"base": "build_features.py", "v2": "build_features_v2.py",
                  "v3": "build_features_v3.py", "v4": "build_features_v4.py",
                  "v7": "build_features_v7.py"}[tier]
        arg = "" if tier == "v3" else f" --anchors {','.join(miss)}"
        out.append((f"признаки {tier} ({len(miss)} якорей)",
                    str(FEATURES / pat.format(a=miss[0])),
                    f".venv/bin/python work/scripts/{script}{arg}"))
    seq3 = ROOT / "work" / "seq3" / f"anchor={TEST_ANCHOR_ISO}.npy"
    if not seq3.exists():
        out.append(("тензоры seq3 (fusion_v3*, 3.4 ГБ)", str(seq3),
                    "POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/build_seq3.py --max-train 8"))
    seq2 = ROOT / "work" / "seq2" / f"anchor={TEST_ANCHOR_ISO}.npy"
    if not seq2.exists():
        out.append(("тензоры seq2 (seq2tr_f, ~11 ГБ, удалены)", str(seq2),
                    "POLARS_MAX_THREADS=3 .venv/bin/python work/scripts/build_seq2.py"))
    return out


def stage_features() -> None:
    for f in ("train.parquet", "sample_submit.csv"):
        if not (ROOT / f).exists():
            raise MissingArtifact(f"НЕ ХВАТАЕТ ВХОДНОГО ФАЙЛА: {ROOT / f}")
    anchors = ",".join([TEST_ANCHOR_ISO, VAL_ANCHOR_ISO] + SILENCE_ANCHORS)
    run_script("build_features.py", "--anchors", anchors)
    run_script("build_features_v2.py", "--anchors", anchors)
    run_script("build_features_v3.py")          # сам пропускает уже собранные срезы
    run_script("build_features_v4.py", "--anchors", anchors)
    run_script("build_features_v7.py", "--anchors", TEST_ANCHOR_ISO,
               "--states", "4", "--sims", "300", "--win", "120",
               "--em-cap", "15000", "--seed", "42")
    run_script("build_seq3.py", "--max-train", "8", env={"POLARS_MAX_THREADS": "3"})
    if not (ROOT / "work" / "seq2" / f"anchor={TEST_ANCHOR_ISO}.npy").exists():
        run_script("build_seq2.py", env={"POLARS_MAX_THREADS": "3"})
    log("признаки и тензоры готовы")


# =========================== СТАДИЯ: ПРЕДСКАЗАНИЯ ============================ #

def load_test_matrix(base: str, meta: dict):
    """(X, user_id) для модели: колонки строго в том порядке, в каком обучали."""
    import polars as pl
    for k in ("USE_V2", "USE_V3", "USE_V4", "USE_V5", "USE_V6", "USE_V7", "USE_V8",
              "USE_V10", "USE_SEQOOF"):
        os.environ.pop(k, None)
    os.environ.update(meta.get("feature_flags") or BASES[base]["env"])
    from common import TEST_ANCHOR, load_anchor
    df = load_anchor(TEST_ANCHOR)
    cols = meta["feature_cols"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise MissingArtifact(
            f"модель {base}: в тестовом срезе нет {len(missing)} признаков, "
            f"например {missing[:5]}.\n"
            f"  вероятно не собран нужный набор (флаги {meta.get('feature_flags')}) — "
            f"см. reproduce_training.md §2")
    X = df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    uid = df["user_id"].to_numpy()
    del df
    return np.ascontiguousarray(X), uid


def _torch_device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _apply_stats(mod, X: np.ndarray, stats_file: Path) -> None:
    z = np.load(stats_file)
    mod.apply_stats(X, {k: z[k] for k in ("med", "lo", "hi", "mean", "std")})


def predict_mlp_family(base: str, meta: dict, X: np.ndarray) -> np.ndarray:
    """mlpziln / mlp2: усреднение сидов в СЫРОЙ шкале (как в трейнере)."""
    import torch
    mod = __import__("train_mlpziln" if meta["kind"] == "mlpziln" else "train_mlp2")
    _apply_stats(mod, X, find(meta["stats_npz"], f"статистики препроцессинга {base}", base))
    dev, cfg = _torch_device(), meta["cfg"]
    preds = []
    for seed in meta["seeds"]:
        w = find(f"{base}_seed{seed}.pt", f"веса {base}, сид {seed}", base)
        net = mod.build_model(X.shape[1], cfg["hidden"], cfg["dropout"]).to(dev)
        net.load_state_dict(torch.load(w, map_location=dev))
        preds.append(np.expm1(np.clip(mod.predict_log(net, X, dev), 0, None)))
        del net
    return np.mean(preds, axis=0)


def predict_seq2(base: str, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import train_seq2 as mod
    from common import TEST_ANCHOR, user_universe
    uids = user_universe()["user_id"].to_numpy()
    x_mm = mod.open_x(TEST_ANCHOR)
    dev, preds = _torch_device(), []
    for seed in meta["seeds"]:
        w = find(f"{base}_seed{seed}.pt", f"веса {base}, сид {seed}", base)
        net = mod.build_model(meta["arch"], dev)
        net.load_state_dict(torch.load(w, map_location=dev))
        _, lp = mod.predict_main(net, x_mm, dev, 4096, 1)
        preds.append(np.expm1(np.clip(lp, 0, None)).astype(np.float64))
        del net
    return np.mean(preds, axis=0), uids


def _lgb(fname: str, what: str, base: str):
    import lightgbm as lgb
    return lgb.Booster(model_file=str(find(fname, what, base)))


def predict_gbdt(base: str, meta: dict, X: np.ndarray) -> np.ndarray:
    """train_gbdt: two_stage = expm1(p*mu), иначе expm1(raw + m_hat) / clip."""
    kind = meta.get("model", "lgb")
    if meta["objective"] == "two_stage":
        p = _lgb(f"{base}__stage1.txt", f"{base}: бустер P(y>0)", base).predict(X)
        mu = _lgb(f"{base}__stage2.txt", f"{base}: бустер E[log1p|y>0]", base).predict(X)
        return np.expm1(np.clip(p * np.clip(mu, 0, None), 0, None))
    if kind == "xgb":
        import xgboost as xgb
        b = xgb.Booster()
        b.load_model(str(find(f"{base}.xgb.json", f"бустер XGBoost {base}", base)))
        raw = b.predict(xgb.DMatrix(X))
    else:
        raw = _lgb(f"{base}.txt", f"бустер LightGBM {base}", base).predict(X)
    if meta["objective"] == "log_mse":
        return np.expm1(np.clip(raw + float(meta.get("m_hat_test", 0.0)), 0, None))
    return np.clip(raw, 0, None)


def predict_countaov(base: str, meta: dict, X: np.ndarray) -> np.ndarray:
    from train_countaov import COMBINE
    pc = _lgb(f"{base}__count.txt", f"{base}: голова числа заказов", base).predict(X)
    pa = _lgb(f"{base}__aov.txt", f"{base}: голова среднего чека", base).predict(X)
    return COMBINE[meta["mode"]](pc, pa, meta["aov_damp"])


def predict_hmmsim(base: str) -> tuple[np.ndarray, np.ndarray]:
    """Весов нет по построению: пересчитываем симулятор с теми же гиперпараметрами."""
    import polars as pl
    from common import PREDS_DIR
    out = PREDS_DIR / f"{base}_test.parquet"
    if not out.exists():
        log(f"{base}: сохранённых весов нет по построению (генеративный симулятор) — "
            f"пересчитываю, ~{BASES[base]['secs'] // 60} мин")
        run_script("train_hmm_sim.py", "--name", base, "--states", "4", "--sims", "500",
                   "--win", "120", "--em-cap", "25000", "--splits", "test",
                   env={"THREADS": os.environ.get("OMP_NUM_THREADS", "6")})
    d = pl.read_parquet(out).sort("user_id")
    return d["pred"].to_numpy(), d["user_id"].to_numpy()


def cached_pred(base: str, why: str) -> tuple[np.ndarray, np.ndarray]:
    """Прогноз-артефакт из work/preds (для трейнеров без сохранения весов)."""
    import polars as pl
    p = WORK_PREDS / f"{base}_test.parquet"
    if not p.exists():
        raise MissingArtifact(
            f"НЕ ХВАТАЕТ ПРОГНОЗА: {p}  ({why})\n"
            f"  создаётся: {howto(base)}\n"
            f"  ~{BASES[base]['secs'] // 60} мин; трейнер этой модели не сохраняет веса "
            f"(в нём нет вызовов work/scripts/model_io.py), поэтому её артефакт — сам прогноз.")
    d = pl.read_parquet(p).sort("user_id")
    return d["pred"].to_numpy().astype(np.float64), d["user_id"].to_numpy()


def predict_base(base: str) -> tuple[np.ndarray, np.ndarray]:
    spec = BASES[base]
    if spec["persist"] == "stateless":
        return predict_hmmsim(base)
    if spec["persist"] == "preds":
        return cached_pred(base, "трейнер не сохраняет веса")
    if not have(f"{base}_meta.json"):
        log(f"ВНИМАНИЕ: {base}: нет {base}_meta.json — беру кэш прогноза work/preds")
        return cached_pred(base, "нет сохранённых весов")
    meta = load_meta(base)
    kind = meta["kind"]
    if kind == "seq2":
        return predict_seq2(base, meta)
    X, uid = load_test_matrix(base, meta)
    if kind in ("mlpziln", "mlp2"):
        return predict_mlp_family(base, meta, X), uid
    if kind == "countaov":
        return predict_countaov(base, meta, X), uid
    if kind == "gbdt":
        return predict_gbdt(base, meta, X), uid
    raise MissingArtifact(f"неизвестный тип модели {kind!r} в {base}_meta.json")


def needed_bases() -> list[str]:
    seen, out = set(), []
    for member in BLEND_WEIGHTS:
        for b in MEMBER_PARTS[member][0]:
            if b not in seen:
                seen.add(b)
                out.append(b)
    return out


def stage_predict() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    uid_ref = None
    ref_path = CACHE_DIR / "user_ids.npy"
    if ref_path.exists():
        uid_ref = np.load(ref_path)
    for base in needed_bases():
        dst = CACHE_DIR / f"{base}.npy"
        if dst.exists():
            log(f"{base}: прогноз уже в кэше, пропускаю")
            continue
        t0 = time.time()
        pred, uid = predict_base(base)
        order = np.argsort(uid)
        uid, pred = uid[order], np.asarray(pred, dtype=np.float64)[order]
        if uid_ref is None:
            uid_ref = uid
            np.save(ref_path, uid)
        elif not np.array_equal(uid, uid_ref):
            raise MissingArtifact(f"{base}: набор user_id не совпал с остальными")
        np.save(dst, np.clip(pred, 0, None))
        log(f"{base}: готово за {time.time() - t0:.0f}s, "
            f"mean_log1p={np.log1p(np.clip(pred, 0, None)).mean():.4f}")


# ====================== СТАДИЯ: КАЛИБРОВКА + БЛЕНД (2-3) ===================== #

def stage_ensemble() -> np.ndarray:
    from calibrate import apply_shifts          # ровно та функция, что писала таблицу
    uid_path = CACHE_DIR / "user_ids.npy"
    if not uid_path.exists():
        raise MissingArtifact(f"нет {uid_path} — сначала выполните стадию predict")
    n = len(np.load(uid_path))
    lp_blend = np.zeros(n, dtype=np.float64)
    total_w = 0.0
    for member, w in sorted(BLEND_WEIGHTS.items(), key=lambda kv: -kv[1]):
        parts, cal = MEMBER_PARTS[member]
        acc = np.zeros(n, dtype=np.float64)
        for b in parts:
            f = CACHE_DIR / f"{b}.npy"
            if not f.exists():
                raise MissingArtifact(f"нет прогноза {b} — сначала стадия predict")
            acc += np.log1p(np.clip(np.load(f), 0, None)) / len(parts)
        raw_mean = float(acc.mean())
        if cal:
            z = np.load(find(f"{cal}.npz", f"таблица калибровки члена {member}", parts[0]))
            acc = apply_shifts(acc, z["centers"], z["shifts"])
        lp_blend += w * acc
        total_w += w
        log(f"{member:<20} w={w:.4f}  mean_log1p {raw_mean:.4f} -> {acc.mean():.4f}"
            + (f"  ({len(parts)} сида усреднены)" if len(parts) > 1 else ""))
    log(f"бленд: {len(BLEND_WEIGHTS)} членов, сумма весов {total_w:.4f}, "
        f"mean_log1p {lp_blend.mean():.4f}, sd {lp_blend.std():.4f}")
    np.save(CACHE_DIR / "lp_blend.npy", lp_blend)
    return lp_blend


# ==================== СТАДИЯ: МОМЕНТЫ + ЦЕПОЧКА (шаги 4-5) =================== #

def load_chain() -> dict:
    p = None
    for d in (MODELS_DIR, WORK_MODELS):
        if (d / CHAIN_NPZ).exists():
            p = d / CHAIN_NPZ
            break
    if p is None:
        raise MissingArtifact(
            f"НЕ ХВАТАЕТ ФАЙЛА: {CHAIN_NPZ} (замороженная цепочка поправок)\n"
            f"  искали в: {MODELS_DIR}\n            {WORK_MODELS}\n"
            f"  создаётся: .venv/bin/python final_submission/inference.py --stage freeze\n"
            f"work/preds/blend_cal_test.parquet)")
    z = np.load(p)
    return {k: z[k] for k in z.files}


def stage_moments(lp_blend: np.ndarray | None = None) -> np.ndarray:
    """Шаг 4: привести к моментам REF_MEAN/REF_SD. Шаг 5: цепочка + сила шага."""
    if lp_blend is None:
        lp_blend = np.load(CACHE_DIR / "lp_blend.npy")
    ch = load_chain()
    ref, carry = ch["ref_lp"], ch["carry_lp"]
    if len(ref) != len(lp_blend):
        raise MissingArtifact("chain_test.npz собран на другом наборе пользователей")

    b = REF_SD / lp_blend.std()
    matched = (REF_MEAN - b * lp_blend.mean()) + b * lp_blend
    log(f"шаг 4 (моменты): {REF_MEAN - b * lp_blend.mean():+.4f} + {b:.4f} * lp  ->  "
        f"среднее {matched.mean():.4f} разброс {matched.std():.4f}")

    full = matched + carry
    log(f"шаг 5a (цепочка): разброс переносимого остатка {carry.std():.4f}, "
        f"после переноса среднее {full.mean():.4f} разброс {full.std():.4f}")

    base = ref + STEP * (full - ref)
    log(f"шаг 5b (сила {STEP}): направление к опоре ужато с разброса "
        f"{(full - ref).std():.4f} до {(base - ref).std():.4f}; "
        f"среднее {base.mean():.4f} разброс {base.std():.4f}")
    np.save(CACHE_DIR / "lp_base.npy", base)
    return base


# ======================= СТАДИЯ: МОЛЧАЩИЕ (шаг 6) ============================ #

def silence_p_test() -> np.ndarray:
    """Вероятность нуля событий в тестовом окне для всех 250000.

    Кэш: models/silence_p_test.npz. Иначе — обучение по протоколу
    work/scripts/silence_model.py (примитивы импортируются оттуда, ничего не
    дублируется): пять чистых якорей на обучение, шестой на калибровку наклона
    Платта, признаки — внутриякорные ранги, отсев проводников артефакта отбора,
    смесь логрегрессии и бустинга 50/50.
    """
    import polars as pl
    for d in (MODELS_DIR, WORK_MODELS):
        p = d / "silence_p_test.npz"
        if p.exists():
            z = np.load(p)
            log(f"молчание: беру готовое p из {p} (среднее {z['p'].mean():.5f})")
            return z["p"]

    log("молчание: готового p нет — обучаю модель (5 якорей, ~10 мин)")
    os.environ.update(USE_V2="1", USE_V3="1", USE_V4="1")
    import silence_model as SM
    from common import feature_cols, load_anchor

    C = SM.build_cumsum()
    fit_anchors, cal_anchor = SM.EVAL_TRAIN[:-1], SM.EVAL_TRAIN[-1]
    cols = feature_cols(load_anchor(fit_anchors[0]))
    Xtr, ytr, atr = SM.load_block(fit_anchors, cols, C, "raw")
    keep, _drop, rdrift = SM.pick_cols(Xtr, atr, cols)
    keep = np.array([j for j in keep if rdrift[j] <= 0.30])
    log(f"молчание: признаков в модели {len(keep)} из {len(cols)}")
    for k in range(atr.max() + 1):
        sub = np.ascontiguousarray(Xtr[atr == k])
        SM.rank_inplace(sub)
        Xtr[atr == k] = sub
    A = np.ascontiguousarray(Xtr[:, keep])
    del Xtr

    Xca, yca, _ = SM.load_block([cal_anchor], cols, C, "rank")
    Bc = np.ascontiguousarray(Xca[:, keep])
    del Xca

    # все 250000 удовлетворяют условию отбора по построению — проверяем, а не верим
    assert SM.sel_mask(C, SM.TEST_ANCHOR).all(), \
        "на тестовом якоре население НЕ совпадает с условием отбора"
    dte = load_anchor(SM.TEST_ANCHOR)
    uid_te = dte["user_id"].to_numpy()
    Bt = np.ascontiguousarray(dte.select(cols).to_numpy().astype(np.float32)[:, keep])
    del dte
    SM.rank_inplace(Bt)

    threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    parts = []
    for name in ("логрегрессия", "бустинг"):
        mdl = (SM.fit_gbm(A, ytr, atr, atr.max() + 1, threads=threads) if name == "бустинг"
               else SM.fit_lr(A, ytr, atr, atr.max() + 1))
        a_, b_ = SM.platt(SM.score(mdl, Bc), yca)
        p = 1.0 / (1.0 + np.exp(-(a_ * SM.score(mdl, Bt) + b_)))
        log(f"молчание: {name}: наклон Платта {a_:.4f}, сдвиг {b_:.4f}, "
            f"среднее p {p.mean():.5f}")
        parts.append(p)
    p_te = 0.5 * parts[0] + 0.5 * parts[1]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(MODELS_DIR / "silence_p_test.npz", user_id=uid_te, p=p_te)
    log(f"молчание: записан {MODELS_DIR / 'silence_p_test.npz'}")
    return p_te


def mean_preserving_dir(p: np.ndarray, m: np.ndarray) -> np.ndarray:
    """d = -(p*m - mean(p*m)), приведённое к общему размеру q = Q_REF."""
    u = p * m
    d = -(u - float(u.mean()))
    q = float((d ** 2).mean())
    return d * float(np.sqrt(Q_REF / max(q, 1e-30)))


def stage_silence(base: np.ndarray | None = None) -> np.ndarray:
    if base is None:
        base = np.load(CACHE_DIR / "lp_base.npy")
    ch = load_chain()
    mdl_tektit = ch["dir_old"]

    import silence_model as SM

    # ортогональная к старому направлению часть нового

    lp = base + A_OLD * mdl_tektit + A_NEW * e
    log(f"шаг 6: применено {A_OLD}*старое + {A_NEW}*новизна; "
        f"среднее {base.mean():.4f} -> {lp.mean():.4f} (сохраняется), "
        f"разброс {base.std():.4f} -> {lp.std():.4f}")
    for what, got, exp in (("среднее", lp.mean(), EXPECT_MEAN), ("разброс", lp.std(), EXPECT_SD)):
        if abs(got - exp) > 0.02:
            print(f"ВНИМАНИЕ: {what} итогового прогноза {got:.4f}, ожидалось ~{exp:.4f} — "
                  f"проверьте состав бленда и chain_test.npz", file=sys.stderr)
    np.save(CACHE_DIR / "final_lp.npy", lp)
    return lp


# ========================== СТАДИЯ: SUBMISSION =============================== #

def stage_submission(lp: np.ndarray | None = None) -> None:
    import polars as pl
    from common import SAMPLE_SUBMIT
    if lp is None:
        lp = np.load(CACHE_DIR / "final_lp.npy")
    uid = np.load(CACHE_DIR / "user_ids.npy")
    vals = np.expm1(np.clip(lp, 0, None))
    sample = pl.read_csv(SAMPLE_SUBMIT, schema_overrides={"user_id": pl.Int64})
    out = (sample.select("user_id")
           .join(pl.DataFrame({"user_id": uid.astype(np.int64), "predict": vals}),
                 on="user_id", how="left"))
    assert out.height == sample.height, "число строк не совпало с sample_submit"
    assert out["predict"].null_count() == 0, "есть user_id без прогноза"
    assert float(out["predict"].min()) >= 0.0, "есть отрицательные прогнозы"


# ============================ СТАДИЯ: FREEZE ================================= #

def stage_freeze() -> None:
    """Заморозить измеренную цепочку в models/chain_test.npz.

    Три вектора по 250000, каждый — РАЗНОСТЬ ОТПРАВЛЕННЫХ ФАЙЛОВ, а не выход модели,
    поэтому их нельзя пересчитать из весов и они входят в пакет как данные:

               его моменты и есть REF_MEAN/REF_SD.
      carry_lp накопленная цепочка LB-поправок: ref_lp минус приведённый к тем же
               моментам старый бленд (work/preds/blend_cal_test.parquet). Ровно то,
               что make_candidate.py делает флагом --carry-from blend_cal.
      dir_old  старое (табличное) направление поправки на молчащих, приведённое к
               сабмиты, отличающиеся ровно этим направлением с силой 0.5.
    """
    import polars as pl
    sys.path.insert(0, str(SCRIPTS))
    from subs import lp as sub_lp

    p = WORK_PREDS / "blend_cal_test.parquet"
    if not p.exists():
        raise MissingArtifact(f"нет {p} — без него не восстановить перенос цепочки")
    d = pl.read_parquet(p).sort("user_id")
    assert np.array_equal(d["user_id"].to_numpy(), uid), "user_id старого бленда не совпал"
    xo = np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))
    bo = REF_SD / xo.std()
    carry = ref - ((REF_MEAN - bo * xo.mean()) + bo * xo)

    q = float((mdl_tektit ** 2).mean())
    mdl_tektit = mdl_tektit * float(np.sqrt(Q_REF / q))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / CHAIN_NPZ
    np.savez(out, user_id=uid.astype(np.int64), ref_lp=ref, carry_lp=carry, dir_old=mdl_tektit)
    log(f"записан {out}: ref среднее {ref.mean():.6f} разброс {ref.std():.6f}; "
        f"цепочка разброс {carry.std():.5f}; старое направление q={q:.6f} -> {Q_REF:.6f}")


# ============================== СТАДИЯ: CHECK ================================ #

def stage_check(verify_blend: bool = False) -> int:
    """Что есть, чего не хватает и какой командой каждое получается."""
    miss: list[str] = []
    print(f"OZON_ROOT:  {ROOT}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"work/models:{WORK_MODELS}")

    print("\n=== 1. входные данные ===")
    for f in ("train.parquet", "sample_submit.csv"):
        ok = (ROOT / f).exists()
        print(f"  {f:<24} {'есть' if ok else 'НЕТ'}")
        if not ok:
            miss.append(f"входной файл {f}")

    print("\n=== 2. признаки и тензоры ===")
    print(f"  якоря: тест {TEST_ANCHOR_ISO}, валидация {VAL_ANCHOR_ISO}, "
          f"{len(SILENCE_ANCHORS)} якорей модели молчания")
    fm = missing_features()
    bad = {w.split(" (")[0] for w, _, _ in fm}
    for tier in list(FEATURE_TIERS) + ["тензоры seq3", "тензоры seq2"]:
        label = tier if tier.startswith("тензоры") else f"признаки {tier}"
        if label not in bad:
            print(f"  есть {label}")
    for what, sample, cmd in fm:
        print(f"  НЕТ  {what}\n       нет, например: {sample}\n       команда: {cmd}")
        miss.append(what)

    print(f"\n=== 3. базовые модели бленда ({len(needed_bases())} шт) ===")
    print("  состояние «готово» у строк «прогноз-артефакт» означает файл в work/preds:\n"
          "  их трейнеры не сохраняют веса (нет вызовов work/scripts/model_io.py), и\n"
          "  в отгружаемом пакете, где work/preds пуст, они требуют повторного обучения.")
    print(f"  {'модель':<18}{'вклад':>7}  {'состояние':<9} {'мин':>5}  пояснение")
    print("  " + "-" * 88)
    contrib = {}
    for member, w in BLEND_WEIGHTS.items():
        for b in MEMBER_PARTS[member][0]:
            contrib[b] = contrib.get(b, 0.0) + w / len(MEMBER_PARTS[member][0])
    need_secs = 0
    seen_groups = set()
    for b in sorted(needed_bases(), key=lambda x: -contrib[x]):
        state, why = base_state(b)
        spec = BASES[b]
        if state != "готово":
            g = spec.get("group", b)
            if g not in seen_groups:
                seen_groups.add(g)
                need_secs += spec["secs"]
            miss.append(f"модель {b}")
        print(f"  {b:<18}{contrib[b]:>7.4f}  {state:<9} {spec['secs'] // 60:>5}  {why}")
        if state != "готово":
            print(f"      {howto(b)}")

    print("\n=== 4. таблицы поквантильной калибровки (24 бина) ===")
    for member in sorted(BLEND_WEIGHTS, key=lambda m: -BLEND_WEIGHTS[m]):
        parts, cal = MEMBER_PARTS[member]
        if cal is None:
            continue
        ok = have(f"{cal}.npz")
        src = parts[0] if len(parts) == 1 else f"{cal[:-4]} (усреднение {len(parts)} сидов)"
        print(f"  {cal + '.npz':<28} {'есть' if ok else 'НЕТ '}"
              + ("" if ok else f"   .venv/bin/python work/scripts/calibrate.py --pred {cal[:-4]}"))
        if not ok:
            miss.append(f"калибровка {cal}.npz (из {src})")

    print("\n=== 5. замороженная цепочка (шаги 4-6) ===")
    ok = have(CHAIN_NPZ)
    print(f"  {CHAIN_NPZ:<28} {'есть' if ok else 'НЕТ '}"
          + ("" if ok else "   .venv/bin/python final_submission/inference.py --stage freeze"))
    if not ok:
        miss.append(CHAIN_NPZ)

    print("\n=== 6. модель молчания (шаг 6) ===")
    ok = have("silence_p_test.npz")
    print(f"  silence_p_test.npz           {'есть' if ok else 'нет'}"
          f"   {'' if ok else '-> будет обучена на месте, ~10 мин (5 чистых якорей)'}")
    print(f"  обучающие якоря: {', '.join(SILENCE_ANCHORS[:-1])}")
    print(f"  якорь калибровки наклона: {SILENCE_ANCHORS[-1]}")

    if verify_blend:
        print("\n=== сверка состава бленда с work/reports/blend_reopt.json ===")
        rep = ROOT / "work" / "reports" / "blend_reopt.json"
        if not rep.exists():
            print(f"  нет {rep} — сверять не с чем (это нормально для отгружаемого пакета)")
        else:
            cur = json.loads(rep.read_text())["winner"]["weights"]
            only_here = set(BLEND_WEIGHTS) - set(cur)
            only_there = set(cur) - set(BLEND_WEIGHTS)
            drift = {k: (BLEND_WEIGHTS[k], cur[k]) for k in set(cur) & set(BLEND_WEIGHTS)
                     if abs(BLEND_WEIGHTS[k] - cur[k]) > 1e-6}
            if not (only_here or only_there or drift):
                print("  совпадает полностью")
            for k in sorted(only_here):
                print(f"  только в пакете: {k} {BLEND_WEIGHTS[k]:.6f}")
            for k in sorted(only_there):
                print(f"  только в отчёте: {k} {cur[k]:.6f}")
            for k, (a, b) in sorted(drift.items()):
                print(f"  вес разошёлся: {k} пакет {a:.6f} отчёт {b:.6f}")

    print("\n" + "=" * 90)
    ready = len(needed_bases()) - sum(1 for m in miss if m.startswith("модель "))
    print(f"базовых моделей готово: {ready}/{len(needed_bases())}   "
          f"(их суммарный вклад в бленд "
          f"{sum(contrib[b] for b in needed_bases() if base_state(b)[0] == 'готово'):.3f} "
          f"из {sum(BLEND_WEIGHTS.values()):.3f})")
    print(f"всего не хватает артефактов: {len(miss)}")
    clean_secs, groups = 0, set()
    for b in needed_bases():
        spec = BASES[b]
        if spec["persist"] != "weights" or base_state(b)[0] != "готово":
            g = spec.get("group", b)
            if g not in groups:
                groups.add(g)
                clean_secs += spec["secs"]
    if need_secs:
        print(f"переобучение здесь и сейчас (work/preds на месте): "
              f"{need_secs / 3600:.1f} ч")
    print(f"переобучение на ЧИСТОЙ машине (только веса из models/, work/preds пуст): "
          f"{clean_secs / 3600:.1f} ч")
    print("последовательно, без параллели: на 16 ГБ параллельные тяжёлые прогоны "
          "дважды роняли машину.")
    print("плюс сборка признаков/тензоров из раздела 2, калибровки из раздела 4 "
          "(~1 мин каждая) и обучение модели молчания (~10 мин)")
    print("ВАЖНО про упаковку: веса лежат в work/models. Перед отгрузкой их надо "
          "скопировать\nв final_submission/models — см. README.md, раздел «Упаковка».")
    if miss:
        print("\nПорядок восстановления: раздел 2 -> раздел 3 -> раздел 4 -> раздел 5.\n"
              "Полная последовательность с флагами — final_submission/reproduce_training.md.")
    return len(miss)


# ================================== MAIN ===================================== #

STAGES = ["check", "features", "predict", "ensemble", "moments", "silence",
          "submission", "freeze"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES + ["all"], default="all")
    ap.add_argument("--verify-blend", action="store_true",
                    help="в стадии check сверить зафиксированные веса с текущим "
                         "work/reports/blend_reopt.json")
    args = ap.parse_args()

    if args.stage == "check":
        return 1 if stage_check(args.verify_blend) else 0
    try:
        if args.stage == "freeze":
            stage_freeze()
            return 0
        if args.stage in ("features", "all"):
            stage_features()
        if args.stage in ("predict", "all"):
            stage_predict()
        lp_blend = stage_ensemble() if args.stage in ("ensemble", "all") else None
        base = stage_moments(lp_blend) if args.stage in ("moments", "all") else None
        lp = stage_silence(base) if args.stage in ("silence", "all") else None
        if args.stage in ("submission", "all"):
            stage_submission(lp)
    except MissingArtifact as e:
        print(f"\nОШИБКА: {e}\n", file=sys.stderr)
        print("Состояние артефактов: python inference.py --stage check", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
