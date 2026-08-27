"""Пути и режимы для воспроизводимого пайплайна kostya46.
OZON_ROOT — корень репо с train.parquet (default: два уровня вверх от этого файла).
KWORK — рабочая директория сборки (default: OZON_ROOT/work_kostya/_build).
KSMOKE=1 — смоук-режим: 8 раундов, 1 сид, 1 срез (только проверка, что код работает;
по правилу команды смоуки НЕ ранжируют конфигурации).
KSEED_OFFSET — сдвиг сидов (default 0 = канон).
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("OZON_ROOT", Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get("KWORK", ROOT / "work_kostya" / "_build"))
WORK.mkdir(parents=True, exist_ok=True)
TRAIN_PARQUET = ROOT / "train.parquet"
SMOKE = os.environ.get("KSMOKE", "") == "1"
SEED_OFF = int(os.environ.get("KSEED_OFFSET", "0"))
NUM_THREADS = int(os.environ.get("KTHREADS", "2"))  # бит-повтор шipped-файлов требует 2

def wp(name: str) -> str:
    return str(WORK / name)
