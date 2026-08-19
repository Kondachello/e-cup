"""Persistence of fitted model artifacts (freeze step).

Why this exists: every trainer used to keep its model in memory, write the
predictions and exit, so the submission could not be reproduced without a full
retrain.  Each production trainer now calls into this module at the end of its
retrain phase and drops the fitted artifacts into ``work/models/``, which is
what ``final_submission/inference.py`` loads.

Naming contract (NAME = the trainer's ``--name``):

  ``NAME.txt``               LightGBM booster (``Booster.save_model``)
  ``NAME__TAG.txt``          LightGBM booster of sub-model TAG (channel models)
  ``NAME.xgb.json``          XGBoost booster (``Booster.save_model``)
  ``NAME__TAG.xgb.json``     XGBoost sub-model
  ``NAME.cbm``               CatBoost model (``save_model``)
  ``NAME_seed{S}.pt``        torch ``state_dict`` for seed S (retrain weights)
  ``NAME_meta.json``         everything inference needs besides the weights:
                             feature-column order, architecture cfg, seed list,
                             per-seed epoch counts, env feature flags
  ``NAME_stats.npz``         preprocessing stats — written by the trainers
                             themselves, unchanged, listed here for reference

Two invariants this module must never break, because trainings are long and
already running:

1. **Nothing here changes training numerics.**  Only serialization happens.
2. **Nothing here can fail a run.**  Every entry point swallows its exceptions
   and prints a ``[model_io] WARN`` line: a run that already produced valid
   predictions must not die because a file could not be written.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import WORK

MODELS_DIR = WORK / "models"

# Env flags that select which feature tiers load_anchor() joins in. Recorded in
# the meta file because the column order of a model depends on them.
FEATURE_FLAGS = ("USE_V2", "USE_V3", "USE_V4", "USE_V6", "USE_V7", "USE_V8",
                 "USE_V10", "USE_SEQOOF")


def models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def _stem(name: str, tag: str | None) -> str:
    return name if tag is None else f"{name}__{tag}"


def _ok(path: Path) -> None:
    size = path.stat().st_size if path.exists() else 0
    print(f"[model_io] saved {path.name} ({size / 2**20:.1f} MB)", flush=True)


def _warn(what: str, exc: Exception) -> None:
    print(f"[model_io] WARN could not save {what}: {type(exc).__name__}: {exc}",
          flush=True)


def save_lgb(name: str, booster, tag: str | None = None,
             num_iteration: int | None = None) -> Path | None:
    """LightGBM booster -> work/models/NAME[__TAG].txt (native text format)."""
    try:
        p = models_dir() / f"{_stem(name, tag)}.txt"
        booster.save_model(str(p), num_iteration=num_iteration)
        _ok(p)
        return p
    except Exception as e:                                    # noqa: BLE001
        _warn(f"lgb model {_stem(name, tag)}", e)
        return None


def save_xgb(name: str, booster, tag: str | None = None) -> Path | None:
    """XGBoost booster -> work/models/NAME[__TAG].xgb.json (portable UBJ/JSON)."""
    try:
        p = models_dir() / f"{_stem(name, tag)}.xgb.json"
        booster.save_model(str(p))
        _ok(p)
        return p
    except Exception as e:                                    # noqa: BLE001
        _warn(f"xgb model {_stem(name, tag)}", e)
        return None


def save_cb(name: str, model, tag: str | None = None) -> Path | None:
    """CatBoost model -> work/models/NAME[__TAG].cbm (native binary format)."""
    try:
        p = models_dir() / f"{_stem(name, tag)}.cbm"
        model.save_model(str(p))
        _ok(p)
        return p
    except Exception as e:                                    # noqa: BLE001
        _warn(f"catboost model {_stem(name, tag)}", e)
        return None


BOOSTER_EXT = {"lgb": ".txt", "xgb": ".xgb.json", "cb": ".cbm"}


def booster_filename(kind: str, name: str, tag: str | None = None) -> str:
    """File name save_booster() would use — for recording in the meta json."""
    return f"{_stem(name, tag)}{BOOSTER_EXT.get(kind, '.model')}"


def save_booster(kind: str, name: str, model, tag: str | None = None) -> Path | None:
    """Dispatch on the trainer's --model value ('lgb' / 'xgb' / 'cb')."""
    fn = {"lgb": save_lgb, "xgb": save_xgb, "cb": save_cb}.get(kind)
    if fn is None:
        _warn(f"model {name} (unknown kind {kind!r})", ValueError(kind))
        return None
    return fn(name, model, tag)


def save_torch(name: str, model, seed: int | str, tag: str | None = None) -> Path | None:
    """torch module (or plain state_dict) -> work/models/NAME[__TAG]_seed{S}.pt.

    Tensors are moved to CPU first so weights trained on MPS/CUDA load anywhere.
    """
    try:
        import torch
        state = model.state_dict() if hasattr(model, "state_dict") else model
        cpu_state = {k: v.detach().cpu() if hasattr(v, "detach") else v
                     for k, v in state.items()}
        p = models_dir() / f"{_stem(name, tag)}_seed{seed}.pt"
        torch.save(cpu_state, p)
        _ok(p)
        return p
    except Exception as e:                                    # noqa: BLE001
        _warn(f"torch weights {_stem(name, tag)} seed {seed}", e)
        return None


def save_npz(stem: str, **arrays) -> Path | None:
    """Small numeric artifact (calibration tables, ...) -> work/models/STEM.npz."""
    try:
        import numpy as np
        p = models_dir() / f"{stem}.npz"
        np.savez(p, **arrays)
        _ok(p)
        return p
    except Exception as e:                                    # noqa: BLE001
        _warn(f"npz {stem}", e)
        return None


def save_meta(name: str, **fields) -> Path | None:
    """Everything inference needs besides the weights -> NAME_meta.json.

    Always records the feature-tier env flags, the trainer command line and a
    timestamp; callers add ``feature_cols``, ``cfg``, ``seeds`` etc.
    """
    try:
        meta = {
            "name": name,
            "script": Path(sys.argv[0]).name,
            "argv": sys.argv[1:],
            "feature_flags": {k: os.environ.get(k) for k in FEATURE_FLAGS
                              if os.environ.get(k)},
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        meta.update(fields)
        p = models_dir() / f"{name}_meta.json"
        p.write_text(json.dumps(meta, indent=1, default=str))
        _ok(p)
        return p
    except Exception as e:                                    # noqa: BLE001
        _warn(f"meta {name}", e)
        return None


def load_meta(name: str, models_root: Path | None = None) -> dict:
    root = Path(models_root) if models_root else MODELS_DIR
    return json.loads((root / f"{name}_meta.json").read_text())
