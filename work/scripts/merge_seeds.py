"""Усреднение групп сидов в log1p и пересборка калиброванного пула.

Запуск: python work/scripts/merge_seeds.py
Находит группы моделей одного семейства с разными сидами, усредняет их предсказания
в log1p-пространстве, калибрует результат и логирует.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, str(Path(__file__).parent))
from common import PREDS_DIR, VAL_ANCHOR, load_anchor, rmsle
from exp_lib import save_preds, log_score

GROUPS = {
    "c_ts2_avg":   ["c_ts2_s42", "c_ts2_s1337", "c_ts2_s7", "c_ts2_s2024", "c_ts2_s555"],
    "mlpziln_avg": ["mlpziln", "mlpziln_b", "mlpziln_c"],
    "countaov_avg":["countaov", "countaov_s1337", "countaov_s7"],
    "fusion_avg":  ["fusion_f", "fusion_s7", "fusion_s2024"],
    "channel_avg": ["channel2", "channel_s1337"],
    "twl_v7_avg":  ["twl_v7", "twl_v7_s1337", "twl_v7_s7"],
    "behavonly_avg":["behavonly", "behavonly_s1337", "behavonly_s7"],
    "mlpbin_avg":  ["mlpbin", "mlpbin_b", "mlpbin_c", "mlpbin_d"],
    "fusion_v3_rb_avg":["fusion_v3_rb555", "fusion_v3_rb2024", "fusion_v3_rb1337"],
    "fusion_v3_fine_avg":["fusion_v3_fine42", "fusion_v3_fine1337", "fusion_v3_fine7",
                        "fusion_v3_fine555", "fusion_v3_fine2024"],
    "fusion_v3_avg":["fusion_v3", "fusion_v3_s1337", "fusion_v3_s7", "fusion_v3_s2024", "fusion_v3_s555"],
}

def main():
    val = load_anchor(VAL_ANCHOR, columns=["user_id", "target"]).sort("user_id")
    y = val["target"].to_numpy(); uid = val["user_id"].to_numpy()
    for out, members in GROUPS.items():
        have = [m for m in members
                if (PREDS_DIR / f"{m}_val.parquet").exists()
                and (PREDS_DIR / f"{m}_test.parquet").exists()]
        if len(have) < 2:
            print(f"{out}: только {len(have)} членов, пропуск")
            continue
        for split, ids in (("val", uid), ("test", None)):
            acc, uid_s = None, None
            for m in have:
                d = pl.read_parquet(PREDS_DIR / f"{m}_{split}.parquet").sort("user_id")
                if uid_s is None:
                    uid_s = d["user_id"].to_numpy()
                lp = np.log1p(np.clip(d["pred"].to_numpy().astype(np.float64), 0, None))
                acc = lp if acc is None else acc + lp
            acc /= len(have)
            save_preds(out, split, uid_s, np.expm1(np.clip(acc, 0, None)))
            if split == "val":
                s = rmsle(y, np.expm1(acc))
                solo = min(rmsle(y, np.expm1(np.log1p(np.clip(
                    pl.read_parquet(PREDS_DIR / f"{m}_val.parquet").sort("user_id")["pred"]
                    .to_numpy().astype(np.float64), 0, None)))) for m in have)
                log_score(out, s, f"seed-avg of {len(have)}: {have}; лучший одиночный {solo:.6f}")
                print(f"{out}: {s:.6f} из {len(have)} сидов (лучший одиночный {solo:.6f})")

if __name__ == "__main__":
    main()
