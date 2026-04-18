from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "feature and train.py"
SPLITS_DIR = ROOT / "splits"
SEEDS = [42, 52, 62, 72, 82]
SCENARIO_TO_L2 = {"Normal": 0, "充电短路": 1, "GZ": 2}


def load_module():
    spec = importlib.util.spec_from_file_location("mstf_hidet_split_export", str(BASE_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module()


def build_labels(samples):
    l2 = np.array([SCENARIO_TO_L2.get(sample["scenario"], 0) for sample in samples], dtype=int)
    l1 = (l2 > 0).astype(int)
    return {"L1": l1, "L2": l2}


def route_b_split(samples, labels, seed: int):
    real_idx = np.array([i for i, s in enumerate(samples) if s["source"] == "real"])
    virtual_idx = np.array([i for i, s in enumerate(samples) if s["source"] == "virtual"])

    real_samples_sub = [samples[i] for i in real_idx]
    real_labels_sub = {k: v[real_idx] for k, v in labels.items()}
    real_tr_local, real_te_local = BASE.split_by_battery_id(
        real_samples_sub, real_labels_sub, test_size=0.50, random_state=seed
    )
    real_tr_global = real_idx[real_tr_local]
    real_te_global = real_idx[real_te_local]

    virtual_samples = [samples[i] for i in virtual_idx]
    virtual_labels = {k: v[virtual_idx] for k, v in labels.items()}
    v_tr_local, v_val_local = BASE.split_by_battery_id(
        virtual_samples, virtual_labels, test_size=0.10, random_state=seed
    )
    v_tr = virtual_idx[v_tr_local]
    v_val = virtual_idx[v_val_local]

    n_real_val = max(1, len(real_tr_global) // 5)
    np.random.seed(seed)
    perm = np.random.permutation(len(real_tr_global))
    real_val_pick = real_tr_global[perm[:n_real_val]]
    real_tr_kept = real_tr_global[perm[n_real_val:]]

    idx_tr = np.concatenate([v_tr, real_tr_kept])
    idx_val = np.concatenate([v_val, real_val_pick])
    idx_te = real_te_global
    return idx_tr, idx_val, idx_te


def rows_for_seed(samples, seed: int):
    labels = build_labels(samples)
    idx_tr, idx_val, idx_te = route_b_split(samples, labels, seed)
    split_map = {}
    for idx in idx_tr.tolist():
        split_map[idx] = "train"
    for idx in idx_val.tolist():
        split_map[idx] = "val"
    for idx in idx_te.tolist():
        split_map[idx] = "test"

    rows = []
    for idx, sample in enumerate(samples):
        rows.append(
            {
                "seed": seed,
                "index": idx,
                "split": split_map[idx],
                "source": sample.get("source"),
                "scenario": sample.get("scenario"),
                "filename": sample.get("filename"),
                "battery_id": sample.get("battery_id"),
                "resistance": sample.get("resistance"),
            }
        )
    return rows


def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    samples = BASE.load_all_data()
    all_rows = []
    for seed in SEEDS:
        all_rows.extend(rows_for_seed(samples, seed))

    full_df = pd.DataFrame(all_rows)
    full_df.to_csv(SPLITS_DIR / "repeated_route_b_splits.csv", index=False, encoding="utf-8-sig")
    full_df[full_df["seed"] == 42].to_csv(SPLITS_DIR / "route_b_seed42_split.csv", index=False, encoding="utf-8-sig")

    summary = full_df.groupby(["seed", "split", "source", "scenario"]).size().reset_index(name="count")
    summary.to_csv(SPLITS_DIR / "split_summary_by_seed.csv", index=False, encoding="utf-8-sig")

    meta = {
        "seeds": SEEDS,
        "note": "Exported from the AE9 original Route B split logic without changing the method."
    }
    (SPLITS_DIR / "split_export_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
