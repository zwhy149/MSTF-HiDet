import argparse
import csv
import importlib.util
import json
import os
import pickle
import subprocess
import sys
from collections import Counter, defaultdict
from importlib.machinery import SourceFileLoader
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
ORIGINAL_SCRIPT = ROOT / "feature and train.py"
RUNNER_SCRIPT = ROOT / "run_single_seed_avg_eval.py"
DEFAULT_VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON_EXE = Path(os.environ.get("MSTF_PYTHON_EXE", str(DEFAULT_VENV_PYTHON if DEFAULT_VENV_PYTHON.exists() else sys.executable)))
AVG_ROOT = ROOT / "avg_evaluation_results"
RUNS_ROOT = AVG_ROOT / "runs"
DEFAULT_SEEDS = [42, 52, 62, 72, 82]
FIXED_BASELINES = ["Random Forest", "1D-CNN", "XGBoost", "KNN"]
L2_NAMES = ["Normal", "Charging short-circuit", "Full-SOC Resting Short-circuit"]
L1_NAMES = ["Normal", "Fault"]


def load_original_module(module_path: Path):
    loader = SourceFileLoader("mstf_hidet_original_avg_tools", str(module_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def ensure_dirs():
    AVG_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def format_mean_std(mean_value, std_value, multiplier=1.0, digits=3, pct=False):
    mean_scaled = mean_value * multiplier
    std_scaled = std_value * multiplier
    if pct:
        return f"{mean_scaled:.1f}% ± {std_scaled:.1f}%"
    return f"{mean_scaled:.{digits}f} ± {std_scaled:.{digits}f}"


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def run_seed(seed: int):
    run_dir = RUNS_ROOT / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON_EXE),
        str(RUNNER_SCRIPT),
        "--seed",
        str(seed),
        "--output-dir",
        str(run_dir),
    ]
    completed = subprocess.run(cmd, cwd=str(ROOT), check=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Seed {seed} failed with exit code {completed.returncode}")
    return run_dir


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CNN1DBase(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(x.unsqueeze(1)).squeeze(-1))


def quick_train_cnn(X_tr, y_tr, X_te, y_te, seed: int, device: str):
    set_seed(seed)
    num_classes = int(max(np.max(y_tr), np.max(y_te)) + 1)
    model = CNN1DBase(num_classes).to(device)
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.LongTensor(y_tr).to(device)
    X_te_t = torch.FloatTensor(X_te).to(device)
    counts = Counter(y_tr.tolist())
    cw = torch.FloatTensor([1.0 / max(counts.get(i, 1), 1) for i in range(num_classes)]).to(device)
    cw = cw / cw.sum() * num_classes
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30, eta_min=1e-5)
    bs = min(256, len(X_tr_t))
    for _ in range(30):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=device)
        for i in range(0, len(X_tr_t), bs):
            idx = perm[i:i + bs]
            logits = model(X_tr_t[idx])
            loss = F.cross_entropy(logits, y_tr_t[idx], weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pred = model(X_te_t).argmax(1).cpu().numpy()
    return {
        "accuracy": accuracy_score(y_te, pred),
        "f1_macro": f1_score(y_te, pred, average="macro", zero_division=0),
        "predictions": pred,
    }


def run_fixed_baselines(cache: dict, seed: int):
    X_tr = cache["X_train_s"]
    y_tr = cache["labels_train"]["L2"]
    X_te = cache["X_test_s"]
    y_te = cache["labels_test"]["L2"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=seed,
        verbosity=0,
    )
    xgb.fit(X_tr, y_tr)
    xgb_pred = xgb.predict(X_te)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr, y_tr)
    knn_pred = knn.predict(X_te)

    cnn_res = quick_train_cnn(X_tr, y_tr, X_te, y_te, seed=seed, device=device)

    return {
        "Random Forest": {
            "accuracy": accuracy_score(y_te, rf_pred),
            "f1_macro": f1_score(y_te, rf_pred, average="macro", zero_division=0),
            "predictions": rf_pred,
        },
        "1D-CNN": cnn_res,
        "XGBoost": {
            "accuracy": accuracy_score(y_te, xgb_pred),
            "f1_macro": f1_score(y_te, xgb_pred, average="macro", zero_division=0),
            "predictions": xgb_pred,
        },
        "KNN": {
            "accuracy": accuracy_score(y_te, knn_pred),
            "f1_macro": f1_score(y_te, knn_pred, average="macro", zero_division=0),
            "predictions": knn_pred,
        },
    }


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_exact_mstf_metrics(checkpoint: dict):
    labels_test = checkpoint["labels_test"]
    pred_l1 = checkpoint["pred_l1"]
    pred_l2 = checkpoint["pred_l2"]
    return {
        "L1_acc": float(accuracy_score(labels_test["L1"], pred_l1)),
        "L2_acc": float(accuracy_score(labels_test["L2"], pred_l2)),
        "L2_f1": float(f1_score(labels_test["L2"], pred_l2, average="macro", zero_division=0)),
    }


def build_average_confusions(run_payloads):
    l1_cms = []
    l2_cms = []
    for payload in run_payloads:
        ckpt = payload["checkpoint"]
        labels_test = ckpt["labels_test"]
        pred_l1 = ckpt["pred_l1"]
        pred_l2 = ckpt["pred_l2"]
        cm_l1 = confusion_matrix(labels_test["L1"], pred_l1, labels=[0, 1]).astype(float)
        cm_l2 = confusion_matrix(labels_test["L2"], pred_l2, labels=[0, 1, 2]).astype(float)
        cm_l1 = cm_l1 / np.clip(cm_l1.sum(axis=1, keepdims=True), 1, None)
        cm_l2 = cm_l2 / np.clip(cm_l2.sum(axis=1, keepdims=True), 1, None)
        l1_cms.append(cm_l1)
        l2_cms.append(cm_l2)
    return np.stack(l1_cms), np.stack(l2_cms)


def plot_average_confusions(run_payloads):
    l1_stack, l2_stack = build_average_confusions(run_payloads)
    l1_mean = l1_stack.mean(axis=0)
    l1_std = l1_stack.std(axis=0)
    l2_mean = l2_stack.mean(axis=0)
    l2_std = l2_stack.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    specs = [
        (axes[0], l1_mean, l1_std, L1_NAMES, "(a) L1 Mean Confusion"),
        (axes[1], l2_mean, l2_std, L2_NAMES, "(b) L2 Mean Confusion"),
    ]
    for ax, mean_cm, std_cm, names, title in specs:
        im = ax.imshow(mean_cm, cmap="YlOrRd", vmin=0, vmax=1)
        for i in range(mean_cm.shape[0]):
            for j in range(mean_cm.shape[1]):
                text = f"{mean_cm[i, j]:.2f}\n±{std_cm[i, j]:.2f}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white" if mean_cm[i, j] > 0.5 else "black",
                )
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontweight="bold")
        ax.set_yticklabels(names, fontweight="bold")
        ax.set_xlabel("Predicted", fontweight="bold")
        ax.set_ylabel("True", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(AVG_ROOT / "Fig2_confusion_matrices_avg.png", dpi=600)
    plt.savefig(AVG_ROOT / "Fig2_confusion_matrices_avg.pdf", format="pdf")
    plt.close(fig)


def plot_pooled_tsne(run_payloads):
    embeds = []
    labels = []
    run_ids = []
    for idx, payload in enumerate(run_payloads):
        ckpt = payload["checkpoint"]
        embeds.append(ckpt["embed_np"])
        labels.append(ckpt["labels_test"]["L2"])
        run_ids.extend([payload["seed"]] * len(ckpt["labels_test"]["L2"]))
    X = np.vstack(embeds)
    y = np.concatenate(labels)
    if len(X) > 2000:
        sel = np.random.RandomState(42).choice(len(X), 2000, replace=False)
        X = X[sel]
        y = y[sel]
        run_ids = [run_ids[i] for i in sel]
    coords = TSNE(
        n_components=2,
        perplexity=min(30, max(5, len(X) // 20)),
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for cls, color in enumerate(colors):
        mask = y == cls
        if np.any(mask):
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=26,
                alpha=0.75,
                color=color,
                label=L2_NAMES[cls],
                edgecolors="white",
                linewidths=0.35,
            )
    ax.set_title("Pooled Test Embeddings Across Repeated Route B Runs", fontweight="bold")
    ax.set_xlabel("t-SNE-1", fontweight="bold")
    ax.set_ylabel("t-SNE-2", fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.15, linestyle="--")
    plt.tight_layout()
    plt.savefig(AVG_ROOT / "Fig3_tsne_repeated_pooled.png", dpi=600)
    plt.savefig(AVG_ROOT / "Fig3_tsne_repeated_pooled.pdf", format="pdf")
    plt.close(fig)


def plot_sota_average(summary_by_method):
    names = FIXED_BASELINES + ["MSTF-HiDet\n(Ours)"]
    acc_means = []
    acc_stds = []
    f1_means = []
    f1_stds = []
    for name in FIXED_BASELINES:
        acc_means.append(summary_by_method[name]["accuracy"]["mean"])
        acc_stds.append(summary_by_method[name]["accuracy"]["std"])
        f1_means.append(summary_by_method[name]["f1_macro"]["mean"])
        f1_stds.append(summary_by_method[name]["f1_macro"]["std"])
    acc_means.append(summary_by_method["MSTF-HiDet"]["L2_acc"]["mean"])
    acc_stds.append(summary_by_method["MSTF-HiDet"]["L2_acc"]["std"])
    f1_means.append(summary_by_method["MSTF-HiDet"]["L2_f1"]["mean"])
    f1_stds.append(summary_by_method["MSTF-HiDet"]["L2_f1"]["std"])

    x = np.arange(len(names))
    colors = ["#78909C"] * len(FIXED_BASELINES) + ["#E53935"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    bars1 = ax1.bar(x, acc_means, yerr=acc_stds, width=0.62, color=colors, edgecolor="black", linewidth=0.8, capsize=4)
    bars2 = ax2.bar(x, f1_means, yerr=f1_stds, width=0.62, color=colors, edgecolor="black", linewidth=0.8, capsize=4)
    for ax, bars, vals, stds, ylabel, title, panel in [
        (ax1, bars1, acc_means, acc_stds, "Accuracy", "Accuracy", "(a)"),
        (ax2, bars2, f1_means, f1_stds, "F1-Score (Macro)", "F1-Score", "(b)"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10, fontweight="bold")
        ax.set_ylim(max(0.75, min(vals) - 0.05), 1.01)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=8)
        ax.text(0.02, 0.95, panel, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", ha="left")
        ax.grid(True, alpha=0.15, axis="y", linestyle="--")
        for bar, val, std in zip(bars, vals, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + std + 0.005,
                f"{val:.3f}\n±{std:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    plt.tight_layout()
    plt.savefig(AVG_ROOT / "Fig4_sota_fixed_baselines_avg.png", dpi=600)
    plt.savefig(AVG_ROOT / "Fig4_sota_fixed_baselines_avg.pdf", format="pdf")
    plt.close(fig)


def write_csv(path: Path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_files(run_rows, summary_by_method, safety_summary, seeds):
    repeated_path = AVG_ROOT / "repeated_eval_results.csv"
    fieldnames = list(run_rows[0].keys())
    write_csv(repeated_path, run_rows, fieldnames)

    summary_rows = []
    for method, metrics in summary_by_method.items():
        for metric_name, stats in metrics.items():
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric_name,
                    "mean": f"{stats['mean']:.6f}",
                    "std": f"{stats['std']:.6f}",
                    "min": f"{stats['min']:.6f}",
                    "max": f"{stats['max']:.6f}",
                }
            )
    write_csv(
        AVG_ROOT / "repeated_eval_summary.csv",
        summary_rows,
        ["method", "metric", "mean", "std", "min", "max"],
    )

    table_rows = []
    for name in FIXED_BASELINES:
        stats = summary_by_method[name]
        table_rows.append(
            {
                "Method": name,
                "Accuracy": format_mean_std(stats["accuracy"]["mean"], stats["accuracy"]["std"]),
                "F1-Macro": format_mean_std(stats["f1_macro"]["mean"], stats["f1_macro"]["std"]),
            }
        )
    ours = summary_by_method["MSTF-HiDet"]
    table_rows.append(
        {
            "Method": "MSTF-HiDet (Ours)",
            "Accuracy": format_mean_std(ours["L2_acc"]["mean"], ours["L2_acc"]["std"]),
            "F1-Macro": format_mean_std(ours["L2_f1"]["mean"], ours["L2_f1"]["std"]),
        }
    )
    write_csv(AVG_ROOT / "table2_avg_results.csv", table_rows, ["Method", "Accuracy", "F1-Macro"])

    with open(AVG_ROOT / "table2_avg_results.md", "w", encoding="utf-8") as f:
        f.write("| Method | Accuracy (mean ± std) | F1-Macro (mean ± std) |\n")
        f.write("|---|---:|---:|\n")
        for row in table_rows:
            f.write(f"| {row['Method']} | {row['Accuracy']} | {row['F1-Macro']} |\n")

    ours_acc = summary_by_method["MSTF-HiDet"]["L2_acc"]
    ours_f1 = summary_by_method["MSTF-HiDet"]["L2_f1"]
    paragraph = (
        "In the revised reporting, the original AE7-copy Route B evaluation protocol was repeated across "
        f"{len(seeds)} random seeds ({', '.join(map(str, seeds))}) without altering the underlying method, feature set, losses, "
        "or the original code logic in the source folder. Under this repeated evaluation setting, MSTF-HiDet "
        f"achieved an average L2 accuracy of {format_mean_std(ours_acc['mean'], ours_acc['std'])} "
        f"and an average L2 macro-F1 of {format_mean_std(ours_f1['mean'], ours_f1['std'])}. "
        "Accordingly, the previously reported single-run perfect result should be interpreted as one realization "
        "of the original protocol rather than the sole result used for manuscript-facing statistical reporting."
    )
    with open(AVG_ROOT / "section4_4_avg_result_text.md", "w", encoding="utf-8") as f:
        f.write(paragraph + "\n")

    with open(AVG_ROOT / "absolute_claim_revisions.md", "w", encoding="utf-8") as f:
        f.write(
            "Recommended replacement sentences:\n\n"
            f"1. Across repeated evaluations under the original Route B protocol, MSTF-HiDet achieved an average L2 accuracy of {format_mean_std(ours_acc['mean'], ours_acc['std'])}.\n\n"
            f"2. The proposed method maintained stable diagnosis performance across repeated held-out evaluations, with an average L2 macro-F1 of {format_mean_std(ours_f1['mean'], ours_f1['std'])}.\n\n"
            "3. The single-run 1.00 result is retained as an individual run outcome, whereas the manuscript-facing result is reported as mean ± standard deviation over repeated evaluations.\n"
        )

    final_report_avg = {
        "protocol": "Original AE7-copy logic preserved; repeated-average reporting added in AE9 only.",
        "seeds": seeds,
        "MSTF-HiDet": summary_by_method["MSTF-HiDet"],
        "SOTA_fixed_baselines": {
            k: summary_by_method[k] for k in FIXED_BASELINES
        },
        "safety_metrics": safety_summary,
        "figure_files": {
            "avg_confusion": "Fig2_confusion_matrices_avg.png",
            "pooled_tsne": "Fig3_tsne_repeated_pooled.png",
            "avg_sota": "Fig4_sota_fixed_baselines_avg.png",
        },
    }
    with open(AVG_ROOT / "final_report_avg.json", "w", encoding="utf-8") as f:
        json.dump(final_report_avg, f, indent=2, ensure_ascii=False)

    with open(AVG_ROOT / "README_avg_evaluation.md", "w", encoding="utf-8") as f:
        f.write(
            "# AE9 Repeated-Average Evaluation\n\n"
            "- Source logic: copied from `D:\\AE7 - Copy` without editing the original folder.\n"
            "- Repeated seeds: " + ", ".join(map(str, seeds)) + "\n"
            "- Fixed baseline set for the new SOTA figure: Random Forest, 1D-CNN, XGBoost, KNN.\n"
            "- The averaged SOTA figure is produced by the wrapper script and does not overwrite the original single-run figure.\n"
            "- The new pooled t-SNE figure aggregates embeddings across repeated test runs because a literal arithmetic average of t-SNE coordinates is not statistically meaningful.\n"
        )


def build_safety_summary(run_payloads):
    bucket = defaultdict(lambda: defaultdict(list))
    for payload in run_payloads:
        report = payload["report"]
        for cls_name, metrics in report["safety_metrics"].items():
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    bucket[cls_name][metric_name].append(metric_value)
    out = {}
    for cls_name, metrics in bucket.items():
        out[cls_name] = {metric_name: summarize(values) for metric_name, values in metrics.items()}
    return out


def aggregate_results(run_payloads):
    run_rows = []
    mstf_bucket = defaultdict(list)
    baseline_bucket = {name: defaultdict(list) for name in FIXED_BASELINES}

    for payload in run_payloads:
        seed = payload["seed"]
        report = payload["report"]
        baselines = payload["fixed_baselines"]
        mstf_exact = payload["mstf_exact"]
        row = {
            "seed": seed,
            "chosen_route": report["split"]["chosen_route"],
            "train_samples": report["split"]["train"],
            "val_samples": report["split"]["val"],
            "test_samples": report["split"]["test"],
            "mstf_l1_acc": mstf_exact["L1_acc"],
            "mstf_l2_acc": mstf_exact["L2_acc"],
            "mstf_l2_f1": mstf_exact["L2_f1"],
        }
        mstf_bucket["L1_acc"].append(mstf_exact["L1_acc"])
        mstf_bucket["L2_acc"].append(mstf_exact["L2_acc"])
        mstf_bucket["L2_f1"].append(mstf_exact["L2_f1"])

        for name in FIXED_BASELINES:
            row[f"{name}_acc"] = baselines[name]["accuracy"]
            row[f"{name}_f1"] = baselines[name]["f1_macro"]
            baseline_bucket[name]["accuracy"].append(baselines[name]["accuracy"])
            baseline_bucket[name]["f1_macro"].append(baselines[name]["f1_macro"])
        run_rows.append(row)

    summary_by_method = {
        "MSTF-HiDet": {metric_name: summarize(values) for metric_name, values in mstf_bucket.items()}
    }
    for name in FIXED_BASELINES:
        summary_by_method[name] = {
            metric_name: summarize(values) for metric_name, values in baseline_bucket[name].items()
        }
    return run_rows, summary_by_method


def main():
    parser = argparse.ArgumentParser(description="Repeated average evaluation for the copied AE7 pipeline.")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    ensure_dirs()
    if not PYTHON_EXE.exists():
        raise FileNotFoundError(f"Python executable not found: {PYTHON_EXE}")

    run_payloads = []
    for seed in args.seeds:
        run_dir = RUNS_ROOT / f"seed_{seed}"
        final_report = run_dir / "final_report.json"
        feature_cache = run_dir / "_feature_cache.pkl"
        checkpoint = run_dir / "checkpoint_results.pkl"
        if not final_report.exists():
            print(f"[RUN] seed={seed}")
            run_dir = run_seed(seed)
        else:
            print(f"[SKIP] seed={seed} already exists: {run_dir}")

        report = read_json(final_report)
        cache = read_pickle(feature_cache)
        ckpt = read_pickle(checkpoint)
        mstf_exact = compute_exact_mstf_metrics(ckpt)
        fixed_baseline_json = run_dir / "fixed_baseline_results.json"
        if fixed_baseline_json.exists():
            fixed_baselines = read_json(fixed_baseline_json)
        else:
            fixed_baselines = run_fixed_baselines(cache, seed)
            baseline_json_ready = {
                name: {
                    "accuracy": float(values["accuracy"]),
                    "f1_macro": float(values["f1_macro"]),
                }
                for name, values in fixed_baselines.items()
            }
            with open(fixed_baseline_json, "w", encoding="utf-8") as f:
                json.dump(baseline_json_ready, f, indent=2, ensure_ascii=False)

        fixed_baselines = {
            name: {
                "accuracy": float(values["accuracy"]),
                "f1_macro": float(values["f1_macro"]),
            }
            for name, values in fixed_baselines.items()
        }

        run_payloads.append(
            {
                "seed": seed,
                "run_dir": run_dir,
                "report": report,
                "cache": cache,
                "checkpoint": ckpt,
                "fixed_baselines": fixed_baselines,
                "mstf_exact": mstf_exact,
            }
        )

    run_rows, summary_by_method = aggregate_results(run_payloads)
    safety_summary = build_safety_summary(run_payloads)
    write_summary_files(run_rows, summary_by_method, safety_summary, args.seeds)
    plot_average_confusions(run_payloads)
    plot_pooled_tsne(run_payloads)
    plot_sota_average(summary_by_method)

    ours = summary_by_method["MSTF-HiDet"]
    print("=" * 70)
    print("Repeated-average evaluation completed")
    print(f"Runs: {len(args.seeds)}")
    print(f"L2 Accuracy: {format_mean_std(ours['L2_acc']['mean'], ours['L2_acc']['std'])}")
    print(f"L2 Macro-F1: {format_mean_std(ours['L2_f1']['mean'], ours['L2_f1']['std'])}")
    print(f"Outputs: {AVG_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
