import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parent
AVG_DIR = ROOT / "avg_evaluation_results"
RUNS_DIR = AVG_DIR / "runs"
SEED_DIRS = sorted([p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name.startswith("seed_")])
L2_NAMES = ["Normal", "Charging short-circuit", "Full-SOC Resting Short-circuit"]
L1_NAMES = ["Normal", "Fault"]
THRESHOLDS = np.linspace(0.01, 0.99, 300)


def setup_fonts():
    candidates = [
        "Times New Roman",
        "DejaVu Serif",
        "Liberation Serif",
        "Nimbus Roman",
        "FreeSerif",
        "serif",
    ]
    available = set(f.name for f in font_manager.fontManager.ttflist)
    chosen = "serif"
    for c in candidates:
        if c in available:
            chosen = c
            break
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [chosen, "DejaVu Serif"],
            "font.size": 12,
            "font.weight": "bold",
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": False,
            "ytick.right": False,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "axes.linewidth": 1.5,
            "axes.edgecolor": "black",
            "legend.fontsize": 11,
            "legend.frameon": True,
            "legend.edgecolor": "black",
            "legend.fancybox": False,
            "legend.framealpha": 1.0,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavuserif",
        }
    )
    return chosen


def sci_ax_style(ax):
    ax.tick_params(axis="both", which="both", direction="in", top=False, right=False, width=1.2)
    ax.tick_params(axis="both", which="minor", direction="in", top=False, right=False, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("black")


def add_panel_label(ax, label_text):
    ax.text(
        0.02,
        0.98,
        label_text,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
    )


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_runs():
    runs = []
    for seed_dir in SEED_DIRS:
        seed = int(seed_dir.name.split("_")[-1])
        runs.append(
            {
                "seed": seed,
                "dir": seed_dir,
                "checkpoint": load_pickle(seed_dir / "checkpoint_results.pkl"),
                "report": load_json(seed_dir / "final_report.json"),
                "training_summary": load_json(seed_dir / "training_run_summary.json"),
            }
        )
    if not runs:
        raise FileNotFoundError(f"No repeated run directories found in {RUNS_DIR}")
    return runs


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _stack_histories(runs, key):
    series = []
    max_len = max(len(run["checkpoint"]["history"].get(key, [])) for run in runs)
    out = np.full((len(runs), max_len), np.nan, dtype=float)
    for idx, run in enumerate(runs):
        values = np.asarray(run["checkpoint"]["history"].get(key, []), dtype=float)
        out[idx, : len(values)] = values
    return out


def plot_avg_training_curves(runs, out_dir: Path):
    eps = np.arange(1, max(len(run["checkpoint"]["history"].get("train_loss", [])) for run in runs) + 1)
    train_loss = _stack_histories(runs, "train_loss")
    val_task_loss = _stack_histories(runs, "val_task_loss")
    val_l1_acc = _stack_histories(runs, "val_l1_acc")
    val_l2_acc = _stack_histories(runs, "val_l2_acc")
    val_l2_f1 = _stack_histories(runs, "val_l2_f1")
    val_l3_f1 = _stack_histories(runs, "val_l3_f1")
    lr = _stack_histories(runs, "lr")

    def nan_mean_std(arr):
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

    tl_m, tl_s = nan_mean_std(train_loss)
    vl_m, vl_s = nan_mean_std(val_task_loss)
    l1_m, l1_s = nan_mean_std(val_l1_acc)
    l2_m, l2_s = nan_mean_std(val_l2_acc)
    f1_m, f1_s = nan_mean_std(val_l2_f1)
    l3_m, l3_s = nan_mean_std(val_l3_f1)
    lr_m, lr_s = nan_mean_std(lr)

    warmup_epochs = [run["training_summary"].get("warmup_end_epoch", 10) for run in runs]
    best_epochs = [run["training_summary"].get("best_epoch") for run in runs if run["training_summary"].get("best_epoch") is not None]
    stop_epochs = [run["training_summary"].get("early_stop_epoch") for run in runs if run["training_summary"].get("early_stop_epoch") is not None]
    warmup_epoch = int(round(np.mean(warmup_epochs))) if warmup_epochs else 10
    best_epoch = int(round(np.mean(best_epochs))) if best_epochs else None
    stop_epoch = int(round(np.mean(stop_epochs))) if stop_epochs else None

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    label_xy = (0.01, 0.985)
    label_bbox = dict(facecolor="white", edgecolor="black", alpha=0.90, boxstyle="square,pad=0.18", linewidth=0.8)
    legend_style = dict(frameon=True, fancybox=False, edgecolor="black", framealpha=0.95)
    note_bbox = dict(facecolor="white", edgecolor="black", alpha=0.90, boxstyle="square,pad=0.22", linewidth=0.9)

    lw_main = 2.0
    lw_primary = 2.3
    lw_aux = 1.8

    axes[0].plot(eps, tl_m, color="blue", lw=lw_main, label="Train total (mean)")
    axes[0].fill_between(eps, tl_m - tl_s, tl_m + tl_s, color="blue", alpha=0.12)
    axes[0].plot(eps, vl_m, color="red", ls="--", lw=lw_main, label="Val task (mean)")
    axes[0].fill_between(eps, np.clip(vl_m - vl_s, 0, None), vl_m + vl_s, color="red", alpha=0.10)
    axes[0].set_xlabel("Epoch", fontweight="bold")
    axes[0].set_ylabel("Loss", fontweight="bold")
    axes[0].set_title("Objective Tracking", fontweight="bold", pad=8)
    axes[0].text(label_xy[0], label_xy[1], "(a)", transform=axes[0].transAxes, fontsize=14, fontweight="bold", va="top", ha="left", bbox=label_bbox, zorder=10)
    leg = axes[0].legend(loc="upper right", **legend_style)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    leg.get_frame().set_linewidth(0.9)
    axes[0].grid(True, alpha=0.15, linestyle="--")

    axes[1].plot(eps, l1_m, color="red", ls="-", lw=lw_main, label="L1 Acc (mean)")
    axes[1].fill_between(eps, np.clip(l1_m - l1_s, 0, 1), np.clip(l1_m + l1_s, 0, 1), color="red", alpha=0.10)
    axes[1].plot(eps, l2_m, color="blue", ls="--", lw=lw_main, label="L2 Acc (mean)")
    axes[1].fill_between(eps, np.clip(l2_m - l2_s, 0, 1), np.clip(l2_m + l2_s, 0, 1), color="blue", alpha=0.10)
    axes[1].plot(eps, f1_m, color="black", ls="-", lw=lw_primary, label="L2 Macro-F1 (mean)")
    axes[1].fill_between(eps, np.clip(f1_m - f1_s, 0, 1), np.clip(f1_m + f1_s, 0, 1), color="black", alpha=0.08)
    l3_valid_mask = np.isfinite(l3_m)
    if l3_valid_mask.any():
        axes[1].plot(eps[l3_valid_mask], l3_m[l3_valid_mask], color="#2ca02c", ls="--", lw=lw_aux, alpha=0.85, label="L3 Macro-F1 (mean)")
        axes[1].fill_between(
            eps[l3_valid_mask],
            np.clip(l3_m[l3_valid_mask] - l3_s[l3_valid_mask], 0, 1),
            np.clip(l3_m[l3_valid_mask] + l3_s[l3_valid_mask], 0, 1),
            color="#2ca02c",
            alpha=0.08,
        )
    axes[1].set_xlabel("Epoch", fontweight="bold")
    axes[1].set_ylabel("Score", fontweight="bold")
    axes[1].set_title("Primary and Auxiliary Validation Metrics", fontweight="bold", pad=8)
    axes[1].set_ylim(0.0, 1.03)
    axes[1].text(label_xy[0], label_xy[1], "(b)", transform=axes[1].transAxes, fontsize=14, fontweight="bold", va="top", ha="left", bbox=label_bbox, zorder=10)
    leg = axes[1].legend(loc="lower right", **legend_style)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    leg.get_frame().set_linewidth(0.9)
    axes[1].grid(True, alpha=0.15, linestyle="--")

    axes[2].plot(eps, lr_m, color="green", lw=lw_main)
    axes[2].fill_between(eps, np.clip(lr_m - lr_s, 0, None), lr_m + lr_s, color="green", alpha=0.10)
    axes[2].set_xlabel("Epoch", fontweight="bold")
    axes[2].set_ylabel("Learning Rate", fontweight="bold")
    axes[2].set_title("Learning Rate", fontweight="bold", pad=8)
    axes[2].text(label_xy[0], label_xy[1], "(c)", transform=axes[2].transAxes, fontsize=14, fontweight="bold", va="top", ha="left", bbox=label_bbox, zorder=10)
    axes[2].grid(True, alpha=0.15, linestyle="--")

    marker_handles = []
    if warmup_epoch and 1 <= warmup_epoch <= len(eps):
        for ax in axes:
            ax.axvline(int(warmup_epoch), color="#6b6b6b", ls=":", lw=1.2, alpha=0.9)
        marker_handles.append(Line2D([0], [0], color="#6b6b6b", lw=1.2, ls=":", label=f"Warm-up end (ep {int(warmup_epoch)})"))

    if best_epoch is not None and 1 <= int(best_epoch) <= len(eps):
        bi = int(best_epoch) - 1
        axes[0].scatter([best_epoch], [vl_m[bi]], color="black", s=26, zorder=6)
        axes[1].scatter([best_epoch], [f1_m[bi]], color="black", s=26, zorder=6)
        axes[2].scatter([best_epoch], [lr_m[bi]], color="black", s=26, zorder=6)
        marker_handles.append(Line2D([0], [0], marker="o", color="black", lw=0, markersize=5, label=f"Best L2 F1 epoch (mean ep {int(best_epoch)})"))

    if stop_epoch is not None and 1 <= int(stop_epoch) <= len(eps):
        for ax in axes:
            ax.axvline(int(stop_epoch), color="#8B0000", ls="--", lw=1.2, alpha=0.9)
        marker_handles.append(Line2D([0], [0], color="#8B0000", lw=1.2, ls="--", label=f"Early-stop epoch (mean ep {int(stop_epoch)})"))

    status_text = f"Best L2 F1 epoch: {best_epoch if best_epoch is not None else 'N/A'}\nMean early stop: {stop_epoch if stop_epoch is not None else 'N/A'}"
    axes[0].text(0.90, 0.14, status_text, transform=axes[0].transAxes, ha="right", va="bottom", fontsize=10, fontweight="bold", bbox=note_bbox)

    if marker_handles:
        leg = axes[2].legend(handles=marker_handles, loc="upper right", **legend_style)
        for t in leg.get_texts():
            t.set_fontweight("bold")
        leg.get_frame().set_linewidth(0.9)

    for ax in axes:
        ax.set_xlim(1, len(eps))
        sci_ax_style(ax)
    plt.tight_layout(pad=0.9, w_pad=1.1)
    plt.savefig(out_dir / "Fig1_training_avg.pdf", format="pdf")
    plt.savefig(out_dir / "Fig1_training_avg.png", dpi=600)
    plt.close(fig)


def collect_confusions(runs):
    l1_counts = []
    l1_norms = []
    l2_counts = []
    l2_norms = []
    for run in runs:
        ckpt = run["checkpoint"]
        y_true_l1 = ckpt["labels_test"]["L1"]
        y_pred_l1 = ckpt["pred_l1"]
        y_true_l2 = ckpt["labels_test"]["L2"]
        y_pred_l2 = ckpt["pred_l2"]
        cm1 = confusion_matrix(y_true_l1, y_pred_l1, labels=[0, 1]).astype(float)
        cm2 = confusion_matrix(y_true_l2, y_pred_l2, labels=[0, 1, 2]).astype(float)
        l1_counts.append(cm1)
        l2_counts.append(cm2)
        l1_norms.append(cm1 / np.clip(cm1.sum(axis=1, keepdims=True), 1, None))
        l2_norms.append(cm2 / np.clip(cm2.sum(axis=1, keepdims=True), 1, None))
    return {
        "l1_count_mean": np.mean(np.stack(l1_counts), axis=0),
        "l1_count_std": np.std(np.stack(l1_counts), axis=0),
        "l1_norm_mean": np.mean(np.stack(l1_norms), axis=0),
        "l1_norm_std": np.std(np.stack(l1_norms), axis=0),
        "l2_count_mean": np.mean(np.stack(l2_counts), axis=0),
        "l2_count_std": np.std(np.stack(l2_counts), axis=0),
        "l2_norm_mean": np.mean(np.stack(l2_norms), axis=0),
        "l2_norm_std": np.std(np.stack(l2_norms), axis=0),
    }


def plot_avg_confusion_blue(conf_stats, out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    im1 = ax1.imshow(conf_stats["l1_norm_mean"], cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for i in range(2):
        for j in range(2):
            pct = conf_stats["l1_norm_mean"][i, j]
            pct_std = conf_stats["l1_norm_std"][i, j]
            count_mean = conf_stats["l1_count_mean"][i, j]
            txt = f"{count_mean:.1f}\n({pct:.1%}±{pct_std:.1%})"
            ax1.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color="white" if pct > 0.5 else "black",
            )
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(L1_NAMES, fontweight="bold")
    ax1.set_yticklabels(L1_NAMES, fontweight="bold")
    ax1.set_xlabel("Predicted", fontweight="bold")
    ax1.set_ylabel("True", fontweight="bold")
    ax1.set_title("L1: Fault Detection", fontweight="bold", pad=8)
    add_panel_label(ax1, "(a)")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    im2 = ax2.imshow(conf_stats["l2_norm_mean"], cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for i in range(3):
        for j in range(3):
            pct = conf_stats["l2_norm_mean"][i, j]
            pct_std = conf_stats["l2_norm_std"][i, j]
            count_mean = conf_stats["l2_count_mean"][i, j]
            txt = f"{count_mean:.1f}\n({pct:.1%}±{pct_std:.1%})"
            ax2.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white" if pct > 0.5 else "black",
            )
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(L2_NAMES, fontsize=9, fontweight="bold")
    ax2.set_yticklabels(L2_NAMES, fontsize=9, fontweight="bold")
    ax2.set_xlabel("Predicted", fontweight="bold")
    ax2.set_ylabel("True", fontweight="bold")
    ax2.set_title("L2: Scenario", fontweight="bold", pad=8)
    add_panel_label(ax2, "(b)")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    for ax in [ax1, ax2]:
        sci_ax_style(ax)
    plt.tight_layout()
    plt.savefig(out_dir / "Fig2_confusion_matrices_avg_blue.pdf", format="pdf")
    plt.savefig(out_dir / "Fig2_confusion_matrices_avg_blue.png", dpi=600)
    plt.close(fig)


def collect_tsne_payload(runs):
    embeds = []
    y_l1 = []
    y_l2 = []
    resistances = []
    for run in runs:
        ckpt = run["checkpoint"]
        embeds.append(ckpt["embed_np"])
        y_l1.append(ckpt["labels_test"]["L1"])
        y_l2.append(ckpt["labels_test"]["L2"])
        resistances.extend([s.get("resistance") for s in ckpt["samples_test_info"]])
    embed = np.vstack(embeds)
    y_l1 = np.concatenate(y_l1)
    y_l2 = np.concatenate(y_l2)
    perp = min(30, max(5, len(embed) - 1))
    Z = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perp,
        learning_rate="auto",
        init="pca",
    ).fit_transform(embed)
    return {"Z": Z, "y_l1": y_l1, "y_l2": y_l2, "resistances": np.asarray(resistances, dtype=object)}


def plot_avg_tsne_style(tsne_payload, out_dir: Path):
    Z = tsne_payload["Z"]
    y_l1 = tsne_payload["y_l1"]
    y_l2 = tsne_payload["y_l2"]
    resistances = tsne_payload["resistances"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.32)

    color_map_l1 = {0: "#4A90D9", 1: "#E74C3C"}
    marker_map_l1 = {0: "o", 1: "^"}
    name_map_l1 = {0: "Normal", 1: "Fault"}
    for label in [0, 1]:
        m = y_l1 == label
        if m.sum() > 0:
            ax1.scatter(
                Z[m, 0],
                Z[m, 1],
                c=color_map_l1[label],
                marker=marker_map_l1[label],
                s=70,
                alpha=0.78,
                edgecolors="white",
                linewidths=0.45,
                label=name_map_l1[label],
                zorder=3,
            )
    ax1.set_xlabel("t-SNE Dim 1", fontweight="bold")
    ax1.set_ylabel("t-SNE Dim 2", fontweight="bold")
    ax1.set_title("(a) Fault Detection", fontweight="bold", pad=10)
    leg1 = ax1.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=12, markerscale=1.2)
    leg1.get_frame().set_linewidth(1.2)
    for t in leg1.get_texts():
        t.set_fontweight("bold")
    ax1.grid(True, alpha=0.15, linewidth=0.5, linestyle="--", zorder=0)

    cmaps = {1: "Reds", 2: "Blues"}
    marker_l2 = {0: "o", 1: "^", 2: "s"}
    all_r = [np.log10(float(r)) for r in resistances if r is not None and float(r) > 0]
    r_min = min(all_r) if all_r else -2
    r_max = max(all_r) if all_r else 1
    handles = []

    mn = y_l2 == 0
    if mn.sum() > 0:
        ax2.scatter(
            Z[mn, 0],
            Z[mn, 1],
            c="#B0B0B0",
            marker="o",
            s=55,
            alpha=0.68,
            edgecolors="white",
            linewidths=0.35,
            zorder=2,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#B0B0B0",
                markersize=10,
                label="Normal",
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
        )

    for sc_id in [1, 2]:
        mask = y_l2 == sc_id
        if mask.sum() == 0:
            continue
        cmap = plt.get_cmap(cmaps[sc_id])
        mk = marker_l2[sc_id]
        for i in np.where(mask)[0]:
            r = resistances[i]
            norm_val = (
                np.clip(1.0 - (np.log10(float(r)) - r_min) / (r_max - r_min + 1e-8), 0.2, 0.9)
                if r is not None and float(r) > 0
                else 0.5
            )
            ax2.scatter(
                Z[i, 0],
                Z[i, 1],
                c=[cmap(norm_val)],
                marker=mk,
                s=65,
                alpha=0.78,
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )
        handles.append(
            Line2D(
                [0],
                [0],
                marker=mk,
                color="w",
                markerfacecolor=cmap(0.7),
                markersize=10,
                label=L2_NAMES[sc_id],
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
        )

    leg2 = ax2.legend(handles=handles, frameon=True, fancybox=False, edgecolor="black", fontsize=11)
    leg2.get_frame().set_linewidth(1.2)
    for t in leg2.get_texts():
        t.set_fontweight("bold")
    ax2.set_xlabel("t-SNE Dim 1", fontweight="bold")
    ax2.set_ylabel("t-SNE Dim 2", fontweight="bold")
    ax2.set_title("(b) Scenario & Severity", fontweight="bold", pad=10)
    ax2.grid(True, alpha=0.15, linewidth=0.5, linestyle="--", zorder=0)

    for ax in [ax1, ax2]:
        sci_ax_style(ax)
    plt.tight_layout()
    plt.savefig(out_dir / "Fig3_tsne_avg_style.pdf", format="pdf")
    plt.savefig(out_dir / "Fig3_tsne_avg_style.png", dpi=600)
    plt.close(fig)


def collect_avg_fpr(runs):
    class_names = ["Normal", "Charging short-circuit", "Full-SOC Resting Short-circuit"]
    default_fpr = {name: [] for name in class_names}
    curves = {name: [] for name in class_names}
    for run in runs:
        ckpt = run["checkpoint"]
        report = run["report"]
        y_true = ckpt["labels_test"]["L2"]
        probs = ckpt["probs_final"]
        for cls_id, cls_name in enumerate(class_names):
            for thr in THRESHOLDS:
                pred_bin = (probs[:, cls_id] >= thr).astype(int)
                true_bin = (y_true == cls_id).astype(int)
                fp = int(((pred_bin == 1) & (true_bin == 0)).sum())
                tn = int(((pred_bin == 0) & (true_bin == 0)).sum())
                val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                curves[cls_name].append((thr, val))
            default_fpr[cls_name].append(float(report["safety_metrics"][cls_name]["FPR"]))

    curve_summary = {}
    for cls_name in class_names:
        cls_vals = np.asarray([v for _, v in curves[cls_name]], dtype=float).reshape(len(runs), len(THRESHOLDS))
        curve_summary[cls_name] = {
            "mean": np.mean(cls_vals, axis=0),
            "std": np.std(cls_vals, axis=0),
            "default_mean": float(np.mean(default_fpr[cls_name])),
            "default_std": float(np.std(default_fpr[cls_name])),
        }
    return curve_summary


def plot_avg_fpr(curve_summary, out_dir: Path):
    colors = {"Normal": "#7F8C8D", "Charging short-circuit": "#E74C3C", "Full-SOC Resting Short-circuit": "#2E86C1"}
    linestyles = {"Normal": "-", "Charging short-circuit": "--", "Full-SOC Resting Short-circuit": "-."}
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for cls_name in L2_NAMES:
        mean_curve = curve_summary[cls_name]["mean"]
        std_curve = curve_summary[cls_name]["std"]
        ax.plot(
            THRESHOLDS,
            mean_curve,
            color=colors[cls_name],
            linestyle=linestyles[cls_name],
            lw=2.3,
            label=f"{cls_name} (FPR={curve_summary[cls_name]['default_mean']:.3f}±{curve_summary[cls_name]['default_std']:.3f})",
        )
        ax.fill_between(
            THRESHOLDS,
            np.clip(mean_curve - std_curve, 0, 1),
            np.clip(mean_curve + std_curve, 0, 1),
            color=colors[cls_name],
            alpha=0.12,
        )
    ax.axhline(y=0.05, color="gray", linestyle=":", linewidth=1.2, alpha=0.8, label="FPR = 0.05 ref.")
    ax.set_xlabel("Confidence Threshold", fontweight="bold")
    ax.set_ylabel("False Positive Rate (FPR)", fontweight="bold")
    ax.set_xlim([0, 1.0])
    ymax = max(0.15, max(float(np.max(curve_summary[k]["mean"] + curve_summary[k]["std"])) for k in L2_NAMES) * 1.12)
    ax.set_ylim([-0.02, ymax])
    leg = ax.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=9, loc="upper right")
    leg.get_frame().set_linewidth(1.2)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    ax.grid(True, alpha=0.15, linewidth=0.5, linestyle="--", zorder=0)
    sci_ax_style(ax)
    plt.tight_layout()
    plt.savefig(out_dir / "Fig_FPR_avg_curves.pdf", format="pdf")
    plt.savefig(out_dir / "Fig_FPR_avg_curves.png", dpi=600)
    plt.close(fig)

    rows = []
    for cls_name in L2_NAMES:
        rows.append(
            {
                "class": cls_name,
                "default_fpr_mean": curve_summary[cls_name]["default_mean"],
                "default_fpr_std": curve_summary[cls_name]["default_std"],
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "fpr_avg_summary.csv", index=False, encoding="utf-8-sig")


def collect_avg_ablation(runs):
    module_order = ["Base MLP", "+ Attention", "+ Contrastive", "+ MSTF", "Full System"]
    data = {name: [] for name in module_order}
    for run in runs:
        abl = run["report"].get("Ablation", {})
        for name in module_order:
            data[name].append(float(abl[name]))
    rows = []
    for name in module_order:
        stats = summarize(data[name])
        rows.append({"module": name, **stats})
    return rows


def plot_avg_ablation(rows, out_dir: Path):
    modules = [r["module"] for r in rows]
    means = [r["mean"] for r in rows]
    stds = [r["std"] for r in rows]
    colors = ["#BDBDBD", "#B0BEC5", "#90CAF9", "#64B5F6", "#E53935"]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    x = np.arange(len(modules))
    bars = ax.bar(x, means, yerr=stds, width=0.55, color=colors, edgecolor="black", linewidth=0.8, capsize=4)
    for bar, mean_val, std_val in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean_val + std_val + 0.006,
            f"{mean_val:.3f}\n±{std_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(modules, fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_title("Averaged Ablation Study", fontweight="bold", pad=10)
    ax.set_ylim(max(0, min(means) - 0.1), 1.08)
    ax.grid(True, alpha=0.15, axis="y", linestyle="--")
    sci_ax_style(ax)
    plt.tight_layout()
    plt.savefig(out_dir / "Fig7_ablation_avg.pdf", format="pdf")
    plt.savefig(out_dir / "Fig7_ablation_avg.png", dpi=600)
    plt.close(fig)

    pd.DataFrame(rows).to_csv(out_dir / "ablation_avg_summary.csv", index=False, encoding="utf-8-sig")


def plot_combined_2x2_avg(tsne_payload, conf_stats, curve_summary, out_dir: Path):
    Z = tsne_payload["Z"]
    y_l1 = tsne_payload["y_l1"]
    y_l2 = tsne_payload["y_l2"]
    resistances = tsne_payload["resistances"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_fpr, ax_cm = axes[0]
    ax_tsne1, ax_tsne2 = axes[1]
    fig.subplots_adjust(hspace=0.32, wspace=0.30)

    colors = {"Normal": "#7F8C8D", "Charging short-circuit": "#E74C3C", "Full-SOC Resting Short-circuit": "#2E86C1"}
    linestyles = {"Normal": "-", "Charging short-circuit": "--", "Full-SOC Resting Short-circuit": "-."}
    for cls_name in L2_NAMES:
        mean_curve = curve_summary[cls_name]["mean"]
        std_curve = curve_summary[cls_name]["std"]
        ax_fpr.plot(
            THRESHOLDS,
            mean_curve,
            color=colors[cls_name],
            linestyle=linestyles[cls_name],
            lw=2.4,
            label=f"{cls_name} ({curve_summary[cls_name]['default_mean']:.3f}±{curve_summary[cls_name]['default_std']:.3f})",
        )
        ax_fpr.fill_between(
            THRESHOLDS,
            np.clip(mean_curve - std_curve, 0, 1),
            np.clip(mean_curve + std_curve, 0, 1),
            color=colors[cls_name],
            alpha=0.12,
        )
    ax_fpr.axhline(y=0.05, color="gray", linestyle=":", linewidth=1.2, alpha=0.7, label="FPR = 0.05 ref.")
    ax_fpr.set_xlabel("Confidence Threshold", fontweight="bold")
    ax_fpr.set_ylabel("False Positive Rate (FPR)", fontweight="bold")
    ax_fpr.set_xlim([0, 1.0])
    ymax = max(0.15, max(float(np.max(curve_summary[k]["mean"] + curve_summary[k]["std"])) for k in L2_NAMES) * 1.12)
    ax_fpr.set_ylim([-0.02, ymax])
    leg_fpr = ax_fpr.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8.5, loc="upper right")
    leg_fpr.get_frame().set_linewidth(1.2)
    for t in leg_fpr.get_texts():
        t.set_fontweight("bold")
    ax_fpr.grid(True, alpha=0.15, linewidth=0.5, linestyle="--", zorder=0)
    add_panel_label(ax_fpr, "(a)")

    im = ax_cm.imshow(conf_stats["l2_norm_mean"], cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for i in range(3):
        for j in range(3):
            pct = conf_stats["l2_norm_mean"][i, j]
            pct_std = conf_stats["l2_norm_std"][i, j]
            count_mean = conf_stats["l2_count_mean"][i, j]
            txt = f"{count_mean:.1f}\n({pct:.1%}±{pct_std:.1%})"
            ax_cm.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=8.8,
                fontweight="bold",
                color="white" if pct > 0.5 else "black",
            )
    ax_cm.set_xticks(range(3))
    ax_cm.set_yticks(range(3))
    ax_cm.set_xticklabels(L2_NAMES, fontsize=9, fontweight="bold")
    ax_cm.set_yticklabels(L2_NAMES, fontsize=9, fontweight="bold")
    ax_cm.set_xlabel("Predicted", fontweight="bold")
    ax_cm.set_ylabel("True", fontweight="bold")
    fig.colorbar(im, ax=ax_cm, fraction=0.046)
    add_panel_label(ax_cm, "(b)")

    color_map_l1 = {0: "#4A90D9", 1: "#E74C3C"}
    marker_map_l1 = {0: "o", 1: "^"}
    name_map_l1 = {0: "Normal", 1: "Fault"}
    for label in [0, 1]:
        mask = y_l1 == label
        if mask.sum() > 0:
            ax_tsne1.scatter(
                Z[mask, 0],
                Z[mask, 1],
                c=color_map_l1[label],
                marker=marker_map_l1[label],
                s=62,
                alpha=0.76,
                edgecolors="white",
                linewidths=0.4,
                label=name_map_l1[label],
                zorder=3,
            )
    ax_tsne1.set_xlabel("t-SNE Dim 1", fontweight="bold")
    ax_tsne1.set_ylabel("t-SNE Dim 2", fontweight="bold")
    leg_t1 = ax_tsne1.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=10, markerscale=1.1)
    leg_t1.get_frame().set_linewidth(1.2)
    for t in leg_t1.get_texts():
        t.set_fontweight("bold")
    ax_tsne1.grid(True, alpha=0.15, linewidth=0.5, linestyle="--", zorder=0)
    add_panel_label(ax_tsne1, "(c)")

    cmaps = {1: "Reds", 2: "Blues"}
    marker_l2 = {0: "o", 1: "^", 2: "s"}
    all_r = [np.log10(float(r)) for r in resistances if r is not None and float(r) > 0]
    r_min = min(all_r) if all_r else -2
    r_max = max(all_r) if all_r else 1
    handles = []
    mn = y_l2 == 0
    if mn.sum() > 0:
        ax_tsne2.scatter(
            Z[mn, 0],
            Z[mn, 1],
            c="#B0B0B0",
            marker="o",
            s=52,
            alpha=0.65,
            edgecolors="white",
            linewidths=0.35,
            zorder=2,
        )
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#B0B0B0", markersize=9, label="Normal", markeredgecolor="white", markeredgewidth=0.5))
    for sc_id in [1, 2]:
        mask = y_l2 == sc_id
        if mask.sum() == 0:
            continue
        cmap = plt.get_cmap(cmaps[sc_id])
        mk = marker_l2[sc_id]
        for i in np.where(mask)[0]:
            r = resistances[i]
            norm_val = (
                np.clip(1.0 - (np.log10(float(r)) - r_min) / (r_max - r_min + 1e-8), 0.2, 0.9)
                if r is not None and float(r) > 0
                else 0.5
            )
            ax_tsne2.scatter(
                Z[i, 0],
                Z[i, 1],
                c=[cmap(norm_val)],
                marker=mk,
                s=58,
                alpha=0.76,
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )
        handles.append(Line2D([0], [0], marker=mk, color="w", markerfacecolor=cmap(0.7), markersize=9, label=L2_NAMES[sc_id], markeredgecolor="white", markeredgewidth=0.5))
    leg_t2 = ax_tsne2.legend(handles=handles, frameon=True, fancybox=False, edgecolor="black", fontsize=10)
    leg_t2.get_frame().set_linewidth(1.2)
    for t in leg_t2.get_texts():
        t.set_fontweight("bold")
    ax_tsne2.set_xlabel("t-SNE Dim 1", fontweight="bold")
    ax_tsne2.set_ylabel("t-SNE Dim 2", fontweight="bold")
    ax_tsne2.grid(True, alpha=0.15, linewidth=0.5, linestyle="--", zorder=0)
    add_panel_label(ax_tsne2, "(d)")

    for ax in [ax_fpr, ax_tsne1, ax_tsne2, ax_cm]:
        sci_ax_style(ax)
    plt.tight_layout()
    plt.savefig(out_dir / "Fig_combined_2x2_avg_publication.pdf", format="pdf")
    plt.savefig(out_dir / "Fig_combined_2x2_avg_publication.png", dpi=600)
    plt.close(fig)


def write_tsne_reviewer_note(out_dir: Path):
    note = (
        "Assessment for Q1-style review:\n\n"
        "1. The repeated pooled t-SNE is acceptable as a qualitative support figure, not as the primary evidence of model validity.\n"
        "2. In the manuscript text, it should be described as a visualization of embedding consistency across repeated held-out evaluations.\n"
        "3. It should not be used to claim statistical significance, robustness by itself, or strict class separability guarantees.\n"
        "4. The primary evidence should remain the repeated mean ± std quantitative results, class-wise safety metrics, and averaged confusion matrices.\n"
        "5. If space is tight, this figure is safer as a secondary or supplementary figure than as the core performance figure.\n"
    )
    with open(out_dir / "tsne_q1_reviewer_note.md", "w", encoding="utf-8") as f:
        f.write(note)


def main():
    setup_fonts()
    runs = load_runs()
    plot_avg_training_curves(runs, AVG_DIR)
    conf_stats = collect_confusions(runs)
    tsne_payload = collect_tsne_payload(runs)
    curve_summary = collect_avg_fpr(runs)
    ablation_rows = collect_avg_ablation(runs)

    plot_avg_confusion_blue(conf_stats, AVG_DIR)
    plot_avg_tsne_style(tsne_payload, AVG_DIR)
    plot_avg_fpr(curve_summary, AVG_DIR)
    plot_avg_ablation(ablation_rows, AVG_DIR)
    plot_combined_2x2_avg(tsne_payload, conf_stats, curve_summary, AVG_DIR)
    write_tsne_reviewer_note(AVG_DIR)

    print("=" * 70)
    print("Average publication figures generated")
    print(f"Runs used: {len(runs)}")
    print(f"Output directory: {AVG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
