"""
Generate 2x2 combined figure:
  Top:    (a) FPR vs Threshold    (b) Confusion Matrix
  Bottom: (c) t-SNE (L1)          (d) t-SNE (L2 + Severity)
Only left & bottom axes have ticks; panel labels in top-left corner.
"""

import os, pickle, warnings
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')

# ── Font setup ──
def setup_fonts():
    candidates = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif',
                  'Nimbus Roman', 'FreeSerif', 'serif']
    available = set(f.name for f in font_manager.fontManager.ttflist)
    chosen = 'serif'
    for c in candidates:
        if c in available:
            chosen = c
            break
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [chosen, 'DejaVu Serif'],
        'font.size': 12,
        'font.weight': 'bold',
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': False,
        'ytick.right': False,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'axes.linewidth': 1.5,
        'axes.edgecolor': 'black',
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        'legend.framealpha': 1.0,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.unicode_minus': False,
        'mathtext.fontset': 'dejavuserif',
    })
    print(f"  Font: {chosen}")
    return chosen

FONT_NAME = setup_fonts()


def sci_ax_style(ax):
    """Only left and bottom spines/ticks; hide top and right."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    ax.tick_params(axis='both', which='both', direction='in',
                   top=False, right=False, left=True, bottom=True, width=1.2)
    ax.tick_params(axis='both', which='minor', direction='in',
                   top=False, right=False, left=True, bottom=True, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')


def add_panel_label(ax, label_text):
    """Panel label in top-left, offset to avoid overlap with data."""
    ax.text(-0.02, 1.06, label_text,
            transform=ax.transAxes, fontsize=16, fontweight='bold',
            va='top', ha='left', color='black',
            fontfamily='serif')


# ── Load checkpoint ──
REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_PATH = REPO_ROOT / 'checkpoints' / 'checkpoint_results.pkl'
OUTPUT_DIR = REPO_ROOT / 'results' / 'supplementary' / 'detection_delay'

print("Loading checkpoint...")
with open(CKPT_PATH, 'rb') as f:
    ckpt = pickle.load(f)

embed_np     = ckpt['embed_np']
labels_test  = ckpt['labels_test']
pred_l2      = ckpt['pred_l2']
pred_l1      = ckpt['pred_l1']
probs_final  = ckpt['probs_final']
samples_info = ckpt['samples_test_info']
L2_NAMES     = ckpt['L2_NAMES']

# ── Filter out 20Ω and 50Ω ──
EXCLUDE_R = {20.0, 50.0}
keep = np.ones(len(samples_info), dtype=bool)
for i, s in enumerate(samples_info):
    r = s.get('resistance')
    if r is not None:
        r_val = float(r)
        if any(abs(r_val - ex) < 0.01 for ex in EXCLUDE_R):
            keep[i] = False

n_total   = len(samples_info)
n_kept    = int(keep.sum())
n_removed = n_total - n_kept
print(f"  Total: {n_total} | Kept: {n_kept} | Removed (20Ω/50Ω): {n_removed}")

embed_f   = embed_np[keep]
y_true_l2 = labels_test['L2'][keep]
y_true_l1 = labels_test['L1'][keep]
y_pred_l2 = pred_l2[keep]
y_pred_l1 = pred_l1[keep]
probs_f   = probs_final[keep]
samples_f = [s for i, s in enumerate(samples_info) if keep[i]]
resistances = np.array([s.get('resistance') for s in samples_f], dtype=object)

# ── t-SNE ──
print("  Computing t-SNE...")
perp = min(30, max(5, len(embed_f) - 1))
Z = TSNE(n_components=2, random_state=42, perplexity=perp,
         learning_rate='auto', init='pca').fit_transform(embed_f)

# ── Confusion Matrix ──
present = sorted(set(y_true_l2) | set(y_pred_l2))
cm  = confusion_matrix(y_true_l2, y_pred_l2, labels=present)
cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

# Per-class FPR at default threshold
fpr_at_default = {}
for idx_cls, cls in enumerate(present):
    tp = cm[idx_cls, idx_cls]
    fn = cm[idx_cls, :].sum() - tp
    fp = cm[:, idx_cls].sum() - tp
    tn = cm.sum() - tp - fn - fp
    fpr_at_default[cls] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

# ── FPR vs confidence threshold curves ──
thresholds = np.linspace(0.01, 0.99, 300)
fpr_curves = {}
for cls in present:
    true_bin = (y_true_l2 == cls).astype(int)
    fprs = []
    for t in thresholds:
        pred_bin = (probs_f[:, cls] >= t).astype(int)
        fp = int(((pred_bin == 1) & (true_bin == 0)).sum())
        tn = int(((pred_bin == 0) & (true_bin == 0)).sum())
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    fpr_curves[cls] = np.array(fprs)

# ══════════════════════════════════════════════════════════════
#  2×2 Combined Figure
#  (a) FPR vs Threshold      (b) Confusion Matrix
#  (c) t-SNE L1              (d) t-SNE L2 + Severity
# ══════════════════════════════════════════════════════════════
print("  Drawing 2×2 combined figure...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax_fpr, ax_cm = axes[0]
ax_tsne1, ax_tsne2 = axes[1]
fig.subplots_adjust(hspace=0.32, wspace=0.30)

# ────── (a) FPR vs Confidence Threshold ──────
line_colors = ['#7F8C8D', '#E74C3C', '#2E86C1']
line_styles = ['-', '--', '-.']
for idx, cls in enumerate(present):
    ax_fpr.plot(thresholds, fpr_curves[cls],
                color=line_colors[idx % len(line_colors)],
                linestyle=line_styles[idx % len(line_styles)],
                lw=2.5,
                label=f'{L2_NAMES[cls]} (FPR$_{{opt}}$={fpr_at_default[cls]:.3f})')
ax_fpr.axhline(y=0.05, color='gray', linestyle=':', linewidth=1.2, alpha=0.7,
               label='FPR = 0.05 ref.')
ax_fpr.set_xlabel('Confidence Threshold', fontweight='bold')
ax_fpr.set_ylabel('False Positive Rate (FPR)', fontweight='bold')
ax_fpr.set_xlim([0, 1.0])
ax_fpr.set_ylim([-0.02, max(0.15, float(max(c.max() for c in fpr_curves.values())) * 1.2)])
leg_fpr = ax_fpr.legend(frameon=True, fancybox=False, edgecolor='black',
                        fontsize=9, loc='upper right')
leg_fpr.get_frame().set_linewidth(1.2)
for t in leg_fpr.get_texts():
    t.set_fontweight('bold')
ax_fpr.grid(True, alpha=0.15, linewidth=0.5, linestyle='--', zorder=0)
add_panel_label(ax_fpr, '(a)')

# ────── (b) L2 Confusion Matrix ──────
im = ax_cm.imshow(cm_n, cmap='Blues', vmin=0, vmax=1, aspect='equal')
for i in range(len(present)):
    for j in range(len(present)):
        ax_cm.text(j, i, f'{cm[i,j]}\n({cm_n[i,j]:.1%})',
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   color='white' if cm_n[i, j] > 0.5 else 'black')
used_names = [L2_NAMES[k] for k in present]
ax_cm.set_xticks(range(len(present)))
ax_cm.set_yticks(range(len(present)))
ax_cm.set_xticklabels(used_names, fontsize=9, fontweight='bold')
ax_cm.set_yticklabels(used_names, fontsize=9, fontweight='bold')
ax_cm.set_xlabel('Predicted', fontweight='bold')
ax_cm.set_ylabel('True', fontweight='bold')
# Confusion matrix: show all 4 spines (it's a matrix, not a plot)
ax_cm.tick_params(axis='both', which='both', direction='in',
                  top=False, right=False, left=True, bottom=True)
ax_cm.tick_params(which='minor', left=False, bottom=False)
fig.colorbar(im, ax=ax_cm, fraction=0.046)
add_panel_label(ax_cm, '(b)')

# ────── (c) t-SNE L1: Normal vs Fault ──────
color_l1  = {0: '#4A90D9', 1: '#E74C3C'}
marker_l1 = {0: 'o',       1: '^'}
name_l1   = {0: 'Normal',  1: 'Fault'}
for label in [0, 1]:
    m = y_true_l1 == label
    if m.sum() > 0:
        ax_tsne1.scatter(Z[m, 0], Z[m, 1],
                         c=color_l1[label], marker=marker_l1[label],
                         s=85, alpha=0.85, edgecolors='white', linewidths=0.5,
                         label=name_l1[label], zorder=3)
ax_tsne1.set_xlabel('t-SNE Dim 1', fontweight='bold')
ax_tsne1.set_ylabel('t-SNE Dim 2', fontweight='bold')
leg_t1 = ax_tsne1.legend(frameon=True, fancybox=False, edgecolor='black',
                         fontsize=10, markerscale=1.2)
leg_t1.get_frame().set_linewidth(1.2)
for t in leg_t1.get_texts():
    t.set_fontweight('bold')
ax_tsne1.grid(True, alpha=0.15, linewidth=0.5, linestyle='--', zorder=0)
add_panel_label(ax_tsne1, '(c)')

# ────── (d) t-SNE L2: Scenario & Severity ──────
color_l2  = {0: '#7F8C8D', 1: '#E74C3C', 2: '#2E86C1'}
marker_l2 = {0: 'o',       1: '^',       2: 's'}
sc_labels = {0: 'Normal',  1: 'Charging Short', 2: 'Rest-Stage Short'}
cmaps_l2  = {1: 'Reds', 2: 'Blues'}

all_r = [np.log10(float(s.get('resistance')))
         for s in samples_f
         if s.get('resistance') is not None and float(s.get('resistance')) > 0]
r_min = min(all_r) if all_r else -2
r_max = max(all_r) if all_r else 1

handles_l2 = []
# Normal
mn = y_true_l2 == 0
if mn.sum() > 0:
    ax_tsne2.scatter(Z[mn, 0], Z[mn, 1], c='#B0B0B0', marker='o',
                     s=70, alpha=0.7, edgecolors='white', linewidths=0.4, zorder=2)
    handles_l2.append(Line2D([0], [0], marker='o', color='w',
                             markerfacecolor='#B0B0B0', markersize=10,
                             label='Normal', markeredgecolor='white',
                             markeredgewidth=0.5))
# Fault classes with severity coloring
for sc_id in [1, 2]:
    ms = y_true_l2 == sc_id
    if ms.sum() == 0:
        continue
    cmap = plt.get_cmap(cmaps_l2[sc_id])
    mk = marker_l2[sc_id]
    for i in np.where(ms)[0]:
        r = resistances[i]
        norm_val = (np.clip(1.0 - (np.log10(float(r)) - r_min) / (r_max - r_min + 1e-8), 0.2, 0.9)
                    if r is not None and float(r) > 0 else 0.5)
        ax_tsne2.scatter(Z[i, 0], Z[i, 1], c=[cmap(norm_val)], marker=mk,
                         s=85, alpha=0.85, edgecolors='white', linewidths=0.4, zorder=3)
    handles_l2.append(Line2D([0], [0], marker=mk, color='w',
                             markerfacecolor=cmap(0.7), markersize=10,
                             label=sc_labels[sc_id],
                             markeredgecolor='white', markeredgewidth=0.5))

leg_t2 = ax_tsne2.legend(handles=handles_l2, frameon=True, fancybox=False,
                         edgecolor='black', fontsize=10)
leg_t2.get_frame().set_linewidth(1.2)
for t in leg_t2.get_texts():
    t.set_fontweight('bold')
ax_tsne2.set_xlabel('t-SNE Dim 1', fontweight='bold')
ax_tsne2.set_ylabel('t-SNE Dim 2', fontweight='bold')
ax_tsne2.grid(True, alpha=0.15, linewidth=0.5, linestyle='--', zorder=0)
add_panel_label(ax_tsne2, '(d)')

# ── Apply axis style to all panels ──
for ax in [ax_fpr, ax_tsne1, ax_tsne2]:
    sci_ax_style(ax)

# Confusion matrix: keep left & bottom spines only, hide top & right
for spine in ax_cm.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_color('black')
for label in ax_cm.get_xticklabels() + ax_cm.get_yticklabels():
    label.set_fontweight('bold')

plt.tight_layout()
out_pdf = os.path.join(OUTPUT_DIR, 'Fig_combined_2x2.pdf')
out_png = os.path.join(OUTPUT_DIR, 'Fig_combined_2x2.png')
plt.savefig(out_pdf, format='pdf')
plt.savefig(out_png, dpi=600)
plt.close()

print(f"\n  Saved: {out_pdf}")
print(f"  Saved: {out_png}")
print("  Done!")
