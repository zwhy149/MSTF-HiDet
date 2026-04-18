"""
Generate combined figure: (a) Confusion Matrix, (b) t-SNE, (c) FPR vs Threshold
Excludes 20Ω and 50Ω samples.
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

# ── Font setup (Times New Roman preferred) ──
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
        'legend.fontsize': 11,
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
    ax.tick_params(axis='both', which='both', direction='in',
                   top=False, right=False, width=1.2)
    ax.tick_params(axis='both', which='minor', direction='in',
                   top=False, right=False, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')


def add_panel_label(ax, label_text):
    """Add (a)/(b)/(c) label: black Times New Roman, transparent background."""
    ax.text(0.02, 0.98, label_text,
            transform=ax.transAxes, fontsize=16, fontweight='bold',
            va='top', ha='left', color='black',
            fontfamily='serif',
            bbox=dict(facecolor='none', edgecolor='none', pad=1))


# ── Load checkpoint ──
REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_PATH = REPO_ROOT / 'checkpoints' / 'checkpoint_results.pkl'
OUTPUT_DIR = REPO_ROOT / 'results' / 'supplementary' / 'detection_delay'

print("Loading checkpoint...")
with open(CKPT_PATH, 'rb') as f:
    ckpt = pickle.load(f)

embed_np    = ckpt['embed_np']
labels_test = ckpt['labels_test']
pred_l2     = ckpt['pred_l2']
pred_l1     = ckpt['pred_l1']
probs_final = ckpt['probs_final']
samples_info = ckpt['samples_test_info']
L2_NAMES    = ckpt['L2_NAMES']   # ['Normal', 'Charging Short', 'Rest-Stage Short']

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

# Apply filter
embed_f   = embed_np[keep]
y_true_l2 = labels_test['L2'][keep]
y_true_l1 = labels_test['L1'][keep]
y_pred_l2 = pred_l2[keep]
y_pred_l1 = pred_l1[keep]
probs_f   = probs_final[keep]
samples_f = [s for i, s in enumerate(samples_info) if keep[i]]

# ── t-SNE ──
print("  Computing t-SNE on filtered data...")
perp = min(30, max(5, len(embed_f) - 1))
Z = TSNE(n_components=2, random_state=42, perplexity=perp,
         learning_rate='auto', init='pca').fit_transform(embed_f)

# ── Confusion matrix ──
present = sorted(set(y_true_l2) | set(y_pred_l2))
cm  = confusion_matrix(y_true_l2, y_pred_l2, labels=present)
cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

# ── Per-class FPR at the default threshold ──
fpr_at_default = {}
for idx_cls, cls in enumerate(present):
    tp = cm[idx_cls, idx_cls]
    fn = cm[idx_cls, :].sum() - tp
    fp = cm[:, idx_cls].sum() - tp
    tn = cm.sum() - tp - fn - fp
    fpr_at_default[cls] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
print(f"  Per-class FPR (default threshold): "
      f"{', '.join(f'{L2_NAMES[c]}={fpr_at_default[c]:.4f}' for c in present)}")

# ── FPR vs confidence threshold (for panel c) ──
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
#  Combined figure:  (a) Confusion Matrix  (b) t-SNE  (c) FPR
# ══════════════════════════════════════════════════════════════
print("  Drawing combined figure...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
fig.subplots_adjust(wspace=0.35)

# ────── (a) L2 Confusion Matrix ──────
im = ax1.imshow(cm_n, cmap='Blues', vmin=0, vmax=1, aspect='equal')
for i in range(len(present)):
    for j in range(len(present)):
        ax1.text(j, i, f'{cm[i,j]}\n({cm_n[i,j]:.1%})',
                 ha='center', va='center', fontsize=10, fontweight='bold',
                 color='white' if cm_n[i,j] > 0.5 else 'black')
used_names = [L2_NAMES[k] for k in present]
ax1.set_xticks(range(len(present)))
ax1.set_yticks(range(len(present)))
ax1.set_xticklabels(used_names, fontsize=10, fontweight='bold')
ax1.set_yticklabels(used_names, fontsize=10, fontweight='bold')
ax1.set_xlabel('Predicted', fontweight='bold')
ax1.set_ylabel('True', fontweight='bold')
fig.colorbar(im, ax=ax1, fraction=0.046)
add_panel_label(ax1, '(a)')

# ────── (b) t-SNE (L2 classes) ──────
color_l2  = {0: '#7F8C8D', 1: '#E74C3C', 2: '#2E86C1'}
marker_l2 = {0: 'o',       1: '^',       2: 's'}
sc_labels = {0: 'Normal',  1: 'Charging Short', 2: 'Rest-Stage Short'}

handles = []
for cls in present:
    mask = y_true_l2 == cls
    if mask.sum() == 0:
        continue
    ax2.scatter(Z[mask, 0], Z[mask, 1],
                c=color_l2.get(cls, '#333'),
                marker=marker_l2.get(cls, 'o'),
                s=85, alpha=0.85, edgecolors='white', linewidths=0.5,
                zorder=3)
    handles.append(Line2D([0], [0],
                          marker=marker_l2.get(cls, 'o'), color='w',
                          markerfacecolor=color_l2.get(cls, '#333'),
                          markersize=10,
                          label=sc_labels.get(cls, f'Class {cls}'),
                          markeredgecolor='white', markeredgewidth=0.5))

ax2.set_xlabel('t-SNE Dim 1', fontweight='bold')
ax2.set_ylabel('t-SNE Dim 2', fontweight='bold')
leg2 = ax2.legend(handles=handles, frameon=True, fancybox=False,
                  edgecolor='black', fontsize=10)
leg2.get_frame().set_linewidth(1.2)
for t in leg2.get_texts():
    t.set_fontweight('bold')
ax2.grid(True, alpha=0.15, linewidth=0.5, linestyle='--', zorder=0)
add_panel_label(ax2, '(b)')

# ────── (c) FPR vs Confidence Threshold ──────
line_colors = ['#7F8C8D', '#E74C3C', '#2E86C1']
line_styles = ['-', '--', '-.']

for idx, cls in enumerate(present):
    ax3.plot(thresholds, fpr_curves[cls],
             color=line_colors[idx % len(line_colors)],
             linestyle=line_styles[idx % len(line_styles)],
             lw=2.5,
             label=f'{L2_NAMES[cls]} (FPR$_{{opt}}$={fpr_at_default[cls]:.3f})')

# Reference line
ax3.axhline(y=0.05, color='gray', linestyle=':', linewidth=1.2, alpha=0.7,
            label='FPR = 0.05 ref.')
ax3.set_xlabel('Confidence Threshold', fontweight='bold')
ax3.set_ylabel('False Positive Rate (FPR)', fontweight='bold')
ax3.set_xlim([0, 1.0])
ax3.set_ylim([-0.02, max(0.15, float(max(c.max() for c in fpr_curves.values())) * 1.2)])
leg3 = ax3.legend(frameon=True, fancybox=False, edgecolor='black',
                  fontsize=10, loc='upper right')
leg3.get_frame().set_linewidth(1.2)
for t in leg3.get_texts():
    t.set_fontweight('bold')
ax3.grid(True, alpha=0.15, linewidth=0.5, linestyle='--', zorder=0)
add_panel_label(ax3, '(c)')

# ── Final styling ──
for ax in [ax1, ax2, ax3]:
    sci_ax_style(ax)

plt.tight_layout()
plt.savefig(os.path.join(str(OUTPUT_DIR), 'Fig_combined_CM_tSNE_FPR.pdf'), format='pdf')
plt.savefig(os.path.join(str(OUTPUT_DIR), 'Fig_combined_CM_tSNE_FPR.png'), dpi=600)
plt.close()

print(f"\n  Saved: {os.path.join(str(OUTPUT_DIR), 'Fig_combined_CM_tSNE_FPR.pdf')}")
print(f"  Saved: {os.path.join(str(OUTPUT_DIR), 'Fig_combined_CM_tSNE_FPR.png')}")
print("  Done!")
