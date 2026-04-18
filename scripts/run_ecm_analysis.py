"""
Physics-Informed 2RC ECM Fitting with Confidence Intervals — 1Ω ISC
=====================================================================
Reviewer: "The RMSE of 5.56 mV is reported without confidence intervals
or prediction bounds, undermining the claimed detection reliability."

Approach:
1. Fit a 2RC Equivalent Circuit Model to the real 1Ω ISC experiment data
2. scipy.optimize.curve_fit → parameter covariance → 95% CI / prediction bands
3. Bootstrap resampling → RMSE 95% confidence interval
4. Generate 4-panel publication figure

ECM Model:
  V(t) = V0 + α·t + β·t² + γ·(1-exp(-t/τ1)) + δ·(1-exp(-t/τ2))
         − Δ_isc·(1-exp(-(t-t_f)/τ_isc))  [after fault onset]
"""

import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.filterwarnings('ignore')

# ── Font ──
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
        'font.size': 12, 'font.weight': 'bold',
        'axes.labelsize': 14, 'axes.titlesize': 15,
        'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': False, 'ytick.right': False,
        'xtick.major.size': 5, 'ytick.major.size': 5,
        'xtick.minor.size': 3, 'ytick.minor.size': 3,
        'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'xtick.minor.width': 0.8, 'ytick.minor.width': 0.8,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'axes.linewidth': 1.5, 'axes.edgecolor': 'black',
        'legend.fontsize': 10, 'legend.frameon': True,
        'legend.edgecolor': 'black', 'legend.fancybox': False,
        'legend.framealpha': 1.0,
        'figure.dpi': 300, 'savefig.dpi': 600,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
        'axes.unicode_minus': False, 'mathtext.fontset': 'dejavuserif',
    })
    print(f"  Font: {chosen}")
    return chosen

FONT_NAME = setup_fonts()


def sci_ax_style(ax):
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
    ax.text(0.04, 0.94, label_text,
            transform=ax.transAxes, fontsize=16, fontweight='bold',
            va='top', ha='left', color='black', fontfamily='serif')


# ── Paths ──
REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA_ROOT = Path(os.environ.get('MSTF_REAL_DATA', r'D:\AE\dataset_holographic'))
REAL_FILE = str(REAL_DATA_ROOT / '充电短路' / '1Ω充电短路.xlsx')
OUT_DIR   = str(REPO_ROOT / 'results' / 'supplementary' / 'ecm_confidence_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ──
print("Loading experiment data (1Ω ISC)...")
df = pd.read_excel(REAL_FILE)
t_all = df['Time'].values.astype(float)
v_all = df['Voltage'].values.astype(float)
lab   = df['Label'].values.astype(int)

fault_idx = int(np.argmax(lab == 1))
t_fault = t_all[fault_idx]
print(f"  Points: {len(t_all)}, Fault onset: t={t_fault:.0f} s")

# ══════════════════════════════════════════════════════════════
#  2RC ECM Model
# ══════════════════════════════════════════════════════════════
def ecm_model(t, V0, alpha, beta, gamma, tau1, delta, tau2, D_isc, tau_isc):
    v = (V0 + alpha * t + beta * t**2
         + gamma * (1 - np.exp(-t / tau1))
         + delta * (1 - np.exp(-t / tau2)))
    mask = t >= t_fault
    dt = np.where(mask, t - t_fault, 0.0)
    v = v - np.where(mask, D_isc * (1 - np.exp(-dt / tau_isc)), 0.0)
    return v


# Initial guess
v0_init = v_all[0]
slope_init = (v_all[fault_idx] - v_all[0]) / t_fault
p0 = [v0_init, slope_init * 0.8, 1e-8, 0.02, 10.0, 0.01, 300.0, 0.01, 50.0]
lb = [v0_init - 0.1, 0,    -1e-5, -0.5,   1.0, -0.5,   10.0, 0.0,   1.0]
ub = [v0_init + 0.1, 1e-2,  1e-5,  0.5, 200.0,  0.5, 2000.0, 0.5, 500.0]

print("\nFitting 2RC ECM model...")
popt, pcov = curve_fit(ecm_model, t_all, v_all, p0=p0,
                       bounds=(lb, ub), maxfev=100000,
                       method='trf', ftol=1e-14, xtol=1e-14, gtol=1e-14)
param_names = ['V0', 'α', 'β', 'γ', 'τ1', 'δ', 'τ2', 'Δ_isc', 'τ_isc']
param_units = ['V', 'V/s', 'V/s²', 'V', 's', 'V', 's', 'V', 's']
perr = np.sqrt(np.diag(pcov))

print("  Fitted parameters:")
for name, val, err, unit in zip(param_names, popt, perr, param_units):
    print(f"    {name:8s} = {val:>14.6g} ± {err:.4g} {unit}")

# ── Residuals ──
v_fit = ecm_model(t_all, *popt)
residuals = v_all - v_fit
resid_mV = residuals * 1000

n_params = len(popt)
dof = len(t_all) - n_params
mse = np.sum(residuals**2) / dof
s_resid = np.sqrt(mse)

RMSE = np.sqrt(np.mean(residuals**2)) * 1000
MAE  = np.mean(np.abs(residuals)) * 1000
maxE = np.max(np.abs(residuals)) * 1000
R2   = 1 - np.sum(residuals**2) / np.sum((v_all - np.mean(v_all))**2)

print(f"\n{'='*60}")
print(f"  RMSE  = {RMSE:.3f} mV")
print(f"  MAE   = {MAE:.3f} mV")
print(f"  MaxAE = {maxE:.3f} mV")
print(f"  R²    = {R2:.8f}")

rmse_pre  = np.sqrt(np.mean(residuals[:fault_idx]**2)) * 1000
rmse_post = np.sqrt(np.mean(residuals[fault_idx:]**2)) * 1000
print(f"  RMSE pre-fault:  {rmse_pre:.3f} mV")
print(f"  RMSE post-fault: {rmse_post:.3f} mV")

# ── Prediction & Confidence bands ──
print("\n  Computing CI and prediction bands...")
eps = 1e-8
J = np.zeros((len(t_all), n_params))
for j in range(n_params):
    p_up = popt.copy(); p_up[j] += eps
    p_dn = popt.copy(); p_dn[j] -= eps
    J[:, j] = (ecm_model(t_all, *p_up) - ecm_model(t_all, *p_dn)) / (2 * eps)

var_pred = np.sum((J @ pcov) * J, axis=1)
se_pred = np.sqrt(np.maximum(var_pred, 0))
t_crit = stats.t.ppf(0.975, df=dof)

ci_lo = v_fit - t_crit * se_pred
ci_hi = v_fit + t_crit * se_pred
pi_lo = v_fit - t_crit * np.sqrt(var_pred + mse)
pi_hi = v_fit + t_crit * np.sqrt(var_pred + mse)

ci_cov = np.mean((v_all >= ci_lo) & (v_all <= ci_hi))
pi_cov = np.mean((v_all >= pi_lo) & (v_all <= pi_hi))
print(f"  95% CI coverage:         {ci_cov*100:.1f}%")
print(f"  95% Prediction coverage: {pi_cov*100:.1f}%")

# ── Bootstrap RMSE CI ──
print("  Bootstrap RMSE CI (n=10000)...")
np.random.seed(42)
n_boot = 10000
boot_rmse = np.zeros(n_boot)
n_pts = len(residuals)
for b in range(n_boot):
    idx = np.random.choice(n_pts, size=n_pts, replace=True)
    boot_rmse[b] = np.sqrt(np.mean(residuals[idx]**2)) * 1000

ci_lo_b = np.percentile(boot_rmse, 2.5)
ci_hi_b = np.percentile(boot_rmse, 97.5)
print(f"  RMSE = {RMSE:.3f} mV  [95% CI: {ci_lo_b:.3f} – {ci_hi_b:.3f} mV]")

# Normality
_, p_sw = stats.shapiro(residuals[::10])
_, p_ks = stats.kstest(residuals / np.std(residuals), 'norm')
print(f"  Shapiro-Wilk p={p_sw:.4f}, KS p={p_ks:.4f}")

# ── Save stats ──
stats_path = os.path.join(OUT_DIR, 'fitting_statistics.txt')
with open(stats_path, 'w', encoding='utf-8') as f:
    f.write("Physics-Informed 2RC ECM Fitting Analysis — 1 Ohm ISC\n")
    f.write("=" * 58 + "\n\n")
    f.write("Model: V(t) = V0 + a*t + b*t^2 + g*(1-exp(-t/t1)) + d*(1-exp(-t/t2))\n")
    f.write("       - D_isc*(1-exp(-(t-tf)/t_isc))   [t >= tf]\n\n")
    f.write(f"Data points:        {len(t_all)}\n")
    f.write(f"Fault onset:        t = {t_fault:.0f} s\n")
    f.write(f"Parameters:         {n_params}\n")
    f.write(f"DoF:                {dof}\n\n")
    f.write("Fitted Parameters:\n")
    for name, val, err, unit in zip(param_names, popt, perr, param_units):
        f.write(f"  {name:8s} = {val:>14.6g}  +/- {err:.4g} {unit}\n")
    f.write(f"\nGoodness of Fit:\n")
    f.write(f"  RMSE             = {RMSE:.3f} mV\n")
    f.write(f"  RMSE (pre-fault) = {rmse_pre:.3f} mV\n")
    f.write(f"  RMSE (post-fault)= {rmse_post:.3f} mV\n")
    f.write(f"  MAE              = {MAE:.3f} mV\n")
    f.write(f"  Max |error|      = {maxE:.3f} mV\n")
    f.write(f"  R^2              = {R2:.8f}\n\n")
    f.write(f"Bootstrap RMSE (n={n_boot}):\n")
    f.write(f"  Mean   = {np.mean(boot_rmse):.3f} mV\n")
    f.write(f"  95% CI = [{ci_lo_b:.3f}, {ci_hi_b:.3f}] mV\n\n")
    f.write(f"Band Coverage:\n")
    f.write(f"  95% Confidence band:  {ci_cov*100:.1f}%\n")
    f.write(f"  95% Prediction band:  {pi_cov*100:.1f}%\n\n")
    f.write(f"Normality Tests:\n")
    f.write(f"  Shapiro-Wilk p = {p_sw:.4f}\n")
    f.write(f"  KS test p      = {p_ks:.4f}\n")
print(f"  Saved: {stats_path}")


# ══════════════════════════════════════════════════════════════
#  2×2 Figure
# ══════════════════════════════════════════════════════════════
print("\n  Drawing combined figure...")
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
ax_fit, ax_hist = axes[0]
ax_resid, ax_boot = axes[1]
fig.subplots_adjust(hspace=0.35, wspace=0.30)

# ── (a) Voltage fit ──
ax_fit.fill_between(t_all, pi_lo, pi_hi,
                    color='#27AE60', alpha=0.20, label='95% Prediction Band', zorder=1)
ax_fit.fill_between(t_all, ci_lo, ci_hi,
                    color='#3498DB', alpha=0.40, label='95% Confidence Band', zorder=2)
ax_fit.plot(t_all, v_fit, color='#2E86C1', lw=2.0,
            label='Physics-Informed ECM', zorder=3)
ax_fit.plot(t_all, v_all, color='#E74C3C', lw=1.8, linestyle='--',
            label='Experiment Data (1 $\\Omega$)', zorder=4)
ax_fit.axvline(x=t_fault, color='#2C3E50', linestyle=':', lw=1.5, alpha=0.7,
               label=f'ESC Onset ($t$={int(t_fault)} s)')
ax_fit.set_xlabel('Time (s)', fontweight='bold')
ax_fit.set_ylabel('Voltage (V)', fontweight='bold')
ax_fit.set_xlim([0, t_all[-1]])
ax_fit.set_ylim([2.8, 4.0])
ax_fit.set_yticks(np.arange(2.8, 4.01, 0.2))
leg = ax_fit.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=9, loc='upper right')
leg.get_frame().set_linewidth(1.2)
for txt in leg.get_texts(): txt.set_fontweight('bold')
ax_fit.grid(True, alpha=0.15, linewidth=0.5, linestyle='--', zorder=0)
add_panel_label(ax_fit, '(a)')

# ── (b) Residual histogram ──
n_bins = 50
ax_hist.hist(resid_mV, bins=n_bins, density=True,
             color='#3498DB', alpha=0.6, edgecolor='white', linewidth=0.5, zorder=2)
mu_r, std_r = np.mean(resid_mV), np.std(resid_mV)
x_n = np.linspace(resid_mV.min() - 2, resid_mV.max() + 2, 300)
ax_hist.plot(x_n, stats.norm.pdf(x_n, mu_r, std_r), color='#E74C3C', lw=2.5,
             label=f'Normal: $\\mu$={mu_r:.2f}, $\\sigma$={std_r:.2f} mV', zorder=3)
kde = gaussian_kde(resid_mV)
ax_hist.plot(x_n, kde(x_n), color='#27AE60', lw=2.0, linestyle='--',
             label='KDE', zorder=3)
ax_hist.axvline(x=0, color='gray', linestyle=':', lw=1.2, alpha=0.7)
ax_hist.set_xlabel('Residual (mV)', fontweight='bold')
ax_hist.set_ylabel('Density', fontweight='bold')
lg = ax_hist.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=9, loc='upper right')
lg.get_frame().set_linewidth(1.2)
for txt in lg.get_texts(): txt.set_fontweight('bold')
ax_hist.grid(True, alpha=0.15, linewidth=0.5, linestyle='--', zorder=0)
add_panel_label(ax_hist, '(b)')

# ── (c) Residual vs time ──
ax_resid.plot(t_all, resid_mV, color='black', lw=2.0, alpha=1.0, zorder=2,
              label='Residual')
ax_resid.axhline(y=0, color='gray', linestyle=':', lw=1.2, alpha=0.7)
ax_resid.axhline(y=RMSE, color='#2E86C1', linestyle='--', lw=1.5, alpha=0.7,
                 label=f'$\\pm$RMSE = $\\pm${RMSE:.2f} mV')
ax_resid.axhline(y=-RMSE, color='#2E86C1', linestyle='--', lw=1.5, alpha=0.7)
sigma_mV = s_resid * 1000
ax_resid.axhline(y=2*sigma_mV, color='#27AE60', linestyle='-.', lw=1.2, alpha=0.5,
                 label=f'$\\pm 2\\sigma$ = $\\pm${2*sigma_mV:.2f} mV')
ax_resid.axhline(y=-2*sigma_mV, color='#27AE60', linestyle='-.', lw=1.2, alpha=0.5)
ax_resid.axvline(x=t_fault, color='#2C3E50', linestyle=':', lw=1.5, alpha=0.7,
                 label='ESC Onset')
ax_resid.set_xlabel('Time (s)', fontweight='bold')
ax_resid.set_ylabel('Residual (mV)', fontweight='bold')
ax_resid.set_xlim([0, t_all[-1]])
lg = ax_resid.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=9, loc='upper right')
lg.get_frame().set_linewidth(1.2)
for txt in lg.get_texts(): txt.set_fontweight('bold')
ax_resid.grid(True, alpha=0.15, linewidth=0.5, linestyle='--', zorder=0)
add_panel_label(ax_resid, '(c)')

# ── (d) Bootstrap RMSE ──
ax_boot.hist(boot_rmse, bins=60, density=True,
             color='#27AE60', alpha=0.6, edgecolor='white', linewidth=0.5, zorder=2)
mu_b, std_b = np.mean(boot_rmse), np.std(boot_rmse)
x_b = np.linspace(boot_rmse.min() - 0.5, boot_rmse.max() + 0.5, 300)
ax_boot.plot(x_b, stats.norm.pdf(x_b, mu_b, std_b), color='#E74C3C', lw=2.5,
             label=f'$\\mu$={mu_b:.2f} mV', zorder=3)
ax_boot.axvline(x=ci_lo_b, color='#2C3E50', linestyle='--', lw=2.0, zorder=4)
ax_boot.axvline(x=ci_hi_b, color='#2C3E50', linestyle='--', lw=2.0, zorder=4,
                label=f'95% CI: [{ci_lo_b:.2f}, {ci_hi_b:.2f}] mV')
ax_boot.axvline(x=RMSE, color='#E74C3C', linestyle='-', lw=2.0, alpha=0.7, zorder=4,
                label=f'RMSE = {RMSE:.2f} mV')
ax_boot.axvspan(ci_lo_b, ci_hi_b, alpha=0.15, color='#2C3E50', zorder=1)
ax_boot.set_xlabel('RMSE (mV)', fontweight='bold')
ax_boot.set_ylabel('Density', fontweight='bold')
lg = ax_boot.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=9, loc='upper right')
lg.get_frame().set_linewidth(1.2)
for txt in lg.get_texts(): txt.set_fontweight('bold')
ax_boot.grid(True, alpha=0.15, linewidth=0.5, linestyle='--', zorder=0)
add_panel_label(ax_boot, '(d)')

for ax in axes.flat:
    sci_ax_style(ax)

plt.tight_layout()
out_pdf = os.path.join(OUT_DIR, 'Fig_ECM_confidence_analysis.pdf')
out_png = os.path.join(OUT_DIR, 'Fig_ECM_confidence_analysis.png')
plt.savefig(out_pdf, format='pdf')
plt.savefig(out_png, dpi=600)
plt.close()
print(f"\n  Saved: {out_pdf}")
print(f"  Saved: {out_png}")
print("  Done!")
