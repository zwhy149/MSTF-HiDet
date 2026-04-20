"""
MSTF-HiDet Fault Detector
Usage:
  python detector.py --file  test_file.xlsx
  python detector.py --dir   /path/to/test_files/
  python detector.py --file  test_file.xlsx --plot

Loads a trained model bundle and performs:
  1. File-level classification: Normal / Charging short-circuit / Full-SOC Resting Short-circuit
  2. Sliding-window detection to locate fault onset
  3. Detection delay measurement
"""

import os, re, sys, pickle, warnings, argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

from scipy.signal import find_peaks, savgol_filter
from scipy.stats import kurtosis, skew

SCENARIO_DISPLAY_NAMES = {
    'Normal': 'Normal',
    '充电短路': 'Charging short-circuit',
    'GZ': 'Full-SOC Resting Short-circuit',
}


# --- Network definition (matches training code) ---
class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention for 1D feature vectors."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x)


class ResidualBlock(nn.Module):
    """FC residual block: Linear->LN->GELU->Dropout->Linear->LN + SE + skip."""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.se = SEBlock(dim, reduction=4)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.drop(self.se(self.net(x))))


class MSTFHiDetNet(nn.Module):
    def __init__(self, feat_dim, num_l2_classes=3, num_l3_classes=8, hidden=256, num_heads=4, dropout=0.20):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.encoder = nn.Sequential(
            ResidualBlock(hidden, dropout),
            ResidualBlock(hidden, dropout),
        )
        self.n_tokens = num_heads
        self.attn = nn.MultiheadAttention(hidden // self.n_tokens, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)
        self.attn_drop = nn.Dropout(dropout)
        self.head_l1 = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(hidden//2, 2))
        self.head_l2 = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(hidden//2, num_l2_classes))
        self.head_l3 = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(hidden//2, num_l3_classes))
        self.projector = nn.Sequential(nn.Linear(hidden, 128), nn.GELU(), nn.Linear(128, 64))

    def forward(self, x, return_embed=False):
        h = self.input_proj(x)
        h = self.encoder(h)
        B = h.size(0)
        h2 = h.view(B, self.n_tokens, -1)
        attn_out, _ = self.attn(h2, h2, h2)
        h = self.attn_norm(h + self.attn_drop(attn_out.reshape(B, -1)))
        o1, o2, o3 = self.head_l1(h), self.head_l2(h), self.head_l3(h)
        proj = self.projector(h)
        if return_embed:
            return o1, o2, o3, proj, h
        return o1, o2, o3, proj


# --- Feature extraction (matches training code) ---
SEGMENT_COUNTS = [4, 8, 16]
CTAM_WINDOWS = [5, 15, 30, 60]

def _stat(sig):
    if len(sig) == 0: return np.zeros(15)
    return np.array([
        np.mean(sig), np.std(sig), np.min(sig), np.max(sig), np.ptp(sig),
        np.median(sig), kurtosis(sig), skew(sig),
        np.percentile(sig, 25), np.percentile(sig, 75),
        np.mean(np.abs(np.diff(sig))) if len(sig)>1 else 0,
        np.std(np.diff(sig)) if len(sig)>1 else 0,
        np.max(np.abs(np.diff(sig))) if len(sig)>1 else 0,
        len(find_peaks(sig)[0]) / max(len(sig), 1),
        len(find_peaks(-sig)[0]) / max(len(sig), 1),
    ])

def _segment(sig):
    feats = []
    for ns in SEGMENT_COUNTS:
        sl = max(1, len(sig) // ns); means, stds, slopes, ranges_ = [], [], [], []
        for j in range(ns):
            seg = sig[j*sl: min((j+1)*sl, len(sig))]
            if len(seg) == 0: seg = np.array([sig[-1]])
            means.append(np.mean(seg)); stds.append(np.std(seg)); ranges_.append(np.ptp(seg))
            slopes.append(np.polyfit(np.arange(len(seg)), seg, 1)[0] if len(seg) > 1 else 0)
        feats += [np.mean(means), np.std(means), np.mean(stds), np.max(stds),
                  np.mean(slopes), np.std(slopes), np.max(ranges_), np.mean(ranges_),
                  np.max(np.abs(np.diff(means))) if len(means) > 1 else 0,
                  np.max(np.abs(np.diff(stds))) if len(stds) > 1 else 0]
    return np.array(feats)

def _ctam_windows(sig):
    feats = []
    for w in CTAM_WINDOWS:
        if len(sig) < w: feats += [0]*6; continue
        seg = sig[-w:]
        feats += [np.mean(seg), np.std(seg), np.max(seg)-np.min(seg),
                  np.polyfit(np.arange(len(seg)), seg, 1)[0] if len(seg)>1 else 0,
                  kurtosis(seg) if len(seg)>3 else 0, skew(seg) if len(seg)>3 else 0]
    return np.array(feats)

def _transient(sig):
    if len(sig) < 10: return np.zeros(10)
    diff = np.diff(sig)
    abs_d = np.abs(diff); thr = np.mean(abs_d) + 2 * np.std(abs_d)
    spikes = abs_d > thr; n_spikes = np.sum(spikes)
    first = np.argmax(spikes) if n_spikes > 0 else -1
    last = len(spikes) - 1 - np.argmax(spikes[::-1]) if n_spikes > 0 else -1
    try:
        sm = savgol_filter(sig, min(15, len(sig)//2*2+1), min(3, len(sig)//2*2), mode='nearest')
        noise = sig - sm; snr = np.std(sm) / (np.std(noise)+1e-10)
    except: snr = 0
    return np.array([n_spikes, n_spikes/len(sig), np.max(abs_d), np.mean(abs_d),
                     first/len(sig) if first>=0 else -1, last/len(sig) if last>=0 else -1,
                     snr, np.std(abs_d), kurtosis(diff) if len(diff)>3 else 0,
                     skew(diff) if len(diff)>3 else 0])

def _transient_morphology_features(sig):
    """Transient morphology features: sharpness, smoothness, monotonicity, etc."""
    n = len(sig)
    if n < 20: return np.zeros(10)
    diff = np.diff(sig)
    abs_d = np.abs(diff)
    sharpness = np.max(abs_d) / (np.mean(abs_d) + 1e-10)
    d2 = np.diff(diff)
    smoothness = np.std(d2) if len(d2) > 1 else 0
    signs = np.sign(diff)
    same_sign = np.sum(signs[1:] == signs[:-1]) / max(len(signs)-1, 1) if len(signs) > 1 else 0
    x = np.arange(n)
    corr = np.corrcoef(x, sig)[0, 1]
    r_squared = corr ** 2 if not np.isnan(corr) else 0
    steepest_pos = np.argmin(diff) / max(len(diff)-1, 1)
    sorted_d = np.sort(abs_d)[::-1]
    top5_energy = np.sum(sorted_d[:max(1, len(sorted_d)//20)])
    concentration = top5_energy / (np.sum(abs_d) + 1e-10)
    mid = n // 2
    slope1 = np.polyfit(np.arange(mid), sig[:mid], 1)[0] if mid > 1 else 0
    slope2 = np.polyfit(np.arange(n-mid), sig[mid:], 1)[0] if n-mid > 1 else 0
    slope_ratio = slope1 / (slope2 + 1e-10) if abs(slope2) > 1e-10 else 0
    slope_ratio = np.clip(slope_ratio, -10, 10)
    tail = np.mean(sig[-max(1,n//10):]) - np.mean(sig[:max(1,n//10)])
    dk = kurtosis(diff) if len(diff) > 3 else 0
    lag = max(1, n // 10)
    if n > lag:
        ac = np.corrcoef(sig[:-lag], sig[lag:])[0, 1]
        ac = ac if not np.isnan(ac) else 0
    else:
        ac = 0
    return np.array([sharpness, smoothness, same_sign, r_squared, steepest_pos,
                     concentration, slope_ratio, tail, dk, ac])

def extract_features(voltage, target_len=500):
    v = voltage.copy()
    if len(v) > target_len: v = v[np.linspace(0, len(v)-1, target_len).astype(int)]
    elif len(v) < target_len and len(v) > 10:
        v = np.interp(np.linspace(0, len(v)-1, target_len), np.arange(len(v)), v)
    # z-score normalization with physical floor for low-resistance scenarios
    std_v = max(np.std(v), 0.05)
    v = (v - np.mean(v)) / std_v
    parts = [_stat(v), _segment(v), _ctam_windows(v), _transient(v), _transient_morphology_features(v)]
    feat = np.concatenate(parts).astype(np.float64)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


# --- Fault detector ---
# ==========================================
class MSTFHiDetDetector:
    """
    MSTF-HiDet Fault Detector.
    Loads a trained model for:
      1. File-level classification
      2. Sliding-window fault localization
      3. Detection delay measurement
    """

    def __init__(self, bundle_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Detector] Loading bundle: {bundle_path}")
        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)

        self.feat_dim = bundle['feat_dim']
        self.l2_names = bundle['l2_names']
        self.target_len = bundle.get('target_len', 500)
        self.scaler_mean = bundle['scaler_mean']
        self.scaler_scale = bundle['scaler_scale']

        # Rebuild model
        self.model = MSTFHiDetNet(
            feat_dim=bundle['feat_dim'],
            num_l2_classes=bundle['num_l2'],
            num_l3_classes=bundle['num_l3'],
            hidden=bundle['hidden_dim'],
            num_heads=bundle['num_heads'],
            dropout=bundle['dropout'],
        ).to(self.device)
        self.model.load_state_dict(bundle['model_state'])
        self.model.eval()

        # NN-Only inference (no ensemble override)
        print(f"[Detector] Model loaded | feat_dim={self.feat_dim} | mode=NN Only | device={self.device}")

    def _scale(self, X):
        return (X - self.scaler_mean) / (self.scaler_scale + 1e-10)

    def _predict(self, X_scaled):
        """NN-Only prediction — neural network is the sole source of truth"""
        with torch.no_grad():
            x = torch.FloatTensor(X_scaled).to(self.device)
            if x.dim() == 1: x = x.unsqueeze(0)
            o1, o2, o3, _ = self.model(x)
            probs = F.softmax(o2, dim=1).cpu().numpy()
            pred_l2 = o2.argmax(1).cpu().numpy()
            pred_l1 = o1.argmax(1).cpu().numpy()
        return pred_l1, pred_l2, probs

    def detect_file(self, filepath):
        print(f"\n{'='*60}")
        print(f"  Detecting: {os.path.basename(filepath)}")
        print(f"{'='*60}")

        # Read file
        try: df = pd.read_excel(filepath)
        except: print("  Cannot read file"); return None

        volt_cols = [c for c in df.columns if 'volt' in c.lower() or 'Volt' in c]
        if not volt_cols: print("  No voltage column found"); return None
        mv = [c for c in volt_cols if 'module' in c.lower()]
        voltage = df[mv[0]].values.astype(float) if mv else df[volt_cols[0]].values.astype(float)

        time_cols = [c for c in df.columns if 'time' in c.lower() or 'Time' in c]
        time_arr = df[time_cols[0]].values.astype(float) if time_cols else np.arange(len(voltage), dtype=float)

        lab_cols = [c for c in df.columns if 'label' in c.lower() or 'Label' in c]
        labels = df[lab_cols[0]].values.astype(int) if lab_cols else None

        print(f"  Data points: {len(voltage)} | Range: {time_arr[0]:.1f}s ~ {time_arr[-1]:.1f}s")

        # 1. File-level classification
        feat = extract_features(voltage, self.target_len)
        feat_scaled = self._scale(feat.reshape(1, -1))
        pred_l1, pred_l2, probs = self._predict(feat_scaled)

        result = {
            'file': os.path.basename(filepath),
            'prediction': int(pred_l2[0]),
            'fault_type': self.l2_names[pred_l2[0]],
            'confidence': float(probs[0].max()),
            'is_fault': bool(pred_l1[0] == 1),
            'probs': {self.l2_names[i]: float(probs[0][i]) for i in range(len(self.l2_names))},
        }
 
        print(f"\n  Diagnosis result:")
        print(f"    Type: {result['fault_type']}")
        print(f"    Confidence: {result['confidence']:.2f}")
        print(f"    Class probs: {result['probs']}")

        # 2. Sliding-window detection
        if result['is_fault']:
            print(f"\n  Sliding window detection...")
            window_results = self._sliding_window_detect(voltage, time_arr)
            result['sliding_window'] = window_results

            if window_results['detection_time'] is not None:
                print(f"    Fault detected at: {window_results['detection_time']:.1f}s")
                print(f"    First fault window: [{window_results['first_fault_start']:.1f}s, {window_results['first_fault_end']:.1f}s]")

                # 3. Detection delay
                if labels is not None:
                    true_fault_start = None
                    for i in range(len(labels)):
                        if labels[i] == 1:
                            true_fault_start = time_arr[i]
                            break

                    if true_fault_start is not None:
                        delay = window_results['detection_time'] - true_fault_start
                        result['true_fault_start'] = float(true_fault_start)
                        result['detection_delay'] = float(delay)
                        print(f"\n    True fault onset: {true_fault_start:.1f}s")
                        print(f"    * Detection delay: {delay:.1f}s {'[OK]' if delay < 5 else '[!] Large delay'}")
        else:
            print(f"\n  [OK] No fault detected (Normal)")

        # 4. Ground-truth comparison
        if labels is not None:
            has_true_fault = (labels == 1).any()
            result['has_true_fault'] = bool(has_true_fault)
            correct = result['is_fault'] == has_true_fault
            result['correct'] = correct
            print(f"\n  Ground truth: {'Fault' if has_true_fault else 'Normal'} | Pred: {'Fault' if result['is_fault'] else 'Normal'} | {'[Y]' if correct else '[N]'}")

        return result

    def _sliding_window_detect(self, voltage, time_arr, min_window=50, step_ratio=0.25):

        n = len(voltage)

        # Rule-based bypass: sudden voltage drop
        dt = np.diff(time_arr)
        dv = np.diff(voltage)
        for i in range(len(dv)):
            if dt[i] < 1e-6:
                continue
            # Check accumulated voltage drop in 1-2s window
            j = i
            t_acc = 0.0
            v_drop = 0.0
            while j < len(dv) and t_acc < 2.0:
                t_acc += dt[j]
                v_drop += dv[j]  # dv is negative for a drop
                j += 1
            if t_acc >= 1.0 and v_drop < -0.4:
                fault_time = float(time_arr[i])
                return {
                    'detection_time': fault_time,
                    'first_fault_start': fault_time,
                    'first_fault_end': float(time_arr[min(j, n-1)]),
                    'n_fault_segments': 1,
                    'segments': [{'start_idx': i, 'end_idx': j,
                                  'start_time': fault_time, 'end_time': float(time_arr[min(j, n-1)]),
                                  'prediction': -1, 'fault_type': 'Critical Short (Rule)',
                                  'confidence': 1.0}],
                    'rule_triggered': True,
                }
        n = len(voltage)
        # Multiple window sizes
        window_sizes = [min_window, min_window*2, min_window*4, n//4, n//2]
        window_sizes = sorted(set(max(30, min(w, n)) for w in window_sizes))

        first_fault_time = None
        first_start = None
        first_end = None
        segment_results = []

        for win_size in window_sizes:
            step = max(1, int(win_size * step_ratio))
            for start in range(0, n - win_size + 1, step):
                end = start + win_size
                seg = voltage[start:end]
                if len(seg) < 20: continue

                feat = extract_features(seg, self.target_len)
                feat_scaled = self._scale(feat.reshape(1, -1))
                _, pred_l2, probs = self._predict(feat_scaled)

                is_fault = pred_l2[0] > 0
                conf = probs[0].max()
                t_start = time_arr[start]
                t_end = time_arr[min(end-1, n-1)]

                if is_fault and conf > 0.7:
                    segment_results.append({
                        'start_idx': start, 'end_idx': end,
                        'start_time': float(t_start), 'end_time': float(t_end),
                        'prediction': int(pred_l2[0]),
                        'fault_type': self.l2_names[pred_l2[0]],
                        'confidence': float(conf),
                    })
                    if first_fault_time is None or t_start < first_fault_time:
                        first_fault_time = t_start
                        first_start = t_start
                        first_end = t_end

        return {
            'detection_time': first_fault_time,
            'first_fault_start': first_start,
            'first_fault_end': first_end,
            'n_fault_segments': len(segment_results),
            'segments': segment_results[:20],
        }

    def detect_dir(self, dirpath):
        files = sorted([f for f in os.listdir(dirpath) if f.endswith(('.xlsx', '.csv'))])
        if not files: print(f"No xlsx/csv files in {dirpath}"); return []
        print(f"\nBatch detection: {len(files)} files...\n")
        results = []
        for f in files:
            r = self.detect_file(os.path.join(dirpath, f))
            if r: results.append(r)

        # Summary
        print(f"\n{'='*60}")
        print(f"  Batch Detection Summary ({len(results)} files)")
        print(f"{'='*60}")
        print(f"  {'File':<35s} {'Prediction':>10s} {'Conf':>8s} {'Correct':>6s} {'Delay(s)':>8s}")
        print(f"  {'-'*70}")
        correct_count = 0
        total_with_label = 0
        delays = []
        for r in results:
            delay_str = f"{r.get('detection_delay', 0):.1f}" if 'detection_delay' in r else '-'
            correct_str = '[Y]' if r.get('correct', None) else ('[N]' if r.get('correct', None) is not None else '-')
            if r.get('correct') is not None:
                total_with_label += 1
                if r['correct']: correct_count += 1
            if 'detection_delay' in r: delays.append(r['detection_delay'])
            print(f"  {r['file']:<35s} {r['fault_type']:>10s} {r['confidence']:>8.2f} {correct_str:>6s} {delay_str:>8s}")

        if total_with_label > 0:
            print(f"\n  Accuracy: {correct_count}/{total_with_label} = {correct_count/total_with_label:.2f}")
        if delays:
            print(f"  Avg delay: {np.mean(delays):.2f}s | Max: {np.max(delays):.2f}s | Min: {np.min(delays):.2f}s")

        return results

    def plot_detection(self, filepath, output_path=None):
        try:
            import matplotlib
            matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE = os.path.join(REPO_ROOT, 'checkpoints', 'mstf_hidet_bundle.pkl')
DEFAULT_REAL_DATA = os.environ.get('MSTF_REAL_DATA', r'D:\AE\dataset_holographic')
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, 'results', 'supplementary', 'detection_delay')
        except ImportError:
            print("matplotlib required"); return

        try: df = pd.read_excel(filepath)
        except: return None
        volt_cols = [c for c in df.columns if 'volt' in c.lower() or 'Volt' in c]
        if not volt_cols: return None
        mv = [c for c in volt_cols if 'module' in c.lower()]
        voltage = df[mv[0]].values.astype(float) if mv else df[volt_cols[0]].values.astype(float)
        time_cols = [c for c in df.columns if 'time' in c.lower() or 'Time' in c]
        time_arr = df[time_cols[0]].values.astype(float) if time_cols else np.arange(len(voltage), dtype=float)
        lab_cols = [c for c in df.columns if 'label' in c.lower() or 'Label' in c]
        labels = df[lab_cols[0]].values.astype(int) if lab_cols else None

        result = self.detect_file(filepath)
        if not result: return None

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

        ax = axes[0]
        ax.plot(time_arr, voltage, 'b-', lw=1, alpha=0.8, label='Voltage')

        true_fault_start = None
        if labels is not None:
            fault_mask = labels == 1
            if fault_mask.any():
                ax.fill_between(time_arr, voltage.min()*0.98, voltage.max()*1.02,
                               where=fault_mask, alpha=0.12, color='red',
                               zorder=1, label='Fault Region')
                fault_start_idx = np.argmax(fault_mask)
                true_fault_start = time_arr[fault_start_idx]
                ax.axvline(true_fault_start, color='red', ls='--', lw=1.8,
                          alpha=0.9, zorder=4)
                y_range = voltage.max() - voltage.min()
                ax.annotate('Fault\nOnset',
                           xy=(true_fault_start, voltage.max() - y_range*0.05),
                           fontsize=9, fontweight='bold', color='red',
                           ha='right', va='top',
                           xytext=(-8, 0), textcoords='offset points')

        if result.get('is_fault') and result.get('sliding_window', {}).get('detection_time') is not None:
            det_time = result['sliding_window']['detection_time']
            ax.axvline(det_time, color='#2E7D32', ls='-', lw=2.2, alpha=0.9, zorder=5)

            y_range = voltage.max() - voltage.min()
            ax.annotate('Detection',
                       xy=(det_time, voltage.min() + y_range*0.15),
                       fontsize=9, fontweight='bold', color='#2E7D32',
                       ha='left', va='bottom',
                       xytext=(6, 0), textcoords='offset points')

            if true_fault_start is not None:
                delay = det_time - true_fault_start
                mid_y = voltage.min() + y_range * 0.08
                if delay > 0:
                    ax.annotate('', xy=(det_time, mid_y), xytext=(true_fault_start, mid_y),
                               arrowprops=dict(arrowstyle='<->', color='#E65100', lw=1.8))
                    mid_t = (true_fault_start + det_time) / 2
                    ax.text(mid_t, mid_y + y_range*0.04,
                           f'$\\Delta t$ = {abs(delay):.1f}s',
                           fontsize=10, fontweight='bold', color='#E65100',
                           ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                    edgecolor='#E65100', alpha=0.85))

        ax.set_xlabel('Time (s)'); ax.set_ylabel('Voltage (V)')
        ax.set_title(f'{result["file"]} | Prediction: {result["fault_type"]} ({result["confidence"]:.2%})',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9); ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        if result.get('sliding_window', {}).get('segments'):
            segs = result['sliding_window']['segments']
            times = [(s['start_time']+s['end_time'])/2 for s in segs]
            confs = [s['confidence'] for s in segs]
            colors_seg = ['red' if s['prediction'] > 0 else 'blue' for s in segs]
            ax2.scatter(times, confs, c=colors_seg, s=20, alpha=0.6)
            ax2.axhline(0.7, color='orange', ls='--', lw=1, label='Threshold')
            ax2.set_ylabel('Confidence'); ax2.set_xlabel('Time (s)')
            ax2.set_title('Sliding Window Detection Confidence')
            ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 1.05)
        else:
            ax2.text(0.5, 0.5, 'No sliding window results', ha='center', va='center', transform=ax2.transAxes)

        plt.tight_layout()
        if output_path is None:
            output_path = filepath.rsplit('.', 1)[0] + '_detection.png'
        plt.savefig(output_path, dpi=200); plt.close()
        print(f"  Chart saved: {output_path}")
        return result


def setup_sci_fonts():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    candidates = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif',
                  'Nimbus Roman', 'FreeSerif', 'serif']
    available = set(f.name for f in font_manager.fontManager.ttflist)
    chosen = 'serif'
    for c in candidates:
        if c in available: chosen = c; break
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': [chosen, 'DejaVu Serif'],
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.unicode_minus': False, 'mathtext.fontset': 'dejavuserif',
        'axes.linewidth': 1.3, 'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
        'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
    })
    return chosen


def find_representative_files(base_dir, scenario_folder, target_resistances=[10, 1, 0.1, 0.01]):
    d = os.path.join(base_dir, scenario_folder)
    if not os.path.isdir(d): return {}
    found = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(('.xlsx', '.csv')): continue
        m = re.search(r'([\d.]+)Ω', f)
        if not m: continue
        r_val = float(m.group(1))
        for target in target_resistances:
            if abs(r_val - target) < 1e-6:
                if target not in found:
                    found[target] = os.path.join(d, f)
    return found


def generate_sci_detection_figure(detector, base_dir, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    chosen_font = setup_sci_fonts()
    print(f"\n[SCI Figure] Font: {chosen_font}")

    target_r = [10, 1, 0.1, 0.01]
    scenarios = {
        '充电短路': {'en': 'Charging short-circuit', 'slug': 'charging_short_circuit', 'color': '#D32F2F', 'light': '#FFCDD2'},
        'GZ': {'en': 'Full-SOC Resting Short-circuit', 'slug': 'full_soc_resting_short_circuit', 'color': '#1565C0', 'light': '#BBDEFB'},
    }

    os.makedirs(output_dir, exist_ok=True)

    for sc_cn, sc_info in scenarios.items():
        print(f"\n{'='*60}")
        print(f"  Processing: {sc_info['en']}")
        print(f"{'='*60}")

        files = find_representative_files(base_dir, sc_cn, target_r)
        if not files:
            print(f"  [WARN] No files found for {sc_info['en']}")
            continue

        available_r = sorted(files.keys(), reverse=True)
        n_plots = len(available_r)
        if n_plots == 0: continue

        # Subplot layout
        ncols = min(2, n_plots)
        nrows = (n_plots + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(7.5*ncols, 4.5*nrows))
        if n_plots == 1: axes = np.array([axes])
        axes = np.atleast_2d(axes)

        panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

        for idx, r_val in enumerate(available_r):
            row, col = idx // ncols, idx % ncols
            ax = axes[row, col]
            filepath = files[r_val]
            fname = os.path.basename(filepath)

            print(f"\n  [{panel_labels[idx]}] R = {r_val}Ω : {fname}")

            # Read file
            try: df = pd.read_excel(filepath)
            except: ax.text(0.5, 0.5, 'Read Error', ha='center', va='center', transform=ax.transAxes); continue

            volt_cols = [c for c in df.columns if 'volt' in c.lower() or 'Volt' in c]
            if not volt_cols: continue
            mv = [c for c in volt_cols if 'module' in c.lower()]
            voltage = df[mv[0]].values.astype(float) if mv else df[volt_cols[0]].values.astype(float)
            time_cols = [c for c in df.columns if 'time' in c.lower() or 'Time' in c]
            time_arr = df[time_cols[0]].values.astype(float) if time_cols else np.arange(len(voltage), dtype=float)
            lab_cols = [c for c in df.columns if 'label' in c.lower() or 'Label' in c]
            labels_arr = df[lab_cols[0]].values.astype(int) if lab_cols else None

            # Run detection
            feat = extract_features(voltage, detector.target_len)
            feat_scaled = detector._scale(feat.reshape(1, -1))
            pred_l1, pred_l2, probs = detector._predict(feat_scaled)
            fault_type = detector.l2_names[pred_l2[0]]
            confidence = float(probs[0].max())
            is_fault = pred_l1[0] == 1

            # Sliding window
            sw_result = None
            if is_fault:
                sw_result = detector._sliding_window_detect(voltage, time_arr)

            # === Plot ===
            # Voltage curve
            ax.plot(time_arr, voltage, color='#333333', lw=1.2, alpha=0.9, zorder=3)

            # Fault region (semi-transparent)
            true_fault_start = None
            if labels_arr is not None:
                fault_mask = labels_arr == 1
                if fault_mask.any():
                    ax.fill_between(time_arr, voltage.min()*0.98, voltage.max()*1.02,
                                   where=fault_mask, alpha=0.12, color='red',
                                   zorder=1, label='Fault Region')
                    # Fault onset line
                    fault_start_idx = np.argmax(fault_mask)
                    true_fault_start = time_arr[fault_start_idx]
                    ax.axvline(true_fault_start, color='red', ls='--', lw=1.8,
                              alpha=0.9, zorder=4)
                    # Annotation
                    y_range = voltage.max() - voltage.min()
                    ax.annotate('Fault\nOnset',
                               xy=(true_fault_start, voltage.max() - y_range*0.05),
                               fontsize=9, fontweight='bold', color='red',
                               ha='right', va='top',
                               xytext=(-8, 0), textcoords='offset points')

            # Detection delay (no green line or delay annotation drawn)
            det_time = None
            delay = None
            if sw_result and sw_result['detection_time'] is not None:
                det_time = sw_result['detection_time']
                if true_fault_start is not None:
                    delay = det_time - true_fault_start

            # Axes
            ax.set_xlabel('Time (s)', fontweight='bold')
            ax.set_ylabel('Voltage (V)', fontweight='bold')
            ax.grid(True, alpha=0.2, ls='-', lw=0.5)
            ax.set_xlim(time_arr[0], time_arr[-1])
            v_margin = (voltage.max() - voltage.min()) * 0.08
            ax.set_ylim(voltage.min() - v_margin, voltage.max() + v_margin)

            # Title
            r_str = f'{r_val}' if r_val >= 1 else f'{r_val}'
            ax.set_title(f'$R_{{sc}}$ = {r_str} $\\Omega$',
                        fontsize=14, fontweight='bold', loc='left', pad=8)




            # Print result
            delay_str = f'{abs(delay):.1f}s' if delay is not None else 'N/A'
            print(f"    Pred={fault_type} | Conf={confidence:.2f} | Delay={delay_str}")

        # Hide empty subplots
        for idx in range(n_plots, nrows*ncols):
            row, col = idx // ncols, idx % ncols
            axes[row, col].set_visible(False)



        # Suptitle
        fig.suptitle(f'{sc_info["en"]} Detection Results (MSTF-HiDet)',
                    fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout(rect=[0, 0, 1, 1.0])

        # Save
        out_pdf = os.path.join(output_dir, f'Detection_{sc_info["slug"]}.pdf')
        out_png = os.path.join(output_dir, f'Detection_{sc_info["slug"]}.png')
        fig.savefig(out_pdf, format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {out_pdf}")
        print(f"  Saved: {out_png}")

    # Generate summary table
    generate_summary_table(detector, base_dir, scenarios, target_r, output_dir)


def generate_summary_table(detector, base_dir, scenarios, target_r, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    setup_sci_fonts()

    all_results = []
    for sc_cn, sc_info in scenarios.items():
        files = find_representative_files(base_dir, sc_cn, target_r)
        for r_val in sorted(files.keys(), reverse=True):
            filepath = files[r_val]
            try: df = pd.read_excel(filepath)
            except: continue
            volt_cols = [c for c in df.columns if 'volt' in c.lower() or 'Volt' in c]
            if not volt_cols: continue
            mv = [c for c in volt_cols if 'module' in c.lower()]
            voltage = df[mv[0]].values.astype(float) if mv else df[volt_cols[0]].values.astype(float)
            time_cols = [c for c in df.columns if 'time' in c.lower() or 'Time' in c]
            time_arr = df[time_cols[0]].values.astype(float) if time_cols else np.arange(len(voltage), dtype=float)
            lab_cols = [c for c in df.columns if 'label' in c.lower() or 'Label' in c]
            labels_arr = df[lab_cols[0]].values.astype(int) if lab_cols else None

            feat = extract_features(voltage, detector.target_len)
            feat_scaled = detector._scale(feat.reshape(1, -1))
            pred_l1, pred_l2, probs = detector._predict(feat_scaled)
            is_fault = pred_l1[0] == 1

            true_start = None
            if labels_arr is not None:
                fm = labels_arr == 1
                if fm.any(): true_start = time_arr[np.argmax(fm)]

            det_time = None
            if is_fault:
                sw = detector._sliding_window_detect(voltage, time_arr)
                det_time = sw.get('detection_time')

            delay = (det_time - true_start) if (det_time is not None and true_start is not None) else None
            correct = is_fault == (labels_arr is not None and (labels_arr == 1).any()) if labels_arr is not None else None

            all_results.append({
                'Scenario': sc_info['en'],
                'R (Ω)': r_val,
                'Prediction': detector.l2_names[pred_l2[0]],
                'Confidence': float(probs[0].max()),
                'Correct': 'Y' if correct else 'N',
                'Fault Onset (s)': f'{true_start:.1f}' if true_start else '-',
                'Detection (s)': f'{det_time:.1f}' if det_time else '-',
                'Delay (s)': f'{abs(delay):.1f}' if delay is not None else '-',
            })

    if not all_results: return

    # Table figure
    fig, ax = plt.subplots(figsize=(14, 0.6 * len(all_results) + 2.5))
    ax.axis('off')

    col_names = ['Scenario', 'R (Ω)', 'Prediction', 'Conf.', 'Correct',
                 'Fault Onset (s)', 'Detection (s)', 'Delay (s)']
    cell_text = []
    for r in all_results:
        cell_text.append([r['Scenario'], f"{r['R (Ω)']}", r['Prediction'],
                         f"{r['Confidence']:.2%}", r['Correct'],
                         r['Fault Onset (s)'], r['Detection (s)'], r['Delay (s)']])

    colors = []
    for r in all_results:
        if r['Correct'] == 'Y':
            colors.append(['#E8F5E9']*8)
        else:
            colors.append(['#FFEBEE']*8)

    table = ax.table(cellText=cell_text, colLabels=col_names,
                    cellColours=colors, colColours=['#E3F2FD']*8,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#BDBDBD')
        cell.set_linewidth(0.8)
        if key[0] == 0:  # header
            cell.set_text_props(fontweight='bold', fontsize=12)
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', fontweight='bold', fontsize=11)
        else:
            cell.set_text_props(fontweight='bold', fontsize=10)

    ax.set_title('MSTF-HiDet Detection Results Summary', fontsize=15,
                fontweight='bold', pad=20)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'Detection_Summary_Table.pdf'), format='pdf', bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(output_dir, 'Detection_Summary_Table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Summary table saved to {output_dir}")


# --- CLI entry point ---
def main():
    parser = argparse.ArgumentParser(description='MSTF-HiDet Fault Detector')
    parser.add_argument('--bundle', type=str,
                        default=DEFAULT_BUNDLE,
                        help='Model bundle path')
    parser.add_argument('--file', type=str, default=None, help='Single test file path')
    parser.add_argument('--dir', type=str, default=None, help='Test file directory')
    parser.add_argument('--plot', action='store_true', help='Generate visualization')
    parser.add_argument('--device', type=str, default=None, help='cpu/cuda')
    args = parser.parse_args()

    if not os.path.exists(args.bundle):
        print(f"Bundle not found: {args.bundle}")
        print("Run training first.")
        return

    detector = MSTFHiDetDetector(args.bundle, device=args.device)

    if args.file:
        result = detector.detect_file(args.file)
        if args.plot and result:
            detector.plot_detection(args.file)
    elif args.dir:
        results = detector.detect_dir(args.dir)
        if args.plot:
            for r in results:
                fp = os.path.join(args.dir, r['file'])
                detector.plot_detection(fp)
    else:
        # Default: run on dataset
        base_dir = DEFAULT_REAL_DATA
        output_dir = DEFAULT_OUTPUT_DIR

        if not os.path.isdir(base_dir):
            print(f"Directory not found: {base_dir}")
            return

        print(f"\n{'#'*70}")
        print(f"  MSTF-HiDet Fault Detector")
        print(f"  Test data: {base_dir}")
        print(f"  Output:    {output_dir}")
        print(f"{'#'*70}")

        # 1. Batch detection
        for sc in ['充电短路', 'GZ', 'Normal']:
            d = os.path.join(base_dir, sc)
            if os.path.isdir(d):
                print(f"\n\n{'='*60}")
                print(f"  Scenario: {SCENARIO_DISPLAY_NAMES.get(sc, sc)}")
                print(f"{'='*60}")
                detector.detect_dir(d)

        # 2. Generate detection figures
        print(f"\n\n{'#'*70}")
        print(f"  Generating SCI-level Detection Figures...")
        print(f"{'#'*70}")
        generate_sci_detection_figure(detector, base_dir, output_dir)

        print(f"\n All done! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
