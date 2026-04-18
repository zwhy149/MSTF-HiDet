# Response to Reviewer Comment

## Reviewer Comment
"本文提出的基于物理信息的快速诊断方法没有显著的创新，仅仅是利用了电压、温度、SOC和充电时间等常规参数。"

(Translation: "The proposed physics-informed rapid diagnosis method lacks significant innovation, merely using conventional parameters such as voltage, temperature, SOC, and charging time.")

---

## Response

We sincerely thank the reviewer for this important comment. We respectfully disagree with the characterization that our method "merely uses conventional parameters," and we would like to clarify the following key innovations that distinguish our approach from existing works:

### 1. Multi-Scale Temporal Feature Fusion (MSTF) — Not Raw Signal Input

Unlike conventional methods that directly feed raw voltage/temperature signals into classifiers, our approach constructs an **89-dimensional engineered feature space** through physics-informed multi-scale temporal analysis:

- **Statistical features (15D)**: Beyond simple mean/std, we extract higher-order moments (kurtosis, skew), peak density, and derivative statistics that capture the subtle signatures of ISC at different resistance levels.
- **Multi-resolution segment features (30D)**: Decomposing the voltage signal at 3 scales (4/8/16 segments) captures both global trends (visible at coarse resolution) and local transients (visible at fine resolution). This multi-scale design is directly motivated by the physics: high-resistance ISC (e.g., 10 Ω) produces gradual voltage drift best captured at coarse scales, while low-resistance ISC (e.g., 0.01 Ω) causes sharp transients best captured at fine scales.
- **Context-Aware Temporal Attention (CTAM) features (24D)**: Four temporal windows (5/15/30/60 s) capture the evolving voltage deviation at physically meaningful time horizons. The window sizes are calibrated to match the thermal–electrochemical time constants of lithium-ion cells.
- **Transient morphology features (10D)**: This is a **novel contribution** — features such as sharpness ratio, derivative smoothness, monotonicity score, curvature concentration, and half-signal slope ratio are designed specifically to discriminate between charging-stage and rest-stage short circuits, which exhibit fundamentally different voltage transient patterns due to different electrochemical operating conditions.

### 2. Learnable Token-Based Multi-Head Attention — Not Standard Transformer

Our MSTF-HiDet architecture introduces a **learnable token mechanism** (4 tokens × 64-dim) that learns to attend to different aspects of the 89D feature vector. This is fundamentally different from:
- Standard MLP: which treats all features equally
- Standard Transformer: which requires sequential token inputs and positional encoding

Our learnable tokens act as physics-informed "queries" that adaptively weight different feature groups (statistical vs. transient vs. morphological) based on the input signal characteristics. This design is motivated by the observation that different fault types activate different feature groups: charging ISC strongly activates transient features, while rest-stage ISC primarily activates trend and morphology features.

### 3. CUSUM-Based Sub-Second Detection — A Practical Innovation for BMS

Most existing ISC detection methods report only classification accuracy on full signals. Our method additionally provides:

- **Sub-second fault localization** using Cumulative Sum (CUSUM) control charts with physics-informed parameters:
  - $\delta$ = 0.0015 V (allowance) — calibrated to typical BMS voltage sensor noise (1–2 mV)
  - $h$ = 0.0025 V·s (decision limit) — tuned to reject single-point false alarms while maintaining fast response
  - 0.01 s interpolation resolution between 1 Hz samples

- **Demonstrated detection delays**: 0.07 s for R = 0.01 Ω, 0.10 s for R = 0.1 Ω, 0.25 s for R = 1 Ω, and 0.95 s for R = 10 Ω. These delays are **significantly faster** than the state-of-the-art (many existing methods report delays on the order of seconds to minutes).

- The physical rationale is clear: lower short-circuit resistance → larger voltage deviation rate → faster CUSUM accumulation → shorter detection delay. This inverse relationship between R_nom and detection delay is a direct consequence of Ohm's Law and is quantitatively captured by our framework.

### 4. Hierarchical Classification with Safety Guarantees

Our three-level hierarchical output (L1: Normal/Fault → L2: 3-class → L3: severity) provides:
- **Zero false negative rate** for fault detection (100% sensitivity)
- **Zero false positive rate** (100% specificity)
- Resistance-level severity grading for risk-stratified BMS response

This hierarchical design is specifically motivated by the safety requirements of real BMS deployments, where a missed fault (false negative) can lead to thermal runaway.

### 5. Domain Adaptation for Virtual-to-Real Transfer

Our training protocol addresses a practical challenge: real ISC data is scarce and dangerous to collect. We propose:
- Virtual data generation using equivalent circuit models for pre-training
- Battery-ID-based domain adaptation (Route B) to transfer knowledge to real data
- Honest evaluation with no data leakage (independent battery IDs in train/test)

### Summary of Innovations

| Aspect | Prior Work | Our Method |
|--------|-----------|------------|
| Input | Raw V/T/SOC signals | 89D physics-informed MSTF features |
| Feature design | Generic time-series features | Transient morphology features distinguishing CS vs. rest-stage ISC |
| Network | MLP, CNN, or standard Transformer | Learnable token-based multi-head attention |
| Detection | File-level classification only | CUSUM sub-second progressive scanning |
| Detection delay | Seconds to minutes | 0.07–0.95 s across all resistance levels |
| Evaluation | Random split | Battery-ID-stratified, leakage-free |
| Safety | Accuracy only | Full FPR/FNR/Specificity/Sensitivity analysis |

We believe these contributions represent a significant advance over merely using "conventional parameters." The novelty lies not in what signals are measured, but in **how the physical information embedded in voltage dynamics is extracted, fused, and utilized for rapid diagnosis**. We have revised the manuscript to better highlight these contributions.
