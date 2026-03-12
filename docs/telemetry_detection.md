# Telemetry Anomaly Detection

End-to-end guide: what the synthetic data looks like, how each detector works, and what performance to expect as you change the generator parameters.

---

## 1. Synthetic Data Generator

### Channel models

The generator produces a `(T, C)` tensor of `C=7` channels, each with a physically-motivated model.

| Index | Name | Model | Typical std |
|---|---|---|---|
| 0 | `power_v` | Sinusoidal orbital + correlated noise | ≈ 0.71 |
| 1 | `batt_soc` | Same orbital sine as `power_v` + shared noise (correlation ≈ 0.8) | ≈ 0.71 |
| 2 | `temp_rf` | Sinusoidal thermal cycling (amplitude 2×, phase +0.3 rad) | ≈ 1.42 |
| 3 | `temp_obc` | Sinusoidal thermal cycling (amplitude 1×, phase +0.7 rad) | ≈ 0.71 |
| 4 | `gyro_x` | Ornstein-Uhlenbeck, fast mean-reversion (θ=0.15) | ≈ 0.16 |
| 5 | `gyro_y` | Ornstein-Uhlenbeck, fast mean-reversion (θ=0.15) | ≈ 0.16 |
| 6 | `reaction_wheel` | Ornstein-Uhlenbeck, slow mean-reversion (θ=0.005) | ≈ 0.50 |

> **Why two channel types?**  Power and thermal channels follow the orbital period (sinusoidal).  Attitude channels follow a random-walk-like process that mean-reverts — they can drift for hundreds of steps before returning to zero.

### Anomaly types

All anomaly magnitudes are expressed as multiples of each channel's own standard deviation, so detection difficulty is comparable across channels.

| Type | Shape | Affected channels | Magnitude | Detectable by |
|---|---|---|---|---|
| `spike` | 1 timestep | 1 (random) | ±8σ | ZScore, Mahalanobis, OCSVM, LSTM |
| `step` | 5–30 timesteps | 1 (random) | ±6σ | ZScore, Mahalanobis, CUSUM, OCSVM, LSTM |
| `ramp` | 5–30 timesteps | 1 (random) | 0→6σ | ZScore (end only), Mahalanobis, CUSUM, LSTM |
| `correlation_break` | 5–30 timesteps | 0+1 (power channels) | same amplitude, zero correlation | Mahalanobis, LSTM only |

> `correlation_break` keeps channel 1's amplitude identical to normal.  A univariate z-score cannot see it; only multivariate methods that track cross-channel relationships can.

### Key config parameters and their effects

```yaml
# configs/data/telemetry.yaml  (defaults)
n_channels: 7
n_timesteps: 1000
noise_std: 0.05          # Base Gaussian noise added to every channel
orbital_period_steps: 200
anomaly_ratio: 0.05      # Fraction of timesteps that are anomalous
anomaly_types: [spike, step, ramp, correlation_break]
anomaly_min_duration: 5
anomaly_max_duration: 30
seed: 42
```

| Parameter | Effect on detection difficulty |
|---|---|
| `noise_std` ↑ | Harder for all detectors; signal-to-noise drops. Below 0.02 → trivial; above 0.2 → even LSTM struggles. |
| `anomaly_ratio` ↑ | More anomalous timesteps → higher recall at same threshold, but the detector's training split may contain contamination. |
| `anomaly_types = ["spike", "step", "ramp"]` | Removes `correlation_break`; all detectors perform better because the hardest type is gone. |
| `anomaly_types = ["correlation_break"]` | Only Mahalanobis and LSTM work; ZScore/CUSUM/IF will be at chance. |
| `orbital_period_steps` ↑ | Slower sine → less variation per window → slightly easier for windowed detectors. |
| `anomaly_min_duration = anomaly_max_duration = 1` | All anomalies become spikes → ZScore excels (high precision), CUSUM loses (needs sustained drift). |
| `anomaly_min_duration = 50` | Long sustained anomalies → CUSUM and LSTM excel; ZScore still works on amplitude. |
| `n_timesteps` ↑ | More training data → better-calibrated detectors across the board. Especially important for LSTM. |

---

## 2. Detectors

### 2.1 PerChannelZScore (statistical)

**How it works.** Computes per-channel z-scores `|x − μ| / σ` using the global mean and std estimated from normal training data. Each channel's score is then normalised by its training 99th-percentile z (`z99`), so that every channel's normal variation maps to ≤ 1.0 regardless of its absolute scale. The per-timestep anomaly score is the max normalised z across all channels.

**Strengths.**
- No fitting cost; near-instant inference.
- Excellent precision on spikes and steps when anomaly magnitude >> natural channel variance.
- Interpretable: a score > 1.0 means at least one channel is above its training 99th percentile.

**Weaknesses.**
- Blind to `correlation_break` (no cross-channel reasoning).
- Sensitive to slow OU drift: channel 6 (`reaction_wheel`) naturally drifts to z ≈ 4 over long windows. Without per-channel calibration (z99), this channel dominates the score and floods the output with false positives.
- Smoothing (window > 1) with mean-pooling dilutes spikes; use the default `window=1`.

**Expected performance on default config.**

| Metric | Value |
|---|---|
| AUC | ≈ 0.71 |
| Best-F1 precision | ≈ 0.96 |
| Best-F1 recall | ≈ 0.40 |

Low recall because `correlation_break` anomalies (≈ 25% of all anomaly timesteps) are invisible.

**When to choose it.** Anomaly types are known to be spikes/steps/ramps; you need fast inference and interpretable scores.

---

### 2.2 MahalanobisDetector (statistical)

**How it works.** Extracts overlapping sliding windows of shape `(window, C)`, flattens them to `window×C` features, fits a multivariate Gaussian (mean + inverse covariance) to normal training windows, and computes the Mahalanobis distance of each test window from that distribution.

**Strengths.**
- Captures **cross-channel correlations** — the only statistical detector that can flag `correlation_break`.
- Strong performance on all anomaly types with no feature engineering.
- Works even when individual channels look normal (e.g., correlation_break, where amplitude does not change).

**Weaknesses.**
- Quadratic cost in `window × C` at fit time (covariance matrix inversion). Slow for large windows.
- Requires enough training data to estimate the `(window×C)²` covariance matrix well.

**Expected performance on default config.**

| Metric | Value |
|---|---|
| AUC | ≈ 0.96 |
| Best-F1 precision | ≈ 0.74 |
| Best-F1 recall | ≈ 0.47 |

**Recommended default** for the Statistical tab.

**When to choose it.** You have all anomaly types and want the best statistical detector. Use `window=20` as a starting point; reduce if fitting is slow.

---

### 2.3 CUSUMDetector (statistical)

**How it works.** Accumulates a one-sided sum of deviations above a slack parameter `k` (in units of σ). The cumulative sum never resets; it rises when the channel drifts above `μ + k·σ` and decays back to zero when the channel returns to normal. Per-timestep score = max CUSUM over all channels.

**Strengths.**
- Designed for sustained **drift and step** anomalies; accumulates evidence over many timesteps.
- Very fast; O(T·C) inference.

**Weaknesses.**
- **Not suitable for sinusoidal channels.** The sine wave creates periodic accumulations (score rises on each peak) that are indistinguishable from a real step anomaly.
- Non-resetting: if an anomaly occurs early in the series, the high CUSUM persists for all subsequent normal data, badly distorting precision.
- Completely blind to spikes (1-step anomaly) and `correlation_break`.

**Expected performance on default config (mixed sinusoidal+OU channels).**

| Metric | Value |
|---|---|
| AUC on default data | ≈ 0.69 (well below its ≥ 0.8 on stationary data) |
| AUC on stationary data + tail step | > 0.8 |

**When to choose it.** Your channels are **stationary** (no orbital sine) and you expect slow drift or step shifts. Not recommended with the default 7-channel mixed dataset.

---

### 2.4 IsolationForestDetector (ML)

**How it works.** Extracts sliding windows of shape `(window, C)`, flattens and standardises them (StandardScaler per feature position), then trains an Isolation Forest. Anomaly score = negated `score_samples()` (higher = more anomalous). Anomalies are isolated with shorter average path lengths in randomly-split trees.

**Strengths.**
- Non-parametric; no distributional assumption.
- Handles arbitrary non-linear patterns.

**Weaknesses.**
- Suffers from the **curse of dimensionality**. The window feature space has `window × C` dimensions (e.g., 70 for `window=10`). Random splits rarely land on the single anomalous dimension (one channel per anomaly), so isolation is weak.
- Precision is systematically low even after feature scaling; the score distribution barely separates normal and anomalous windows.

**Expected performance on default config.**

| Metric | Value |
|---|---|
| AUC | ≈ 0.73 |
| Best-F1 precision | ≈ 0.14 |
| Best-F1 recall | ≈ 0.62 |

**When to choose it.** Exploratory baseline when you have no prior on the anomaly structure. Do not use as a production detector on this dataset.

---

### 2.5 OneClassSVMDetector (ML)

**How it works.** Same sliding window + StandardScaler pipeline as IsolationForest, but fits a One-Class SVM with an RBF kernel. The RBF kernel computes pairwise distances in the full feature space, making it far more sensitive to multi-dimensional deviations than tree-based isolation.

**Strengths.**
- Handles the high-dimensional window space better than IF because the kernel operates holistically on all `window×C` features simultaneously.
- Strong precision when `nu` (contamination estimate) is calibrated.

**Weaknesses.**
- Quadratic fit time in number of training windows (slow for large datasets).
- Recall is limited for `correlation_break` — same window-based limitation as IF.

**Expected performance on default config.**

| Metric | Value |
|---|---|
| AUC | ≈ 0.79 |
| Best-F1 precision | ≈ 0.88 |
| Best-F1 recall | ≈ 0.38 |

**When to choose it.** Better ML baseline than IsolationForest; use `window=10` for a good precision/speed tradeoff.

---

### 2.6 LSTMAutoencoderDetector (deep)

**How it works.** An LSTM encoder–decoder autoencoder trained to reconstruct sliding windows of normal telemetry. At inference, the per-timestep anomaly score is the mean squared reconstruction error of the window centred on that timestep. Anomalies produce high reconstruction error because the model has never seen those patterns.

**Architecture.**

```
Input  (window, C)
  → LSTM encoder (hidden_size)  → last hidden state
  → repeated hidden_size times
  → LSTM decoder (hidden_size)  → (window, hidden_size)
  → Linear projection           → (window, C)
Output (window, C)
```

**Strengths.**
- Learns **temporal dynamics and cross-channel correlations** jointly.
- Handles all four anomaly types including `correlation_break` (the correlation pattern is baked into the model's weights).
- Reconstruction error provides a natural, interpretable score (units: MSE).
- Normal reconstruction error ≈ 0.035; anomalous windows produce errors 10–30× higher.

**Weaknesses.**
- Requires more training time (`n_epochs` × dataset size).
- Performance degrades if training data is too short (< 300 normal timesteps) or if `noise_std` is very high.
- Not interpretable per channel — the score is an aggregate over the whole window.

**Expected performance on default config (`hidden_size=32`, `n_epochs=30`).**

| Metric | Value |
|---|---|
| AUC | ≈ 0.95 |
| Best-F1 precision | ≈ 0.80 |
| Best-F1 recall | ≈ 0.78 |

**Best overall detector.** Recommended default for the Deep tab and Comparison tab.

**When to choose it.** You have at least a few hundred normal timesteps for training and care about detecting all anomaly types, including correlation breaks.

---

## 3. Detector Comparison

### Default config (`noise_std=0.05`, all 4 anomaly types)

| Detector | AUC | Precision | Recall | Notes |
|---|---|---|---|---|
| PerChannelZScore | 0.71 | **0.96** | 0.40 | Blind to `correlation_break` |
| MahalanobisDetector | **0.96** | 0.74 | 0.47 | Best statistical |
| CUSUMDetector | ~0.69 | — | — | Fails on sinusoidal channels; use only on stationary data |
| IsolationForestDetector | 0.73 | 0.14 | 0.62 | Weak: dimensionality curse |
| OneClassSVMDetector | 0.79 | 0.88 | 0.38 | Good precision |
| **LSTMAutoencoderDetector** | **0.95** | **0.80** | **0.78** | Best overall |

### Choosing anomaly types only (`correlation_break` removed)

| Detector | Expected AUC change |
|---|---|
| PerChannelZScore | ↑ ~0.85 (recall improves; was missing 25% of anomalies) |
| MahalanobisDetector | ↑ ~0.98 |
| CUSUM | unchanged (still fails on sine) |
| IsolationForest | ↑ slight |
| OCSVM | ↑ ~0.85 |
| LSTM | ↑ ~0.97 |

### Increasing `noise_std` to 0.15

All detectors degrade roughly proportionally. At `noise_std=0.15`, anomaly magnitudes (6–8 × channel std) overlap more with the natural noise floor. Expect:
- ZScore AUC ≈ 0.65, precision drops to ~0.6
- Mahalanobis AUC ≈ 0.80
- LSTM AUC ≈ 0.85 (most robust to noise increase)

### Very short series (`n_timesteps=200`)

Training split has ≈ 80 normal samples after the 50% split. Expect LSTM to struggle (too few samples to learn temporal patterns); statistical methods are more robust to small training sets.

---

## 4. Choosing a Detector

```
Is the channel stationary (no orbital sine)?
  YES → CUSUMDetector for step/drift; PerChannelZScore for spikes
  NO  ↓
Do you need to detect correlation_break?
  NO  → PerChannelZScore (fast, high precision on spikes/steps)
  YES ↓
Do you have training time budget?
  NO  → MahalanobisDetector (best statistical, no training loop)
  YES → LSTMAutoencoderDetector (best overall, all anomaly types)
```
