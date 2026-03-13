# SAR Anomaly Detection

End-to-end guide: what the synthetic data looks like, how the RX detector works, and what performance to expect as you change the generator parameters.

---

## 1. Synthetic Data Generator

### Image model

The generator produces patches that mimic SAR intensity images using a multiplicative speckle model:

**Y = X × N**

where X is the true (speckle-free) intensity and N is the speckle noise term. For multilook processing with L looks, N follows **Gamma(L, 1/L)**: mean 1, variance 1/L. Higher L means less speckle and smoother patches.

### Anomaly type: bright target

Each anomalous patch has a small rectangular region (`anomaly_size × anomaly_size` pixels) set to a fixed higher intensity (`anomaly_intensity`). This simulates point targets in real SAR — ships, vehicles, corner reflectors, or other man-made structures with high radar backscatter.

### Key config parameters and their effects

```yaml
# configs/data/synthetic.yaml  (defaults)
patch_size: 64          # Spatial size of each patch (H × W)
n_looks: 4              # Number of looks; controls speckle variance (1/L)
anomaly_ratio: 0.1      # Fraction of patches that contain a bright target
anomaly_size: 3         # Size (pixels) of the bright target region
base_intensity: 1.0     # Background mean intensity
anomaly_intensity: 3.0  # Intensity of the injected bright target
seed: null
```

| Parameter | Effect on detection difficulty |
|---|---|
| `n_looks` ↓ | More speckle variance → brighter random pixels → more false positives. |
| `n_looks` ↑ | Smoother background → easier detection; near-perfect AUC even at moderate intensities. |
| `anomaly_intensity` ↓ toward `base_intensity` | Bright target blends with high-speckle background → harder to separate from normal variance. |
| `anomaly_size` ↓ (→ 1) | Fewer anomalous pixels per patch → weaker top-k signal. Single-pixel targets are hardest. |
| `anomaly_ratio` ↑ | More anomaly patches → more training contamination if normal-only split is not enforced. |
| `patch_size` ↑ | More background pixels → top-k averaging dilutes the bright target less. Slightly easier. |

---

## 2. Detector

### RXDetector (statistical)

**How it works.** A simplified Reed-Xiaoli detector operating on single-channel intensity patches.

1. **Fit**: computes a global mean (μ) and standard deviation (σ) from all pixels of the normal training patches.
2. **Score**: z-scores every pixel in a test patch as `(x − μ) / σ`, then averages the top-k highest z-scores (default `k=10`) as the patch-level anomaly score. A bright target lifts a small cluster of pixels far above the background mean, which dominates the top-k average.
3. **Predict**: thresholds the score — patches above the threshold are flagged as anomalies.

**Strengths.**
- Extremely fast: O(N × H × W) fit and score.
- No hyperparameters beyond `top_k`; default `k=10` works well across patch sizes.
- Naturally targets spatially localised bright regions — exactly what we want to detect.
- Interpretable: score ≈ "how many σ above normal is the brightest cluster in this patch?".

**Weaknesses.**
- Sensitive to high speckle (`n_looks=1`): single-look images produce many bright pixels by chance, flooding the top-k with normal speckle peaks.
- Cannot detect anomalies that are not brighter than the background (e.g. dark targets or texture changes).
- Global μ/σ assumes a statistically homogeneous background; heterogeneous scenes (varying terrain) would require a local estimate.

**Expected performance on default config.**

| Metric | Value |
|---|---|
| AUC | > 0.90 |
| Best-F1 | > 0.70 |

On the controlled-data test (`anomaly_intensity` set well above all background pixels) the detector achieves AUC = 1.0.

**When to choose it.** Always — it is the only detector implemented for SAR patches. It is also the natural baseline: any future detector should exceed its AUC on the same data.

---

## 3. Sensitivity to Generator Parameters

### Effect of `n_looks`

| n_looks | Speckle variance | Expected AUC |
|---|---|---|
| 1 | High (σ² = 1) | Moderate (bright targets partially hidden by speckle peaks) |
| 4 (default) | Medium (σ² = 0.25) | > 0.90 |
| 16 | Low (σ² = 0.0625) | Near-perfect |

### Effect of `anomaly_intensity`

| anomaly_intensity | Contrast ratio | Expected AUC |
|---|---|---|
| ≈ base_intensity (1.0) | No contrast | ~0.5 (chance) |
| 2.0 | 2× | Moderate |
| 3.0 (default) | 3× | > 0.90 |
| >> base_intensity | Very high | ~1.0 |

---

## 4. Choosing Parameters

```
Is speckle variance high (n_looks < 4)?
  YES → Increase n_looks, or increase anomaly_intensity to compensate
  NO  ↓
Is AUC below 0.90?
  YES → Check anomaly_intensity relative to base_intensity (ratio < 2×?)
      → Check anomaly_size (1-pixel targets are hardest)
  NO  → Default config is well-calibrated; run run_generate and use the dashboard
```
