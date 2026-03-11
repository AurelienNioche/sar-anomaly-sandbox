# Baseline Anomaly Detectors

Classical, unsupervised anomaly detectors that serve as reference points before introducing deep models. They require only normal (background) patches to fit and produce a scalar anomaly score per patch.

## RX Detector (Reed-Xiaoli)

`rx_detector.py` — `RXDetector`

### What it is

The RX detector is the standard anomaly detection baseline for SAR and hyperspectral imagery. It scores each pixel by how far it deviates from the background distribution, then returns the maximum pixel score as the patch-level anomaly score.

For single-channel data, the background distribution is characterised by a mean μ and standard deviation σ estimated from normal training patches. Each pixel is scored by its z-score:

```
s(x) = (x - μ) / σ
```

The patch score is `max(s)` over all pixels, which naturally picks up small bright targets anywhere in the patch.

### Why it works for SAR

SAR intensity speckle is multiplicative Gamma noise with unit mean. Under the central limit theorem, for L≥4 looks the pixel distribution is well approximated by a Gaussian — which is exactly what the RX detector assumes. The RX detector is the Neyman-Pearson optimal test for a Gaussian background: it maximises detection probability at a fixed false alarm rate (CFAR — Constant False Alarm Rate).

### API

```python
from src.models.baselines import RXDetector

det = RXDetector()
det.fit(normal_patches)          # (N, 1, H, W)
scores = det.score(test_patches) # (N,) anomaly scores
labels = det.predict(test_patches, threshold=3.0)  # (N,) binary
```

### Limitations

- Assumes a single global Gaussian background. In heterogeneous scenes (e.g. mixed land/sea), a local or adaptive version would be needed.
- The threshold must be set manually (e.g. via ROC analysis on a validation set).
- Only exploits pixel-level intensity; spatial context is ignored.
