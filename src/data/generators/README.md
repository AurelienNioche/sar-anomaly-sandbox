# Synthetic SAR Generators

This module produces SAR-like imagery for anomaly detection experiments. The main generator is `SpeckleSARGenerator` in `speckle.py`.

## How It Works

The generator creates small patches that mimic SAR intensity images:

1. **Background**: A homogeneous base intensity is multiplied by speckle noise.
2. **Anomalies** (optional): A bright target is inserted at a random position in a fraction of patches.

Labels are 0 (normal) or 1 (anomaly).

## Why Gamma Speckle?

SAR speckle arises from the coherent nature of radar: many scatterers within a resolution cell add up with random phases, producing a granular texture. In intensity images, speckle is well modeled as **multiplicative** noise:

**Y = X × N**

where X is the true (speckle-free) intensity and N is the noise. For multilook processing with L looks, N follows a **Gamma(L, 1/L)** distribution: mean 1, variance 1/L. Higher L means less speckle (smoother images).

The Gamma distribution is the standard choice because it emerges from the physics of coherent summation and matches empirical SAR statistics. See e.g. Goodman (1976) on laser speckle and its extension to SAR.

## What Is a Bright Target?

A **bright target** is a small region (e.g. 3×3 pixels) with higher backscatter than the background. In real SAR, such targets can be:

- **Point targets**: Ships, vehicles, corner reflectors, or man-made structures that reflect strongly.
- **Anomalies of interest**: Objects we want to detect (e.g. vessels, infrastructure).

In our synthetic data, we simulate this by setting a small patch to a fixed higher intensity. This gives a simple, controllable anomaly for training and evaluating detectors.

## Usage

```python
from src.data.generators import SpeckleSARGenerator
from src.data.generators.speckle import SpeckleSARGeneratorConfig

config = SpeckleSARGeneratorConfig(patch_size=64, n_looks=4, anomaly_ratio=0.1, seed=42)
gen = SpeckleSARGenerator(config)
patches, labels = gen.generate(1000)
```

Or via CLI:

```bash
run_generate --output data/synthetic/sar --n_samples 1000
```
