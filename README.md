# SAR Anomaly Sandbox

Anomaly detection for SAR (Synthetic Aperture Radar) satellite imagery using Python, PyTorch, and SciPy.

## Setup

Create virtual environment and install (using [uv](https://docs.astral.sh/uv/)):

```bash
uv venv
uv pip install -e ".[dev]"
```

Activate the environment:

```bash
source .venv/bin/activate
```

Pre-commit hooks (ruff) are installed. To set them up in a fresh clone:

```bash
pre-commit install
```

## Project Structure

```
├── data/
│   ├── raw/                # Real SAR data (if available)
│   ├── synthetic/           # Generated SAR-like data
│   └── processed/          # Preprocessed tensors/patches
├── configs/
│   ├── model/              # Model configs (AE, UNet, ViT…)
│   ├── data/               # Data generation + preprocessing configs
│   └── experiment/         # Training/eval configs
├── src/
│   ├── data/               # Generators, loaders, transforms
│   ├── models/             # Baselines + deep models
│   ├── training/           # Trainer, evaluator, callbacks
│   ├── visualization/     # Plots, SAR-specific viz
│   ├── utils/              # Config, logging, seed
│   └── experiments/        # CLI entry points
├── notebooks/
└── tests/
```

## CLI Entry Points

- `run_train` — Train anomaly detection model
- `run_eval` — Evaluate model on SAR data
- `run_generate` — Generate synthetic SAR-like data

## Synthetic Data

Synthetic SAR patches with speckle and bright-target anomalies can be generated via `run_generate`. See [src/data/generators/README.md](src/data/generators/README.md) for how it works, why we use Gamma speckle, and what bright targets represent.

## Dashboards

**SAR imagery** — experiment with the speckle generator and run the RX detector:

```bash
streamlit run src/visualization/dashboards/sar_dashboard.py
```

**Satellite telemetry** — generate multivariate time series and compare statistical, ML, and deep learning anomaly detectors:

```bash
streamlit run src/visualization/dashboards/telemetry_dashboard.py
```

See [src/visualization/dashboards/README.md](src/visualization/dashboards/README.md) for details.

## Telemetry Anomaly Detection

Synthetic satellite telemetry (power, thermal, attitude, RF channels) with four anomaly types: spike, step, ramp, correlation break. Three detector families:

- **Statistical**: `PerChannelZScore`, `MahalanobisDetector`, `CUSUMDetector`
- **ML**: `IsolationForestDetector`, `OneClassSVMDetector`
- **Deep**: `LSTMAutoencoderDetector`

Config: `configs/data/telemetry.yaml`.

For a full explanation of how each detector works, its strengths and weaknesses, and expected performance under different generator parameterisations, see **[docs/telemetry_detection.md](docs/telemetry_detection.md)**.
