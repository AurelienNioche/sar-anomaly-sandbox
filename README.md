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
