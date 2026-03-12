# SAR Anomaly Sandbox

Anomaly detection for SAR satellite imagery **and** satellite telemetry, using Python, PyTorch, and scikit-learn.

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
source .venv/bin/activate
pre-commit install
```

## Project Structure

```
├── configs/data/          # Generator configs (speckle.yaml, telemetry.yaml)
├── data/
│   ├── synthetic/         # Saved SAR patch runs (user-local, not committed)
│   └── telemetry/         # Saved telemetry runs (user-local, not committed)
├── docs/
│   └── telemetry_detection.md   # Detector guide with expected performance
├── src/
│   ├── data/generators/   # SpeckleSARGenerator, TelemetryGenerator
│   ├── models/
│   │   ├── baselines/     # RXDetector, PerChannelZScore, MahalanobisDetector, CUSUMDetector
│   │   ├── classical/     # IsolationForestDetector, OneClassSVMDetector
│   │   └── deep/          # LSTMAutoencoderDetector
│   ├── experiments/       # run_generate CLI
│   ├── utils/             # config loader, seed
│   └── visualization/dashboards/  # sar_dashboard.py, telemetry_dashboard.py
└── tests/
```

## Dashboards

**SAR imagery** — speckle generator + RX detector:

```bash
streamlit run src/visualization/dashboards/sar_dashboard.py
```

**Satellite telemetry** — multivariate time series + 5 anomaly detectors:

```bash
streamlit run src/visualization/dashboards/telemetry_dashboard.py
```

### Telemetry dashboard tabs

| Tab | What it does |
|---|---|
| Generator | Configure and generate synthetic telemetry; auto-saves to `data/telemetry/` |
| Visualize | Browse any saved run; auto-syncs path to all detector tabs |
| Statistical | PerChannelZScore, Mahalanobis, CUSUM |
| ML | Isolation Forest, One-Class SVM |
| Deep | LSTM Autoencoder |
| Comparison | All five detectors side-by-side (ROC curves + metrics table) |

## Synthetic Telemetry

Seven channels with physically motivated baselines:

| Channel | Model |
|---|---|
| `power_v`, `batt_soc` | Orbital sinusoidal + cross-correlated noise |
| `temp_rf`, `temp_obc` | Sinusoidal thermal cycling |
| `gyro_x`, `gyro_y` | Ornstein-Uhlenbeck (fast mean-reverting) |
| `reaction_wheel` | Ornstein-Uhlenbeck (slow mean-reverting) |

Four anomaly types: **spike** (±8σ), **step** (±6σ), **ramp** (0→6σ), **correlation_break**.

Config: `configs/data/telemetry.yaml`. Full detector guide: [docs/telemetry_detection.md](docs/telemetry_detection.md).

## CLI

```bash
run_generate --output data/synthetic --n_samples 64
```

## Tests

```bash
pytest                  # run all 89 tests
pytest --cov=src        # with coverage (overall ~46%; core detectors 95-99%)
```
