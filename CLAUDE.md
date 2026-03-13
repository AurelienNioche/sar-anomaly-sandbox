# CLAUDE.md — Project Guide for AI Assistants

## What this project is

A research sandbox for anomaly detection on two types of satellite sensor data:

- **SAR (Synthetic Aperture Radar)**: 2-D image patches; the detector (`RXDetector`) flags bright-target anomalies using a per-patch Reed-Xiaoli statistic.
- **Telemetry**: Multivariate time series (7 channels of housekeeping data); multiple detectors covering statistical, ML, and deep-learning approaches.

Everything is synthetic — there is no real sensor data in the repository.

---

## Module layout

```
src/
  data/generators/
    speckle.py          # SpeckleSARGenerator  — SAR patch factory
    telemetry.py        # TelemetryGenerator   — multivariate time series factory
  models/
    baselines/
      rx_detector.py              # RXDetector (SAR)
      telemetry_statistical.py    # PerChannelZScore, MahalanobisDetector, CUSUMDetector
    classical/
      telemetry_ml.py             # IsolationForestDetector, OneClassSVMDetector
    deep/
      lstm_autoencoder.py         # LSTMAutoencoderDetector
  utils/
    config.py       # load_config(path) — YAML → dict
    metrics.py      # _best_f1_threshold — shared metric helper
    seed.py         # set_seed(seed)
  experiments/
    run_generate.py   # CLI: generate and save SAR data
  visualization/
    data_io.py              # Shared I/O: save_run, load_tensors_from_dir, list_runs
    sar_dashboard.py        # Streamlit app — SAR
    telemetry_dashboard.py  # Streamlit app — Telemetry
```

Generated data lands in `data/synthetic/sar/` (SAR) and `data/synthetic/telemetry/` (telemetry) as timestamped sub-folders containing `.pt` files. These are **not** committed to the repository.

---

## Detector contract

Every telemetry detector exposes exactly this API:

```python
detector.fit(normal: Tensor[T, C]) -> self
detector.score(data: Tensor[T, C]) -> Tensor[T]   # higher = more anomalous
detector.predict(data, threshold: float) -> Tensor[T]  # 0 / 1
```

`fit` receives only **normal** (anomaly-free) training data. `score` returns a real-valued anomaly score per timestep. `predict` thresholds the score. There is no shared ABC or Protocol — the contract is documented in the module-level docstring of `telemetry_statistical.py`.

---

## Synthetic telemetry channels

| Index | Name | Model |
|---|---|---|
| 0 | `power_v` | Orbital sine, correlated with `batt_soc` |
| 1 | `batt_soc` | Orbital sine, correlated with `power_v` |
| 2 | `temp_rf` | Thermal sine (larger amplitude, phase-shifted) |
| 3 | `temp_obc` | Thermal sine (smaller amplitude) |
| 4 | `gyro_x` | Ornstein-Uhlenbeck, fast mean-reverting |
| 5 | `gyro_y` | Ornstein-Uhlenbeck, fast mean-reverting |
| 6 | `reaction_wheel` | Ornstein-Uhlenbeck, slow mean-reverting |

Anomaly magnitudes are expressed as multiples of each channel's own standard deviation (computed analytically), so detection difficulty is consistent regardless of scale. Current values: spike ±8σ, step ±6σ, ramp 0→6σ.

---

## Test strategy

Tests live in `tests/` and are split by concern:

| File | What it tests |
|---|---|
| `test_models.py` | RXDetector API + AUC floor on SAR data |
| `test_generators.py` | SAR generator output shapes and label ratios |
| `test_telemetry_generators.py` | Telemetry generator channel models, OU process, anomaly injection |
| `test_telemetry_detectors.py` | Detector API contracts (shapes, fit-before-score guard) |
| `test_detector_properties.py` | **Performance floors** — AUC, precision, F1 against documented claims |
| `test_dashboard.py` | SAR dashboard: data I/O helpers, `load_patches_labels_from_dir` |
| `test_dashboard_telemetry.py` | Telemetry dashboard: tab structure, session-state sync, data flow |
| `test_integration.py` | SAR end-to-end: generate → save → load → detect |
| `test_telemetry_integration.py` | Telemetry end-to-end: generate → fit each detector → score |

**`test_detector_properties.py` is the regression guard for detector performance.** Each test pins a documented claim (e.g., "Mahalanobis AUC ≥ 0.85 on mixed data"). Run these before and after any refactor that touches model code.

Run all tests:
```bash
pytest tests/
```

Run only performance tests:
```bash
pytest tests/test_detector_properties.py -v
```

---

## Key conventions

**Seeds**: `set_seed(seed)` in `src/utils/seed.py` seeds both `random` (stdlib) and `numpy.random`. Generators accept `seed: int | None` and call `set_seed` in `__init__`. Tests use fixed seeds (typically `42`) for reproducibility.

**Anomaly magnitude**: Always expressed in units of the channel's own `σ` (stored in `self._channel_std`), computed analytically from the channel model parameters. `_channel_std` is set as a side-effect of `_make_baseline()` and is `None` before `generate()` is called.

**Config dataclasses**: Both generators use a `@dataclass` config (`SpeckleSARGeneratorConfig`, `TelemetryGeneratorConfig`). Detectors use flat constructor kwargs — config dataclasses are not used for detectors.

**Dashboard data flow**:
```
Generator tab  →  save_run()  →  data/synthetic/telemetry/<timestamp>/
Visualize tab  →  list_runs() → selectbox → session_state sync
Detector tabs  ←  list_runs() → selectbox → load tensors
```
All tabs share a `list_runs(base_dir, filenames)` helper from `data_io.py` that returns run directories newest-first.

**`_best_f1_threshold`**: Shared helper in `src/utils/metrics.py`. Iterates ROC-curve thresholds and returns the one maximising F1. Used by dashboard threshold sliders (initial/reset value). The comparison tab uses a coarser percentile sweep — the two are intentionally different (interactive reset needs precision; comparison needs speed).

**Linting**: `ruff` is configured in `pyproject.toml`. Run `ruff check src tests` before committing.

---

## Dashboard architecture

Both dashboards are single-file Streamlit apps. They are launched with:

```bash
streamlit run src/visualization/telemetry_dashboard.py
streamlit run src/visualization/sar_dashboard.py
```

Streamlit caches Python module imports at the process level. After editing an imported module (e.g., a detector), restart the Streamlit server — a browser hard-refresh is not sufficient.

Session state keys follow the pattern `tel_{tab_key}_{thing}` (e.g., `tel_stat_scores`, `tel_deep_threshold`). String literals are scattered through the dashboard files; a typo silently drops state.

---

## Common pitfalls

- **Calling `_inject_*` before `generate()`**: `_channel_std` is `None` until `_make_baseline()` runs inside `generate()`. Each injector guards against this with an explicit `RuntimeError`.
- **`fill_between` with stale ylim**: `ax.get_ylim()` read immediately after `ax.plot()` may not reflect `tight_layout()` adjustments. Use `ax.axvspan(start, end, ...)` per anomaly region instead.
- **`_INJECTORS` dict bypasses subclass dispatch**: Use `getattr(self, f"_inject_{atype}")` to allow subclass overrides.
- **Committed `.pt` files**: `test_no_stale_data_in_telemetry_dir` checks `git ls-files` to ensure run artifacts are not committed. The equivalent check for `data/synthetic/sar/` is not yet automated — avoid `git add data/`.
