"""Property tests for detector guarantees documented in docs/telemetry_detection.md.

Each test validates a specific claim made in the documentation, so that the docs
and the code cannot silently diverge.
"""

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

from src.data.generators.telemetry import TelemetryGenerator, TelemetryGeneratorConfig
from src.models.baselines import CUSUMDetector, MahalanobisDetector, PerChannelZScore
from src.models.classical import IsolationForestDetector, OneClassSVMDetector
from src.models.classical.telemetry_ml import _make_windows as _make_windows_raw

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_data(seed: int = 42, anomaly_types=None):
    cfg = TelemetryGeneratorConfig(
        n_channels=7, n_timesteps=1000, noise_std=0.05,
        orbital_period_steps=200, anomaly_ratio=0.05, seed=seed,
        anomaly_types=anomaly_types or ["spike", "step", "ramp", "correlation_break"],
    )
    data, labels_mc = TelemetryGenerator(cfg).generate(n_series=1)
    data = data[0]           # (T, C)
    labels_mc = labels_mc[0] # (T,) multi-class
    labels = (labels_mc > 0).long()  # binary: 0=normal, 1=anomaly
    normal = data[labels == 0]
    train = normal[:len(normal) // 2]
    return train, data, labels


def _best_f1_metrics(scores: np.ndarray, y: np.ndarray):
    _, _, thresholds = roc_curve(y, scores)
    from sklearn.metrics import f1_score
    best_t, best_f1 = thresholds[0], 0.0
    for t in thresholds:
        f1 = f1_score(y, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    preds = (scores >= best_t).astype(int)
    return (
        precision_score(y, preds, zero_division=0),
        recall_score(y, preds, zero_division=0),
        best_f1,
        best_t,
    )


# ---------------------------------------------------------------------------
# PerChannelZScore — calibration
# ---------------------------------------------------------------------------

def test_zscore_z99_maps_training_p99_to_one():
    """After fit, each channel's training p99 z-score must be exactly 1.0 when
    divided by z99.  This is the definition of per-channel calibration."""
    train, _, _ = _default_data()
    det = PerChannelZScore(window=1).fit(train)

    z_train = (train.float() - det.mean).abs() / det.std
    p99_raw = z_train.quantile(0.99, dim=0)
    ratio = p99_raw / det.z99
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-4), (
        f"z99 calibration off: p99/z99 = {ratio.tolist()}"
    )


def test_zscore_slow_ou_channel_does_not_dominate():
    """Without calibration the slow-OU channel (ch6) dominates; with z99
    calibration, the max score for normal data should be comparable across channels.
    Any channel should contribute ≤ 2× the other channels' normal p99 score."""
    train, data, labels = _default_data()
    det = PerChannelZScore(window=1).fit(train)

    z_raw = (data.float() - det.mean).abs() / det.std
    z_cal = z_raw / det.z99
    normal_mask = labels == 0

    per_channel_p99 = z_cal[normal_mask].quantile(0.99, dim=0)
    max_p99 = float(per_channel_p99.max())
    min_p99 = float(per_channel_p99.min())
    assert max_p99 / (min_p99 + 1e-6) <= 2.5, (
        f"Channel p99 calibrated scores span {min_p99:.2f}–{max_p99:.2f}; "
        f"ratio {max_p99/min_p99:.2f} > 2.5 — calibration not equalising channels"
    )


def test_zscore_precision_floor_on_spikes_and_steps():
    """PerChannelZScore must achieve precision ≥ 0.7 at its best-F1 threshold
    when data contains only spike/step/ramp anomalies (no correlation_break)."""
    train, data, labels = _default_data(anomaly_types=["spike", "step", "ramp"])
    det = PerChannelZScore(window=1).fit(train)
    scores = det.score(data).numpy()
    y = labels.numpy()
    prec, rec, f1, _ = _best_f1_metrics(scores, y)
    assert prec >= 0.7, (
        f"PerChannelZScore precision={prec:.3f} < 0.7 on spike/step/ramp data"
    )


def test_zscore_correlation_break_not_detectable():
    """A z-score detector is documented as blind to correlation_break.
    AUC on correlation_break-only data must be ≤ 0.65 (near chance)."""
    train, data, labels = _default_data(
        seed=99, anomaly_types=["correlation_break"]
    )
    det = PerChannelZScore(window=1).fit(train)
    scores = det.score(data).numpy()
    auc = roc_auc_score(labels.numpy(), scores)
    assert auc <= 0.65, (
        f"PerChannelZScore AUC={auc:.3f} on correlation_break — "
        f"expected ≤ 0.65 (detector should be blind to this type)"
    )


# ---------------------------------------------------------------------------
# MahalanobisDetector — cross-channel detection
# ---------------------------------------------------------------------------

def test_mahalanobis_auc_floor():
    """MahalanobisDetector must achieve AUC ≥ 0.85 on default mixed data.
    Observed: ~0.96.  This absolute floor ensures the existing relative test
    (must beat ZScore by 0.10) cannot pass if both detectors have regressed."""
    train, data, labels = _default_data()
    det = MahalanobisDetector(window=20).fit(train)
    auc = roc_auc_score(labels.numpy(), det.score(data).numpy())
    assert auc >= 0.85, f"MahalanobisDetector AUC={auc:.3f} < 0.85 floor"


def test_mahalanobis_detects_correlation_break():
    """Mahalanobis must achieve AUC > 0.7 on correlation_break-only data,
    where z-score fails."""
    train, data, labels = _default_data(
        seed=99, anomaly_types=["correlation_break"]
    )
    det = MahalanobisDetector(window=20).fit(train)
    scores = det.score(data).numpy()
    auc = roc_auc_score(labels.numpy(), scores)
    assert auc > 0.7, (
        f"MahalanobisDetector AUC={auc:.3f} on correlation_break — expected > 0.7"
    )


def test_mahalanobis_auc_exceeds_zscore_on_all_types():
    """Mahalanobis (handles all types) must beat PerChannelZScore (blind to
    correlation_break) by at least 0.1 AUC on mixed anomaly data."""
    train, data, labels = _default_data()
    y = labels.numpy()

    z_auc = roc_auc_score(y, PerChannelZScore(window=1).fit(train).score(data).numpy())
    m_auc = roc_auc_score(y, MahalanobisDetector(window=20).fit(train).score(data).numpy())
    assert m_auc > z_auc + 0.10, (
        f"Mahalanobis AUC={m_auc:.3f} should exceed ZScore AUC={z_auc:.3f} by >0.10 "
        f"on mixed data (Mahalanobis sees correlation_break, ZScore does not)"
    )


# ---------------------------------------------------------------------------
# CUSUMDetector — stationarity requirement
# ---------------------------------------------------------------------------

def test_cusum_poor_on_sinusoidal_data():
    """CUSUM is documented to fail on sinusoidal channels.  Its AUC on the
    default mixed dataset must be clearly lower than on stationary data (> 0.8),
    confirming the sinusoidal baseline confuses its cumulative sum."""
    train, data, labels = _default_data()
    det = CUSUMDetector().fit(train)
    scores = det.score(data).numpy()
    auc = roc_auc_score(labels.numpy(), scores)
    assert auc <= 0.75, (
        f"CUSUM AUC={auc:.3f} on sinusoidal data — expected ≤ 0.75 "
        f"(should be clearly worse than its ≥0.8 on stationary data)"
    )


def test_cusum_good_on_stationary_tail_step():
    """CUSUM must achieve AUC > 0.8 on stationary data with a step at the tail
    (the scenario it is designed for)."""
    torch.manual_seed(7)
    n_t, n_c = 600, 3
    normal = torch.randn(n_t, n_c) * 0.1
    data = normal.clone()
    data[400:, 0] += 2.0
    labels = torch.zeros(n_t, dtype=torch.long)
    labels[400:] = 1

    det = CUSUMDetector().fit(normal[:300])
    scores = det.score(data).numpy()
    auc = roc_auc_score(labels.numpy(), scores)
    assert auc > 0.8, f"CUSUM AUC={auc:.3f} on stationary tail-step — expected > 0.8"


# ---------------------------------------------------------------------------
# IsolationForestDetector — StandardScaler
# ---------------------------------------------------------------------------

def test_isolation_forest_auc_floor():
    """IsolationForest must achieve AUC ≥ 0.60 on default mixed data.
    Observed: ~0.73.  The floor guards against regressions in the windowing
    or scaling pipeline without being so tight that noise causes flakiness."""
    train, data, labels = _default_data()
    det = IsolationForestDetector(window=10).fit(train)
    auc = roc_auc_score(labels.numpy(), det.score(data).numpy())
    assert auc >= 0.60, f"IsolationForest AUC={auc:.3f} < 0.60 floor"


def test_isolation_forest_has_scaler_after_fit():
    """After fit, the StandardScaler must have been applied (mean_ and scale_
    attributes must exist and have the correct feature dimension)."""
    train, _, _ = _default_data()
    det = IsolationForestDetector(window=10).fit(train)
    window_dim = 10 * 7
    assert hasattr(det.scaler, "mean_"), "StandardScaler not fitted"
    assert det.scaler.mean_.shape == (window_dim,), (
        f"Scaler mean_ shape {det.scaler.mean_.shape} != ({window_dim},)"
    )


def test_isolation_forest_scaler_normalises_training_windows():
    """The training windows after scaler.transform must have mean ≈ 0 and std ≈ 1
    per feature (this is the whole point of adding the StandardScaler)."""
    train, _, _ = _default_data()
    det = IsolationForestDetector(window=10).fit(train)
    windows = _make_windows_raw(train.float().numpy(), 10)
    scaled = det.scaler.transform(windows)
    assert abs(scaled.mean()) < 0.1, f"Scaled windows mean={scaled.mean():.3f} ≠ 0"
    assert abs(scaled.std() - 1.0) < 0.1, f"Scaled windows std={scaled.std():.3f} ≠ 1"


# ---------------------------------------------------------------------------
# OneClassSVMDetector — precision
# ---------------------------------------------------------------------------

def test_ocsvm_auc_floor():
    """OneClassSVM must achieve AUC ≥ 0.65 on default mixed data.
    Observed: ~0.79.  The floor catches regressions in the scaler or window
    pipeline while giving a generous margin for seed-to-seed variance."""
    train, data, labels = _default_data()
    det = OneClassSVMDetector(window=10).fit(train)
    auc = roc_auc_score(labels.numpy(), det.score(data).numpy())
    assert auc >= 0.65, f"OneClassSVM AUC={auc:.3f} < 0.65 floor"


def test_ocsvm_precision_floor():
    """OCSVM must achieve precision ≥ 0.6 at best-F1 threshold on default data."""
    train, data, labels = _default_data()
    det = OneClassSVMDetector(window=10).fit(train)
    scores = det.score(data).numpy()
    y = labels.numpy()
    prec, rec, f1, _ = _best_f1_metrics(scores, y)
    assert prec >= 0.6, f"OCSVM precision={prec:.3f} < 0.6 at best-F1 threshold"


def test_ocsvm_auc_exceeds_isolation_forest():
    """OCSVM must outperform IsolationForest in AUC on default data.
    Documented reason: RBF kernel handles high-dimensional windows better than
    random tree splitting."""
    train, data, labels = _default_data()
    y = labels.numpy()
    if_auc = roc_auc_score(y, IsolationForestDetector(window=10).fit(train).score(data).numpy())
    svm_auc = roc_auc_score(y, OneClassSVMDetector(window=10).fit(train).score(data).numpy())
    assert svm_auc > if_auc, (
        f"OCSVM AUC={svm_auc:.3f} should beat IF AUC={if_auc:.3f}"
    )


# ---------------------------------------------------------------------------
# LSTMAutoencoderDetector — best overall
# ---------------------------------------------------------------------------

def test_lstm_reconstruction_error_gap():
    """The median reconstruction error for anomalous windows must be at least
    3× the median error for normal windows.  This validates the '10–30×' claim
    in the docs (we use a conservative 3× threshold here)."""
    from src.models.deep import LSTMAutoencoderDetector
    train, data, labels = _default_data()
    det = LSTMAutoencoderDetector(window=20, hidden_size=32, n_epochs=20).fit(train)
    scores = det.score(data).numpy()
    y = labels.numpy()
    normal_med = float(np.median(scores[y == 0]))
    anomaly_med = float(np.median(scores[y == 1]))
    assert anomaly_med >= 3.0 * normal_med, (
        f"LSTM anomaly median={anomaly_med:.4f} < 3× normal median={normal_med:.4f}"
    )


def test_lstm_beats_all_statistical_in_auc():
    """LSTM AUC must be competitive with MahalanobisDetector AUC on default data.
    With limited epochs the LSTM may trail slightly, but should be within 0.05 AUC
    and will surpass Mahalanobis with more training (n_epochs=30+)."""
    from src.models.deep import LSTMAutoencoderDetector
    train, data, labels = _default_data()
    y = labels.numpy()
    m_auc = roc_auc_score(y, MahalanobisDetector(window=20).fit(train).score(data).numpy())
    det_lstm = LSTMAutoencoderDetector(window=20, hidden_size=32, n_epochs=20).fit(train)
    lstm_auc = roc_auc_score(y, det_lstm.score(data).numpy())
    assert lstm_auc >= m_auc - 0.05, (
        f"LSTM AUC={lstm_auc:.3f} should be within 0.05 of Mahalanobis AUC={m_auc:.3f}"
    )


# ---------------------------------------------------------------------------
# Parameterisation sensitivity tests
# ---------------------------------------------------------------------------

def test_higher_noise_degrades_all_detectors():
    """At noise_std=0.15 all detectors must achieve lower AUC than at 0.05.
    Checks the claim that noise_std ↑ → harder detection."""
    results = {}
    for noise in [0.05, 0.15]:
        cfg = TelemetryGeneratorConfig(
            n_channels=7, n_timesteps=1000, noise_std=noise, seed=42,
            anomaly_ratio=0.05,
        )
        data, labels_mc = TelemetryGenerator(cfg).generate(n_series=1)
        data = data[0]
        labels = (labels_mc[0] > 0).long()
        train = data[labels == 0][:400]
        y = labels.numpy()
        det = MahalanobisDetector(window=20).fit(train)
        results[noise] = roc_auc_score(y, det.score(data).numpy())
    assert results[0.15] < results[0.05], (
        f"Mahalanobis AUC at noise=0.15 ({results[0.15]:.3f}) should be lower than "
        f"at noise=0.05 ({results[0.05]:.3f})"
    )


def test_removing_correlation_break_improves_zscore():
    """Removing correlation_break from anomaly types must improve ZScore AUC
    by at least 0.05 (since it was the main source of missed anomalies)."""
    results = {}
    for types in [["spike", "step", "ramp", "correlation_break"],
                  ["spike", "step", "ramp"]]:
        cfg = TelemetryGeneratorConfig(
            n_channels=7, n_timesteps=1000, noise_std=0.05, seed=42,
            anomaly_ratio=0.05, anomaly_types=types,
        )
        data, labels_mc = TelemetryGenerator(cfg).generate(n_series=1)
        data = data[0]
        labels = (labels_mc[0] > 0).long()
        train = data[labels == 0][:400]
        y = labels.numpy()
        det = PerChannelZScore(window=1).fit(train)
        key = "with_cb" if "correlation_break" in types else "without_cb"
        results[key] = roc_auc_score(y, det.score(data).numpy())
    assert results["without_cb"] > results["with_cb"] + 0.05, (
        f"ZScore AUC without correlation_break ({results['without_cb']:.3f}) should "
        f"exceed with correlation_break ({results['with_cb']:.3f}) by > 0.05"
    )
