"""Tests for all five telemetry detectors on controlled (noise-free) data."""

import torch
from sklearn.metrics import roc_auc_score

from src.models.baselines import CUSUMDetector, MahalanobisDetector, PerChannelZScore
from src.models.classical import IsolationForestDetector, OneClassSVMDetector
from src.models.deep import LSTMAutoencoderDetector


def _make_controlled_data(
    n_t: int = 300,
    n_c: int = 3,
    anomaly_value: float = 5.0,
    anomaly_start: int = 100,
    anomaly_len: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (normal, all_data, labels) with a clean step anomaly."""
    normal = torch.zeros(n_t, n_c)
    data = normal.clone()
    data[anomaly_start : anomaly_start + anomaly_len, 0] = anomaly_value
    labels = torch.zeros(n_t, dtype=torch.long)
    labels[anomaly_start : anomaly_start + anomaly_len] = 1
    return normal, data, labels


def _assert_detector_api(det, normal, data, labels) -> None:
    det.fit(normal)
    scores = det.score(data)
    assert scores.shape == (len(data),), f"score shape mismatch: {scores.shape}"
    preds = det.predict(data, threshold=float(scores.mean()))
    assert preds.shape == (len(data),)
    assert set(preds.tolist()).issubset({0, 1})


def test_per_channel_zscore_api() -> None:
    normal, data, labels = _make_controlled_data()
    _assert_detector_api(PerChannelZScore(window=10), normal, data, labels)


def test_mahalanobis_api() -> None:
    normal, data, labels = _make_controlled_data()
    _assert_detector_api(MahalanobisDetector(window=10), normal, data, labels)


def test_cusum_api() -> None:
    normal, data, labels = _make_controlled_data()
    _assert_detector_api(CUSUMDetector(), normal, data, labels)


def test_isolation_forest_api() -> None:
    normal, data, labels = _make_controlled_data()
    _assert_detector_api(IsolationForestDetector(window=10), normal, data, labels)


def test_ocsvm_api() -> None:
    normal, data, labels = _make_controlled_data()
    _assert_detector_api(OneClassSVMDetector(window=10), normal, data, labels)


def test_lstm_autoencoder_api() -> None:
    normal, data, labels = _make_controlled_data(n_t=150)
    _assert_detector_api(
        LSTMAutoencoderDetector(window=10, hidden_size=8, n_epochs=3),
        normal, data, labels,
    )


def test_per_channel_zscore_detects_step() -> None:
    normal, data, labels = _make_controlled_data()
    det = PerChannelZScore(window=10).fit(normal)
    scores = det.score(data).numpy()
    auc = roc_auc_score(labels.numpy(), scores)
    assert auc > 0.9, f"PerChannelZScore AUC={auc:.3f} on controlled data"


def test_cusum_detects_drift() -> None:
    n_t = 300
    normal = torch.zeros(n_t, 2)
    data = normal.clone()
    data[150:, 0] += torch.linspace(0, 3.0, 150)
    labels = torch.zeros(n_t, dtype=torch.long)
    labels[150:] = 1
    det = CUSUMDetector().fit(normal)
    scores = det.score(data).numpy()
    auc = roc_auc_score(labels.numpy(), scores)
    assert auc > 0.9, f"CUSUM AUC={auc:.3f} on drift data"


def test_fit_before_score_raises() -> None:
    import pytest
    data = torch.zeros(50, 3)
    for det in [PerChannelZScore(), MahalanobisDetector(), CUSUMDetector()]:
        with pytest.raises(RuntimeError):
            det.score(data)
