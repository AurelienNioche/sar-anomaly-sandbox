"""ML anomaly detectors for multivariate telemetry time series.

Both detectors operate on sliding windows: each timestep is represented
by a flattened window of (window, C) values. This captures local temporal
context without requiring deep learning.

API (same as statistical detectors):
    fit(normal: Tensor[T, C]) -> self
    score(data: Tensor[T, C]) -> Tensor[T]
    predict(data, threshold) -> Tensor[T]
"""

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


def _make_windows(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (windows, center_indices) for a (T, C) array."""
    n_t = len(x)
    half = window // 2
    indices = []
    windows = []
    for i in range(n_t):
        start = max(0, i - half)
        end = min(n_t, start + window)
        start = max(0, end - window)
        w = x[start:end].flatten()
        if len(w) < window * x.shape[1]:
            w = np.pad(w, (0, window * x.shape[1] - len(w)), mode="edge")
        windows.append(w)
        indices.append(i)
    return np.array(windows, dtype=np.float32), np.array(indices)


class IsolationForestDetector:
    """Isolation Forest over sliding windows of telemetry data."""

    def __init__(self, window: int = 10, n_estimators: int = 100, random_state: int = 0) -> None:
        self.window = window
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",
            random_state=random_state,
        )

    def fit(self, normal: torch.Tensor) -> "IsolationForestDetector":
        windows, _ = _make_windows(normal.float().numpy(), self.window)
        windows = self.scaler.fit_transform(windows)
        self.model.fit(windows)
        return self

    def score(self, data: torch.Tensor) -> torch.Tensor:
        windows, _ = _make_windows(data.float().numpy(), self.window)
        windows = self.scaler.transform(windows)
        raw = self.model.score_samples(windows)
        scores = -raw.astype(np.float32)
        return torch.from_numpy(scores)

    def predict(self, data: torch.Tensor, threshold: float) -> torch.Tensor:
        return (self.score(data) > threshold).long()


class OneClassSVMDetector:
    """One-Class SVM over sliding windows of telemetry data."""

    def __init__(self, window: int = 10, nu: float = 0.1, kernel: str = "rbf") -> None:
        self.window = window
        self.scaler = StandardScaler()
        self.model = OneClassSVM(nu=nu, kernel=kernel)

    def fit(self, normal: torch.Tensor) -> "OneClassSVMDetector":
        windows, _ = _make_windows(normal.float().numpy(), self.window)
        windows = self.scaler.fit_transform(windows)
        self.model.fit(windows)
        return self

    def score(self, data: torch.Tensor) -> torch.Tensor:
        windows, _ = _make_windows(data.float().numpy(), self.window)
        windows = self.scaler.transform(windows)
        raw = self.model.score_samples(windows)
        scores = -raw.astype(np.float32)
        return torch.from_numpy(scores)

    def predict(self, data: torch.Tensor, threshold: float) -> torch.Tensor:
        return (self.score(data) > threshold).long()
