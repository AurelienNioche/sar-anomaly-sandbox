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
from sklearn.preprocessing import StandardScaler


def _make_windows(x: np.ndarray, window: int) -> np.ndarray:
    """Return a (T, window*C) array of boundary-clamped sliding windows."""
    n_t = len(x)
    half = window // 2
    windows = []
    for i in range(n_t):
        start = max(0, i - half)
        end = min(n_t, start + window)
        start = max(0, end - window)
        w = x[start:end].flatten()
        if len(w) < window * x.shape[1]:
            w = np.pad(w, (0, window * x.shape[1] - len(w)), mode="edge")
        windows.append(w)
    return np.array(windows, dtype=np.float32)


class _SklearnWindowDetector:
    """Base class for sliding-window sklearn anomaly detectors."""

    def __init__(self, window: int, model) -> None:
        self.window = window
        self.scaler = StandardScaler()
        self.model = model

    def fit(self, normal: torch.Tensor) -> "_SklearnWindowDetector":
        windows = _make_windows(normal.float().numpy(), self.window)
        windows = self.scaler.fit_transform(windows)
        self.model.fit(windows)
        return self

    def score(self, data: torch.Tensor) -> torch.Tensor:
        windows = _make_windows(data.float().numpy(), self.window)
        windows = self.scaler.transform(windows)
        raw = self.model.score_samples(windows)
        return torch.from_numpy(-raw.astype(np.float32))

    def predict(self, data: torch.Tensor, threshold: float) -> torch.Tensor:
        return (self.score(data) > threshold).long()


class IsolationForestDetector(_SklearnWindowDetector):
    """Isolation Forest over sliding windows of telemetry data."""

    def __init__(
        self,
        window: int = 10,
        n_estimators: int = 100,
        random_state: int = 0,
    ) -> None:
        from sklearn.ensemble import IsolationForest
        super().__init__(
            window,
            IsolationForest(
                n_estimators=n_estimators,
                contamination="auto",
                random_state=random_state,
            ),
        )


class OneClassSVMDetector(_SklearnWindowDetector):
    """One-Class SVM over sliding windows of telemetry data."""

    def __init__(
        self,
        window: int = 10,
        nu: float = 0.1,
        kernel: str = "rbf",
    ) -> None:
        from sklearn.svm import OneClassSVM
        super().__init__(window, OneClassSVM(nu=nu, kernel=kernel))
