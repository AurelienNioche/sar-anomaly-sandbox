"""Statistical anomaly detectors for multivariate telemetry time series.

All detectors share the same API:
    fit(normal: Tensor[T, C]) -> self
    score(data: Tensor[T, C]) -> Tensor[T]   (higher = more anomalous)
    predict(data, threshold) -> Tensor[T]    (0/1)
"""

import numpy as np
import torch


class PerChannelZScore:
    """Per-channel rolling z-score. Anomaly score = max |z| across channels."""

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, normal: torch.Tensor) -> "PerChannelZScore":
        self.mean = normal.float().mean(dim=0)
        self.std = normal.float().std(dim=0).clamp(min=1e-6)
        return self

    def score(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit() first.")
        z = (data.float() - self.mean) / self.std
        scores = z.abs()
        if self.window > 1:
            kernel = torch.ones(1, 1, self.window) / self.window
            smoothed = []
            n_t = scores.shape[0]
            for c in range(scores.shape[1]):
                ch = scores[:, c].unsqueeze(0).unsqueeze(0)
                pad_l = (self.window - 1) // 2
                pad_r = self.window // 2
                ch_pad = torch.nn.functional.pad(ch, (pad_l, pad_r), mode="replicate")
                conv_out = torch.nn.functional.conv1d(ch_pad, kernel).squeeze()
                smoothed.append(conv_out[:n_t])
            scores = torch.stack(smoothed, dim=1)
        return scores.max(dim=1).values

    def predict(self, data: torch.Tensor, threshold: float) -> torch.Tensor:
        return (self.score(data) > threshold).long()


class MahalanobisDetector:
    """Sliding-window multivariate Mahalanobis distance detector."""

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.mean: np.ndarray | None = None
        self.cov_inv: np.ndarray | None = None

    def fit(self, normal: torch.Tensor) -> "MahalanobisDetector":
        x = normal.float().numpy()
        n_t, n_c = x.shape
        windows = np.stack(
            [x[i : i + self.window] for i in range(n_t - self.window + 1)]
        ).reshape(-1, self.window * n_c)
        self.mean = windows.mean(axis=0)
        cov = np.cov(windows, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        self.cov_inv = np.linalg.inv(cov)
        return self

    def score(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.cov_inv is None:
            raise RuntimeError("Call fit() first.")
        x = data.float().numpy()
        n_t, n_c = x.shape
        scores = np.zeros(n_t, dtype=np.float32)
        half = self.window // 2
        for i in range(n_t):
            start = max(0, i - half)
            end = min(n_t, start + self.window)
            start = max(0, end - self.window)
            w = x[start:end].flatten()
            if len(w) < self.window * n_c:
                w = np.pad(w, (0, self.window * n_c - len(w)), mode="edge")
            diff = w - self.mean
            scores[i] = float(diff @ self.cov_inv @ diff)
        return torch.from_numpy(np.sqrt(np.clip(scores, 0, None)))

    def predict(self, data: torch.Tensor, threshold: float) -> torch.Tensor:
        return (self.score(data) > threshold).long()


class CUSUMDetector:
    """CUSUM (cumulative sum) detector for drift/step anomalies per channel.

    Score at time t = max over channels of the CUSUM statistic, which
    accumulates evidence of a mean shift above a slack parameter k.
    """

    def __init__(self, k_sigma: float = 0.5) -> None:
        self.k_sigma = k_sigma
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, normal: torch.Tensor) -> "CUSUMDetector":
        self.mean = normal.float().mean(dim=0)
        self.std = normal.float().std(dim=0).clamp(min=1e-6)
        return self

    def score(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit() first.")
        x = data.float()
        z = (x - self.mean) / self.std
        k = self.k_sigma
        n_t, n_c = z.shape
        cusum_pos = torch.zeros(n_t, n_c)
        cusum_neg = torch.zeros(n_t, n_c)
        for t in range(1, n_t):
            cusum_pos[t] = torch.clamp(cusum_pos[t - 1] + z[t] - k, min=0.0)
            cusum_neg[t] = torch.clamp(cusum_neg[t - 1] - z[t] - k, min=0.0)
        return torch.maximum(cusum_pos, cusum_neg).max(dim=1).values

    def predict(self, data: torch.Tensor, threshold: float) -> torch.Tensor:
        return (self.score(data) > threshold).long()
