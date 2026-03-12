import random
from dataclasses import dataclass, field

import numpy as np
import torch

from src.utils.seed import set_seed

CHANNEL_NAMES = ["power_v", "batt_soc", "temp_rf", "temp_obc", "gyro_x", "gyro_y", "reaction_wheel"]

ANOMALY_TYPES = ("spike", "step", "ramp", "correlation_break")


@dataclass
class TelemetryGeneratorConfig:
    n_channels: int = 7
    n_timesteps: int = 1000
    sampling_hz: float = 1.0
    noise_std: float = 0.1
    orbital_period_steps: int = 200
    anomaly_ratio: float = 0.05
    anomaly_types: list[str] = field(default_factory=lambda: list(ANOMALY_TYPES))
    anomaly_min_duration: int = 5
    anomaly_max_duration: int = 30
    seed: int | None = None


class TelemetryGenerator:
    """Generates synthetic multivariate satellite telemetry time series.

    Each channel has a sinusoidal baseline (orbital period) plus Gaussian noise.
    Channels 0-1 are positively correlated by design (power-related).
    Anomalies are injected as non-overlapping windows of configurable type.
    """

    def __init__(self, config: TelemetryGeneratorConfig) -> None:
        self.config = config
        if config.seed is not None:
            set_seed(config.seed)

    def _make_baseline(self) -> np.ndarray:
        n_t = self.config.n_timesteps
        n_c = self.config.n_channels
        t = np.arange(n_t, dtype=np.float32)
        omega = 2 * np.pi / self.config.orbital_period_steps
        phases = np.linspace(0, np.pi, n_c, dtype=np.float32)
        amplitudes = np.ones(n_c, dtype=np.float32)
        amplitudes[2] = 2.0
        signal = np.outer(np.sin(omega * t), amplitudes) + phases[np.newaxis, :]
        noise = np.random.randn(n_t, n_c).astype(np.float32) * self.config.noise_std
        corr = np.random.randn(n_t).astype(np.float32) * self.config.noise_std
        noise[:, 0] += corr
        noise[:, 1] += corr * 0.8
        return signal + noise

    def _find_free_window(
        self,
        labels: np.ndarray,
        duration: int,
        max_tries: int = 50,
    ) -> int | None:
        n_t = self.config.n_timesteps
        for _ in range(max_tries):
            start = random.randint(0, n_t - duration - 1)
            if labels[start : start + duration].sum() == 0:
                return start
        return None

    def _inject_spike(self, data: np.ndarray, labels: np.ndarray) -> None:
        duration = 1
        start = self._find_free_window(labels, duration)
        if start is None:
            return
        ch = random.randint(0, self.config.n_channels - 1)
        sign = 1 if random.random() > 0.5 else -1
        data[start, ch] += sign * 5.0 * self.config.noise_std * 10
        labels[start] = 1

    def _inject_step(self, data: np.ndarray, labels: np.ndarray) -> None:
        duration = random.randint(
            self.config.anomaly_min_duration, self.config.anomaly_max_duration
        )
        start = self._find_free_window(labels, duration)
        if start is None:
            return
        ch = random.randint(0, self.config.n_channels - 1)
        shift = (1 if random.random() > 0.5 else -1) * self.config.noise_std * 8
        data[start : start + duration, ch] += shift
        labels[start : start + duration] = 1

    def _inject_ramp(self, data: np.ndarray, labels: np.ndarray) -> None:
        duration = random.randint(
            self.config.anomaly_min_duration, self.config.anomaly_max_duration
        )
        start = self._find_free_window(labels, duration)
        if start is None:
            return
        ch = random.randint(0, self.config.n_channels - 1)
        ramp = np.linspace(0, self.config.noise_std * 10, duration, dtype=np.float32)
        data[start : start + duration, ch] += ramp
        labels[start : start + duration] = 1

    def _inject_correlation_break(self, data: np.ndarray, labels: np.ndarray) -> None:
        duration = random.randint(
            self.config.anomaly_min_duration, self.config.anomaly_max_duration
        )
        start = self._find_free_window(labels, duration)
        if start is None:
            return
        # channels 0 and 1 are normally correlated; break the correlation by
        # replacing channel 1 with independent noise
        data[start : start + duration, 1] = (
            np.random.randn(duration).astype(np.float32) * self.config.noise_std * 3
        )
        labels[start : start + duration] = 1

    _INJECTORS = {
        "spike": _inject_spike,
        "step": _inject_step,
        "ramp": _inject_ramp,
        "correlation_break": _inject_correlation_break,
    }

    def generate(self) -> tuple[torch.Tensor, torch.Tensor]:
        n_t = self.config.n_timesteps
        data = self._make_baseline()
        labels = np.zeros(n_t, dtype=np.int64)

        n_anomaly_steps = int(n_t * self.config.anomaly_ratio)
        injected = 0
        max_iter = n_t * 10
        itr = 0
        while injected < n_anomaly_steps and itr < max_iter:
            atype = random.choice(self.config.anomaly_types)
            before = int(labels.sum())
            self._INJECTORS[atype](self, data, labels)
            injected += int(labels.sum()) - before
            itr += 1

        return torch.from_numpy(data), torch.from_numpy(labels)
