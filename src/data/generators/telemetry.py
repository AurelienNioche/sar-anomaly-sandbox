import random
from dataclasses import dataclass, field

import numpy as np
import torch

from src.utils.seed import set_seed

CHANNEL_NAMES = [
    "power_v",        # Orbital sinusoidal; correlated with batt_soc
    "batt_soc",       # Orbital sinusoidal; correlated with power_v
    "temp_rf",        # Thermal cycling — sinusoidal, larger amplitude, phase-shifted
    "temp_obc",       # Thermal cycling — sinusoidal, smaller amplitude
    "gyro_x",         # Attitude error — Ornstein-Uhlenbeck, fast mean-reverting
    "gyro_y",         # Attitude error — Ornstein-Uhlenbeck, fast mean-reverting
    "reaction_wheel", # Angular-momentum bias — Ornstein-Uhlenbeck, slow mean-reverting
]

ANOMALY_TYPES = ("spike", "step", "ramp", "correlation_break")


@dataclass
class TelemetryGeneratorConfig:
    n_channels: int = 7
    n_timesteps: int = 1000
    sampling_hz: float = 1.0
    noise_std: float = 0.05       # Base Gaussian noise (fraction of channel amplitude)
    orbital_period_steps: int = 200
    anomaly_ratio: float = 0.05
    anomaly_types: list[str] = field(default_factory=lambda: list(ANOMALY_TYPES))
    anomaly_min_duration: int = 5
    anomaly_max_duration: int = 30
    seed: int | None = None


class TelemetryGenerator:
    """Synthetic multivariate satellite telemetry generator.

    Each channel uses a physically motivated baseline model:

    - power_v / batt_soc  : orbital sinusoidal + cross-correlated Gaussian noise
    - temp_rf / temp_obc  : sinusoidal thermal cycling with different phases / amplitudes
    - gyro_x / gyro_y     : Ornstein-Uhlenbeck (fast mean-reverting attitude errors)
    - reaction_wheel       : Ornstein-Uhlenbeck (slow mean-reverting momentum bias)

    All anomaly magnitudes are expressed as multiples of each channel's own standard
    deviation (computed analytically from the channel model), so detection difficulty
    is consistent regardless of channel scale.

    Anomaly types:
    - spike             : one timestep, one channel shifts ±5σ
    - step              : window, one channel shifts ±3σ
    - ramp              : window, one channel drifts 0→4σ
    - correlation_break : window, channel 1 decouples from channel 0 (same amplitude,
                          zero correlation) — only detectable by multivariate methods
    """

    def __init__(self, config: TelemetryGeneratorConfig) -> None:
        self.config = config
        self._channel_std: np.ndarray | None = None
        if config.seed is not None:
            set_seed(config.seed)

    def _ou_process(self, n_t: int, theta: float, sigma: float) -> np.ndarray:
        """Ornstein-Uhlenbeck: x[t] = (1-theta)*x[t-1] + sigma*N(0,1).

        Stationary with mean=0 and std = sigma / sqrt(2*theta) (for small theta).
        """
        eps = np.random.randn(n_t).astype(np.float32) * sigma
        x = np.zeros(n_t, dtype=np.float32)
        for i in range(1, n_t):
            x[i] = (1.0 - theta) * x[i - 1] + eps[i]
        return x

    def _make_baseline(self) -> np.ndarray:
        n_t = self.config.n_timesteps
        n_c = self.config.n_channels
        noise = self.config.noise_std
        omega = 2.0 * np.pi / self.config.orbital_period_steps
        t = np.arange(n_t, dtype=np.float32)

        data = np.zeros((n_t, n_c), dtype=np.float32)
        channel_std = np.ones(n_c, dtype=np.float32)

        # --- Channels 0-1: power_v, batt_soc —————————————————————————————
        # Same orbital sine + a shared noise term (induces correlation).
        amp01 = 1.0
        sine01 = amp01 * np.sin(omega * t).astype(np.float32)
        corr = np.random.randn(n_t).astype(np.float32) * noise
        data[:, 0] = sine01 + corr + np.random.randn(n_t).astype(np.float32) * noise
        data[:, 1] = sine01 + corr * 0.9 + np.random.randn(n_t).astype(np.float32) * noise
        # Analytical std: sqrt(amp^2/2 + var_corr + var_indep)
        std01 = float(np.sqrt(amp01 ** 2 / 2 + 2 * noise ** 2))
        channel_std[0] = channel_std[1] = std01

        # --- Channels 2-3: temp_rf, temp_obc ——————————————————————————————
        if n_c > 2:
            amp2 = 2.0
            data[:, 2] = amp2 * np.sin(omega * t + 0.3).astype(np.float32) \
                         + np.random.randn(n_t).astype(np.float32) * noise
            channel_std[2] = float(np.sqrt(amp2 ** 2 / 2 + noise ** 2))
        if n_c > 3:
            amp3 = 1.0
            data[:, 3] = amp3 * np.sin(omega * t + 0.7).astype(np.float32) \
                         + np.random.randn(n_t).astype(np.float32) * noise
            channel_std[3] = float(np.sqrt(amp3 ** 2 / 2 + noise ** 2))

        # --- Channels 4-5: gyro_x, gyro_y (fast OU) ———————————————————————
        ou_theta_fast = 0.15
        ou_sigma_fast = noise * 2
        for ch in range(4, min(6, n_c)):
            data[:, ch] = self._ou_process(n_t, ou_theta_fast, ou_sigma_fast)
            channel_std[ch] = float(ou_sigma_fast / np.sqrt(2 * ou_theta_fast))

        # --- Channel 6: reaction_wheel (slow OU) ——————————————————————————
        if n_c > 6:
            ou_theta_slow = 0.005
            ou_sigma_slow = noise
            data[:, 6] = self._ou_process(n_t, ou_theta_slow, ou_sigma_slow)
            channel_std[6] = float(ou_sigma_slow / np.sqrt(2 * ou_theta_slow))

        self._channel_std = channel_std[:n_c]
        return data[:, :n_c]

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
        start = self._find_free_window(labels, 1)
        if start is None:
            return
        ch = random.randint(0, self.config.n_channels - 1)
        sign = 1 if random.random() > 0.5 else -1
        data[start, ch] += sign * 8.0 * float(self._channel_std[ch])
        labels[start] = 1

    def _inject_step(self, data: np.ndarray, labels: np.ndarray) -> None:
        duration = random.randint(
            self.config.anomaly_min_duration, self.config.anomaly_max_duration
        )
        start = self._find_free_window(labels, duration)
        if start is None:
            return
        ch = random.randint(0, self.config.n_channels - 1)
        sign = 1 if random.random() > 0.5 else -1
        shift = sign * 6.0 * float(self._channel_std[ch])
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
        final_mag = 6.0 * float(self._channel_std[ch])
        ramp = np.linspace(0, final_mag, duration, dtype=np.float32)
        data[start : start + duration, ch] += ramp
        labels[start : start + duration] = 1

    def _inject_correlation_break(self, data: np.ndarray, labels: np.ndarray) -> None:
        duration = random.randint(
            self.config.anomaly_min_duration, self.config.anomaly_max_duration
        )
        start = self._find_free_window(labels, duration)
        if start is None:
            return
        # Channels 0 and 1 are normally correlated via a shared noise term.
        # Break the correlation by replacing channel 1 with independent noise of the
        # same amplitude — the z-score won't change much, but Mahalanobis will flag it.
        std1 = float(self._channel_std[1])
        data[start : start + duration, 1] = (
            np.random.randn(duration).astype(np.float32) * std1
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
