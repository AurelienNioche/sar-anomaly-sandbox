import random
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import gamma as gamma_dist

from src.utils.seed import set_seed


@dataclass
class SpeckleSARGeneratorConfig:
    patch_size: int = 64
    n_looks: int = 4
    anomaly_ratio: float = 0.1
    anomaly_size: int = 3
    base_intensity: float = 1.0
    anomaly_intensity: float = 3.0
    seed: int | None = None


class SpeckleSARGenerator:
    def __init__(self, config: SpeckleSARGeneratorConfig) -> None:
        self.config = config
        if config.seed is not None:
            set_seed(config.seed)

    def _generate_background(self) -> np.ndarray:
        shape = (self.config.patch_size, self.config.patch_size)
        base = np.ones(shape, dtype=np.float32) * self.config.base_intensity
        speckle = gamma_dist.rvs(
            self.config.n_looks,
            scale=1.0 / self.config.n_looks,
            size=shape,
            random_state=None,
        ).astype(np.float32)
        return base * speckle

    def _add_bright_target(self, patch: np.ndarray) -> np.ndarray:
        h, w = patch.shape
        s = self.config.anomaly_size
        i = random.randint(0, h - s)
        j = random.randint(0, w - s)
        patch = patch.copy()
        patch[i : i + s, j : j + s] = self.config.anomaly_intensity
        return patch

    def _generate_patch(self) -> tuple[torch.Tensor, int]:
        patch = self._generate_background()
        if random.random() < self.config.anomaly_ratio:
            patch = self._add_bright_target(patch)
            label = 1
        else:
            label = 0
        return torch.from_numpy(patch).unsqueeze(0), label

    def generate(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        patches: list[torch.Tensor] = []
        labels: list[int] = []
        for _ in range(n_samples):
            patch, label = self._generate_patch()
            patches.append(patch)
            labels.append(label)
        return torch.stack(patches), torch.tensor(labels, dtype=torch.long)
