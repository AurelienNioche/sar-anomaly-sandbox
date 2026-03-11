import torch


class RXDetector:
    def __init__(self) -> None:
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, patches: torch.Tensor) -> "RXDetector":
        pixels = patches.float().reshape(-1)
        self.mean = pixels.mean()
        self.std = pixels.std().clamp(min=1e-6)
        return self

    def score(self, patches: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("RXDetector must be fit before scoring.")
        z = (patches.float() - self.mean) / self.std
        return z.reshape(len(patches), -1).max(dim=1).values

    def predict(self, patches: torch.Tensor, threshold: float) -> torch.Tensor:
        return (self.score(patches) > threshold).long()
