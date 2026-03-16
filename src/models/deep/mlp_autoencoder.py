"""MLP autoencoder for unsupervised telemetry anomaly detection.

This is the "no-memory" deep detector: each timestep is treated as an
independent C-dimensional point.  There is no window, no convolution, and no
recurrence — the model learns only what a normal point in the feature space
looks like, ignoring any temporal ordering.

Strengths:
- Extremely fast to train and score (no windowing, pure matrix ops)
- Detects spikes and correlation breaks well (point-level deviations)
- No sequential coupling means it generalises cleanly across series

Limitations:
- Cannot detect gradual anomalies such as ramps: each individual timestep of a
  ramp still lies close to the normal distribution, so reconstruction error
  stays low throughout the drift.

API:
    fit(normal: Tensor[T, C]) -> self
    score(data: Tensor[T, C]) -> Tensor[T]   per-timestep reconstruction MSE
    predict(data, threshold) -> Tensor[T]
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _MLPAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int) -> None:
        super().__init__()
        bottleneck = max(1, hidden_size // 2)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class MLPAutoencoderDetector:
    """Trains a bottleneck MLP autoencoder on normal telemetry timesteps.

    Each timestep is scored independently — there is no window and no temporal
    context.  Anomaly score = per-timestep reconstruction MSE.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        lr: float = 1e-3,
        n_epochs: int = 20,
        batch_size: int = 256,
    ) -> None:
        self.hidden_size = hidden_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model: _MLPAutoencoder | None = None
        self.train_losses: list[float] = []

    def fit(self, normal: torch.Tensor) -> "MLPAutoencoderDetector":
        n_features = normal.shape[1]
        x = normal.float()
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = _MLPAutoencoder(n_features, self.hidden_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.train_losses = []
        self.model.train()
        for _ in range(self.n_epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)
            self.train_losses.append(epoch_loss / len(x))

        self.model.eval()
        return self

    def score(self, data: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        self.model.eval()
        x = data.float()
        with torch.no_grad():
            recon = self.model(x)
        return ((x - recon) ** 2).mean(dim=1)

    def predict(self, data: torch.Tensor, threshold: float) -> torch.Tensor:
        return (self.score(data) > threshold).long()
