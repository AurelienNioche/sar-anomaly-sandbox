"""LSTM autoencoder for unsupervised telemetry anomaly detection.

The model is trained on normal windows only. Anomalies are detected as
timesteps where the per-timestep reconstruction MSE exceeds a threshold.

API:
    fit(normal: Tensor[T, C], ...) -> self
    score(data: Tensor[T, C]) -> Tensor[T]   per-timestep reconstruction MSE
    predict(data, threshold) -> Tensor[T]
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _make_windows_tensor(x: torch.Tensor, window: int, step: int = 1) -> torch.Tensor:
    """Return (N, window, C) tensor of sliding windows."""
    n_t = len(x)
    starts = range(0, n_t - window + 1, step)
    return torch.stack([x[s : s + window] for s in starts])


class _LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, n_layers: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(
            n_features, hidden_size, n_layers, batch_first=True
        )
        self.decoder = nn.LSTM(
            hidden_size, hidden_size, n_layers, batch_first=True
        )
        self.output_proj = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, c) = self.encoder(x)
        seq_len = x.shape[1]
        decoder_input = h[-1].unsqueeze(1).expand(-1, seq_len, -1)
        out, _ = self.decoder(decoder_input, (h, c))
        return self.output_proj(out)


class LSTMAutoencoderDetector:
    """Trains an LSTM autoencoder on normal telemetry; scores by reconstruction MSE."""

    def __init__(
        self,
        window: int = 30,
        hidden_size: int = 32,
        n_layers: int = 1,
        lr: float = 1e-3,
        n_epochs: int = 20,
        batch_size: int = 64,
    ) -> None:
        self.window = window
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model: _LSTMAutoencoder | None = None
        self.train_losses: list[float] = []

    def fit(self, normal: torch.Tensor) -> "LSTMAutoencoderDetector":
        n_features = normal.shape[1]
        windows = _make_windows_tensor(normal.float(), self.window)
        dataset = TensorDataset(windows)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = _LSTMAutoencoder(n_features, self.hidden_size, self.n_layers)
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
            self.train_losses.append(epoch_loss / len(windows))

        self.model.eval()
        return self

    def score(self, data: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        n_t = data.shape[0]
        x = data.float()
        scores = torch.zeros(n_t)
        counts = torch.zeros(n_t)
        self.model.eval()
        with torch.no_grad():
            windows = _make_windows_tensor(x, self.window)
            recon = self.model(windows)
            mse = ((windows - recon) ** 2).mean(dim=2)
            for i, start in enumerate(range(n_t - self.window + 1)):
                scores[start : start + self.window] += mse[i]
                counts[start : start + self.window] += 1
        counts = counts.clamp(min=1)
        return scores / counts

    def predict(self, data: torch.Tensor, threshold: float) -> torch.Tensor:
        return (self.score(data) > threshold).long()
