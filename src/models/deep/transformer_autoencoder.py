"""Transformer autoencoder for unsupervised telemetry anomaly detection.

Each window of length W is encoded by a TransformerEncoder (self-attention over
all W positions in parallel — no sequential hidden state) and then projected
back to the original feature space.  Anomaly score = per-timestep reconstruction
MSE averaged over all overlapping windows that include that timestep, identical
aggregation to LSTMAutoencoderDetector.

Compared to the LSTM:
- No hidden state at inference: the full window is processed in one forward pass
- Attention is O(W²) in the window length — fine for the short windows used here
  (default W=30), and faster on hardware that parallelises matrix multiply well

API:
    fit(normal: Tensor[T, C]) -> self
    score(data: Tensor[T, C]) -> Tensor[T]   per-timestep reconstruction MSE
    predict(data, threshold) -> Tensor[T]
"""

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _make_windows_tensor(x: torch.Tensor, window: int, step: int = 1) -> torch.Tensor:
    """Return (N, window, C) tensor of sliding windows."""
    n_t = len(x)
    starts = range(0, n_t - window + 1, step)
    return torch.stack([x[s : s + window] for s in starts])


class _SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class _TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = _SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_proj = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos_enc(self.input_proj(x))
        h = self.encoder(h)
        return self.output_proj(h)


class TransformerAutoencoderDetector:
    """Trains a Transformer autoencoder on normal telemetry; scores by reconstruction MSE.

    The model processes each sliding window with full self-attention (O(W²) per window).
    Unlike the LSTM, there is no sequential hidden state — all positions in a window
    are computed in parallel, making it straightforward to batch and accelerate.
    """

    def __init__(
        self,
        window: int = 30,
        d_model: int = 32,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 64,
        lr: float = 1e-3,
        n_epochs: int = 20,
        batch_size: int = 64,
    ) -> None:
        self.window = window
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model: _TransformerAutoencoder | None = None
        self.train_losses: list[float] = []

    def fit(self, normal: torch.Tensor) -> "TransformerAutoencoderDetector":
        n_features = normal.shape[1]
        windows = _make_windows_tensor(normal.float(), self.window)
        dataset = TensorDataset(windows)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = _TransformerAutoencoder(
            n_features=n_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
        )
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
