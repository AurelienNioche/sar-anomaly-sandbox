# Synthetic SAR Data

This directory holds synthetically generated SAR-like imagery for anomaly detection experiments. The data is produced by `run_generate` and stored as `patches.pt` and `labels.pt`.

## Output Format

- `patches.pt`: Tensor of shape (N, 1, H, W) — N patches, 1 channel, H×W spatial size.
- `labels.pt`: Tensor of shape (N,) — 0 = normal, 1 = anomaly.

For how the generation works, why we use Gamma speckle, and what bright targets represent, see [src/data/generators/README.md](../../src/data/generators/README.md).
