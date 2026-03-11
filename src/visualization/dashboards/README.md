# SAR Dashboards

## SAR Anomaly Dashboard

Interactive Streamlit app for experimenting with the synthetic SAR generator and visualizing generated data.

### Run

```bash
streamlit run src/visualization/dashboards/sar_dashboard.py
```

### Tabs

**Generator**

- Sliders for all generator parameters: patch size, number of looks, anomaly ratio, anomaly size, base intensity, anomaly intensity.
- Optional seed for reproducibility.
- "Generate" button produces a grid of sample patches with labels (Normal/Anomaly).
- Use this to explore how each parameter affects the synthetic imagery.

**Visualize**

- Drag and drop a folder (or browse) containing `patches.pt` and `labels.pt` produced by `run_generate`.
- Displays total patch count, anomaly/normal counts.
- Index slider to browse individual patches.
- Grid preview of the first 16 patches.

### Folder Format

The Visualize tab expects a folder with:

- `patches.pt` — Tensor of shape (N, 1, H, W)
- `labels.pt` — Tensor of shape (N,) with values 0 (normal) or 1 (anomaly)

This is the default output of `run_generate --output <folder>`.
