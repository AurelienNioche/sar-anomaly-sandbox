import torch
from sklearn.metrics import roc_auc_score

from src.data.generators import SpeckleSARGenerator
from src.data.generators.speckle import SpeckleSARGeneratorConfig
from src.models.baselines import RXDetector


def _make_dataset(
    n: int = 200,
    anomaly_ratio: float = 0.2,
    seed: int = 0,
    anomaly_intensity: float = 5.0,
):
    config = SpeckleSARGeneratorConfig(
        patch_size=32,
        n_looks=4,
        anomaly_ratio=anomaly_ratio,
        anomaly_intensity=anomaly_intensity,
        seed=seed,
    )
    gen = SpeckleSARGenerator(config)
    return gen.generate(n)


def test_rx_fit_normal_patches() -> None:
    patches, labels = _make_dataset()
    normal = patches[labels == 0]
    det = RXDetector()
    det.fit(normal)
    assert det.mean is not None
    assert det.std is not None
    assert det.std > 0


def test_rx_score_anomaly_higher() -> None:
    patches, labels = _make_dataset()
    normal = patches[labels == 0]
    det = RXDetector().fit(normal)
    scores = det.score(patches)
    mean_normal = scores[labels == 0].mean().item()
    mean_anomaly = scores[labels == 1].mean().item()
    assert mean_anomaly > mean_normal


def test_rx_predict_shape() -> None:
    patches, labels = _make_dataset()
    normal = patches[labels == 0]
    det = RXDetector().fit(normal)
    preds = det.predict(patches, threshold=3.0)
    assert preds.shape == (len(patches),)
    assert set(preds.tolist()).issubset({0, 1})


def test_rx_auc_reasonable() -> None:
    patches, labels = _make_dataset(n=500, anomaly_ratio=0.2, seed=42)
    normal = patches[labels == 0]
    det = RXDetector().fit(normal)
    scores = det.score(patches).numpy()
    auc = roc_auc_score(labels.numpy(), scores)
    assert auc > 0.8


def test_rx_on_controlled_data() -> None:
    """Verify detector correctness with hand-crafted tensors, no randomness."""
    bg_value = 1.0
    anomaly_value = 5.0
    patch_h, patch_w = 32, 32

    background = torch.full((10, 1, patch_h, patch_w), bg_value)
    anomaly = background.clone()
    anomaly[:, :, 14:17, 14:17] = anomaly_value

    all_patches = torch.cat([background, anomaly], dim=0)
    all_labels = torch.cat([torch.zeros(10), torch.ones(10)]).long()

    det = RXDetector().fit(background)
    scores = det.score(all_patches)

    bg_scores = scores[all_labels == 0]
    anom_scores = scores[all_labels == 1]
    assert (anom_scores > bg_scores.max()).all(), (
        "Every anomaly patch should score above every background patch"
    )

    auc = roc_auc_score(all_labels.numpy(), scores.numpy())
    assert auc == 1.0
