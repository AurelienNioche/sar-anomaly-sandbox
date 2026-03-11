from sklearn.metrics import roc_auc_score

from src.data.generators import SpeckleSARGenerator
from src.data.generators.speckle import SpeckleSARGeneratorConfig
from src.models.baselines import RXDetector


def _make_dataset(
    n: int = 200,
    anomaly_ratio: float = 0.2,
    seed: int = 0,
    anomaly_intensity: float = 8.0,
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
