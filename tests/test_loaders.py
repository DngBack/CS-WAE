from src.datasets.loaders import get_dataloader


def test_get_dataloader_is_available():
    assert callable(get_dataloader)
