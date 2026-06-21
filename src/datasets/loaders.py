"""Dataset loader registry."""

from .mnist import get_mnist_loaders, get_data_info as get_mnist_info

SUPPORTED_DATASETS = ("mnist",)


def get_loaders(dataset: str = "mnist", data_dir: str = "./data", seed: int | None = None, **kwargs):
    if dataset == "mnist":
        return get_mnist_loaders(data_dir=data_dir, seed=seed, **kwargs)
    raise ValueError(f"Unsupported dataset: {dataset}. Choose from {SUPPORTED_DATASETS}")


def get_dataset_info(dataset: str = "mnist") -> dict:
    if dataset == "mnist":
        return get_mnist_info()
    raise ValueError(f"Unsupported dataset: {dataset}")
