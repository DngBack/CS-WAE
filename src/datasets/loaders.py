"""Dataset loader registry."""

from torchvision import datasets, transforms

from .mnist import get_mnist_loaders, get_data_info as get_mnist_info
from .fashion_mnist import get_fashion_mnist_loaders, get_data_info as get_fashion_mnist_info

SUPPORTED_DATASETS = ("mnist", "fashion_mnist")


def get_loaders(dataset: str = "mnist", data_dir: str = "./data", seed: int | None = None, **kwargs):
    if dataset == "mnist":
        return get_mnist_loaders(data_dir=data_dir, seed=seed, **kwargs)
    if dataset == "fashion_mnist":
        return get_fashion_mnist_loaders(data_dir=data_dir, seed=seed, **kwargs)
    raise ValueError(f"Unsupported dataset: {dataset}. Choose from {SUPPORTED_DATASETS}")


def get_dataset_info(dataset: str = "mnist") -> dict:
    if dataset == "mnist":
        return get_mnist_info()
    if dataset == "fashion_mnist":
        return get_fashion_mnist_info()
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_default_runs_dir(dataset: str = "mnist") -> str:
    """Default output root for a dataset, e.g. runs/fashion_mnist."""
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return f"runs/{dataset}"


def build_test_dataset(dataset: str = "mnist", data_dir: str = "./data"):
    """Build torchvision test set used for FID reference images."""
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == "mnist":
        return datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    if dataset == "fashion_mnist":
        return datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset: {dataset}")
