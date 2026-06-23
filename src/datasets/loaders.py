"""Dataset loader registry."""

from torchvision import datasets, transforms

from .cifar10 import get_cifar10_loaders, get_data_info as get_cifar10_info
from .emnist_letters import get_emnist_letters_loaders, get_data_info as get_emnist_letters_info
from .fashion_mnist import get_fashion_mnist_loaders, get_data_info as get_fashion_mnist_info
from .kmnist import get_kmnist_loaders, get_data_info as get_kmnist_info
from .mnist import get_mnist_loaders, get_data_info as get_mnist_info
from .svhn import get_svhn_loaders, get_data_info as get_svhn_info

SUPPORTED_DATASETS = (
    "mnist",
    "fashion_mnist",
    "kmnist",
    "emnist_letters",
    "cifar10",
    "svhn",
)


def get_loaders(dataset: str = "mnist", data_dir: str = "./data", seed: int | None = None, **kwargs):
    if dataset == "mnist":
        return get_mnist_loaders(data_dir=data_dir, seed=seed, **kwargs)
    if dataset == "fashion_mnist":
        return get_fashion_mnist_loaders(data_dir=data_dir, seed=seed, **kwargs)
    if dataset == "kmnist":
        return get_kmnist_loaders(data_dir=data_dir, seed=seed, **kwargs)
    if dataset == "emnist_letters":
        return get_emnist_letters_loaders(data_dir=data_dir, seed=seed, **kwargs)
    if dataset == "cifar10":
        return get_cifar10_loaders(data_dir=data_dir, seed=seed, **kwargs)
    if dataset == "svhn":
        return get_svhn_loaders(data_dir=data_dir, seed=seed, **kwargs)
    raise ValueError(f"Unsupported dataset: {dataset}. Choose from {SUPPORTED_DATASETS}")


def get_dataset_info(dataset: str = "mnist") -> dict:
    if dataset == "mnist":
        return get_mnist_info()
    if dataset == "fashion_mnist":
        return get_fashion_mnist_info()
    if dataset == "kmnist":
        return get_kmnist_info()
    if dataset == "emnist_letters":
        return get_emnist_letters_info()
    if dataset == "cifar10":
        return get_cifar10_info()
    if dataset == "svhn":
        return get_svhn_info()
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
    if dataset == "kmnist":
        return datasets.KMNIST(data_dir, train=False, download=True, transform=transform)
    if dataset == "emnist_letters":
        from .emnist_letters import _emnist_orientation_fix, _RemapLabels

        emnist_transform = transforms.Compose(
            [
                transforms.Lambda(_emnist_orientation_fix),
                transforms.ToTensor(),
            ]
        )
        return _RemapLabels(
            datasets.EMNIST(
                data_dir,
                split="letters",
                train=False,
                download=True,
                transform=emnist_transform,
            )
        )
    if dataset == "cifar10":
        return datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    if dataset == "svhn":
        return datasets.SVHN(
            data_dir, split="test", download=True, transform=transform
        )
    raise ValueError(f"Unsupported dataset: {dataset}")
