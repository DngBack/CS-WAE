"""Train the fixed independent pixel classifier used by Stage-0 evaluation.

MNIST/Fashion-MNIST: four-convolution CNN, Adam, 20 epochs, no augmentation.
CIFAR-10: WideResNet-28-10, SGD + cosine schedule, 200 epochs, crop/flip
augmentation on the classifier training split only.

The official test split is used once for the reported real-test accuracy;
checkpoint selection uses a stratified 90/10 split of the official training
set and never uses generated images.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.audit_protocol import AUDIT_PROTOCOL_VERSION
from src.models.external_classifiers import build_external_classifier, external_classifier_name
from src.utils.provenance import sha256_file
from src.utils.seed import set_seed


def _targets(dataset) -> np.ndarray:
    values = dataset.targets
    return values.cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)


def _datasets(name: str, data_dir: str):
    tensor = transforms.ToTensor()
    if name == "mnist":
        cls = datasets.MNIST
        train_aug = tensor
    elif name == "fashion_mnist":
        cls = datasets.FashionMNIST
        train_aug = tensor
    elif name == "cifar10":
        cls = datasets.CIFAR10
        train_aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            tensor,
        ])
    else:
        raise ValueError(name)
    train_for_aug = cls(data_dir, train=True, download=True, transform=train_aug)
    train_for_eval = cls(data_dir, train=True, download=True, transform=tensor)
    test = cls(data_dir, train=False, download=True, transform=tensor)
    return train_for_aug, train_for_eval, test


def _stratified_train_validation(labels: np.ndarray, seed: int, validation_fraction: float = 0.1):
    from sklearn.model_selection import train_test_split

    indices = np.arange(labels.shape[0])
    train, validation = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=labels,
        shuffle=True,
    )
    return np.sort(train), np.sort(validation)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
        correct += int((logits.argmax(1) == labels).sum().item())
        total += labels.numel()
    return correct / total, loss_sum / total


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Stage-0 external evaluator")
    parser.add_argument("--dataset", required=True, choices=["mnist", "fashion_mnist", "cifar10"])
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="external_classifiers")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    epochs = args.epochs or (200 if args.dataset == "cifar10" else 20)
    train_aug, train_eval, test = _datasets(args.dataset, args.data_dir)
    train_idx, validation_idx = _stratified_train_validation(_targets(train_eval), args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(train_aug, train_idx.tolist()), batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=args.num_workers, pin_memory=True,
    )
    validation_loader = DataLoader(
        Subset(train_eval, validation_idx.tolist()), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = build_external_classifier(args.dataset).to(device)
    if args.dataset == "cifar10":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = None

    best_state = None
    best_validation = -1.0
    history = []
    for epoch in range(epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"external {args.dataset} epoch {epoch + 1}/{epochs}")
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            loss = F.cross_entropy(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        if scheduler is not None:
            scheduler.step()
        validation_accuracy, validation_loss = evaluate(model, validation_loader, device)
        history.append({
            "epoch": epoch + 1,
            "validation_accuracy": validation_accuracy,
            "validation_loss": validation_loss,
        })
        if validation_accuracy > best_validation:
            best_validation = validation_accuracy
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("External classifier training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    test_accuracy, test_loss = evaluate(model, test_loader, device)

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{external_classifier_name(args.dataset)}_seed{args.seed}.pth"
    payload = {
        "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "architecture": external_classifier_name(args.dataset),
        "n_classes": 10,
        "seed": args.seed,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "train_size": int(train_idx.size),
        "validation_size": int(validation_idx.size),
        "test_size": int(len(test)),
        "best_validation_accuracy": best_validation,
        "real_test_accuracy": test_accuracy,
        "real_test_loss": test_loss,
        "preprocessing": {
            "input_range": [0.0, 1.0],
            "normalization": "CIFAR-10 channel mean/std inside model" if args.dataset == "cifar10" else "none",
            "train_augmentation": "random crop 4px reflect + horizontal flip" if args.dataset == "cifar10" else "none",
            "validation_augmentation": "none",
            "test_augmentation": "none",
        },
        "model_state_dict": best_state,
    }
    torch.save(payload, checkpoint_path)
    metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
    metadata["checkpoint_sha256"] = sha256_file(checkpoint_path)
    metadata["history"] = history
    metadata_path = checkpoint_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved classifier: {checkpoint_path}")
    print(f"Real-test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
