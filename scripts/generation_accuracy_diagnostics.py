"""Large-sample external Gen-ACC with a training-only style bank.

This script avoids rerunning probes/HSIC and fixes two ambiguities in older
artifacts: the post-hoc conditional sampler is fitted only on training data,
and generated-image counts are large enough to report binomial uncertainty.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.compute_leakage_diagnostics import (
    compute_external_generation_metrics,
    extract_full_encodings,
    get_eval_subset_loader,
    load_fcswae_checkpoint,
)
from src.models.external_classifiers import load_external_classifier_checkpoint
from src.utils.provenance import build_manifest, save_result_with_manifest
from src.utils.seed import set_seed


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [center - radius, center + radius]


def attach_intervals(metric: dict) -> dict:
    confusion = np.asarray(metric["confusion_counts"], dtype=np.int64)
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    result = dict(metric)
    result["correct"] = correct
    result["total"] = total
    result["micro_accuracy_wilson_ci95"] = wilson_interval(correct, total)
    result["per_class_wilson_ci95"] = [
        wilson_interval(int(confusion[index, index]), int(confusion[index].sum()))
        for index in range(confusion.shape[0])
    ]
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--external-classifier-checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=["mnist", "fashion_mnist", "cifar10"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-per-class", type=int, default=1000)
    parser.add_argument("--style-bank-size", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "runs_diag" / "generation_accuracy" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_fcswae_checkpoint(args.checkpoint, args.dataset, device)
    classifier, classifier_metadata = load_external_classifier_checkpoint(
        args.external_classifier_checkpoint, args.dataset, device
    )
    train_loader = get_eval_subset_loader(
        args.dataset, args.style_bank_size, batch_size=256, seed=args.seed, split="train"
    )
    _mu_c, mu_s, labels = extract_full_encodings(model, train_loader, device)
    means, stds, counts = [], [], []
    for class_index in range(model.n_classes):
        class_styles = mu_s[labels == class_index]
        counts.append(int(class_styles.shape[0]))
        means.append(class_styles.mean(0))
        stds.append(class_styles.std(0).clamp_min(1e-6))
    style_stats = {
        "means": torch.stack(means).to(device),
        "stds": torch.stack(stds).to(device),
    }

    metrics = {}
    for strategy in ("global_gaussian", "class_diag_t025"):
        # Give each sampling strategy its own deterministic RNG stream.
        strategy_offset = 30_000 if strategy == "global_gaussian" else 40_000
        set_seed(args.seed + strategy_offset)
        metric = compute_external_generation_metrics(
            model, classifier, model.n_classes, args.n_per_class, device,
            strategy, style_stats,
        )
        metrics[strategy] = attach_intervals(metric)

    results = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "tag": args.tag,
        "style_bank": {
            "split": "train",
            "requested_size": args.style_bank_size,
            "actual_size": int(labels.shape[0]),
            "per_class_counts": counts,
            "fit": "per-class diagonal Gaussian on posterior means mu_s",
        },
        "external_classifier_metadata": classifier_metadata,
        "generation": metrics,
    }
    manifest = build_manifest(
        repository_root=ROOT,
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        seed=args.seed,
        evaluation_config={
            "n_generated_per_class": args.n_per_class,
            "style_bank_split": "train",
            "style_bank_size": int(labels.shape[0]),
            "sampling_seed_global_gaussian": args.seed + 30_000,
            "sampling_seed_class_diag_t025": args.seed + 40_000,
        },
        model_config={
            "semantic_dim": model.semantic_dim,
            "style_dim": model.style_dim,
            "n_classes": model.n_classes,
        },
        external_classifier_path=args.external_classifier_checkpoint,
    )
    save_result_with_manifest(out_dir / "results.json", results, manifest)
    concise = {
        name: {
            "macro_gen_accuracy": value["macro_gen_accuracy"],
            "micro_gen_accuracy": value["micro_gen_accuracy"],
            "micro_accuracy_wilson_ci95": value["micro_accuracy_wilson_ci95"],
        }
        for name, value in metrics.items()
    }
    print(json.dumps(concise, indent=2))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
