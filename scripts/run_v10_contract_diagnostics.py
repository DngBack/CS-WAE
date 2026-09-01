#!/usr/bin/env python3
"""Post-specified held-out diagnostics for the complete latent contract.

This script does not train or modify a model.  It loads the fixed three-seed
F-CS-WAE Rotated-MNIST checkpoints and adds two tests requested during the v10
paper audit:

1. term 3: q(z_c | y=k) versus the model's requested p(z_c | y=k);
2. the complete encoded joint q(z_c,z_s | y=k) versus independently sampled
   p(z_c | y=k)p(z_s).

Both tests use independent posterior/prior sample banks.  Content lives on a
sphere, so its RBF kernel uses chordal distance (ordinary Euclidean distance
between unit vectors).  The joint test uses the product of the spherical
content kernel and the canonical Euclidean style kernel.  Class statistics
are averaged with equal weights because the audit population is balanced;
the permutation null independently relabels the two banks within each class.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_native_cross_model_audit import (  # noqa: E402
    EVAL_PER_CLASS_PER_DOMAIN,
    _load_trained_model,
    _loaders,
)
from src.datasets.rotated_mnist import DomainSubset  # noqa: E402
from src.metrics.audit_protocol import (  # noqa: E402
    EUCLIDEAN_SIGMAS,
    SPHERICAL_SIGMAS,
    multiscale_rbf_kernel,
)
from src.metrics.factorized_adapter import build_factorized_audit_adapter  # noqa: E402
from src.utils.provenance import sha256_file  # noqa: E402


PROTOCOL_VERSION = "v10-complete-contract-1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-root", default="runs_cross_model/native_v1")
    parser.add_argument(
        "--output",
        default="runs_diag/v10_contract_diagnostics/results.json",
    )
    return parser.parse_args()


def _git_metadata() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _mmd_values_from_kernel(
    kernel: torch.Tensor,
    n: int,
    *,
    seed: int,
    permutations: int,
) -> torch.Tensor:
    """Return observed plus permutation MMD values for two equal banks."""

    if kernel.shape != (2 * n, 2 * n):
        raise ValueError("kernel shape does not match two equal sample banks")
    kernel = kernel.clone()
    kernel.fill_diagonal_(0.0)
    total_off_diagonal = kernel.sum()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = [
        torch.cat(
            [
                torch.ones(n, dtype=kernel.dtype),
                -torch.ones(n, dtype=kernel.dtype),
            ]
        )
    ]
    for _ in range(permutations):
        permutation = torch.randperm(2 * n, generator=generator)
        sign = -torch.ones(2 * n, dtype=kernel.dtype)
        sign[permutation[:n]] = 1.0
        signs.append(sign)
    assignments = torch.stack(signs, dim=1).to(kernel.device)
    quadratic = (assignments * (kernel @ assignments)).sum(dim=0)
    within_ordered = (total_off_diagonal + quadratic) / 2.0
    cross_ordered = (total_off_diagonal - quadratic) / 2.0
    return (
        within_ordered / float(n * (n - 1))
        - cross_ordered / float(n * n)
    )


def _summarize_values(values: torch.Tensor) -> dict:
    observed = float(values[0].item())
    null = values[1:].detach().cpu().numpy().astype(np.float64)
    exceedances = int(np.count_nonzero(null >= observed))
    null_std = float(null.std(ddof=1)) if null.size > 1 else None
    standardized = None
    if null_std is not None and null_std > 0:
        standardized = float((observed - null.mean()) / null_std)
    return {
        "statistic": observed,
        "p_value": float((exceedances + 1) / (null.size + 1)),
        "null_mean": float(null.mean()),
        "null_std": null_std,
        "standardized_excess": standardized,
        "null_quantiles": {
            "q50": float(np.quantile(null, 0.50)),
            "q95": float(np.quantile(null, 0.95)),
            "q99": float(np.quantile(null, 0.99)),
        },
    }


def _holm(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    rejected = [False] * len(p_values)
    for rank, index in enumerate(order):
        threshold = alpha / (len(p_values) - rank)
        if p_values[int(index)] > threshold:
            break
        rejected[int(index)] = True
    return rejected


def _sample_fcswae_prior(
    model: torch.nn.Module,
    label: int,
    n: int,
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    # F-CS-WAE's historical sampler uses the global Torch RNG.  fork_rng
    # makes this diagnostic deterministic without altering surrounding state.
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        return model.sample_from_class_prior(label, n, device)


@torch.no_grad()
def _collect_domain(
    args: argparse.Namespace,
    model: torch.nn.Module,
    seed: int,
    domain: int,
    test_dataset,
    device: torch.device,
) -> dict:
    domain_dataset = DomainSubset(test_dataset, domain)
    count = 10 * EVAL_PER_CLASS_PER_DOMAIN
    loader = DataLoader(
        Subset(domain_dataset, list(range(count))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    adapter = build_factorized_audit_adapter(model)
    posterior_generator = torch.Generator(device="cpu").manual_seed(
        210_000 + 100 * seed + domain
    )
    content, style, labels = [], [], []
    for images, batch_labels, domains in loader:
        images = images.to(device)
        batch_labels = batch_labels.to(device)
        domains = domains.to(device)
        views = adapter.encode_views(
            images,
            generator=posterior_generator,
            domain=domains,
            labels=batch_labels,
        )
        content.append(views.content_sample.detach())
        style.append(views.style_sample.detach())
        labels.append(batch_labels.detach())
    content = torch.cat(content)
    style = torch.cat(style)
    labels = torch.cat(labels)

    term3_values = []
    joint_values = []
    class_rows = []
    for label in range(10):
        mask = labels == label
        posterior_content = content[mask]
        posterior_style = style[mask]
        n = int(mask.sum().item())
        prior_content, prior_style = _sample_fcswae_prior(
            model,
            label,
            n,
            seed=220_000 + 10_000 * seed + 100 * domain + label,
            device=device,
        )

        pooled_content = F.normalize(
            torch.cat([posterior_content, prior_content], dim=0), p=2, dim=1
        )
        content_kernel = multiscale_rbf_kernel(
            pooled_content, pooled_content, SPHERICAL_SIGMAS
        )
        term3 = _mmd_values_from_kernel(
            content_kernel,
            n,
            seed=230_000 + 10_000 * seed + 100 * domain + label,
            permutations=args.permutations,
        )

        pooled_style = torch.cat([posterior_style, prior_style], dim=0)
        style_kernel = multiscale_rbf_kernel(
            pooled_style, pooled_style, EUCLIDEAN_SIGMAS
        )
        joint = _mmd_values_from_kernel(
            content_kernel * style_kernel,
            n,
            seed=240_000 + 10_000 * seed + 100 * domain + label,
            permutations=args.permutations,
        )
        term3_values.append(term3)
        joint_values.append(joint)
        class_rows.append(
            {
                "label": label,
                "n_per_bank": n,
                "term3": _summarize_values(term3),
                "joint_contract": _summarize_values(joint),
            }
        )

    term3_aggregate = torch.stack(term3_values).mean(dim=0)
    joint_aggregate = torch.stack(joint_values).mean(dim=0)
    term3_p = [row["term3"]["p_value"] for row in class_rows]
    joint_p = [row["joint_contract"]["p_value"] for row in class_rows]
    term3_holm = _holm(term3_p)
    joint_holm = _holm(joint_p)
    for row, term3_reject, joint_reject in zip(
        class_rows, term3_holm, joint_holm
    ):
        row["term3"]["holm_reject_0.05"] = term3_reject
        row["joint_contract"]["holm_reject_0.05"] = joint_reject

    return {
        "domain": domain,
        "rotation_degrees": [-30.0, 30.0][domain],
        "n_samples": int(labels.numel()),
        "class_counts": torch.bincount(labels, minlength=10).cpu().tolist(),
        "term3_aggregate": _summarize_values(term3_aggregate),
        "joint_contract_aggregate": _summarize_values(joint_aggregate),
        "term3_holm_rejections": int(sum(term3_holm)),
        "joint_contract_holm_rejections": int(sum(joint_holm)),
        "classes": class_rows,
    }


def _mean_std(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "values": array.tolist(),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    rows = []
    checkpoint_hashes = {}
    split_hashes = {}
    for seed in args.seeds:
        model, _run_config, checkpoint = _load_trained_model(
            args, "fcswae", seed, device
        )
        _train_loader, test_loader, split_manifest = _loaders(args, seed=seed)
        checkpoint_hashes[str(seed)] = sha256_file(checkpoint)
        split_hashes[str(seed)] = hashlib.sha256(
            json.dumps(split_manifest, sort_keys=True).encode("utf-8")
        ).hexdigest()
        for domain in (0, 1):
            result = _collect_domain(
                args,
                model,
                seed,
                domain,
                test_loader.dataset,
                device,
            )
            result["seed"] = seed
            rows.append(result)
            print(
                f"seed={seed} domain={domain} "
                f"T3={result['term3_aggregate']['statistic']:.6g} "
                f"p={result['term3_aggregate']['p_value']:.4g} "
                f"Joint={result['joint_contract_aggregate']['statistic']:.6g} "
                f"p={result['joint_contract_aggregate']['p_value']:.4g}",
                flush=True,
            )

    summary = {
        "n_seed_domain_audits": len(rows),
        "term3_mmd2_u": _mean_std(
            [row["term3_aggregate"]["statistic"] for row in rows]
        ),
        "joint_contract_mmd2_u": _mean_std(
            [row["joint_contract_aggregate"]["statistic"] for row in rows]
        ),
        "term3_all_aggregate_reject": all(
            row["term3_aggregate"]["p_value"] <= 0.05 for row in rows
        ),
        "joint_contract_all_aggregate_reject": all(
            row["joint_contract_aggregate"]["p_value"] <= 0.05 for row in rows
        ),
        "term3_total_holm_rejections": int(
            sum(row["term3_holm_rejections"] for row in rows)
        ),
        "joint_contract_total_holm_rejections": int(
            sum(row["joint_contract_holm_rejections"] for row in rows)
        ),
        "total_class_tests": 10 * len(rows),
    }
    payload = {
        "manifest": {
            "schema_version": "v10-complete-contract-result-1.0.0",
            "protocol_version": PROTOCOL_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "post_specified_after_v9_feedback": True,
            "model_family": "F-CS-WAE",
            "dataset": "two-domain Rotated-MNIST",
            "seeds": args.seeds,
            "domains": [-30.0, 30.0],
            "samples_per_class_per_domain": EVAL_PER_CLASS_PER_DOMAIN,
            "permutations": args.permutations,
            "content_kernel": {
                "geometry": "chordal Euclidean distance on unit sphere",
                "sigmas": list(SPHERICAL_SIGMAS),
            },
            "style_kernel": {
                "geometry": "Euclidean",
                "sigmas": list(EUCLIDEAN_SIGMAS),
            },
            "joint_kernel": "product of content and style multiscale RBF kernels",
            "repository": _git_metadata(),
            "checkpoint_sha256": checkpoint_hashes,
            "split_manifest_sha256": split_hashes,
            "implementation_sha256": sha256_file(Path(__file__)),
        },
        "summary": summary,
        "records": rows,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, output)
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {output} ({digest})", flush=True)


if __name__ == "__main__":
    main()
