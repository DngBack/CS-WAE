#!/usr/bin/env python3
"""Official SCFlow cross-image recombination audit in CLIP embedding space.

The test grid constructs the exact forward task used by SCFlow: a content
donor with the requested content but a different style, a style donor with the
requested style but a different content, and the held-out target combination.
It audits reverse-endpoint compatibility, cross-factor leakage, forward
retrieval competence, and a forward/reverse cycle.  It does not claim pixel
fidelity because the separate UnCLIP decoder is intentionally out of scope.
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

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SCFLOW = ROOT / "external" / "SCFlow"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCFLOW))

from scflow.trainer_module import TrainerModuleSCFlow  # noqa: E402
from src.metrics.audit_protocol import (  # noqa: E402
    energy_distance_permutation_test,
    hsic_permutation_test,
    mmd2_permutation_test,
)


PROTOCOL = "scflow-contract-audit-1.1.0"
GROUP_SIZE = 3000
N_STYLES = 51
EMBEDDING_DIM = 768
EXPECTED_CHECKPOINT_BYTES = 9_904_269_044
EXPECTED_TEST_BYTES = 337_096_482


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="external/SCFlow/ckpts/scflow_last.ckpt")
    parser.add_argument("--test-h5", default="external/SCFlow/dataset/test.h5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-content", type=int, default=10)
    parser.add_argument("--n-styles", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ode-steps", type=int, default=1)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", default="runs_modern/scflow_audit/pilot.json")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "null_values"}


def decode_h5_strings(values: np.ndarray) -> np.ndarray:
    """Decode variable- and fixed-width HDF5 strings as UTF-8."""
    return np.asarray(
        [
            value.decode("utf-8")
            if isinstance(value, (bytes, np.bytes_))
            else str(value)
            for value in values
        ]
    )


def cosine_summary(left: torch.Tensor, right: torch.Tensor) -> dict:
    values = F.cosine_similarity(left.float(), right.float(), dim=1).cpu().numpy()
    return {
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.50)),
    }


def retrieval(
    queries: torch.Tensor,
    bank: torch.Tensor,
    desired_content: torch.Tensor,
    desired_style: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> dict:
    bank_normalized = F.normalize(bank.float(), dim=1)
    predictions = []
    similarities = []
    for start in range(0, queries.shape[0], chunk_size):
        query = F.normalize(queries[start : start + chunk_size].float(), dim=1)
        scores = query @ bank_normalized.T
        similarity, index = scores.max(dim=1)
        predictions.append(index)
        similarities.append(similarity)
    predicted = torch.cat(predictions)
    similarity = torch.cat(similarities)
    predicted_style = torch.div(predicted, GROUP_SIZE, rounding_mode="floor")
    predicted_content = predicted.remainder(GROUP_SIZE)
    content_correct = predicted_content == desired_content
    style_correct = predicted_style == desired_style
    return {
        "content_accuracy": float(content_correct.float().mean().item()),
        "style_accuracy": float(style_correct.float().mean().item()),
        "joint_accuracy": float((content_correct & style_correct).float().mean().item()),
        "nearest_cosine_mean": float(similarity.mean().item()),
    }


def build_grid(bank: torch.Tensor, args: argparse.Namespace) -> dict:
    if not 2 <= args.n_content <= GROUP_SIZE:
        raise ValueError(f"n-content must be in [2, {GROUP_SIZE}]")
    if not 2 <= args.n_styles <= N_STYLES:
        raise ValueError(f"n-styles must be in [2, {N_STYLES}]")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    contents = torch.randperm(GROUP_SIZE, generator=generator)[: args.n_content]
    styles = torch.randperm(N_STYLES, generator=generator)[: args.n_styles]
    desired_content = contents.repeat_interleave(args.n_styles)
    desired_style = styles.repeat(args.n_content)
    n = desired_content.numel()

    content_offsets = torch.randint(1, N_STYLES, (n,), generator=generator)
    content_donor_style = (desired_style + content_offsets) % N_STYLES
    style_donor_content = torch.randint(0, GROUP_SIZE, (n,), generator=generator)
    collision = style_donor_content == desired_content
    style_donor_content[collision] = (style_donor_content[collision] + 1) % GROUP_SIZE

    target_index = desired_style * GROUP_SIZE + desired_content
    content_index = content_donor_style * GROUP_SIZE + desired_content
    style_index = desired_style * GROUP_SIZE + style_donor_content
    return {
        "content_ids": contents,
        "style_ids": styles,
        "desired_content": desired_content,
        "desired_style": desired_style,
        "target": bank[target_index],
        "content_donor": bank[content_index],
        "style_donor": bank[style_index],
        "indices": {
            "target": target_index,
            "content_donor": content_index,
            "style_donor": style_index,
        },
    }


def load_module(checkpoint: Path, device: torch.device, ode_steps: int):
    config = OmegaConf.load(SCFLOW / "configs/inference.yaml")
    module = TrainerModuleSCFlow.load_from_checkpoint(
        checkpoint_path=str(checkpoint),
        fm_cfg=config.model.fm,
        test_vis=False,
        unclip_ckpt=None,
        val_step_num=ode_steps,
        reverse_inference=False,
        strict=False,
        map_location="cpu",
    )
    module.eval().to(device)
    return module


@torch.no_grad()
def run_flow(module, grid: dict, args: argparse.Namespace, device: torch.device) -> dict:
    source_outputs, target_outputs, cycle_outputs = [], [], []
    target = grid["target"]
    content_donor = grid["content_donor"]
    style_donor = grid["style_donor"]
    for start in range(0, target.shape[0], args.batch_size):
        stop = start + args.batch_size
        batch_target = target[start:stop].to(device).unsqueeze(1)
        batch_source = torch.cat(
            [content_donor[start:stop], style_donor[start:stop]], dim=1
        ).to(device).unsqueeze(1)
        forward = module.predict_forward(batch_source)
        reverse_target = module.predict_reverse(
            torch.cat([batch_target, batch_target], dim=2)
        )
        reverse_cycle = module.predict_reverse(forward)
        source_outputs.append(forward.cpu())
        target_outputs.append(reverse_target.cpu())
        cycle_outputs.append(reverse_cycle.cpu())
    return {
        "forward": torch.cat(source_outputs).squeeze(1),
        "reverse_target": torch.cat(target_outputs).squeeze(1),
        "reverse_cycle": torch.cat(cycle_outputs).squeeze(1),
    }


def main() -> None:
    args = parse_args()
    checkpoint = ROOT / args.checkpoint
    test_h5 = ROOT / args.test_h5
    if checkpoint.stat().st_size != EXPECTED_CHECKPOINT_BYTES:
        raise RuntimeError("SCFlow checkpoint is incomplete or differs from the gated artifact")
    if test_h5.stat().st_size != EXPECTED_TEST_BYTES:
        raise RuntimeError("SCFlow test split is incomplete or differs from the gated artifact")
    with h5py.File(test_h5, "r") as handle:
        bank = torch.from_numpy(np.asarray(handle["images"], dtype=np.float32))
        content_keys = decode_h5_strings(
            np.asarray(handle["metadata/content_idx"][:GROUP_SIZE])
        )
        style_names = decode_h5_strings(
            np.asarray(handle["metadata/style_name"][::GROUP_SIZE])
        )
    if bank.shape != (N_STYLES * GROUP_SIZE, EMBEDDING_DIM):
        raise RuntimeError(f"unexpected SCFlow test bank shape {tuple(bank.shape)}")
    grid = build_grid(bank, args)
    device = torch.device(args.device)
    module = load_module(checkpoint, device, args.ode_steps)
    outputs = run_flow(module, grid, args, device)
    del module
    torch.cuda.empty_cache()

    forward = outputs["forward"]
    predicted_target = (forward[:, :EMBEDDING_DIM] + forward[:, EMBEDDING_DIM:]) / 2.0
    reverse = outputs["reverse_target"]
    reverse_content = reverse[:, :EMBEDDING_DIM]
    reverse_style = reverse[:, EMBEDDING_DIM:]
    cycle = outputs["reverse_cycle"]
    cycle_content = cycle[:, :EMBEDDING_DIM]
    cycle_style = cycle[:, EMBEDDING_DIM:]
    source_joint = torch.cat([grid["content_donor"], grid["style_donor"]], dim=1)

    device_for_tests = device if device.type == "cuda" else torch.device("cpu")
    reverse_joint_device = reverse.to(device_for_tests)
    source_joint_device = source_joint.to(device_for_tests)
    direct_mmd = mmd2_permutation_test(
        reverse_joint_device,
        source_joint_device,
        seed=args.seed + 100,
        n_permutations=args.permutations,
    )
    direct_energy = energy_distance_permutation_test(
        reverse_joint_device,
        source_joint_device,
        seed=args.seed + 100,
        n_permutations=args.permutations,
    )
    content_mmd = mmd2_permutation_test(
        reverse_content.to(device_for_tests),
        grid["content_donor"].to(device_for_tests),
        seed=args.seed + 110,
        n_permutations=args.permutations,
    )
    content_energy = energy_distance_permutation_test(
        reverse_content.to(device_for_tests),
        grid["content_donor"].to(device_for_tests),
        seed=args.seed + 110,
        n_permutations=args.permutations,
    )
    style_mmd = mmd2_permutation_test(
        reverse_style.to(device_for_tests),
        grid["style_donor"].to(device_for_tests),
        seed=args.seed + 120,
        n_permutations=args.permutations,
    )
    style_energy = energy_distance_permutation_test(
        reverse_style.to(device_for_tests),
        grid["style_donor"].to(device_for_tests),
        seed=args.seed + 120,
        n_permutations=args.permutations,
    )
    content_local = torch.arange(args.n_content).repeat_interleave(args.n_styles)
    style_local = torch.arange(args.n_styles).repeat(args.n_content)
    content_style_leakage = hsic_permutation_test(
        reverse_content.to(device_for_tests),
        style_local.to(device_for_tests),
        seed=args.seed + 200,
        n_permutations=args.permutations,
    )
    style_content_leakage = hsic_permutation_test(
        reverse_style.to(device_for_tests),
        content_local.to(device_for_tests),
        seed=args.seed + 300,
        n_permutations=args.permutations,
    )

    bank_device = bank.to(device_for_tests)
    desired_content = grid["desired_content"].to(device_for_tests)
    desired_style = grid["desired_style"].to(device_for_tests)
    results = {
        "sampler_compatibility": {
            "direct_joint_mmd": compact(direct_mmd),
            "direct_joint_energy": compact(direct_energy),
            "decision_agreement": (
                direct_mmd["p_value"] <= 0.05
            ) == (direct_energy["p_value"] <= 0.05),
            "content_marginal_mmd": compact(content_mmd),
            "content_marginal_energy": compact(content_energy),
            "style_marginal_mmd": compact(style_mmd),
            "style_marginal_energy": compact(style_energy),
            "reverse_content_style_hsic": compact(content_style_leakage),
            "reverse_style_content_hsic": compact(style_content_leakage),
        },
        "decoder_transport": {
            "target_cosine": cosine_summary(predicted_target, grid["target"]),
            "source_mean_target_cosine": cosine_summary(
                (grid["content_donor"] + grid["style_donor"]) / 2.0,
                grid["target"],
            ),
            "cycle_content_cosine": cosine_summary(cycle_content, grid["content_donor"]),
            "cycle_style_cosine": cosine_summary(cycle_style, grid["style_donor"]),
            "forward_half_agreement": cosine_summary(
                forward[:, :EMBEDDING_DIM], forward[:, EMBEDDING_DIM:]
            ),
        },
        "clip_space_competence": {
            "forward_target_retrieval": retrieval(
                predicted_target.to(device_for_tests),
                bank_device,
                desired_content,
                desired_style,
            ),
            "source_mean_baseline_retrieval": retrieval(
                ((grid["content_donor"] + grid["style_donor"]) / 2.0).to(device_for_tests),
                bank_device,
                desired_content,
                desired_style,
            ),
            "reverse_content_retrieval": retrieval(
                reverse_content.to(device_for_tests),
                bank_device,
                desired_content,
                desired_style,
            ),
            "reverse_style_retrieval": retrieval(
                reverse_style.to(device_for_tests),
                bank_device,
                desired_content,
                desired_style,
            ),
        },
    }
    payload = {
        "protocol": {
            "version": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "post_specified": True,
            "model": "official SCFlow (ICCV 2025)",
            "model_repository_commit": subprocess.check_output(
                ["git", "-C", str(SCFLOW), "rev-parse", "HEAD"], text=True
            ).strip(),
            "device": str(device),
            "seed": args.seed,
            "n_content": args.n_content,
            "n_styles": args.n_styles,
            "n_combinations": int(grid["target"].shape[0]),
            "ode_steps": args.ode_steps,
            "permutations": args.permutations,
            "factor_population": "Cartesian subset of official 51-style x 3000-content test grid",
            "claim_boundary": "CLIP-space compatibility/transport/competence; no pixel fidelity or diversity claim",
            "checkpoint": {
                "path": str(checkpoint.relative_to(ROOT)),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
            },
            "test_h5": {
                "path": str(test_h5.relative_to(ROOT)),
                "bytes": test_h5.stat().st_size,
                "sha256": sha256(test_h5),
            },
            "selected_content_ids": grid["content_ids"].tolist(),
            "selected_style_ids": grid["style_ids"].tolist(),
            "selected_content_keys": [
                str(content_keys[index]) for index in grid["content_ids"].tolist()
            ],
            "selected_style_names": [
                str(style_names[index]) for index in grid["style_ids"].tolist()
            ],
        },
        "results": results,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(results, indent=2), flush=True)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
