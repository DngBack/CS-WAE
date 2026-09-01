#!/usr/bin/env python3
"""Calibrate minibatch posterior-mean conditional HSIC on a checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_leakage_diagnostics import (
    get_eval_subset_loader,
    load_fcswae_checkpoint,
)
from src.utils.loss_f_cs_wae import mean_conditional_hsic_constraint
from src.utils.seed import set_seed


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": ordered[0], "median": statistics.median(ordered),
        "max": ordered[-1],
    }


def round_weight(value: float) -> float:
    if value <= 0.0:
        return 0.0
    exponent = math.floor(math.log10(value))
    scale = 10.0 ** exponent
    normalized = value / scale
    rounded = min((1.0, 2.0, 3.0, 5.0, 10.0), key=lambda x: abs(x - normalized))
    return rounded * scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=320)
    parser.add_argument("--null-draws", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batches < 1 or args.batch_size < 4 or args.null_draws < 1:
        raise ValueError("batches/null-draws must be positive and batch-size >= 4")
    set_seed(args.seed)
    device = torch.device(args.device)
    model = load_fcswae_checkpoint(args.checkpoint, "mnist", device)
    loader = get_eval_subset_loader(
        "mnist", args.batches * args.batch_size,
        batch_size=args.batch_size, seed=args.seed, split="train", num_workers=0,
    )
    rows = []
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= args.batches:
                break
            data, labels = batch[:2]
            data, labels = data.to(device), labels.to(device)
            mean_content, _rho, mean_style, _logvar = model.encode(data)
            excess, raw, null, represented = mean_conditional_hsic_constraint(
                mean_content,
                mean_style,
                labels,
                model.n_classes,
                null_draws=args.null_draws,
                style_only=True,
            )
            rows.append({
                "batch": index, "raw": float(raw), "null": float(null),
                "excess": float(excess), "represented_classes": represented,
            })
    if not rows:
        raise RuntimeError("calibration loader produced no batches")
    excess_mean = statistics.mean(row["excess"] for row in rows)
    target_contributions = [0.001, 0.003, 0.01]
    weights = [
        round_weight(target / max(excess_mean, 1e-12))
        for target in target_contributions
    ]
    payload = {
        "protocol": "fact-audit-aligned-mean-hsic-calibration-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(args.checkpoint).resolve().relative_to(ROOT)),
        "seed": args.seed, "batch_size": args.batch_size,
        "batches": len(rows), "null_draws": args.null_draws,
        "raw": summarize([row["raw"] for row in rows]),
        "null": summarize([row["null"] for row in rows]),
        "positive_excess": summarize([row["excess"] for row in rows]),
        "target_objective_contributions": target_contributions,
        "suggested_weights": weights,
        "rows": rows,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({
        "raw": payload["raw"], "null": payload["null"],
        "positive_excess": payload["positive_excess"],
        "suggested_weights": weights,
    }, indent=2))


if __name__ == "__main__":
    main()
