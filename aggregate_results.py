"""
Aggregate multi-seed metrics into mean ± std summary tables.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLUMNS = ["ACC", "NMI", "ARI", "FID", "LPIPS", "SSIM", "PSNR"]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed run metrics")
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs/mnist",
        help="Root directory containing seed_* subdirectories",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="seed_*",
        help="Glob pattern for seed run directories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: <runs-dir>/aggregated_metrics.json)",
    )
    return parser.parse_args()


def load_seed_metrics(runs_dir: Path, pattern: str) -> dict[int, dict]:
    seed_metrics = {}
    for run_dir in sorted(runs_dir.glob(pattern)):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"Skip {run_dir.name}: no metrics.json")
            continue

        seed = None
        if run_dir.name.startswith("seed_"):
            try:
                seed = int(run_dir.name.split("_", 1)[1])
            except ValueError:
                pass

        with metrics_path.open() as f:
            metrics = json.load(f)

        key = seed if seed is not None else run_dir.name
        seed_metrics[key] = metrics
        print(f"Loaded {run_dir.name}: {metrics_path}")

    return seed_metrics


def aggregate_metrics(seed_metrics: dict) -> pd.DataFrame:
    rows = []
    for seed, metrics in seed_metrics.items():
        row = {"seed": seed}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("seed")
    available = [c for c in METRIC_COLUMNS if c in df.columns]
    summary = pd.DataFrame(index=available)

    for col in available:
        values = df[col].astype(float)
        summary.loc[col, "mean"] = values.mean()
        summary.loc[col, "std"] = values.std(ddof=0)
        summary.loc[col, "min"] = values.min()
        summary.loc[col, "max"] = values.max()
        summary.loc[col, "n"] = len(values)

    return df, summary


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    seed_metrics = load_seed_metrics(runs_dir, args.pattern)

    if not seed_metrics:
        raise SystemExit(f"No metrics found under {runs_dir}/{args.pattern}")

    per_seed_df, summary_df = aggregate_metrics(seed_metrics)

    output_json = Path(args.output) if args.output else runs_dir / "aggregated_metrics.json"
    output_csv = output_json.with_suffix(".csv")
    per_seed_csv = runs_dir / "per_seed_metrics.csv"

    per_seed_df.to_csv(per_seed_csv)
    summary_df.to_csv(output_csv)

    payload = {
        "n_seeds": len(seed_metrics),
        "seeds": list(seed_metrics.keys()),
        "per_seed": {str(k): v for k, v in seed_metrics.items()},
        "summary": summary_df.to_dict(orient="index"),
    }
    with output_json.open("w") as f:
        json.dump(payload, f, indent=2)

    print("\nPer-seed metrics:")
    print(per_seed_df.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nAggregated (mean ± std):")
    for metric in summary_df.index:
        mean = summary_df.loc[metric, "mean"]
        std = summary_df.loc[metric, "std"]
        print(f"  {metric}: {mean:.4f} ± {std:.4f}")

    print(f"\nSaved: {per_seed_csv}")
    print(f"Saved: {output_csv}")
    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()
