"""
plot_delta_pareto.py

Aggregates the F-CS-WAE per-class-style-MMD (delta) sweep into a summary
table and three Pareto scatter plots, extending the paper's two-point
(delta=0 vs delta=1) tradeoff (Figure 7 / sec:tradeoff, Limitations item 2)
into a 6-point sweep: delta in {0.0, 0.03, 0.1, 0.3, 1.0, 3.0}.

Merges, per (dataset, delta):
  - runs_f/{dataset}/<run_dir>/metrics.json           (ACC, NMI, ARI, FID, ...)
  - runs_diag/delta_sweep/{dataset}_delta{delta}.json (global_mmd, delta_inter,
                                                        lp_accuracy, hsic,
                                                        gen_self_acc_*)

Idempotent / re-runnable: point it at whatever (dataset, delta) files exist
right now (e.g. only the two original delta=0/1 checkpoints before the rest
of the sweep finishes) -- missing points are skipped with a warning, not a
crash, so this can (and should) be validated before the sweep completes.

Usage
-----
python scripts/plot_delta_pareto.py
python scripts/plot_delta_pareto.py --deltas 0.0 1.0   # just the 2 existing extremes
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_delta_sweep import resolve_run_dir

DEFAULT_DELTAS = ["0.0", "0.03", "0.1", "0.3", "1.0", "3.0"]
DEFAULT_DATASETS = ["mnist", "cifar10"]


def load_point(dataset: str, delta: str, seed: int):
    """Return a merged dict for one (dataset, delta) point, or None if missing."""
    run_dir = resolve_run_dir(dataset, seed, delta)
    metrics_path = run_dir / "metrics.json"
    diag_path = ROOT / "runs_diag" / "delta_sweep" / f"{dataset}_delta{delta}.json"

    if not metrics_path.exists():
        print(f"  [skip] {dataset} delta={delta}: missing {metrics_path}")
        return None
    if not diag_path.exists():
        print(f"  [skip] {dataset} delta={delta}: missing {diag_path}")
        return None

    with metrics_path.open() as f:
        metrics = json.load(f)
    with diag_path.open() as f:
        diag = json.load(f)

    return {
        "dataset": dataset,
        "delta": float(delta),
        "ACC": metrics.get("ACC"),
        "NMI": metrics.get("NMI"),
        "ARI": metrics.get("ARI"),
        "FID": metrics.get("FID"),
        "SSIM": metrics.get("SSIM"),
        "LPIPS": metrics.get("LPIPS"),
        "global_mmd": diag.get("global_mmd"),
        "delta_inter": diag.get("delta_inter"),
        "lp_accuracy": diag.get("lp_accuracy"),
        "hsic": diag.get("hsic"),
        "gen_self_acc_global_gaussian": diag.get("gen_self_acc_global_gaussian"),
    }


def plot_pareto(points_by_dataset, x_key, y_key, title, xlabel, ylabel, out_path, chance_line=None):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = {"mnist": "tab:blue", "cifar10": "tab:orange"}
    plotted_any = False

    for dataset, points in points_by_dataset.items():
        pts = [p for p in points if p.get(x_key) is not None and p.get(y_key) is not None]
        pts = sorted(pts, key=lambda p: p["delta"])
        if not pts:
            continue
        plotted_any = True
        xs = [p[x_key] for p in pts]
        ys = [p[y_key] for p in pts]
        ax.plot(xs, ys, marker="o", color=colors.get(dataset), label=dataset)
        for p in pts:
            ax.annotate(
                f"δ={p['delta']:g}", (p[x_key], p[y_key]), fontsize=7,
                textcoords="offset points", xytext=(4, 4),
            )

    if not plotted_any:
        plt.close(fig)
        print(f"  [skip-plot] {out_path.name}: no points with both {x_key} and {y_key}")
        return

    if chance_line is not None:
        ax.axvline(chance_line, color="gray", linestyle="--", linewidth=1, label=f"chance = {chance_line:.2f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.text(0.5, -0.02, "Single seed per δ; no error bars.", ha="center", fontsize=7, style="italic")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate the F-CS-WAE delta sweep into Pareto plots")
    p.add_argument("--deltas", nargs="+", default=DEFAULT_DELTAS)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=["mnist", "cifar10"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None, help="Defaults to runs_diag/delta_sweep/")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "runs_diag" / "delta_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading available (dataset, delta) points...")
    points_by_dataset = {}
    all_points = []
    for dataset in args.datasets:
        points = []
        for delta in args.deltas:
            pt = load_point(dataset, delta, args.seed)
            if pt is not None:
                points.append(pt)
                all_points.append(pt)
        points_by_dataset[dataset] = points

    if not all_points:
        print("No (dataset, delta) points found yet — nothing to plot.")
        return

    with (out_dir / "summary_table.json").open("w") as f:
        json.dump(all_points, f, indent=2)
    with (out_dir / "summary_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_points[0].keys()))
        writer.writeheader()
        writer.writerows(all_points)
    print(f"Wrote summary table ({len(all_points)} points) to {out_dir}/summary_table.{{csv,json}}")

    print("Plotting...")
    plot_pareto(
        points_by_dataset, "lp_accuracy", "ACC",
        "Clustering ACC vs style-leakage linear-probe accuracy",
        "LP(z_s -> y) accuracy", "Clustering ACC",
        out_dir / "pareto_acc_vs_lp.png", chance_line=0.10,
    )
    plot_pareto(
        points_by_dataset, "lp_accuracy", "gen_self_acc_global_gaussian",
        "Naive-sampling self-ACC vs style-leakage linear-probe accuracy",
        "LP(z_s -> y) accuracy", "Gen self-ACC (z_s ~ N(0,I))",
        out_dir / "pareto_selfacc_vs_lp.png", chance_line=0.10,
    )
    plot_pareto(
        points_by_dataset, "delta_inter", "FID",
        "Generation quality (FID) vs inter-class style separation",
        "Δ_inter (↓ = less leakage)", "FID (↓ = better)",
        out_dir / "pareto_fid_vs_deltainter.png",
    )

    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
