#!/usr/bin/env python3
"""Generate paper-ready F-CS-WAE CIFAR-10 result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_outputs" / "f_cs_wae_cifar10"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

METRICS = ["ACC", "NMI", "ARI", "FID", "LPIPS", "SSIM", "PSNR"]
PERCENT_METRICS = {"ACC", "NMI", "ARI", "SSIM"}

ABLATION_VARIANTS = [
    ("full", "F-CS-WAE (Full)"),
    ("no_z_s", "No style latent z_s"),
    ("no_class_mmd", "No supervised class MMD"),
    ("no_style_mmd", "No style MMD"),
    ("no_classifier", "No auxiliary classifier"),
    ("gaussian_class_prior", "Euclidean Gaussian class prior"),
    ("vmf_class_prior", "vMF class prior"),
    ("single_center", "Single center per class"),
    ("learnable_centers", "Learnable class centers"),
]


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def metric_value(metrics: dict, key: str) -> float:
    return float(metrics[key]) if key in metrics else float("nan")


def fmt_metric(key: str, value: float, digits: int = 2) -> str:
    if np.isnan(value):
        return "--"
    if key in PERCENT_METRICS:
        return f"{100.0 * value:.{digits}f}"
    return f"{value:.{digits}f}"


def fmt_mean_std(key: str, mean: float, std: float, digits: int = 2) -> str:
    if np.isnan(mean):
        return "--"
    if key in PERCENT_METRICS:
        return f"{100.0 * mean:.{digits}f} $\\pm$ {100.0 * std:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_f_seed_metrics() -> pd.DataFrame:
    rows = []
    for seed_dir in sorted((ROOT / "runs_f" / "cifar10").glob("seed_*")):
        metrics_path = seed_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        seed = int(seed_dir.name.split("_", 1)[1])
        metrics = read_json(metrics_path)
        rows.append({"seed": seed, **{m: metric_value(metrics, m) for m in METRICS}})
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def export_main_tables(seed_df: pd.DataFrame) -> pd.DataFrame:
    seed_df.to_csv(TABLE_DIR / "table_f01_f_cs_wae_cifar10_per_seed.csv", index=False)

    summary_rows = []
    for metric in METRICS:
        values = seed_df[metric].astype(float)
        summary_rows.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(ddof=0),
                "min": values.min(),
                "max": values.max(),
                "n": values.count(),
                "paper": fmt_mean_std(metric, values.mean(), values.std(ddof=0)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLE_DIR / "table_f01_f_cs_wae_cifar10_summary.csv", index=False)

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{F-CS-WAE results on CIFAR-10 over three random seeds. ACC, NMI, ARI, and SSIM are reported as percentages.}",
        r"\label{tab:f-cs-wae-cifar10-main}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Run & ACC$\uparrow$ & NMI$\uparrow$ & ARI$\uparrow$ & FID$\downarrow$ & LPIPS$\downarrow$ & SSIM$\uparrow$ & PSNR$\uparrow$ \\",
        r"\midrule",
    ]
    for _, row in seed_df.iterrows():
        vals = " & ".join(fmt_metric(m, float(row[m])) for m in METRICS)
        tex.append(f"Seed {int(row['seed'])} & {vals} \\\\")
    tex.append(r"\midrule")
    vals = " & ".join(summary_df.loc[summary_df["metric"] == m, "paper"].iloc[0] for m in METRICS)
    tex.append(f"Mean $\\pm$ std & {vals} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLE_DIR / "table_f01_f_cs_wae_cifar10_main.tex").write_text("\n".join(tex))

    return summary_df


def load_comparison_rows(seed_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add_row(method: str, source: str, metrics: dict, status: str) -> None:
        row = {"Method": method, "Source": source, "Status": status}
        row.update({m: metric_value(metrics, m) for m in METRICS})
        rows.append(row)

    mean_metrics = {
        row["metric"]: float(row["mean"]) for _, row in summary_df.iterrows()
    }
    add_row("F-CS-WAE", "runs_f/cifar10/seed_0..2", mean_metrics, "3-seed mean")

    seed0 = seed_df.loc[seed_df["seed"] == 0].iloc[0].to_dict()
    add_row("F-CS-WAE", "runs_f/cifar10/seed_0", seed0, "seed 0")

    full_path = ROOT / "runs_f" / "cifar10" / "f_ablation_20260625_183045" / "full" / "metrics.json"
    if full_path.exists():
        add_row("F-CS-WAE (full ablation)", str(full_path.relative_to(ROOT)), read_json(full_path), "completed")

    old_baselines = ROOT / "runs" / "cifar10" / "baselines" / "seed_0" / "comparison_results.csv"
    if old_baselines.exists():
        df = pd.read_csv(old_baselines, index_col=0)
        for method in ["WAE-MMD", "VAE", "VaDE"]:
            if method in df.index:
                add_row(method, str(old_baselines.relative_to(ROOT)), df.loc[method].to_dict(), "seed 0")

    resnet_path = ROOT / "runs_f" / "cifar10" / "baselines_extended" / "seed_0" / "ResNetAE" / "metrics.json"
    if resnet_path.exists():
        add_row("ResNetAE", str(resnet_path.relative_to(ROOT)), read_json(resnet_path), "seed 0")

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(TABLE_DIR / "table_f02_cifar10_comparison_available.csv", index=False)

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Available CIFAR-10 comparisons. F-CS-WAE is reported as both a 3-seed mean and seed-0 runs; remaining baselines are the completed seed-0 outputs available in the experiment directory. ACC, NMI, ARI, and SSIM are percentages.}",
        r"\label{tab:f-cs-wae-cifar10-comparison}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Method & Status & ACC$\uparrow$ & NMI$\uparrow$ & ARI$\uparrow$ & FID$\downarrow$ \\",
        r"\midrule",
    ]
    for _, row in comp_df.iterrows():
        method = row["Method"]
        if method.startswith("F-CS-WAE"):
            method = r"\textbf{" + method + "}"
        tex.append(
            f"{method} & {row['Status']} & {fmt_metric('ACC', row['ACC'])} & "
            f"{fmt_metric('NMI', row['NMI'])} & {fmt_metric('ARI', row['ARI'])} & "
            f"{fmt_metric('FID', row['FID'])} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLE_DIR / "table_f02_cifar10_comparison_available.tex").write_text("\n".join(tex))
    return comp_df


def export_ablation_status() -> pd.DataFrame:
    root = ROOT / "runs_f" / "cifar10" / "f_ablation_20260625_183045"
    rows = []
    for key, label in ABLATION_VARIANTS:
        variant_dir = root / key
        metrics_path = variant_dir / "metrics.json"
        if metrics_path.exists():
            metrics = read_json(metrics_path)
            status = "completed"
        elif variant_dir.exists():
            metrics = {}
            status = "interrupted, no metrics"
        else:
            metrics = {}
            status = "not completed"
        rows.append(
            {
                "Key": key,
                "Variant": label,
                "Status": status,
                **{m: metric_value(metrics, m) if metrics else np.nan for m in METRICS},
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "table_f03_ablation_status.csv", index=False)

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{F-CS-WAE ablation status for CIFAR-10. Only completed variants are reported with metrics. The run was interrupted by disk quota during logging after the full variant completed.}",
        r"\label{tab:f-cs-wae-ablation-status}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Variant & Status & ACC$\uparrow$ & NMI$\uparrow$ & FID$\downarrow$ \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        tex.append(
            f"{row['Variant']} & {row['Status']} & {fmt_metric('ACC', row['ACC'])} & "
            f"{fmt_metric('NMI', row['NMI'])} & {fmt_metric('FID', row['FID'])} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLE_DIR / "table_f03_ablation_status.tex").write_text("\n".join(tex))
    return df


def load_history(seed: int) -> pd.DataFrame:
    path = ROOT / "runs_f" / "cifar10" / f"seed_{seed}" / "training_history.json"
    data = read_json(path)
    df = pd.DataFrame(data)
    df.insert(0, "epoch", np.arange(1, len(df) + 1))
    df.insert(1, "seed", seed)
    return df


def plot_main_metrics(seed_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    seeds = [f"Seed {int(s)}" for s in seed_df["seed"]]

    acc = seed_df["ACC"].to_numpy() * 100.0
    axes[0].bar(seeds, acc, color="#2878B5")
    axes[0].axhline(float(summary_df.loc[summary_df["metric"] == "ACC", "mean"].iloc[0]) * 100.0, color="#C82423", linestyle="--", linewidth=1.2, label="Mean")
    axes[0].set_ylabel("ACC (%)")
    axes[0].set_title("CIFAR-10 Clustering Accuracy")
    axes[0].set_ylim(0, max(100, acc.max() + 5))
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    fid = seed_df["FID"].to_numpy()
    axes[1].bar(seeds, fid, color="#54B345")
    axes[1].axhline(float(summary_df.loc[summary_df["metric"] == "FID", "mean"].iloc[0]), color="#C82423", linestyle="--", linewidth=1.2, label="Mean")
    axes[1].set_ylabel("FID")
    axes[1].set_title("CIFAR-10 Generation Quality")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_f01_f_cs_wae_cifar10_main_metrics.png", dpi=300)
    fig.savefig(FIG_DIR / "fig_f01_f_cs_wae_cifar10_main_metrics.pdf")
    plt.close(fig)


def plot_training_curves() -> None:
    histories = [load_history(seed) for seed in (0, 1, 2)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = ["#2878B5", "#C82423", "#54B345"]

    for df, color in zip(histories, colors):
        seed = int(df["seed"].iloc[0])
        axes[0].plot(df["epoch"], df["total"], label=f"Seed {seed}", color=color, linewidth=1.2)
        axes[1].plot(df["epoch"], df["rec"], label=f"Seed {seed}", color=color, linewidth=1.2)

    for ax, title, ylabel in (
        (axes[0], "Training Total Loss", "Total loss"),
        (axes[1], "Reconstruction Term", "L_rec"),
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_f02_f_cs_wae_training_curves.png", dpi=300)
    fig.savefig(FIG_DIR / "fig_f02_f_cs_wae_training_curves.pdf")
    plt.close(fig)


def plot_comparison(comp_df: pd.DataFrame) -> None:
    plot_df = comp_df[comp_df["Status"].isin(["seed 0", "completed"])].copy()
    method_labels = plot_df["Method"].replace({"F-CS-WAE (full ablation)": "F-CS-WAE\n(full)"})

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].bar(method_labels, plot_df["ACC"] * 100.0, color="#2878B5")
    axes[0].set_ylabel("ACC (%)")
    axes[0].set_title("Available CIFAR-10 Accuracy")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(method_labels, plot_df["FID"], color="#54B345")
    axes[1].set_ylabel("FID")
    axes[1].set_title("Available CIFAR-10 FID")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_f03_cifar10_comparison_available.png", dpi=300)
    fig.savefig(FIG_DIR / "fig_f03_cifar10_comparison_available.pdf")
    plt.close(fig)


def write_summary(seed_df: pd.DataFrame, summary_df: pd.DataFrame, comp_df: pd.DataFrame, abl_df: pd.DataFrame) -> None:
    acc = summary_df.loc[summary_df["metric"] == "ACC"].iloc[0]
    fid = summary_df.loc[summary_df["metric"] == "FID"].iloc[0]
    nmi = summary_df.loc[summary_df["metric"] == "NMI"].iloc[0]
    ari = summary_df.loc[summary_df["metric"] == "ARI"].iloc[0]

    full = abl_df.loc[abl_df["Key"] == "full"].iloc[0]
    completed_ablation = int((abl_df["Status"] == "completed").sum())

    lines = [
        "# F-CS-WAE CIFAR-10 Paper Outputs",
        "",
        "Generated from repository-local experiment artifacts only.",
        "",
        "## Main Result",
        "",
        f"- F-CS-WAE over {len(seed_df)} seeds: ACC {100 * acc['mean']:.2f} ± {100 * acc['std']:.2f}, "
        f"NMI {100 * nmi['mean']:.2f} ± {100 * nmi['std']:.2f}, "
        f"ARI {100 * ari['mean']:.2f} ± {100 * ari['std']:.2f}, "
        f"FID {fid['mean']:.2f} ± {fid['std']:.2f}.",
        f"- Completed full ablation variant: ACC {100 * full['ACC']:.2f}, "
        f"NMI {100 * full['NMI']:.2f}, ARI {100 * full['ARI']:.2f}, FID {full['FID']:.2f}.",
        "",
        "## Caveats",
        "",
        f"- Ablation completion: {completed_ablation}/{len(abl_df)} variants have metrics. The run stopped during `no_z_s` because logging hit disk quota.",
        "- Extended baselines are partial: `ResNetAE` has metrics; `AEWithCE` has a checkpoint but no metrics file in the current output directory.",
        "- The CIFAR-10 comparison table marks whether each row is a 3-seed mean, seed-0 result, or completed ablation run.",
        "",
        "## Files",
        "",
        "- `tables/table_f01_f_cs_wae_cifar10_main.tex`: main per-seed + mean table.",
        "- `tables/table_f02_cifar10_comparison_available.tex`: available baseline comparison table.",
        "- `tables/table_f03_ablation_status.tex`: ablation completion/status table.",
        "- `figures/fig_f01_f_cs_wae_cifar10_main_metrics.*`: ACC/FID across seeds.",
        "- `figures/fig_f02_f_cs_wae_training_curves.*`: training curves over 300 epochs.",
        "- `figures/fig_f03_cifar10_comparison_available.*`: available ACC/FID comparison.",
    ]
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    ensure_dirs()
    seed_df = load_f_seed_metrics()
    if seed_df.empty:
        raise SystemExit("No F-CS-WAE seed metrics found under runs_f/cifar10/seed_*")

    summary_df = export_main_tables(seed_df)
    comp_df = load_comparison_rows(seed_df, summary_df)
    abl_df = export_ablation_status()

    plot_main_metrics(seed_df, summary_df)
    plot_training_curves()
    plot_comparison(comp_df)
    write_summary(seed_df, summary_df, comp_df, abl_df)

    print(f"Generated F-CS-WAE paper outputs under {OUT_DIR.relative_to(ROOT)}")
    print(f"Tables:  {TABLE_DIR.relative_to(ROOT)}")
    print(f"Figures: {FIG_DIR.relative_to(ROOT)}")
    print(f"Summary: {(OUT_DIR / 'SUMMARY.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
