#!/usr/bin/env python3
"""Create compact JSON/CSV/figures/report for a two-arm face audit pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--treatment-dir", required=True)
    parser.add_argument("--treatment-label", default="FACT transfer")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _read(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _metric_rows(label: str, audit: dict) -> list[dict]:
    rows = []
    for latent, result in audit["direct_conditional_equality"].items():
        rows.append({
            "arm": label,
            "family": "direct_conditional_equality",
            "metric": latent,
            "statistic": result["statistic"],
            "p_value": result["p_value"],
            "reject_0.05": result["reject_0.05"],
        })
    for latent, result in audit["condition_dependence"].items():
        rows.append({
            "arm": label,
            "family": "condition_dependence",
            "metric": latent,
            "statistic": result["statistic"],
            "p_value": result["p_value"],
            "reject_0.05": result["reject_0.05"],
        })
    return rows


def _save_audit_figure(rows: list[dict], output: Path) -> None:
    direct_names = ["content", "style", "joint"]
    leakage_names = ["content", "style"]
    arms = list(dict.fromkeys(row["arm"] for row in rows))
    colors = ["#4472C4", "#ED7D31"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for axis, family, names, title in [
        (axes[0], "direct_conditional_equality", direct_names, "Conditional MMD$^2$ (absolute)"),
        (axes[1], "condition_dependence", leakage_names, "Condition dependence (HSIC)"),
    ]:
        x = np.arange(len(names), dtype=float)
        width = 0.36
        for arm_index, arm in enumerate(arms):
            selected = {
                row["metric"]: row for row in rows
                if row["arm"] == arm and row["family"] == family
            }
            values = [selected[name]["statistic"] for name in names]
            bars = axis.bar(
                x + (arm_index - (len(arms) - 1) / 2) * width,
                values,
                width,
                color=colors[arm_index % len(colors)],
                label=arm,
            )
            for bar, name in zip(bars, names):
                result = selected[name]
                marker = "R" if result["reject_0.05"] else "NR"
                axis.annotate(
                    f"{marker}\np={result['p_value']:.3g}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7,
                )
        axis.set_xticks(x, names)
        axis.set_title(title)
        axis.set_ylabel("test statistic")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("UTKFace continuous-age pilot (R = reject, NR = do not reject at 0.05)")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _save_training_figure(histories: dict[str, list[dict]], output: Path) -> None:
    keys = ["rec", "total", "style_mmd", "agg_mmd", "fact_label_dependence"]
    available = [key for key in keys if any(key in epoch for history in histories.values() for epoch in history)]
    fig, axes = plt.subplots(1, len(available), figsize=(3.5 * len(available), 3.3), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, key in zip(axes, available):
        for label, history in histories.items():
            axis.plot(np.arange(1, len(history) + 1), [epoch.get(key, np.nan) for epoch in history], label=label)
        axis.set_title(key)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("epoch mean")
    axes[0].legend(frameon=False)
    fig.suptitle("Training diagnostics (descriptive only)")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    control_dir, treatment_dir = Path(args.control_dir), Path(args.treatment_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audits = {
        "Control": _read(control_dir / "test_audit.json"),
        args.treatment_label: _read(treatment_dir / "test_audit.json"),
    }
    histories = {
        "Control": _read(control_dir / "training_history.json"),
        args.treatment_label: _read(treatment_dir / "training_history.json"),
    }
    reconstructions = {
        "Control": _read(control_dir / "test_reconstruction.json"),
        args.treatment_label: _read(treatment_dir / "test_reconstruction.json"),
    }
    run_configs = {
        "Control": _read(control_dir / "run_config.json"),
        args.treatment_label: _read(treatment_dir / "run_config.json"),
    }
    rows = [row for label, audit in audits.items() for row in _metric_rows(label, audit)]
    csv_path = output / "audit_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _save_audit_figure(rows, output / "audit_comparison.png")
    _save_training_figure(histories, output / "training_curves.png")

    probes = {label: audit.get("continuous_probes", {}) for label, audit in audits.items()}
    summary = {
        "scope": {
            "dataset": "UTKFace",
            "condition": "continuous chronological age",
            "arms": list(audits),
            "seeds": [0],
            "status": "exploratory one-pair pilot; not multi-seed confirmatory evidence",
        },
        "direct_conditional_equality": {
            label: audit["direct_conditional_equality"] for label, audit in audits.items()
        },
        "condition_dependence": {
            label: audit["condition_dependence"] for label, audit in audits.items()
        },
        "latent_descriptives": {
            label: audit["latent_descriptives"] for label, audit in audits.items()
        },
        "continuous_probes": probes,
        "held_out_reconstruction": reconstructions,
        "run_configs": run_configs,
        "training_final_epoch": {label: history[-1] for label, history in histories.items()},
    }
    (output / "pilot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = [
        "# UTKFace continuous-age pilot", "",
        "This is an exploratory matched seed-0 run, not multi-seed confirmatory evidence. ",
        "FACT hyperparameters were transferred without tuning from the earlier setup.", "",
        "## Direct conditional equality", "",
        "| Arm | Latent | Absolute MMD² | p | Decision at 0.05 |", "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["family"] != "direct_conditional_equality":
            continue
        decision = "reject equality" if row["reject_0.05"] else "do not reject"
        lines.append(f"| {row['arm']} | {row['metric']} | {row['statistic']:.6g} | {row['p_value']:.4g} | {decision} |")
    lines += ["", "The MMD estimator is unbiased and can be negative at finite sample size; a negative value is not evidence of equality.", ""]
    lines += ["## Continuous-age dependence", "", "| Arm | Latent | HSIC | p | Decision at 0.05 |", "|---|---:|---:|---:|---|"]
    for row in rows:
        if row["family"] != "condition_dependence":
            continue
        decision = "reject independence" if row["reject_0.05"] else "do not reject"
        lines.append(f"| {row['arm']} | {row['metric']} | {row['statistic']:.6g} | {row['p_value']:.4g} | {decision} |")
    lines += ["", "## Demographic-attribute dependence", "", "| Arm | Attribute | Latent | HSIC | p | Decision at 0.05 |", "|---|---|---:|---:|---:|---|"]
    for label, audit in audits.items():
        for attribute, latent_results in audit.get("content_style_attribute_dependence", {}).items():
            for latent, result in latent_results.items():
                reject = result.get("reject_0.05", result["p_value"] <= 0.05)
                decision = "reject independence" if reject else "do not reject"
                lines.append(
                    f"| {label} | {attribute} | {latent} | {result['statistic']:.6g} | "
                    f"{result['p_value']:.4g} | {decision} |"
                )
    lines += ["", "## Continuous-age probes", "", "| Arm | Probe | MAE (years) | R² |", "|---|---|---:|---:|"]
    for label, arm_probes in probes.items():
        for name, result in arm_probes.items():
            if result is not None:
                lines.append(f"| {label} | {name} | {result['mae_years']:.4g} | {result['r2']:.4g} |")
    lines += ["", "## Held-out reconstruction", "", "| Arm | Reconstruction | L1/pixel | PSNR (dB) |", "|---|---|---:|---:|"]
    for label, result in reconstructions.items():
        for key, display in [
            ("posterior_center_reconstruction", "posterior center"),
            ("fixed_one_draw_posterior_reconstruction", "fixed posterior draw"),
        ]:
            metrics = result[key]
            lines.append(
                f"| {label} | {display} | {metrics['l1_per_pixel']['mean']:.5g} | "
                f"{metrics['psnr_db_from_mean_mse']:.4g} |"
            )
    lines += ["", "## Absolute style scale", "", "| Arm | Bank | Median norm | 99th pct. norm | Max norm | Mean coordinate SD |", "|---|---|---:|---:|---:|---:|"]
    for label, audit in audits.items():
        for key, display in [("posterior_style", "posterior"), ("sampler_style", "sampler")]:
            values = audit["latent_descriptives"][key]
            norms, std = values["row_l2_norm"], values["coordinate_std"]
            lines.append(
                f"| {label} | {display} | {norms['q50']:.5g} | {norms['q99']:.5g} | "
                f"{norms['max']:.5g} | {std['mean']:.5g} |"
            )
    lines += [
        "", "## Interpretation guardrails", "",
        "- Report absolute post-training statistics and equality decisions, not only relative reductions.",
        "- A non-rejection is not evidence that the two conditional laws are practically equivalent.",
        "- The ridge probe measures decodability from a held-out split; it does not establish causal disentanglement.",
        "- Generated grids are qualitative checks and are not age-retention scores.",
        "- Absolute latent scale must be inspected alongside standardized-kernel MMD; otherwise rare variance explosions can be hidden.", "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"output_dir": str(output), "files": sorted(path.name for path in output.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
