"""
End-to-end CS-WAE experiment pipeline (Python replacement for shell scripts).

Examples:
  python run_pipeline.py all --dataset cifar10 --device cuda:0 --backbone cnn
  python run_pipeline.py all --dataset mnist --device cuda:0 --device2 cuda:1
  python run_pipeline.py train --dataset fashion_mnist --seeds 0 1 2 --device cuda:0
  python run_pipeline.py aggregate --runs-dir runs/mnist
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from src.datasets.loaders import SUPPORTED_DATASETS, get_default_runs_dir

ROOT = Path(__file__).resolve().parent
ABLATION_VARIANTS = ["baseline", "no_sup_mmd", "minimal", "euclidean", "vmf_prior"]
RGB_DATASETS = {"cifar10", "svhn"}
HEAVY_VIZ_SKIP = {"emnist_letters", "cifar10", "svhn"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CS-WAE experiment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--dataset",
            type=str,
            required=True,
            choices=list(SUPPORTED_DATASETS),
        )
        p.add_argument("--device", type=str, default="cuda:0")
        p.add_argument(
            "--device2",
            type=str,
            default=None,
            help="Second GPU for parallel seed-0/1 or ablation/baseline splits",
        )
        p.add_argument(
            "--backbone",
            type=str,
            default=None,
            choices=["cnn", "resnet18"],
            help="Encoder backbone (default: cnn; resnet18 suggested for cifar10/svhn)",
        )
        p.add_argument("--runs-dir", type=str, default=None)
        p.add_argument("--epochs", type=int, default=None)
        p.add_argument("--log-file", type=str, default=None)
        p.add_argument("--skip-viz", action="store_true")
        p.add_argument("--skip-advanced-viz", action="store_true")

    train_p = sub.add_parser("train", help="Multi-seed CS-WAE training")
    add_common(train_p)
    train_p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    ablation_p = sub.add_parser("ablation", help="Ablation study (5 variants)")
    add_common(ablation_p)
    ablation_p.add_argument("--seed", type=int, default=0)
    ablation_p.add_argument("--results-dir", type=str, default=None)

    baseline_p = sub.add_parser("baselines", help="Train and compare baselines")
    add_common(baseline_p)
    baseline_p.add_argument("--seed", type=int, default=0)
    baseline_p.add_argument("--output-dir", type=str, default=None)

    agg_p = sub.add_parser("aggregate", help="Aggregate multi-seed metrics")
    agg_p.add_argument("--runs-dir", type=str, required=True)
    agg_p.add_argument("--log-file", type=str, default=None)

    all_p = sub.add_parser("all", help="Full pipeline: train → ablation → baselines → aggregate")
    add_common(all_p)
    all_p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    return parser.parse_args()


class Pipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dataset = args.dataset
        self.backbone = resolve_backbone(args.dataset, args.backbone)
        self.runs_dir = Path(
            args.runs_dir
            if getattr(args, "runs_dir", None)
            else get_default_runs_dir(self.dataset, self.backbone)
        )
        self.device = args.device
        self.device2 = args.device2
        self.parallel = self.device2 is not None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = Path(
            args.log_file
            if getattr(args, "log_file", None)
            else f"logs/pipeline_{self.dataset}_{self.backbone}_{self.timestamp}.log"
        )
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self.log_file.open("a") as f:
            f.write(line + "\n")

    def train_viz_flags(self, seed: int) -> list[str]:
        flags: list[str] = []
        if self.args.skip_viz or self.dataset in RGB_DATASETS:
            flags.append("--skip-viz")
        if self.args.skip_advanced_viz or self.dataset in HEAVY_VIZ_SKIP:
            flags.append("--skip-advanced-viz")
        elif seed != 0:
            flags.append("--skip-advanced-viz")
        return flags

    def run_cmd(self, cmd: list[str], label: str = "") -> None:
        if label:
            self.log(f"START [{label}] {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        with self.log_file.open("a") as f:
            if result.stdout:
                f.write(result.stdout)
            if result.stderr:
                f.write(result.stderr)
        if result.returncode != 0:
            self.log(f"FAILED [{label}] exit={result.returncode}")
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            raise subprocess.CalledProcessError(result.returncode, cmd)

    def _train_seed_cmd(self, seed: int, device: str) -> list[str]:
        cmd = [
            sys.executable,
            "train_cs_wae.py",
            "--dataset",
            self.dataset,
            "--seed",
            str(seed),
            "--device",
            device,
            "--output-dir",
            str(self.runs_dir / f"seed_{seed}"),
            "--backbone",
            self.backbone,
            *self.train_viz_flags(seed),
        ]
        if self.args.epochs is not None:
            cmd.extend(["--epochs", str(self.args.epochs)])
        return cmd

    def phase_train(self, seeds: list[int]) -> None:
        self.log(
            f"Training seeds={seeds} dataset={self.dataset} backbone={self.backbone} "
            f"device={self.device} device2={self.device2}"
        )
        if self.parallel and len(seeds) >= 2:
            first, second = seeds[0], seeds[1]
            rest = seeds[2:]
            with ProcessPoolExecutor(max_workers=2) as pool:
                futures = {
                    pool.submit(
                        subprocess.run,
                        self._train_seed_cmd(first, self.device),
                        cwd=ROOT,
                        check=True,
                    ): first,
                    pool.submit(
                        subprocess.run,
                        self._train_seed_cmd(second, self.device2),
                        cwd=ROOT,
                        check=True,
                    ): second,
                }
                for fut in as_completed(futures):
                    seed = futures[fut]
                    fut.result()
                    self.log(f"Done seed {seed}")
            for seed in rest:
                self.run_cmd(self._train_seed_cmd(seed, self.device), f"seed_{seed}")
                self.log(f"Done seed {seed}")
        else:
            for seed in seeds:
                self.run_cmd(self._train_seed_cmd(seed, self.device), f"seed_{seed}")
                self.log(f"Done seed {seed}")

    def phase_ablation(self, results_dir: Path | None = None) -> Path:
        ablation_dir = Path(results_dir or self.runs_dir / f"ablation_{self.timestamp}")
        ablation_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"Ablation → {ablation_dir}")

        def ablation_cmd(variants: list[str], device: str, skip_aggregate: bool) -> list[str]:
            cmd = [
                sys.executable,
                "run_ablation_study.py",
                "--dataset",
                self.dataset,
                "--seed",
                "0",
                "--device",
                device,
                "--variants",
                *variants,
                "--results-dir",
                str(ablation_dir),
                "--backbone",
                self.backbone,
            ]
            if skip_aggregate:
                cmd.append("--skip-aggregate")
            return cmd

        if self.parallel:
            with ProcessPoolExecutor(max_workers=2) as pool:
                f1 = pool.submit(
                    subprocess.run,
                    ablation_cmd(["baseline", "no_sup_mmd", "minimal"], self.device, True),
                    cwd=ROOT,
                    check=True,
                )
                f2 = pool.submit(
                    subprocess.run,
                    ablation_cmd(["euclidean", "vmf_prior"], self.device2, True),
                    cwd=ROOT,
                    check=True,
                )
                f1.result()
                f2.result()
        else:
            self.run_cmd(
                ablation_cmd(ABLATION_VARIANTS, self.device, skip_aggregate=False),
                "ablation_all",
            )
            return ablation_dir

        self.run_cmd(
            [
                sys.executable,
                "run_ablation_study.py",
                "--summarize-only",
                "--results-dir",
                str(ablation_dir),
            ],
            "ablation_summarize",
        )
        return ablation_dir

    def phase_baselines(self, output_dir: Path | None = None) -> Path:
        baseline_dir = Path(output_dir or self.runs_dir / "baselines" / "seed_0")
        baseline_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"Baselines → {baseline_dir}")

        def baseline_cmd(models: list[str], device: str) -> list[str]:
            cmd = [
                sys.executable,
                "compare_baselines.py",
                "--dataset",
                self.dataset,
                "--seed",
                "0",
                "--device",
                device,
                "--epochs",
                str(self.args.epochs if self.args.epochs is not None else 50),
                "--output-dir",
                str(baseline_dir),
                "--models",
                *models,
                "--backbone",
                self.backbone,
                "--skip-summary",
            ]
            return cmd

        if self.parallel:
            with ProcessPoolExecutor(max_workers=2) as pool:
                f1 = pool.submit(
                    subprocess.run,
                    baseline_cmd(["VAE", "WAE-MMD"], self.device),
                    cwd=ROOT,
                    check=True,
                )
                f2 = pool.submit(
                    subprocess.run,
                    baseline_cmd(["VaDE", "CS-WAE"], self.device2),
                    cwd=ROOT,
                    check=True,
                )
                f1.result()
                f2.result()
            self.run_cmd(
                [
                    sys.executable,
                    "compare_baselines.py",
                    "--dataset",
                    self.dataset,
                    "--summarize-only",
                    "--output-dir",
                    str(baseline_dir),
                ],
                "baselines_summarize",
            )
        else:
            cmd = [
                sys.executable,
                "compare_baselines.py",
                "--dataset",
                self.dataset,
                "--seed",
                "0",
                "--device",
                self.device,
                "--epochs",
                str(self.args.epochs if self.args.epochs is not None else 50),
                "--output-dir",
                str(baseline_dir),
                "--backbone",
                self.backbone,
            ]
            self.run_cmd(cmd, "baselines_all")

        return baseline_dir

    def phase_aggregate(self) -> None:
        self.log(f"Aggregate → {self.runs_dir}")
        self.run_cmd(
            [
                sys.executable,
                "aggregate_results.py",
                "--runs-dir",
                str(self.runs_dir),
            ],
            "aggregate",
        )

    def run_all(self, seeds: list[int]) -> None:
        self.log(
            f"=== CS-WAE Pipeline — dataset={self.dataset} backbone={self.backbone} "
            f"runs={self.runs_dir} ==="
        )
        self.phase_train(seeds)
        ablation_dir = self.phase_ablation()
        baseline_dir = self.phase_baselines()
        self.phase_aggregate()
        self.log("=== Pipeline complete ===")
        self.log(f"Runs:       {self.runs_dir}/seed_{{0,1,2}}/")
        self.log(f"Ablation:   {ablation_dir}/")
        self.log(f"Baselines:  {baseline_dir}/")
        self.log(f"Aggregate:  {self.runs_dir}/aggregated_metrics.json")
        self.log(f"Log:        {self.log_file}")


def resolve_backbone(dataset: str, backbone: str | None) -> str:
    return backbone if backbone is not None else "cnn"


def main() -> None:
    args = parse_args()
    pipeline = Pipeline(args)

    if args.command == "train":
        pipeline.phase_train(args.seeds)
    elif args.command == "ablation":
        pipeline.phase_ablation(
            Path(args.results_dir) if args.results_dir else None
        )
    elif args.command == "baselines":
        pipeline.phase_baselines(
            Path(args.output_dir) if args.output_dir else None
        )
    elif args.command == "aggregate":
        pipeline.log_file = Path(args.log_file) if args.log_file else pipeline.log_file
        pipeline.phase_aggregate()
    elif args.command == "all":
        if not args.skip_viz and pipeline.dataset in RGB_DATASETS:
            args.skip_viz = True
        if not args.skip_advanced_viz and pipeline.dataset in HEAVY_VIZ_SKIP:
            args.skip_advanced_viz = True
        pipeline.run_all(args.seeds)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
