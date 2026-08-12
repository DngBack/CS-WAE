"""CPU-only tests for crash-safe F-CS-WAE training checkpoints."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

from src.trainers.trainer_f_cs_wae import FCSWAETrainer


RUN_CONFIG = {
    "dataset": "mnist",
    "seed": 7,
    "epochs": 10,
    "n_centers": 1,
    "semantic_dim": 3,
    "style_dim": 4,
    "delta_final": 0.0,
    "style_sigma_floor": 0.15,
    "phase_a_end": 2,
    "phase_b_end": 4,
}


def make_minimal_trainer(
    seed: int,
    device: torch.device | None = None,
) -> FCSWAETrainer:
    trainer = object.__new__(FCSWAETrainer)
    trainer.device = device or torch.device("cpu")
    trainer.model = nn.Linear(3, 2).to(trainer.device)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(
        trainer.optimizer, step_size=2, gamma=0.5
    )
    loader_generator = torch.Generator().manual_seed(seed)
    trainer.train_loader = SimpleNamespace(generator=loader_generator)
    return trainer


class TrainingResumeTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for map-location regression")
    def test_cuda_resume_keeps_cpu_rng_snapshot_on_cpu(self):
        trainer = make_minimal_trainer(seed=5, device=torch.device("cuda:0"))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "training_checkpoint.pt"
            trainer.save_training_checkpoint(
                checkpoint_path,
                completed_epochs=0,
                history=[],
                run_config=RUN_CONFIG,
            )
            resumed = make_minimal_trainer(seed=99, device=torch.device("cuda:0"))
            completed, history = resumed.load_training_checkpoint(
                checkpoint_path,
                expected_run_config=RUN_CONFIG,
            )
            self.assertEqual((completed, history), (0, []))

    def test_checkpoint_restores_training_and_rng_state(self):
        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)
        trainer = make_minimal_trainer(seed=11)

        loss = trainer.model(torch.ones((2, 3))).square().mean()
        loss.backward()
        trainer.optimizer.step()
        trainer.scheduler.step()
        saved_weights = {
            key: value.detach().clone()
            for key, value in trainer.model.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "training_checkpoint.pt"
            trainer.save_training_checkpoint(
                checkpoint_path,
                completed_epochs=3,
                history=[{"rec": 1.0}, {"rec": 0.5}, {"rec": 0.25}],
                run_config=RUN_CONFIG,
            )
            self.assertTrue(checkpoint_path.exists())
            self.assertFalse((checkpoint_path.parent / ".training_checkpoint.pt.tmp").exists())

            expected_python = random.random()
            expected_numpy = float(np.random.rand())
            expected_torch = torch.rand(4)
            expected_loader = torch.randperm(12, generator=trainer.train_loader.generator)

            random.random()
            np.random.rand()
            torch.rand(10)
            for parameter in trainer.model.parameters():
                parameter.data.zero_()

            resumed = make_minimal_trainer(seed=999)
            completed, history = resumed.load_training_checkpoint(
                checkpoint_path,
                expected_run_config=RUN_CONFIG,
            )
            self.assertEqual(completed, 3)
            self.assertEqual(len(history), 3)
            self.assertEqual(resumed.scheduler.last_epoch, trainer.scheduler.last_epoch)
            for key, expected in saved_weights.items():
                self.assertTrue(torch.equal(resumed.model.state_dict()[key], expected))
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(float(np.random.rand()), expected_numpy)
            self.assertTrue(torch.equal(torch.rand(4), expected_torch))
            self.assertTrue(
                torch.equal(
                    torch.randperm(12, generator=resumed.train_loader.generator),
                    expected_loader,
                )
            )

    def test_checkpoint_rejects_changed_experiment_configuration(self):
        trainer = make_minimal_trainer(seed=3)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "training_checkpoint.pt"
            trainer.save_training_checkpoint(
                checkpoint_path,
                completed_epochs=0,
                history=[],
                run_config=RUN_CONFIG,
            )
            changed = dict(RUN_CONFIG, style_sigma_floor=0.35)
            with self.assertRaisesRegex(ValueError, "Resume configuration mismatch"):
                trainer.load_training_checkpoint(
                    checkpoint_path,
                    expected_run_config=changed,
                )


if __name__ == "__main__":
    unittest.main()
