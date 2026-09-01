"""Tests for the exploratory late-phase objective-weight cap."""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from src.config_f_cs_wae import f_cs_wae_config as cfg
from src.trainers.trainer_f_cs_wae import FCSWAETrainer


class PhaseWeightFreezeTests(TestCase):
    def test_freeze_reuses_boundary_weights_without_disabling_fact(self):
        trainer = object.__new__(FCSWAETrainer)
        settings = {
            "phase_a_end": 5,
            "phase_b_end": 12,
            "phase_c_end": 26,
            "phase_d_end": 40,
            "phase_weight_freeze_epoch": 34,
            "alpha_final": 2.0,
            "beta_final": 5.0,
            "gamma_final": 1.0,
            "delta_final": 1.0,
            "eta_init": 0.1,
            "eta_final": 0.3,
            "fact_enabled": True,
            "fact_start_epoch": 12,
        }
        with patch.multiple(cfg, **settings):
            boundary = trainer.get_phase_weights(34)
            self.assertEqual(trainer.get_phase_weights(39), boundary)
            self.assertAlmostEqual(boundary[0], 1.0 + 8.0 / 14.0)
            self.assertAlmostEqual(boundary[1], 2.5 + 2.5 * 8.0 / 14.0)
            self.assertTrue(trainer.fact_is_active(39))

    def test_unfrozen_schedule_continues_to_ramp(self):
        trainer = object.__new__(FCSWAETrainer)
        with patch.multiple(
            cfg,
            phase_a_end=5,
            phase_b_end=12,
            phase_c_end=26,
            phase_d_end=40,
            phase_weight_freeze_epoch=None,
            alpha_final=2.0,
            beta_final=5.0,
            gamma_final=1.0,
            delta_final=1.0,
            eta_init=0.1,
            eta_final=0.3,
        ):
            at_34 = trainer.get_phase_weights(34)
            at_39 = trainer.get_phase_weights(39)
            self.assertGreater(at_39[0], at_34[0])
            self.assertGreater(at_39[1], at_34[1])
