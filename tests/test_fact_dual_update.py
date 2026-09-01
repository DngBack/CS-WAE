"""Tests for scale-normalized FACT dual ascent."""

from __future__ import annotations

import unittest

from src.config_f_cs_wae import f_cs_wae_config as cfg
from src.trainers.trainer_f_cs_wae import FCSWAETrainer


class FactDualUpdateTests(unittest.TestCase):
    def test_reference_scales_make_equal_relative_violations_equal_updates(self):
        names = (
            "fact_dual_lr",
            "fact_dual_max",
            "fact_style_tolerance",
            "fact_content_tolerance",
            "fact_dependence_tolerance",
            "fact_dual_reference_style",
            "fact_dual_reference_content",
            "fact_dual_reference_dependence",
            "fact_dual_step_max",
        )
        original = {name: getattr(cfg, name) for name in names}
        try:
            cfg.fact_dual_lr = 0.1
            cfg.fact_dual_max = 10.0
            cfg.fact_style_tolerance = 0.0
            cfg.fact_content_tolerance = 0.0
            cfg.fact_dependence_tolerance = 0.0
            cfg.fact_dual_reference_style = 0.002
            cfg.fact_dual_reference_content = 0.0015
            cfg.fact_dual_reference_dependence = 0.000025
            cfg.fact_dual_step_max = None
            trainer = object.__new__(FCSWAETrainer)
            trainer.fact_dual_weights = {
                "style": 1.0,
                "content": 1.0,
                "dependence": 1.0,
            }
            trainer.update_fact_dual_weights(
                {
                    "fact_style": 0.002,
                    "fact_content": 0.0015,
                    "fact_dependence": 0.000025,
                }
            )
            for value in trainer.fact_dual_weights.values():
                self.assertAlmostEqual(value, 1.1)
        finally:
            for name, value in original.items():
                setattr(cfg, name, value)

    def test_epoch_step_cap_prevents_transient_saturation(self):
        names = (
            "fact_dual_lr",
            "fact_dual_max",
            "fact_style_tolerance",
            "fact_content_tolerance",
            "fact_dependence_tolerance",
            "fact_dual_reference_style",
            "fact_dual_reference_content",
            "fact_dual_reference_dependence",
            "fact_dual_step_max",
        )
        original = {name: getattr(cfg, name) for name in names}
        try:
            cfg.fact_dual_lr = 0.3
            cfg.fact_dual_max = 10.0
            cfg.fact_style_tolerance = 0.0
            cfg.fact_content_tolerance = 0.0
            cfg.fact_dependence_tolerance = 0.0
            cfg.fact_dual_reference_style = 0.002
            cfg.fact_dual_reference_content = 0.0015
            cfg.fact_dual_reference_dependence = 0.000025
            cfg.fact_dual_step_max = 0.25
            trainer = object.__new__(FCSWAETrainer)
            trainer.fact_dual_weights = dict.fromkeys(
                ("style", "content", "dependence"), 1.0
            )
            trainer.update_fact_dual_weights(
                {
                    "fact_style": 1.0,
                    "fact_content": 1.0,
                    "fact_dependence": 1.0,
                }
            )
            for value in trainer.fact_dual_weights.values():
                self.assertAlmostEqual(value, 1.25)
        finally:
            for name, value in original.items():
                setattr(cfg, name, value)


if __name__ == "__main__":
    unittest.main()
