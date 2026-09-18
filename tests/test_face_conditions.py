"""Face dataset and continuous-condition audit tests (no dataset download)."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.datasets.face_conditions import (
    DEFAULT_AGE_BIN_EDGES,
    load_celebahq_populations,
    load_utkface_populations,
)
from src.metrics.continuous_condition import (
    continuous_hsic_permutation_test,
    paired_conditional_mmd_permutation_test,
    paired_factorized_joint_mmd_permutation_test,
)
from src.models.f_cs_wae import FCSWAE, ResBlockDecoder


def _image(path: Path, value: int) -> None:
    pixels = np.full((20, 20, 3), value % 255, dtype=np.uint8)
    Image.fromarray(pixels).save(path)


class FaceConditionDatasetTests(unittest.TestCase):
    def test_utkface_parses_real_age_and_keeps_trainer_label_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "UTKFace"
            root.mkdir()
            expected = {}
            for index in range(60):
                age = (index * 2) % 117
                name = f"{age}_{index % 2}_{index % 5}_201701010000{index:03d}.jpg"
                _image(root / name, index)
                expected[name] = age
            _image(root / "bad_name.jpg", 0)
            populations, metadata = load_utkface_populations(
                data_dir=directory, image_size=64, split_seed=17
            )
            self.assertEqual(sum(map(len, populations.values())), 60)
            self.assertEqual(metadata["rejected_filenames"], 1)
            self.assertEqual(metadata["n_classes"], len(DEFAULT_AGE_BIN_EDGES) - 1)
            group_sets = [population.groups for population in populations.values()]
            self.assertFalse(group_sets[0] & group_sets[1])
            self.assertFalse(group_sets[0] & group_sets[2])
            self.assertFalse(group_sets[1] & group_sets[2])
            nonempty = next(population for population in populations.values() if len(population))
            image, label, age, attributes = nonempty[0]
            record = nonempty.records[0]
            self.assertEqual(tuple(image.shape), (3, 64, 64))
            self.assertEqual(float(age), expected[record.path.name])
            self.assertEqual(label.dtype, torch.long)
            self.assertEqual(tuple(attributes.shape), (2,))

    def test_celebahq_uses_explicit_mapping_and_binary_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "CelebA-HQ"
            images = root / "images"
            images.mkdir(parents=True)
            metadata_path = root / "metadata.csv"
            rows = []
            for index, split in enumerate(("train", "train", "validation", "validation", "test", "test")):
                name = f"{index:05d}.jpg"
                _image(images / name, index * 20)
                rows.append(
                    {
                        "image": name,
                        "identity": f"person-{index}",
                        "split": split,
                        "Smiling": -1 if index % 2 == 0 else 1,
                        "Eyeglasses": 1 if index % 3 == 0 else -1,
                    }
                )
            with metadata_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            populations, metadata = load_celebahq_populations(
                data_dir=directory,
                image_size=64,
                attribute_columns=("Eyeglasses",),
            )
            self.assertEqual([len(populations[name]) for name in ("train", "validation", "test")], [2, 2, 2])
            self.assertEqual(metadata["condition_label_map"], {"-1.0": 0, "1.0": 1})
            image, label, condition, attributes = populations["test"][0]
            self.assertEqual(tuple(image.shape), (3, 64, 64))
            self.assertIn(int(label), (0, 1))
            self.assertIn(float(condition), (-1.0, 1.0))
            self.assertEqual(metadata["attribute_names"], ["identity_id", "Eyeglasses"])
            self.assertIn(float(attributes[-1]), (0.0, 1.0))

    def test_celebahq_hash_split_keeps_identities_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "CelebA-HQ"
            images = root / "images"
            images.mkdir(parents=True)
            rows = []
            for identity in range(30):
                for view in range(2):
                    name = f"{identity:03d}_{view}.jpg"
                    _image(images / name, identity)
                    rows.append(
                        {
                            "image": name,
                            "identity": f"person-{identity}",
                            "Smiling": -1 if identity % 2 else 1,
                        }
                    )
            metadata_path = root / "metadata.csv"
            with metadata_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            populations, _ = load_celebahq_populations(
                data_dir=directory, image_size=64, split_seed=23
            )
            groups = [populations[name].groups for name in ("train", "validation", "test")]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            for population in populations.values():
                counts = {}
                for record in population.records:
                    counts[record.group] = counts.get(record.group, 0) + 1
                self.assertTrue(all(count == 2 for count in counts.values()))


class ContinuousConditionAuditTests(unittest.TestCase):
    def test_decoder_supports_face_resolution(self):
        decoder = ResBlockDecoder(semantic_dim=4, style_dim=5, image_size=128)
        output = decoder(torch.randn(1, 9))
        self.assertEqual(tuple(output.shape), (1, 3, 128, 128))

    def test_continuous_prior_is_seeded_and_interpolated(self):
        model = FCSWAE(4, 3, n_classes=3, image_size=64)
        with torch.no_grad():
            model.ema_centers.copy_(
                F.normalize(
                    torch.tensor(
                        [[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0, 0.0]]]
                    ),
                    dim=-1,
                )
            )
        conditions = torch.tensor([10.0, 20.0, 30.0, 40.0])
        knots = torch.tensor([10.0, 25.0, 40.0])
        first = model.sample_from_continuous_prior(
            conditions, knots, generator=torch.Generator().manual_seed(11)
        )
        second = model.sample_from_continuous_prior(
            conditions, knots, generator=torch.Generator().manual_seed(11)
        )
        self.assertTrue(torch.allclose(first[0], second[0]))
        self.assertTrue(torch.allclose(first[1], second[1]))
        self.assertEqual(tuple(first[0].shape), (4, 4))
        self.assertTrue(torch.allclose(first[0].norm(dim=1), torch.ones(4), atol=1e-5))

    def test_continuous_hsic_detects_age_leakage(self):
        generator = torch.Generator().manual_seed(31)
        age = torch.linspace(0.0, 1.0, 96)
        style = age[:, None].repeat(1, 4) + 0.03 * torch.randn((96, 4), generator=generator)
        result = continuous_hsic_permutation_test(
            style, age, seed=31, n_permutations=49
        )
        self.assertLessEqual(result["p_value"], 0.05)
        self.assertTrue(result["reject_0.05"])

    def test_paired_conditional_joint_mmd_detects_sampler_shift(self):
        generator = torch.Generator().manual_seed(41)
        n = 64
        age = torch.linspace(0.0, 1.0, n)
        q_content = F.normalize(torch.randn((n, 5), generator=generator), dim=1)
        p_content = F.normalize(torch.randn((n, 5), generator=generator), dim=1)
        q_style = torch.randn((n, 4), generator=generator) + age[:, None]
        p_style = torch.randn((n, 4), generator=generator) + age[:, None] + 3.0
        style_result = paired_conditional_mmd_permutation_test(
            q_style,
            p_style,
            age,
            seed=41,
            n_permutations=49,
            condition_kind="continuous",
        )
        joint_result = paired_factorized_joint_mmd_permutation_test(
            q_content,
            q_style,
            p_content,
            p_style,
            age,
            seed=42,
            n_permutations=49,
        )
        self.assertLessEqual(style_result["p_value"], 0.05)
        self.assertLessEqual(joint_result["p_value"], 0.05)


if __name__ == "__main__":
    unittest.main()
