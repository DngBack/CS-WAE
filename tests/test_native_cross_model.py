import torch
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.rotated_mnist import balanced_source_indices
from src.metrics.factorized_adapter import build_factorized_audit_adapter
from src.models.native_factorized_baselines import NativeDIVA, NativeDRIT
from src.trainers.native_factorized_trainers import DIVATrainer, DRITTrainer


def test_balanced_source_indices_is_reproducible_and_interleaved():
    targets = torch.arange(10).repeat_interleave(8)
    first = balanced_source_indices(targets, per_class=3, seed=17)
    second = balanced_source_indices(targets, per_class=3, seed=17)
    assert torch.equal(first, second)
    assert torch.equal(targets[first].view(3, 10), torch.arange(10).repeat(3, 1))


def test_native_adapters_return_reproducible_named_views():
    images = torch.rand(4, 1, 28, 28)
    domains = torch.zeros(4, dtype=torch.long)
    for model in (
        NativeDRIT(content_dim=12, style_dim=5),
        NativeDIVA(domain_dim=6, style_dim=5, semantic_dim=12),
    ):
        model.eval()
        adapter = build_factorized_audit_adapter(model)
        first = adapter.encode_views(
            images,
            domain=domains,
            generator=torch.Generator().manual_seed(123),
        )
        second = adapter.encode_views(
            images,
            domain=domains,
            generator=torch.Generator().manual_seed(123),
        )
        assert first.content_mean.shape == (4, 12)
        assert first.style_sample.shape == (4, 5)
        assert torch.equal(first.style_sample, second.style_sample)
        assert first.style_logvar is not None
        decoded = adapter.decode(first.content_sample, first.style_sample, domain=0)
        assert decoded.shape == images.shape


def test_drit_adapter_rejects_pooled_domain_specific_style_spaces():
    adapter = build_factorized_audit_adapter(NativeDRIT(content_dim=8, style_dim=4))
    images = torch.rand(2, 1, 28, 28)
    try:
        adapter.encode_views(
            images,
            domain=torch.tensor([0, 1]),
            generator=torch.Generator().manual_seed(0),
        )
    except ValueError as error:
        assert "one domain at a time" in str(error)
    else:
        raise AssertionError("DRIT adapter silently pooled domain-specific style spaces")


def test_native_trainers_complete_one_small_cpu_epoch(tmp_path):
    images = torch.rand(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3])
    domains = torch.tensor([0, 1, 0, 1])
    diva_loader = DataLoader(TensorDataset(images, labels, domains), batch_size=4)
    diva = NativeDIVA(domain_dim=4, style_dim=4, semantic_dim=4)
    diva_trainer = DIVATrainer(diva, diva_loader, torch.device("cpu"))
    diva_metrics = diva_trainer.train_epoch(0)
    assert torch.isfinite(torch.tensor(diva_metrics["total"]))

    domain_zero = DataLoader(
        TensorDataset(images[:2], labels[:2], torch.zeros(2, dtype=torch.long)),
        batch_size=2,
    )
    domain_one = DataLoader(
        TensorDataset(images[2:], labels[2:], torch.ones(2, dtype=torch.long)),
        batch_size=2,
    )
    drit = NativeDRIT(content_dim=4, style_dim=3)
    drit_trainer = DRITTrainer(drit, (domain_zero, domain_one), torch.device("cpu"))
    drit_metrics = drit_trainer.train_epoch(0)
    assert torch.isfinite(torch.tensor(drit_metrics["generator_total"]))
