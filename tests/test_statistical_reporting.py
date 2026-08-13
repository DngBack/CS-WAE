import pytest

from src.metrics.statistical_reporting import holm_adjust, holm_family


def test_holm_adjust_preserves_input_order_and_monotonicity():
    raw = [0.04, 0.01, 0.03]
    adjusted = holm_adjust(raw)
    assert adjusted == pytest.approx([0.06, 0.03, 0.06])
    result = holm_family(raw, alpha=0.05)
    assert result["reject"] == [False, True, False]


def test_holm_rejects_invalid_values():
    with pytest.raises(ValueError):
        holm_adjust([])
    with pytest.raises(ValueError):
        holm_adjust([0.1, 1.1])
