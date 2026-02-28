"""Tests for FLOPs computation and reporting.

Validates that forward (and backward) FLOPs are computed for logged models
and that the summary utilities from torchlens.flops work correctly.
"""

import pytest
import torch
import torchvision

import torchlens as tl
from torchlens.flops import format_flops, get_total_flops, get_flops_by_type


# ---------------------------------------------------------------------------
# format_flops unit tests
# ---------------------------------------------------------------------------


class TestFormatFlops:
    """Unit tests for the format_flops helper."""

    def test_none_returns_na(self):
        assert format_flops(None) == "N/A"

    def test_small_value(self):
        assert format_flops(42) == "42"

    def test_kilo(self):
        result = format_flops(1500)
        assert "K" in result

    def test_mega(self):
        result = format_flops(2_500_000)
        assert "M" in result

    def test_giga(self):
        result = format_flops(3_000_000_000)
        assert "G" in result

    def test_tera(self):
        result = format_flops(1_200_000_000_000)
        assert "T" in result

    def test_precision(self):
        result = format_flops(1_234_567, precision=1)
        assert "M" in result
        # With precision=1 we expect one decimal place
        assert "." in result


# ---------------------------------------------------------------------------
# FLOPs on real models
# ---------------------------------------------------------------------------


class TestModelFlops:
    """Verify FLOPs are computed for traced models."""

    @pytest.fixture
    def img_input(self):
        return torch.rand(1, 3, 224, 224)

    def _log_model(self, model, x):
        model.eval()
        return tl.log_forward_pass(
            model, x, vis_opt="none", save_function_args=True
        )

    def test_resnet18_has_flops(self, img_input):
        model = torchvision.models.resnet18(weights=None)
        mh = self._log_model(model, img_input)
        totals = get_total_flops(mh.layer_list)
        assert totals["forward"] > 0, "Forward FLOPs should be > 0 for ResNet-18"

    def test_vgg11_has_flops(self, img_input):
        model = torchvision.models.vgg11(weights=None)
        mh = self._log_model(model, img_input)
        totals = get_total_flops(mh.layer_list)
        assert totals["forward"] > 0, "Forward FLOPs should be > 0 for VGG-11"

    def test_vit_b_32_has_flops(self, img_input):
        model = torchvision.models.vit_b_32(weights=None)
        mh = self._log_model(model, img_input)
        totals = get_total_flops(mh.layer_list)
        assert totals["forward"] > 0, "Forward FLOPs should be > 0 for ViT-B/32"

    def test_get_total_flops_keys(self, img_input):
        model = torchvision.models.resnet18(weights=None)
        mh = self._log_model(model, img_input)
        totals = get_total_flops(mh.layer_list)
        assert "forward" in totals
        assert "backward" in totals
        assert "total" in totals
        assert totals["total"] >= totals["forward"]

    def test_get_flops_by_type(self, img_input):
        model = torchvision.models.resnet18(weights=None)
        mh = self._log_model(model, img_input)
        by_type = get_flops_by_type(mh.layer_list)
        assert isinstance(by_type, dict)
        # Should have at least one layer type entry
        assert len(by_type) > 0
        for info in by_type.values():
            assert "forward" in info
            assert "backward" in info
            assert "count" in info


# ---------------------------------------------------------------------------
# FLOPs summary printing (smoke test)
# ---------------------------------------------------------------------------


class TestFlopsSummaryPrint:
    """Smoke-test the FLOPs summary output (similar to debug_torchlens.py)."""

    @pytest.fixture
    def img_input(self):
        return torch.rand(1, 3, 224, 224)

    def test_print_flops_summary(self, img_input, capsys):
        """Verify we can print a FLOPs summary without errors."""
        model = torchvision.models.resnet18(weights=None).eval()
        mh = tl.log_forward_pass(
            model, img_input, vis_opt="none", save_function_args=True
        )
        layer_list = mh.layer_list
        totals = get_total_flops(layer_list)

        # Print a summary (mimics debug_torchlens.py print_flops_summary)
        total_forward = 0
        total_backward = 0
        for layer in layer_list:
            fwd = layer.flops
            bwd = getattr(layer, "backward_flops", None)
            if fwd is not None:
                total_forward += fwd
            if bwd is not None:
                total_backward += bwd
            if layer.layer_type not in ("input", "output", "buffer"):
                print(
                    f"{layer.layer_label:<35} "
                    f"{layer.layer_type:<20} "
                    f"{format_flops(fwd):>12} "
                    f"{format_flops(bwd):>12}"
                )

        captured = capsys.readouterr()
        assert len(captured.out) > 0, "Expected printed output"
        assert total_forward == totals["forward"]
