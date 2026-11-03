"""Tests for Layer: aggregate per-layer metadata."""

import pytest
import torch
from torch import nn

import torchlens as tl

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class SimpleModel(nn.Module):
    def forward(self, x):
        return torch.relu(x + 1)


class RecurrentModel(nn.Module):
    """Model that applies the same linear layer in a loop (triggers loop detection)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        for _ in range(3):
            x = self.fc(x)
        return x


class ColonNamedModule(nn.Module):
    """Model with a legal PyTorch submodule name containing a colon."""

    def __init__(self) -> None:
        """Register a colon-bearing submodule name."""

        super().__init__()
        self.add_module("a:b", nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the colon-bearing submodule."""

        return self._modules["a:b"](x)


@pytest.fixture
def simple_log():
    model = SimpleModel()
    return tl.trace(model, torch.randn(1, 5), save="all")


@pytest.fixture
def recurrent_log():
    model = RecurrentModel()
    return tl.trace(model, torch.randn(1, 5), save="all")


# ---------------------------------------------------------------------------
# Basic Layer construction
# ---------------------------------------------------------------------------


class TestLayerLogConstruction:
    def test_single_pass_layer_has_one_pass(self, simple_log):
        """Non-recurrent model: every Layer has exactly one pass."""
        for layer_log in simple_log._layer_logs.values():
            assert layer_log.num_passes == 1
            assert len(layer_log.ops) == 1
            assert 0 in layer_log.ops


# ---------------------------------------------------------------------------
# Multi-pass (recurrent) Layer
# ---------------------------------------------------------------------------


class TestMultiPassLayerLog:
    def test_multi_pass_aggregate_graph_union(self, recurrent_log):
        """Aggregate graph properties are unions across ops."""
        for layer_log in recurrent_log._layer_logs.values():
            if layer_log.num_passes > 1 and layer_log.has_children:
                # Verify children is a union
                all_children = set()
                for pass_log in layer_log.ops.values():
                    for child in pass_log.children:
                        all_children.add(recurrent_log[child].layer_label)
                assert set(layer_log.children) == all_children
                break


# ---------------------------------------------------------------------------
# __str__ and __repr__
# ---------------------------------------------------------------------------


class TestLayerLogDisplay:
    def test_str_single_pass(self, simple_log):
        """The card's first line IS the repr (F10; lovely D15)."""
        for layer_log in simple_log._layer_logs.values():
            s = str(layer_log)
            assert layer_log.layer_label in s
            assert s.splitlines()[0] == repr(layer_log)
            assert len(s.splitlines()) <= 8

    def test_str_multi_pass(self, recurrent_log):
        """A multi-pass layer says 'passes', never a pooled statistic (D32)."""
        for layer_log in recurrent_log._layer_logs.values():
            if layer_log.num_passes > 1:
                s = str(layer_log)
                assert f"x{layer_log.num_passes} passes" in s
                assert "pass 1/" in s  # per-pass cores, never pooled
                break


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------


class TestConvenienceAliases:
    def test_params_accessor(self, recurrent_log):
        for layer_log in recurrent_log._layer_logs.values():
            if layer_log.has_params:
                params = layer_log.params
                assert params is not None
                break


# ---------------------------------------------------------------------------
# Bugfix regression tests
# ---------------------------------------------------------------------------


class _SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class TestSliceIndexing:
    """Slice indexing should return list."""

    def test_slice_returns_list(self):
        model = _SimpleLinear()
        log = tl.trace(model, torch.randn(2, 10))
        result = log[0:2]
        assert isinstance(result, list)
        assert len(result) == 2


class TestAmbiguousSubstring:
    """Ambiguous substring should list matching layers."""

    def test_lists_matches(self):
        class TwoLinears(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = nn.Linear(10, 10)
                self.lin2 = nn.Linear(10, 10)

            def forward(self, x):
                return self.lin1(x) + self.lin2(x)

        model = TwoLinears()
        log = tl.trace(model, torch.randn(2, 10))
        with pytest.raises(ValueError, match="Ambiguous"):
            log["linear"]


class TestCaseInsensitiveLookup:
    """Case-insensitive lookup for layer/module names."""

    def test_case_insensitive_module_lookup(self):
        model = _SimpleLinear()
        log = tl.trace(model, torch.randn(2, 10))
        result = log["FC"]
        assert result is not None
