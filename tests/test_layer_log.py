"""Tests for LayerLog: aggregate per-layer metadata."""

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes.layer_log import LayerLog
from torchlens.data_classes.layer_pass_log import LayerPassLog


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


@pytest.fixture
def simple_log():
    model = SimpleModel()
    return tl.log_forward_pass(model, torch.randn(1, 5), layers_to_save="all")


@pytest.fixture
def recurrent_log():
    model = RecurrentModel()
    return tl.log_forward_pass(model, torch.randn(1, 5), layers_to_save="all")


# ---------------------------------------------------------------------------
# Basic LayerLog construction
# ---------------------------------------------------------------------------


class TestLayerLogConstruction:
    def test_layer_logs_populated(self, simple_log):
        """layer_logs dict is populated after postprocessing."""
        assert len(simple_log.layer_logs) > 0

    def test_layer_logs_keyed_by_no_pass_label(self, simple_log):
        """Keys are no-pass layer labels."""
        for key, layer_log in simple_log.layer_logs.items():
            assert ":" not in key
            assert isinstance(layer_log, LayerLog)

    def test_single_pass_layer_has_one_pass(self, simple_log):
        """Non-recurrent model: every LayerLog has exactly one pass."""
        for layer_log in simple_log.layer_logs.values():
            assert layer_log.num_passes == 1
            assert len(layer_log.passes) == 1
            assert 1 in layer_log.passes

    def test_back_reference_set(self, simple_log):
        """parent_layer_log is set on each LayerPassLog."""
        for layer_log in simple_log.layer_logs.values():
            for pass_log in layer_log.passes.values():
                assert pass_log.parent_layer_log is layer_log


# ---------------------------------------------------------------------------
# Single-pass delegation
# ---------------------------------------------------------------------------


class TestSinglePassDelegation:
    def test_tensor_contents_delegation(self, simple_log):
        """tensor_contents delegates to passes[1] for single-pass."""
        for layer_log in simple_log.layer_logs.values():
            pass_log = layer_log.passes[1]
            assert layer_log.tensor_contents is pass_log.tensor_contents

    def test_has_saved_activations_delegation(self, simple_log):
        for layer_log in simple_log.layer_logs.values():
            pass_log = layer_log.passes[1]
            assert layer_log.has_saved_activations == pass_log.has_saved_activations

    def test_operation_num_delegation(self, simple_log):
        for layer_log in simple_log.layer_logs.values():
            pass_log = layer_log.passes[1]
            assert layer_log.operation_num == pass_log.operation_num

    def test_fallback_getattr_delegation(self, simple_log):
        """__getattr__ fallback delegates arbitrary per-pass fields."""
        for layer_log in simple_log.layer_logs.values():
            pass_log = layer_log.passes[1]
            # func_autocast_state is per-pass, not an explicit @property
            assert layer_log.func_autocast_state is pass_log.func_autocast_state

    def test_pass_finished_reads_from_model_log(self, simple_log):
        for layer_log in simple_log.layer_logs.values():
            assert layer_log._pass_finished is True


# ---------------------------------------------------------------------------
# Aggregate fields
# ---------------------------------------------------------------------------


class TestAggregateFields:
    def test_aggregate_fields_match_first_pass(self, simple_log):
        """Aggregate fields on LayerLog match the corresponding first-pass values."""
        for layer_log in simple_log.layer_logs.values():
            fp = layer_log.passes[1]
            assert layer_log.layer_type == fp.layer_type
            assert layer_log.func_applied_name == fp.func_applied_name
            assert layer_log.tensor_shape == fp.tensor_shape
            assert layer_log.tensor_dtype == fp.tensor_dtype
            assert layer_log.is_input_layer == fp.is_input_layer
            assert layer_log.is_output_layer == fp.is_output_layer
            assert layer_log.computed_with_params == fp.computed_with_params

    def test_layer_label_is_no_pass(self, simple_log):
        for layer_log in simple_log.layer_logs.values():
            fp = layer_log.passes[1]
            assert layer_log.layer_label == fp.layer_label_no_pass


# ---------------------------------------------------------------------------
# Multi-pass (recurrent) LayerLog
# ---------------------------------------------------------------------------


class TestMultiPassLayerLog:
    def test_recurrent_layer_has_multiple_passes(self, recurrent_log):
        """The repeated fc layer should have 2 passes."""
        multi_pass_found = False
        for layer_log in recurrent_log.layer_logs.values():
            if layer_log.num_passes > 1:
                multi_pass_found = True
                assert len(layer_log.passes) == layer_log.num_passes
        assert multi_pass_found, "Expected at least one multi-pass layer in recurrent model"

    def test_multi_pass_tensor_contents_raises(self, recurrent_log):
        """Accessing per-pass field on multi-pass LayerLog raises ValueError."""
        for layer_log in recurrent_log.layer_logs.values():
            if layer_log.num_passes > 1:
                with pytest.raises(ValueError, match="has .* passes"):
                    _ = layer_log.tensor_contents
                break

    def test_multi_pass_getattr_raises(self, recurrent_log):
        """__getattr__ fallback raises for multi-pass layers."""
        for layer_log in recurrent_log.layer_logs.values():
            if layer_log.num_passes > 1:
                with pytest.raises(AttributeError):
                    _ = layer_log.func_autocast_state
                break

    def test_multi_pass_aggregate_graph_union(self, recurrent_log):
        """Aggregate graph properties are unions across passes."""
        for layer_log in recurrent_log.layer_logs.values():
            if layer_log.num_passes > 1 and layer_log.has_children:
                # Verify child_layers is a union
                all_children = set()
                for pass_log in layer_log.passes.values():
                    for child in pass_log.child_layers:
                        all_children.add(recurrent_log[child].layer_label_no_pass)
                assert set(layer_log.child_layers) == all_children
                break

    def test_getitem_returns_layer_log_for_multi_pass(self, recurrent_log):
        """log['label'] returns LayerLog for multi-pass layers."""
        for layer_log in recurrent_log.layer_logs.values():
            if layer_log.num_passes > 1:
                result = recurrent_log[layer_log.layer_label]
                assert isinstance(result, LayerLog)
                assert result is layer_log
                break

    def test_getitem_pass_notation_returns_pass_log(self, recurrent_log):
        """log['label:2'] still returns LayerPassLog."""
        for layer_log in recurrent_log.layer_logs.values():
            if layer_log.num_passes > 1:
                for pass_num, pass_log in layer_log.passes.items():
                    result = recurrent_log[pass_log.layer_label]
                    assert isinstance(result, LayerPassLog)
                break

    def test_pass_labels_populated(self, recurrent_log):
        """pass_labels list contains the w-pass labels of each pass."""
        for layer_log in recurrent_log.layer_logs.values():
            assert len(layer_log.pass_labels) == layer_log.num_passes
            for i, pass_label in enumerate(layer_log.pass_labels, 1):
                assert layer_log.passes[i].layer_label == pass_label


# ---------------------------------------------------------------------------
# __str__ and __repr__
# ---------------------------------------------------------------------------


class TestLayerLogDisplay:
    def test_str_single_pass(self, simple_log):
        for layer_log in simple_log.layer_logs.values():
            s = str(layer_log)
            assert "Layer " in s
            assert layer_log.layer_label in s

    def test_str_multi_pass(self, recurrent_log):
        for layer_log in recurrent_log.layer_logs.values():
            if layer_log.num_passes > 1:
                s = str(layer_log)
                assert "passes" in s.lower()
                break

    def test_repr_eq_str(self, simple_log):
        for layer_log in simple_log.layer_logs.values():
            assert repr(layer_log) == str(layer_log)


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------


class TestConvenienceAliases:
    def test_layer_passes_total_alias(self, simple_log):
        for layer_log in simple_log.layer_logs.values():
            assert layer_log.layer_passes_total == layer_log.num_passes

    def test_layer_label_no_pass_alias(self, simple_log):
        for layer_log in simple_log.layer_logs.values():
            assert layer_log.layer_label_no_pass == layer_log.layer_label

    def test_params_accessor(self, recurrent_log):
        for layer_log in recurrent_log.layer_logs.values():
            if layer_log.computed_with_params:
                params = layer_log.params
                assert params is not None
                break


# ---------------------------------------------------------------------------
# Integration: layer_logs on ModelLog
# ---------------------------------------------------------------------------


class TestModelLogIntegration:
    def test_layer_logs_attr_exists(self, simple_log):
        assert hasattr(simple_log, "layer_logs")

    def test_layer_logs_ordered(self, simple_log):
        """layer_logs preserves layer execution order."""
        labels = list(simple_log.layer_logs.keys())
        # Should be in the same order as layer_list (by first appearance)
        seen = []
        for pass_log in simple_log.layer_list:
            if pass_log.layer_label_no_pass not in seen:
                seen.append(pass_log.layer_label_no_pass)
        assert labels == seen

    def test_len_layer_logs(self, simple_log):
        """Number of LayerLogs equals number of unique no-pass labels."""
        unique = set(pl.layer_label_no_pass for pl in simple_log.layer_list)
        assert len(simple_log.layer_logs) == len(unique)


# ---------------------------------------------------------------------------
# Bugfix regression tests
# ---------------------------------------------------------------------------


class _SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class TestLayerNumPasses:
    """Bug #53: layer_num_passes should be keyed correctly."""

    def test_returns_integer(self):
        model = _SimpleLinear()
        log = tl.log_forward_pass(model, torch.randn(2, 10))
        for label in log.layer_labels_no_pass:
            passes = log.layer_num_passes.get(label)
            assert passes is not None, f"No passes for {label}"
            assert isinstance(passes, int), f"Expected int, got {type(passes)} for {label}"


class TestSliceIndexing:
    """Bug #78: Slice indexing should return list."""

    def test_slice_returns_list(self):
        model = _SimpleLinear()
        log = tl.log_forward_pass(model, torch.randn(2, 10))
        result = log[0:2]
        assert isinstance(result, list)
        assert len(result) == 2


class TestToPandasGuard:
    """Bug #124: to_pandas() should guard against unfinished pass."""

    def test_works_after_pass(self):
        model = _SimpleLinear()
        log = tl.log_forward_pass(model, torch.randn(2, 10))
        try:
            import pandas

            df = log.to_pandas()
            assert df is not None
        except ImportError:
            pytest.skip("pandas not installed")


class TestAmbiguousSubstring:
    """Bug #125: Ambiguous substring should list matching layers."""

    def test_lists_matches(self):
        class TwoLinears(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = nn.Linear(10, 10)
                self.lin2 = nn.Linear(10, 10)

            def forward(self, x):
                return self.lin1(x) + self.lin2(x)

        model = TwoLinears()
        log = tl.log_forward_pass(model, torch.randn(2, 10))
        with pytest.raises(ValueError, match="Ambiguous"):
            log["linear"]


class TestBug23CaseInsensitiveLookup:
    """#23: Case-insensitive lookup for layer/module names."""

    def test_case_insensitive_exact_match(self):
        model = _SimpleLinear()
        log = tl.log_forward_pass(model, torch.randn(2, 10))
        result = log["Linear_1_1"]
        assert result is not None
        assert result.layer_label == "linear_1_1"

    def test_case_insensitive_module_lookup(self):
        model = _SimpleLinear()
        log = tl.log_forward_pass(model, torch.randn(2, 10))
        result = log["FC"]
        assert result is not None


class TestBug83ParentLayerArgLocs:
    """#83: LayerLog.parent_layer_arg_locs should return strings, not sets."""

    def test_arg_locs_are_strings(self):
        model = _SimpleLinear()
        log = tl.log_forward_pass(model, torch.randn(2, 10))
        for layer in log:
            arg_locs = layer.parent_layer_arg_locs
            for arg_type in ["args", "kwargs"]:
                for key, val in arg_locs[arg_type].items():
                    assert isinstance(val, str), (
                        f"Expected string for arg_locs['{arg_type}'][{key}], got {type(val)}"
                    )
