"""Tests for Layer: aggregate per-layer metadata."""

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes.layer import Layer
from torchlens.data_classes.trace import Trace
from torchlens.data_classes.op import Op


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
    return tl.trace(model, torch.randn(1, 5), layers_to_save="all")


@pytest.fixture
def recurrent_log():
    model = RecurrentModel()
    return tl.trace(model, torch.randn(1, 5), layers_to_save="all")


def _assert_layer_labels_are_not_doubled(trace: Trace) -> None:
    """Assert trace-level layer labels correspond one-to-one with layer logs.

    Parameters
    ----------
    trace:
        Trace whose public layer label accessor should be checked.
    """

    assert len(trace.layer_labels) == len(trace.layers)
    assert all(
        previous_label != next_label
        for previous_label, next_label in zip(trace.layer_labels, trace.layer_labels[1:])
    )


# ---------------------------------------------------------------------------
# Basic Layer construction
# ---------------------------------------------------------------------------


class TestLayerLogConstruction:
    @pytest.mark.smoke
    def test_layer_logs_populated(self, simple_log):
        """layer_logs dict is populated after postprocessing."""
        assert len(simple_log.layer_logs) > 0

    def test_layer_logs_keyed_by_no_call_label(self, simple_log):
        """Keys are no-pass layer labels."""
        for key, layer_log in simple_log.layer_logs.items():
            assert ":" not in key
            assert isinstance(layer_log, Layer)

    @pytest.mark.smoke
    def test_trace_layer_labels_not_doubled_for_recurrent_model(
        self: "TestLayerLogConstruction",
        recurrent_log: Trace,
    ) -> None:
        """Trace layer labels are not doubled for a multi-call recurrent model."""

        _assert_layer_labels_are_not_doubled(recurrent_log)

    def test_single_pass_layer_has_one_pass(self, simple_log):
        """Non-recurrent model: every Layer has exactly one pass."""
        for layer_log in simple_log.layer_logs.values():
            assert layer_log.num_passes == 1
            assert len(layer_log.ops) == 1
            assert 0 in layer_log.ops

    def test_trace_layer_labels_not_doubled_for_tiny_transformers_model(
        self: "TestLayerLogConstruction",
    ) -> None:
        """Trace layer labels are not doubled for an optional tiny Transformers model."""

        transformers = pytest.importorskip("transformers")
        config = transformers.DistilBertConfig(
            vocab_size=101,
            max_position_embeddings=16,
            n_layers=1,
            n_heads=2,
            dim=16,
            hidden_dim=32,
            dropout=0.0,
            attention_dropout=0.0,
        )
        model = transformers.DistilBertModel(config)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        trace = tl.trace(model, input_ids, layers_to_save="all")

        _assert_layer_labels_are_not_doubled(trace)


# ---------------------------------------------------------------------------
# Single-pass delegation
# ---------------------------------------------------------------------------


class TestSinglePassDelegation:
    @pytest.mark.smoke
    def test_tensor_contents_delegation(self, simple_log):
        """out delegates to ops[0] for single-pass."""
        for layer_log in simple_log.layer_logs.values():
            pass_log = layer_log.ops[0]
            assert layer_log.out is pass_log.out

    def test_has_saved_outs_delegation(self, simple_log):
        for layer_log in simple_log.layer_logs.values():
            pass_log = layer_log.ops[0]
            assert layer_log.has_saved_activation == pass_log.has_saved_activation

    def test_compute_index_delegation(self, simple_log):
        for layer_log in simple_log.layer_logs.values():
            pass_log = layer_log.ops[0]
            assert layer_log.step_index == pass_log.step_index

    def test_fallback_getattr_delegation(self, simple_log):
        """__getattr__ fallback delegates arbitrary per-pass fields."""
        for layer_log in simple_log.layer_logs.values():
            pass_log = layer_log.ops[0]
            # func_autocast_state is per-pass, not an explicit @property
            assert layer_log.func_autocast_state is pass_log.func_autocast_state

    def test_tracing_finished_reads_from_trace(self, simple_log):
        for layer_log in simple_log.layer_logs.values():
            assert layer_log._tracing_finished is True


# ---------------------------------------------------------------------------
# Aggregate fields
# ---------------------------------------------------------------------------


class TestAggregateFields:
    def test_aggregate_fields_match_first_pass(self, simple_log):
        """Aggregate fields on Layer match the corresponding first-pass values."""
        for layer_log in simple_log.layer_logs.values():
            fp = layer_log.ops[0]
            assert layer_log.layer_type == fp.layer_type
            assert layer_log.func_name == fp.func_name
            assert layer_log.shape == fp.shape
            assert layer_log.dtype == fp.dtype
            assert layer_log.is_input == fp.is_input
            assert layer_log.is_output == fp.is_output
            assert layer_log.uses_params == fp.uses_params

    def test_layer_label_is_no_pass(self, simple_log):
        for layer_log in simple_log.layer_logs.values():
            fp = layer_log.ops[0]
            assert layer_log.layer_label == fp.layer_label


# ---------------------------------------------------------------------------
# Multi-pass (recurrent) Layer
# ---------------------------------------------------------------------------


class TestMultiPassLayerLog:
    @pytest.mark.smoke
    def test_recurrent_layer_has_multiple_ops(self, recurrent_log):
        """The repeated fc layer should have 2 ops."""
        multi_pass_found = False
        for layer_log in recurrent_log.layer_logs.values():
            if layer_log.num_passes > 1:
                multi_pass_found = True
                assert len(layer_log.ops) == layer_log.num_passes
        assert multi_pass_found, "Expected at least one multi-pass layer in recurrent model"

    def test_multi_pass_tensor_contents_raises(self, recurrent_log):
        """Accessing per-pass field on multi-pass Layer raises ValueError."""
        for layer_log in recurrent_log.layer_logs.values():
            if layer_log.num_passes > 1:
                with pytest.raises(ValueError, match="has .* ops"):
                    _ = layer_log.out
                break

    def test_multi_pass_getattr_raises(self, recurrent_log):
        """__getattr__ fallback raises for multi-pass layers."""
        for layer_log in recurrent_log.layer_logs.values():
            if layer_log.num_passes > 1:
                with pytest.raises(AttributeError):
                    _ = layer_log.func_autocast_state
                break

    def test_multi_pass_aggregate_graph_union(self, recurrent_log):
        """Aggregate graph properties are unions across ops."""
        for layer_log in recurrent_log.layer_logs.values():
            if layer_log.num_passes > 1 and layer_log.has_children:
                # Verify children is a union
                all_children = set()
                for pass_log in layer_log.ops.values():
                    for child in pass_log.children:
                        all_children.add(recurrent_log[child].layer_label)
                assert set(layer_log.children) == all_children
                break

    def test_getitem_returns_layer_log_for_multi_pass(self, recurrent_log):
        """log['label'] returns Layer for multi-pass layers."""
        for layer_log in recurrent_log.layer_logs.values():
            if layer_log.num_passes > 1:
                result = recurrent_log[layer_log.layer_label]
                assert isinstance(result, Layer)
                assert result is layer_log
                break

    def test_getitem_pass_notation_returns_pass_log(self, recurrent_log):
        """log['label:2'] still returns Op."""
        for layer_log in recurrent_log.layer_logs.values():
            if layer_log.num_passes > 1:
                for pass_index, pass_log in layer_log.ops.items():
                    result = recurrent_log[pass_log.label]
                    assert isinstance(result, Op)
                break

    def test_call_labels_populated(self, recurrent_log):
        """call_labels list contains the w-pass labels of each pass."""
        for layer_log in recurrent_log.layer_logs.values():
            assert len(layer_log.call_labels) == layer_log.num_passes
            for i, call_label in enumerate(layer_log.call_labels):
                assert layer_log.ops[i].label == call_label


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
                assert "ops" in s.lower()
                break

    def test_repr_eq_str(self, simple_log):
        for layer_log in simple_log.layer_logs.values():
            assert repr(layer_log) == str(layer_log)


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------


class TestConvenienceAliases:
    def test_layer_label_no_pass_alias(self, simple_log):
        for layer_log in simple_log.layer_logs.values():
            assert layer_log.layer_label == layer_log.layer_label

    def test_params_accessor(self, recurrent_log):
        for layer_log in recurrent_log.layer_logs.values():
            if layer_log.uses_params:
                params = layer_log.params
                assert params is not None
                break


# ---------------------------------------------------------------------------
# Integration: layer_logs on Trace
# ---------------------------------------------------------------------------


class TestTraceIntegration:
    def test_layer_logs_attr_exists(self, simple_log):
        assert hasattr(simple_log, "layer_logs")

    def test_layer_logs_ordered(self, simple_log):
        """layer_logs preserves layer execution order."""
        labels = list(simple_log.layer_logs.keys())
        # Should be in the same order as layer_list (by first appearance)
        seen = []
        for pass_log in simple_log.layer_list:
            if pass_log.layer_label not in seen:
                seen.append(pass_log.layer_label)
        assert labels == seen

    def test_len_layer_logs(self, simple_log):
        """Number of LayerLogs equals number of unique no-pass labels."""
        unique = set(pl.layer_label for pl in simple_log.layer_list)
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
    """layer_num_calls should be keyed correctly."""

    def test_returns_integer(self):
        model = _SimpleLinear()
        log = tl.trace(model, torch.randn(2, 10))
        for label in log.layer_labels:
            ops = log.layer_num_calls.get(label)
            assert ops is not None, f"No ops for {label}"
            assert isinstance(ops, int), f"Expected int, got {type(ops)} for {label}"


class TestSliceIndexing:
    """Slice indexing should return list."""

    def test_slice_returns_list(self):
        model = _SimpleLinear()
        log = tl.trace(model, torch.randn(2, 10))
        result = log[0:2]
        assert isinstance(result, list)
        assert len(result) == 2


class TestToPandasGuard:
    """to_pandas() should guard against unfinished pass."""

    def test_works_after_pass(self):
        model = _SimpleLinear()
        log = tl.trace(model, torch.randn(2, 10))
        try:
            import pandas

            df = log.to_pandas()
            assert df is not None
        except ImportError:
            pytest.skip("pandas not installed")


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

    def test_case_insensitive_exact_match(self):
        model = _SimpleLinear()
        log = tl.trace(model, torch.randn(2, 10))
        result = log["Linear_1_1"]
        assert result is not None
        assert result.layer_label == "linear_1_1"

    def test_case_insensitive_module_lookup(self):
        model = _SimpleLinear()
        log = tl.trace(model, torch.randn(2, 10))
        result = log["FC"]
        assert result is not None


class TestParentLayerArgLocs:
    """Layer.parent_arg_positions should return strings, not sets."""

    def test_arg_locs_are_strings(self):
        model = _SimpleLinear()
        log = tl.trace(model, torch.randn(2, 10))
        for layer in log:
            arg_locs = layer.parent_arg_positions
            for arg_type in ["args", "kwargs"]:
                for key, val in arg_locs[arg_type].items():
                    assert isinstance(val, str), (
                        f"Expected string for arg_locs['{arg_type}'][{key}], got {type(val)}"
                    )
