"""Tests for Module, ModuleCall, and ModuleAccessor."""

import example_models
import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens import trace as trace_fn
from torchlens.types import Module, ModuleCall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_model():
    return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))


def _simple_input():
    return torch.randn(1, 10)


def _make_nested_model():
    """Model with nested submodules for hierarchy tests."""
    return nn.Sequential(
        nn.Sequential(nn.Linear(10, 8), nn.ReLU()),
        nn.Sequential(nn.Linear(8, 4), nn.Sigmoid()),
        nn.Linear(4, 2),
    )


def _nested_input():
    return torch.randn(1, 10)


# ---------------------------------------------------------------------------
# TestModuleLogBasic
# ---------------------------------------------------------------------------


class TestModuleLogBasic:
    def test_root_module_alias(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        assert log.modules[""] is log.modules["self"]

    def test_getitem_multi_pass_returns_module_log(self, input_2d):
        """log["fc1"] for a multi-pass module should return Module (instead of error)."""
        model = example_models.RecurrentParamsSimple()
        log = trace_fn(model, input_2d)
        result = log["fc1"]
        assert isinstance(result, Module)


# ---------------------------------------------------------------------------
# TestModuleLogFields
# ---------------------------------------------------------------------------


class TestModuleLogFields:
    def test_call_depth(self):
        log = trace_fn(_make_nested_model(), _nested_input())
        root = log.modules["self"]
        assert root.call_depth == 0

        # Top-level module should be depth 1
        top = log.modules["0"]
        assert top.call_depth == 1

        # Nested inside "0" should be depth 2
        nested = log.modules["0.0"]
        assert nested.call_depth == 2

    def test_address_depth(self):
        log = trace_fn(_make_nested_model(), _nested_input())
        assert log.modules["self"].address_depth == 0
        assert log.modules["0"].address_depth == 1
        assert log.modules["0.0"].address_depth == 2


# ---------------------------------------------------------------------------
# TestMultiPassModules
# ---------------------------------------------------------------------------


class TestMultiPassModules:
    def test_per_call_field_raises(self, input_2d):
        model = example_models.RecurrentParamsSimple()
        log = trace_fn(model, input_2d)
        ml = log.modules["fc1"]
        assert ml.num_calls > 1
        assert ml.layer_labels
        with pytest.raises(AttributeError, match="ops"):
            _ = ml.forward_args

    def test_pass_access(self, input_2d):
        model = example_models.RecurrentParamsSimple()
        log = trace_fn(model, input_2d)
        ml = log.modules["fc1"]
        assert 0 in ml.ops
        assert 1 in ml.ops
        assert isinstance(ml.ops[0], ModuleCall)
        assert isinstance(ml.ops[1], ModuleCall)


# ---------------------------------------------------------------------------
# TestModuleAccessorSummary
# ---------------------------------------------------------------------------


class TestModuleAccessorSummary:
    def test_recurrent_to_pandas_and_root_aggregates(self, input_2d):
        """Recurrent models must not crash Module.to_pandas.

        Regression: the root ``self:1`` call stores bare Layer labels (per the
        function-root-module invariant), and a recurrent layer label resolves
        to MULTIPLE Ops. Aggregate ModuleCall properties that iterate
        ``trace.ops[label]`` used to raise ``AmbiguousOpLookupError`` on such a
        label, crashing the field-order-driven module table. The root aggregate
        must instead sum over every pass, and multi-call submodules must not
        crash the per-pass columns.
        """

        from torchlens.validation.invariants import check_metadata_invariants

        model = example_models.RecurrentParamsSimple()
        # activation_transform=identity forces transformed_out to be populated
        # (it stays None with no transform), so the single-pass-vs-multi-pass
        # None-guard assertions below are meaningful rather than trivially true.
        log = trace_fn(
            model, input_2d, capture=tl.options.CaptureOptions(activation_transform=lambda x: x)
        )

        # The whole table renders without raising.
        df = log.modules.to_pandas()
        assert len(df) == len(log.modules)

        # Root aggregates sum over ALL ops (every recurrent pass), with no
        # double-count and no miss versus a manual sum over log.ops.
        root = log.root_module
        all_ops = list(log._layer_list)
        manual_func = sum(getattr(op, "func_duration", 0.0) or 0.0 for op in all_ops)
        manual_autograd = sum(int(getattr(op, "autograd_memory", 0) or 0) for op in all_ops)
        out_label_set = set(root.calls[0].output_ops)
        manual_out_act = sum(
            int(getattr(op, "activation_memory", 0) or 0)
            for op in all_ops
            if op.label in out_label_set or op.layer_label in out_label_set
        )
        assert manual_out_act > 0  # cross-check actually exercises some ops
        assert float(root.func_calls_duration) == pytest.approx(manual_func)
        assert int(root.total_autograd_memory) == manual_autograd
        assert int(root.total_output_activation_memory) == manual_out_act

        # A multi-call submodule reports per-pass columns as None (mirroring the
        # output_structure precedent) rather than raising; total_* still carry
        # the aggregate.
        fc1 = log.modules["fc1"]
        assert fc1.num_calls > 1
        fc1_row = df[df["address"] == "fc1"].iloc[0]
        assert (
            fc1_row["func_calls_duration"] is None
            or fc1_row["func_calls_duration"] != (fc1_row["func_calls_duration"])
        )  # None or NaN
        assert float(fc1_row["total_func_calls_duration"]) >= 0.0

        # BLOCKER regression (cert5/cert6): the SAME bug class one layer down --
        # ``Layer.to_pandas()``, ``LayerAccessor.to_pandas()`` (trace.layers.to_pandas()),
        # and a single-module ``Module["addr"].to_pandas()`` all delegate to
        # ``transformed_out``/``transformed_grad``, which raise ValueError via
        # ``Layer._single_pass_or_error()`` for any multi-pass (recurrent) layer.
        # None of these three surfaces were covered by the assertions above, which
        # is exactly why the regression slipped through hotfix2.
        multi_pass_labels = [
            label for label in fc1.layer_labels if log.layers[label].num_passes > 1
        ]
        assert multi_pass_labels, "fixture must contain a multi-pass layer to exercise this gap"

        # 1) Layer.to_pandas() directly.
        multi_pass_layer = log.layers[multi_pass_labels[0]]
        layer_df = multi_pass_layer.to_pandas()
        assert len(layer_df) == 1
        assert layer_df.iloc[0]["transformed_out"] is None
        assert layer_df.iloc[0]["transformed_grad"] is None

        # 2) trace.layers.to_pandas() (LayerAccessor).
        layers_df = log.layers.to_pandas()
        assert len(layers_df) == len(log.layers)
        multi_pass_row = layers_df[layers_df["layer_label"] == multi_pass_labels[0]].iloc[0]
        assert multi_pass_row["transformed_out"] is None
        assert multi_pass_row["transformed_grad"] is None
        # Single-pass layers keep their real (non-None) per-pass values.
        single_pass_row = layers_df[layers_df["num_passes"] == 1].iloc[0]
        assert single_pass_row["transformed_out"] is not None

        # 3) trace.modules["addr"].to_pandas() -- the per-module layer export.
        fc1_layers_df = fc1.to_pandas()
        assert len(fc1_layers_df) == len(fc1.layer_labels)
        fc1_layer_row = fc1_layers_df[fc1_layers_df["layer_label"] == multi_pass_labels[0]].iloc[0]
        assert fc1_layer_row["transformed_out"] is None
        assert fc1_layer_row["transformed_grad"] is None

        assert check_metadata_invariants(log) is True

    def test_summary(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        s = log.modules.summary()
        assert isinstance(s, str)
        assert len(s) > 0
        assert "Address" in s

    def test_repr(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        r = repr(log.modules)
        assert "ModuleAccessor" in r


# ---------------------------------------------------------------------------
# TestModuleLogIntegration
# ---------------------------------------------------------------------------


class TestModuleLogIntegration:
    def test_module_log_iter(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        ml = log.modules["0"]
        entries = list(ml)
        assert len(entries) == len(ml.layer_labels)

    def test_module_log_getitem(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        ml = log.modules["0"]
        if len(ml.layer_labels) > 0:
            entry = ml[0]
            assert entry.layer_label == ml.layer_labels[0]


# ---------------------------------------------------------------------------
# Bugfix regression tests
# ---------------------------------------------------------------------------


class _SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
