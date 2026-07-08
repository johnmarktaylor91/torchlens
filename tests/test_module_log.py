"""Tests for Module, ModuleCall, and ModuleAccessor."""

import pytest
import torch
import torch.nn as nn

import example_models
from torchlens import trace as trace_fn
from torchlens.types import Module, ModuleCall
from torchlens.data_classes import ModuleAccessor, ParamAccessor


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


class _RepeatedBatchNormLeaf(nn.Module):
    """Leaf module with a BatchNorm site and registered running buffers."""

    def __init__(self, channels: int) -> None:
        """Initialize the convolution and BatchNorm layers.

        Parameters
        ----------
        channels:
            Number of feature channels preserved by the leaf block.
        """

        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the leaf block.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Activated output feature map.
        """

        return torch.relu(self.bn(self.conv(x)))


class _RepeatedBatchNormStack(nn.Module):
    """Stack of identical BatchNorm-containing leaf modules."""

    def __init__(self, depth: int = 4, channels: int = 3) -> None:
        """Initialize the repeated BatchNorm stack.

        Parameters
        ----------
        depth:
            Number of repeated leaf modules.
        channels:
            Number of channels preserved by every leaf module.
        """

        super().__init__()
        self.blocks = nn.Sequential(*(_RepeatedBatchNormLeaf(channels) for _ in range(depth)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the repeated BatchNorm stack.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Output feature map after every leaf module.
        """

        return self.blocks(x)


# ---------------------------------------------------------------------------
# TestModuleLogBasic
# ---------------------------------------------------------------------------


class TestModuleLogBasic:
    @pytest.mark.smoke
    def test_modules_accessor_exists(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        assert isinstance(log.modules, ModuleAccessor)

    def test_root_module_exists(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        root = log.modules["self"]
        assert isinstance(root, Module)
        assert root.address == "self"

    def test_root_module_alias(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        assert log.modules[""] is log.modules["self"]

    def test_root_module_property(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        assert log.root_module is log.modules["self"]

    def test_module_count(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        # Sequential has 3 children: Linear, ReLU, Linear → 3 submodules + root = 4
        assert len(log.modules) >= 4

    @pytest.mark.smoke
    def test_access_by_address(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        ml = log.modules["0"]
        assert isinstance(ml, Module)
        assert ml.address == "0"

    def test_access_by_index(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        ml = log.modules[0]
        assert isinstance(ml, Module)

    def test_call_access_by_pass_notation(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        # All modules have 1 pass in a non-recurrent model
        addresses = [ml.address for ml in log.modules if ml.address != "self"]
        if addresses:
            addr = addresses[0]
            assert isinstance(log.modules[f"{addr}:1"], Module)
            mpl = log.module_calls[f"{addr}:1"]
            assert isinstance(mpl, ModuleCall)

    def test_contains(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        assert "0" in log.modules
        assert "self" in log.modules
        assert "nonexistent" not in log.modules

    def test_iter(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        modules_list = list(log.modules)
        assert len(modules_list) == len(log.modules)
        assert all(isinstance(ml, Module) for ml in modules_list)

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
    def test_identity_fields(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        ml = log.modules["0"]
        assert ml.address == "0"
        assert ml.name == "0"
        assert ml.class_name == "Linear"

    def test_source_info(self):
        log = trace_fn(_make_simple_model(), _simple_input(), save_code_context=True)
        ml = log.modules["0"]
        assert ml.class_source_file is not None  # nn.Linear has inspectable source
        assert ml.forward_signature is not None

    def test_hierarchy_address(self):
        log = trace_fn(_make_nested_model(), _nested_input())
        # "0.0" is Linear inside first Sequential
        ml = log.modules["0.0"]
        assert ml.address_parent == "0"
        assert ml.address_depth == 2

        # "0" is the first Sequential
        parent = log.modules["0"]
        assert parent.address_parent == "self"
        assert "0.0" in parent.address_children

    def test_hierarchy_call(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        root = log.modules["self"]
        # Root's call_children should include top-level modules
        assert len(root.call_children) > 0

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

    def test_layers_populated(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        ml = log.modules["0"]
        assert len(ml.layers) > 0
        assert ml.num_layers == len(ml.layers)

    def test_params_accessor(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        ml = log.modules["0"]  # Linear layer
        assert isinstance(ml.params, ParamAccessor)
        assert len(ml.params) == 2  # weight + bias

    def test_training_mode(self):
        model = _make_simple_model()
        model.eval()
        log = trace_fn(model, _simple_input())
        for ml in log.modules:
            if ml.address != "self":
                assert ml.training is False

    def test_hooks_detected_false(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        for ml in log.modules:
            assert ml.has_forward_hooks is False
            assert ml.has_backward_hooks is False

    def test_repr(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        ml = log.modules["0"]
        r = repr(ml)
        assert "Module" in r
        assert ml.address in r
        assert ml.class_name in r


# ---------------------------------------------------------------------------
# TestModuleCallLog
# ---------------------------------------------------------------------------


class TestModuleCallLog:
    def test_pass_layers(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        ml = log.modules["0"]
        mpl = ml.ops[0]
        assert isinstance(mpl, ModuleCall)
        # Pass layers should be a subset of parent layers
        assert all(label.split(":", 1)[0] in ml.layer_labels for label in mpl.ops)

    def test_input_output_layers(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        ml = log.modules["0"]
        mpl = ml.ops[0]
        assert isinstance(mpl.input_layers, list)
        assert isinstance(mpl.output_layers, list)

    def test_call_children(self):
        log = trace_fn(_make_nested_model(), _nested_input())
        # "0" contains "0.0" and "0.1" as submodules
        ml = log.modules["0"]
        mpl = ml.ops[0]
        assert isinstance(mpl.call_children, list)

    def test_repr(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        ml = log.modules["0"]
        mpl = ml.ops[0]
        r = repr(mpl)
        assert "ModuleCall" in r
        assert len(r) > 0


# ---------------------------------------------------------------------------
# TestMultiPassModules
# ---------------------------------------------------------------------------


class TestMultiPassModules:
    def test_num_calls_gt_1(self, input_2d):
        model = example_models.RecurrentParamsSimple()
        log = trace_fn(model, input_2d)
        # fc1 is used 4 times
        ml = log.modules["fc1"]
        assert ml.num_calls >= 2

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

    def test_pass_notation_accessor(self, input_2d):
        model = example_models.RecurrentParamsSimple()
        log = trace_fn(model, input_2d)
        assert isinstance(log.modules["fc1:2"], Module)
        mpl = log.module_calls["fc1:2"]
        assert isinstance(mpl, ModuleCall)
        assert mpl.call_index == 2


# ---------------------------------------------------------------------------
# TestSinglePassDelegation
# ---------------------------------------------------------------------------


class TestSinglePassDelegation:
    def test_layers_delegates(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        ml = log.modules["0"]
        assert ml.num_calls == 1
        assert ml.layer_labels == [label.split(":", 1)[0] for label in ml.ops[0].ops]

    def test_forward_args_delegates(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        ml = log.modules["0"]
        # forward_args should be accessible for single-pass, delegating to
        # the single ModuleCall's own forward_args (mirrors test_layers_delegates).
        assert ml.forward_args == ml.ops[0].forward_args


# ---------------------------------------------------------------------------
# TestModuleAccessorSummary
# ---------------------------------------------------------------------------


class TestModuleAccessorSummary:
    def test_to_pandas(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        df = log.modules.to_pandas()
        assert len(df) == len(log.modules)
        assert "address" in df.columns
        assert "class_name" in df.columns
        assert "call_depth" in df.columns
        assert "num_params" in df.columns

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
        log = trace_fn(model, input_2d, activation_transform=lambda x: x)

        # The whole table renders without raising.
        df = log.modules.to_pandas()
        assert len(df) == len(log.modules)

        # Root aggregates sum over ALL ops (every recurrent pass), with no
        # double-count and no miss versus a manual sum over log.ops.
        root = log.root_module
        all_ops = list(log.ops)
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
        multi_pass_labels = [label for label in fc1.layer_labels if log[label].num_passes > 1]
        assert multi_pass_labels, "fixture must contain a multi-pass layer to exercise this gap"

        # 1) Layer.to_pandas() directly.
        multi_pass_layer = log[multi_pass_labels[0]]
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
        assert len(fc1_layers_df) == fc1.num_layers
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
    def test_root_layers_equals_model_layers(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        root = log.root_module
        assert root.layer_labels == [layer.layer_label for layer in log.layer_list]

    def test_root_params_count(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        root = log.root_module
        assert root.num_params == log.num_params

    def test_class_name_matches(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        # Module "0" is Linear
        assert log.modules["0"].class_name == "Linear"
        # Module "1" is ReLU
        assert log.modules["1"].class_name == "ReLU"
        # Module "2" is Linear
        assert log.modules["2"].class_name == "Linear"

    def test_nested_model_hierarchy(self):
        log = trace_fn(_make_nested_model(), _nested_input())
        # Check that nesting is consistent
        for ml in log.modules:
            if ml.address == "self":
                continue
            # address_parent should be a valid module
            assert ml.address_parent in log.modules

    def test_module_log_to_pandas(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        ml = log.modules["0"]
        df = ml.to_pandas()
        assert len(df) == ml.num_layers

    def test_module_log_iter(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        ml = log.modules["0"]
        entries = list(ml)
        assert len(entries) == ml.num_layers

    def test_module_log_getitem(self):
        log = trace_fn(_make_simple_model(), _simple_input())
        ml = log.modules["0"]
        if ml.num_layers > 0:
            entry = ml[0]
            assert entry.layer_label == ml.layer_labels[0]

    def test_nested_modules_model(self, input_2d):
        """Integration test with the NestedModules example model."""
        model = example_models.NestedModules()
        log = trace_fn(model, input_2d)
        assert len(log.modules) > 1
        root = log.root_module
        assert root.address == "self"
        # Should have nested hierarchy
        max_depth = max(ml.call_depth for ml in log.modules)
        assert max_depth >= 2  # At least 3 levels of nesting

    def test_first_repeated_batchnorm_module_owns_only_local_layers(self) -> None:
        """Internal buffer-address probes must not inflate first BatchNorm module layers."""

        model = _RepeatedBatchNormStack(depth=4).eval()
        log = trace_fn(model, torch.randn(1, 3, 8, 8))

        block_layers = [log.modules[f"blocks.{idx}"].layer_labels for idx in range(4)]
        block_counts = [log.modules[f"blocks.{idx}"].num_layers for idx in range(4)]
        batchnorm_counts = [log.modules[f"blocks.{idx}.bn"].num_layers for idx in range(4)]

        assert block_counts == [7, 7, 7, 7]
        assert [len(labels) for labels in block_layers] == block_counts
        assert batchnorm_counts == [5, 5, 5, 5]
        for labels in block_layers:
            assert [label.split("_", 1)[0] for label in labels].count("buffer") == 4
            assert [label.split("_", 1)[0] for label in labels].count("batchnorm") == 1
            assert [label.split("_", 1)[0] for label in labels].count("conv2d") == 1
            assert [label.split("_", 1)[0] for label in labels].count("relu") == 1


# ---------------------------------------------------------------------------
# Bugfix regression tests
# ---------------------------------------------------------------------------


class _SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class TestModuleLogStringIndexing:
    """Module should support string label lookup."""

    def test_module_string_lookup(self):
        model = _SimpleLinear()
        log = trace_fn(model, torch.randn(2, 10))
        if hasattr(log, "_module_logs") and log._module_logs:
            first_key = list(log._module_logs._dict.keys())[0]
            mod = log._module_logs[first_key]
            assert mod is not None


class TestTupleStringNormalization:
    """modules should handle both tuple and string formats."""

    def test_module_hierarchy_with_nested_model(self):
        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()

            def forward(self, x):
                return self.inner(x)

        model = Outer()
        log = trace_fn(model, torch.randn(2, 10))
        assert len(log.modules) > 0
        assert "inner" in log.modules
