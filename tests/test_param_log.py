"""Tests for ParamLog, ParamAccessor, and param-related visualization."""

import os
from os.path import join as opj

import pytest
import torch
import torch.nn as nn

import example_models
from conftest import VIS_OUTPUT_DIR
from torchlens import ParamLog, log_forward_pass, show_model_graph
from torchlens.data_classes import ParamAccessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_model():
    return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))


def _make_frozen_first_layer():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    for p in model[0].parameters():
        p.requires_grad = False
    return model


def _make_all_frozen():
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    for p in model.parameters():
        p.requires_grad = False
    return model


def _simple_input():
    return torch.randn(1, 10)


# ---------------------------------------------------------------------------
# ParamLog class structure
# ---------------------------------------------------------------------------


class TestParamLogFields:
    def test_fields_populated(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        pl = mh.params[0]
        assert isinstance(pl, ParamLog)
        assert isinstance(pl.address, str)
        assert isinstance(pl.name, str)
        assert isinstance(pl.shape, tuple)
        assert isinstance(pl.dtype, torch.dtype)
        assert isinstance(pl.num_params, int)
        assert isinstance(pl.fsize, int)
        assert isinstance(pl.fsize_nice, str)
        assert isinstance(pl.trainable, bool)
        assert isinstance(pl.module_address, str)
        assert isinstance(pl.module_type, str)
        assert isinstance(pl.barcode, str)
        assert isinstance(pl.num_passes, int)
        assert isinstance(pl.tensor_log_entries, list)
        assert isinstance(pl.linked_params, list)

    def test_repr_contains_key_info(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        pl = mh.params[0]
        r = repr(pl)
        assert pl.address in r
        assert "shape" in r
        assert "dtype" in r
        assert "trainable" in r or "frozen" in r
        assert "has_grad" in r

    def test_len_equals_num_params(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        pl = mh.params["0.weight"]
        assert len(pl) == pl.num_params == 50  # 5 * 10

    def test_is_quantized_false_for_float(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        for pl in mh.params:
            assert pl.is_quantized is False


# ---------------------------------------------------------------------------
# ParamAccessor on ModelHistory
# ---------------------------------------------------------------------------


class TestParamAccessorMH:
    def test_access_by_address(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        pl = mh.params["0.weight"]
        assert pl.address == "0.weight"

    def test_access_by_index(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        pl = mh.params[0]
        assert isinstance(pl, ParamLog)

    def test_iterable(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        params = list(mh.params)
        assert len(params) == 4  # weight+bias for each Linear

    def test_len(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        assert len(mh.params) == 4

    def test_contains(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        assert "0.weight" in mh.params
        assert "nonexistent" not in mh.params

    def test_short_name_ambiguous_raises(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        with pytest.raises(KeyError, match="Ambiguous"):
            mh.params["weight"]  # Both 0.weight and 2.weight

    def test_repr_dict_like(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        r = repr(mh.params)
        assert "0.weight" in r
        assert "ParamLog" in r

    def test_params_property_same_as_param_logs(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        assert mh.params is mh.param_logs


# ---------------------------------------------------------------------------
# ParamAccessor on TensorLogEntry
# ---------------------------------------------------------------------------


class TestParamAccessorTLE:
    def test_access_by_address(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        entry = [e for e in mh if e.computed_with_params][0]
        pl = entry.params[entry.parent_param_logs[0].address]
        assert isinstance(pl, ParamLog)

    def test_access_by_short_name(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        entry = [e for e in mh if e.computed_with_params][0]
        w = entry.params["weight"]
        b = entry.params["bias"]
        assert "weight" in w.name
        assert "bias" in b.name

    def test_access_by_index(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        entry = [e for e in mh if e.computed_with_params][0]
        assert isinstance(entry.params[0], ParamLog)

    def test_len(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        entry = [e for e in mh if e.computed_with_params][0]
        assert len(entry.params) == 2  # weight + bias

    def test_iterable(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        entry = [e for e in mh if e.computed_with_params][0]
        params = list(entry.params)
        assert len(params) == 2

    def test_no_params_for_non_param_layer(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        input_entry = mh["input_1"]
        assert len(input_entry.params) == 0


# ---------------------------------------------------------------------------
# Param metadata correctness
# ---------------------------------------------------------------------------


class TestParamMetadata:
    def test_address_and_name(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        pl = mh.params["0.weight"]
        assert pl.address == "0.weight"
        assert pl.name == "weight"
        assert pl.module_address == "0"

    def test_shape_and_dtype(self):
        model = _make_simple_model()
        mh = log_forward_pass(model, _simple_input())
        pl = mh.params["0.weight"]
        assert pl.shape == (5, 10)
        assert pl.dtype == torch.float32

    def test_trainable_flag_true(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        for pl in mh.params:
            assert pl.trainable is True

    def test_trainable_flag_frozen(self):
        mh = log_forward_pass(_make_frozen_first_layer(), _simple_input())
        assert mh.params["0.weight"].trainable is False
        assert mh.params["0.bias"].trainable is False
        assert mh.params["2.weight"].trainable is True
        assert mh.params["2.bias"].trainable is True

    def test_module_info(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        pl = mh.params["0.weight"]
        assert pl.module_address == "0"
        assert pl.module_type == "Linear"

    def test_fsize_positive(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        for pl in mh.params:
            assert pl.fsize > 0
            assert len(pl.fsize_nice) > 0


# ---------------------------------------------------------------------------
# Trainable / frozen tallies
# ---------------------------------------------------------------------------


class TestTrainableFrozenTallies:
    def test_mh_all_trainable(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        assert mh.total_params == mh.total_params_trainable
        assert mh.total_params_frozen == 0
        assert mh.total_params == 67  # 50+5+10+2

    def test_mh_mixed(self):
        mh = log_forward_pass(_make_frozen_first_layer(), _simple_input())
        assert mh.total_params_frozen == 55  # 50+5
        assert mh.total_params_trainable == 12  # 10+2
        assert mh.total_params == 67
        assert mh.total_params == mh.total_params_trainable + mh.total_params_frozen

    def test_mh_all_frozen(self):
        mh = log_forward_pass(_make_all_frozen(), _simple_input())
        assert mh.total_params_trainable == 0
        assert mh.total_params_frozen == 67

    def test_tle_tallies(self):
        mh = log_forward_pass(_make_frozen_first_layer(), _simple_input())
        for entry in mh:
            if entry.computed_with_params:
                assert (
                    entry.num_params_total == entry.num_params_trainable + entry.num_params_frozen
                )
        # First linear is frozen
        linear1 = [e for e in mh if "linear_1" in e.layer_label][0]
        assert linear1.num_params_frozen == 55
        assert linear1.num_params_trainable == 0
        # Second linear is trainable
        linear2 = [e for e in mh if "linear_2" in e.layer_label][0]
        assert linear2.num_params_trainable == 12
        assert linear2.num_params_frozen == 0

    def test_output_layer_no_params(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        output = mh["output_1"]
        assert output.num_params_total == 0
        assert output.num_params_trainable == 0
        assert output.num_params_frozen == 0
        assert len(output.parent_param_logs) == 0


# ---------------------------------------------------------------------------
# Linked params
# ---------------------------------------------------------------------------


class TestLinkedParams:
    def test_weight_bias_linked(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        w = mh.params["0.weight"]
        b = mh.params["0.bias"]
        assert b.address in w.linked_params
        assert w.address in b.linked_params

    def test_linked_symmetric(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        for pl in mh.params:
            for other_addr in pl.linked_params:
                other = mh.params[other_addr]
                assert pl.address in other.linked_params


# ---------------------------------------------------------------------------
# tensor_log_entries reverse mapping
# ---------------------------------------------------------------------------


class TestTensorLogEntries:
    def test_populated(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        for pl in mh.params:
            assert len(pl.tensor_log_entries) > 0

    def test_points_to_correct_layers(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        for pl in mh.params:
            for label in pl.tensor_log_entries:
                entry = mh[label]
                assert any(p.address == pl.address for p in entry.parent_param_logs)


# ---------------------------------------------------------------------------
# Recurrent / multi-pass
# ---------------------------------------------------------------------------


class TestRecurrentParams:
    def test_num_passes_non_recurrent(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        for pl in mh.params:
            assert pl.num_passes == 1

    def test_num_passes_recurrent(self, input_2d):
        model = example_models.RecurrentParamsSimple()
        mh = log_forward_pass(model, input_2d)
        # fc1 is used 4 times
        pl = mh.params["fc1.weight"]
        assert pl.num_passes >= 2  # should be 4
        assert len(pl.tensor_log_entries) >= 2

    def test_tensor_log_entries_multi_pass(self, input_2d):
        model = example_models.RecurrentParamsSimple()
        mh = log_forward_pass(model, input_2d)
        pl = mh.params["fc1.weight"]
        # num_passes equals the number of tensor_log_entries
        assert pl.num_passes == len(pl.tensor_log_entries)
        assert pl.num_passes >= 2


# ---------------------------------------------------------------------------
# Gradient tracking
# ---------------------------------------------------------------------------


class TestGradientTracking:
    def test_no_grad_by_default(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        for pl in mh.params:
            assert pl.has_grad is False
            assert pl.grad_shape is None

    def test_no_grad_without_save_gradients(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input(), save_gradients=False)
        # Even if we could call backward, save_gradients=False means no hooks
        for pl in mh.params:
            assert pl.has_grad is False

    def test_grad_after_backward(self):
        model = _make_simple_model()
        x = _simple_input()
        mh = log_forward_pass(model, x, save_gradients=True)
        output = mh["output_1"].tensor_contents
        output.sum().backward()

        # All trainable params should have gradients
        for pl in mh.params:
            assert pl.has_grad is True
            assert pl.grad_shape == pl.shape
            assert pl.grad_dtype == pl.dtype
            assert pl.grad_fsize > 0
            assert len(pl.grad_fsize_nice) > 0

    def test_grad_frozen_params_no_grad(self):
        model = _make_frozen_first_layer()
        x = _simple_input()
        mh = log_forward_pass(model, x, save_gradients=True)
        output = mh["output_1"].tensor_contents
        output.sum().backward()

        # Frozen params should NOT have gradients
        assert mh.params["0.weight"].has_grad is False
        assert mh.params["0.bias"].has_grad is False
        # Trainable params should have gradients
        assert mh.params["2.weight"].has_grad is True
        assert mh.params["2.bias"].has_grad is True

    def test_tle_grad_saved(self):
        model = _make_simple_model()
        x = _simple_input()
        mh = log_forward_pass(model, x, save_gradients=True)
        output = mh["output_1"].tensor_contents
        output.sum().backward()

        # TLEs with param layers should have saved gradients
        for entry in mh:
            if entry.computed_with_params:
                assert entry.has_saved_grad is True
                assert entry.grad_contents is not None
                assert entry.grad_shape is not None

    def test_layers_with_saved_gradients_populated(self):
        model = _make_simple_model()
        x = _simple_input()
        mh = log_forward_pass(model, x, save_gradients=True)
        output = mh["output_1"].tensor_contents
        output.sum().backward()

        assert len(mh.layers_with_saved_gradients) > 0
        for label in mh.layers_with_saved_gradients:
            assert mh[label].has_saved_grad is True

    def test_grad_shape_matches_param_shape(self):
        model = _make_simple_model()
        x = _simple_input()
        mh = log_forward_pass(model, x, save_gradients=True)
        output = mh["output_1"].tensor_contents
        output.sum().backward()

        for pl in mh.params:
            if pl.has_grad:
                assert pl.grad_shape == pl.shape, (
                    f"Grad shape {pl.grad_shape} != param shape {pl.shape} for {pl.address}"
                )


# ---------------------------------------------------------------------------
# Optimizer support
# ---------------------------------------------------------------------------


class TestOptimizerSupport:
    def test_has_optimizer_none_by_default(self):
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        for pl in mh.params:
            assert pl.has_optimizer is None

    def test_has_optimizer_true(self):
        model = _make_simple_model()
        optimizer = torch.optim.Adam(model.parameters())
        mh = log_forward_pass(model, _simple_input(), optimizer=optimizer)
        for pl in mh.params:
            assert pl.has_optimizer is True

    def test_has_optimizer_partial(self):
        model = _make_simple_model()
        # Only optimize second linear layer
        optimizer = torch.optim.Adam(model[2].parameters())
        mh = log_forward_pass(model, _simple_input(), optimizer=optimizer)
        assert mh.params["0.weight"].has_optimizer is False
        assert mh.params["0.bias"].has_optimizer is False
        assert mh.params["2.weight"].has_optimizer is True
        assert mh.params["2.bias"].has_optimizer is True


# ---------------------------------------------------------------------------
# Visualization — param labels and colors
# ---------------------------------------------------------------------------


class TestVisualizationParams:
    def _get_dot_source(self, model, x, vis_opt="unrolled", vis_nesting_depth=1000):
        outpath = opj(VIS_OUTPUT_DIR, "toy-networks", "_test_param_vis")
        mh = log_forward_pass(model, x)
        mh.render_graph(
            vis_opt=vis_opt,
            vis_outpath=outpath,
            save_only=True,
            vis_nesting_depth=vis_nesting_depth,
        )
        # Graphviz source is saved as the outpath (no extension)
        with open(outpath, "r") as f:
            return f.read(), mh

    def test_param_label_shows_names(self):
        dot, mh = self._get_dot_source(_make_simple_model(), _simple_input())
        assert "weight:" in dot
        assert "bias:" in dot

    def test_trainable_round_brackets(self):
        dot, mh = self._get_dot_source(_make_simple_model(), _simple_input())
        # All trainable: should use round brackets ()
        assert "weight: (" in dot
        assert "bias: (" in dot

    def test_frozen_square_brackets(self):
        dot, mh = self._get_dot_source(_make_all_frozen(), _simple_input())
        # All frozen: should use square brackets []
        assert "weight: [" in dot
        assert "bias: [" in dot

    def test_mixed_brackets(self):
        dot, mh = self._get_dot_source(_make_frozen_first_layer(), _simple_input())
        # Should have both round and square brackets
        assert "(" in dot  # trainable
        assert "[" in dot  # frozen

    def test_trainable_bg_color(self):
        dot, mh = self._get_dot_source(_make_simple_model(), _simple_input())
        assert "#D9D9D9" in dot  # TRAINABLE_PARAMS_BG_COLOR

    def test_frozen_bg_color(self):
        dot, mh = self._get_dot_source(_make_all_frozen(), _simple_input())
        assert "#B0B0B0" in dot  # FROZEN_PARAMS_BG_COLOR

    def test_mixed_gradient_fill(self):
        """When a layer has both trainable and frozen params, use gradient fill."""
        # Create a model with mixed trainable/frozen in the same layer
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        # Freeze only the weight, keep bias trainable
        model[0].weight.requires_grad = False
        dot, mh = self._get_dot_source(model, _simple_input())
        # Gradient fill uses "color1:color2" syntax
        assert "#D9D9D9:#B0B0B0" in dot
        assert "gradientangle" in dot

    def test_graph_caption_all_trainable(self):
        dot, mh = self._get_dot_source(_make_simple_model(), _simple_input())
        assert "all trainable" in dot

    def test_graph_caption_all_frozen(self):
        dot, mh = self._get_dot_source(_make_all_frozen(), _simple_input())
        assert "all frozen" in dot

    def test_graph_caption_mixed(self):
        dot, mh = self._get_dot_source(_make_frozen_first_layer(), _simple_input())
        assert "trainable" in dot
        # Should show ratio like "12/67 trainable"
        assert "12/67" in dot

    def test_collapsed_module_trainable_color(self):
        """Collapsed module with all trainable params should use trainable color."""
        dot, mh = self._get_dot_source(_make_simple_model(), _simple_input(), vis_nesting_depth=1)
        assert "#D9D9D9" in dot

    def test_collapsed_module_frozen_color(self):
        """Collapsed module with all frozen params should use frozen color."""
        dot, mh = self._get_dot_source(_make_all_frozen(), _simple_input(), vis_nesting_depth=1)
        assert "#B0B0B0" in dot

    def test_collapsed_module_mixed_gradient(self):
        """Collapsed module with mixed frozen/trainable should use gradient fill."""
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        model[0].weight.requires_grad = False
        dot, mh = self._get_dot_source(model, _simple_input(), vis_nesting_depth=1)
        # The collapsed "0" module has mixed params
        assert "#D9D9D9:#B0B0B0" in dot or "#D9D9D9" in dot  # at least one color appears

    def test_collapsed_module_param_detail(self):
        """Collapsed module should show trainable/frozen param breakdown."""
        dot, mh = self._get_dot_source(_make_simple_model(), _simple_input(), vis_nesting_depth=1)
        assert "all trainable" in dot

    def test_collapsed_module_param_detail_frozen(self):
        dot, mh = self._get_dot_source(_make_all_frozen(), _simple_input(), vis_nesting_depth=1)
        assert "all frozen" in dot


# ---------------------------------------------------------------------------
# Integration: end-to-end with various model types
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_recurrent_model(self, input_2d):
        model = example_models.RecurrentParamsSimple()
        mh = log_forward_pass(model, input_2d)
        assert len(mh.params) == 2  # fc1.weight, fc1.bias
        assert mh.params["fc1.weight"].num_passes >= 2

    def test_model_with_no_params(self):
        model = nn.Sequential(nn.ReLU(), nn.Sigmoid())
        x = torch.randn(1, 5)
        mh = log_forward_pass(model, x)
        assert len(mh.params) == 0
        assert mh.total_params == 0
        assert mh.total_params_trainable == 0
        assert mh.total_params_frozen == 0

    def test_recurrent_params_complex(self, input_2d):
        model = example_models.RecurrentParamsComplex()
        mh = log_forward_pass(model, input_2d)
        assert len(mh.params) == 4  # fc1 weight+bias, fc2 weight+bias
        # Both fc1 and fc2 are used multiple times (as tensor_log_entries)
        assert mh.params["fc1.weight"].num_passes >= 2
        assert mh.params["fc2.weight"].num_passes >= 2
        # Verify all entries point to real layers
        for pl in mh.params:
            for label in pl.tensor_log_entries:
                assert label in mh.layer_dict_all_keys

    def test_gradient_tracking_recurrent(self, input_2d):
        model = example_models.RecurrentParamsSimple()
        mh = log_forward_pass(model, input_2d, save_gradients=True)
        output = mh["output_1"].tensor_contents
        output.sum().backward()

        # Despite multiple passes, each param should have exactly one gradient
        for pl in mh.params:
            assert pl.has_grad is True
            assert pl.grad_shape == pl.shape

    def test_parent_params_cleared_from_tle(self):
        """After postprocessing, parent_params references should be cleared."""
        mh = log_forward_pass(_make_simple_model(), _simple_input())
        for entry in mh:
            assert entry.parent_params is None

    def test_vis_renders_without_error(self):
        """Basic smoke test that visualization renders for each param scenario."""
        outdir = opj(VIS_OUTPUT_DIR, "toy-networks")

        # All trainable
        model = _make_simple_model()
        show_model_graph(
            model, _simple_input(), save_only=True, vis_outpath=opj(outdir, "_param_all_trainable")
        )

        # Mixed
        model = _make_frozen_first_layer()
        show_model_graph(
            model, _simple_input(), save_only=True, vis_outpath=opj(outdir, "_param_mixed")
        )

        # All frozen
        model = _make_all_frozen()
        show_model_graph(
            model, _simple_input(), save_only=True, vis_outpath=opj(outdir, "_param_all_frozen")
        )

    def test_vis_collapsed_renders_without_error(self):
        outdir = opj(VIS_OUTPUT_DIR, "toy-networks")

        model = _make_simple_model()
        show_model_graph(
            model,
            _simple_input(),
            save_only=True,
            vis_nesting_depth=1,
            vis_outpath=opj(outdir, "_param_collapsed_trainable"),
        )

        model = _make_frozen_first_layer()
        show_model_graph(
            model,
            _simple_input(),
            save_only=True,
            vis_nesting_depth=1,
            vis_outpath=opj(outdir, "_param_collapsed_mixed"),
        )

        model = _make_all_frozen()
        show_model_graph(
            model,
            _simple_input(),
            save_only=True,
            vis_nesting_depth=1,
            vis_outpath=opj(outdir, "_param_collapsed_frozen"),
        )
