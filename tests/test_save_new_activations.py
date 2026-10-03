"""Tests for save_new_outs — the fast-path out re-extraction.

Covers: successful re-extraction on simple models, multiple sequential calls,
out value correctness, and the known failure mode on models with
identity-propagated operations.
"""

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens import trace as trace_fn
from torchlens.postprocess.graph_traversal import output_payload_aliases_parent
from torchlens.validation import check_metadata_invariants

# =============================================================================
# Test models
# =============================================================================


class _SimpleFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class _RecurrentFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        for _ in range(3):
            x = self.relu(self.fc(x))
        return x


class _BranchingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(h) + h.sum()


# =============================================================================
# Positive tests — save_new_outs works correctly
# =============================================================================


def test_save_new_outs_basic():
    """save_new_outs replaces outs on a simple model."""
    model = _SimpleFF()
    x1 = torch.randn(2, 5)
    log = trace_fn(model, x1, capture=tl.options.CaptureOptions(random_seed=42))

    x2 = torch.randn(2, 5)
    log.save_new_outs(model, x2, random_seed=42)

    # Verify output matches direct model execution
    model.eval()
    with torch.no_grad():
        expected = model(x2)
    actual = log[log.output_layers[0]].out
    assert torch.allclose(actual, expected, atol=1e-5)
    log.cleanup()


def test_save_new_outs_multiple_calls():
    """save_new_outs works correctly on repeated calls."""
    model = _SimpleFF()
    log = trace_fn(model, torch.randn(2, 5), capture=tl.options.CaptureOptions(random_seed=42))

    for _i in range(5):
        x = torch.randn(2, 5)
        log.save_new_outs(model, x, random_seed=42)

    # Still valid after 5 calls
    assert len(log.layer_list) > 0
    assert log[log.output_layers[0]].has_saved_activation
    log.cleanup()


def test_save_new_outs_outs_change():
    """Activations actually change when new input is provided."""
    model = _SimpleFF()
    torch.manual_seed(0)
    x1 = torch.randn(2, 5)
    log = trace_fn(model, x1, capture=tl.options.CaptureOptions(random_seed=42))
    act1 = log[log.output_layers[0]].out.clone()

    torch.manual_seed(99)
    x2 = torch.randn(2, 5)
    log.save_new_outs(model, x2, random_seed=42)
    act2 = log[log.output_layers[0]].out.clone()

    assert not torch.equal(act1, act2), "Activations should differ for different inputs"
    log.cleanup()


def test_save_new_outs_metadata_preserved():
    """Metadata invariants hold after save_new_outs."""
    model = _SimpleFF()
    log = trace_fn(model, torch.randn(2, 5), capture=tl.options.CaptureOptions(random_seed=42))
    log.save_new_outs(model, torch.randn(2, 5), random_seed=42)

    assert check_metadata_invariants(log) is True
    log.cleanup()


def test_save_new_outs_recurrent():
    """save_new_outs works on recurrent models."""
    model = _RecurrentFF()
    log = trace_fn(model, torch.randn(2, 5), capture=tl.options.CaptureOptions(random_seed=42))
    assert log.is_recurrent

    log.save_new_outs(model, torch.randn(2, 5), random_seed=42)
    assert log[log.output_layers[0]].has_saved_activation
    log.cleanup()


def test_save_new_outs_branching():
    """save_new_outs works on branching models."""
    model = _BranchingModel()
    log = trace_fn(model, torch.randn(2, 5), capture=tl.options.CaptureOptions(random_seed=42))

    log.save_new_outs(model, torch.randn(2, 5), random_seed=42)
    assert log[log.output_layers[0]].has_saved_activation
    log.cleanup()


def test_save_new_outs_layers_to_save():
    """save_new_outs respects layers_to_save parameter."""
    model = _SimpleFF()
    log = trace_fn(model, torch.randn(2, 5), capture=tl.options.CaptureOptions(random_seed=42))

    # Only save the first layer
    first_label = log.layer_labels[0]
    log.save_new_outs(model, torch.randn(2, 5), random_seed=42, layers_to_save=[first_label])

    assert log[first_label].has_saved_activation
    log.cleanup()


def test_save_new_outs_fast_path_does_not_attach_streaming_refs() -> None:
    """The fast-path re-extraction flow should not create streaming bundle refs."""

    model = _SimpleFF()
    log = trace_fn(model, torch.randn(2, 5), capture=tl.options.CaptureOptions(random_seed=42))
    output_label = log.output_layers[0]

    log.save_new_outs(model, torch.randn(2, 5), random_seed=42, layers_to_save=[output_label])

    assert getattr(log, "_out_writer", None) is None
    assert all(getattr(layer, "out_ref", None) is None for layer in log.layer_list)
    log.cleanup()


# =============================================================================
# Torchvision models with identity-propagated ops
# =============================================================================


def _assert_save_new_outs_matches_fresh_log(
    model: nn.Module, x1: torch.Tensor, x2: torch.Tensor
) -> None:
    """Verify fast out refresh matches a fresh exhaustive log.

    Parameters
    ----------
    model:
        Model to log and refresh.
    x1:
        Initial input for the exhaustive log.
    x2:
        Replacement input for ``save_new_outs`` and the fresh log.
    """
    log = trace_fn(model, x1, capture=tl.options.CaptureOptions(random_seed=42))
    fresh_log = None
    try:
        log.save_new_outs(model, x2, random_seed=42)
        fresh_log = trace_fn(model, x2, capture=tl.options.CaptureOptions(random_seed=42))
        fresh_layers_by_label = {layer.layer_label: layer for layer in fresh_log.layer_list}

        compared_layers = 0
        for layer in log.layer_list:
            if layer.out is None:
                continue
            assert layer.layer_label in fresh_layers_by_label
            fresh_out = fresh_layers_by_label[layer.layer_label].out
            assert fresh_out is not None
            assert layer.out.shape == fresh_out.shape
            assert layer.out.dtype == fresh_out.dtype
            assert torch.allclose(layer.out, fresh_out, rtol=1e-4, atol=1e-5)
            compared_layers += 1
        assert compared_layers > 0
    finally:
        log.cleanup()
        if fresh_log is not None:
            fresh_log.cleanup()


@pytest.mark.slow
def test_save_new_outs_alexnet_matches_fresh_log() -> None:
    """AlexNet fast out refresh matches a fresh exhaustive log."""
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.alexnet(weights=None)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    _assert_save_new_outs_matches_fresh_log(model, x, torch.randn(1, 3, 224, 224))


@pytest.mark.slow
def test_save_new_outs_resnet_rejects_buffer_sink_refresh() -> None:
    """ResNet18 buffer-sink refresh: eval refreshes (D18), train refuses typed.

    Rewritten IN the D18 merge per the honesty rule: the historical eval-mode
    refusal was the mode-BLIND condition (sink presence, not write evidence)
    and is the capability D18 unlocks; the train-mode refusal is pinned forever.
    """
    torchvision = pytest.importorskip("torchvision")
    from torchlens.errors import BufferSinkRoutingError

    model = torchvision.models.resnet18(weights=None)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    log = trace_fn(model, x, capture=tl.options.CaptureOptions(random_seed=42))
    try:
        assert any(
            log.layer_dict_all_keys[label].layer_type == "buffer" for label in log.internal_sink_ops
        )
        x2 = torch.randn(1, 3, 224, 224)
        log.save_new_outs(model, x2, random_seed=42)
        fresh = trace_fn(model, x2, capture=tl.options.CaptureOptions(random_seed=42))
        try:
            fresh_by_label = {layer.layer_label: layer for layer in fresh.layer_list}
            compared = 0
            for layer in log.layer_list:
                if layer.layer_type != "batchnorm" or layer.out is None:
                    continue
                fresh_out = fresh_by_label[layer.layer_label].out
                assert torch.allclose(layer.out, fresh_out, rtol=1e-4, atol=1e-5)
                compared += 1
            assert compared > 0
        finally:
            fresh.cleanup()
    finally:
        log.cleanup()

    train_model = torchvision.models.resnet18(weights=None)
    train_model.train()
    train_log = trace_fn(train_model, x, capture=tl.options.CaptureOptions(random_seed=42))
    try:
        with pytest.raises(BufferSinkRoutingError, match="computational graph changed"):
            train_log.save_new_outs(train_model, torch.randn(1, 3, 224, 224), random_seed=42)
    finally:
        train_log.cleanup()


# =============================================================================
# Bugfix regression tests
# =============================================================================


class _SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class _SharedBufferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor([2.0]))
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        x = x * self.scale
        x = self.fc(x)
        x = x * self.scale
        return x


class _OperandOrderSwapModel(nn.Module):
    """Model whose subtraction operand order can drift across reruns."""

    def __init__(self) -> None:
        """Initialize the operand-order toggle."""

        super().__init__()
        self.reverse = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subtract the two branches in the configured operand order."""

        add_branch = x + 1
        mul_branch = x * 2
        if self.reverse:
            return mul_branch - add_branch
        return add_branch - mul_branch


class TestSaveNewActivationsRegression:
    """Zombie OpLogs on repeated calls."""

    def test_save_new_outs_3x(self) -> None:
        """Three default-selector refreshes should keep outputs aligned with the model."""
        model = _SimpleLinear()
        x = torch.randn(2, 10)
        log = trace_fn(model, x)
        try:
            for _ in range(3):
                next_x = torch.randn(2, 10)
                log.save_new_outs(model, next_x)
                expected = model(next_x)
                output = log[log.output_layers[0]].out
                assert output is not None
                assert log.num_saved_ops > 0
                assert torch.allclose(output, expected)
        finally:
            log.cleanup()

    def test_save_new_outs_different_values(self) -> None:
        """Activations should change with new inputs."""
        model = _SimpleLinear()
        x1 = torch.randn(2, 10)
        log = trace_fn(model, x1)
        try:
            first_output = log[log.output_layers[0]].out.clone()
            x2 = torch.randn(2, 10) + 10
            log.save_new_outs(model, x2)
            second_output = log[log.output_layers[0]].out
            assert not torch.equal(first_output, second_output)
        finally:
            log.cleanup()


class TestSaveNewActivationsStateReset:
    """Stale state in save_new_outs."""

    def test_timing_reset(self) -> None:
        """func_calls_duration should be fresh."""
        model = _SimpleLinear()
        log = trace_fn(
            model, torch.randn(2, 10), capture=tl.options.CaptureOptions(layers_to_save="all")
        )
        try:
            log.save_new_outs(model, torch.randn(2, 10), layers_to_save="all")
            assert log.func_calls_duration >= 0
        finally:
            log.cleanup()

    def test_lookup_keys_clean(self) -> None:
        """Lookup caches should not have stale entries."""
        model = _SimpleLinear()
        log = trace_fn(
            model, torch.randn(2, 10), capture=tl.options.CaptureOptions(layers_to_save="all")
        )
        try:
            labels_pass1 = set(log.layer_labels)
            log.save_new_outs(model, torch.randn(2, 10), layers_to_save="all")
            labels_pass2 = set(log.layer_labels)
            assert labels_pass1 == labels_pass2
        finally:
            log.cleanup()

    @pytest.mark.parametrize(
        "trace_kwargs",
        [{}, {"layers_to_save": "all"}],
        ids=["default_selector", "explicit_all"],
    )
    def test_5x_stress(self, trace_kwargs: dict[str, str]) -> None:
        """Stress test: repeated refreshes stay aligned for default and explicit-all saves."""
        model = _SimpleLinear()
        capture_kwargs = (
            {"capture": tl.options.CaptureOptions(**trace_kwargs)} if trace_kwargs else {}
        )
        log = trace_fn(model, torch.randn(2, 10), **capture_kwargs)
        try:
            for _ in range(5):
                next_x = torch.randn(2, 10)
                log.save_new_outs(model, next_x, **trace_kwargs)
                output = log[log.output_layers[0]].out
                assert output is not None
                assert log.num_saved_ops > 0
                assert torch.allclose(output, model(next_x))
        finally:
            log.cleanup()

    def test_different_values(self) -> None:
        """Each pass should reflect new input values."""
        model = _SimpleLinear()
        log = trace_fn(
            model, torch.ones(2, 10), capture=tl.options.CaptureOptions(layers_to_save="all")
        )
        try:
            input_val_1 = log["input_1"].out.clone()
            log.save_new_outs(model, torch.zeros(2, 10), layers_to_save="all")
            input_val_2 = log["input_1"].out
            assert not torch.equal(input_val_1, input_val_2)
        finally:
            log.cleanup()


class TestOutputTensorIndependence:
    """Fast-mode out shared reference.

    HONESTY 13-R1 (``postprocess.graph_traversal.output_payload_aliases_parent``,
    mirrored for the fast refresh path in
    ``capture.projectors._separate_output_payloads``) deliberately has the output
    pseudo-row RIDE its producer's one retained payload when the refreshed value IS
    that payload bit-for-bit (same shape/dtype/device, NaN-equal) -- a documented
    memory-savings contract, not a bug. Independence is the guarantee only for the
    OTHER case: a differing value always keeps its own physical copy. This test
    checks that real guarantee instead of assuming every pair is independent
    regardless of the contract (the historical bug class it exists to catch is an
    UNDOCUMENTED alias outside that contract, e.g. a differing value that still
    shares storage).
    """

    def test_output_independent_of_parent(self) -> None:
        model = _SimpleLinear()
        x = torch.randn(2, 10)
        log = trace_fn(model, x)
        try:
            log.save_new_outs(model, torch.randn(2, 10))
            checked = False
            for label in log.output_layers:
                output_entry = log[label]
                if output_entry.parents and output_entry.out is not None:
                    parent_label = output_entry.parents[0]
                    parent_entry = log[parent_label]
                    if parent_entry.out is not None:
                        checked = True
                        aliases_parent = output_payload_aliases_parent(
                            output_entry.out, parent_entry.out
                        )
                        if aliases_parent:
                            # The documented HONESTY 13-R1 case: one payload by
                            # design, so a mutation is visible on both views.
                            assert output_entry.out.data_ptr() == parent_entry.out.data_ptr()
                        else:
                            original_parent = parent_entry.out.clone()
                            output_entry.out.fill_(999)
                            assert torch.equal(parent_entry.out, original_parent)
                        break
            assert checked, "no output/parent pair with saved outs to exercise"
        finally:
            log.cleanup()


class TestFastPathModuleLogs:
    """Refresh capture should preserve module logs from the exhaustive pass."""

    def test_fast_path_preserves_module_logs(self) -> None:
        model = _SimpleLinear()
        x = torch.randn(2, 10)
        log = trace_fn(model, x)
        try:
            original_module_count = len(log.modules)
            original_addresses = [m.address for m in log.modules]
            assert original_module_count > 0
            log.save_new_outs(model, torch.randn(2, 10))
            assert len(log.modules) == original_module_count
            assert [m.address for m in log.modules] == original_addresses
        finally:
            log.cleanup()


class TestDescriptiveValueError:
    """log_source_tensor_fast should give descriptive error on graph change."""

    def test_dynamic_graph_descriptive_error(self):
        class DynamicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 10)
                self.linear2 = nn.Linear(10, 10)
                self.call_count = 0

            def forward(self, x):
                self.call_count += 1
                x = self.linear1(x)
                if self.call_count > 1:
                    x = self.linear2(x)
                    x = torch.relu(x)
                    x = self.linear2(x)
                return x

        model = DynamicModel()
        log = trace_fn(model, torch.randn(2, 10))
        with pytest.raises(ValueError, match="computational graph changed"):
            log.save_new_outs(model, torch.randn(2, 10))


class TestFastPassBufferOrphan:
    """Fast-pass should not KeyError on models with shared buffers."""

    def test_shared_buffer_fast_path(self):
        model = _SharedBufferModel()
        log = trace_fn(model, torch.randn(2, 10), capture=tl.options.CaptureOptions(random_seed=42))
        # Should not raise KeyError on fast pass
        log.save_new_outs(model, torch.randn(2, 10), random_seed=42)
        assert log[log.output_layers[0]].has_saved_activation
        log.cleanup()

    def test_shared_buffer_fast_path_3x(self):
        model = _SharedBufferModel()
        log = trace_fn(model, torch.randn(2, 10), capture=tl.options.CaptureOptions(random_seed=42))
        for _ in range(3):
            log.save_new_outs(model, torch.randn(2, 10), random_seed=42)
        assert log[log.output_layers[0]].has_saved_activation
        log.cleanup()


class TestGraphConsistencyValidation:
    """log_source_tensor_fast warns on shape mismatch."""

    def test_shape_mismatch_warns(self):
        model = _SimpleLinear()
        log = trace_fn(model, torch.randn(2, 10))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log.save_new_outs(model, torch.randn(4, 10))
            shape_warnings = [x for x in w if "shape changed" in str(x.message)]
            assert len(shape_warnings) > 0

    def test_save_new_outs_rejects_operand_order_drift(self) -> None:
        """save_new_outs must refuse reruns whose operand routing changed."""

        model = _OperandOrderSwapModel()
        x = torch.randn(2, 3)
        log = trace_fn(model, x, capture=tl.options.CaptureOptions(save_arg_values=True))
        try:
            model.reverse = True
            with pytest.raises(ValueError, match="computational graph changed") as exc_info:
                log.save_new_outs(model, x)
            assert "parent_arg_positions" in str(exc_info.value)
        finally:
            log.cleanup()

    def test_save_new_outs_accepts_same_graph_operand_model(self) -> None:
        """save_new_outs must still refresh same-graph reruns for the same model."""

        model = _OperandOrderSwapModel()
        x1 = torch.randn(2, 3)
        x2 = torch.randn(2, 3)
        log = trace_fn(model, x1, capture=tl.options.CaptureOptions(save_arg_values=True))
        fresh_log = None
        try:
            log.save_new_outs(model, x2)
            fresh_log = trace_fn(model, x2, capture=tl.options.CaptureOptions(save_arg_values=True))
            refreshed_sub = next(op for op in log.layer_list if op.layer_type == "sub")
            fresh_sub = next(op for op in fresh_log.layer_list if op.layer_type == "sub")
            assert refreshed_sub.parents == fresh_sub.parents
            assert refreshed_sub.parent_arg_positions == fresh_sub.parent_arg_positions
            assert refreshed_sub.out is not None
            assert fresh_sub.out is not None
            assert torch.allclose(refreshed_sub.out, fresh_sub.out)
        finally:
            log.cleanup()
            if fresh_log is not None:
                fresh_log.cleanup()


# =============================================================================
# D18: mode-aware buffer-sink projector (decided capability narrowing)
#
# The refusal decision is CLOSED-FORM: refuse iff any buffer sink carries
# ``buffer_value_changed is not False``, with the recorded mode claims as a
# contradiction belt, the refreshed rerun's own journal as the O1 write
# tripwire, and target-vs-refreshed evidence equality as the O2 widening.
# Every refusal is the TYPED BufferSinkRoutingError carrying the frozen
# ``buffer_sink_routing_mutable`` code and the pinned "computational graph
# changed" message term.
#
# O3 (honesty rule): the train-mode refusal tests below are PINNED FOREVER --
# the D18 narrowing is closed-vocabulary, not a beachhead. They must never be
# deleted or weakened.
# =============================================================================


class _BatchNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(self.bn(x))


class _BufferCounterModel(nn.Module):
    """Counter-only buffer writer: one inplace add_ on a registered buffer."""

    def __init__(self):
        super().__init__()
        self.register_buffer("step", torch.zeros(1))
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        self.step.add_(1)
        return self.lin(x)


def _buffer_sinks(log):
    """Return the buffer-typed internal sink layers of one trace."""

    return [
        log.layer_dict_all_keys[label]
        for label in log.internal_sink_ops
        if log.layer_dict_all_keys[label].layer_type == "buffer"
    ]


def _assert_buffer_sink_refusal(exc_info):
    """Assert one refusal is the typed D18 arm with the pinned term and code."""

    from torchlens.runnable import RunnableErrorCode

    assert "computational graph changed" in str(exc_info.value)
    assert exc_info.value.fields["code"] == RunnableErrorCode.BUFFER_SINK_ROUTING_MUTABLE.value


def _tamper_recorded_training_literal(log, sink, value):
    """Flip the recorded literal mode argument on one sink's producing op."""

    source = log.layer_dict_all_keys[sink.buffer_source]
    args = list(source.non_tensor_pos_args)
    index = next(i for i, arg in enumerate(args) if isinstance(arg, bool))
    args[index] = value
    source.non_tensor_pos_args = args


def test_eval_mode_batchnorm_refresh_allowed_default_path():
    """D18 NEW capability: eval-mode BatchNorm is refresh-eligible by default."""

    model = _BatchNormModel()
    model.eval()
    x = torch.randn(3, 4)
    log = trace_fn(model, x, capture=tl.options.CaptureOptions(random_seed=1))
    try:
        sinks = _buffer_sinks(log)
        assert sinks and all(sink.buffer_value_changed is False for sink in sinks)
        x2 = torch.randn(3, 4)
        log.save_new_outs(model, x2, random_seed=1)
        fresh = trace_fn(model, x2, capture=tl.options.CaptureOptions(random_seed=1))
        try:
            refreshed_bn = next(op for op in log.layer_list if op.layer_type == "batchnorm")
            fresh_bn = next(op for op in fresh.layer_list if op.layer_type == "batchnorm")
            assert torch.allclose(refreshed_bn.out, fresh_bn.out)
        finally:
            fresh.cleanup()
    finally:
        log.cleanup()


def test_eval_mode_batchnorm_run_default_path_verified():
    """D18 end-to-end: run() on an eval-BN model settles VERIFIED by default."""

    from torchlens.runnable import PathFaithfulness

    model = _BatchNormModel()
    model.eval()
    x = torch.randn(3, 4)
    log = trace_fn(model, x)
    try:
        x2 = torch.randn(3, 4)
        result = log.run(inputs=x2)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
        with torch.no_grad():
            expected = model(x2)
        assert torch.allclose(result.output, expected)
    finally:
        log.cleanup()


def test_train_mode_batchnorm_refresh_still_refuses():
    """O3 PINNED FOREVER: train-mode BatchNorm refresh refuses, typed."""

    from torchlens.errors import BufferSinkRoutingError

    model = _BatchNormModel()
    model.train()
    log = trace_fn(model, torch.randn(3, 4))
    try:
        assert any(sink.buffer_value_changed is True for sink in _buffer_sinks(log))
        with pytest.raises(BufferSinkRoutingError) as exc_info:
            log.save_new_outs(model, torch.randn(3, 4))
        _assert_buffer_sink_refusal(exc_info)
    finally:
        log.cleanup()


def test_train_mode_counter_only_still_refuses():
    """O3: a lone inplace buffer counter (num_batches_tracked shape) refuses."""

    from torchlens.errors import BufferSinkRoutingError

    model = _BufferCounterModel()
    log = trace_fn(model, torch.randn(3, 4))
    try:
        sinks = _buffer_sinks(log)
        assert sinks and all(sink.buffer_write_kind == "inplace" for sink in sinks)
        with pytest.raises(BufferSinkRoutingError) as exc_info:
            log.save_new_outs(model, torch.randn(3, 4))
        _assert_buffer_sink_refusal(exc_info)
    finally:
        log.cleanup()


def test_tampered_buffer_value_changed_claim_refuses_typed():
    """Tamper: a stored write-evidence bit flipped to False cannot buy a pass.

    The recorded mode claims (train) contradict the tampered evidence (False),
    so the belt refuses typed even though the primary key alone would pass.
    The model is switched to eval before the refresh so ONLY the belt (not the
    O1 fresh-journal tripwire) can catch the tamper.
    """

    from torchlens.errors import BufferSinkRoutingError

    model = _BatchNormModel()
    model.train()
    log = trace_fn(model, torch.randn(3, 4))
    try:
        for sink in _buffer_sinks(log):
            sink._internal_set("buffer_value_changed", False)
        model.eval()
        with pytest.raises(BufferSinkRoutingError) as exc_info:
            log.save_new_outs(model, torch.randn(3, 4))
        _assert_buffer_sink_refusal(exc_info)
        assert "mode claim" in str(exc_info.value) or "claims" in str(exc_info.value)
    finally:
        log.cleanup()


def test_train_claim_with_unchanged_values_refuses():
    """7.2 belt, other direction: a train claim over unchanged values refuses."""

    from torchlens.errors import BufferSinkRoutingError

    model = _BatchNormModel()
    model.eval()
    log = trace_fn(model, torch.randn(3, 4))
    try:
        sinks = _buffer_sinks(log)
        assert all(sink.buffer_value_changed is False for sink in sinks)
        bn_sink = next(sink for sink in sinks if sink.buffer_source_func_name == "batch_norm")
        _tamper_recorded_training_literal(log, bn_sink, True)
        with pytest.raises(BufferSinkRoutingError) as exc_info:
            log.save_new_outs(model, torch.randn(3, 4))
        _assert_buffer_sink_refusal(exc_info)
    finally:
        log.cleanup()


def test_tampered_training_literal_claim_refuses_typed():
    """Direct tamper of the recorded literal training argument refuses typed."""

    from torchlens.errors import BufferSinkRoutingError

    model = _BatchNormModel()
    model.train()
    log = trace_fn(model, torch.randn(3, 4))
    try:
        bn_sink = next(
            sink for sink in _buffer_sinks(log) if sink.buffer_source_func_name == "batch_norm"
        )
        _tamper_recorded_training_literal(log, bn_sink, False)
        with pytest.raises(BufferSinkRoutingError) as exc_info:
            log.save_new_outs(model, torch.randn(3, 4))
        _assert_buffer_sink_refusal(exc_info)
    finally:
        log.cleanup()


def test_tampered_module_mode_record_refuses_typed():
    """Direct tamper of the module_training_modes record refuses typed."""

    from torchlens.errors import BufferSinkRoutingError

    model = _BatchNormModel()
    model.train()
    log = trace_fn(model, torch.randn(3, 4))
    try:
        modes = log._runnable.module_training_modes
        assert modes.get("bn") is True
        modes["bn"] = False
        with pytest.raises(BufferSinkRoutingError) as exc_info:
            log.save_new_outs(model, torch.randn(3, 4))
        _assert_buffer_sink_refusal(exc_info)
    finally:
        log.cleanup()


def test_mode_flip_between_runs_refuses():
    """O1 live tripwire: eval capture, model.train() before refresh, typed."""

    from torchlens.errors import BufferSinkRoutingError

    model = _BatchNormModel()
    model.eval()
    log = trace_fn(model, torch.randn(3, 4))
    try:
        model.train()
        with pytest.raises(BufferSinkRoutingError) as exc_info:
            log.save_new_outs(model, torch.randn(3, 4))
        _assert_buffer_sink_refusal(exc_info)
    finally:
        log.cleanup()


def test_buffer_value_changed_none_fails_closed():
    """Unproven write evidence (None) refuses typed -- never widens past evidence."""

    from torchlens.errors import BufferSinkRoutingError

    model = _BatchNormModel()
    model.eval()
    log = trace_fn(model, torch.randn(3, 4))
    try:
        for sink in _buffer_sinks(log):
            sink._internal_set("buffer_value_changed", None)
        with pytest.raises(BufferSinkRoutingError) as exc_info:
            log.save_new_outs(model, torch.randn(3, 4))
        _assert_buffer_sink_refusal(exc_info)
        assert "unproven" in str(exc_info.value)
    finally:
        log.cleanup()


def test_state_restore_does_not_loosen_projector():
    """5.2 red-stays-red: snapshot-restore never loosens the buffer-sink refusal.

    Snapshot-restore fixes VALUES; the buffer-sink refusal is about ROUTING
    analysis and keeps its own authority. A train-mode BatchNorm (genuine
    value-changing buffer writes) still refuses on the default run() path
    after the restore bracket ships.
    """

    from torchlens.errors import BufferSinkRoutingError

    model = _BatchNormModel()
    model.train()
    log = trace_fn(model, torch.randn(3, 4))
    try:
        with pytest.raises(BufferSinkRoutingError) as exc_info:
            log.run(inputs=torch.randn(3, 4))
        _assert_buffer_sink_refusal(exc_info)
    finally:
        log.cleanup()


class _RoutingFlipModel(nn.Module):
    """Halves its weights each forward; routing flips once they decay enough."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.threshold = float(self.lin.weight.abs().sum()) * 0.375

    def forward(self, x):
        h = self.lin(x)
        if float(self.lin.weight.abs().sum()) > self.threshold:
            out = torch.relu(h)
        else:
            out = torch.tanh(h)
        with torch.no_grad():
            self.lin.weight.mul_(0.5)
        return out


def test_carry_state_then_routing_change_still_refuses():
    """5.3 red-stays-red: carried state that changes routing refuses next run.

    carry_state never touches verification: the NEXT run() from mutated state
    faces every gate as usual, so a mutation-driven routing change trips the
    graph-change tripwire with the pinned term.
    """

    model = _RoutingFlipModel()
    log = trace_fn(model, torch.randn(2, 4))
    try:
        first = log.run(inputs=torch.randn(2, 4), carry_state=True)
        assert first.report.state_carried is True
        with pytest.raises(ValueError, match="computational graph changed"):
            log.run(inputs=torch.randn(2, 4), carry_state=True)
    finally:
        log.cleanup()
