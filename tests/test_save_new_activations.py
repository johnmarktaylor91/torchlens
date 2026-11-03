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
from torchlens._record_type import record_type
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


# Kept: sole branch coverage of _data_classes/layer.py, validation/_invariants_buffers.py, validation/_invariants_equivalence.py; a sole mutation kill.
def test_save_new_outs_metadata_preserved():
    """Metadata invariants hold after save_new_outs."""
    model = _SimpleFF()
    log = trace_fn(model, torch.randn(2, 5), capture=tl.options.CaptureOptions(seed=42))
    log._save_new_outs(model, torch.randn(2, 5), seed=42)

    assert check_metadata_invariants(log) is True
    log.cleanup()


# Kept: named by a manifest, the interaction matrix or the validation mutation suite.
def test_save_new_outs_recurrent():
    """save_new_outs works on recurrent models."""
    model = _RecurrentFF()
    log = trace_fn(model, torch.randn(2, 5), capture=tl.options.CaptureOptions(seed=42))
    assert log.graph.is_recurrent

    log._save_new_outs(model, torch.randn(2, 5), seed=42)
    assert log[log.output_layers[0]].has_saved_activation
    log.cleanup()


# Kept: sole branch coverage of _capture/_run_setup.py, _capture/finalize/settle.py, _capture/projectors.py and more.
def test_save_new_outs_fast_path_does_not_attach_streaming_refs() -> None:
    """The fast-path re-extraction flow should not create streaming bundle refs."""

    model = _SimpleFF()
    log = trace_fn(model, torch.randn(2, 5), capture=tl.options.CaptureOptions(seed=42))
    output_label = log.output_layers[0]

    log._save_new_outs(model, torch.randn(2, 5), seed=42, save=[output_label])

    assert getattr(log, "_out_writer", None) is None
    assert all(getattr(layer, "out_ref", None) is None for layer in log._layer_list)
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
    log = trace_fn(model, x1, capture=tl.options.CaptureOptions(seed=42))
    fresh_log = None
    try:
        log._save_new_outs(model, x2, seed=42)
        fresh_log = trace_fn(model, x2, capture=tl.options.CaptureOptions(seed=42))
        fresh_layers_by_label = {layer.layer_label: layer for layer in fresh_log._layer_list}

        compared_layers = 0
        for layer in log._layer_list:
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


class TestSaveNewActivationsStateReset:
    """Stale state in save_new_outs."""

    # Kept: sole branch coverage of _data_classes/_trace_groups.py.
    def test_timing_reset(self) -> None:
        """func_calls_duration should be fresh."""
        model = _SimpleLinear()
        log = trace_fn(model, torch.randn(2, 10), save="all")
        try:
            log._save_new_outs(model, torch.randn(2, 10), save="all")
            assert log.cost.func_calls_duration >= 0
        finally:
            log.cleanup()


class TestDescriptiveValueError:
    """log_source_tensor_fast should give descriptive error on graph change."""

    # Kept: sole branch coverage of _capture/projectors.py.
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
        # The capture restores ``call_count`` (pure observer); an ordinary forward
        # advances it, so the refresh takes the other branch.
        assert model.call_count == 0
        model(torch.randn(2, 10))
        with pytest.raises(ValueError, match="computational graph changed"):
            log._save_new_outs(model, torch.randn(2, 10))


class TestGraphConsistencyValidation:
    """log_source_tensor_fast warns on shape mismatch."""

    # Kept: sole branch coverage of _capture/projectors.py.
    def test_shape_mismatch_warns(self):
        model = _SimpleLinear()
        log = trace_fn(model, torch.randn(2, 10))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log._save_new_outs(model, torch.randn(4, 10))
            shape_warnings = [x for x in w if "shape changed" in str(x.message)]
            assert len(shape_warnings) > 0

    # Kept: sole branch coverage of _capture/projectors.py.
    def test_save_new_outs_rejects_operand_order_drift(self) -> None:
        """save_new_outs must refuse reruns whose operand routing changed."""

        model = _OperandOrderSwapModel()
        x = torch.randn(2, 3)
        log = trace_fn(model, x, capture=tl.options.CaptureOptions(save_args=True))
        try:
            model.reverse = True
            with pytest.raises(ValueError, match="computational graph changed") as exc_info:
                log._save_new_outs(model, x)
            assert "parent_arg_positions" in str(exc_info.value)
        finally:
            log.cleanup()

    # Kept: sole branch coverage of _capture/finalize/settle.py.
    def test_save_new_outs_accepts_same_graph_operand_model(self) -> None:
        """save_new_outs must still refresh same-graph reruns for the same model."""

        model = _OperandOrderSwapModel()
        x1 = torch.randn(2, 3)
        x2 = torch.randn(2, 3)
        log = trace_fn(model, x1, capture=tl.options.CaptureOptions(save_args=True))
        fresh_log = None
        try:
            log._save_new_outs(model, x2)
            fresh_log = trace_fn(model, x2, capture=tl.options.CaptureOptions(save_args=True))
            refreshed_sub = next(op for op in log._layer_list if op.type == "sub")
            fresh_sub = next(op for op in fresh_log._layer_list if op.type == "sub")
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
# ``buffer_sink_routing_unsupported`` code and the pinned "computational graph
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
        log._layer_dict_all_keys[label]
        for label in log.internal_sink_ops
        if record_type(log._layer_dict_all_keys[label]) == "buffer"
    ]


def _assert_buffer_sink_refusal(exc_info):
    """Assert one refusal is the typed D18 arm with the pinned term and code."""

    from torchlens._runnable import RunnableErrorCode

    assert "computational graph changed" in str(exc_info.value)
    assert exc_info.value.fields["code"] == RunnableErrorCode.BUFFER_SINK_ROUTING_MUTABLE.value


def _tamper_recorded_training_literal(log, sink, value):
    """Flip the recorded literal mode argument on one sink's producing op."""

    source = log._layer_dict_all_keys[sink.buffer_source]
    args = list(source.non_tensor_pos_args)
    index = next(i for i, arg in enumerate(args) if isinstance(arg, bool))
    args[index] = value
    source.non_tensor_pos_args = args


# Kept: sole branch coverage of _capture/projectors.py.
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
            log._save_new_outs(model, torch.randn(3, 4))
        _assert_buffer_sink_refusal(exc_info)
        assert "mode claim" in str(exc_info.value) or "claims" in str(exc_info.value)
    finally:
        log.cleanup()


# Kept: sole branch coverage of _capture/projectors.py.
def test_mode_flip_between_runs_refuses():
    """O1 live tripwire: eval capture, model.train() before refresh, typed."""

    from torchlens.errors import BufferSinkRoutingError

    model = _BatchNormModel()
    model.eval()
    log = trace_fn(model, torch.randn(3, 4))
    try:
        model.train()
        with pytest.raises(BufferSinkRoutingError) as exc_info:
            log._save_new_outs(model, torch.randn(3, 4))
        _assert_buffer_sink_refusal(exc_info)
    finally:
        log.cleanup()


# Kept: sole branch coverage of _capture/projectors.py, _data_classes/_trace_rerun_state.py, intervention/_trace_run.py.
def test_state_restore_admits_only_proven_train_mode_norm_writers():
    """5.2 under the pure-observer default: run() admits a proven train-mode BatchNorm, nothing looser.

    The default run() restores the declared state it changed and the capture is a
    pure observer, so a train-mode BatchNorm (proven norm writes plus its counter)
    runs and leaves its statistics as they were. A lone counter buffer is not a
    norm writer and still refuses on the same path.
    """

    from torchlens.errors import BufferSinkRoutingError

    model = _BatchNormModel()
    model.train()
    log = trace_fn(model, torch.randn(3, 4))
    try:
        before = {name: buf.clone() for name, buf in model.named_buffers()}
        log.run(inputs=torch.randn(3, 4))
        for name, buf in model.named_buffers():
            assert torch.equal(before[name], buf), name
    finally:
        log.cleanup()
    counter = _BufferCounterModel()
    log = trace_fn(counter, torch.randn(3, 4))
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
