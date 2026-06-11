"""Static buffers returned from ``forward()`` must yield real output layers.

Regression coverage for the buffer-sole-output bug (raised 2026-06-05): a model
whose ONLY output is a static registered buffer (``def forward(self, x): return
self.buf``) used to capture no output layer at all -- ``tl.trace`` "succeeded"
with an empty ``output_layers`` list and ``validate_forward_pass`` raised
``MetadataInvariantError: No output layers found``.

The fix lives in postprocess Step 1 (``graph_traversal._add_output_layers``):
an output tensor that is a registered buffer never touched by any traced op is
logged as a late buffer source node, and the synthetic ``output_N`` node binds
to its value. These tests pin the fixed behavior plus the two no-regression
cases (buffer used in compute; buffer returned alongside computed outputs).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens import trace as trace_fn
from torchlens.validation.invariants import check_metadata_invariants


class BufferOnlyOutputModel(nn.Module):
    """Model whose sole output is a static registered buffer."""

    def __init__(self) -> None:
        """Register the static buffer."""

        super().__init__()
        self.register_buffer("buf", torch.arange(3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the untouched buffer, ignoring the input.

        Parameters
        ----------
        x:
            Ignored input tensor.

        Returns
        -------
        torch.Tensor
            The registered buffer itself.
        """

        return self.buf


class BufferInComputeModel(nn.Module):
    """Model that reads its buffer inside the computation (no-regression case)."""

    def __init__(self) -> None:
        """Register the static buffer."""

        super().__init__()
        self.register_buffer("buf", torch.arange(3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the buffer to the input.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Input plus buffer.
        """

        return x + self.buf


class BufferAlongsideComputeModel(nn.Module):
    """Model returning a static buffer alongside a computed output."""

    def __init__(self) -> None:
        """Register the static buffer and a linear layer."""

        super().__init__()
        self.register_buffer("buf", torch.arange(3.0))
        self.lin = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the untouched buffer and a computed tensor.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The raw buffer and the linear output.
        """

        return self.buf, self.lin(x)


class DuplicateBufferOutputModel(nn.Module):
    """Model returning the same untouched registered buffer twice."""

    def __init__(self) -> None:
        """Register the shared static buffer."""

        super().__init__()
        self.register_buffer("buf", torch.arange(3.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the same buffer object twice.

        Parameters
        ----------
        x:
            Ignored input tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Two references to the same registered buffer.
        """

        return self.buf, self.buf


class ForeignTensorBetweenOutputsModel(nn.Module):
    """Model returning an unattributed tensor between attributed outputs."""

    def __init__(self, foreign_tensor: torch.Tensor) -> None:
        """Store a tensor created before tracing begins.

        Parameters
        ----------
        foreign_tensor:
            Tensor that TorchLens should not attribute to a graph producer.
        """

        super().__init__()
        self.foreign_tensor = foreign_tensor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return attributed, foreign, and attributed tensors.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Computed output, unattributed tensor, computed output.
        """

        y1 = x + 1
        y2 = x * 2
        return y1, self.foreign_tensor, y2


@pytest.mark.smoke
def test_buffer_sole_output_traces_with_output_layer() -> None:
    """A buffer-only return must produce ``output_1`` bound to the buffer value."""

    model = BufferOnlyOutputModel()
    log = trace_fn(model, torch.rand(3), layers_to_save="all")
    try:
        assert log.output_layers == ["output_1"]
        output_layer = log["output_1"]
        assert torch.equal(output_layer.out, model.buf)

        # The output node hangs off a real buffer source node, not thin air.
        assert len(output_layer.parents) == 1
        parent = log[output_layer.parents[0]]
        assert parent.is_buffer
        assert parent.address == "buf"
        assert parent.layer_label in {label.rsplit(":", 1)[0] for label in log.buffer_layers} | set(
            log.buffer_layers
        )

        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_buffer_sole_output_does_not_leak_labels_onto_model_state() -> None:
    """Late buffer logging must not leave capture labels on the live buffer."""

    from torchlens.backends.torch._tl import get_tensor_label

    model = BufferOnlyOutputModel()
    log = trace_fn(model, torch.rand(3), layers_to_save="all")
    try:
        assert get_tensor_label(model.buf) is None
    finally:
        log.cleanup()

    # A second trace of the same instance must behave identically.
    second_log = trace_fn(model, torch.rand(3), layers_to_save="all")
    try:
        assert second_log.output_layers == ["output_1"]
        assert torch.equal(second_log["output_1"].out, model.buf)
    finally:
        second_log.cleanup()


def test_buffer_sole_output_validates_clean() -> None:
    """``validate(..., scope="forward")`` must pass for a buffer-only model."""

    assert tl.validate(BufferOnlyOutputModel(), torch.rand(3), scope="forward") is True


def test_buffer_in_compute_model_unchanged() -> None:
    """Models that use buffers inside computation keep their existing shape."""

    model = BufferInComputeModel()
    x = torch.rand(3)
    log = trace_fn(model, x, layers_to_save="all")
    try:
        assert log.output_layers == ["output_1"]
        assert torch.equal(log["output_1"].out, x + model.buf)
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()
    assert tl.validate(BufferInComputeModel(), torch.rand(3), scope="forward") is True


def test_buffer_alongside_computed_outputs_pairs_values_correctly() -> None:
    """A buffer returned next to computed outputs yields correctly paired outputs."""

    model = BufferAlongsideComputeModel()
    x = torch.rand(3)
    log = trace_fn(model, x, layers_to_save="all")
    try:
        assert log.output_layers == ["output_1", "output_2"]

        buffer_output = log["output_1"]
        assert torch.equal(buffer_output.out, model.buf)
        assert buffer_output.io_role == "output.0"
        assert log[buffer_output.parents[0]].is_buffer

        computed_output = log["output_2"]
        assert torch.equal(computed_output.out, model.lin(x))
        assert computed_output.io_role == "output.1"
        assert not log[computed_output.parents[0]].is_buffer

        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()

    assert tl.validate(BufferAlongsideComputeModel(), torch.rand(3), scope="forward") is True


def test_duplicate_direct_buffer_outputs_share_one_buffer_parent() -> None:
    """Repeated direct-buffer outputs must not late-log the buffer twice."""

    model = DuplicateBufferOutputModel()
    log = trace_fn(model, torch.rand(3), layers_to_save="all")
    try:
        assert log.output_layers == ["output_1", "output_2"]
        output_1 = log["output_1"]
        output_2 = log["output_2"]
        assert torch.equal(output_1.out, model.buf)
        assert torch.equal(output_2.out, model.buf)
        assert output_1.parents == output_2.parents

        buffer_parent = log[output_1.parents[0]]
        assert buffer_parent.is_buffer
        assert buffer_parent.address == "buf"
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()

    assert tl.validate(DuplicateBufferOutputModel(), torch.rand(3), scope="forward") is True


def test_foreign_tensor_between_attributed_outputs_does_not_shift_alignment() -> None:
    """Unattributed middle outputs must not shift later output bindings."""

    foreign_tensor = torch.full((3,), 7.0)
    model = ForeignTensorBetweenOutputsModel(foreign_tensor)
    x = torch.rand(3)
    log = trace_fn(model, x, layers_to_save="all")
    try:
        assert log.output_layers == ["output_1", "output_2"]

        first_output = log["output_1"]
        assert first_output.io_role == "output.0"
        assert torch.equal(first_output.out, x + 1)
        assert torch.equal(log[first_output.parents[0]].out, x + 1)

        second_output = log["output_2"]
        assert second_output.io_role == "output.2"
        assert torch.equal(second_output.out, x * 2)
        assert torch.equal(log[second_output.parents[0]].out, x * 2)

        assert not any(
            torch.equal(log[output_label].out, foreign_tensor) for output_label in log.output_layers
        )
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


class UnloggedPlusReassignedBufferModel(nn.Module):
    """Model returning an untouched buffer ahead of a reassigned one.

    Pins the pairing logic when a never-logged buffer output precedes a
    capture-attributed buffer output (whose live label was stripped during
    session cleanup): the reassigned buffer must keep its capture-time write
    node, and the untouched buffer must get a late source node -- in the
    right order.
    """

    def __init__(self) -> None:
        """Register both buffers."""

        super().__init__()
        self.register_buffer("static_buf", torch.arange(3.0))
        self.register_buffer("state", torch.zeros(3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reassign one buffer, return both.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The untouched static buffer and the freshly reassigned state.
        """

        self.state = torch.tanh(self.state + x)
        return self.static_buf, self.state


def test_unlogged_buffer_before_reassigned_buffer_output() -> None:
    """Late-logged and capture-logged buffer outputs pair to the right values."""

    model = UnloggedPlusReassignedBufferModel()
    x = torch.rand(3)
    log = trace_fn(model, x, layers_to_save="all")
    try:
        assert log.output_layers == ["output_1", "output_2"]

        static_output = log["output_1"]
        assert torch.equal(static_output.out, model.static_buf)
        static_parent = log[static_output.parents[0]]
        assert static_parent.is_buffer
        assert static_parent.address == "static_buf"

        state_output = log["output_2"]
        assert torch.equal(state_output.out, model.state)
        state_parent = log[state_output.parents[0]]
        assert state_parent.is_buffer
        assert state_parent.address == "state"

        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()
