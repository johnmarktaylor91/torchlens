"""``copy.deepcopy`` of a tensor inside a captured forward is a new tensor, not its source.

``Tensor.__deepcopy__`` ends with ``new.__dict__ = deepcopy(self.__dict__, memo)``,
which copies TorchLens's ``._tl`` metadata onto the copy. Because the storage was
deep-copied through the same memo, the copied label's storage pin even maps onto
the copy's own storage, so the copy used to read as the SAME current-session
tensor: the deepcopy call looked like it returned its input, no op was logged,
every consumer of the copy was wired to the source node, and the storage copy's
dispatches stayed unaccounted (forward validation failed on completeness).
"""

from __future__ import annotations

import copy

import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch import _tl


class _DeepcopyBuffer(nn.Module):
    """Use a deep copy of a buffer next to the buffer itself."""

    def __init__(self) -> None:
        """Register the buffer."""

        super().__init__()
        self.register_buffer("buf", torch.arange(4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply by the copy, add the buffer.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ``x * deepcopy(buf) + buf``.
        """

        copied = copy.deepcopy(self.buf)
        return x * copied + self.buf


class _DeepcopyThenMutateSource(nn.Module):
    """Copy an intermediate, then write the source in place."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``deepcopy(y) * y`` where ``y`` is incremented after the copy.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Product of the pre-increment copy and the incremented source.
        """

        y = x * 2
        snapshot = copy.deepcopy(y)
        y.add_(1)
        return snapshot * y


def test_deepcopy_of_a_buffer_is_its_own_node_and_validates() -> None:
    """The copy is logged as an op on the buffer, and forward replay validates."""

    model = _DeepcopyBuffer()
    x = torch.ones(4)
    trace = tl.trace(model, x)
    try:
        (buffer_op,) = [op for op in trace.layer_list if op.is_buffer]
        (mul_op,) = [op for op in trace.layer_list if op.func_name == "__mul__"]
        assert buffer_op.label not in mul_op.parents, mul_op.parents
        (copy_parent,) = [trace[label] for label in mul_op.parents if label != "input_1"]
        assert copy_parent.func_name == "__deepcopy__"
        assert copy_parent.parents == (buffer_op.label,)
    finally:
        trace.cleanup()
    assert tl.validate(_DeepcopyBuffer(), x, scope="forward") is True


def test_deepcopy_snapshot_of_a_later_mutated_intermediate_validates() -> None:
    """A deep-copied snapshot keeps its pre-write value through replay."""

    assert tl.validate(_DeepcopyThenMutateSource(), torch.ones(4), scope="forward") is True


def test_deepcopied_tensor_meta_drops_only_the_session_identity() -> None:
    """The copy keeps inert raw history but never the source object's session anchor."""

    source = torch.zeros(3)
    session = _tl.begin_label_session()
    try:
        _tl.set_tensor_label(source, "relu_1_1_raw")
        _tl.set_buffer_address(source, "block.buf")
        _tl.mark_tensor_data_alias(source)
        assert _tl.get_tensor_label(source) == "relu_1_1_raw"

        copied = copy.deepcopy(source)

        meta = _tl.get_tensor_meta(copied)
        assert meta is not None
        assert meta.label_session is None
        assert meta.label_storage is None
        assert meta.data_alias is False
        assert meta.same_object_mutation is None
        assert meta.label_raw == "relu_1_1_raw"
        assert meta.address == "block.buf"
        # Inside the capture the copy is not the labeled object.
        assert _tl.get_tensor_label(copied) is None
        assert not _tl.is_tensor_data_alias(copied)
        # The source keeps its own identity.
        assert _tl.get_tensor_label(source) == "relu_1_1_raw"
        assert _tl.get_tensor_meta(source).label_session == session
    finally:
        _tl.end_label_session()
    # Outside a capture raw labels read unchanged, as before.
    assert _tl.get_tensor_label(copied) == "relu_1_1_raw"
