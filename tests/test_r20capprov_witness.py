"""Regression tests for the unattributed-tensor-argument witness false positive.

Context
-------
The postprocess "no graph/source provenance" witness
(``torchlens._postprocess._warn_unattributed_tensor_args``, fed by
``torchlens.backends.torch.ops._unattributed_tensor_arg_positions``) exists to
catch capture gaps: tensors that flow into an op with no traced provenance. Its
branch (2) additionally flags a *provenanced* tensor whose producer label is not
one of the op's recorded parent edges.

That branch (2) mis-fired on dynamic-shape families: a scalar tensor that is
fully traced (``has_known_provenance``) but is consumed as a **size/shape**
argument (a ``torch.zeros`` size dim, a ``view``/``reshape``/``as_strided`` size
arg, a ``new_zeros`` size element) is deliberately excluded from the op's
recorded parents, so branch (2) wrongly promoted it to an error. This broke
``test_packed_sequence`` (LSTM internal ``h0``/``c0`` allocation) and
``test_longformer`` (windowed-attention dynamic dims).

Round-31 H2 superseded the schema classifier that used to narrow branch (2):
a runtime TENSOR at any input slot -- including schema-typed ``int``/``Scalar``
control slots such as factory size dims -- is a real data dependency, and the
extraction coverage guard now records it as a graph PARENT. The dynamic-shape
families that motivated the old narrowing therefore no longer produce
unattributed provenanced tensors at all (the size producer is an attributed
parent), and branch (2) stays armed at EVERY slot: a provenanced tensor missing
from the recorded parents is a dropped edge wherever it sits. An UN-provenanced
tensor at any position is caught earlier by branch (1), unchanged.
"""

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl

_PROVENANCE_MATCH = "no graph/source provenance"

# Foreign tensors constructed OUTSIDE any capture session carry no TorchLens
# provenance at all -- the "real leak" the witness must never miss.
_FOREIGN_SIZE_A = torch.tensor(4)
_FOREIGN_SIZE_B = torch.tensor(6)
_FOREIGN_OPERAND = torch.ones(24)


class _ProvenancedDynamicSize(nn.Module):
    """Feed a fully-traced scalar tensor as a ``reshape`` size argument."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``x`` using a size derived from a traced op.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ``x`` reshaped with a dynamically computed final dimension.
        """

        cols = torch.tensor([x.shape[1] * x.shape[2]])  # traced producer
        return x.reshape(x.shape[0], cols[0])  # cols[0] is provenanced, at arg2 (size)


class _ForeignSize(nn.Module):
    """Feed UN-provenanced foreign scalar tensors as ``reshape`` size args."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape a flattened view with foreign size tensors.

        Parameters
        ----------
        x:
            Input tensor with 24 elements.

        Returns
        -------
        torch.Tensor
            ``x`` reshaped to ``(4, 6)`` via foreign size tensors.
        """

        return x.flatten().reshape(_FOREIGN_SIZE_A, _FOREIGN_SIZE_B)


class _ForeignOperand(nn.Module):
    """Add an UN-provenanced foreign tensor at a genuine data-operand position."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add a foreign tensor to a flattened view.

        Parameters
        ----------
        x:
            Input tensor with 24 elements.

        Returns
        -------
        torch.Tensor
            The elementwise sum with the foreign operand.
        """

        return x.flatten() + _FOREIGN_OPERAND  # foreign at add arg1 (data operand)


def _provenance_warnings(model: nn.Module, x: torch.Tensor) -> list[str]:
    """Return every provenance-witness warning message raised while tracing.

    Parameters
    ----------
    model:
        Module to trace.
    x:
        Model input.

    Returns
    -------
    list[str]
        Messages of captured provenance-witness ``UserWarning`` instances.
    """

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.trace(model, x)
    return [str(w.message) for w in caught if _PROVENANCE_MATCH in str(w.message)]


def test_provenanced_dynamic_size_arg_no_false_positive() -> None:
    """A fully-traced scalar used as a size argument must NOT be flagged."""

    assert _provenance_warnings(_ProvenancedDynamicSize().eval(), torch.randn(2, 3, 4)) == []


def test_foreign_unprovenanced_size_arg_still_flags() -> None:
    """An un-provenanced foreign tensor at a size position MUST still fire.

    Guards the no-false-negative direction: the narrowing gates on
    ``has_known_provenance`` (branch 1), so a real leak at a size position is
    never masked.
    """

    with pytest.warns(UserWarning, match=_PROVENANCE_MATCH):
        tl.trace(_ForeignSize().eval(), torch.randn(2, 3, 4))


def test_foreign_unprovenanced_operand_arg_still_flags() -> None:
    """An un-provenanced foreign tensor at a data-operand position MUST fire."""

    with pytest.warns(UserWarning, match=_PROVENANCE_MATCH):
        tl.trace(_ForeignOperand().eval(), torch.randn(2, 3, 4))


def test_provenanced_dynamic_size_arg_becomes_recorded_parent() -> None:
    """Round-31 H2: a traced tensor consumed as a size arg IS a graph parent.

    The old schema classifier suppressed the witness at size/shape slots
    because the graph builder deliberately excluded them from parents. That
    premise is gone: extraction's runtime coverage guard records the traced
    size producer as a real parent (its value determines the output), so the
    benign case that motivated the suppression no longer exists and the
    witness can stay armed at every slot.
    """

    trace = tl.trace(_ProvenancedDynamicSize().eval(), torch.randn(2, 3, 4))
    reshape_op = next(op for op in trace.ops if op.func_name == "reshape")
    getitem_labels = [op.layer_label for op in trace.ops if op.func_name == "__getitem__"]
    assert any(parent in getitem_labels for parent in reshape_op.parents)
    assert reshape_op.unattributed_tensor_args == ()
