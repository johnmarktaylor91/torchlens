"""Tests for Phase 3 repr and tensor display ergonomics."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


class _InfModel(nn.Module):
    """Tiny model that produces a non-finite out."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an infinite tensor."""

        return x / 0


def test_first_nonfinite_reports_context() -> None:
    """Trace.first_nonfinite reports the first saved NaN or Inf site."""

    log = tl.trace(_InfModel(), torch.ones(1, 2))
    answer = log.first_nonfinite()
    assert "First non-finite" in answer
    assert "shape=" in answer
    assert "dtype=" in answer
    assert "parents=" in answer
    assert "source=" in answer


class _TwoStage(nn.Module):
    """Two-op model so relation interpolation has real parents/children."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear followed by relu."""

        return torch.relu(self.lin(x))


def test_layer_repr_after_trace_collection_degrades_not_raises() -> None:
    """repr/str/format on a Layer whose Trace was collected must not raise.

    R52-A shape 1 (round-4 b7-opus): the natural one-liner
    ``tl.trace(model, x)[label]`` leaves the Layer holding a dead weakref;
    ``__repr__`` is data-model API and must degrade, never raise.
    """

    import gc

    layer = tl.trace(_TwoStage(), torch.randn(2, 4))["relu_1_2"]
    gc.collect()

    text = repr(layer)
    assert "relu_1_2" in text
    assert "detached" in text
    assert str(layer) == text
    assert f"context: {layer}"  # f-string interpolation must not raise
    # Relation accessors still refuse TYPED - repr degrading must not
    # loosen the accessor contract.
    from torchlens._errors import RecordBindingError

    with pytest.raises(RecordBindingError) as exc_info:
        _ = layer.parents
    assert exc_info.value.fields["code"] == "trace_reference_collected"


def test_layer_standalone_pickle_repr_and_accessors_typed() -> None:
    """A standalone-pickled Layer must repr fine and refuse accessors TYPED.

    R52-A shape 2 (round-4 b7-opus): ``__getstate__`` strips the trace
    weakref, so ``source_trace`` used to return a bare ``None`` behind a
    ``-> Trace`` signature and ``repr()`` crashed with an UNTYPED
    ``TypeError: 'NoneType' object is not subscriptable``.
    """

    import pickle

    from torchlens._errors import RecordBindingError

    trace = tl.trace(_TwoStage(), torch.randn(2, 4))
    restored = pickle.loads(pickle.dumps(trace["relu_1_2"]))

    text = repr(restored)
    assert "relu_1_2" in text
    assert "detached" in text
    assert str(restored) == text

    with pytest.raises(RecordBindingError) as exc_info:
        _ = restored.source_trace
    assert exc_info.value.fields["code"] == "record_not_bound"
    assert exc_info.value.fields["remedy"]

    with pytest.raises(RecordBindingError) as parents_exc:
        _ = restored.parents
    assert parents_exc.value.fields["code"] == "record_not_bound"


def test_op_repr_degrades_under_predicate_save() -> None:
    """repr/str of an UNSAVED Op under a predicate save must not raise.

    R01 (b1-opus r5): ``_tensor_contents_str_helper`` read ``self.out``
    unguarded, and ``Op.__getattribute__`` refuses that payload read with
    ``PayloadUnavailableError`` once a predicate save was used, so
    ``repr(trace._layer_list)`` raised on 6 of 7 ops.
    """

    trace = tl.trace(_TwoStage(), torch.randn(2, 4), save=tl.func("relu"))
    text = repr(trace._layer_list)
    assert text
    unsaved = next(op for op in trace._layer_list if not op.has_saved_activation)
    unsaved_text = f"{unsaved}"
    assert "not saved" in unsaved_text
    saved = trace["relu_1_2"]
    assert "not saved" not in f"{saved.ops[0]}"


def test_op_repr_after_trace_collection_degrades_not_raises() -> None:
    """repr/str on an Op whose Trace was collected must not raise (R52-B).

    The r4 fix (1a2b715e) landed on ``Layer`` only; ``Op`` -- the more
    numerous record class -- kept the identical type-lie: ``source_trace``
    returned a bare ``None`` behind ``-> Trace`` and ``__str__`` silently
    printed an unknown denominator (``operation 1/?``).
    """

    import gc

    from torchlens._errors import RecordBindingError

    op = tl.trace(_TwoStage(), torch.randn(2, 4))["relu_1_2"].ops[0]
    gc.collect()

    text = repr(op)
    assert "relu_1_2" in text
    assert "detached" in text
    assert str(op) == text
    with pytest.raises(RecordBindingError) as exc_info:
        _ = op.source_trace
    assert exc_info.value.fields["code"] == "trace_reference_collected"


def test_op_standalone_pickle_accessors_refuse_typed() -> None:
    """A standalone-pickled Op refuses relation getters TYPED, never TypeError (R52-B)."""

    import pickle

    from torchlens._errors import RecordBindingError

    trace = tl.trace(_TwoStage(), torch.randn(2, 4))
    restored = pickle.loads(pickle.dumps(trace["relu_1_2"].ops[0]))

    text = repr(restored)
    assert "detached" in text

    with pytest.raises(RecordBindingError) as exc_info:
        _ = restored.source_trace
    assert exc_info.value.fields["code"] == "record_not_bound"
    with pytest.raises(RecordBindingError) as children_exc:
        restored.get_children()
    assert children_exc.value.fields["code"] == "record_not_bound"
    with pytest.raises(RecordBindingError) as parents_exc:
        restored.get_parents()
    assert parents_exc.value.fields["code"] == "record_not_bound"


def test_op_repr_on_live_trace_unchanged() -> None:
    """An Op bound to a live Trace keeps the full informative repr (R52-B)."""

    trace = tl.trace(_TwoStage(), torch.randn(2, 4))
    op = trace["relu_1_2"].ops[0]
    text = repr(op)
    assert "relu_1_2" in text
    assert "detached" not in text
    assert "/?" not in text  # denominator is the real op count, not unknown
    assert op.get_parents() and op.get_children() is not None
