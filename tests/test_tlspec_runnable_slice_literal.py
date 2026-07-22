"""Sparse runnable literal-grammar coverage for Python slice/index literals.

``x[:, 3:]``-style subscripting lowers ``__getitem__`` calls whose key argument is a
``slice`` (and, for multi-axis indexing, a tuple mixing ``slice``/``int``/``None``/
``Ellipsis``). These are inert value types -- no callables, no imports -- so the sparse
runnable literal grammar admits them as a new ``LiteralSlice`` literal kind plus the
existing ``LiteralAtom`` machinery for ``Ellipsis``/``None``. This file locks that the
save -> load -> run round trip is VERIFIED and value-correct on both the original
capture input and a changed input (slicing is a deterministic, input-independent
operation, so a changed input must still verify), and that the literal grammar stays
fail-closed for genuinely unsupported literal types.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.runnable import (
    _UnsupportedLiteralError,
    _encode_literal,
    build_sparse_run_descriptor,
)
from torchlens.errors import RunnablePreflightError
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness, RunnableErrorCode


class OpenEndedSlice(nn.Module):
    """``x[:, 3:]`` -- an open-ended slice on the trailing axis."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Slice from a fixed offset to the end."""

        return value[:, 3:] + 1


class BoundedSlice(nn.Module):
    """``x[:, 1:5]`` -- a fully bounded slice."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Slice a fixed sub-range."""

        return value[:, 1:5] + 1


class IntegerIndex(nn.Module):
    """``x[:, 0]`` -- a plain integer index, not a slice."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Select a single column by integer index."""

        return value[:, 0] + 1


class SteppedSlice(nn.Module):
    """``x[:, ::2]`` -- a slice with a non-trivial step."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Select every other column."""

        return value[:, ::2] + 1


class NegativeSlice(nn.Module):
    """``x[:, -1:]`` -- a slice anchored with a negative start."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Select the trailing column via a negative-start slice."""

        return value[:, -1:] + 1


class EllipsisIndex(nn.Module):
    """``x[..., 0]`` -- an ``Ellipsis`` literal in the index tuple."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Select the last-axis column via an ``Ellipsis`` prefix."""

        return value[..., 0] + 1


class NewaxisIndex(nn.Module):
    """``x[:, None]`` -- a bare ``None`` (newaxis) literal in the index tuple."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Insert a new axis, then remove it again for shape parity."""

        return (value[:, None] + 1).squeeze(1)


class GeneratorLiteralKwarg(nn.Module):
    """A ``torch.Generator`` keyword literal -- genuinely outside the literal grammar."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Pass a non-tensor, non-scalar object as a literal op argument."""

        generator = torch.Generator()
        generator.manual_seed(0)
        return torch.rand(value.shape, generator=generator) + value


def _save_runnable(model: nn.Module, capture_input: torch.Tensor, path: Path) -> Path:
    """Capture with intervention-ready state and save a sparse runnable artifact."""

    trace = tl.trace(
        model,
        capture_input,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    trace.save(path, level="runnable")
    return path


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("model_cls", "reference_fn"),
    (
        (OpenEndedSlice, lambda x: x[:, 3:] + 1),
        (BoundedSlice, lambda x: x[:, 1:5] + 1),
        (IntegerIndex, lambda x: x[:, 0] + 1),
        (SteppedSlice, lambda x: x[:, ::2] + 1),
        (NegativeSlice, lambda x: x[:, -1:] + 1),
        (EllipsisIndex, lambda x: x[..., 0] + 1),
        (NewaxisIndex, lambda x: (x[:, None] + 1).squeeze(1)),
    ),
    ids=(
        "open_ended_slice",
        "bounded_slice",
        "integer_index",
        "stepped_slice",
        "negative_slice",
        "ellipsis_index",
        "newaxis_index",
    ),
)
def test_slice_index_literal_save_load_run_verified_on_original_input(
    tmp_path: Path,
    model_cls: type[nn.Module],
    reference_fn: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """A slice/index literal saves, loads, and runs VERIFIED on the capture input."""

    model = model_cls()
    capture_input = torch.randn(2, 8)
    path = _save_runnable(model, capture_input, tmp_path / "slice.tlspec")

    result = tl.load(path).run(inputs=capture_input)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.allclose(result.output, reference_fn(capture_input))


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("model_cls", "reference_fn"),
    (
        (OpenEndedSlice, lambda x: x[:, 3:] + 1),
        (BoundedSlice, lambda x: x[:, 1:5] + 1),
        (IntegerIndex, lambda x: x[:, 0] + 1),
        (SteppedSlice, lambda x: x[:, ::2] + 1),
        (NegativeSlice, lambda x: x[:, -1:] + 1),
        (EllipsisIndex, lambda x: x[..., 0] + 1),
        (NewaxisIndex, lambda x: (x[:, None] + 1).squeeze(1)),
    ),
    ids=(
        "open_ended_slice",
        "bounded_slice",
        "integer_index",
        "stepped_slice",
        "negative_slice",
        "ellipsis_index",
        "newaxis_index",
    ),
)
def test_slice_index_literal_save_load_run_verified_on_changed_input(
    tmp_path: Path,
    model_cls: type[nn.Module],
    reference_fn: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """The same slice/index literal must still verify on a changed same-shape input.

    Slicing is a deterministic, input-value-independent operation, so a same-shape
    changed input replays the recorded taken path faithfully and must report
    VERIFIED with a value-correct result -- never a silently stale/false-VERIFIED.
    """

    model = model_cls()
    capture_input = torch.randn(2, 8)
    path = _save_runnable(model, capture_input, tmp_path / "slice.tlspec")

    changed_input = torch.randn(2, 8)
    result = tl.load(path).run(inputs=changed_input)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.allclose(result.output, reference_fn(changed_input))


def test_callable_literal_argument_still_refuses_with_unsupported_literal(
    tmp_path: Path,
) -> None:
    """A genuinely unsupported literal type (``torch.Generator``) must still refuse.

    Adding the slice/Ellipsis/None literal kinds must not widen the grammar beyond
    those inert value types -- an object outside the frozen grammar (here a
    ``torch.Generator`` keyword literal) must keep failing closed with
    ``UNSUPPORTED_LITERAL`` end to end at save time.
    """

    model = GeneratorLiteralKwarg()
    capture_input = torch.randn(3)
    trace = tl.trace(
        model,
        capture_input,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    descriptor = build_sparse_run_descriptor(trace)
    path = tmp_path / "unsupported.tlspec"

    assert not descriptor.preflight.passed
    assert RunnableErrorCode.UNSUPPORTED_LITERAL in {
        diagnostic.code for diagnostic in descriptor.preflight.diagnostics
    }
    with pytest.raises(RunnablePreflightError, match="producer preflight failed"):
        trace.save(path, level="runnable")
    assert not path.exists()


def test_encode_literal_rejects_a_bare_callable() -> None:
    """The literal encoder itself must reject a bare Python callable."""

    def _callable_literal() -> None:
        """An arbitrary Python callable -- outside the frozen literal grammar."""

    with pytest.raises(_UnsupportedLiteralError):
        _encode_literal(_callable_literal)


def test_encode_literal_rejects_an_arbitrary_object() -> None:
    """The literal encoder itself must reject an arbitrary non-grammar object."""

    class _Opaque:
        """An arbitrary object with no literal-grammar representation."""

    with pytest.raises(_UnsupportedLiteralError):
        _encode_literal(_Opaque())


def test_encode_literal_rejects_a_non_integer_slice_component() -> None:
    """A slice component outside ``{None, int}`` must still fail closed."""

    with pytest.raises(_UnsupportedLiteralError):
        _encode_literal(slice(1.5, None, None))


def test_encode_literal_round_trips_slice_ellipsis_and_none() -> None:
    """Unit-level encode/decode parity for the three new literal shapes."""

    from torchlens._runnable_execution import _decode_literal

    assert _decode_literal(_encode_literal(slice(3, None, None))) == slice(3, None, None)
    assert _decode_literal(_encode_literal(slice(1, 5, None))) == slice(1, 5, None)
    assert _decode_literal(_encode_literal(slice(None, None, 2))) == slice(None, None, 2)
    assert _decode_literal(_encode_literal(slice(-1, None, None))) == slice(-1, None, None)
    assert _decode_literal(_encode_literal(Ellipsis)) is Ellipsis
    assert _decode_literal(_encode_literal(None)) is None
    # An Ellipsis and a bare None must not be confused: both carry `.value is None`
    # on the wire, so decoding must key off the atom KIND, not the stored value.
    assert _decode_literal(_encode_literal(Ellipsis)) is not None


# ======================================================================================
# r71 B -- composite-literal semantic laundering (secA-F1: slice components)
# ======================================================================================


import enum as _enum  # noqa: E402
import warnings as _warnings  # noqa: E402
from typing import Any  # noqa: E402

import numpy as _np  # noqa: E402

from torchlens.errors import PathDivergenceError, RunPreconditionError  # noqa: E402


class _Mode(_enum.IntEnum):
    FAST = 0
    SLOW = 5


class _CustomInt(int):
    pass


_SEMANTIC_INT_SPECIMENS: dict[str, Any] = {
    "int_enum": _Mode.SLOW,
    "custom_int": _CustomInt(5),
    "np_int64_subclass": type("WeirdNp", (_np.int64,), {})(5),
}


def _b_capture(model: nn.Module, inputs: Any) -> tl.Trace:
    return tl.trace(
        model,
        inputs,
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )


def _b_save(trace: tl.Trace, path: Path) -> Path:
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        trace.save(path, level="runnable", include_weights=True)
    return path


@pytest.mark.parametrize("name", sorted(_SEMANTIC_INT_SPECIMENS))
def test_r71b_semantic_slice_component_refuses_typed_at_save(name: str, tmp_path: Path) -> None:
    """Every semantic-int specimen refuses typed at BOTH the encoder and save
    across slice start/stop/step, nested slice-in-list/tuple/mapping-value."""

    value = _SEMANTIC_INT_SPECIMENS[name]
    # Direct encoder: classifier-first, refuses before any int() coercion.
    for component_slice in (slice(value), slice(0, value), slice(0, 9, value)):
        with pytest.raises(_UnsupportedLiteralError):
            _encode_literal(component_slice)
    # Nested composite edges must refuse identically (sequence item, tuple item,
    # mapping value).
    with pytest.raises(_UnsupportedLiteralError):
        _encode_literal([slice(value)])
    with pytest.raises(_UnsupportedLiteralError):
        _encode_literal((slice(0, value),))
    with pytest.raises(_UnsupportedLiteralError):
        _encode_literal({"k": slice(0, 9, value)})

    # A slice model-input leaf carrying a semantic component refuses at save.
    class _SliceInput(nn.Module):
        def forward(self, x: torch.Tensor, idx: slice) -> torch.Tensor:
            return x[idx] + 1.0

    x = torch.arange(10.0)
    with pytest.raises(RunnablePreflightError) as excinfo:
        _b_save(_b_capture(_SliceInput(), [x, slice(0, value)]), tmp_path / f"b_{name}.tlspec")
    diagnostics = str(excinfo.value.fields.get("diagnostics"))
    assert "missing_input_container_contract" in diagnostics
    assert "semantic_scalar_type" in diagnostics


@pytest.mark.smoke
def test_r71b_secA_pin_plain_capture_semantic_runtime_diverges(tmp_path: Path) -> None:
    """The named secA-F1 pin: a captured forward branching on a slice-component TYPE
    replayed with a same-VALUE plain-int input VERIFIES on the plain twin and DIVERGES
    on the semantic twin (never false VERIFIED)."""

    class _TypeSteer(nn.Module):
        def forward(self, x: torch.Tensor, idx: slice) -> torch.Tensor:
            if isinstance(idx.start, _Mode):
                return x * 2.0
            return x + 100.0

    x = torch.tensor([1.0, 2.0, 3.0])
    # Plain-slice capture (the admitted lane).
    path = _b_save(_b_capture(_TypeSteer(), [x, slice(0, 3)]), tmp_path / "b_pin.tlspec")
    assert (
        tl.load(path).run(inputs=[x, slice(0, 3)]).report.path_faithfulness
        is PathFaithfulness.VERIFIED
    )
    # Same-VALUE semantic runtime component diverges (component-by-component compare,
    # never slice.__eq__).
    with pytest.raises((PathDivergenceError, RunPreconditionError)):
        tl.load(path).run(inputs=[x, slice(_Mode.FAST, 3)])


def test_r71b_ratified_stock_wrapper_slice_component_normalizes() -> None:
    """The ratified stock-numpy wrapper int lane normalizes to the exact builtin atom
    at the encoder (value-transparent; no semantic refusal)."""

    from torchlens._runnable_execution import _decode_literal

    encoded = _encode_literal(slice(_np.int64(1), _np.int64(5), _np.int64(2)))
    assert _decode_literal(encoded) == slice(1, 5, 2)


def test_r71b_greens_ordinary_slices_stay_verified(tmp_path: Path) -> None:
    """No over-trigger: negative/open/bounded/stepped slices, None, and plain ints all
    stay admitted + VERIFIED."""

    class _SliceInput(nn.Module):
        def forward(self, x: torch.Tensor, idx: slice) -> torch.Tensor:
            return x[idx] + 1.0

    x = torch.arange(10.0)
    for index, idx in enumerate(
        (
            slice(None),
            slice(2, None),
            slice(1, 5),
            slice(None, None, 2),
            slice(-3, None),
        )
    ):
        path = _b_save(_b_capture(_SliceInput(), [x, idx]), tmp_path / f"b_green_{index}.tlspec")
        result = tl.load(path).run(inputs=[x, idx])
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.smoke
def test_r71b_composite_component_policy_covers_every_edge() -> None:
    """Meta-test: the composite policy table keys == every composite NonTensorLiteral
    node kind; a new composite without a declared component policy REDs here."""

    from torchlens._input_walk import COMPOSITE_LITERAL_COMPONENT_POLICY
    from torchlens.runnable import (
        LiteralAtom,
        LiteralMapping,
        LiteralSequence,
        LiteralSlice,
        LiteralTorchSymbol,
        LiteralTupleKey,
    )

    # The recursive (composite) literal node kinds -- leaves excluded.
    composite_kinds = {
        LiteralSlice.__name__,
        LiteralSequence.__name__,
        LiteralMapping.__name__,
        LiteralTupleKey.__name__,
    }
    leaf_kinds = {LiteralAtom.__name__, LiteralTorchSymbol.__name__}
    assert set(COMPOSITE_LITERAL_COMPONENT_POLICY) == composite_kinds
    assert not (composite_kinds & leaf_kinds)
    for policy in COMPOSITE_LITERAL_COMPONENT_POLICY.values():
        assert policy, "every composite edge declares its classifier lane"
        for lane in policy:
            assert lane in {"classify_scalar", "_encode_literal", "encode_mapping_key"}, lane

    # Classifier-before-coercion at the slice encoder (source scan).
    import inspect

    from torchlens._io import runnable as io_runnable

    slice_source = inspect.getsource(io_runnable._encode_slice_component)
    assert slice_source.index("classify_scalar") < slice_source.index("isinstance(")

    # _literal_leaf_equal descends slices component-by-component, never slice.__eq__.
    from torchlens import _runnable_execution

    compare_source = inspect.getsource(_runnable_execution._literal_leaf_equal)
    assert "isinstance(recorded, slice)" in compare_source
