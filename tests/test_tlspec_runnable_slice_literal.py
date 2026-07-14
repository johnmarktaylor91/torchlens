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
