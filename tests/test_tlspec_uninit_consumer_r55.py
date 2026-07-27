"""r55 hon_2 + hon_1-consumer immunizers -- the load-side taint classifier (W2 lane).

hon_2 (MED, unsound sanitizer): the r53 ``out=`` taint sanitizer dropped the
destination's uninitialized-memory taint at ANY ``out=`` write, on the false
assumption of exact value semantics. ``torch.add(a, x, out=a)`` computes ``a + x``,
which READS ``a``'s prior (uninitialized) bytes -- yet the sanitizer laundered the
result to untainted, defeating the r53 structural ceiling. The fix is
aliasing-aware: preserve the destination's taint when the ``out=`` destination is
ALSO a value operand of the same op (fail closed, r35 unknown-alias precedent).

hon_1 (MED, missing family member consumer): ``Tensor.new(*sizes)`` is an
uninitialized-memory allocator with no aten spelling. W1 tabled it (size-gated);
this lane WIRES that predicate into ``_nondeterministic_value_sources`` so the
size form is recognized as uninitialized while the data form
(``new([values])``/``new(tensor)``) stays clean and an undecidable form fails
closed to tainted.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._runnable_execution import (
    _nondeterministic_value_sources,
    _uninit_taint_reaches,
)

pytestmark = pytest.mark.smoke

_CAPTURE = dict(intervention_ready=True)


@contextmanager
def _realistic_nondeterministic_fill() -> Iterator[None]:
    """Capture under the DEFAULT user context (deterministic algorithms OFF).

    ``tests/conftest.py`` enables ``torch.use_deterministic_algorithms(True)``
    globally, under which the ``empty`` family is deterministically NaN-filled and
    the uninit-nondeterminism class does not exist. This restores the ordinary
    eager default for the capture window so the r52/r53 uninit taint is real.
    """

    was = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was, warn_only=warn_only)


def _output_taint_reaches(bundle: Path) -> bool:
    loaded = tl.load(str(bundle))
    descriptor = loaded.__dict__["_runnable_descriptor"]
    taint = _nondeterministic_value_sources(descriptor)
    output_slots = [
        slot.slot_id
        for slot in descriptor.tensor_slots
        if slot.role.name == "OUTPUT" or slot.output_path is not None
    ]
    return _uninit_taint_reaches(taint, output_slots)


def _build(tmp_path: Path, name: str, model: nn.Module, x: torch.Tensor) -> Path:
    with _realistic_nondeterministic_fill():
        trace = tl.trace(model.eval(), x, **_CAPTURE)
    bundle = tmp_path / name
    tl.save(trace, str(bundle), level="runnable", include_weights=True)
    return bundle


# --------------------------------------------------------------------------- #
# hon_2 -- aliasing-aware out= sanitizer                                        #
# --------------------------------------------------------------------------- #


class _AliasOut(nn.Module):
    """``torch.add(a, x, out=a)`` -- out= ALIASES a value operand (reads a's bytes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.empty_like(x)  # uninitialized
        torch.add(a, x, out=a)  # a + x -> depends on a's prior uninit bytes
        return a


class _NonAliasOut(nn.Module):
    """``torch.add(x, x, out=c)`` -- c is a distinct dest, totally overwritten -> clean."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.empty_like(x)  # uninitialized dest, but result independent of its bytes
        torch.add(x, x, out=c)
        return c


def test_aliased_out_preserves_uninit_taint(tmp_path: Path) -> None:
    """hon_2: an out= that aliases a value operand PRESERVES the uninit taint."""

    bundle = _build(tmp_path, "alias_out.tlspec", _AliasOut(), torch.randn(4))
    assert _output_taint_reaches(bundle) is True


def test_nonaliased_out_still_sanitizes(tmp_path: Path) -> None:
    """hon_2 non-regression: a genuine total-write out= to a distinct dest sanitizes."""

    bundle = _build(tmp_path, "nonalias_out.tlspec", _NonAliasOut(), torch.randn(4))
    assert _output_taint_reaches(bundle) is False


# --------------------------------------------------------------------------- #
# hon_1 consumer -- Tensor.new size-form vs data-form                          #
# --------------------------------------------------------------------------- #


class _NewSize(nn.Module):
    """``x.new(5)`` -- legacy SIZE-form allocator (uninitialized memory)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x.new(5)
        return a, x.sum()


class _NewData(nn.Module):
    """``x.new([1.0, 2.0, 3.0])`` -- DATA-form copy constructor (deterministic)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x.new([1.0, 2.0, 3.0])
        return a, x.sum()


def test_new_size_form_recognized_uninit(tmp_path: Path) -> None:
    """hon_1: ``Tensor.new(size)`` is recognized as uninitialized -> taint reaches output."""

    bundle = _build(tmp_path, "new_size.tlspec", _NewSize(), torch.randn(4))
    assert _output_taint_reaches(bundle) is True


def test_new_data_form_stays_clean(tmp_path: Path) -> None:
    """hon_1 over-ceiling guard: ``Tensor.new([data])`` is a deterministic copy -> clean."""

    bundle = _build(tmp_path, "new_data.tlspec", _NewData(), torch.randn(4))
    assert _output_taint_reaches(bundle) is False
