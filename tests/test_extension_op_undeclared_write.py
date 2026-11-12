"""An extension operator that fills an argument its schema does not mark as written.

torch_scatter registers ``scatter_max(Tensor src, Tensor index, int dim, Tensor?
optional_out, int? dim_size)`` from C++ with no alias annotations and fills
``optional_out`` in place. HARP-NeXt (Menagerie m17214) passes a ``torch.zeros``
buffer there, so the deferred saved out of ``zeros`` showed the scatter's later write
and replay of ``zeros`` disagreed with it. The ``torch.ops`` recorder now copies out
pending payload aliases of every tensor argument of an operator outside the ATen and
prims namespaces, whose schemas are the only ones trusted to declare their writes.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import pytest
import torch
import torch.nn as nn

import torchlens as tl
import torchlens.backends.torch._torch_ops_calls as _ops_calls
import torchlens.backends.torch._wrapper_session as _wrapper_session
from torchlens._utils import tensor_utils as _tu
from torchlens.validation import last_validation_failure, validate_forward_pass

# These tests assert that payloads defer, which no capture does once Dynamo is loaded.
pytestmark = pytest.mark.usefixtures("dynamo_free_process")

_LIB = torch.library.Library("torchlens_test_rtl4", "DEF")  # noqa: TOR901
_LIB.define("fill_out(Tensor src, Tensor? out) -> Tensor")


def _fill_out_impl(src: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    """Accumulate ``src`` into ``out`` below every wrapped surface; return a fresh copy.

    Like ``scatter_max`` with ``out=``, the result depends on ``out``'s initial value.
    """

    assert out is not None
    torch.ops.aten.add_.Tensor(out, src)
    return torch.ops.aten.clone.default(out)


_LIB.impl("fill_out", _fill_out_impl, "CompositeExplicitAutograd")


@contextlib.contextmanager
def _deferred_payload_clones() -> Iterator[None]:
    """Force deferred (clone-on-write) payload clones inside the block."""

    prev_tu, prev_w = _tu._DEFER_ENABLED, _wrapper_session._COW_ENABLED
    _tu._DEFER_ENABLED = True
    _wrapper_session._COW_ENABLED = True
    try:
        yield
    finally:
        _tu._DEFER_ENABLED = prev_tu
        _wrapper_session._COW_ENABLED = prev_w


class _ScatterIntoZeros(nn.Module):
    """Accumulate into a ``zeros`` buffer through an operator that does not declare the write."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        buffer = torch.zeros(x.shape)
        filled = torch.ops.torchlens_test_rtl4.fill_out(x * 2.0, buffer)
        return filled + 1.0


def test_undeclared_write_keeps_the_zeros_payload() -> None:
    """The ``zeros`` saved out keeps the zeros it produced."""

    torch.manual_seed(0)
    with _deferred_payload_clones():
        trace = tl.trace(_ScatterIntoZeros(), torch.randn(3, 4))
    (zeros,) = [op for op in trace._layer_list if op.func_name == "zeros"]
    assert torch.equal(zeros.out, torch.zeros(3, 4))


def test_undeclared_write_validates() -> None:
    """Forward replay of ``zeros`` and of the operator reproduces their saved outs."""

    torch.manual_seed(0)
    with _deferred_payload_clones():
        assert tl.validate(_ScatterIntoZeros(), torch.randn(3, 4), seed=0)
    torch.manual_seed(0)
    assert validate_forward_pass(_ScatterIntoZeros(), torch.randn(3, 4))


def test_without_the_copy_out_the_argument_check_fails(monkeypatch) -> None:  # noqa: ANN001
    """Planted bug: with the copy-out disabled, the operator's saved argument fails as before.

    The deferred ``zeros`` payload is the buffer the operator received, so its saved
    argument shows the operator's own write and argument logging rejects it, the
    signature m17214 showed at ``scatter_max``.
    """

    monkeypatch.setattr(_ops_calls, "materialize_deferred_for_call", lambda tensors: None)
    torch.manual_seed(0)
    with _deferred_payload_clones():
        assert not tl.validate(_ScatterIntoZeros(), torch.randn(3, 4), seed=0)
    failure = last_validation_failure()
    assert failure is not None and failure.func_name == "fill_out", failure
    assert failure.check_name == "argument_logging", failure
