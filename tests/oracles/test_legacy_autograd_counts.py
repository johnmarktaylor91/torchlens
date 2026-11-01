"""Raw-torch recount oracles for the legacy autograd-saved quantities.

The expected bytes and tensor counts come from running each op of the fixture
again in plain torch under ``torch.autograd.graph.saved_tensors_hooks`` and
recording every tensor autograd packs for backward, never from TorchLens. Per
op, a storage counts once (by data pointer) and a view counts in full, which is
the per-op band's definition; the trace, module and call doors sum those bands.

Capture runs floating Parameters with ``requires_grad=True`` unless
``backward_ready=True`` (the counts describe the captured forward), so the
default-capture recount sets the same flag on a copy of the model, and the
``backward_ready=True`` case is recounted on the model exactly as the user
froze it. These are the I1 oracles the census
(``tests/oracles/data/legacy_quantities.tsv``) names for those doors.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from tests.oracles.test_legacy_graph_counts import Grad
from torchlens.options import CaptureOptions


def _packed(fn: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run ``fn`` and return its output and every tensor autograd saved."""

    packed: list[torch.Tensor] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        packed.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = fn()
    return out, packed


def _band(out: torch.Tensor, packed: list[torch.Tensor]) -> tuple[int | None, int | None]:
    """Per-op band: bytes and count of saved tensors, one per data pointer; None without grad_fn."""

    if out.grad_fn is None:
        return None, None
    seen: dict[int, int] = {}
    for tensor in packed:
        seen.setdefault(tensor.data_ptr(), tensor.numel() * tensor.element_size())
    return sum(seen.values()), len(seen)


def _recount(model: Grad, x: torch.Tensor) -> tuple[dict[str, tuple[int | None, int | None]], int]:
    """Recount the three ops of ``Grad`` and the non-Parameter bytes the whole graph saves."""

    a, saved_a = _packed(lambda: model.a(x))
    r, saved_r = _packed(lambda: torch.relu(a))
    b, saved_b = _packed(lambda: model.b(r))
    bands = {
        "linear_1_1": _band(a, saved_a),
        "relu_1_2": _band(r, saved_r),
        "linear_2_3": _band(b, saved_b),
    }
    # Whole graph: every saved tensor once, Parameters and views of them excluded.
    pinned: dict[int, int] = {}
    for tensor in [*saved_a, *saved_r, *saved_b]:
        if isinstance(tensor, nn.Parameter) or isinstance(tensor._base, nn.Parameter):
            continue
        pinned.setdefault(tensor.data_ptr(), tensor.numel() * tensor.element_size())
    return bands, sum(pinned.values())


def _as_flagged(model: Grad) -> Grad:
    """A copy of ``model`` with the flag capture forces on floating Parameters."""

    flagged = copy.deepcopy(model)
    for param in flagged.parameters():
        param.requires_grad_(True)
    return flagged


def _sum(values: list[int | None]) -> int:
    return sum(value or 0 for value in values)


@pytest.mark.parametrize("variant", ["trainable", "frozen_default", "frozen_backward_ready"])
def test_autograd_saved_bytes_on_every_door(variant: str) -> None:
    torch.manual_seed(0)
    model = Grad()
    x = torch.randn(2, 4)  # the input does not require grad
    if variant != "trainable":
        model.a.requires_grad_(False)
    reference = model if variant == "frozen_backward_ready" else _as_flagged(model)
    bands, pinned = _recount(reference, x)
    if variant == "frozen_backward_ready":
        # Eager: the frozen Linear and the ReLU after it build no graph; the
        # second Linear saves only its 2x8 float input (its weight needs no
        # input gradient), 64 B.
        assert bands == {
            "linear_1_1": (None, None),
            "relu_1_2": (None, None),
            "linear_2_3": (64, 1),
        }
        capture = CaptureOptions(backward_ready=True)
    else:
        # Under the flag: addmm saves its 2x4 input (32 B); the ReLU its 2x8
        # output (64 B); the second addmm its 2x8 input and weight.t() (64 + 64).
        assert bands == {"linear_1_1": (32, 1), "relu_1_2": (64, 1), "linear_2_3": (128, 2)}
        capture = CaptureOptions()
    trace = tl.trace(model, x, capture=capture)
    try:
        for op in trace.compute_ops:
            expected_bytes, expected_count = bands[op.layer_label]
            got_bytes = None if op.autograd_memory is None else int(op.autograd_memory)
            assert (got_bytes, op.num_autograd_tensors) == (expected_bytes, expected_count)
            layer = trace.layers[op.layer_label]
            layer_bytes = None if layer.autograd_memory is None else int(layer.autograd_memory)
            assert layer_bytes == expected_bytes, op.layer_label
            assert layer.num_autograd_tensors == expected_count, op.layer_label
            total = layer.total_autograd_memory
            assert (None if total is None else int(total)) == expected_bytes, op.layer_label
        a_bytes, b_bytes = bands["linear_1_1"][0], bands["linear_2_3"][0]
        whole = _sum([band[0] for band in bands.values()])
        assert int(trace.module_calls["a:1"].autograd_memory) == (a_bytes or 0)
        assert int(trace.module_calls["b:1"].autograd_memory) == (b_bytes or 0)
        assert int(trace.module_calls["self:1"].autograd_memory) == whole
        assert int(trace.modules["a"].total_autograd_memory) == (a_bytes or 0)
        assert int(trace.modules["self"].total_autograd_memory) == whole
        assert int(trace.cost.total_autograd_memory) == whole
        assert int(trace.cost.pinned_autograd_memory) == pinned
    finally:
        trace.cleanup()
