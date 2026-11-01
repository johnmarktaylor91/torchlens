"""Hand-counted oracles for ``CaptureFacts.advisories`` and ``Trace.cost.total_backward_memory``.

``advisories`` is the number of capture-advisory records on the trace, one per
advisory kind the capture raised (``Trace.annotations["capture_advisories"]``,
each row carrying how many instances it aggregates). ``total_backward_memory``
is the bytes of the gradient payloads the trace retains, counted once per
storage: a payload that is a view keeps its whole base storage alive, and an
expanded view holds less than its logical size.

The storage probe behind the second oracle (Grad, batch 3, two backward passes):
raw saved gradients are always dense copies, so with plain ``save_grads`` no
two payloads share storage; a ``grad_transform`` that returns autograd's own
gradient (or a slice of it) retains autograd's storage, and the gradient that
``sum`` sends back to its input is an expanded 0-d tensor (stride 0, a 4-byte
storage) where its logical size is 24 bytes.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from tests.oracles.test_legacy_graph_counts import Grad
from torchlens.errors import ScalarEscapeWarning
from torchlens.options import CaptureOptions, SaveOptions

G = torch.randn(4)  # a forward-global tensor a module consumes directly
H = torch.randn(4)  # a forward-global tensor only a scalar read consumes


class Advise(nn.Module):
    """Plants three advisory kinds with known instance counts."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        a = h.sum().item()  # scalar escape 1
        b = float(h.mean())  # scalar escape 2
        y = self.fc(G)  # module entry adopts a forward-global tensor: 1 adoption
        g = H.sum().item()  # scalar escape 3; H.sum() is a source-less orphan: 1
        return h * (a + b + g) + y


def test_capture_advisories_match_the_planted_kinds_and_counts() -> None:
    torch.manual_seed(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(Advise().eval(), torch.randn(2, 4))
    try:
        assert any(issubclass(item.category, ScalarEscapeWarning) for item in caught)
        rows = trace.annotations["capture_advisories"]
        assert sorted((row["kind"], row["count"]) for row in rows) == [
            ("module_boundary_adoption", 1),
            ("orphan_unattributed_tensor_args", 1),
            ("scalar_escape", 3),
        ]
        assert trace.summary().capture.advisories == 3
    finally:
        trace.cleanup()
    torch.manual_seed(0)
    clean = tl.trace(Grad().eval(), torch.randn(3, 4))
    try:
        assert not clean.annotations.get("capture_advisories")
        assert clean.summary().capture.advisories == 0
    finally:
        clean.cleanup()


def _two_pass_trace(transform: Callable[[torch.Tensor], torch.Tensor] | None) -> tl.Trace:
    torch.manual_seed(0)
    save = SaveOptions(grad_transform=transform) if transform is not None else None
    trace = tl.trace(
        Grad().train(), torch.randn(3, 4), capture=CaptureOptions(save_grads=True), save=save
    )
    loss = trace[trace.output_layers[0]].out.sum()
    trace.backward(loss, retain_graph=True)
    trace.backward(loss)
    return trace


def test_sum_backward_sends_an_expanded_zero_dim_gradient() -> None:
    # Raw torch, no TorchLens: the probe's storage finding for the b output.
    torch.manual_seed(0)
    out = Grad()(torch.randn(3, 4))
    seen: list[torch.Tensor] = []
    out.register_hook(seen.append)  # a tensor hook sees autograd's own gradient
    out.sum().backward()
    (grad,) = seen
    assert grad.stride() == (0, 0)
    assert grad.untyped_storage().nbytes() == 4
    assert grad.nelement() * grad.element_size() == 24


# Per pass, the retained gradients belong to a's output (3x8 float32, 96 B), the
# relu output (96 B) and b's output (3x2, 24 B); two passes double each sum.
# Raw payloads are dense copies: 96 + 96 + 24 = 216 B per pass. The identity
# transform also retains autograd's own gradients: 96 + 96 + 4 (b's expanded
# gradient) = 196 B; the slicing transform keeps those same three storages
# alive (96 + 96 + 4); a cloning transform retains dense copies (216 B).
@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        (None, 2 * 216),
        (lambda grad: grad, 2 * (216 + 196)),
        (lambda grad: grad[..., :1], 2 * (216 + 196)),
        (lambda grad: grad.clone(), 2 * (216 + 216)),
    ],
    ids=["raw", "identity", "slice", "clone"],
)
def test_total_backward_memory_counts_unique_storage_bytes(
    transform: Callable[[torch.Tensor], torch.Tensor] | None, expected: int
) -> None:
    trace = _two_pass_trace(transform)
    try:
        assert int(trace.cost.total_backward_memory) == expected
    finally:
        trace.cleanup()
