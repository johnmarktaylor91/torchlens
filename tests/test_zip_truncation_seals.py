"""grind-r5 b7 R23: pairwise comparisons must refuse, never truncate.

``Bundle`` output comparison zipped ``output_layers`` shortest-prefix, so a
member that LOST an output compared clean (a 2-vs-1 probe passed). The fast
replay path zipped ``output_slot_ids`` against the independent persisted
``op_labels`` field with no parity contract, so a short-labelled descriptor
truncated the per-slot shape/dtype/device guards while reporting clean.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import BundleMemberError


class _OneOutputModel(nn.Module):
    """Model with a single output tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


class _TwoOutputModel(nn.Module):
    """Model returning two output tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.lin(x)
        return torch.relu(y), torch.sigmoid(y)


@pytest.mark.smoke
def test_bundle_output_delta_refuses_output_arity_mismatch() -> None:
    """A member missing an output must refuse, not report only the survivor."""

    torch.manual_seed(3)
    x = torch.randn(2, 3)
    two = tl.trace(_TwoOutputModel(), x, capture=tl.options.CaptureOptions(intervention_ready=True))
    one = tl.trace(_OneOutputModel(), x, capture=tl.options.CaptureOptions(intervention_ready=True))
    bundle = tl.bundle({"baseline": two, "lost_output": one}, baseline="baseline")

    with pytest.raises(BundleMemberError, match="structurally divergent"):
        bundle.output_delta("baseline")


def test_bundle_output_delta_still_works_on_matching_arity() -> None:
    """Equal output arity keeps the comparison green."""

    torch.manual_seed(4)
    x = torch.randn(2, 3)
    first = tl.trace(
        _TwoOutputModel(), x, capture=tl.options.CaptureOptions(intervention_ready=True)
    )
    second = tl.trace(
        _TwoOutputModel(), x, capture=tl.options.CaptureOptions(intervention_ready=True)
    )
    bundle = tl.bundle({"baseline": first, "other": second}, baseline="baseline")

    delta = bundle.output_delta("baseline")
    assert set(delta) == {"baseline", "other"}
    assert len(delta["other"]) == 2, "expected one delta row per output"


def test_fast_run_refuses_slot_label_parity_tamper(tmp_path) -> None:
    """A descriptor with fewer op_labels than output slots must refuse the
    fast iteration typed instead of silently skipping the trailing guards."""

    model = _OneOutputModel()
    x = torch.randn(2, 3)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    bundle = tmp_path / "runnable.tlspec"
    tl.save(trace, str(bundle), level="runnable", include_weights=True)
    loaded = tl.load(str(bundle))

    loaded.run(inputs=x, fast=True)  # first fast run = ordinary verified run

    descriptor = None
    for holder in vars(loaded).values():
        calls = getattr(holder, "calls", None)
        if calls is None and hasattr(holder, "descriptor"):
            calls = getattr(holder.descriptor, "calls", None)
        if calls:
            descriptor = calls
            break
    if descriptor is None:
        from torchlens._runnable_seam import runnable_trace_state

        descriptor = runnable_trace_state(loaded).descriptor.calls
    tampered = [call for call in descriptor if getattr(call, "output_slot_ids", ())]
    assert tampered, "expected at least one output-bearing call descriptor"
    object.__setattr__(tampered[-1], "op_labels", ())

    with pytest.raises(Exception, match="op label|parity|tampered"):
        loaded.run(inputs=x, fast=True)
