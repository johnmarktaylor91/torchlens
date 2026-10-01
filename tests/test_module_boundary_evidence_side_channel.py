"""Module-boundary intervention evidence must survive attr-rejecting tensors.

grind-p3-fixplan T3 (fix-patching lane, HIGH): the module-boundary live-hook
dispatcher attached its fire results and replaced-parent provenance to the
replacement tensor with a bare ``setattr`` under ``except: pass``. A
replacement tensor that rejects dynamic attributes silently DISCARDED the
evidence: the fresh value was misclassified as an ``internal_source`` op with
no intervention provenance and no parents. The evidence now routes through the
op-level storage-owned side table (fire results) and a trace-scoped identity
table (parent labels), so classification no longer depends on stamping
attributes onto the replacement tensor.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch._modes import SubclassConstructionUnderDispatchModeError
from torchlens.options import CaptureOptions
from torchlens.utils._torch_compat import HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE


class RejectingTensor(torch.Tensor):
    """Tensor subclass that rejects TorchLens intervention-evidence attributes."""

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject only the transient intervention-evidence attributes."""

        if name in ("_tl_live_fire_results", "_tl_module_intervention_parent_labels"):
            raise RuntimeError("dynamic attributes disabled")
        super().__setattr__(name, value)


class TwoLinear(nn.Module):
    """Two-linear model with a hookable interior module boundary."""

    def __init__(self) -> None:
        """Initialize the two linear maps."""

        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run fc1 then fc2."""

        return self.fc2(self.fc1(x))


def _rejecting_replace_hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
    """Replace the module output with a fresh attr-rejecting tensor."""

    del hook
    return torch.zeros_like(out).as_subclass(RejectingTensor)


def _plain_replace_hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
    """Replace the module output with a fresh ordinary tensor."""

    del hook
    return torch.zeros_like(out)


def _boundary_capture(hook: Any) -> Any:
    """Capture ``TwoLinear`` with a module-boundary replacement hook on fc1."""

    return tl.trace(
        TwoLinear(),
        torch.randn(2, 4),
        capture=CaptureOptions(
            intervention_ready=True,
            hooks={tl.module("fc1"): hook},
        ),
    )


@pytest.mark.smoke
def test_boundary_evidence_survives_attr_rejecting_replacement() -> None:
    """An attr-rejecting replacement keeps its intervention provenance.

    Pre-fix, the silent ``setattr`` failure dropped the fire results AND the
    replaced-parent labels, so the fresh tensor was minted as an
    ``internal_source`` with no interventions and no parents.
    """

    if not HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE:
        # Floor-torch capability gap (torch 2.1/2.2): constructing the
        # attr-rejecting RejectingTensor subclass inside the hook, while an
        # intervention-ready capture's TorchLens dispatch mode is active,
        # refuses typed instead of crashing on torch's own "already
        # associated to a python object" RuntimeError. See
        # SubclassConstructionUnderDispatchModeError.
        with pytest.raises(SubclassConstructionUnderDispatchModeError):
            _boundary_capture(_rejecting_replace_hook)
        return

    log = _boundary_capture(_rejecting_replace_hook)

    boundary = next(
        op for op in log.layer_list if str(op.layer_label).startswith("interventionreplacement")
    )
    assert boundary.interventions
    assert boundary.intervention_replaced
    assert any(record.replaced for record in boundary.interventions)
    # Replaced-parent provenance (fc1's real output) survived the rejection.
    assert tuple(boundary.parents)
    # The replacement is never laundered into an untraceable internal source.
    assert not any(bool(getattr(op, "is_internal_source", False)) for op in log.layer_list)


@pytest.mark.smoke
def test_rejecting_and_plain_replacements_classify_identically() -> None:
    """Attr-rejecting and ordinary replacement tensors produce the same structure."""

    if not HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE:
        # Same floor-torch gap as test_boundary_evidence_survives_attr_rejecting_replacement.
        with pytest.raises(SubclassConstructionUnderDispatchModeError):
            _boundary_capture(_rejecting_replace_hook)
        return

    rejecting_log = _boundary_capture(_rejecting_replace_hook)
    plain_log = _boundary_capture(_plain_replace_hook)

    def _shape(log: Any) -> list[tuple[str, bool, bool]]:
        return [
            (
                str(op.layer_label),
                bool(op.interventions),
                bool(getattr(op, "intervention_replaced", False)),
            )
            for op in log.layer_list
        ]

    assert _shape(rejecting_log) == _shape(plain_log)
