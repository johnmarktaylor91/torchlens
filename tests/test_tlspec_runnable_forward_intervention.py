"""Round-11 F5 regression: forward-modifying interventions must not lie.

A forward-modifying (value-override) intervention makes the captured forward diverge
from the recorded sparse DAG (which stores only the original op recipe). A runnable
replay could only recompute the un-intervened value, so since deephunt F1 the producer
refuses ``level="runnable"`` at SAVE time with the named
``user_intervention_unsupported`` diagnostic (historically the save succeeded and
the run ceilinged UNVERIFIABLE + NOT_APPLICABLE with no findable cause).
Observe-only/backward interventions and plain captures are unchanged and still VERIFY.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._runnable import NumericAttestationStatus, PathFaithfulness
from torchlens.errors import RunnablePreflightError
from torchlens.intervention.errors import EngineDispatchError
from torchlens.options import CaptureOptions


class _AblationModel(nn.Module):
    """Parameterized graph whose ReLU output can be forward-intervened."""

    def __init__(self) -> None:
        """Initialize a deterministic linear layer."""

        torch.manual_seed(7)
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return a scaled ReLU activation."""

        return torch.relu(self.linear(value)) * 2.0


def _capture(model: nn.Module, value: torch.Tensor, **kwargs: object) -> tl.Trace:
    """Capture an intervention-ready trace with container structure."""

    return tl.trace(
        model,
        value,
        capture=CaptureOptions(
            replayable=True,
            capture_container_structure=True,
            cache=False,
        ),
        save="all",
        **kwargs,
    )


# Kept: a sole mutation kill; the catch of its regression fault.
def test_forward_override_intervention_refuses_runnable_save(tmp_path: Path) -> None:
    """A zero-ablated capture refuses runnable save with a named diagnostic.

    REVIEWED REBASELINE (deephunt F1): this test previously pinned the honest
    run-time ceiling (save succeeds, replay of the UN-ablated DAG reports
    UNVERIFIABLE + NOT_APPLICABLE). The disclosure gap -- a replay output from a
    different computation than the artifact's provenance, with no diagnostic
    naming the dropped intervention -- is now closed EARLIER, at save time, so
    the artifact is never produced. The never-false-VERIFIED contract is
    unchanged; the honesty moved from an unfindable ceiling to a typed refusal.
    """

    torch.manual_seed(0)
    value = torch.randn(2, 4)
    trace = _capture(
        _AblationModel().eval(),
        value,
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )
    path = tmp_path / "ablated.tlspec"
    with pytest.raises(RunnablePreflightError) as excinfo:
        trace.save(path, level="runnable", include_weights=True)
    assert "user_intervention_unsupported" in str(excinfo.value.fields.get("diagnostics"))


# Kept: sole branch coverage of _capture/prepare/resolve.py, _ir/selector_eval.py, _selection/resolver.py and more; a sole mutation kill; named by a manifest, the interaction matrix or the validation mutation suite.
def test_backward_grad_intervention_still_verifies(tmp_path: Path) -> None:
    """A backward/grad intervention leaves the forward reproducible -> VERIFIED."""

    torch.manual_seed(0)
    value = torch.randn(2, 4)
    trace = _capture(
        _AblationModel().eval(),
        value,
        intervene=tl.when(tl.grad(tl.func("relu")), tl.scale(2.0)),
    )
    path = tmp_path / "grad.tlspec"
    trace.save(path, level="runnable", include_weights=True, include_activations=True)

    loaded = tl.load(path)
    # Format 3 persists the recipe and the reader restages it, so a fresh verified run
    # refuses rather than silently dropping the recorded grad hook; detach it explicitly.
    with pytest.raises(EngineDispatchError, match="staged intervention spec"):
        loaded.run(inputs=value, seed=0)
    loaded._clear_hooks()
    result = loaded.run(inputs=value, seed=0)

    # The grad intervention never modified the forward output, so the sparse DAG
    # reproduces it byte-for-byte: an op-representable intervention still VERIFIES.
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
