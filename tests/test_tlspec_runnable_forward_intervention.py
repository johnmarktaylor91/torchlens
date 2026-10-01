"""Round-11 F5 regression: forward-modifying interventions must not lie.

A forward-modifying (value-override) intervention makes the captured forward diverge
from the recorded sparse DAG (which stores only the original op recipe). A runnable
replay could only recompute the un-intervened value, so since deephunt F1 the producer
refuses ``level="runnable"`` at SAVE time with the named
``user_intervention_not_replayable`` diagnostic (historically the save succeeded and
the run ceilinged UNVERIFIABLE + NOT_APPLICABLE with no findable cause).
Observe-only/backward interventions and plain captures are unchanged and still VERIFY.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import RunnablePreflightError
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness


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
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
            layers_to_save="all",
        ),
        **kwargs,
    )


@pytest.mark.smoke
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
        tl.save(trace, path, level="runnable", include_weights=True)
    assert "user_intervention_not_replayable" in str(excinfo.value.fields.get("diagnostics"))


def test_forward_override_with_activations_also_refuses(tmp_path: Path) -> None:
    """The refusal is independent of the activation-archive payload flags.

    REVIEWED REBASELINE (deephunt F1): previously asserted the saved artifact's
    run reported UNVERIFIABLE + NOT_APPLICABLE without a contradicting
    attestation error; the save itself now refuses first (same rationale as
    above).
    """

    torch.manual_seed(0)
    value = torch.randn(2, 4)
    trace = _capture(
        _AblationModel().eval(),
        value,
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )
    path = tmp_path / "ablated-with-acts.tlspec"
    with pytest.raises(RunnablePreflightError) as excinfo:
        tl.save(
            trace,
            path,
            level="runnable",
            include_weights=True,
            include_activations=True,
        )
    assert "user_intervention_not_replayable" in str(excinfo.value.fields.get("diagnostics"))
    assert not path.exists()


@pytest.mark.smoke
def test_plain_capture_still_verifies(tmp_path: Path) -> None:
    """A non-intervened capture is unchanged: VERIFIED (+ ATTESTED with activations)."""

    torch.manual_seed(0)
    value = torch.randn(2, 4)
    trace = _capture(_AblationModel().eval(), value)
    path = tmp_path / "plain.tlspec"
    tl.save(trace, path, level="runnable", include_weights=True, include_activations=True)

    result = tl.load(path).run(inputs=value, seed=0)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


def test_backward_grad_intervention_still_verifies(tmp_path: Path) -> None:
    """A backward/grad intervention leaves the forward reproducible -> VERIFIED."""

    torch.manual_seed(0)
    value = torch.randn(2, 4)
    trace = _capture(
        _AblationModel().eval(),
        value,
        intervene=tl.when(tl.func("relu"), tl.grad_scale(2.0)),
    )
    path = tmp_path / "grad.tlspec"
    tl.save(trace, path, level="runnable", include_weights=True, include_activations=True)

    result = tl.load(path).run(inputs=value, seed=0)

    # The grad intervention never modified the forward output, so the sparse DAG
    # reproduces it byte-for-byte: an op-representable intervention still VERIFIES.
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
