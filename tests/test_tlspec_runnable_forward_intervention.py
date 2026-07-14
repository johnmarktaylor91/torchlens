"""Round-11 F5 regression: forward-modifying interventions must not lie.

A forward-modifying (value-override) intervention makes the captured forward diverge
from the recorded sparse DAG (which stores only the original op recipe). The runnable
replay recomputes the un-intervened value, so the run must report UNVERIFIABLE +
NOT_APPLICABLE (never a false VERIFIED, never a contradicting NumericAttestationError)
so both honesty layers AGREE. Observe-only/backward interventions and plain captures
are unchanged and still VERIFY.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
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
        layers_to_save="all",
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
        **kwargs,
    )


@pytest.mark.smoke
def test_forward_override_intervention_is_unverifiable_not_verified(tmp_path: Path) -> None:
    """A zero-ablated capture never reports a false VERIFIED sparse-only."""

    torch.manual_seed(0)
    value = torch.randn(2, 4)
    trace = _capture(
        _AblationModel().eval(),
        value,
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )
    path = tmp_path / "ablated.tlspec"
    tl.save(trace, path, level="runnable", include_weights=True)

    result = tl.load(path).run(inputs=value, seed=0)

    # The DAG replay recomputes the UN-ablated value, so the honest ceiling is
    # UNVERIFIABLE + NOT_APPLICABLE -- the two layers must agree, never VERIFIED.
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_forward_override_with_activations_does_not_contradict(tmp_path: Path) -> None:
    """Archived activations must not raise a contradicting attestation error."""

    torch.manual_seed(0)
    value = torch.randn(2, 4)
    trace = _capture(
        _AblationModel().eval(),
        value,
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )
    path = tmp_path / "ablated-with-acts.tlspec"
    tl.save(
        trace,
        path,
        level="runnable",
        include_weights=True,
        include_activations=True,
    )

    # Previously this raised NumericAttestationError while faithfulness said VERIFIED.
    result = tl.load(path).run(inputs=value, seed=0)

    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


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
