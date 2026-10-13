"""Agent JSON carries the structure-only evidence facts (R2-11 export parity).

``Trace.to_agent_json`` assembles its ``capture`` block from the shared
honesty helpers; a structure-only dump must carry the claim-status ladder and
the evidence envelope the shared fact block carries, while a plain dump's
``capture`` block keeps exactly its previous keys.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens._capture_honesty import capture_honesty_facts
from torchlens.options import CaptureOptions

#: The plain capture block's keys before the structure-only facts joined it.
_PLAIN_CAPTURE_KEYS = frozenset(
    {
        "capture_status",
        "capture_verified",
        "capture_verification_reason",
        "rescue_rerun",
        "backend",
        "model_class",
        "structure_only",
        "grouping",
        "has_backward_pass",
        "device_summary",
        "poisoned",
    }
)

_STRUCTURE_KEYS = ("claim_status", "values_available", "substrate", "structure_evidence")


class _Tiny(nn.Module):
    """Linear then relu."""

    def __init__(self) -> None:
        """Build one linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear then relu."""

        return torch.relu(self.fc(x))


def _structure(model: nn.Module) -> tl.Trace:
    """A structure-only capture of ``model``."""

    return tl.trace(model, torch.randn(2, 4), capture=CaptureOptions(structure_only=True))


def test_structure_only_dump_carries_the_evidence_facts() -> None:
    """claim_status, values_available, substrate and the envelope ride the capture block."""

    trace = _structure(_Tiny())
    capture = trace.to_agent_json(max_ops=1)["capture"]
    facts = capture_honesty_facts(trace)
    assert capture["structure_only"] is True
    for key in _STRUCTURE_KEYS:
        assert key in capture, key
        assert capture[key] == facts[key], key
    assert capture["claim_status"] == "hypothesis"
    assert capture["values_available"] is False
    assert "discharge" not in capture


def test_discharged_dump_carries_the_discharge() -> None:
    """After a discharge the dump names its verdict and the corroborated ladder rung."""

    torch.manual_seed(0)
    model = _Tiny()
    trace = _structure(model)
    trace.discharge_against(tl.trace(model, torch.randn(2, 4)))
    capture = trace.to_agent_json(max_ops=1)["capture"]
    facts = capture_honesty_facts(trace)
    assert capture["claim_status"] == "corroborated"
    assert capture["discharge"] == facts["discharge"]
    assert capture["discharge"]["verdict"] == "corroborated"


def test_plain_dump_capture_block_is_unchanged() -> None:
    """A plain capture's block gains no structure-only keys."""

    trace = tl.trace(_Tiny(), torch.randn(2, 4))
    capture = trace.to_agent_json(max_ops=1)["capture"]
    assert set(capture) == _PLAIN_CAPTURE_KEYS
    assert capture["structure_only"] is False
