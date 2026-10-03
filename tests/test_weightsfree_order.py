"""Construction-order factorial (W1-ORD / D8 / defect L4), memo 8.1 item 2.

Honest twins refute each other purely from the ORDER in which the two model
objects were constructed relative to TorchLens's first wrap: whichever model
was built pre-wrap holds pre-wrap function references (``self.act =
F.gelu``) and records an adoption node where the other side records the
named op. The ``realpre`` cells must produce a TYPED PREFLIGHT REFUSAL,
never a REFUTED verdict.

True pre-wrap construction needs a FRESH process (torch wraps once per
process at the first capture), so the factorial's realpre leg runs in a
subprocess and is tiered slow; the in-process rows pin the stamp plumbing
and the discriminant.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.capture._weightsfree_admission import (
    stamp_wrap_generation,
    wrap_generation_of,
)
from torchlens.options import CaptureOptions


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def test_every_capture_is_generation_stamped() -> None:
    """W1-ORD plumbing: ordinary and structure-only captures both carry the
    session wrap-generation stamp the discharge preflight keys on."""

    ordinary = tl.trace(Toy(), torch.randn(2, 4))
    structure = tl.trace(Toy(), torch.randn(2, 4), capture=CaptureOptions(structure_only=True))
    assert wrap_generation_of(ordinary) is not None
    assert wrap_generation_of(structure) is not None
    assert wrap_generation_of(ordinary) == wrap_generation_of(structure)
    envelope = structure.structure_evidence
    assert envelope is not None
    assert envelope["wrap_generation"] == wrap_generation_of(structure)


def test_generation_mismatch_refuses_before_any_verdict() -> None:
    """The D8 condition as a stamp fact (the full factorial rides the
    subprocess leg): mismatched generations refuse typed pre-digest."""

    from torchlens.capture.structure_only import StructureOnlyCapabilityError

    structure = tl.trace(Toy(), torch.randn(2, 4), capture=CaptureOptions(structure_only=True))
    real = tl.trace(Toy(), torch.randn(2, 4))
    stamp_wrap_generation(structure, 3)
    stamp_wrap_generation(real, 4)
    with pytest.raises(StructureOnlyCapabilityError) as excinfo:
        structure.discharge_against(real)
    assert excinfo.value.fields["reason"] == "wrap_generation"


_REALPRE_SCRIPT = textwrap.dedent(
    """
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SavedRef(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)
            self.act = F.gelu  # bound PRE-WRAP in this fresh process
        def forward(self, x):
            return self.act(self.fc(x))

    torch.manual_seed(0)
    real = SavedRef()   # constructed BEFORE the first capture (pre-wrap)
    real.eval()

    import torchlens as tl
    from torchlens.options import CaptureOptions

    real_trace = tl.trace(real, torch.randn(2, 8))  # first capture wraps torch

    with torch.device("meta"):
        twin = SavedRef()  # constructed AFTER the wrap: named-op side
    twin.eval()
    meta_trace = tl.trace(
        twin, torch.empty(2, 8, device="meta"),
        capture=CaptureOptions(structure_only=True),
    )
    try:
        verdict = meta_trace.discharge_against(real_trace)
    except Exception as exc:
        code = getattr(exc, "fields", {}).get("code")
        reason = getattr(exc, "fields", {}).get("reason")
        print(f"OUTCOME refused code={code} reason={reason}")
    else:
        print(f"OUTCOME verdict={verdict.verdict.value}")
    """
)


@pytest.mark.slow
def test_realpre_cell_refuses_typed_never_refutes() -> None:
    """The factorial's realpre cell in a fresh process: the real twin built
    pre-wrap. Acceptable outcomes are the typed construction-order refusal or
    honest parity (torch build without the saved-ref asymmetry) — NEVER a
    REFUTED verdict from honest twins."""

    result = subprocess.run(
        [sys.executable, "-c", _REALPRE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    outcome = [line for line in result.stdout.splitlines() if line.startswith("OUTCOME")]
    assert outcome, result.stdout[-2000:]
    line = outcome[-1]
    assert "verdict=refuted" not in line, (
        f"honest twins REFUTED each other on construction order alone: {line}"
    )
    # Any TYPED comparability refusal is the honest outcome; measured on this
    # torch build the pre-wrap saved reference trips the shipped rescue
    # re-run on the real side, so the construction-order artifact surfaces
    # through the rescue-asymmetry disclosure.
    assert (
        "reason=construction_order" in line
        or "reason=wrap_generation" in line
        or "reason=rescue_asymmetry" in line
        or "verdict=corroborated" in line
    ), line
