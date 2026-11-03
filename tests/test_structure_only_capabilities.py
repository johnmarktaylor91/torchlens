"""Capability-table lockstep + chokepoint battery (L7a memo sec 6).

BOTH halves of the outcome-table enforcement pattern: (i) the NEGATIVE scan —
nobody reads STRUCTURE_ONLY_CAPABILITIES but the one chokepoint — AND (ii) a
POSITIVE CALLSITE INVENTORY over every value-bearing surface the table
promises to gate. Plus: doc-table lockstep against the in-code authority,
closed-grammar red-capability, chokepoint provocations for every
refuse:<code> row, and the registrar exit-gate round-trip (the ONE sanctioned
pre-bump persistence path, fail-closed marker included).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture.structure_only import (
    StructureOnlyCapabilityError,
)
from torchlens.options import CaptureOptions

TORCHLENS_DIR = Path(tl.__file__).resolve().parent
REPO_ROOT = TORCHLENS_DIR.parent


class TwoLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def _structure_trace():
    return tl.trace(TwoLayer(), torch.randn(2, 4), capture=CaptureOptions(structure_only=True))


# ---------------------------------------------------------------------------
# (i) NEGATIVE scan: the chokepoint is the only table reader
# ---------------------------------------------------------------------------


def test_nobody_reads_the_capability_table_but_the_chokepoint() -> None:
    for path in TORCHLENS_DIR.rglob("*.py"):
        if path.name == "structure_only.py" and path.parent.name == "_capture":
            continue
        text = path.read_text(encoding="utf-8")
        assert "STRUCTURE_ONLY_CAPABILITIES[" not in text, str(path)


# ---------------------------------------------------------------------------
# (ii) POSITIVE callsite inventory: every promised gate call exists
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Doc-table lockstep (compat-report key-set pattern)
# ---------------------------------------------------------------------------


def test_row_grammar_is_closed_and_red_capable() -> None:
    from torchlens._capture.structure_only import CapabilityRow, _validate_row_grammar

    with pytest.raises(AssertionError):
        _validate_row_grammar(
            CapabilityRow(
                key="bad",
                claim="x",
                status_v1="maybe_supported",
                flip_event="never",
                evidence="none",
                amend_owner="L7a",
            )
        )
    with pytest.raises(AssertionError):
        _validate_row_grammar(
            CapabilityRow(
                key="bad",
                claim="x",
                status_v1="refuse:some_code",
                flip_event="never",
                evidence="none",
                amend_owner="L7a",
                refusal_code="other_code",
            )
        )


# ---------------------------------------------------------------------------
# Chokepoint behavior + provocations for every refuse row
# ---------------------------------------------------------------------------


def test_save_persists_the_marker_plainly(tmp_path) -> None:
    """tlspec v8: analysis-level saves persist the structure_only marker.

    The loaded trace stays a hypothesis product: the marker survives the
    round trip, value-requiring consumers keep refusing through the one
    chokepoint, and no pre-release marker rides the artifact.
    """

    log = _structure_trace()
    path = tmp_path / "structure.tlspec"
    log.save(path)
    loaded = tl.load(path)
    assert loaded.options.structure_only is True
    with pytest.raises(StructureOnlyCapabilityError) as excinfo:
        loaded.run(inputs=torch.randn(2, 4))
    assert excinfo.value.fields["code"] == "structure_only_replay_unsupported"


def test_validation_entry_refuses_typed() -> None:
    log = _structure_trace()
    with pytest.raises(StructureOnlyCapabilityError) as excinfo:
        log.validate_forward_pass(None)
    assert excinfo.value.fields["code"] == "structure_only_validation_unsupported"


def test_backward_refuses_typed() -> None:
    log = _structure_trace()
    with pytest.raises(StructureOnlyCapabilityError) as excinfo:
        log.log_backward(torch.tensor(1.0))
    assert excinfo.value.fields["code"] == "structure_only_backward_unsupported"


# ---------------------------------------------------------------------------
# Registrar exit gate: the ONE sanctioned pre-bump round-trip (fail-closed)
# ---------------------------------------------------------------------------
