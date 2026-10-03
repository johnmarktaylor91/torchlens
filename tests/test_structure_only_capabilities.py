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

import re
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.prerelease import activate_prerelease_fields
from torchlens.capture.structure_only import (
    STRUCTURE_ONLY_CAPABILITIES,
    StructureOnlyCapabilityError,
    require_structure_only_capability,
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
        if path.name == "structure_only.py" and path.parent.name == "capture":
            continue
        text = path.read_text(encoding="utf-8")
        assert "STRUCTURE_ONLY_CAPABILITIES[" not in text, str(path)


# ---------------------------------------------------------------------------
# (ii) POSITIVE callsite inventory: every promised gate call exists
# ---------------------------------------------------------------------------


def test_gated_entries_consult_the_structure_only_chokepoint() -> None:
    """Source lockstep mirroring test_capture_outcome_matrix's inventory: a
    refactor that drops a structure-only gate call from a value-bearing
    surface fails here."""

    validation_source = (TORCHLENS_DIR / "data_classes" / "_trace_validation.py").read_text(
        encoding="utf-8"
    )
    gate_calls = re.findall(
        r'require_structure_only_capability\(self, "([a-z_]+)"\)', validation_source
    )
    # One sibling call per capture-outcome gate site: save_new_outs, push,
    # push_from, run (entry pre-gate + loaded-sparse + halted-analysis +
    # unified-live + legacy-live), log_backward, recording_backward,
    # validate_forward_pass, check_metadata_invariants.
    assert sorted(gate_calls) == sorted(
        [
            "live_replay",  # save_new_outs
            "live_replay",  # push
            "live_replay",  # push_from
            "live_replay",  # run(): entry pre-gate
            "live_replay",  # run(): loaded sparse provider (values still required)
            "live_replay",  # run(): halted analysis-load pre-gate
            "live_replay",  # run(): unified live provider
            "live_replay",  # run(): legacy rerun surface
            "backward_grads",  # log_backward
            "backward_grads",  # recording_backward
            "validation_entry",  # validate_forward_pass
            "validation_entry",  # check_metadata_invariants
        ]
    )
    bundle_source = (TORCHLENS_DIR / "_io" / "bundle.py").read_text(encoding="utf-8")
    assert 'require_structure_only_capability(trace, "save_analysis_artifact")' in bundle_source
    assert 'require_structure_only_capability(trace, "save_runnable")' in bundle_source


# ---------------------------------------------------------------------------
# Doc-table lockstep (compat-report key-set pattern)
# ---------------------------------------------------------------------------


def test_doc_table_locksteps_with_the_in_code_authority() -> None:
    doc = (REPO_ROOT / "docs" / "reference" / "structure_only_capabilities.md").read_text(
        encoding="utf-8"
    )
    doc_keys = set(re.findall(r"^\| `([a-z_0-9]+)` \|", doc, re.M))
    assert doc_keys == set(STRUCTURE_ONLY_CAPABILITIES), (
        "doc table keys drifted from STRUCTURE_ONLY_CAPABILITIES — update "
        "table + code + tests in one change"
    )
    for row in STRUCTURE_ONLY_CAPABILITIES.values():
        assert f"`{row.status_v1}`" in doc, row.key
        assert row.flip_event in doc, row.key
        if row.refusal_code is not None:
            assert f"`{row.refusal_code}`" in doc, row.key
    # The no-over-claiming rule is printed in the file header.
    assert "never promises for a wave it hasn't shipped" in doc


def test_row_grammar_is_closed_and_red_capable() -> None:
    from torchlens.capture.structure_only import CapabilityRow, _validate_row_grammar

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


def test_chokepoint_is_a_noop_on_ordinary_traces() -> None:
    log = tl.trace(TwoLayer(), torch.randn(2, 4))
    assert require_structure_only_capability(log, "save_analysis_artifact") is None
    assert require_structure_only_capability(log, "live_replay") is None


def test_save_persists_the_marker_plainly(tmp_path) -> None:
    """tlspec v8: analysis-level saves persist the structure_only marker.

    The loaded trace stays a hypothesis product: the marker survives the
    round trip, value-requiring consumers keep refusing through the one
    chokepoint, and no pre-release marker rides the artifact.
    """

    log = _structure_trace()
    path = tmp_path / "structure.tlspec"
    tl.save(log, path)
    loaded = tl.load(path)
    assert loaded.structure_only is True
    with pytest.raises(StructureOnlyCapabilityError) as excinfo:
        loaded.run(inputs=torch.randn(2, 4))
    assert excinfo.value.fields["code"] == "structure_only_replay_unsupported"


def test_runnable_save_row_refuses_typed() -> None:
    log = _structure_trace()
    with pytest.raises(StructureOnlyCapabilityError) as excinfo:
        require_structure_only_capability(log, "save_runnable")
    assert excinfo.value.fields["code"] == "structure_only_runnable_unsupported"


def test_live_replay_refuses_typed() -> None:
    log = _structure_trace()
    with pytest.raises(StructureOnlyCapabilityError) as excinfo:
        log.run(inputs=torch.randn(2, 4))
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


def test_episode_composition_row_refuses_typed() -> None:
    log = _structure_trace()
    with pytest.raises(StructureOnlyCapabilityError) as excinfo:
        require_structure_only_capability(log, "episode_composition")
    assert excinfo.value.fields["code"] == "structure_only_episode_unsupported"


# ---------------------------------------------------------------------------
# Registrar exit gate: the ONE sanctioned pre-bump round-trip (fail-closed)
# ---------------------------------------------------------------------------


def test_registrar_switch_round_trips_the_marker_and_load_fails_closed(tmp_path) -> None:
    log = _structure_trace()
    artifact = tmp_path / "structure.tlspec"
    with activate_prerelease_fields():
        tl.save(log, artifact)
        reloaded = tl.load(artifact)
        assert reloaded.structure_only is True
    # Outside the switch the stamped artifact can NEVER pass as a real one.
    from torchlens._io import PreReleaseArtifactError

    with pytest.raises(PreReleaseArtifactError):
        tl.load(artifact)
