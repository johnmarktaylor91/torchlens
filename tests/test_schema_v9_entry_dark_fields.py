"""tlspec v9 entry-dark field slots (C07 coordinated schema write).

Three declared-now, written-later field families, so their Phase-3 writers
need no further version bump:

- ``Op.injection_provenance`` (surgery memo 3.5 / items 8-9, 12; F01 writes):
  the injected-op durable structural key. ``None`` on every model op.
- ``Trace.source_snapshots`` (convert memo item 24; F30 writes): the
  per-(path, digest) source snapshot table. Empty until then.
- ``Trace.structure_evidence`` (weightsfree memo section 5 / item 14; F33
  writes): the immutable structure-only evidence envelope. ``None`` until
  then, and only a structure-only capture may carry one.

Every present value validates fail-closed at the load boundary; the defaults
round-trip byte-plainly on ordinary captures.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import TorchLensIOError

pytestmark = [pytest.mark.smoke]

_DIGEST = hashlib.sha256(b"schema-v9-entry-dark").hexdigest()


def _traced():
    torch.manual_seed(0)
    return tl.trace(nn.Sequential(nn.Linear(4, 3), nn.ReLU()), torch.randn(2, 4))


def _roundtrip(trace, tmp_path, name):
    path = tmp_path / f"{name}.tlspec"
    tl.save(trace, path)
    return tl.load(path)


def _injection_record(**overrides):
    record = {
        "host_site_key": "s1|linear/Linear/out/0",
        "spec_rule_id": "rule-1",
        "host_pass": 1,
        "firing_index": 0,
        "nesting_path": [0],
        "local_op_ordinal": 0,
        "output_slot": 0,
    }
    record.update(overrides)
    return record


def _evidence_envelope(**overrides):
    envelope = {
        "capture_mode": "structure_only",
        "substrate": "meta",
        "values_available": False,
        "outcome": "complete",
        "factory_device_policy": "torchlens_owned",
        "ambient_mode_present": False,
        "wrap_generation": 1,
        "input_plan": {"source": "user", "synthesized": [], "omitted": []},
        "claims": {
            "graph_structure": "observed_canonical",
            "shapes_dtypes": "hypothesis",
            "flops_geometry_bytes": "hypothesis_estimate",
            "declared_buffer_mutations": "hypothesis",
            "measured_values": "unavailable",
            "timing_allocator_memory": "unavailable",
            "declared_branch_assumptions": "none",
        },
        "discharge": "absent",
    }
    envelope.update(overrides)
    return envelope


def test_defaults_round_trip_on_ordinary_captures(tmp_path):
    trace = _traced()
    assert all(op.injection_provenance is None for op in trace.ops)
    assert trace.source_snapshots == []
    assert trace.structure_evidence is None
    loaded = _roundtrip(trace, tmp_path, "defaults")
    assert all(op.injection_provenance is None for op in loaded.ops)
    assert loaded.source_snapshots == []
    assert loaded.structure_evidence is None


def test_injection_provenance_round_trips_and_validates(tmp_path):
    trace = _traced()
    trace.ops[-1].injection_provenance = _injection_record()
    loaded = _roundtrip(trace, tmp_path, "injection_ok")
    assert loaded.ops[-1].injection_provenance == _injection_record()


@pytest.mark.parametrize(
    "mutation",
    [
        {"host_site_key": ""},
        {"spec_rule_id": 3},
        {"host_pass": 0},
        {"firing_index": -1},
        {"nesting_path": [0, -2]},
        {"nesting_path": "0"},
        {"local_op_ordinal": True},
        {"output_slot": None},
        {"surprise": 1},
    ],
)
def test_malformed_injection_provenance_refuses_at_load(tmp_path, mutation):
    trace = _traced()
    trace.ops[-1].injection_provenance = _injection_record(**mutation)
    with pytest.raises(TorchLensIOError) as excinfo:
        _roundtrip(trace, tmp_path, "injection_bad")
    assert (
        getattr(excinfo.value, "fields", {}).get("code") == "artifact_injection_provenance_invalid"
    )


def test_source_snapshots_round_trip_and_validate(tmp_path):
    rows = [
        {"path": "model.py", "digest": _DIGEST, "text": "def forward(self, x): ..."},
        {"path": "frame_filtered.py", "digest": _DIGEST, "text": None},
    ]
    trace = _traced()
    trace.source_snapshots = rows
    loaded = _roundtrip(trace, tmp_path, "snapshots_ok")
    assert loaded.source_snapshots == rows


@pytest.mark.parametrize(
    "rows",
    [
        "not-a-list",
        [{"path": "", "digest": _DIGEST, "text": None}],
        [{"path": "a.py", "digest": "nope", "text": None}],
        [{"path": "a.py", "digest": _DIGEST, "text": 7}],
        [{"path": "a.py", "digest": _DIGEST}],
        [
            {"path": "a.py", "digest": _DIGEST, "text": None},
            {"path": "a.py", "digest": _DIGEST, "text": "dup"},
        ],
    ],
)
def test_malformed_source_snapshots_refuse_at_load(tmp_path, rows):
    trace = _traced()
    trace.source_snapshots = rows
    with pytest.raises(TorchLensIOError) as excinfo:
        _roundtrip(trace, tmp_path, "snapshots_bad")
    assert getattr(excinfo.value, "fields", {}).get("code") == "artifact_source_snapshots_invalid"


def test_structure_evidence_requires_the_structure_only_marker(tmp_path):
    trace = _traced()
    trace.structure_evidence = _evidence_envelope()
    with pytest.raises(TorchLensIOError) as excinfo:
        _roundtrip(trace, tmp_path, "evidence_unmarked")
    assert getattr(excinfo.value, "fields", {}).get("code") == "artifact_structure_evidence_invalid"
    assert "structure_only marker is False" in str(excinfo.value)


def test_structure_evidence_round_trips_on_structure_only_captures(tmp_path):
    trace = tl.trace(
        nn.Sequential(nn.Linear(4, 3), nn.ReLU()),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(structure_only=True),
    )
    trace.structure_evidence = _evidence_envelope()
    loaded = _roundtrip(trace, tmp_path, "evidence_ok")
    assert loaded.structure_evidence == _evidence_envelope()


@pytest.mark.parametrize(
    "mutation",
    [
        {"capture_mode": "eager"},
        {"substrate": "quantum"},
        {"values_available": True},
        {"outcome": "great"},
        {"factory_device_policy": "user_owned"},
        {"ambient_mode_present": "yes"},
        {"wrap_generation": -1},
        {"input_plan": "plan"},
        {"discharge": 3},
        {"surprise": 1},
    ],
)
def test_malformed_structure_evidence_refuses_at_load(tmp_path, mutation):
    trace = tl.trace(
        nn.Sequential(nn.Linear(4, 3), nn.ReLU()),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(structure_only=True),
    )
    trace.structure_evidence = _evidence_envelope(**mutation)
    with pytest.raises(TorchLensIOError) as excinfo:
        _roundtrip(trace, tmp_path, "evidence_bad")
    assert getattr(excinfo.value, "fields", {}).get("code") == "artifact_structure_evidence_invalid"


def test_malformed_evidence_claims_refuse_at_load(tmp_path):
    trace = tl.trace(
        nn.Sequential(nn.Linear(4, 3), nn.ReLU()),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(structure_only=True),
    )
    claims = dict(_evidence_envelope()["claims"], measured_values="available")
    trace.structure_evidence = _evidence_envelope(claims=claims)
    with pytest.raises(TorchLensIOError) as excinfo:
        _roundtrip(trace, tmp_path, "evidence_claims_bad")
    assert getattr(excinfo.value, "fields", {}).get("code") == "artifact_structure_evidence_invalid"
