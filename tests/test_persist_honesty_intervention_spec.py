"""Persistence honesty: the intervention-spec door validates, gates, and keeps
edge addresses.

WT1 A-IV item 20 (lane A08): the intervention-spec load door silently
discarded ``FireRecord.edge_address``; ``save_intervention`` had no settled-
outcome gate; and ``spec.json`` deserialized with no structural validation (3.0: ``intervention.json``)
(a missing ``intervention_spec`` key died with a bare ``KeyError``). Deeper
per-record staging refusals continue in lane C03.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture.outcome import CaptureOutcomeError
from torchlens._io import ArtifactError
from torchlens.errors import TraceCaptureError
from torchlens.intervention._spec_codec import _deserialize_fire_record, _serialize_fire_record
from torchlens.intervention._spec_values import _SerializedState
from torchlens.intervention.errors import ReplayPreconditionError
from torchlens.intervention.save import SaveLevel, load_intervention_spec, save_intervention
from torchlens.intervention.types import FireRecord


def _spec_dir(tmp_path):
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    x = torch.randn(2, 4)
    trace = tl.trace(
        model,
        x,
        save=tl.func("relu"),
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )
    path = tmp_path / "spec.tlspec"
    save_intervention(trace, path)
    return path


# --- edge_address survives the load door -------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of intervention/_spec_codec.py, campaign mutant kills.
def test_fire_record_edge_address_round_trips():
    record = FireRecord(
        target_label="linear_1_1",
        edge_address=(7, "positional", (0,)),
    )
    state = _SerializedState(tensor_entries=[], tensor_refs={})
    payload = _serialize_fire_record(record, SaveLevel.EXECUTABLE_WITH_CALLABLES, state)
    # Simulate the JSON round-trip (tuples become lists on disk).
    payload = json.loads(json.dumps(payload))
    rebuilt = _deserialize_fire_record(payload, {})
    assert rebuilt.edge_address == (7, "positional", (0,))


# Kept by shrink-r4 (per-id subset rule): sole coverage of intervention/_spec_codec.py, campaign mutant kills.
def test_fire_record_edge_address_malformed_refuses():
    with pytest.raises(ReplayPreconditionError, match="edge_address"):
        _deserialize_fire_record({"target_label": "x", "edge_address": "bogus"}, {})


# --- settled-outcome gate on the spec door ------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of intervention/save.py, campaign mutant kills.
def test_save_intervention_refuses_failed_capture(tmp_path):
    class Boom(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.fc(x))
            raise RuntimeError("mid-forward failure")

    with pytest.raises(TraceCaptureError, match="mid-forward failure") as excinfo:
        tl.trace(Boom(), torch.randn(2, 4))
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    partial = excinfo.value.partial_trace
    assert partial is not None
    with pytest.raises(CaptureOutcomeError) as gate_exc:
        save_intervention(partial, tmp_path / "spec.tlspec")
    assert gate_exc.value.fields["code"] == "N1"
    assert not (tmp_path / "spec.tlspec").exists()


def test_save_intervention_complete_capture_round_trips(tmp_path):
    path = _spec_dir(tmp_path)
    spec = load_intervention_spec(path)
    assert spec.metadata["save_level"] == "executable_with_callables"


# --- recipe structural validation ---------------------------------------------


def _rewrite_json(path, name, mutate):
    target = path / name
    data = json.loads(target.read_text())
    mutate(data)
    target.write_text(json.dumps(data))


# Kept by shrink-r4 (per-id subset rule): sole coverage of intervention/_spec_codec.py, campaign mutant kills.
def test_missing_intervention_spec_object_refuses_typed(tmp_path):
    path = _spec_dir(tmp_path)
    _rewrite_json(path, "intervention.json", lambda data: data.pop("rules"))
    with pytest.raises(ArtifactError, match="rules") as info:
        load_intervention_spec(path)
    assert info.value.fields["code"] == "manifest_schema_invalid"


# Kept by shrink-r4 (per-id subset rule): sole coverage of intervention/_spec_codec.py, campaign mutant kills.
def test_unknown_save_level_refuses_typed(tmp_path):
    path = _spec_dir(tmp_path)
    _rewrite_json(path, "intervention.json", lambda data: data.update(save_level="banana"))
    with pytest.raises(ArtifactError, match="save_level") as info:
        load_intervention_spec(path)
    assert info.value.fields["code"] == "artifact_save_level_invalid"


# Kept by shrink-r4 (per-id subset rule): sole coverage of intervention/_spec_codec.py, campaign mutant kills.
def test_non_list_target_manifest_refuses_typed(tmp_path):
    path = _spec_dir(tmp_path)
    _rewrite_json(
        path,
        "manifest.json",
        lambda data: data["intervention_compat_metadata"].update(target_manifest=42),
    )
    with pytest.raises(ArtifactError, match="target_manifest") as info:
        load_intervention_spec(path)
    assert info.value.fields["code"] == "manifest_schema_invalid"
