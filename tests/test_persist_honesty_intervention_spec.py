"""Persistence honesty: the intervention-spec door validates, gates, and keeps
edge addresses.

WT1 A-IV item 20 (lane A08): the intervention-spec load door silently
discarded ``FireRecord.edge_address``; ``save_intervention`` had no settled-
outcome gate; and ``spec.json`` deserialized with no structural validation
(a missing ``intervention_spec`` key died with a bare ``KeyError``). Deeper
per-record staging refusals continue in lane C03.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.capture.outcome import CaptureOutcomeError
from torchlens.intervention.errors import ReplayPreconditionError
from torchlens.intervention.save import (
    SaveLevel,
    _deserialize_fire_record,
    _serialize_fire_record,
    _SerializedState,
    load_intervention_spec,
    save_intervention,
)
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


def test_fire_record_edge_address_absent_stays_none():
    record = FireRecord(target_label="linear_1_1")
    state = _SerializedState(tensor_entries=[], tensor_refs={})
    payload = json.loads(
        json.dumps(_serialize_fire_record(record, SaveLevel.EXECUTABLE_WITH_CALLABLES, state))
    )
    assert _deserialize_fire_record(payload, {}).edge_address is None


def test_fire_record_edge_address_malformed_refuses():
    with pytest.raises(ReplayPreconditionError, match="edge_address"):
        _deserialize_fire_record({"target_label": "x", "edge_address": "bogus"}, {})


# --- settled-outcome gate on the spec door ------------------------------------


def test_save_intervention_refuses_failed_capture(tmp_path):
    class Boom(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.fc(x))
            raise RuntimeError("mid-forward failure")

    with pytest.raises(RuntimeError, match="mid-forward failure") as excinfo:
        tl.trace(Boom(), torch.randn(2, 4))
    partial = tl.partial.from_failed_capture(excinfo.value)
    assert partial is not None
    with pytest.raises(CaptureOutcomeError) as gate_exc:
        save_intervention(partial, tmp_path / "spec.tlspec")
    assert gate_exc.value.fields["code"] == "N1"
    assert not (tmp_path / "spec.tlspec").exists()


def test_save_intervention_complete_capture_round_trips(tmp_path):
    path = _spec_dir(tmp_path)
    spec = load_intervention_spec(path)
    assert spec.metadata["save_level"] == "executable_with_callables"


# --- spec.json structural validation ------------------------------------------


def _rewrite_spec_json(path, mutate):
    spec_file = path / "spec.json"
    data = json.loads(spec_file.read_text())
    mutate(data)
    spec_file.write_text(json.dumps(data))


def test_missing_intervention_spec_object_refuses_typed(tmp_path):
    path = _spec_dir(tmp_path)
    _rewrite_spec_json(path, lambda data: data.pop("intervention_spec"))
    with pytest.raises(ReplayPreconditionError, match="intervention_spec"):
        load_intervention_spec(path)


def test_unknown_save_level_refuses_typed(tmp_path):
    path = _spec_dir(tmp_path)
    _rewrite_spec_json(path, lambda data: data.update(save_level="banana"))
    with pytest.raises(ReplayPreconditionError, match="save_level"):
        load_intervention_spec(path)


def test_non_list_target_manifest_refuses_typed(tmp_path):
    path = _spec_dir(tmp_path)
    _rewrite_spec_json(path, lambda data: data.update(target_manifest=42))
    with pytest.raises(ReplayPreconditionError, match="target_manifest"):
        load_intervention_spec(path)
