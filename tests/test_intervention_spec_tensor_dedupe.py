"""One tensor object, one sidecar, however many places a saved spec references it.

A sticky hook spec serializes its helper twice (as ``hook`` and as ``helper``) and
every fire record carries the edit again, so a ``tl.steer(direction)`` hook used to
write three byte-identical ``tensors/value_*.safetensors`` blobs for its one direction
tensor, one more per fire record. The value codec now keys sidecar ids on the tensor
object, so the saved spec references one blob and reloads to the same outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention._spec_values import _serialize_value, _SerializedState
from torchlens.intervention.save import SaveLevel
from torchlens.intervention.types import InterventionSpec
from torchlens.options import CaptureOptions, InterventionOptions

pytestmark = pytest.mark.usefixtures("root_edits_in_place")  # D.12: no per-call mutation keyword


class _SteerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x)) + 1


def _capture() -> Any:
    torch.manual_seed(0)
    return tl.trace(
        _SteerModel(), torch.randn(2, 3), capture=CaptureOptions(replayable=True, seed=0)
    )


def _relu_label(log: Any) -> str:
    return next(str(op.layer_label) for op in log._layer_list if op.type == "relu")


def _tensor_refs(payload: Any) -> list[str]:
    """Every ``__tensor_ref__`` id in ``payload``, in document order."""

    if isinstance(payload, dict):
        if "__tensor_ref__" in payload:
            return [str(payload["__tensor_ref__"])]
        return [ref for value in payload.values() for ref in _tensor_refs(value)]
    if isinstance(payload, list):
        return [ref for value in payload for ref in _tensor_refs(value)]
    return []


def _save_sticky_steer(tmp_path: Path, direction: torch.Tensor) -> tuple[Any, str, Path]:
    log = _capture()
    label = _relu_label(log)
    log.do(
        tl.func("relu"),
        tl.steer(direction),
        intervention=InterventionOptions(),
    )
    # A later propagating do re-fires the sticky steer: a second fire record, one
    # more reference to the same direction (an exact identity scale adds no tensor).
    log.do(
        tl.func("relu"),
        tl.scale(1.0),
        intervention=InterventionOptions(),
    )
    path = tmp_path / "steer.tlspec"
    log.save_intervention(path, level="portable")
    return log, label, path


def test_sticky_steer_hook_writes_one_sidecar_for_every_reference(tmp_path: Path) -> None:
    _, _, path = _save_sticky_steer(tmp_path, torch.ones(2, 3))

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    spec_json = json.loads((path / "spec.json").read_text(encoding="utf-8"))
    hook_spec = spec_json["intervention_spec"]["hook_specs"][0]
    refs = _tensor_refs(spec_json["intervention_spec"])

    assert len(manifest["tensor_entries"]) == 1
    assert sorted(f.name for f in (path / "tensors").glob("*.safetensors")) == [
        f"{manifest['tensor_entries'][0]['blob_id']}.safetensors"
    ]
    assert _tensor_refs(hook_spec["hook"]) and _tensor_refs(hook_spec["helper"])
    assert len(refs) >= 3  # hook, helper and the fire records
    assert set(refs) == {manifest["tensor_entries"][0]["blob_id"]}


def test_sticky_steer_hook_reloads_to_the_live_outputs(tmp_path: Path) -> None:
    direction = torch.full((2, 3), 0.25)
    live, label, path = _save_sticky_steer(tmp_path, direction)

    loaded = tl.load(path)
    assert isinstance(loaded, InterventionSpec)
    hook_spec = loaded.hook_specs[0]
    assert hook_spec.helper is not None
    assert torch.equal(hook_spec.helper.args[0], direction)
    assert torch.equal(hook_spec.hook.args[0], direction)

    fresh = _capture()
    fresh.do(
        hook_spec.site_target,
        hook_spec.helper,
        intervention=InterventionOptions(),
    )
    assert torch.equal(fresh[label].out, live[label].out)
    assert torch.equal(fresh[label].out, _capture()[label].out + direction)


def test_codec_keys_sidecar_ids_on_the_tensor_object() -> None:
    state = _SerializedState(tensor_entries=[], tensor_refs={})
    shared = torch.ones(2)
    twin = torch.ones(2)  # equal values, a different object: never merged by value

    first = _serialize_value(shared, SaveLevel.PORTABLE, state)
    assert _serialize_value(shared, SaveLevel.PORTABLE, state) == first
    assert _serialize_value((shared, shared), SaveLevel.PORTABLE, state) == [first, first]
    assert _serialize_value(twin, SaveLevel.PORTABLE, state) != first
    assert len(state.tensor_refs) == 2
