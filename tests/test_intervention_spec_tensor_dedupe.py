"""One tensor object, one stored tensor, however many places a saved spec references it.

A sticky hook spec serializes its helper twice (as ``hook`` and as ``helper``) and
every fire record carries the edit again, so a ``tl.steer(direction)`` hook used to
write three byte-identical tensor blobs for its one direction tensor, one more per
fire record. Both value encoders (the intervention codec and the 3.0 format's recipe
encoder, which writes ``blobs/recipe-<nnnn>.safetensors``) key stored tensors on the
tensor object, so the saved recipe references one tensor and reloads to the same
outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

import torchlens as tl
from torchlens._io.format3.contract import BLOBS_DIR, INTERVENTION_FILE, TYPED_VALUE_KEY
from torchlens._io.format3.payloads import RECIPE_CHUNK_STEM, PayloadStore
from torchlens._io.format3.recipe import _RecipeEncoder
from torchlens.intervention._spec_values import _serialize_value, _SerializedState
from torchlens.intervention.save import SaveLevel
from torchlens.intervention.types import InterventionSpec
from torchlens.options import CaptureOptions, InterventionOptions


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


def _tensor_refs(payload: Any) -> list[tuple[str, str]]:
    """Every recipe tensor reference ``(blob, key)`` in ``payload``, in document order."""

    if isinstance(payload, dict):
        if payload.get(TYPED_VALUE_KEY) == "tensor":
            return [(str(payload["blob"]), str(payload["key"]))]
        return [ref for value in payload.values() for ref in _tensor_refs(value)]
    if isinstance(payload, list):
        return [ref for value in payload for ref in _tensor_refs(value)]
    return []


def _recipe_chunk_keys(path: Path) -> list[tuple[str, str]]:
    """Every ``(chunk relative path, key)`` stored in the recipe's chunks."""

    stored: list[tuple[str, str]] = []
    for chunk in sorted((path / BLOBS_DIR).glob(f"{RECIPE_CHUNK_STEM}-*.safetensors")):
        with safe_open(str(chunk), framework="pt") as handle:
            relative = f"{BLOBS_DIR}/{chunk.name}"
            stored.extend((relative, key) for key in handle.keys())
    return stored


def _save_sticky_steer(tmp_path: Path, direction: torch.Tensor) -> tuple[Any, str, Path]:
    log = _capture()
    label = _relu_label(log)
    log.do(
        tl.func("relu"),
        tl.steer(direction),
        intervention=InterventionOptions(confirm_mutation=True),
    )
    log.push()  # a second fire record: one more reference to the same direction
    path = tmp_path / "steer.tlspec"
    log.save_intervention(path, level="portable")
    return log, label, path


def test_sticky_steer_hook_writes_one_sidecar_for_every_reference(tmp_path: Path) -> None:
    _, _, path = _save_sticky_steer(tmp_path, torch.ones(2, 3))

    recipe = json.loads((path / INTERVENTION_FILE).read_text(encoding="utf-8"))
    refs = _tensor_refs(recipe)
    stored = _recipe_chunk_keys(path)

    assert len(stored) == 1, stored
    assert refs, "the recipe references no tensor"
    assert set(refs) == set(stored)


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
        intervention=InterventionOptions(confirm_mutation=True),
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


def test_recipe_encoder_keys_stored_tensors_on_the_tensor_object(tmp_path: Path) -> None:
    store = PayloadStore(tmp_path)
    encoder = _RecipeEncoder(store)
    shared = torch.ones(2)
    twin = torch.ones(2)  # equal values, a different object: never merged by value

    first = encoder.encode(shared, "0.edit")
    assert encoder.encode(shared, "1.edit") == first
    assert encoder.encode([shared, shared], "2.edit") == [first, first]
    assert encoder.encode(twin, "3.edit") != first
    store.close()
    assert len(_recipe_chunk_keys(tmp_path)) == 2
