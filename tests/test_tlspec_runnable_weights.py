"""Stage 7 optional runnable weight-payload tests."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any

import pytest
from safetensors.torch import load_file
import torch
from torch import nn

import torchlens as tl
from torchlens._io import TorchLensIOError
from torchlens._io.runnable import (
    assert_sparse_core_has_no_tensor_payload,
    build_sparse_run_descriptor,
)
from torchlens._runnable_state import runnable_tensor_byte_digest
from torchlens.errors import StateBindingError
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness, StateSource


class WeightPayloadModel(nn.Module):
    """Small deterministic model with parameter and buffer state."""

    def __init__(self) -> None:
        """Initialize real weights plus persistent and non-persistent buffers."""

        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.register_buffer("scale", torch.tensor([1.5, -0.5]))
        self.register_buffer("unused", torch.tensor([7.0, 8.0]))
        self.register_buffer("scratch", torch.tensor([99.0]), persistent=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, 2.0, -1.0], [-2.0, 0.5, 3.0]]))
            self.linear.bias.copy_(torch.tensor([0.25, -0.75]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the state-bearing recorded graph."""

        return self.linear(value) * self.scale


class UsedBufferPersistenceModel(nn.Module):
    """Use both a non-persistent buffer and ordinary persistent buffer state."""

    def __init__(self) -> None:
        """Register buffers on opposite sides of the canonical state contract."""

        super().__init__()
        self.register_buffer("offset", torch.tensor([7.0]), persistent=False)
        self.register_buffer("persistent_guard", torch.tensor([0.0]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Use both buffers so producer bindings must classify each honestly."""

        return value + self.offset + self.persistent_guard


def _capture(model: nn.Module) -> tl.Trace:
    """Capture one model with the sparse runnable prerequisites enabled."""

    return tl.trace(
        model,
        torch.ones(2, 3),
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )


def _manifest(path: Path) -> dict[str, Any]:
    """Read one test-produced public manifest."""

    with (path / "manifest.json").open(encoding="utf-8") as handle:
        value = json.load(handle)
    assert isinstance(value, dict)
    return value


def _metadata(path: Path) -> dict[str, Any]:
    """Read trusted test-produced sparse metadata for invariant checks."""

    with (path / "metadata.pkl").open("rb") as handle:
        value = pickle.load(handle)  # noqa: S301 - created in this test process
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize("field_name", ("annotations", "interventions"))
def test_runnable_save_rejects_tensor_in_slotted_keep_field(
    tmp_path: Path,
    field_name: str,
) -> None:
    """Reject tensor payloads hidden in any persisted slotted Op field."""

    trace = _capture(WeightPayloadModel().eval())
    op = next(op for op in trace.layer_list if not op.is_input)
    object.__setattr__(op, field_name, {"hidden_tensor": torch.tensor([1.0])})

    with pytest.raises((AssertionError, TorchLensIOError)) as caught:
        trace.save(tmp_path / f"{field_name}.tlspec", level="runnable")

    if isinstance(caught.value, TorchLensIOError):
        assert isinstance(caught.value.__cause__, AssertionError)


def test_runnable_save_normal_model_remains_tensor_payload_free(tmp_path: Path) -> None:
    """Keep the strengthened sparse-core tripwire transparent to normal saves."""

    path = tmp_path / "normal.tlspec"
    _capture(WeightPayloadModel().eval()).save(path, level="runnable")

    assert path.is_dir()


@pytest.mark.smoke
def test_used_nonpersistent_buffer_is_embedded_without_becoming_canonical_state(
    tmp_path: Path,
) -> None:
    """Reproduce used non-persistent buffers while preserving persistent state rules."""

    model = UsedBufferPersistenceModel().eval()
    inputs = torch.tensor([2.0])
    trace = tl.trace(
        model,
        inputs,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    descriptor = build_sparse_run_descriptor(trace)
    bindings = {
        slot.state_binding.state_dict_name: slot.state_binding
        for slot in descriptor.tensor_slots
        if slot.state_binding is not None
    }

    assert descriptor.preflight.passed
    assert bindings["offset"].persistent is False
    assert bindings["persistent_guard"].persistent is True
    assert "offset" not in model.state_dict()
    assert descriptor.payload_layers.nonpersistent_buffers.present is True

    sparse_path = tmp_path / "nonpersistent.tlspec"
    weighted_path = tmp_path / "nonpersistent-weighted.tlspec"
    trace.save(sparse_path, level="runnable")
    trace.save(weighted_path, level="runnable", include_weights=True)

    sparse_manifest = _manifest(sparse_path)
    payload = sparse_manifest["run"]["payload_layers"]["nonpersistent_buffers"]
    assert payload == {
        "present": True,
        "schema": "runnable_nonpersistent_buffer_v1",
    }
    nonpersistent_entries = [
        entry
        for entry in sparse_manifest["tensors"]
        if entry["kind"] == "runnable_nonpersistent_buffer"
    ]
    assert [entry["label"] for entry in nonpersistent_entries] == ["offset"]
    stored_offset = load_file(str(sparse_path / nonpersistent_entries[0]["relative_path"]))["data"]
    assert torch.equal(stored_offset, model.offset)

    loaded = tl.load(sparse_path)
    random_result = loaded.run(inputs=inputs, seed=1)
    assert torch.equal(random_result.output, model(inputs))
    assert torch.equal(random_result.output, torch.tensor([9.0]))
    assert random_result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert all("offset" not in slot_id for slot_id in random_result.report.random_filled_slot_ids)

    loaded.load_state_dict(model.state_dict())
    staged_result = loaded.run(inputs=inputs, seed=1)
    assert torch.equal(staged_result.output, model(inputs))
    assert staged_result.report.state_source is StateSource.USER_STATE_DICT

    weighted_manifest = _manifest(weighted_path)
    weight_entries = [
        entry for entry in weighted_manifest["tensors"] if entry["kind"] == "runnable_weight"
    ]
    assert [entry["label"] for entry in weight_entries] == ["persistent_guard"]
    weighted = tl.load(weighted_path)
    weighted_result = weighted.run(inputs=inputs, seed=1)
    assert torch.equal(weighted_result.output, model(inputs))
    assert weighted_result.report.state_source is StateSource.EMBEDDED_CAPTURE_STATE


def test_runnable_tensor_digest_includes_dtype_and_shape() -> None:
    """Distinguish logical tensors that share only their raw byte representation."""

    float_value = torch.tensor([1.0], dtype=torch.float32)
    integer_value = float_value.view(torch.int32)
    reshaped_value = float_value.reshape(1, 1)

    assert torch.equal(float_value.view(torch.uint8), integer_value.view(torch.uint8))
    assert runnable_tensor_byte_digest(float_value) != runnable_tensor_byte_digest(integer_value)
    assert runnable_tensor_byte_digest(float_value) != runnable_tensor_byte_digest(reshaped_value)


@pytest.mark.smoke
def test_include_weights_writes_full_state_dict_as_separate_blob_family(
    tmp_path: Path,
) -> None:
    """Persist parameters and persistent buffers without touching the sparse core."""

    model = WeightPayloadModel().eval()
    trace = _capture(model)
    sparse_path = tmp_path / "sparse.tlspec"
    weighted_path = tmp_path / "weighted.tlspec"

    trace.save(sparse_path, level="runnable")
    trace.save(weighted_path, level="runnable", include_weights=True)

    sparse_manifest = _manifest(sparse_path)
    weighted_manifest = _manifest(weighted_path)
    assert sparse_manifest["run"]["payload_layers"]["weights"] == {
        "present": False,
        "schema": "state_dict_v1",
    }
    assert sparse_manifest["body_index"] == []
    assert list((sparse_path / "blobs").iterdir()) == []

    assert weighted_manifest["run"]["payload_layers"]["weights"] == {
        "present": True,
        "schema": "state_dict_v1",
    }
    weight_entries = [
        entry for entry in weighted_manifest["tensors"] if entry["kind"] == "runnable_weight"
    ]
    assert {entry["kind"] for entry in weighted_manifest["tensors"]} == {"runnable_weight"}
    expected_state = model.state_dict()
    assert {entry["label"] for entry in weight_entries} == set(expected_state)
    assert "scratch" not in {entry["label"] for entry in weight_entries}
    assert "unused" in {entry["label"] for entry in weight_entries}
    for entry in weight_entries:
        stored = load_file(str(weighted_path / entry["relative_path"]))["data"]
        assert torch.equal(stored, expected_state[entry["label"]])

    metadata = _metadata(weighted_path)
    assert_sparse_core_has_no_tensor_payload(weighted_manifest["run"])
    assert_sparse_core_has_no_tensor_payload(metadata)
    assert metadata["_buffer_initial_values"] == {}
    assert all(op.func is None for op in metadata["layer_list"])
    tl.validation.validate_tlspec(weighted_path)


@pytest.mark.smoke
def test_embedded_weights_run_matches_live_and_reports_capture_state(tmp_path: Path) -> None:
    """Run self-contained real weights exactly without staging user state."""

    model = WeightPayloadModel().eval()
    trace = _capture(model)
    path = tmp_path / "weighted.tlspec"
    trace.save(path, level="runnable", include_weights=True)
    loaded = tl.load(path)
    inputs = torch.tensor([[2.0, -1.0, 0.5], [-3.0, 0.25, 4.0]])

    result = loaded.run(inputs=inputs, seed=91)

    assert torch.equal(result.output, model(inputs))
    assert result.report.state_source is StateSource.EMBEDDED_CAPTURE_STATE
    assert result.report.random_filled_slot_ids == ()
    assert loaded.readiness is not None
    assert loaded.readiness.state_sources_available[0] is StateSource.EMBEDDED_CAPTURE_STATE


def test_user_state_dict_overrides_embedded_capture_state(tmp_path: Path) -> None:
    """Give an explicitly staged user mapping precedence over embedded weights."""

    model = WeightPayloadModel().eval()
    trace = _capture(model)
    path = tmp_path / "weighted.tlspec"
    trace.save(path, level="runnable", include_weights=True)
    loaded = tl.load(path)
    user_state = {name: torch.zeros_like(value) for name, value in model.state_dict().items()}
    user_state["scale"] = torch.ones_like(user_state["scale"])

    loaded.load_state_dict(user_state)
    result = loaded.run(inputs=torch.ones(2, 3), seed=17)

    assert result.report.state_source is StateSource.USER_STATE_DICT
    assert torch.equal(result.output, torch.zeros(2, 2))


def test_mismatched_embedded_state_raises_shared_strict_binding_error(
    tmp_path: Path,
) -> None:
    """Reject a descriptor/blob mismatch through the ordinary strict binder."""

    model = WeightPayloadModel().eval()
    trace = _capture(model)
    path = tmp_path / "weighted.tlspec"
    trace.save(path, level="runnable", include_weights=True)
    manifest = _manifest(path)
    weight_slot = next(
        slot
        for slot in manifest["run"]["tensor_slots"]
        if slot["state_binding"] is not None
        and slot["state_binding"]["state_dict_name"] == "linear.weight"
    )
    weight_slot["shape"] = [1]
    weight_slot["rank"] = 1
    with (path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    with pytest.raises(StateBindingError) as caught:
        tl.load(path)

    assert "state_shape_mismatch" in caught.value.fields["codes"]


def test_include_weights_requires_runnable_save_level(tmp_path: Path) -> None:
    """Reject the public payload switch on analysis-oriented save levels."""

    trace = _capture(WeightPayloadModel().eval())

    with pytest.raises(ValueError, match="requires level='runnable'"):
        trace.save(tmp_path / "portable.tlspec", include_weights=True)


def test_weight_payload_preserves_canonical_alias_records(tmp_path: Path) -> None:
    """Persist every tied state name and bind one coherent shared allocation."""

    class TiedModel(nn.Module):
        """Two modules sharing one parameter tensor."""

        def __init__(self) -> None:
            """Tie the second linear weight to the first."""

            super().__init__()
            self.first = nn.Linear(2, 2, bias=False)
            self.second = nn.Linear(2, 2, bias=False)
            self.second.weight = self.first.weight

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Use both canonical aliases in one recorded graph."""

            return self.first(value) + self.second(value)

    model = TiedModel().eval()
    trace = tl.trace(
        model,
        torch.ones(1, 2),
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path / "tied.tlspec"
    trace.save(path, level="runnable", include_weights=True)
    loaded = tl.load(path)
    descriptor = loaded.runnable_descriptor
    assert descriptor is not None
    tied_slots = [
        slot
        for slot in descriptor.tensor_slots
        if slot.state_binding is not None
        and slot.state_binding.state_dict_name in {"first.weight", "second.weight"}
    ]

    assert len(tied_slots) == 2
    assert len({slot.state_binding.alias_group for slot in tied_slots}) == 1
    inputs = torch.tensor([[1.0, -2.0]])
    assert torch.equal(loaded.run(inputs=inputs).output, model(inputs))


def test_loaded_runnable_embedded_state_trace_pickles_and_deepcopies(tmp_path: Path) -> None:
    """A loaded runnable trace with embedded state binds MappingProxyType views; the

    generic pickle/deepcopy path must neutralize them so the trace (and any run fork
    derived from it) round-trips instead of crashing on ``cannot pickle 'mappingproxy'``.
    """
    import copy

    from torchlens.errors import PoisonedRunError
    from torchlens.runnable import DivergencePolicy

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(3, 2)
            with torch.no_grad():
                self.lin.weight.fill_(1.0)
                self.lin.bias.zero_()

        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.lin(v))

    model = _M().eval()
    trace = tl.trace(
        model,
        torch.ones(2, 3),
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )
    path = tmp_path / "embedded.tlspec"
    trace.save(path, level="runnable", include_weights=True)
    loaded = tl.load(path)

    # Successful run fork carries the embedded MappingProxyType state -> must pickle + deepcopy.
    ok_fork = loaded.run(inputs=torch.ones(2, 3)).trace
    assert pickle.loads(pickle.dumps(ok_fork)) is not None
    assert copy.deepcopy(ok_fork) is not None

    # A DIVERGED fork must round-trip its poison flag through pickle and deepcopy and stay refused.
    diverged = loaded.run(
        inputs=torch.ones(5, 3), on_divergence=DivergencePolicy.RETURN_DIVERGED
    ).trace
    for revived in (pickle.loads(pickle.dumps(diverged)), copy.deepcopy(diverged)):
        assert revived.__dict__.get("_runnable_poisoned") is True
        with pytest.raises(PoisonedRunError):
            revived.to_pandas()
