"""Stage 8 optional runnable activation-payload and numeric-attestation tests."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.runnable import assert_sparse_core_has_no_tensor_payload
from torchlens.errors import NumericAttestationError
from torchlens.options import CaptureOptions, SaveOptions
from torchlens.runnable import NumericAttestationStatus, StateSource


class ActivationPayloadModel(nn.Module):
    """Small deterministic graph with real parameter and buffer state."""

    def __init__(self) -> None:
        """Initialize byte-stable capture state."""

        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.register_buffer("scale", torch.tensor([1.5, -0.5]))
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, 2.0, -1.0], [-2.0, 0.5, 3.0]]))
            self.linear.bias.copy_(torch.tensor([0.25, -0.75]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply a deterministic state-bearing graph."""

        return torch.relu(self.linear(value)) * self.scale


class DropoutActivationModel(nn.Module):
    """Nondeterministic model whose replay is ineligible for byte attestation."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply training-mode dropout at a recorded RNG boundary."""

        return torch.nn.functional.dropout(value, p=0.5, training=True)


def _capture(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    save: Any = None,
) -> tl.Trace:
    """Capture one runnable-ready trace with an optional existing selector."""

    kwargs = {} if save is None else {"save": save}
    return tl.trace(
        model,
        inputs,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
        **kwargs,
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


def _physical_outs(trace: tl.Trace) -> tuple[torch.Tensor | None, ...]:
    """Snapshot physical Op outs without triggering unsaved-payload errors."""

    values: list[torch.Tensor | None] = []
    for op in trace.layer_list:
        value = op._slot("out")
        values.append(value.detach().clone() if isinstance(value, torch.Tensor) else None)
    return tuple(values)


@pytest.mark.smoke
def test_include_activations_persists_exact_save_selected_family_and_digests(
    tmp_path: Path,
) -> None:
    """Archive only existing ``save=`` decisions with explicit slot membership."""

    model = ActivationPayloadModel().eval()
    trace = _capture(model, torch.ones(2, 3), save=tl.func("relu"))
    path = tmp_path / "selected.tlspec"

    trace.save(path, level="runnable", include_activations=True)

    manifest = _manifest(path)
    activation_layer = manifest["run"]["payload_layers"]["activations"]
    members = activation_layer["members"]
    selected_labels = {str(op.label) for op in trace.layer_list if bool(op.has_saved_activation)}
    assert activation_layer["present"] is True
    assert activation_layer["schema"] == "selected_activation_v1"
    assert {member["op_label"] for member in members} == selected_labels
    assert all(len(member["byte_digest"]) == 64 for member in members)
    assert {member["slot_id"] for member in members} == {
        f"slot:{label}" for label in selected_labels
    }
    entries = [entry for entry in manifest["tensors"] if entry["kind"] == "runnable_activation"]
    assert {entry["kind"] for entry in manifest["tensors"]} == {"runnable_activation"}
    assert {entry["blob_id"] for entry in entries} == {member["blob_id"] for member in members}
    assert_sparse_core_has_no_tensor_payload(manifest["run"])
    assert_sparse_core_has_no_tensor_payload(_metadata(path))
    assert all(op.func is None for op in _metadata(path)["layer_list"])
    tl.validation.validate_tlspec(path)


def test_include_activations_default_is_absent_and_requires_runnable_level(
    tmp_path: Path,
) -> None:
    """Keep activation payloads absent by default and reject analysis save levels."""

    trace = _capture(ActivationPayloadModel().eval(), torch.ones(2, 3))
    path = tmp_path / "sparse.tlspec"
    trace.save(path, level="runnable")

    manifest = _manifest(path)
    assert manifest["run"]["payload_layers"]["activations"] == {
        "present": False,
        "schema": "selected_activation_v1",
    }
    assert not any(entry["kind"] == "runnable_activation" for entry in manifest["tensors"])
    assert tl.load(path).archived_activations == {}
    with pytest.raises(ValueError, match="requires level='runnable'"):
        trace.save(tmp_path / "portable.tlspec", include_activations=True)


@pytest.mark.smoke
def test_archived_activations_are_inspectable_but_never_execution_inputs(
    tmp_path: Path,
) -> None:
    """Expose archives separately while changed-input execution remains freshly computed."""

    model = ActivationPayloadModel().eval()
    capture_inputs = torch.ones(2, 3)
    trace = _capture(model, capture_inputs)
    path = tmp_path / "activation-only.tlspec"
    trace.save(path, level="runnable", include_activations=True)
    loaded = tl.load(path)
    loaded.load_state_dict(model.state_dict())

    assert loaded.archived_activations
    assert all(
        isinstance(record.value, torch.Tensor) for record in loaded.archived_activations.values()
    )
    for record in loaded.archived_activations.values():
        record.value.zero_()
    changed_inputs = torch.tensor([[2.0, -1.0, 0.5], [-3.0, 0.25, 4.0]])
    result = loaded.run(inputs=changed_inputs)

    assert torch.equal(result.output, model(changed_inputs))
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


@pytest.mark.smoke
def test_original_input_real_state_attests_for_embedded_and_user_state(
    tmp_path: Path,
) -> None:
    """Attest every archived raw slot with both supported real-state sources."""

    model = ActivationPayloadModel().eval()
    inputs = torch.ones(2, 3)
    trace = _capture(model, inputs)
    both_path = tmp_path / "both.tlspec"
    activation_path = tmp_path / "activation-only.tlspec"
    trace.save(
        both_path,
        level="runnable",
        include_weights=True,
        include_activations=True,
    )
    trace.save(activation_path, level="runnable", include_activations=True)

    embedded_result = tl.load(both_path).run(inputs=inputs)
    user_loaded = tl.load(activation_path)
    user_loaded.load_state_dict(model.state_dict())
    user_result = user_loaded.run(inputs=inputs)

    assert torch.equal(embedded_result.output, model(inputs))
    assert embedded_result.report.state_source is StateSource.EMBEDDED_CAPTURE_STATE
    assert embedded_result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    assert user_result.report.state_source is StateSource.USER_STATE_DICT
    assert user_result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    assert any(
        check.name.startswith("numeric_attestation") for check in user_result.report.contract_checks
    )


def test_rng_model_activation_attestation_is_not_applicable_instead_of_failing(
    tmp_path: Path,
) -> None:
    """Treat expected dropout nondeterminism as ineligible, not archive corruption."""

    model = DropoutActivationModel()
    inputs = torch.ones(32)
    path = tmp_path / "dropout.tlspec"
    _capture(model, inputs).save(
        path,
        level="runnable",
        include_weights=True,
        include_activations=True,
    )

    result = tl.load(path).run(inputs=inputs)

    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_corrupt_archived_digest_fails_tripwire_and_rolls_back(tmp_path: Path) -> None:
    """Fail on the first declared byte-digest mismatch without mutating the source Trace."""

    model = ActivationPayloadModel().eval()
    inputs = torch.ones(2, 3)
    path = tmp_path / "corrupt-digest.tlspec"
    _capture(model, inputs).save(
        path,
        level="runnable",
        include_weights=True,
        include_activations=True,
    )
    manifest = _manifest(path)
    first_member = manifest["run"]["payload_layers"]["activations"]["members"][0]
    first_member["byte_digest"] = "0" * 64
    with (path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    loaded = tl.load(path)
    before = _physical_outs(loaded)

    with pytest.raises(NumericAttestationError) as caught:
        loaded.run(inputs=inputs)

    mismatch = caught.value.fields["first_mismatch"]
    details = dict(mismatch.details)
    assert caught.value.fields["code"] == "numeric_attestation_failed"
    assert (
        caught.value.fields["numeric_attestation"]
        is NumericAttestationStatus.NUMERIC_ATTESTATION_FAILED
    )
    assert mismatch.code.value == "numeric_attestation_failed"
    assert details["slot_id"] == first_member["slot_id"]
    assert details["expected_digest"] == "0" * 64
    assert details["archived_digest"] != details["expected_digest"]
    assert _physical_outs(loaded) == before
    assert not bool(loaded.__dict__.get("_runnable_poisoned", False))


def test_random_state_changed_input_and_non_equivalent_state_are_not_applicable(
    tmp_path: Path,
) -> None:
    """Never report a numeric pass outside original-input capture-equivalent runs."""

    model = ActivationPayloadModel().eval()
    inputs = torch.ones(2, 3)
    path = tmp_path / "activation-only.tlspec"
    _capture(model, inputs).save(path, level="runnable", include_activations=True)

    random_result = tl.load(path).run(inputs=inputs, seed=41)
    changed = tl.load(path)
    changed.load_state_dict(model.state_dict())
    changed_result = changed.run(inputs=torch.full_like(inputs, 2.0))
    different_state = tl.load(path)
    different_state.load_state_dict(
        {name: torch.zeros_like(value) for name, value in model.state_dict().items()}
    )
    different_result = different_state.run(inputs=inputs)

    assert random_result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert changed_result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert different_result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_transformed_outputs_are_archived_for_inspection(tmp_path: Path) -> None:
    """Include both retained raw and transformed payload fields without executing transforms."""

    model = ActivationPayloadModel().eval()
    trace = tl.trace(
        model,
        torch.ones(2, 3),
        save=SaveOptions(activation_transform=lambda value: value.mean()),
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path / "transformed.tlspec"
    trace.save(path, level="runnable", include_activations=True)
    loaded = tl.load(path)

    fields = {record.field for record in loaded.archived_activations.values()}
    assert fields == {"out", "transformed_out"}
