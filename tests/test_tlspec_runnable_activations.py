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
from torchlens.runnable import (
    DivergencePolicy,
    NumericAttestationStatus,
    PathFaithfulness,
    StateSource,
)


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


class InplaceInputActivationModel(nn.Module):
    """Deterministic model that mutates its model input during execution."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Mutate the input and return a deterministic downstream value."""

        value.add_(1)
        return value * 2


class InplaceInternalActivationModel(nn.Module):
    """Model that mutates a previously saved internal activation in place."""

    def __init__(self) -> None:
        """Initialize an identity linear layer for byte-stable activations."""

        super().__init__()
        self.linear = nn.Linear(3, 3)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(3))
            self.linear.bias.zero_()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Consume then mutate the saved internal linear output.

        Parameters
        ----------
        value:
            Model input tensor.

        Returns
        -------
        torch.Tensor
            Deterministic result that includes both pre- and post-mutation values.
        """

        activation = self.linear(value)
        before_mutation = activation * 2.0
        activation.add_(100.0)
        return before_mutation + activation.sum()


class SeededRngActivationModel(nn.Module):
    """Model that exposes one PyTorch seeded-RNG operation per capture."""

    def __init__(self, operation: str) -> None:
        """Store the requested RNG operation name.

        Parameters
        ----------
        operation:
            Name of the seeded PyTorch operation to execute.
        """

        super().__init__()
        self.operation = operation

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run one seeded-RNG operation.

        Parameters
        ----------
        value:
            Tensor used to shape or parameterize the selected operation.

        Returns
        -------
        torch.Tensor
            The selected operation's tensor result.
        """

        if self.operation == "rand":
            return torch.rand(value.shape, device=value.device)
        if self.operation == "randn":
            return torch.randn(value.shape, device=value.device)
        if self.operation == "randint":
            return torch.randint(0, 4, value.shape, device=value.device).to(value.dtype)
        if self.operation == "randperm":
            return value[torch.randperm(value.shape[0])]
        if self.operation == "multinomial":
            return value[torch.multinomial(torch.softmax(value, dim=0), 2, replacement=True)]
        if self.operation == "bernoulli":
            return torch.bernoulli(torch.sigmoid(value))
        if self.operation == "poisson":
            return torch.poisson(torch.ones_like(value))
        if self.operation == "normal":
            return torch.normal(torch.zeros_like(value), torch.ones_like(value))
        if self.operation == "uniform_":
            return value.clone().uniform_()
        if self.operation == "rand_like":
            return torch.rand_like(value)
        if self.operation == "randn_like":
            return torch.randn_like(value)
        if self.operation == "randint_like":
            return torch.randint_like(value, 4).to(value.dtype)
        if self.operation == "dropout":
            return torch.nn.functional.dropout(value, p=0.5, training=True)
        if self.operation == "dropout3d":
            return torch.nn.functional.dropout3d(value, p=0.5, training=True)
        if self.operation == "rrelu":
            return torch.nn.functional.rrelu(value, training=True)
        if self.operation == "gumbel_softmax":
            return torch.nn.functional.gumbel_softmax(value, tau=1.0, hard=False, dim=-1)
        raise ValueError(f"Unsupported seeded RNG operation {self.operation!r}.")


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


def _seeded_rng_inputs(operation: str) -> torch.Tensor:
    """Return a valid original input for one seeded-RNG model.

    Parameters
    ----------
    operation:
        Seeded-RNG operation selected for the parameterized test.

    Returns
    -------
    torch.Tensor
        Input with the rank required by the selected operation.
    """

    if operation == "dropout3d":
        return torch.ones(1, 2, 2, 2, 2)
    return torch.full((8,), 0.5)


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
    assert activation_layer["schema"] == "selected_activation_v2"
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
        "schema": "selected_activation_v2",
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


@pytest.mark.parametrize(
    "operation",
    (
        "rand",
        "randn",
        "randint",
        "randperm",
        "multinomial",
        "bernoulli",
        "poisson",
        "normal",
        "uniform_",
        "rand_like",
        "randn_like",
        "randint_like",
        "dropout",
        "dropout3d",
        "rrelu",
        "gumbel_softmax",
    ),
)
def test_seeded_rng_operations_skip_original_input_numeric_attestation(
    tmp_path: Path,
    operation: str,
) -> None:
    """Use PyTorch's RNG tags instead of a fragile replay-name allowlist."""

    inputs = _seeded_rng_inputs(operation)
    path = tmp_path / f"{operation}.tlspec"
    model = SeededRngActivationModel(operation)
    _capture(model, inputs).save(
        path,
        level="runnable",
        include_activations=True,
    )

    result = tl.load(path).run(
        inputs=inputs.clone(),
        on_divergence=DivergencePolicy.RETURN_DIVERGED,
    )

    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_inplace_input_attestation_uses_pre_execution_input_digest(tmp_path: Path) -> None:
    """Keep original-input attestation applicable when replay mutates its input slot."""

    model = InplaceInputActivationModel()
    capture_inputs = torch.ones(4)
    path = tmp_path / "inplace-input.tlspec"
    _capture(model, capture_inputs).save(
        path,
        level="runnable",
        include_weights=True,
        include_activations=True,
    )

    result = tl.load(path).run(inputs=torch.ones(4))

    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


def test_inplace_internal_activation_attestation_uses_production_snapshot(
    tmp_path: Path,
) -> None:
    """Attest the pre-mutation internal activation rather than its live alias."""

    model = InplaceInternalActivationModel().eval()
    inputs = torch.ones(2, 3)
    path = tmp_path / "inplace-internal.tlspec"
    _capture(model, inputs).save(
        path,
        level="runnable",
        include_weights=True,
        include_activations=True,
    )

    result = tl.load(path).run(inputs=inputs.clone())

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


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


def test_transformed_outputs_are_archived_without_overclaiming_attestation(
    tmp_path: Path,
) -> None:
    """Keep mixed raw/transformed archives inspectable but outside attestation scope."""

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
    trace.save(
        path,
        level="runnable",
        include_weights=True,
        include_activations=True,
    )
    loaded = tl.load(path)

    fields = {record.field for record in loaded.archived_activations.values()}
    assert fields == {"out", "transformed_out"}
    result = loaded.run(inputs=torch.ones(2, 3))

    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
