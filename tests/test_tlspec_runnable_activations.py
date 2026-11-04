"""Stage 8 optional runnable activation-payload and numeric-attestation tests."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from _format3_envelope import read_record, read_runnable
from torch import nn

import torchlens as tl
from torchlens._io.runnable import assert_sparse_core_has_no_tensor_payload
from torchlens._runnable import (
    NumericAttestationStatus,
)
from torchlens.errors import ReplayValueMismatchError
from torchlens.options import CaptureOptions


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
            replayable=True,
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
    """Inspect the authoritative JSON record cells without invoking pickle."""

    return {kind: read_record(path, kind) for kind in _manifest(path)["records"]}


def _physical_outs(trace: tl.Trace) -> tuple[torch.Tensor | None, ...]:
    """Snapshot physical Op outs without triggering unsaved-payload errors."""

    values: list[torch.Tensor | None] = []
    for op in trace._layer_list:
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


def test_include_activations_persists_exact_save_selected_family_and_digests(
    tmp_path: Path,
) -> None:
    """Archive only existing ``save=`` decisions with explicit slot membership."""

    model = ActivationPayloadModel().eval()
    trace = _capture(model, torch.ones(2, 3), save=tl.func("relu"))
    path = tmp_path / "selected.tlspec"

    # r6 H2: this selective archive now also DISCLOSES that it can never be numerically
    # attested (visible-not-fatal per the pyproject filterwarnings convention for
    # option-triggered informational warnings); the disclosure itself is asserted in
    # tests/test_io_runnable_payload_integrity.py.
    trace.save(path, level="runnable", include_activations=True)

    manifest = _manifest(path)
    activation_layer = read_runnable(path)["payload_layers"]["activations"]
    members = activation_layer["members"]
    selected_labels = {str(op.label) for op in trace._layer_list if bool(op.has_saved_activation)}
    assert activation_layer["present"] is True
    assert activation_layer["schema"] == "selected_activation_v2"
    assert {member["op_label"] for member in members} == selected_labels
    assert all(len(member["byte_digest"]) == 64 for member in members)
    assert {member["slot_id"] for member in members} == {
        f"slot:{label}" for label in selected_labels
    }
    entries = manifest["runnable"]["activations_tensors"]
    assert set(entries) == {member["blob_id"] for member in members}
    assert all(ref["blob"].startswith("runnable/activations-") for ref in entries.values())
    assert_sparse_core_has_no_tensor_payload(read_runnable(path))
    assert_sparse_core_has_no_tensor_payload(_metadata(path))
    assert "func" not in _metadata(path)["op"]["columns"]
    tl.validation.validate_tlspec(path)


def test_include_activations_default_is_absent_and_requires_runnable_level(
    tmp_path: Path,
) -> None:
    """Keep activation payloads absent by default and reject analysis save levels."""

    trace = _capture(ActivationPayloadModel().eval(), torch.ones(2, 3))
    path = tmp_path / "sparse.tlspec"
    trace.save(path, level="runnable")

    manifest = _manifest(path)
    assert read_runnable(path)["payload_layers"]["activations"] == {
        "present": False,
        "schema": "selected_activation_v2",
    }
    assert "activations_tensors" not in manifest["runnable"]
    assert tl.load(path).archived_activations == {}
    with pytest.raises(ValueError, match="set level='runnable' or set include_activations=False"):
        trace.save(tmp_path / "portable.tlspec", include_activations=True)


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.smoke_cells(
    "test_seeded_rng_operations_skip_original_input_numeric_attestation[randint]"
)
@pytest.mark.parametrize(
    "operation",
    (
        "randint",
        "dropout3d",
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
        on_divergence="return",
    )

    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_corrupt_archived_digest_fails_tripwire_and_rolls_back(tmp_path: Path) -> None:
    """Fail on the first declared byte-digest mismatch without mutating the source Trace.

    r6 M4 added an EARLIER, load-time verification of the loaded tensor against its own
    declared ``byte_digest`` (covered by
    ``tests/test_io_runnable_payload_integrity.py::test_archived_activation_digest_is_verified_against_the_loaded_tensor``),
    so a manifest-level digest edit no longer survives to ``.run()``. This test keeps the
    RUN-time half of the tripwire -- the ``ReplayValueMismatchError`` fields, the source
    Trace rollback, and the not-poisoned guarantee -- by planting the same corrupt declared
    digest on the already-loaded descriptor, which is exactly the state the loader used to
    hand ``.run()``.
    """

    model = ActivationPayloadModel().eval()
    inputs = torch.ones(2, 3)
    path = tmp_path / "corrupt-digest.tlspec"
    _capture(model, inputs).save(
        path,
        level="runnable",
        include_weights=True,
        include_activations=True,
    )
    loaded = tl.load(path)
    descriptor = loaded._sparse_run_descriptor
    assert descriptor is not None
    layer = descriptor.payload_layers.activations
    first_member = layer.members[0]
    corrupt_layer = dataclasses.replace(
        layer,
        members=(dataclasses.replace(first_member, byte_digest="0" * 64),) + layer.members[1:],
    )
    loaded._runnable.descriptor = dataclasses.replace(
        descriptor,
        payload_layers=dataclasses.replace(descriptor.payload_layers, activations=corrupt_layer),
    )
    before = _physical_outs(loaded)

    with pytest.raises(ReplayValueMismatchError) as caught:
        loaded.run(inputs=inputs)

    mismatch = caught.value.fields["first_mismatch"]
    details = dict(mismatch.details)
    assert caught.value.fields["code"] == "replay_value_mismatch"
    assert (
        caught.value.fields["numeric_attestation"]
        is NumericAttestationStatus.NUMERIC_ATTESTATION_FAILED
    )
    assert mismatch.code.value == "replay_value_mismatch"
    assert details["slot_id"] == first_member.slot_id
    assert details["expected_digest"] == "0" * 64
    assert details["archived_digest"] != details["expected_digest"]
    assert _physical_outs(loaded) == before
    assert not bool(loaded._runnable.poisoned)


@pytest.mark.filterwarnings(
    # A08 random-role-init disclosure (runnable_random_init_run): this test
    # exercises a weight-free runnable run DELIBERATELY; the warning is the
    # feature under test elsewhere (tests/test_persist_honesty_runnable_random.py).
    "default:This runnable artifact carries no model weights"
)
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
        capture=CaptureOptions(
            replayable=True,
            capture_container_structure=True,
            cache=False,
            activation_transform=lambda value: value.mean(),
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
