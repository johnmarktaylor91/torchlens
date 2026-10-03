"""Round-6 non-security ``_io`` / ``validation`` integrity findings (H2, M3, M4, M5, L7).

Each test pins ONE finding: the published record must not be able to contradict itself, the
validator must not certify an artifact the binder cannot bind, an unattestable archive must
SAY it is unattestable rather than look like caller error, an unreachable codec path must
fail closed instead of trapping, and a deprecation warning must describe what actually
happens.
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import MIN_TLSPEC_VERSION, TorchLensIOError
from torchlens._io.manifest import Manifest
from torchlens._io.payload_codec import get_payload_codec
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus
from torchlens.validation import _validate_runnable_payload_entries, validate_tlspec

_CAP = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


@pytest.fixture(autouse=True)
def _preserve_disclosure_flags() -> Any:
    """Leave the process-wide one-time save-disclosure flags exactly as they were found.

    ``_warn_nonpersistent_buffer_disclosure_once`` and its r6 sibling fire ONCE per process.
    Tests in other files assert them with ``pytest.warns``, so a file that merely happens to
    run earlier and trip the one-shot would starve those assertions -- an order-dependent
    cross-file failure. Snapshot and restore both flags around every test here.
    """

    import torchlens._io.bundle as bundle_module

    before = (
        bundle_module._NONPERSISTENT_DISCLOSURE_WARNED,
        bundle_module._UNATTESTABLE_ACTIVATION_DISCLOSURE_WARNED,
    )
    yield
    (
        bundle_module._NONPERSISTENT_DISCLOSURE_WARNED,
        bundle_module._UNATTESTABLE_ACTIVATION_DISCLOSURE_WARNED,
    ) = before


class StateModel(nn.Module):
    """Parameters plus a persistent AND a non-persistent buffer."""

    def __init__(self) -> None:
        """Build byte-stable capture state."""

        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.register_buffer("scale", torch.tensor([1.5, -0.5]))
        self.register_buffer("gain", torch.tensor([2.0, 2.0]), persistent=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, 2.0, -1.0], [-2.0, 0.5, 3.0]]))
            self.linear.bias.copy_(torch.tensor([0.25, -0.75]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply a deterministic state-bearing graph."""

        return torch.relu(self.linear(value)) * self.scale * self.gain


class ParamsOnlyModel(nn.Module):
    """Parameters, no buffers."""

    def __init__(self) -> None:
        """Build parameter-only capture state."""

        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply a deterministic parameter-only graph."""

        return torch.relu(self.linear(value))


class NoStateModel(nn.Module):
    """No parameters and no buffers at all."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply a stateless graph."""

        return torch.relu(value)


def _save(path: Path, model: nn.Module, inputs: torch.Tensor, **kwargs: Any) -> Path:
    """Save one runnable artifact, tolerating the orthogonal buffer disclosure."""

    trace = tl.trace(model, inputs, capture=_CAP, **kwargs.pop("capture_kwargs", {}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        trace.save(path, level="runnable", **kwargs)
    return path


def _manifest(path: Path) -> dict[str, Any]:
    """Read one artifact's public manifest."""

    with (path / "manifest.json").open(encoding="utf-8") as handle:
        value = json.load(handle)
    assert isinstance(value, dict)
    return value


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Overwrite one artifact's public manifest."""

    with (path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


# --------------------------------------------------------------------------- #
# M4: the published ArchivedActivation record cannot contradict itself.
# --------------------------------------------------------------------------- #


def test_archived_activation_digest_is_verified_against_the_loaded_tensor(
    tmp_path: Path,
) -> None:
    """A tampered member ``byte_digest`` is refused AT LOAD, not published beside ``.value``.

    Before r6 the loader verified only the blob FILE's ``sha256`` against the manifest and
    then republished the member's unverified ``byte_digest`` on the public
    ``ArchivedActivation`` next to the tensor it claims to describe -- so editing one
    member digest (blobs untouched) loaded silently and produced a self-contradicting
    inspection record.
    """

    model = StateModel().eval()
    inputs = torch.ones(2, 3)
    path = _save(
        tmp_path / "tampered-digest.tlspec",
        model,
        inputs,
        include_weights=True,
        include_activations=True,
    )
    manifest = _manifest(path)
    members = manifest["run"]["payload_layers"]["activations"]["members"]
    honest_digest = members[0]["byte_digest"]
    members[0]["byte_digest"] = "0" * 64
    _write_manifest(path, manifest)

    with pytest.raises(TorchLensIOError, match="byte-digest mismatch"):
        tl.load(path)

    # The honest artifact still loads and publishes a digest that MATCHES its value.
    members[0]["byte_digest"] = honest_digest
    _write_manifest(path, manifest)
    loaded = tl.load(path)
    from torchlens._runnable_state import runnable_tensor_byte_digest

    archived = loaded._runnable.archived_activations
    assert archived
    for record in archived.values():
        assert runnable_tensor_byte_digest(record.value) == record.byte_digest


# --------------------------------------------------------------------------- #
# M3: the weights branch of the runnable payload validator is bidirectional.
# --------------------------------------------------------------------------- #


def test_validator_rejects_present_weights_with_missing_entries(tmp_path: Path) -> None:
    """``weights.present=true`` with the weight entries deleted is no longer certified.

    Deleting the ``runnable_weight`` entries leaves an artifact the strict binder CANNOT
    bind, yet the one-directional check accepted it -- the gate users are told to run
    certified an unbindable artifact.
    """

    path = _save(
        tmp_path / "weights.tlspec",
        StateModel().eval(),
        torch.ones(2, 3),
        include_weights=True,
    )
    manifest = _manifest(path)
    assert manifest["run"]["payload_layers"]["weights"]["present"] is True
    assert [entry for entry in manifest["tensors"] if entry["kind"] == "runnable_weight"]

    # Honest manifest passes.
    _validate_runnable_payload_entries(manifest)

    stripped = copy.deepcopy(manifest)
    stripped["tensors"] = [
        entry for entry in stripped["tensors"] if entry.get("kind") != "runnable_weight"
    ]
    with pytest.raises(ValueError, match="weight entries disagree with tensor slots"):
        _validate_runnable_payload_entries(stripped)


def test_validator_rejects_relabelled_weight_entries(tmp_path: Path) -> None:
    """Weight LABELS are cross-checked against the descriptor's declared state names."""

    path = _save(
        tmp_path / "relabelled.tlspec",
        StateModel().eval(),
        torch.ones(2, 3),
        include_weights=True,
    )
    manifest = _manifest(path)
    relabelled = copy.deepcopy(manifest)
    for entry in relabelled["tensors"]:
        if entry.get("kind") == "runnable_weight":
            entry["label"] = f"bogus.{entry['label']}"
    with pytest.raises(ValueError, match="weight entries disagree with tensor slots"):
        _validate_runnable_payload_entries(relabelled)


@pytest.mark.parametrize("include_weights", [False, True])
@pytest.mark.parametrize(
    "factory", [NoStateModel, ParamsOnlyModel, StateModel], ids=["no-state", "params", "mixed"]
)
def test_weight_invariant_holds_on_every_honest_shape(
    tmp_path: Path,
    factory: type[nn.Module],
    include_weights: bool,
) -> None:
    """The strengthened invariant is behavior-preserving across the honest shape matrix.

    ``present`` tracks the optional ``include_weights=`` FLAG (unlike the REQUIRED
    non-persistent-buffer family, whose ``present`` tracks EXISTENCE), so the honest matrix
    includes both a params-bearing model at the DEFAULT ``include_weights=False`` and a
    no-state model at ``include_weights=True``.
    """

    path = _save(
        tmp_path / f"shape-{factory.__name__}-{include_weights}.tlspec",
        factory().eval(),
        torch.ones(2, 3),
        include_weights=include_weights,
    )

    _validate_runnable_payload_entries(_manifest(path))
    validate_tlspec(path)


# --------------------------------------------------------------------------- #
# H2: a selectively-saved activation archive DISCLOSES that it is unattestable.
# --------------------------------------------------------------------------- #


def test_selective_activation_save_discloses_unattestable_archive(tmp_path: Path) -> None:
    """A ``save=`` archive without the model input warns at save and NAMES the reason at run.

    The only pre-existing selective-save coverage never called ``.run()``, so the gap was
    untested: attestation was permanently ``NOT_APPLICABLE`` with no warning, which is
    indistinguishable from "the user changed the input". The status itself is deliberately
    UNCHANGED -- only the disclosure is added.
    """

    import torchlens._io.bundle as bundle_module

    model = StateModel().eval()
    inputs = torch.ones(2, 3)
    trace = tl.trace(model, inputs, capture=_CAP, save=tl.func("relu"))
    path = tmp_path / "selective.tlspec"

    bundle_module._UNATTESTABLE_ACTIVATION_DISCLOSURE_WARNED = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace.save(path, level="runnable", include_weights=True, include_activations=True)
    messages = [str(entry.message) for entry in caught]
    assert any("records no original-input digests" in message for message in messages), messages

    manifest = _manifest(path)
    activations = manifest["run"]["payload_layers"]["activations"]
    assert activations["present"] is True
    assert activations["members"], "the archive really did ship blobs"
    assert activations["original_input_digests"] == []

    result = tl.load(path).run(inputs=inputs.clone())

    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    named = [
        check.name
        for check in result.report.contract_checks
        if check.name.startswith("numeric_attestation:not_applicable")
    ]
    assert named == ["numeric_attestation:not_applicable:no_recorded_original_input_eligibility"]
    assert all(check.passed for check in result.report.contract_checks)


def test_full_capture_activation_save_stays_attested_and_silent(tmp_path: Path) -> None:
    """The disclosure is NARROW: a full capture still records inputs and attests."""

    import torchlens._io.bundle as bundle_module

    model = StateModel().eval()
    inputs = torch.ones(2, 3)
    path = tmp_path / "full-capture.tlspec"

    bundle_module._UNATTESTABLE_ACTIVATION_DISCLOSURE_WARNED = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.trace(model, inputs, capture=_CAP).save(
            path, level="runnable", include_weights=True, include_activations=True
        )
    assert not any("records no original-input digests" in str(e.message) for e in caught)

    assert _manifest(path)["run"]["payload_layers"]["activations"]["original_input_digests"]
    result = tl.load(path).run(inputs=inputs.clone())
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


# --------------------------------------------------------------------------- #
# M5: the unreachable torch NumPy transport fails closed instead of trapping.
# --------------------------------------------------------------------------- #


def test_torch_codec_numpy_transport_fails_closed() -> None:
    """``TorchPayloadCodec.to_numpy`` / ``from_numpy`` refuse rather than trap on bfloat16.

    Both ends short-circuit ``logical_backend == "torch"``, so the pair is structurally
    unreachable; its old ``.numpy()`` body would have raised a raw
    ``TypeError: Got unsupported ScalarType BFloat16`` from deep inside torch for a dtype
    ``tensor_policy._SUPPORTED_DTYPES`` declares SUPPORTED.
    """

    import numpy as np

    codec = get_payload_codec("torch")
    assert codec.can_encode(torch.ones(2, dtype=torch.bfloat16))
    with pytest.raises(TypeError, match="no NumPy transport"):
        codec.to_numpy(torch.ones(2, dtype=torch.bfloat16))
    with pytest.raises(TypeError, match="no NumPy transport"):
        codec.from_numpy(np.zeros(2, dtype=np.float32), None, map_location=None)


def test_bfloat16_payload_round_trips_through_the_real_routing(tmp_path: Path) -> None:
    """The REAL torch routing still handles a bfloat16 payload end-to-end."""

    class Bf16(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Apply a bfloat16 op."""

            return torch.relu(value)

    inputs = torch.ones(2, 3, dtype=torch.bfloat16)
    path = tmp_path / "bf16.tlspec"
    tl.trace(Bf16(), inputs, capture=_CAP).save(path, level="runnable", include_weights=True)

    assert tl.load(path) is not None
    validate_tlspec(path)


# --------------------------------------------------------------------------- #
# L7: the older-version DeprecationWarning describes what actually happens.
# --------------------------------------------------------------------------- #


def test_older_version_refuses_at_rehydration_floor(tmp_path: Path) -> None:
    """A sub-floor tlspec_version refuses typed instead of warning.

    Since the v7 schema bump the current ``TLSPEC_VERSION`` sits above the 2.33
    rehydration floor (``MIN_TLSPEC_VERSION``), so the refusal boundary is the
    floor, not the current version: a between-floor-and-current artifact loads
    with the advisory ``DeprecationWarning``, while anything below the floor
    refuses. ``Manifest.from_dict`` refuses first with the floor named.
    """

    from torchlens.errors import ArtifactVersionBelowFloorError

    path = tmp_path / "versioned.tlspec"
    tl.trace(ParamsOnlyModel().eval(), torch.ones(2, 3), capture=_CAP).save(path)
    manifest_dict = _manifest(path)
    manifest_dict["tlspec_version"] = MIN_TLSPEC_VERSION - 1

    with pytest.raises(ArtifactVersionBelowFloorError, match="torchlens 2.33"):
        Manifest.from_dict(manifest_dict)


def test_from_dict_really_requires_every_non_provenance_field(tmp_path: Path) -> None:
    """Pin the behavior the corrected warning now describes: a missing field RAISES."""

    path = tmp_path / "required.tlspec"
    tl.trace(ParamsOnlyModel().eval(), torch.ones(2, 3), capture=_CAP).save(path)
    manifest_dict = _manifest(path)
    manifest_dict.pop("tensors")

    with pytest.raises(TorchLensIOError):
        Manifest.from_dict(manifest_dict)
