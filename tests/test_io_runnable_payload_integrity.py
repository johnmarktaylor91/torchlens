"""Round-6 non-security ``_io`` / ``validation`` integrity findings (H2, M3, M4, M5, L7).

Each test pins ONE finding: the published record must not be able to contradict itself, the
validator must not certify an artifact the binder cannot bind, an unattestable archive must
SAY it is unattestable rather than look like caller error, an unreachable codec path must
fail closed instead of trapping, and a deprecation warning must describe what actually
happens.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from _format3_envelope import (
    read_manifest as _manifest,
    redigest,
    write_manifest as _write_manifest,
)
from torch import nn

import torchlens as tl
from torchlens._io import MIN_TLSPEC_VERSION, ArtifactError
from torchlens._io.format3 import reader, runnable
from torchlens._io.payload_codec import get_payload_codec
from torchlens._runnable import NumericAttestationStatus
from torchlens.options import CaptureOptions
from torchlens.validation import validate_tlspec

_CAP = CaptureOptions(
    replayable=True,
    capture_container_structure=True,
    cache=False,
)


class NoStateModel(nn.Module):
    """No parameters and no buffers at all."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply a stateless graph."""

        return torch.relu(value)


@pytest.fixture(autouse=True)
def _preserve_disclosure_flags() -> Any:
    """Leave the process-wide one-time save-disclosure flags exactly as they were found.

    ``_warn_nonpersistent_buffer_disclosure_once`` and its r6 sibling fire ONCE per process.
    Tests in other files assert them with ``pytest.warns``, so a file that merely happens to
    run earlier and trip the one-shot would starve those assertions -- an order-dependent
    cross-file failure. Snapshot and restore both flags around every test here.
    """

    import torchlens._io._save_gates as bundle_module

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


def _save(path: Path, model: nn.Module, inputs: torch.Tensor, **kwargs: Any) -> Path:
    """Save one runnable artifact, tolerating the orthogonal buffer disclosure."""

    trace = tl.trace(model, inputs, capture=_CAP, **kwargs.pop("capture_kwargs", {}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        trace.save(path, level="runnable", **kwargs)
    return path


def _descriptor(path: Path) -> dict[str, Any]:
    """Read the runnable declaration separately from the artifact envelope."""

    return json.loads((path / runnable.DESCRIPTOR_PATH).read_text(encoding="utf-8"))


def _write_descriptor(path: Path, descriptor: dict[str, Any]) -> None:
    """Restamp an edited descriptor so its deeper payload checks are reached."""

    (path / runnable.DESCRIPTOR_PATH).write_text(
        runnable.canonical_json_text(descriptor), encoding="utf-8"
    )
    redigest(path, runnable.DESCRIPTOR_PATH)


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
    descriptor = _descriptor(path)
    members = descriptor["payload_layers"]["activations"]["members"]
    honest_digest = members[0]["byte_digest"]
    members[0]["byte_digest"] = "0" * 64
    _write_descriptor(path, descriptor)

    with pytest.raises(ArtifactError) as caught:
        tl.load(path)
    assert caught.value.fields["code"] == "bundle_metadata_integrity_refused"

    # The honest artifact still loads and publishes a digest that MATCHES its value.
    members[0]["byte_digest"] = honest_digest
    _write_descriptor(path, descriptor)
    loaded = tl.load(path)
    from torchlens._runnable.state import runnable_tensor_byte_digest

    archived = loaded._runnable.archived_activations
    assert archived
    for record in archived.values():
        assert runnable_tensor_byte_digest(record.value) == record.byte_digest


# --------------------------------------------------------------------------- #
# M3: the weights branch of the runnable payload validator is bidirectional.
# --------------------------------------------------------------------------- #


def test_validator_rejects_present_weights_with_missing_entries(tmp_path: Path) -> None:
    """``weights.present=true`` with the weight entries deleted is no longer certified.

    Deleting the ``state_dict`` family leaves an artifact the strict binder CANNOT
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
    assert _descriptor(path)["payload_layers"]["weights"]["present"] is True
    assert manifest["runnable"]["state_dict_tensors"]

    # Honest manifest passes.
    validate_tlspec(path)

    # Remove the complete family and its files, keeping the envelope internally true.
    # Only the descriptor still claims capture weights, the inconsistency under test.
    for relpath in manifest["runnable"].pop("state_dict"):
        manifest["files"].pop(relpath)
        (path / relpath).unlink()
    manifest["runnable"].pop("state_dict_family")
    manifest["runnable"].pop("state_dict_tensors")
    _write_manifest(path, manifest)
    with pytest.raises(ArtifactError, match="weight entries disagree with tensor slots") as caught:
        validate_tlspec(path)
    assert caught.value.fields["code"] == "manifest_schema_invalid"


def test_validator_rejects_relabelled_weight_entries(tmp_path: Path) -> None:
    """Weight LABELS are cross-checked against the descriptor's declared state names."""

    path = _save(
        tmp_path / "relabelled.tlspec",
        StateModel().eval(),
        torch.ones(2, 3),
        include_weights=True,
    )
    manifest = _manifest(path)
    references = manifest["runnable"]["state_dict_tensors"]
    manifest["runnable"]["state_dict_tensors"] = {
        f"bogus.{name}": reference for name, reference in references.items()
    }
    _write_manifest(path, manifest)
    with pytest.raises(ArtifactError, match="weight entries disagree with tensor slots") as caught:
        validate_tlspec(path)
    assert caught.value.fields["code"] == "manifest_schema_invalid"


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

    import torchlens._io._save_gates as bundle_module

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

    activations = _descriptor(path)["payload_layers"]["activations"]
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

    import torchlens._io._save_gates as bundle_module

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

    assert _descriptor(path)["payload_layers"]["activations"]["original_input_digests"]
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


# --------------------------------------------------------------------------- #
# L7: the older-version DeprecationWarning describes what actually happens.
# --------------------------------------------------------------------------- #


def test_older_version_refuses_at_rehydration_floor(tmp_path: Path) -> None:
    """A sub-floor tlspec_version refuses typed instead of warning.

    A record-contract version below ``MIN_TLSPEC_VERSION`` refuses before any
    records are read. The manifest reader names the rehydration floor rather
    than emitting an advisory and attempting to reconstruct incompatible data.
    """

    from torchlens.errors import ArtifactTooOldError

    path = tmp_path / "versioned.tlspec"
    tl.trace(ParamsOnlyModel().eval(), torch.ones(2, 3), capture=_CAP).save(path)
    manifest_dict = _manifest(path)
    manifest_dict["tlspec_version"] = MIN_TLSPEC_VERSION - 1
    _write_manifest(path, manifest_dict)

    with pytest.raises(ArtifactTooOldError, match="saved by torchlens 2.x"):
        reader.read_manifest(path)


def test_from_dict_really_requires_every_non_provenance_field(tmp_path: Path) -> None:
    """Pin the behavior the corrected warning now describes: a missing field RAISES."""

    path = tmp_path / "required.tlspec"
    tl.trace(ParamsOnlyModel().eval(), torch.ones(2, 3), capture=_CAP).save(path)
    manifest_dict = _manifest(path)
    manifest_dict.pop("records")
    _write_manifest(path, manifest_dict)

    with pytest.raises(ArtifactError):
        reader.read_manifest(path)
