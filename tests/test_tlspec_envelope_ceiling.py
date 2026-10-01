"""G3 ceiling consolidation: one typed above-ceiling refusal at every door.

Every "artifact is newer than this runtime" site raises
``ArtifactVersionAboveRuntimeError`` through the one ``above_ceiling_error``
chokepoint; runtime-axis refusals (torch major mismatch) are the distinct
``ArtifactRuntimeIncompatibleError``; and ``tl.fastlog.recover`` re-raises
governed compatibility refusals instead of salvaging them (D-ECO-9).
"""

from __future__ import annotations

import pytest

from torchlens._io import (
    TLSPEC_VERSION,
    ArtifactRuntimeIncompatibleError,
    ArtifactVersionAboveRuntimeError,
    ArtifactVersionBelowFloorError,
    TorchLensIOError,
    above_ceiling_error,
    read_tlspec_version,
)

pytestmark = [pytest.mark.smoke]

FUTURE_VERSION = TLSPEC_VERSION + 1


def test_chokepoint_builds_structured_refusal() -> None:
    err = above_ceiling_error(observed=FUTURE_VERSION, subject="Bundle", path="/tmp/x")
    assert isinstance(err, ArtifactVersionAboveRuntimeError)
    assert isinstance(err, TorchLensIOError)
    assert err.fields["code"] == "artifact_version_above_runtime"
    assert err.fields["observed"] == FUTURE_VERSION
    assert err.fields["ceiling_tlspec_version"] == TLSPEC_VERSION
    assert err.fields["path"] == "/tmp/x"
    assert err.fields["remedy"]


def test_pickle_state_gate_raises_typed() -> None:
    with pytest.raises(ArtifactVersionAboveRuntimeError) as excinfo:
        read_tlspec_version({"tlspec_version": FUTURE_VERSION}, cls_name="Trace")
    assert excinfo.value.fields["code"] == "artifact_version_above_runtime"


def test_validation_entry_raises_typed_not_bare_valueerror() -> None:
    """The site that used to raise bare ValueError and get laundered codeless."""

    from torchlens.validation import _validate_tlspec_version_ceiling

    with pytest.raises(ArtifactVersionAboveRuntimeError) as excinfo:
        _validate_tlspec_version_ceiling(FUTURE_VERSION)
    assert excinfo.value.fields["code"] == "artifact_version_above_runtime"
    # The load path's ``except ValueError`` laundering wrap can never catch it.
    assert not isinstance(excinfo.value, ValueError)


def _reference_manifest_dict() -> dict:
    """A real harvested writer manifest (main_portable, tlspec 8)."""

    import json
    from pathlib import Path

    path = (
        Path(__file__).parent
        / "release_goldens"
        / "reference_manifests"
        / "main_portable.manifest.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_policy_gate_raises_typed() -> None:
    from torchlens._io.manifest import Manifest, enforce_version_policy

    base = _reference_manifest_dict()
    with pytest.raises(ArtifactVersionAboveRuntimeError):
        Manifest.from_dict({**base, "tlspec_version": FUTURE_VERSION})
    with pytest.raises(ArtifactRuntimeIncompatibleError) as drift:
        enforce_version_policy(Manifest.from_dict({**base, "torch_version": "1.0.0"}))
    assert drift.value.fields["code"] == "bundle_torch_incompatible"


def test_recover_reraises_governed_refusals() -> None:
    """recover() salvages corruption, never a governed compatibility refusal."""

    from torchlens.fastlog.recover import _GOVERNED_COMPATIBILITY_REFUSALS

    assert ArtifactVersionBelowFloorError in _GOVERNED_COMPATIBILITY_REFUSALS
    assert ArtifactVersionAboveRuntimeError in _GOVERNED_COMPATIBILITY_REFUSALS
    assert ArtifactRuntimeIncompatibleError in _GOVERNED_COMPATIBILITY_REFUSALS


def test_recover_refuses_future_version_bundle(tmp_path) -> None:
    """End-to-end: a future-versioned fastlog bundle refuses out of recover()."""

    import json

    bundle = tmp_path / "future.tlspec"
    bundle.mkdir()
    # A real writer manifest with its declared version bumped above the
    # runtime ceiling; recover() must re-raise, not salvage.
    (bundle / "manifest.json").write_text(
        json.dumps({**_reference_manifest_dict(), "tlspec_version": FUTURE_VERSION}),
        encoding="utf-8",
    )
    (bundle / "fastlog_index.jsonl").write_text("", encoding="utf-8")

    from torchlens.fastlog import recover

    with pytest.raises(ArtifactVersionAboveRuntimeError):
        recover(bundle)


def test_public_errors_registry_serves_new_classes() -> None:
    import torchlens.errors as errors

    assert errors.ArtifactVersionAboveRuntimeError is ArtifactVersionAboveRuntimeError
    assert errors.ArtifactRuntimeIncompatibleError is ArtifactRuntimeIncompatibleError
