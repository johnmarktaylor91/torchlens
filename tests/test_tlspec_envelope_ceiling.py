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


@pytest.mark.smoke
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


@pytest.mark.smoke
def test_manifest_policy_warns_not_raises_on_minor_mismatch() -> None:
    """R6 (2026-10-01): a minor torch drift is advisory, never a load refusal.

    The committed cross-env goldens (``tests/godobject_oracle/goldens/``,
    ``tests/test_grouping_stamp.py``'s legacy fixture, and siblings) were
    recorded on a torch 2.13 CUDA build; loading them under a different
    torch MINOR raises no error -- only the advisory ``TorchLensWarning``
    below -- which is correct by design (a same-major, different-minor
    bundle is loadable). Those golden-loading tests expect/filter this exact
    warning narrowly (``tests/_oracle_env.expect_bundle_minor_version_mismatch``);
    this test pins that the warning still fires (never silently drops) and
    that an EXACT match stays silent, independent of the runtime's own torch
    build.
    """

    import warnings as warnings_module

    import torch

    from torchlens._io.manifest import Manifest, enforce_version_policy
    from torchlens.errors import TorchLensWarning

    base = _reference_manifest_dict()
    runtime = torch.__version__.split("+", 1)[0]
    major, minor, *_ = runtime.split(".")
    drifted = f"{major}.{int(minor) + 1}.0+cu130"

    def _is_minor_mismatch(item: warnings_module.WarningMessage) -> bool:
        return issubclass(item.category, TorchLensWarning) and "minor version mismatch" in str(
            item.message
        )

    # The reference manifest's own tlspec_version may also be older than this
    # runtime's, which independently fires the unrelated ArtifactSchemaAgeWarning
    # (also a TorchLensWarning subclass) -- filter by message, not just category,
    # so that advisory never gets conflated with the one under test here.
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        enforce_version_policy(Manifest.from_dict({**base, "torch_version": drifted}))
    mismatch_warnings = [item for item in caught if _is_minor_mismatch(item)]
    assert len(mismatch_warnings) == 1
    message = str(mismatch_warnings[0].message)
    assert drifted in message

    with warnings_module.catch_warnings(record=True) as caught_exact:
        warnings_module.simplefilter("always")
        enforce_version_policy(Manifest.from_dict({**base, "torch_version": torch.__version__}))
    assert not any(_is_minor_mismatch(item) for item in caught_exact)


def test_recover_reraises_governed_refusals() -> None:
    """recover() salvages corruption, never a governed compatibility refusal."""

    from torchlens.fastlog.recover import _GOVERNED_COMPATIBILITY_REFUSALS

    assert ArtifactVersionBelowFloorError in _GOVERNED_COMPATIBILITY_REFUSALS
    assert ArtifactVersionAboveRuntimeError in _GOVERNED_COMPATIBILITY_REFUSALS
    assert ArtifactRuntimeIncompatibleError in _GOVERNED_COMPATIBILITY_REFUSALS


@pytest.mark.smoke
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
