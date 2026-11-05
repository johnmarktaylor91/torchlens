"""G3 ceiling consolidation: one typed above-ceiling refusal at every door.

Every "artifact is newer than this runtime" site raises
``ArtifactTooNewError`` through the one ``above_ceiling_error``
chokepoint; runtime-axis refusals (torch major mismatch) are the distinct
``ArtifactRuntimeIncompatibleError``; and ``tl.recording.recover`` re-raises
governed compatibility refusals instead of salvaging them (D-ECO-9).
"""

from __future__ import annotations

import pytest

from torchlens._io import (
    TLSPEC_VERSION,
    ArtifactRuntimeIncompatibleError,
    ArtifactTooNewError,
    read_tlspec_version,
)

FUTURE_VERSION = TLSPEC_VERSION + 1


def test_pickle_state_gate_raises_typed() -> None:
    with pytest.raises(ArtifactTooNewError) as excinfo:
        read_tlspec_version({"tlspec_version": FUTURE_VERSION}, cls_name="Trace")
    assert excinfo.value.fields["code"] == "artifact_version_above_runtime"


def _reference_manifest_dict() -> dict:
    """A minimal valid manifest in the shape ``Manifest.from_dict`` reads.

    The 3.0 artifact reader parses its own manifest and runs no torch-version policy;
    ``Manifest`` and ``enforce_version_policy`` still gate the fastlog recovery manifest, so
    the reference is a fastlog-shaped dict at the current format version.
    """

    import torch

    return {
        "tlspec_version": TLSPEC_VERSION,
        "torchlens_version": "3.0.0",
        "torch_version": torch.__version__,
        "python_version": "3.10.0",
        "platform": "linux-x86_64",
        "created_at": "2026-01-01T00:00:00Z",
        "bundle_format": "fastlog-directory",
        "n_layers": 0,
        "n_out_blobs": 0,
        "n_grad_blobs": 0,
        "n_auxiliary_blobs": 0,
        "tensors": [],
        "unsupported_tensors": [],
    }


def test_manifest_policy_gate_raises_typed() -> None:
    from torchlens._io.manifest import Manifest, enforce_version_policy

    base = _reference_manifest_dict()
    with pytest.raises(ArtifactTooNewError):
        Manifest.from_dict({**base, "tlspec_version": FUTURE_VERSION})
    with pytest.raises(ArtifactRuntimeIncompatibleError) as drift:
        enforce_version_policy(Manifest.from_dict({**base, "torch_version": "1.0.0"}))
    assert drift.value.fields["code"] == "bundle_torch_incompatible"


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

    # Filter by message, not just category, so no other TorchLensWarning is
    # conflated with the one under test here.
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


@pytest.mark.parametrize("subject", ["Bundle", "Artifact"])
def test_oracle_minor_mismatch_filter_matches_each_loader_and_nothing_else(subject) -> None:
    """The goldens' advisory filter silences the minor-mismatch warning of either loader.

    The 2.x manifest path opens the advisory with ``Bundle`` and the format-3 path with
    ``Artifact`` (``check_runtime_identity(subject=...)``); a golden recorded on another
    torch minor fires one or the other. Any other TorchLens warning still escapes.
    """

    import warnings as warnings_module

    import torch
    from _oracle_env import expect_bundle_minor_version_mismatch

    from torchlens._io._runtime_identity import (
        TORCHLENS_VERSION,
        check_runtime_identity,
        runtime_python_version,
    )
    from torchlens.errors import TorchLensWarning

    runtime = torch.__version__.split("+", 1)[0]
    major, minor, *_ = runtime.split(".")
    stamps = {
        "torch_version": f"{major}.{int(minor) + 1}.0+cpu",
        "torchlens_version": TORCHLENS_VERSION,
        "python_version": runtime_python_version(),
    }
    with warnings_module.catch_warnings(record=True) as unfiltered:
        warnings_module.simplefilter("always")
        check_runtime_identity(subject=subject, **stamps)
    assert [str(item.message).split(" ", 1)[0] for item in unfiltered] == [subject]

    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        with expect_bundle_minor_version_mismatch():
            check_runtime_identity(subject=subject, **stamps)
            warnings_module.warn(f"{subject} unrelated advisory", TorchLensWarning)
    assert [str(item.message) for item in caught] == [f"{subject} unrelated advisory"]


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

    from torchlens.recording import recover

    with pytest.raises(ArtifactTooNewError):
        recover(bundle)
