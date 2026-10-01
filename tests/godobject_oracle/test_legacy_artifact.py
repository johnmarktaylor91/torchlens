"""Frozen legacy-artifact gate: pre-refactor .tlspec files must always load.

The two committed artifacts under ``goldens/`` were produced at the
pre-columnar baseline (tree ``db2bc7a5``) and are NEVER regenerated: they ARE
the old format. Every wave of the columnar re-plumbing must load them through
the (eventually one-way legacy) decoder with a byte-identical public surface
and, for the runnable artifact, a VERIFIED faithful run.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path

import pytest
import torch
from _oracle_env import expect_bundle_minor_version_mismatch
from surface_oracle._snapshot import canonical_dump, snapshot_trace_surface

import torchlens as tl
from torchlens import kernel_telemetry as _kernel_telemetry

from .test_aliases import _SEED

_GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"
_UPDATE_ENV = "TORCHLENS_UPDATE_LEGACY_ARTIFACT_ORACLE"

_ANALYSIS_ARTIFACT = _GOLDEN_DIR / "legacy_baseline_cnn.tlspec"
_RUNNABLE_ARTIFACT = _GOLDEN_DIR / "legacy_baseline_cnn_runnable.tlspec"
_LOADED_SURFACE_GOLDEN = _GOLDEN_DIR / "legacy_baseline_cnn_loaded.json"


@pytest.mark.smoke
def test_legacy_analysis_artifact_loads_byte_identically() -> None:
    """The frozen analysis artifact loads with an identical public surface."""

    assert _kernel_telemetry.KernelLaunch.__name__ == "KernelLaunch"

    from _oracle_env import (
        flag_armed,
        guard_wrap_state_for_golden_update,
        require_env_golden,
        require_update_reason,
        resolve_env_golden,
        write_provenance,
    )

    regen = flag_armed(os.environ, _UPDATE_ENV)
    if regen:
        # The loaded-surface snapshot is generated in-process: refuse to
        # generate golden bytes on a torch already wrapped by earlier tests
        # (SF-53), and require the WHY before the load runs.
        guard_wrap_state_for_golden_update(_UPDATE_ENV)
        require_update_reason(_UPDATE_ENV)
    assert _ANALYSIS_ARTIFACT.exists(), "frozen legacy artifact missing"
    with expect_bundle_minor_version_mismatch():
        loaded = tl.load(str(_ANALYSIS_ARTIFACT))
    actual = canonical_dump(snapshot_trace_surface(loaded))

    if regen:
        golden_path, _ = resolve_env_golden(_GOLDEN_DIR, _LOADED_SURFACE_GOLDEN.name)
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(actual + "\n")
        write_provenance(
            golden_path.parent,
            "tests/godobject_oracle legacy",
            _UPDATE_ENV,
            require_update_reason(_UPDATE_ENV),
        )
        pytest.skip(f"updated legacy loaded-surface golden; re-run without {_UPDATE_ENV} to verify")
    golden_path = require_env_golden(_GOLDEN_DIR, _LOADED_SURFACE_GOLDEN.name, _UPDATE_ENV)
    if not golden_path.exists():
        # Reason BEFORE bytes (b10 R78 round-4): a reasonless record run must
        # fail with the working tree untouched.
        reason = require_update_reason(_UPDATE_ENV)
        golden_path.write_text(actual + "\n")
        write_provenance(
            golden_path.parent,
            "tests/godobject_oracle legacy",
            _UPDATE_ENV,
            reason,
        )
        pytest.skip(f"recorded first-run loaded-surface golden for this environment: {golden_path}")
    expected = golden_path.read_text().rstrip("\n")
    if actual != expected:
        diff = "\n".join(
            list(
                difflib.unified_diff(
                    expected.splitlines(),
                    actual.splitlines(),
                    fromfile="golden",
                    tofile="actual",
                    lineterm="",
                )
            )[:60]
        )
        raise AssertionError(f"legacy loaded surface diverged:\n{diff}")


@pytest.mark.smoke
def test_legacy_artifact_site_key_refusal_teaches() -> None:
    """Pre-site-key artifacts refuse ``site_key``/``site_peers`` TEACHINGLY.

    The frozen v6 artifact predates the site_key_v1 grouping surface, so the
    Layer accessors must refuse typed (``site_key_unavailable``) -- never
    return ``None``, which would present the absence of a key as a fact about
    the model -- and the message must teach at the point of failure: name the
    tlspec v7 pre-release persistence boundary and the re-capture remedy.
    """

    from torchlens._errors import InvalidArgumentError

    with expect_bundle_minor_version_mismatch():
        loaded = tl.load(str(_ANALYSIS_ARTIFACT))
    layer = loaded[loaded.layer_labels[0]]
    with pytest.raises(InvalidArgumentError) as site_exc:
        _ = layer.site_key
    assert site_exc.value.fields["code"] == "site_key_unavailable"
    message = str(site_exc.value)
    assert "site_key_v1" in message
    assert "tlspec v7" in message
    assert "re-capture" in message
    with pytest.raises(InvalidArgumentError) as peers_exc:
        _ = layer.site_peers
    assert peers_exc.value.fields["code"] == "site_key_unavailable"
    peers_message = str(peers_exc.value)
    assert "tlspec v7" in peers_message
    assert "re-capture" in peers_message


@pytest.mark.smoke
def test_legacy_runnable_artifact_runs_verified() -> None:
    """The frozen runnable artifact stages state and replays VERIFIED.

    The run uses a CHANGED input on purpose. The artifact's archived
    ``selected_activation_v2`` digests are byte-exact facts of the ORIGIN
    torch build: an original-input run is attestation-eligible and the
    byte comparison deterministically fails on any other build, which is
    the tripwire working, not a replay bug (a fresh save/load/run on this
    build attests ATTESTED, covered by the runnable attestation suites).
    A changed-input run is ``not_applicable`` by the runnable contract's
    own vocabulary, so this gate asserts exactly what a frozen cross-env
    artifact can honestly prove: v6 load, embedded-state staging, the
    archived-activation family rehydrating, and a VERIFIED replay.
    """

    assert _RUNNABLE_ARTIFACT.exists(), "frozen runnable artifact missing"
    with expect_bundle_minor_version_mismatch():
        loaded = tl.load(str(_RUNNABLE_ARTIFACT))
    assert loaded.archived_activations, "legacy archived-activation family failed to load"
    torch.manual_seed(_SEED)
    x = torch.linspace(-0.5, 0.5, 16).reshape(1, 1, 4, 4)
    result = loaded.run(inputs=x)
    assert result.report.path_faithfulness.name == "VERIFIED"
    assert result.report.numeric_attestation.name == "NOT_APPLICABLE", (
        "changed-input legacy replay must report not_applicable, never a "
        "cross-build byte attestation verdict"
    )
