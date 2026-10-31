"""Anything older than the current format refuses as a file saved by torchlens 2.x.

Artifacts written at the current ``tlspec_version`` load; any older stamp refuses with the
typed ``ArtifactTooOldError`` that names the 2.x origin, instead of being resurrected through
legacy field-alias ladders.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.op import Op
from torchlens._io import (
    MIN_TLSPEC_VERSION,
    TLSPEC_VERSION,
    ArtifactError,
)
from torchlens._io._canonical_pickle import dump_canonical_metadata
from torchlens.errors import ArtifactTooOldError

FLOOR_MATCH = "saved by torchlens 2.x"


def _stamp_between_floor(path: Path, version: int) -> None:
    """Re-stamp a saved bundle as a COHERENT ``tlspec_version=version`` artifact.

    Every torchlens writer stamps ``manifest.json`` AND the pickled root state
    from the same ``TLSPEC_VERSION`` (a stampless state refuses below floor),
    and the one lawful manifest-only bump (``tl.migrate``) leaves a witness.
    A manifest-only edit therefore models NO real writer and trips the
    manifest/metadata integrity anchor (``bundle_manifest_metadata_mismatch``)
    before any age advisory; both stamps move together here.
    """

    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tlspec_version"] = version
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    metadata_path = path / "metadata.pkl"
    state = pickle.loads(metadata_path.read_bytes())  # trusted bytes this test wrote
    state["tlspec_version"] = version
    with metadata_path.open("wb") as handle:
        dump_canonical_metadata(state, handle)


def _build_trace() -> tl.Trace:
    """Capture a small deterministic trace."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    return tl.trace(
        model, torch.randn(2, 4), capture=tl.options.CaptureOptions(layers_to_save="all")
    )


@pytest.mark.smoke
def test_floor_error_is_typed_and_public() -> None:
    """The floor error is a ``ArtifactError`` reachable via ``tl.errors``."""

    assert issubclass(ArtifactTooOldError, ArtifactError)
    assert tl.errors.ArtifactTooOldError is ArtifactTooOldError
    assert MIN_TLSPEC_VERSION == 6


def test_current_artifacts_load_clean(tmp_path: Path) -> None:
    """A current save round-trips with no floor refusal and no version warning."""

    trace = _build_trace()
    path = tmp_path / "current.tlspec"
    trace.save(path)

    loaded = tl.load(path)

    assert isinstance(loaded, tl.Trace)
    assert [layer.layer_label for layer in loaded.layer_list] == [
        layer.layer_label for layer in trace.layer_list
    ]


def test_between_floor_stamp_refuses_as_2x(tmp_path: Path) -> None:
    """A coherent artifact stamped older than the current format refuses, never loads.

    Every stamp from the floor up to the current one was written by torchlens 2.x, so each
    refuses at load with the one typed error naming the 2.x origin and the remedy.
    """

    trace = _build_trace()
    assert MIN_TLSPEC_VERSION < TLSPEC_VERSION, "no between-floor range to exercise"
    for version in range(MIN_TLSPEC_VERSION, TLSPEC_VERSION):
        path = tmp_path / f"between_floor_{version}.tlspec"
        trace.save(path)
        _stamp_between_floor(path, version)
        with pytest.raises(ArtifactTooOldError, match=FLOOR_MATCH) as excinfo:
            tl.load(path)
        assert excinfo.value.fields["code"] == "artifact_version_below_floor"
        assert excinfo.value.fields["observed"] == f"tlspec_version={version}"
    trace.cleanup()


@pytest.mark.parametrize("cls_and_state", ["trace", "op", "layer"])
@pytest.mark.parametrize("version", [None, 0, 2, 5, 9])
def test_pre_floor_object_states_refuse_typed(cls_and_state: str, version: int | None) -> None:
    """Sub-floor and unversioned object states raise the typed floor error."""

    trace = _build_trace()
    source = {
        "trace": trace,
        "op": trace.ops[0],
        "layer": trace.layer_list[0],
    }[cls_and_state]
    state = source.__getstate__()
    if version is None:
        state.pop("tlspec_version", None)
    else:
        state["tlspec_version"] = version

    restored = type(source).__new__(type(source))
    with pytest.raises(ArtifactTooOldError, match=FLOOR_MATCH):
        restored.__setstate__(state)


def test_current_version_object_state_still_restores() -> None:
    """The same forged-state path succeeds at exactly the floor version."""

    trace = _build_trace()
    op_state = trace.ops[0].__getstate__()
    assert op_state["tlspec_version"] == TLSPEC_VERSION

    restored = Op.__new__(Op)
    restored.__setstate__(op_state)
    assert restored.layer_label == trace.ops[0].layer_label


@pytest.mark.smoke
def test_manifest_missing_kind_still_loads_as_trace(tmp_path: Path) -> None:
    """A manifest with ``kind`` deleted is NOT the legacy pre-tlspec format.

    ``tlspec_version`` is itself a tlspec-schema-only marker (the genuine
    v2.16 ModelLog format never carried it); a manifest that has it but lost
    its ``kind`` key is an older-but-still-modern unified manifest, and
    ``kind`` defaults to ``"trace"`` (the only kind that existed before the
    discriminator). Fixes a fast-tier fuzz finding (2026-10,
    ``test_tlspec_parse_fuzz.py::test_every_toplevel_key_deletion_is_adjudicated``):
    this exact shape used to misclassify as ``v2.16_modellog_portable`` and
    raise a false "pre-tlspec ModelLog format" ``ArtifactTooOldError``.
    """

    trace = _build_trace()
    path = tmp_path / "no_kind.tlspec"
    trace.save(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "trace"
    del manifest["kind"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = tl.load(path)
    assert isinstance(loaded, tl.Trace)
    loaded.cleanup()
    trace.cleanup()


@pytest.mark.smoke
def test_manifest_missing_kind_and_below_floor_still_refuses_typed(tmp_path: Path) -> None:
    """Losing ``kind`` must not dodge the real numeric floor check either."""

    trace = _build_trace()
    path = tmp_path / "no_kind_below_floor.tlspec"
    trace.save(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["kind"]
    manifest["tlspec_version"] = MIN_TLSPEC_VERSION - 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactTooOldError) as excinfo:
        tl.load(path)
    # The REAL numeric floor refusal, never the unrelated legacy
    # "pre-tlspec ModelLog format" message a misclassification would give.
    assert excinfo.value.fields["observed"] == f"tlspec_version={MIN_TLSPEC_VERSION - 1}"
    trace.cleanup()


@pytest.mark.smoke
def test_pre_floor_bundle_manifest_refuses_typed(tmp_path: Path) -> None:
    """A bundle whose manifest claims a sub-floor tlspec_version refuses."""

    trace = _build_trace()
    path = tmp_path / "forged.tlspec"
    trace.save(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tlspec_version"] = MIN_TLSPEC_VERSION - 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactTooOldError, match=FLOOR_MATCH):
        tl.load(path)


def test_below_floor_refusal_carries_structured_fields(tmp_path: Path) -> None:
    """The below-floor refusal must carry a branchable code, path, and remedy (R65).

    Every raise site used to hand-copy the message with an empty ``fields`` and
    four of six dropped the artifact path they held in scope.
    """

    trace = _build_trace()
    path = tmp_path / "forged.tlspec"
    trace.save(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tlspec_version"] = MIN_TLSPEC_VERSION - 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactTooOldError) as excinfo:
        tl.load(path)
    fields = excinfo.value.fields
    assert fields["code"] == "artifact_version_below_floor"
    assert fields["observed"] == f"tlspec_version={MIN_TLSPEC_VERSION - 1}"
    assert fields["floor_tlspec_version"] == MIN_TLSPEC_VERSION
    assert fields["remedy"]
    # The one site with the artifact path in scope now reports it.
    assert fields["path"] == str(path)


@pytest.mark.smoke
def test_legacy_intervention_specs_refuse_as_2x_at_the_format_stamp() -> None:
    """A 2.16 intervention spec (``format_version`` 1) refuses with the one 2.x error.

    The Trace floor never sees it: the spec loader reads the format stamp first and
    refuses before any stored edit keyword is decoded.
    """

    from _refuses_as_2x import assert_refuses_as_2x

    fixture_path = Path(__file__).parent / "goldens" / "legacy" / "intervention_v2_16.tlspec"
    error = assert_refuses_as_2x(fixture_path)
    assert error.fields["observed"] == "intervention format_version=1"
    assert error.fields["path"] == str(fixture_path)
