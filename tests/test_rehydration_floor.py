"""The torchlens 2.33 / tlspec_version 6 rehydration floor (drop-not-resurrect).

Artifacts written by torchlens >= 2.33 (``tlspec_version >= 6``) load; anything
older refuses with the typed ``ArtifactVersionBelowFloorError`` naming the
floor, instead of being resurrected through legacy field-alias ladders.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import (
    MIN_TLSPEC_VERSION,
    TLSPEC_VERSION,
    ArtifactSchemaAgeWarning,
    TorchLensIOError,
)
from torchlens.data_classes.op import Op
from torchlens.errors import ArtifactVersionBelowFloorError

FLOOR_MATCH = "torchlens 2.33"


def _build_trace() -> tl.Trace:
    """Capture a small deterministic trace."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    return tl.trace(
        model, torch.randn(2, 4), capture=tl.options.CaptureOptions(layers_to_save="all")
    )


@pytest.mark.smoke
def test_floor_error_is_typed_and_public() -> None:
    """The floor error is a ``TorchLensIOError`` reachable via ``tl.errors``."""

    assert issubclass(ArtifactVersionBelowFloorError, TorchLensIOError)
    assert tl.errors.ArtifactVersionBelowFloorError is ArtifactVersionBelowFloorError
    assert MIN_TLSPEC_VERSION == 6


@pytest.mark.smoke
def test_current_artifacts_load_clean(tmp_path: Path) -> None:
    """A 2.33+ save round-trips with no floor refusal and no version warning."""

    trace = _build_trace()
    path = tmp_path / "current.tlspec"
    tl.save(trace, path)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ArtifactSchemaAgeWarning)
        loaded = tl.load(path)

    assert isinstance(loaded, tl.Trace)
    assert [layer.layer_label for layer in loaded.layer_list] == [
        layer.layer_label for layer in trace.layer_list
    ]


@pytest.mark.smoke
@pytest.mark.parametrize("cls_and_state", ["trace", "op", "layer"])
@pytest.mark.parametrize("version", [None, 0, 2, 5])
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
    with pytest.raises(ArtifactVersionBelowFloorError, match=FLOOR_MATCH):
        restored.__setstate__(state)


@pytest.mark.smoke
def test_current_version_object_state_still_restores() -> None:
    """The same forged-state path succeeds at exactly the floor version."""

    trace = _build_trace()
    op_state = trace.ops[0].__getstate__()
    assert op_state["tlspec_version"] == TLSPEC_VERSION

    restored = Op.__new__(Op)
    restored.__setstate__(op_state)
    assert restored.layer_label == trace.ops[0].layer_label


@pytest.mark.smoke
def test_pre_floor_bundle_manifest_refuses_typed(tmp_path: Path) -> None:
    """A bundle whose manifest claims a sub-floor tlspec_version refuses."""

    trace = _build_trace()
    path = tmp_path / "forged.tlspec"
    tl.save(trace, path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tlspec_version"] = MIN_TLSPEC_VERSION - 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactVersionBelowFloorError, match=FLOOR_MATCH):
        tl.load(path)


@pytest.mark.smoke
def test_below_floor_refusal_carries_structured_fields(tmp_path: Path) -> None:
    """The below-floor refusal must carry a branchable code, path, and remedy (R65).

    Every raise site used to hand-copy the message with an empty ``fields`` and
    four of six dropped the artifact path they held in scope.
    """

    trace = _build_trace()
    path = tmp_path / "forged.tlspec"
    tl.save(trace, path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tlspec_version"] = MIN_TLSPEC_VERSION - 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactVersionBelowFloorError) as excinfo:
        tl.load(path)
    fields = excinfo.value.fields
    assert fields["code"] == "artifact_version_below_floor"
    assert fields["observed"] == f"tlspec_version={MIN_TLSPEC_VERSION - 1}"
    assert fields["floor_tlspec_version"] == MIN_TLSPEC_VERSION
    assert fields["remedy"]
    # The one site with the artifact path in scope now reports it.
    assert fields["path"] == str(path)


@pytest.mark.smoke
def test_between_floor_advisory_is_a_visible_user_warning(tmp_path: Path) -> None:
    """The between-floor artifact-age advisory is visible, not a ``DeprecationWarning``.

    R15-F1: the advisory deprecates no API -- it reports the AGE of one
    artifact -- and Python's default filters hide ``DeprecationWarning`` from
    end users, so the category made the advisory invisible exactly where it
    matters. It is an ``ArtifactSchemaAgeWarning`` (``UserWarning`` subclass),
    reachable through ``tl.errors``, and it must survive the default filters.
    """

    assert issubclass(ArtifactSchemaAgeWarning, UserWarning)
    assert not issubclass(ArtifactSchemaAgeWarning, DeprecationWarning)
    assert issubclass(ArtifactSchemaAgeWarning, tl.errors.TorchLensWarning)
    assert tl.errors.ArtifactSchemaAgeWarning is ArtifactSchemaAgeWarning

    trace = _build_trace()
    path = tmp_path / "between_floor.tlspec"
    tl.save(trace, path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert MIN_TLSPEC_VERSION < TLSPEC_VERSION, "no between-floor range to exercise"
    manifest["tlspec_version"] = MIN_TLSPEC_VERSION
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # resetwarnings() drops pytest's own filters so this asserts against the
    # DEFAULT interpreter filters -- the exact regime that hid the old category.
    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        loaded = tl.load(path)

    assert isinstance(loaded, tl.Trace)
    advisories = [w for w in caught if issubclass(w.category, ArtifactSchemaAgeWarning)]
    assert advisories, [str(w.category) for w in caught]
    assert "older than runtime tlspec_version" in str(advisories[0].message)


@pytest.mark.smoke
def test_inconsistent_producer_pair_refuses_typed(tmp_path: Path) -> None:
    """A (writer, stamp) pair no governed ledger window covers refuses typed.

    Historically this forged pair (an old writer version stamped onto a
    current-schema save) refused through the hand-typed ``< "2.33"``
    inequality as a below-floor error -- a misclassification that ALSO
    orphaned lawful released v2.31.0/v2.32.4 artifacts. Producer
    pair-consistency (ecosystem MEMO 3.1, gate G5) keys the same
    forgery-detection on the governed compatibility ledger: released 2.32.4
    only ever wrote tlspec 6, so claiming tlspec 9 under that writer is an
    inconsistent pair and refuses with the ledger's stable code.
    """

    trace = _build_trace()
    path = tmp_path / "forged_release.tlspec"
    tl.save(trace, path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["tlspec_version"] == TLSPEC_VERSION
    manifest["torchlens_version"] = "2.32.4"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TorchLensIOError, match="torchlens_version=2.32.4") as excinfo:
        tl.load(path)
    assert excinfo.value.fields["code"] == "artifact_producer_pair_ungoverned"


@pytest.mark.smoke
@pytest.mark.parametrize(
    "fixture",
    [
        "golden/io_v3_sample.tlspec",
        "fixtures/tlspec_v2_16/F2_modellog_tiny_cnn.tlspec",
        "fixtures/tlspec_v2_16/F3_modellog_tiny_transformer.tlspec",
    ],
)
def test_real_pre_floor_artifacts_refuse_typed(fixture: str) -> None:
    """Checked-in real pre-2.33 artifacts refuse with the floor named."""

    fixture_path = Path(__file__).parent / fixture
    with pytest.raises(ArtifactVersionBelowFloorError, match=FLOOR_MATCH):
        tl.load(fixture_path, trust_custom_callables=True)


@pytest.mark.smoke
def test_legacy_intervention_specs_stay_loadable() -> None:
    """The floor covers Trace rehydration only; 2.16 intervention specs load."""

    from torchlens.intervention.types import InterventionSpec

    fixture_path = Path(__file__).parent / "fixtures" / "tlspec_v2_16"
    loaded = tl.load(fixture_path / "F1_intervention_default.tlspec", trust_custom_callables=True)
    assert isinstance(loaded, InterventionSpec)
