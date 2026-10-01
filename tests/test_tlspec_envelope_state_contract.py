"""G4 persisted-state contract: symmetric typed refusal + inert inventory.

The MEMO 3.3 test obligations, on real bytes where the spec demands it:

- THREE-PATH partition, parametrized over the ``__dict__``-backed path
  (``Trace`` -- the must-not-drop case: it never crashed, it silently
  absorbed), the hook-restored columnar path (``Op`` -- previously an
  untyped ``AttributeError``), and a slotted value class.
- TWO-DIRECTION load on the harvested golden corpus: every governed old
  writer's artifact loads (or refuses on its governed version window), and a
  newer-shaped state under this reader's inventory refuses typed -- proven
  END-TO-END through disk and the bundle wrappers (the D-ECO-10 on-disk cell
  plus the anti-laundering guarantee: the stable code survives ``tl.load``).
- INERT inventory: ``torchlens.io.inspect_state_contract`` reports unknown
  names with owning record types without loading anything and without
  refusing.
- D-ECO-10's second assertion: the unknown-field event is registered as an
  OUTCOME-DEGRADING condition (an accept mode may never attest COMPLETE).
- SCOPE boundary: the refusal is armed by the governed-artifact-load window
  (the ``.tlspec`` loaders); plain session pickling of live records keeps the
  historical user-extras round-trip, which the save path refuses to persist.
"""

from __future__ import annotations

import json
import pickle
import shutil
import tarfile
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._io import (
    MIN_TLSPEC_VERSION,
    TLSPEC_VERSION,
    TorchLensIOError,
    UnknownPersistedFieldError,
)
from torchlens._io.state_contract import governed_artifact_load

pytestmark = [pytest.mark.heavy]

CORPUS = Path(__file__).parent / "release_goldens" / "genuine_release_artifacts.tar.gz"
UNKNOWN_KEY = "a_field_a_future_release_added"


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("release_corpus")
    with tarfile.open(CORPUS, "r:gz") as tar:
        tar.extractall(root)
    return root


@pytest.fixture(scope="module")
def tiny_trace() -> Iterator[object]:
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    trace = tl.trace(model, torch.randn(1, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


def _assert_typed_unknown_refusal(excinfo: pytest.ExceptionInfo, record_type: str) -> None:
    err = excinfo.value
    assert isinstance(err, UnknownPersistedFieldError)
    assert err.fields["code"] == "unknown_persisted_field"
    assert err.fields["record_type"] == record_type
    assert UNKNOWN_KEY in err.fields["unknown_fields"]
    assert err.fields["remedy"]


def test_trace_dict_path_refuses_unknown_field(tiny_trace) -> None:
    """The must-not-drop case: the path that never crashed is the path checked."""

    from torchlens.data_classes.trace import Trace

    state = tiny_trace.__getstate__()
    state[UNKNOWN_KEY] = 1
    state.setdefault("tlspec_version", TLSPEC_VERSION)
    with governed_artifact_load(), pytest.raises(UnknownPersistedFieldError) as excinfo:
        Trace.__new__(Trace).__setstate__(state)
    _assert_typed_unknown_refusal(excinfo, "Trace")


def test_trace_refusal_covers_both_writer_directions(tiny_trace) -> None:
    """Symmetric refusal: same-stamp drift AND older-writer unknown field."""

    from torchlens.data_classes.trace import Trace

    for declared_version in (TLSPEC_VERSION, 6):
        state = tiny_trace.__getstate__()
        state[UNKNOWN_KEY] = 1
        state["tlspec_version"] = declared_version
        with governed_artifact_load(), pytest.raises(UnknownPersistedFieldError) as excinfo:
            Trace.__new__(Trace).__setstate__(state)
        assert excinfo.value.fields["declared_tlspec_version"] == declared_version


def test_op_columnar_path_refuses_typed_never_attributeerror(tiny_trace) -> None:
    """Acceptance bar: typed refusal, never AttributeError, on the Op path."""

    from torchlens.data_classes._state_adapter import state_new
    from torchlens.data_classes.op import Op

    donor_state = tiny_trace[0].__getstate__()
    donor_state[UNKNOWN_KEY] = 1
    donor_state.setdefault("tlspec_version", TLSPEC_VERSION)
    with governed_artifact_load(), pytest.raises(UnknownPersistedFieldError) as excinfo:
        state_new(Op).__setstate__(donor_state)
    _assert_typed_unknown_refusal(excinfo, "Op")
    assert not isinstance(excinfo.value, AttributeError)


def test_slotted_path_refuses_unknown_field() -> None:
    """The third storage path: a slotted value record refuses identically."""

    from torchlens.data_classes._state_adapter import state_new
    from torchlens.data_classes.grad_fn_call import GradFnCall

    with governed_artifact_load(), pytest.raises(UnknownPersistedFieldError) as excinfo:
        state_new(GradFnCall).__setstate__({"tlspec_version": TLSPEC_VERSION, UNKNOWN_KEY: 1})
    _assert_typed_unknown_refusal(excinfo, "GradFnCall")


def test_absent_fields_keep_tolerant_default_fill(tiny_trace) -> None:
    """The contract refuses UNKNOWN fields only; ABSENT fields still fill."""

    from torchlens.data_classes.trace import Trace

    state = tiny_trace.__getstate__()
    state.setdefault("tlspec_version", TLSPEC_VERSION)
    state.pop("annotations", None)
    restored = Trace.__new__(Trace)
    with governed_artifact_load():
        restored.__setstate__(state)
    assert restored.annotations == {}


def test_session_pickle_stays_outside_the_contract(tiny_trace) -> None:
    """The contract governs ARTIFACT bytes only: session pickling of live
    records keeps the historical open round-trip for user-set extras.

    The save path refuses to persist undeclared record fields, so no governed
    artifact ever carries extras legitimately; refusing them on plain
    ``pickle`` would break the record-extras semantics the substrate pins
    (``test_record_dict_shadow_never_streams``)."""

    param = next(iter(tiny_trace.params.values()))
    param.__dict__["user_note"] = "keep_me"
    try:
        restored = pickle.loads(pickle.dumps(param))
    finally:
        param.__dict__.pop("user_note", None)
    assert restored.user_note == "keep_me"


# ---------------------------------------------------------------------------
# Two-direction gate on the harvested corpus.
# ---------------------------------------------------------------------------


def test_old_writers_direction_on_harvested_bytes(corpus_dir: Path, tmp_path: Path) -> None:
    """Every governed writer's artifact settles per its governed window."""

    import warnings

    from torchlens._io import ArtifactVersionBelowFloorError

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Governed loadable writers: load green, honest outcomes. The
        # rehydration floor is ``tlspec_version >= 6`` (MIN_TLSPEC_VERSION),
        # and the FIRST tlspec-6 writer was released v2.31.0 (measured on
        # genuine wheels; the compat ledger's governed producer windows), so
        # the v2.31.0 / v2.32.4 artifacts LOAD losslessly -- adjudicated in
        # WAVE-0-5.1 (AUD-CODE 2.4): the code was right, this pin and three
        # doc sentences had inherited the false "first shipped in 2.33" claim.
        for name, expected_status in (
            ("art_v2.31.0_portable", "UNATTESTED"),
            ("art_v2.31.0_audit", "UNATTESTED"),
            ("art_v2.32.4_portable", "UNATTESTED"),
            ("art_v2.33.0_portable", "UNATTESTED"),
            ("art_v2.34.1_portable", "UNATTESTED"),
            ("art_main_portable", "COMPLETE"),
        ):
            trace = tl.load(corpus_dir / name)
            assert len(trace) == 151
            assert trace.outcome.status.name == expected_status
        # Below-floor stamps refuse typed with the floor named. The genuine
        # v2.16 bundle predates the integer stamp entirely, so it refuses
        # through the same typed family (erratum FORK F2 owns the remedy
        # text; this pins that it cannot crash untyped or load wrong).
        assert MIN_TLSPEC_VERSION == 6
        with pytest.raises(TorchLensIOError):
            tl.load(corpus_dir / "art_v2.16.0_portable")
        # An integer stamp below the floor refuses with the floor-specific type.
        below_floor = tmp_path / "art_below_floor.tlspec"
        shutil.copytree(corpus_dir / "art_v2.31.0_portable", below_floor)
        manifest_path = below_floor / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["tlspec_version"] = MIN_TLSPEC_VERSION - 1
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ArtifactVersionBelowFloorError):
            tl.load(below_floor)


@pytest.fixture()
def tampered_main_bundle(corpus_dir: Path, tmp_path: Path) -> Path:
    """A byte-real main artifact whose Trace state gained one unknown field."""

    bundle = tmp_path / "art_main_future.tlspec"
    shutil.copytree(corpus_dir / "art_main_portable", bundle)
    metadata_path = bundle / "metadata.pkl"
    with metadata_path.open("rb") as fh:
        scrubbed_state = pickle.load(fh)
    assert isinstance(scrubbed_state, dict)
    scrubbed_state[UNKNOWN_KEY] = 1
    with metadata_path.open("wb") as fh:
        pickle.dump(scrubbed_state, fh)
    return bundle


def test_new_writer_direction_end_to_end_on_disk(tampered_main_bundle: Path) -> None:
    """D-ECO-10's on-disk cell + anti-laundering: the typed refusal survives
    the whole bundle load path with its stable code intact."""

    with pytest.raises(UnknownPersistedFieldError) as excinfo:
        tl.load(tampered_main_bundle)
    assert excinfo.value.fields["code"] == "unknown_persisted_field"
    assert excinfo.value.fields["record_type"] == "Trace"
    assert UNKNOWN_KEY in excinfo.value.fields["unknown_fields"]


def test_inert_inventory_reports_without_loading(
    corpus_dir: Path, tampered_main_bundle: Path
) -> None:
    """inspect_state_contract sees everything and refuses nothing."""

    from torchlens.io import inspect_state_contract

    pristine = inspect_state_contract(corpus_dir / "art_main_portable")
    assert pristine["unknown_field_total"] == 0

    tampered = inspect_state_contract(tampered_main_bundle)
    assert tampered["unknown_field_total"] >= 1
    trace_row = tampered["record_types"]["torchlens.data_classes.trace.Trace"]
    assert UNKNOWN_KEY in trace_row["unknown_fields"]
    assert trace_row["ungoverned"] is False


# ---------------------------------------------------------------------------
# Registration + invariant-feed pins.
# ---------------------------------------------------------------------------


def test_unknown_field_is_registered_outcome_degrading() -> None:
    """D-ECO-10 assertion two: an accept mode may never attest COMPLETE."""

    from torchlens.capture.outcome import (
        OUTCOME_DEGRADING_LOAD_CONDITIONS,
        CaptureStatus,
    )

    degraded = OUTCOME_DEGRADING_LOAD_CONDITIONS["unknown_persisted_field"]
    assert degraded is not CaptureStatus.COMPLETE


def test_dropped_edge_tensor_args_still_feeds_its_invariant() -> None:
    """The field the contract exists to protect keeps protecting its invariant."""

    import inspect

    from torchlens._io.field_registry import persisted_field_names
    from torchlens.validation import core as validation_core

    assert "dropped_edge_tensor_args" in persisted_field_names("op")
    assert "dropped_edge_tensor_args" in inspect.getsource(validation_core)


def test_public_errors_registry_serves_unknown_field_error() -> None:
    import torchlens.errors as errors

    assert errors.UnknownPersistedFieldError is UnknownPersistedFieldError
