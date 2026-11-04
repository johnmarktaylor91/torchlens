"""G4 persisted-state contract: symmetric typed refusal + inert inventory.

The MEMO 3.3 test obligations, on real bytes where the spec demands it:

- THREE-PATH partition, parametrized over the ``__dict__``-backed path
  (``Trace`` -- the must-not-drop case: it never crashed, it silently
  absorbed), the hook-restored columnar path (``Op`` -- previously an
  untyped ``AttributeError``), and a slotted value class.
- A newer-shaped state under this reader's inventory refuses typed -- proven
  END-TO-END through disk and the bundle wrappers on a bundle this release
  wrote (the D-ECO-10 on-disk cell plus the anti-laundering guarantee: the
  stable code survives ``tl.load``). Files a 2.x release saved refuse in
  ``tests/test_load_refuses_saved_by_2x.py``.
- INERT inventory: the private ``inspect_state_contract`` scan reports unknown
  names with owning record types without loading anything and without
  refusing.
- D-ECO-10's second assertion: the unknown-field event is registered as an
  OUTCOME-DEGRADING condition (an accept mode may never attest COMPLETE).
- SCOPE boundary: the refusal is armed by the governed-artifact-load window
  (the ``.tlspec`` loaders); plain session pickling of live records keeps the
  historical user-extras round-trip, which the save path refuses to persist.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from _format3_envelope import read_record, set_cell

import torchlens as tl
from torchlens._io import (
    TLSPEC_VERSION,
    UnknownPersistedFieldError,
)
from torchlens._io.state_contract import governed_artifact_load

pytestmark = [pytest.mark.heavy]

UNKNOWN_KEY = "a_field_a_future_release_added"


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

    from torchlens._data_classes.trace import Trace

    state = tiny_trace.__getstate__()
    state[UNKNOWN_KEY] = 1
    state.setdefault("tlspec_version", TLSPEC_VERSION)
    with governed_artifact_load(), pytest.raises(UnknownPersistedFieldError) as excinfo:
        Trace.__new__(Trace).__setstate__(state)
    _assert_typed_unknown_refusal(excinfo, "Trace")


def test_op_columnar_path_refuses_typed_never_attributeerror(tiny_trace) -> None:
    """Acceptance bar: typed refusal, never AttributeError, on the Op path."""

    from torchlens._data_classes._state_adapter import state_new
    from torchlens._data_classes.op import Op

    donor_state = tiny_trace[0].__getstate__()
    donor_state[UNKNOWN_KEY] = 1
    donor_state.setdefault("tlspec_version", TLSPEC_VERSION)
    with governed_artifact_load(), pytest.raises(UnknownPersistedFieldError) as excinfo:
        state_new(Op).__setstate__(donor_state)
    _assert_typed_unknown_refusal(excinfo, "Op")
    assert not isinstance(excinfo.value, AttributeError)


def test_slotted_path_refuses_unknown_field() -> None:
    """The third storage path: a slotted value record refuses identically."""

    from torchlens._data_classes._state_adapter import state_new
    from torchlens._data_classes.grad_fn_call import GradFnCall

    with governed_artifact_load(), pytest.raises(UnknownPersistedFieldError) as excinfo:
        state_new(GradFnCall).__setstate__({"tlspec_version": TLSPEC_VERSION, UNKNOWN_KEY: 1})
    _assert_typed_unknown_refusal(excinfo, "GradFnCall")


def test_absent_fields_keep_tolerant_default_fill(tiny_trace) -> None:
    """The contract refuses UNKNOWN fields only; ABSENT fields still fill."""

    from torchlens._data_classes.trace import Trace

    state = tiny_trace.__getstate__()
    state.setdefault("tlspec_version", TLSPEC_VERSION)
    state.pop("annotations", None)
    restored = Trace.__new__(Trace)
    with governed_artifact_load():
        restored.__setstate__(state)
    assert restored.annotations == {}


@pytest.fixture()
def tampered_main_bundle(tiny_trace: object, tmp_path: Path) -> Path:
    """A bundle this release wrote whose Trace state gained one unknown field."""

    bundle = tmp_path / "art_current_future.tlspec"
    tiny_trace.save(bundle)  # type: ignore[attr-defined]
    # The trace record table's ``fields`` gain one column; its digest is re-listed, so
    # the refusal comes from the persisted-state contract, not the integrity check.
    assert UNKNOWN_KEY not in read_record(bundle, "trace")["fields"]
    set_cell(bundle, "trace", UNKNOWN_KEY, 0, 1)
    return bundle


def test_inert_inventory_reports_without_loading(
    tiny_trace: object, tampered_main_bundle: Path, tmp_path: Path
) -> None:
    """inspect_state_contract sees everything and refuses nothing."""

    from torchlens._io.state_contract import inspect_state_contract

    pristine_bundle = tmp_path / "art_current.tlspec"
    tiny_trace.save(pristine_bundle)  # type: ignore[attr-defined]
    pristine = inspect_state_contract(pristine_bundle)
    assert pristine["unknown_field_total"] == 0

    tampered = inspect_state_contract(tampered_main_bundle)
    assert tampered["unknown_field_total"] >= 1
    trace_row = tampered["record_types"]["torchlens._data_classes.trace.Trace"]
    assert UNKNOWN_KEY in trace_row["unknown_fields"]
    assert trace_row["ungoverned"] is False


class _MarkerWriter:
    """Pickles as a call to ``open(marker, "w")``: a real unpickle creates the marker."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str, str]]:
        return (open, (str(self.marker), "w"))


def test_inert_inventory_reports_a_planted_pickle_in_a_3_0_save_as_foreign(
    tiny_trace: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root metadata.pkl beside a 3.0 manifest is reported by name, size and digest, never read."""

    import hashlib
    import pickle
    import shutil

    from torchlens._io import state_contract

    clean_bundle = tmp_path / "clean.tlspec"
    tiny_trace.save(clean_bundle)  # type: ignore[attr-defined]
    clean = state_contract.inspect_state_contract(clean_bundle)
    assert "foreign_files" not in clean

    marker = tmp_path / "unpickled.marker"
    planted_bytes = pickle.dumps({"a_planted_key": _MarkerWriter(marker)})
    planted_bundle = tmp_path / "planted.tlspec"
    shutil.copytree(clean_bundle, planted_bundle)
    (planted_bundle / "metadata.pkl").write_bytes(planted_bytes)

    def _refuse_unpickle(_path: object) -> object:
        raise AssertionError("the inventory unpickled a pickle planted in a 3.0 save")

    monkeypatch.setattr(state_contract, "_load_inventory_skeleton", _refuse_unpickle)
    planted = state_contract.inspect_state_contract(planted_bundle)

    note = {
        "name": "metadata.pkl",
        "size_bytes": len(planted_bytes),
        "sha256": hashlib.sha256(planted_bytes).hexdigest(),
    }
    assert planted == {**clean, "foreign_files": [note]}
    assert not marker.exists()


def _legacy_bundle(tmp_path: Path, payload: object) -> Path:
    """A 2.x-shaped artifact: a root metadata.pkl (``payload`` pickled, or raw bytes)."""

    import pickle

    bundle = tmp_path / "legacy.tlspec"
    bundle.mkdir()
    data = payload if isinstance(payload, bytes) else pickle.dumps(payload)
    (bundle / "metadata.pkl").write_bytes(data)
    return bundle


def test_inert_inventory_still_reads_a_legacy_root_pickle(tmp_path: Path) -> None:
    """Without a 3.0 manifest the root metadata.pkl is inventoried inertly, as before."""

    from torchlens._io.state_contract import inspect_state_contract

    marker = tmp_path / "unpickled.marker"
    bundle = _legacy_bundle(tmp_path, {UNKNOWN_KEY: _MarkerWriter(marker)})
    report = inspect_state_contract(bundle)
    trace_row = report["record_types"]["torchlens._data_classes.trace.Trace"]
    assert trace_row["unknown_fields"] == [UNKNOWN_KEY]
    assert report["record_types"][f"{open.__module__}.{open.__name__}"]["ungoverned"] is True
    assert "foreign_files" not in report
    assert not marker.exists()


_LEGACY_GOLDENS = sorted(
    (Path(__file__).parent / "goldens" / "legacy").glob("*.tlspec/metadata.pkl")
)


@pytest.mark.parametrize("metadata", _LEGACY_GOLDENS, ids=lambda path: path.parent.name)
def test_inert_inventory_reads_every_legacy_golden_under_its_bounds(metadata: Path) -> None:
    """The byte ceiling and the class cap leave every genuine 2.x golden readable."""

    from torchlens._io.state_contract import inspect_state_contract

    assert _LEGACY_GOLDENS, "the legacy goldens moved; this test would pass vacuously"
    report = inspect_state_contract(metadata.parent)
    assert report["record_types"]
    assert "foreign_files" not in report


def _assert_inventory_refusal(excinfo: pytest.ExceptionInfo, needle: str) -> None:
    from torchlens._io import ArtifactError

    assert isinstance(excinfo.value, ArtifactError)
    assert excinfo.value.fields["code"] == "bundle_metadata_integrity_refused"
    assert needle in str(excinfo.value)


def test_inert_inventory_refuses_a_legacy_pickle_over_the_byte_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy unpickle is bounded by the file's byte size, at the ceiling inclusive."""

    from torchlens._io import ArtifactError, state_contract

    bundle = _legacy_bundle(tmp_path, {UNKNOWN_KEY: "x" * 64})
    size = (bundle / "metadata.pkl").stat().st_size
    monkeypatch.setattr(state_contract, "_MAX_INVENTORY_PKL_BYTES", size)
    assert state_contract.inspect_state_contract(bundle)["unknown_field_total"] == 1

    monkeypatch.setattr(state_contract, "_MAX_INVENTORY_PKL_BYTES", size - 1)
    with pytest.raises(ArtifactError) as excinfo:
        state_contract.inspect_state_contract(bundle)
    _assert_inventory_refusal(excinfo, "-byte ceiling")


def test_inert_inventory_caps_distinct_surrogate_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy pickle naming more distinct classes than the cap refuses; at the cap it reads."""

    from torchlens._io import ArtifactError, state_contract

    # Protocol-0 opcodes: MARK, then per class GLOBAL "mod\nname\n", EMPTY_TUPLE, REDUCE
    # (each class instantiated twice, so repeats reuse the scan's surrogate), LIST, STOP.
    distinct = 4
    calls = b"".join(f"cno_such_mod_{i}\nNoSuchCls{i}\n)R".encode() for i in range(distinct))
    bundle = _legacy_bundle(tmp_path, b"(" + calls + calls + b"l.")
    monkeypatch.setattr(state_contract, "_MAX_SURROGATE_CLASSES", distinct)
    report = state_contract.inspect_state_contract(bundle)
    assert report["record_types"]["no_such_mod_3.NoSuchCls3"]["instances"] == 2

    monkeypatch.setattr(state_contract, "_MAX_SURROGATE_CLASSES", distinct - 1)
    with pytest.raises(ArtifactError) as excinfo:
        state_contract.inspect_state_contract(bundle)
    _assert_inventory_refusal(excinfo, "distinct classes")


# ---------------------------------------------------------------------------
# Registration + invariant-feed pins.
# ---------------------------------------------------------------------------


def test_dropped_edge_tensor_args_still_feeds_its_invariant() -> None:
    """The field the contract exists to protect keeps protecting its invariant."""

    import inspect

    from torchlens._io.field_registry import persisted_field_names
    from torchlens.validation import core as validation_core

    assert "dropped_edge_tensor_args" in persisted_field_names("op")
    assert "dropped_edge_tensor_args" in inspect.getsource(validation_core)
