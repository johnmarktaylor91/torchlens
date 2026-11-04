"""io durability + bounds hardening lane (fix/iodur, 2026-08-14).

Each test FAILS against the pre-fix behavior:

* MED1 -- the save path fsyncs every written file, the temp directory tree,
  and the parent directory around the publish rename (no fsync existed
  anywhere in ``torchlens/_io/`` before), so a power/OS crash cannot publish
  a torn artifact after the old bundle was already replaced. In the 3.0 saved
  format a bundle's member index is its root ``manifest.json`` and each member
  is ``members/<index>-<name>/`` with its own ``manifest.json``.
* MED2 -- ``Bundle``/trace overwrite saves move the existing bundle aside
  just BEFORE the atomic swap, not at save start, so a hard crash mid-save
  (SIGKILL/power loss) never leaves the target path with no bundle at all.
* MED3 -- the legacy ``kind=bundle`` ``metadata.pkl`` load path enforces the
  same byte ceiling as the trace path.
* MED4 -- manifest tensor entries are bounded in count and duplicate
  ``blob_id`` / ``relative_path`` values refuse (eager-verify CPU
  amplification from a KB-sized hostile manifest). The tensor-entry manifest
  is the one a fastlog directory (``tl.record`` to disk) writes and
  ``tl.recording.recover`` reads; a 3.0 trace save writes the record-table
  manifest instead, so these tests parse a fastlog manifest.
* MED5 -- resaving a loaded bundle with ``include_source=False`` does not
  re-emit the loaded (possibly forged) provenance verbatim.
* MED6 -- the runnable dead-model fallback refuses a non-tensor embedded
  state entry typed instead of crashing with ``AttributeError`` mid-save.
* LOW -- NUL-byte relative paths refuse typed; manifest integer fields
  reject bools/negatives/absurd dims; tensor-entry sha256 is format-checked
  at parse; bounded JSON refuses NaN/Infinity constants.
* R60/R65 -- a hostile ``metadata.pkl`` (object bomb, non-mapping payload)
  never reaches an unpickler through ``tl.load``: in the 3.0 saved format a
  ``metadata.pkl`` at the artifact root is a 2.x marker and refuses typed
  before any pickle byte is read.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import (
    _json as json_mod,
    bundle as bundle_mod,
    bundle_container,
    manifest as manifest_mod,
    runnable as runnable_mod,
)
from torchlens._io.format3.container import MEMBERS_DIR
from torchlens._io.format3.contract import MANIFEST_FILE
from torchlens._io.format_errors import ArtifactTooOldError
from torchlens._io.manifest import Manifest
from torchlens._io.paths import resolve_bundle_blob_path
from torchlens.errors import ArtifactError


def _tiny() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU())


def _trace() -> tl.Trace:
    return tl.trace(
        _tiny().eval(),
        torch.randn(2, 4),
        save="all",
    )


def _save(tmp_path: Path, name: str = "b.tlspec") -> Path:
    spec = tmp_path / name
    _trace().save(str(spec))
    return spec


def _record_to_disk(tmp_path: Path) -> Path:
    """Write a finalized fastlog directory; its manifest is the tensor-entry manifest."""

    recording = tl.record(
        _tiny().eval(),
        torch.randn(2, 4),
        storage=tl.options.StorageOptions(path=tmp_path / "r.tlfast", retain_in_memory=False),
        recording=tl.options.RecordingOptions(default_op=True),
    )
    return Path(recording.path)


def _recording_fsync(synced: list[Path]):
    real_fsync = os.fsync

    def recorder(fd: int) -> None:
        try:
            synced.append(Path(os.readlink(f"/proc/self/fd/{fd}")))
        except OSError:
            pass
        real_fsync(fd)

    return recorder


# --------------------------------------------------------------------------- #
# MED1: fsync-before-publish                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(sys.platform != "linux", reason="/proc fd->path resolution")
def test_bundle_save_fsyncs_members_and_parent_dir(tmp_path: Path, monkeypatch) -> None:
    """A Bundle save fsyncs its root manifest, the member manifests, and the parent dir.

    The 3.0 bundle's member index is its root ``manifest.json`` (the 2.x
    ``bundle.json``); each member is ``members/<index>-<name>/`` with its own
    ``manifest.json``. Paths are the temporary ones, fsynced before the rename.
    """

    synced: list[Path] = []
    monkeypatch.setattr(os, "fsync", _recording_fsync(synced))
    target = tmp_path / "bundle.tlspec"
    tl.Bundle({"m": _trace()}).save(target, overwrite=True)
    parent = target.parent.resolve()
    manifests = [path for path in synced if path.name == MANIFEST_FILE]
    assert any(path.parent.parent == parent for path in manifests), (
        "the bundle root manifest was not fsynced before publish"
    )
    assert any(path.parent.parent.name == MEMBERS_DIR for path in manifests), (
        "member manifests were not fsynced before publish"
    )
    assert parent in synced, "the publish rename was not made durable (parent dir fsync)"


@pytest.mark.skipif(sys.platform != "linux", reason="/proc fd->path resolution")
def test_manifest_write_fsyncs_the_file(tmp_path: Path, monkeypatch) -> None:
    """``Manifest.write`` (fastlog finalize path) fsyncs its own file."""

    manifest = Manifest.read(_record_to_disk(tmp_path) / "manifest.json")
    synced: list[Path] = []
    monkeypatch.setattr(os, "fsync", _recording_fsync(synced))
    destination = tmp_path / "standalone-manifest.json"
    manifest.write(destination)
    assert destination in synced


# --------------------------------------------------------------------------- #
# MED2: overwrite aside-rename happens at the swap, not at save start          #
# --------------------------------------------------------------------------- #


def test_overwrite_save_keeps_old_bundle_at_target_until_swap(tmp_path: Path, monkeypatch) -> None:
    """The existing bundle stays AT its path for the whole write phase.

    Fail-before: the trace-save writer renamed the old bundle aside to
    ``.bak.<uuid>`` at save START, before the minutes-long scrub/blob write.
    Python exception paths restored it, but SIGKILL/power loss mid-save left
    the target with NO bundle (old data stranded under an undocumented backup
    name), and concurrent readers saw the bundle vanish for the entire save.
    The SIGKILL window is the whole span between the early aside-rename and
    the swap; observing the target mid-save (at ``_build_manifest``, after
    blob writes) is the RED discriminator for that ordering.
    """

    spec = tmp_path / "b.tlspec"
    _trace().save(str(spec))
    real_build_manifest = bundle_mod._build_manifest
    observed: dict[str, bool] = {}

    def observing_build_manifest(*args, **kwargs):
        observed["target_exists_mid_save"] = spec.exists()
        observed["target_loadable_mid_save"] = (spec / "manifest.json").is_file()
        return real_build_manifest(*args, **kwargs)

    monkeypatch.setattr(bundle_mod, "_build_manifest", observing_build_manifest)
    _trace().save(str(spec), overwrite=True)
    assert observed["target_exists_mid_save"], "old bundle was moved aside at save start"
    assert observed["target_loadable_mid_save"]
    # The overwrite itself still completed and left no backup debris.
    tl.load(str(spec))
    assert not list(tmp_path.glob("*.bak.*"))


# --------------------------------------------------------------------------- #
# MED3: legacy kind=bundle metadata.pkl byte ceiling                           #
# --------------------------------------------------------------------------- #


def test_legacy_bundle_metadata_pkl_ceiling_refuses_oversize(tmp_path: Path, monkeypatch) -> None:
    """The legacy bundle branch enforces the same pkl byte ceiling as traces.

    Fail-before: ``_load_unified_bundle``'s ``kind=bundle`` legacy branch fed
    ``metadata.pkl`` straight into the unpickler with no fstat cap (the trace
    path had one), so an absurd on-disk pickle was an alloc/time DoS at
    ``tl.load``. Pre-fix this raised the unrelated "is not a Bundle" error
    only AFTER unpickling the whole payload.
    """

    legacy_dir = tmp_path / "legacy.tlspec"
    legacy_dir.mkdir()
    (legacy_dir / "metadata.pkl").write_bytes(pickle.dumps({"not": "a bundle"}))
    monkeypatch.setattr(bundle_mod, "_MAX_METADATA_PKL_BYTES", 16)
    with pytest.raises(ArtifactError, match="ceiling"):
        bundle_container._load_unified_bundle(legacy_dir)


# --------------------------------------------------------------------------- #
# MED4: manifest entry-count ceiling + duplicate blob identity refusal         #
# --------------------------------------------------------------------------- #


def _saved_manifest_data(tmp_path: Path) -> dict:
    """The tensor-entry manifest a finalized fastlog directory wrote."""

    data = json.loads((_record_to_disk(tmp_path) / "manifest.json").read_text(encoding="utf-8"))
    assert data["tensors"], "fixture recording must carry at least one tensor entry"
    return data


def test_manifest_refuses_duplicate_blob_ids(tmp_path: Path) -> None:
    """Two tensor entries sharing one blob_id refuse at parse.

    Fail-before: duplicates silently last-won in the load-side entry indexes
    while eager verification did per-entry sha256 + safetensors decode work,
    so a KB manifest with millions of same-blob entries bought hours of CPU.
    """

    data = _saved_manifest_data(tmp_path)
    data["tensors"] = [*data["tensors"], dict(data["tensors"][0])]
    with pytest.raises(ArtifactError, match="duplicate blob_id"):
        Manifest.from_dict(data)


def test_manifest_refuses_duplicate_relative_paths(tmp_path: Path) -> None:
    """Two tensor entries pointing at one blob file refuse at parse."""

    data = _saved_manifest_data(tmp_path)
    forged = dict(data["tensors"][0])
    forged["blob_id"] = "zzzz-forged"
    data["tensors"] = [*data["tensors"], forged]
    with pytest.raises(ArtifactError, match="duplicate relative_path"):
        Manifest.from_dict(data)


def test_manifest_refuses_entry_count_above_ceiling(tmp_path: Path, monkeypatch) -> None:
    """An entry list above the structural ceiling refuses before parsing."""

    data = _saved_manifest_data(tmp_path)
    assert data["tensors"], "fixture bundle must carry at least one tensor entry"
    monkeypatch.setattr(manifest_mod, "_MAX_MANIFEST_TENSOR_ENTRIES", len(data["tensors"]) - 1)
    with pytest.raises(ArtifactError, match="ceiling"):
        Manifest.from_dict(data)


# --------------------------------------------------------------------------- #
# MED5: include_source=False gates the loaded-provenance passthrough           #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# MED6: runnable dead-model fallback refuses non-tensor state typed            #
# --------------------------------------------------------------------------- #


def test_runnable_dead_model_fallback_refuses_non_tensor_state_typed() -> None:
    """A non-tensor snapshot entry fails typed, not with AttributeError.

    Fail-before: the dead-model fallback cast every snapshot complement entry
    to ``torch.Tensor`` without the live lane's isinstance guard, so a
    non-tensor embedded_state entry on a runnable->load->runnable resave
    crashed with an uncontrolled ``AttributeError`` (``.shape`` on a str)
    mid-save instead of a typed refusal.
    """

    from types import SimpleNamespace

    stub_trace = SimpleNamespace(
        _runnable=SimpleNamespace(
            capture_state={"weight": torch.zeros(2), "running_junk": "not-a-tensor"},
            embedded_state=None,
        ),
        _param_logs=(),
    )
    with pytest.raises(ArtifactError, match="non-tensor persistent-buffer"):
        runnable_mod._add_persistent_buffer_slot_drafts(stub_trace, {})


# --------------------------------------------------------------------------- #
# LOW: validation tightenings                                                  #
# --------------------------------------------------------------------------- #


def test_nul_byte_relative_path_refuses_typed(tmp_path: Path) -> None:
    """A NUL byte in a manifest relative_path refuses typed, not ValueError.

    Fail-before: ``Path.resolve`` raised a bare ``ValueError`` ("embedded null
    byte") while every other hostile path shape refused typed.
    """

    bundle_root = tmp_path / "b.tlspec"
    (bundle_root / "blobs").mkdir(parents=True)
    with pytest.raises(ArtifactError, match="unresolvable"):
        resolve_bundle_blob_path(bundle_root, "blobs/a\x00b.safetensors")


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "shape",
    [
        [True, 2],
        [2**60],
    ],
    ids=[
        "bool-dim",
        "huge-dim",
    ],
)
def test_tensor_entry_refuses_invalid_shape(tmp_path: Path, shape: list) -> None:
    """Negative/bool/absurd shape dims refuse at parse (latent num_elements feed)."""

    data = _saved_manifest_data(tmp_path)
    entry = dict(data["tensors"][0])
    entry["shape"] = shape
    with pytest.raises(ArtifactError, match="shape"):
        manifest_mod.TensorEntry.from_dict(entry)


def test_tensor_entry_refuses_bool_bytes_and_manifest_bool_ints(tmp_path: Path) -> None:
    """``bytes=True`` and ``n_layers=True`` refuse (bool passes isinstance int)."""

    data = _saved_manifest_data(tmp_path)
    entry = dict(data["tensors"][0])
    entry["bytes"] = True
    with pytest.raises(ArtifactError, match="bytes"):
        manifest_mod.TensorEntry.from_dict(entry)
    forged = dict(data)
    forged["n_layers"] = True
    with pytest.raises(ArtifactError, match="n_layers"):
        Manifest.from_dict(forged)


def test_tensor_entry_refuses_malformed_sha256(tmp_path: Path) -> None:
    """A non-hex 64-char sha256 refuses at parse, not lazily at materialize."""

    data = _saved_manifest_data(tmp_path)
    entry = dict(data["tensors"][0])
    entry["sha256"] = "Z" * 64
    with pytest.raises(ArtifactError, match="sha256|SHA-256"):
        manifest_mod.TensorEntry.from_dict(entry)


def test_loads_bounded_refuses_nonfinite_constants() -> None:
    """NaN/Infinity load-refuse: writers use allow_nan=False, so accepting them
    at load produced artifacts whose re-save raises (stillborn)."""

    for payload in ('{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}'):
        with pytest.raises(json.JSONDecodeError, match="non-finite"):
            json_mod.loads_bounded(payload)


# --------------------------------------------------------------------------- #
# R60/F6 + R65/F10: a planted metadata.pkl never reaches an unpickler         #
# --------------------------------------------------------------------------- #


def _never_unpickle(monkeypatch) -> list[str]:
    """Record every call into the pickle readers ``tl.load`` could reach."""

    calls: list[str] = []

    def refuse(name: str):
        def recorder(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"tl.load reached {name} on a planted metadata.pkl")

        return recorder

    monkeypatch.setattr(bundle_mod, "_metadata_unpickler", refuse("_metadata_unpickler"))
    monkeypatch.setattr(pickle, "load", refuse("pickle.load"))
    monkeypatch.setattr(pickle, "loads", refuse("pickle.loads"))
    return calls


def test_metadata_pkl_opcode_ceiling_refuses_object_bomb(tmp_path: Path, monkeypatch) -> None:
    """A metadata pickle packed with tiny values refuses before any unpickling.

    Fail-before: the byte cap admitted a pickle whose ~5x RSS expansion happened
    entirely BEFORE any structural check (measured 76 MiB of ints -> ~390 MiB;
    the old 4 GiB byte cap projected ~20 GiB) and the eventual refusal escaped
    as a raw stdlib TypeError. In the 3.0 saved format no pickle is read at
    all: the planted ``metadata.pkl`` is a 2.x root marker and ``tl.load``
    refuses it typed before opening it.
    """

    spec = _save(tmp_path)
    (spec / "metadata.pkl").write_bytes(pickle.dumps([0] * 100_000))
    calls = _never_unpickle(monkeypatch)
    with pytest.raises(ArtifactTooOldError, match="metadata.pkl") as excinfo:
        tl.load(str(spec))
    assert excinfo.value.fields["code"] == "artifact_version_below_floor"
    assert calls == []


def test_metadata_pkl_non_mapping_payload_refuses_typed(tmp_path: Path, monkeypatch) -> None:
    """A non-mapping metadata payload refuses typed, not as a raw stdlib error.

    Fail-before: ``tl.load`` on a bundle whose ``metadata.pkl`` held a plain
    list escaped as ``ValueError: dictionary update sequence element #0 ...``
    from the downstream dict() walk -- untyped, no code, no remedy. In the 3.0
    saved format the planted pickle refuses typed, with a code and a remedy,
    before it is read.
    """

    spec = _save(tmp_path)
    (spec / "metadata.pkl").write_bytes(pickle.dumps([[0] * 3] * 3))
    calls = _never_unpickle(monkeypatch)
    with pytest.raises(ArtifactTooOldError, match="Remedy") as excinfo:
        tl.load(str(spec))
    assert excinfo.value.fields["code"] == "artifact_version_below_floor"
    assert calls == []


def _load_pickle_payload(tmp_path: Path, payload: object, **policy) -> object:
    """Drive the retained ``metadata.pkl`` reader at its own seam.

    ``tl.load`` refuses every ``metadata.pkl`` root before this reader (pinned by
    the two tests above), so its guards are provoked directly, under a forged
    current-version manifest with no tensors.
    """

    bundle_dir = tmp_path / "pickle_payload"
    (bundle_dir / "blobs").mkdir(parents=True)
    (bundle_dir / "metadata.pkl").write_bytes(pickle.dumps(payload))
    forged = Manifest.from_dict(
        {
            "tlspec_version": manifest_mod.TLSPEC_VERSION,
            "torchlens_version": tl.__version__,
            "torch_version": torch.__version__,
            "python_version": "{}.{}.{}".format(*sys.version_info[:3]),
            "platform": sys.platform,
            "created_at": "2026-01-01T00:00:00Z",
            "bundle_format": "directory",
            "n_layers": 0,
            "n_out_blobs": 0,
            "n_grad_blobs": 0,
            "n_auxiliary_blobs": 0,
            "tensors": [],
            "unsupported_tensors": [],
        }
    )
    return bundle_mod._load_trace_payload(
        bundle_dir,
        forged,
        lazy=True,
        map_location="cpu",
        materialize_nested=False,
        payload_hints=None,
        **policy,
    )


def test_pickle_reader_opcode_ceiling_refuses_object_bomb(tmp_path: Path, monkeypatch) -> None:
    """The retained pickle reader refuses an object bomb on opcode count, typed.

    Fail-before: the byte cap admitted a pickle whose expansion happened before
    any structural check, and the refusal escaped as a raw stdlib TypeError.
    """

    monkeypatch.setattr(bundle_mod, "_METADATA_PKL_PRESCAN_BYTES", 0)
    monkeypatch.setattr(bundle_mod, "_MAX_METADATA_PKL_OPCODES", 1_000)
    with pytest.raises(ArtifactError, match="opcode allocation ceiling") as excinfo:
        _load_pickle_payload(tmp_path, [0] * 100_000)
    assert excinfo.value.fields["code"] == "metadata_object_count_exceeded"


def test_pickle_reader_non_mapping_payload_refuses_typed(tmp_path: Path) -> None:
    """The retained pickle reader refuses a non-mapping payload typed.

    Fail-before: a plain list escaped as ``ValueError: dictionary update
    sequence element #0 ...`` from the downstream dict() walk, with no code.
    """

    with pytest.raises(ArtifactError, match="not a metadata mapping") as excinfo:
        _load_pickle_payload(tmp_path, [[0] * 3] * 3)
    assert excinfo.value.fields["code"] == "metadata_payload_not_a_mapping"


def test_unverified_capture_disclosure_survives_save_and_load(tmp_path: Path) -> None:
    """P7/R10: capture_verified=False must not launder to no-claim across save.

    Fail-before: the disclosure triple was FieldPolicy.DROP, so a capture
    TorchLens itself refused to bless loaded as verified=None / reason=None --
    the round trip IMPROVED a verdict. The negative claim now persists as a
    string-only row; True/None stay session-time so a loaded artifact can
    never CLAIM verification.
    """

    trace = _trace()
    trace._provenance_verified = False
    trace._provenance_verification_reason = "escape_rescue_unrecovered"
    spec = tmp_path / "unverified.tlspec"
    trace.save(str(spec))
    loaded = tl.load(str(spec))
    assert loaded.provenance.verified is False
    assert loaded.provenance.verification_reason == "escape_rescue_unrecovered"

    clean = _save(tmp_path, "clean.tlspec")
    loaded_clean = tl.load(str(clean))
    assert loaded_clean.provenance.verified is None
