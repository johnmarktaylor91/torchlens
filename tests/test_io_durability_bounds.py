"""io durability + bounds hardening lane (fix/iodur, 2026-08-14).

Each test FAILS against the pre-fix behavior:

* MED1 -- the save path fsyncs every written file, the temp directory tree,
  and the parent directory around the publish rename (no fsync existed
  anywhere in ``torchlens/_io/`` before), so a power/OS crash cannot publish
  a torn artifact after the old bundle was already replaced.
* MED2 -- ``Bundle``/trace overwrite saves move the existing bundle aside
  just BEFORE the atomic swap, not at save start, so a hard crash mid-save
  (SIGKILL/power loss) never leaves the target path with no bundle at all.
* MED3 -- the legacy ``kind=bundle`` ``metadata.pkl`` load path enforces the
  same byte ceiling as the trace path.
* MED4 -- manifest tensor entries are bounded in count and duplicate
  ``blob_id`` / ``relative_path`` values refuse (eager-verify CPU
  amplification from a KB-sized hostile manifest).
* MED5 -- resaving a loaded bundle with ``include_source=False`` does not
  re-emit the loaded (possibly forged) provenance verbatim.
* MED6 -- the runnable dead-model fallback refuses a non-tensor embedded
  state entry typed instead of crashing with ``AttributeError`` mid-save.
* LOW -- NUL-byte relative paths refuse typed; manifest integer fields
  reject bools/negatives/absurd dims; tensor-entry sha256 is format-checked
  at parse; bounded JSON refuses NaN/Infinity constants.
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
    manifest as manifest_mod,
    runnable as runnable_mod,
)
from torchlens._io.manifest import Manifest
from torchlens._io.paths import resolve_bundle_blob_path
from torchlens.errors import TorchLensIOError

pytestmark = pytest.mark.smoke


def _tiny() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU())


def _trace() -> tl.Trace:
    return tl.trace(
        _tiny().eval(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )


def _save(tmp_path: Path, name: str = "b.tlspec") -> Path:
    spec = tmp_path / name
    tl.save(_trace(), str(spec))
    return spec


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
def test_trace_save_fsyncs_files_and_parent_dir(tmp_path: Path, monkeypatch) -> None:
    """A trace save fsyncs its sidecars, blobs, and the parent directory.

    Fail-before: ``grep fsync torchlens/_io/`` was empty -- temp+rename survived
    a process crash but not power loss; a "successful" save could hold
    zero-length files while the old bundle was already gone.
    """

    synced: list[Path] = []
    monkeypatch.setattr(os, "fsync", _recording_fsync(synced))
    spec = _save(tmp_path)
    names = {path.name for path in synced}
    assert "manifest.json" in names, "manifest.json was not fsynced before publish"
    assert "metadata.pkl" in names, "metadata.pkl was not fsynced before publish"
    assert any(path.suffix == ".safetensors" for path in synced), "no blob was fsynced"
    assert spec.parent in synced, "the publish rename was not made durable (parent dir fsync)"


@pytest.mark.skipif(sys.platform != "linux", reason="/proc fd->path resolution")
def test_bundle_save_fsyncs_members_and_parent_dir(tmp_path: Path, monkeypatch) -> None:
    """A Bundle save fsyncs bundle.json, nested members, and the parent dir."""

    synced: list[Path] = []
    monkeypatch.setattr(os, "fsync", _recording_fsync(synced))
    target = tmp_path / "bundle.tlspec"
    tl.Bundle({"m": _trace()}).save(target, overwrite=True)
    names = {path.name for path in synced}
    assert "bundle.json" in names, "bundle.json was not fsynced before publish"
    assert "manifest.json" in names, "member manifests were not fsynced before publish"
    assert target.parent in synced, "the publish rename was not made durable (parent dir fsync)"


@pytest.mark.skipif(sys.platform != "linux", reason="/proc fd->path resolution")
def test_manifest_write_fsyncs_the_file(tmp_path: Path, monkeypatch) -> None:
    """``Manifest.write`` (fastlog finalize path) fsyncs its own file."""

    spec = _save(tmp_path)
    manifest = Manifest.read(spec / "manifest.json")
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
    tl.save(_trace(), str(spec))
    real_build_manifest = bundle_mod._build_manifest
    observed: dict[str, bool] = {}

    def observing_build_manifest(*args, **kwargs):
        observed["target_exists_mid_save"] = spec.exists()
        observed["target_loadable_mid_save"] = (spec / "manifest.json").is_file()
        return real_build_manifest(*args, **kwargs)

    monkeypatch.setattr(bundle_mod, "_build_manifest", observing_build_manifest)
    tl.save(_trace(), str(spec), overwrite=True)
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
    with pytest.raises(TorchLensIOError, match="ceiling"):
        bundle_mod._load_unified_bundle(legacy_dir)


# --------------------------------------------------------------------------- #
# MED4: manifest entry-count ceiling + duplicate blob identity refusal         #
# --------------------------------------------------------------------------- #


def _saved_manifest_data(tmp_path: Path) -> dict:
    spec = _save(tmp_path)
    return json.loads((spec / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_refuses_duplicate_blob_ids(tmp_path: Path) -> None:
    """Two tensor entries sharing one blob_id refuse at parse.

    Fail-before: duplicates silently last-won in the load-side entry indexes
    while eager verification did per-entry sha256 + safetensors decode work,
    so a KB manifest with millions of same-blob entries bought hours of CPU.
    """

    data = _saved_manifest_data(tmp_path)
    data["tensors"] = [*data["tensors"], dict(data["tensors"][0])]
    with pytest.raises(TorchLensIOError, match="duplicate blob_id"):
        Manifest.from_dict(data)


def test_manifest_refuses_duplicate_relative_paths(tmp_path: Path) -> None:
    """Two tensor entries pointing at one blob file refuse at parse."""

    data = _saved_manifest_data(tmp_path)
    forged = dict(data["tensors"][0])
    forged["blob_id"] = "zzzz-forged"
    data["tensors"] = [*data["tensors"], forged]
    with pytest.raises(TorchLensIOError, match="duplicate relative_path"):
        Manifest.from_dict(data)


def test_manifest_refuses_entry_count_above_ceiling(tmp_path: Path, monkeypatch) -> None:
    """An entry list above the structural ceiling refuses before parsing."""

    data = _saved_manifest_data(tmp_path)
    assert data["tensors"], "fixture bundle must carry at least one tensor entry"
    monkeypatch.setattr(manifest_mod, "_MAX_MANIFEST_TENSOR_ENTRIES", len(data["tensors"]) - 1)
    with pytest.raises(TorchLensIOError, match="ceiling"):
        Manifest.from_dict(data)


# --------------------------------------------------------------------------- #
# MED5: include_source=False gates the loaded-provenance passthrough           #
# --------------------------------------------------------------------------- #


def _forged_provenance() -> manifest_mod.Provenance:
    return manifest_mod.Provenance(
        provenance_version=1,
        capture_devices=["cpu"],
        dtype_policy={"default_dtype": "torch.float32", "observed_autocast": []},
        rng_state_digests={"forged_engine": "ab" * 32},
        input_hash="cd" * 32,
        model_structure_hash="ef" * 32,
        git_commit_hash="0123456789abcdef0123456789abcdef01234567",  # pragma: allowlist secret
    )


def test_resave_include_source_false_does_not_reemit_loaded_provenance(
    tmp_path: Path,
) -> None:
    """A loaded (possibly forged) provenance is not re-emitted verbatim.

    Fail-before: ``_collect_provenance`` returned ``_source_bundle_provenance``
    BEFORE the ``include_source`` gate, so resaving a loaded bundle with
    ``include_source=False`` still re-emitted git_commit_hash/input_hash/
    rng_state_digests -- propagating an attacker-authored bundle's forged
    provenance into resaves the host appears to attest.
    """

    trace = _trace()
    trace._source_bundle_provenance = _forged_provenance()
    spec = tmp_path / "resave.tlspec"
    tl.save(trace, str(spec), include_source=False)
    provenance = json.loads((spec / "manifest.json").read_text(encoding="utf-8")).get("provenance")
    assert provenance is not None
    assert provenance.get("git_commit_hash") is None, "git commit leaked despite include_source"
    assert "forged_engine" not in provenance.get("rng_state_digests", {})
    assert provenance.get("input_hash") != "cd" * 32
    assert provenance.get("model_structure_hash") != "ef" * 32


def test_resave_include_source_true_keeps_loaded_provenance(tmp_path: Path) -> None:
    """The deliberate capture-time provenance passthrough is preserved."""

    trace = _trace()
    trace._source_bundle_provenance = _forged_provenance()
    spec = tmp_path / "resave-with-source.tlspec"
    tl.save(trace, str(spec), include_source=True)
    provenance = json.loads((spec / "manifest.json").read_text(encoding="utf-8")).get("provenance")
    assert provenance is not None
    assert "forged_engine" in provenance.get("rng_state_digests", {})


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
        param_logs=(),
    )
    with pytest.raises(TorchLensIOError, match="non-tensor persistent-buffer"):
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
    with pytest.raises(TorchLensIOError, match="unresolvable"):
        resolve_bundle_blob_path(bundle_root, "blobs/a\x00b.safetensors")


@pytest.mark.parametrize(
    "shape",
    [[-1, 5], [True, 2], [2**60], list(range(300))],
    ids=["negative-dim", "bool-dim", "huge-dim", "too-many-dims"],
)
def test_tensor_entry_refuses_invalid_shape(tmp_path: Path, shape: list) -> None:
    """Negative/bool/absurd shape dims refuse at parse (latent num_elements feed)."""

    data = _saved_manifest_data(tmp_path)
    entry = dict(data["tensors"][0])
    entry["shape"] = shape
    with pytest.raises(TorchLensIOError, match="shape"):
        manifest_mod.TensorEntry.from_dict(entry)


def test_tensor_entry_refuses_bool_bytes_and_manifest_bool_ints(tmp_path: Path) -> None:
    """``bytes=True`` and ``n_layers=True`` refuse (bool passes isinstance int)."""

    data = _saved_manifest_data(tmp_path)
    entry = dict(data["tensors"][0])
    entry["bytes"] = True
    with pytest.raises(TorchLensIOError, match="bytes"):
        manifest_mod.TensorEntry.from_dict(entry)
    forged = dict(data)
    forged["n_layers"] = True
    with pytest.raises(TorchLensIOError, match="n_layers"):
        Manifest.from_dict(forged)


def test_tensor_entry_refuses_malformed_sha256(tmp_path: Path) -> None:
    """A non-hex 64-char sha256 refuses at parse, not lazily at materialize."""

    data = _saved_manifest_data(tmp_path)
    entry = dict(data["tensors"][0])
    entry["sha256"] = "Z" * 64
    with pytest.raises(TorchLensIOError, match="sha256|SHA-256"):
        manifest_mod.TensorEntry.from_dict(entry)


def test_loads_bounded_refuses_nonfinite_constants() -> None:
    """NaN/Infinity load-refuse: writers use allow_nan=False, so accepting them
    at load produced artifacts whose re-save raises (stillborn)."""

    for payload in ('{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}'):
        with pytest.raises(json.JSONDecodeError, match="non-finite"):
            json_mod.loads_bounded(payload)


# --------------------------------------------------------------------------- #
# R60/F6: metadata.pkl object-count ceiling + R65/F10 typed non-mapping guard  #
# --------------------------------------------------------------------------- #


def test_metadata_pkl_opcode_ceiling_refuses_object_bomb(tmp_path: Path, monkeypatch) -> None:
    """A metadata pickle packed with tiny values refuses on opcode count.

    Fail-before: the byte cap admitted a pickle whose ~5x RSS expansion happened
    entirely BEFORE any structural check (measured 76 MiB of ints -> ~390 MiB;
    the old 4 GiB byte cap projected ~20 GiB) and the eventual refusal escaped
    as a raw stdlib TypeError.
    """

    spec = _save(tmp_path)
    (spec / "metadata.pkl").write_bytes(pickle.dumps([0] * 100_000))
    monkeypatch.setattr(bundle_mod, "_METADATA_PKL_PRESCAN_BYTES", 0)
    monkeypatch.setattr(bundle_mod, "_MAX_METADATA_PKL_OPCODES", 1_000)
    with pytest.raises(TorchLensIOError, match="opcode allocation ceiling") as excinfo:
        tl.load(str(spec))
    assert excinfo.value.fields["code"] == "metadata_object_count_exceeded"


def test_metadata_pkl_non_mapping_payload_refuses_typed(tmp_path: Path) -> None:
    """A non-mapping metadata payload refuses typed, not as a raw stdlib error.

    Fail-before: ``tl.load`` on a bundle whose ``metadata.pkl`` held a plain
    list escaped as ``ValueError: dictionary update sequence element #0 ...``
    from the downstream dict() walk -- untyped, no code, no remedy.
    """

    spec = _save(tmp_path)
    (spec / "metadata.pkl").write_bytes(pickle.dumps([[0] * 3] * 3))
    with pytest.raises(TorchLensIOError, match="not a metadata mapping") as excinfo:
        tl.load(str(spec))
    assert excinfo.value.fields["code"] == "metadata_payload_not_a_mapping"


def test_bundle_save_failure_names_cause_and_code(tmp_path: Path, monkeypatch) -> None:
    """R65: the highest-traffic save door names its cause, code, and remedy.

    Fail-before: every non-typed save failure became the content-free
    ``TorchLensIOError: Failed to save bundle at <path>.`` with empty fields.
    """

    def _boom(state, handle) -> None:
        raise TypeError("cannot pickle '_thread.lock' object")

    monkeypatch.setattr(bundle_mod, "dump_canonical_metadata", _boom)
    with pytest.raises(TorchLensIOError, match="cannot pickle") as excinfo:
        _save(tmp_path)
    assert excinfo.value.fields["code"] == "bundle_save_failed"
    assert excinfo.value.fields["cause_type"] == "TypeError"
    assert "Remedy:" in str(excinfo.value)


def test_unverified_capture_disclosure_survives_save_and_load(tmp_path: Path) -> None:
    """P7/R10: capture_verified=False must not launder to no-claim across save.

    Fail-before: the disclosure triple was FieldPolicy.DROP, so a capture
    TorchLens itself refused to bless loaded as verified=None / reason=None --
    the round trip IMPROVED a verdict. The negative claim now persists as a
    string-only row; True/None stay session-time so a loaded artifact can
    never CLAIM verification.
    """

    trace = _trace()
    trace.capture_verified = False
    trace.capture_verification_reason = "escape_rescue_unrecovered"
    spec = tmp_path / "unverified.tlspec"
    tl.save(trace, str(spec))
    loaded = tl.load(str(spec))
    assert loaded.capture_verified is False
    assert loaded.capture_verification_reason == "escape_rescue_unrecovered"

    clean = _save(tmp_path, "clean.tlspec")
    loaded_clean = tl.load(str(clean))
    assert loaded_clean.capture_verified is None
