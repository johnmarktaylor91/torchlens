"""Untrusted-load follow-up: four loaders refuse links, FIFOs and forged cache entries.

Each test hands a loader one hostile piece and asserts it refuses (or, for the capture
cache, misses) before acting on it:

- a fastlog bundle whose ``metadata.json`` / ``fastlog_index.jsonl`` / blob is a symlink
  or a FIFO (a FIFO blob used to hang the digest forever);
- a history artifact whose ``index.json`` names a chunk outside the artifact (absolute
  path, ``..``, or a symlink) with the chunk's real digest, or a FIFO chunk;
- a supplied capture-cache directory carrying its own secret and an entry tagged with
  it (a planted code-execution pickle used to authenticate and run);
- a bundle artifact whose ``members/`` directory is a symlink to members elsewhere.

FIFO cases run in a daemon thread with a timeout and release the blocked reader after.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import os
import pickle
import shutil
import stat
import threading
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture.prepare import cache as prepare_cache
from torchlens._io import ArtifactError
from torchlens._observability import (
    ObservationRecord,
    RunningStats,
    RunRecord,
    SiteRecord,
    StepBlockRecord,
)
from torchlens._observability._artifact import CommittedBlock, HistoryWriter
from torchlens.errors import HistoryArtifactError
from torchlens.trackers import HistoryReader

# The package re-exports the recover() FUNCTION under the module's name, so the module
# (and its strict ``load``) comes from the import system directly.
recover_module = importlib.import_module("torchlens.recording._recover")

needs_symlinks = pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
needs_fifo = pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX FIFOs")


def _finishes(call: Any, fifo: Path, timeout: float = 20.0) -> tuple[bool, BaseException | None]:
    """Run ``call`` in a daemon thread; return (finished, raised exception).

    A reader still blocked on ``fifo`` after the timeout is released by opening the FIFO's
    write end, so a red run does not leave a thread hanging for the rest of the session.
    """

    outcome: dict[str, BaseException | None] = {"error": None}

    def _target() -> None:
        """Record the call's exception, if any."""

        try:
            call()
        except BaseException as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout)
    finished = not worker.is_alive()
    if not finished:
        try:
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            pass
        worker.join(5.0)
    return finished, outcome["error"]


# --------------------------------------------------------------------------- #
# (1) fastlog loader                                                           #
# --------------------------------------------------------------------------- #


class _Fastlog(nn.Module):
    """Small model for fastlog bundles."""

    def __init__(self) -> None:
        """Build the layer."""

        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Map and rectify."""

        return torch.relu(self.linear(value))


def _fastlog_bundle(tmp_path: Path) -> Path:
    """Write a finalized disk-only fastlog bundle and return its path."""

    path = tmp_path / "run.tlfast"
    tl.record(
        _Fastlog(),
        torch.ones(1, 3),
        storage=tl.options.StorageOptions(path=path, retain_in_memory=False),
        recording=tl.options.RecordingOptions(default_op=True),
    )
    return path


def _first_blob(bundle: Path) -> Path:
    """Return the first persisted safetensors blob."""

    return next((bundle / "blobs").glob("*.safetensors"))


def _relink(path: Path, outside: Path) -> None:
    """Move ``path`` to ``outside`` and leave a symlink to it in its place."""

    shutil.move(str(path), str(outside))
    path.symlink_to(outside)


def test_fastlog_writer_bundle_still_loads(tmp_path: Path) -> None:
    """A bundle as the writer leaves it loads strictly with every record."""

    bundle = _fastlog_bundle(tmp_path)
    loaded = recover_module.load(bundle)
    assert loaded.records and not loaded.recovery_warnings


@needs_symlinks
@pytest.mark.parametrize("name", ["metadata.json", "fastlog_index.jsonl"])
def test_fastlog_symlinked_sidecar_refuses(tmp_path: Path, name: str) -> None:
    """A symlinked sidecar refuses on strict load; recover never reads its target."""

    bundle = _fastlog_bundle(tmp_path)
    _relink(bundle / name, tmp_path / f"outside-{name}")
    with pytest.raises(ArtifactError, match="symlink or not a regular file") as info:
        recover_module.load(bundle)
    assert info.value.fields["code"] == "load_path_symlink_rejected"


@needs_symlinks
def test_fastlog_symlinked_metadata_is_not_read_on_recover(tmp_path: Path) -> None:
    """Recover treats a symlinked ``metadata.json`` as absent instead of reading its target."""

    bundle = _fastlog_bundle(tmp_path)
    (bundle / "manifest.json").unlink()
    outside = tmp_path / "outside-metadata.json"
    _relink(bundle / "metadata.json", outside)
    outside.write_text(json.dumps({"halted": True, "stop_reason": "planted"}), encoding="utf-8")
    recovered = tl.recording.recover(bundle)
    assert recovered.stop_reason is None and not recovered.is_stopped


@needs_symlinks
def test_fastlog_symlinked_blob_refuses(tmp_path: Path) -> None:
    """A blob replaced by a symlink to identical bytes (inside ``blobs/``) refuses on strict load.

    The target stays inside ``blobs/`` so the containment check passes; only the new
    plain-file rule refuses it.
    """

    bundle = _fastlog_bundle(tmp_path)
    blob = _first_blob(bundle)
    _relink(blob, blob.with_name("moved.bin"))
    with pytest.raises(ArtifactError, match="symlink or not a regular file"):
        recover_module.load(bundle)


@needs_fifo
def test_fastlog_fifo_blob_does_not_hang_recover(tmp_path: Path) -> None:
    """A FIFO in place of a blob is skipped with a warning instead of blocking the digest."""

    bundle = _fastlog_bundle(tmp_path)
    blob = _first_blob(bundle)
    blob.unlink()
    os.mkfifo(blob)
    holder: dict[str, Any] = {}
    finished, error = _finishes(
        lambda: holder.setdefault("rec", tl.recording.recover(bundle)), blob
    )
    assert finished, "recover blocked on a FIFO blob"
    assert error is None, error
    assert any("not a regular file" in item for item in holder["rec"].recovery_warnings)


@needs_fifo
def test_fastlog_fifo_index_does_not_hang_recover(tmp_path: Path) -> None:
    """A FIFO ``fastlog_index.jsonl`` refuses promptly on the no-manifest recover path."""

    bundle = _fastlog_bundle(tmp_path)
    (bundle / "manifest.json").unlink()
    index = bundle / "fastlog_index.jsonl"
    index.unlink()
    os.mkfifo(index)
    finished, error = _finishes(lambda: tl.recording.recover(bundle), index)
    assert finished, "recover blocked on a FIFO index"
    assert isinstance(error, tl.errors.RecoveryError), error


# --------------------------------------------------------------------------- #
# (2) history artifact chunk containment                                      #
# --------------------------------------------------------------------------- #


def _history(tmp_path: Path) -> Path:
    """Write a one-chunk history artifact and return its directory."""

    art = tmp_path / "hist"
    writer = HistoryWriter(art, RunRecord(run_id="run-a", segment_id="seg-a"))
    writer.add_site(SiteRecord(site_id="module:enc.0", kind="module", display_label="enc.0"))
    spine = RunningStats()
    spine.update(torch.tensor([1.0, -1.0, 0.0]))
    observation = ObservationRecord(
        global_step=0,
        site_id="module:enc.0",
        stream="activation",
        phase="forward",
        presence="observed",
        spine=spine.compute(),
        sketch=None,
    )
    record = StepBlockRecord(segment_id="seg-a", global_step=0, provenance="explicit")
    writer.append_block(
        CommittedBlock(block=record, observations=(observation,), step_lo=0, step_hi=0)
    )
    return art


def _point_index_at(art: Path, file_value: str) -> None:
    """Rewrite the single index entry's ``file`` (its recorded digest stays the real one)."""

    index_path = art / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["chunks"][0]["file"] = file_value
    index_path.write_text(json.dumps(index), encoding="utf-8")


def _history_refusal(art: Path) -> HistoryArtifactError:
    """Read every observation expecting a typed refusal; return it."""

    with pytest.raises(HistoryArtifactError) as info:
        HistoryReader(art).observations()
    return info.value


def test_history_writer_chunks_still_read(tmp_path: Path) -> None:
    """The writer's own chunk names pass the new name check."""

    assert len(HistoryReader(_history(tmp_path)).observations()) == 1


@pytest.mark.parametrize("form", ["absolute", "dotdot", "subdir"])
def test_history_index_cannot_name_a_chunk_outside(tmp_path: Path, form: str) -> None:
    """An index ``file`` outside the artifact refuses even when the bytes match its digest."""

    art = _history(tmp_path)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (art / "sub").mkdir()
    chunk = art / "chunk-000000.npz"
    targets = {
        "absolute": (outside_dir / "x.npz", str(outside_dir / "x.npz")),
        "dotdot": (outside_dir / "x.npz", "../outside/x.npz"),
        "subdir": (art / "sub" / "chunk-000000.npz", "sub/chunk-000000.npz"),
    }
    target, file_value = targets[form]
    shutil.copyfile(chunk, target)
    _point_index_at(art, file_value)
    assert _history_refusal(art).fields["code"] == "history_artifact_corrupt"


@needs_symlinks
def test_history_symlinked_chunk_refuses(tmp_path: Path) -> None:
    """A chunk replaced by a symlink to identical bytes outside the artifact refuses."""

    art = _history(tmp_path)
    _relink(art / "chunk-000000.npz", tmp_path / "outside.npz")
    error = _history_refusal(art)
    assert error.fields["code"] == "history_artifact_corrupt"
    assert "not a regular file" in str(error)


@needs_fifo
def test_history_fifo_chunk_does_not_hang(tmp_path: Path) -> None:
    """A FIFO chunk refuses promptly instead of blocking the digest."""

    art = _history(tmp_path)
    chunk = art / "chunk-000000.npz"
    chunk.unlink()
    os.mkfifo(chunk)
    finished, error = _finishes(lambda: HistoryReader(art).observations(), chunk)
    assert finished, "HistoryReader blocked on a FIFO chunk"
    assert isinstance(error, HistoryArtifactError), error


# --------------------------------------------------------------------------- #
# (3) capture cache: a supplied directory cannot forge entries                #
# --------------------------------------------------------------------------- #


class _Gadget:
    """A ``__reduce__`` gadget: unpickling it writes a marker file."""

    def __init__(self, marker: Path) -> None:
        self._marker = marker

    def __reduce__(self) -> tuple:  # noqa: D105 - pickle protocol hook
        return (_write_marker, (str(self._marker),))


def _write_marker(path: str) -> str:
    """Stand-in for ``os.system``: proves the gadget ran."""

    Path(path).write_text("pwned", encoding="utf-8")
    return path


@pytest.fixture()
def cache_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Point the cache and the per-user state directory at private per-test paths."""

    monkeypatch.setenv("TORCHLENS_CACHE_DIR", str(tmp_path / "tlcache"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return tmp_path / "tlcache" / "capture", tmp_path / "state" / "torchlens"


def _cached_trace(model: nn.Module, value: torch.Tensor) -> tl.Trace:
    """Trace with the capture cache on."""

    return tl.trace(model, value, capture=tl.options.CaptureOptions(cache=True), save="all")


def test_capture_cache_hits_with_secret_outside_the_cache(cache_env: tuple[Path, Path]) -> None:
    """The normal path still hits, and the secret lives outside the cache directory, 0600."""

    cache_root, state_dir = cache_env
    model, value = nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.rand(2, 4)
    assert _cached_trace(model, value).provenance.cache_hit is False
    assert _cached_trace(model, value).provenance.cache_hit is True
    secret = state_dir / "capture_cache_secret"
    assert secret.is_file() and not secret.is_symlink()
    if os.name == "posix":
        assert stat.S_IMODE(secret.stat().st_mode) == 0o600
    assert not any(path.name.endswith("secret") for path in cache_root.rglob("*"))


def test_capture_cache_secret_directory_that_cannot_be_created_refuses(
    cache_env: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A state home that is a plain file refuses typed, naming the directory and the remedy."""

    blocker = tmp_path / "state-is-a-file"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("XDG_STATE_HOME", str(blocker))
    with pytest.raises(ArtifactError, match="cannot be created or inspected") as info:
        prepare_cache._capture_cache_secret_dir()
    assert str(blocker / "torchlens") in str(info.value)
    assert "XDG_STATE_HOME" in str(info.value)


def test_supplied_cache_directory_cannot_forge_an_entry(
    cache_env: tuple[Path, Path], tmp_path: Path
) -> None:
    """A cache directory carrying its own 0600 secret and a gadget tagged with it is inert.

    Fail-before: the secret was read from the cache directory itself, so a restored or
    copied-in directory (files owned by this user, mode 0600) authenticated its own
    planted pickle and the next cache hit ran it.
    """

    cache_root, _state_dir = cache_env
    model, value = nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.rand(2, 4)
    _cached_trace(model, value)
    entry = next(cache_root.glob("*.pkl"))
    forged_secret = os.urandom(32)
    secret_file = cache_root / ".capture_cache_secret"
    secret_file.unlink(missing_ok=True)
    descriptor = os.open(secret_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(forged_secret)
    marker = tmp_path / "pwned"
    payload = pickle.dumps(_Gadget(marker))
    tag = hmac.new(forged_secret, payload, hashlib.sha256).hexdigest().encode("ascii")
    entry.write_bytes(prepare_cache._CAPTURE_CACHE_MAGIC + tag + b"\n" + payload)

    with pytest.warns(UserWarning, match="Ignoring TorchLens capture cache entry"):
        refreshed = _cached_trace(model, value)
    assert not marker.exists(), "a forged cache entry was unpickled"
    assert refreshed.provenance.cache_hit is False
    assert _cached_trace(model, value).provenance.cache_hit is True


# --------------------------------------------------------------------------- #
# (4) bundle members/ directory symlink                                       #
# --------------------------------------------------------------------------- #


class _Shift(nn.Module):
    """A linear map plus a fixed offset, so two members differ numerically."""

    def __init__(self, offset: float) -> None:
        """Build the map with a fixed seed and keep the offset."""

        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(3, 2)
        self.offset = offset

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Map, offset and rectify."""

        return torch.relu(self.fc(value) + self.offset)


@needs_symlinks
@pytest.mark.parametrize("listed", [True, False], ids=["listed", "unlisted"])
def test_bundle_symlinked_members_directory_refuses(tmp_path: Path, listed: bool) -> None:
    """``members/`` replaced by a symlink to the same members elsewhere refuses typed.

    With the member files still listed in the bundle manifest, the envelope's ancestor
    check refuses first. The crafted shape drops them from the listing, so only the
    members' own envelope hashes are checked: before the fix a symlinked ``members/``
    then loaded every member from outside the container.
    """

    value = torch.randn(2, 3, generator=torch.Generator().manual_seed(1))
    bundle = tl.Bundle(
        {"clean": tl.trace(_Shift(0.0), value), "changed": tl.trace(_Shift(0.5), value)},
        baseline="clean",
    )
    target = tmp_path / "pair.tlspec"
    bundle.save(target)
    assert list(tl.load(target).members) == ["clean", "changed"]
    _relink(target / "members", tmp_path / "outside-members")
    if not listed:
        manifest_path = target / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"] = {
            relpath: entry
            for relpath, entry in manifest["files"].items()
            if not relpath.startswith("members/")
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ArtifactError) as info:
            tl.load(target)
    assert info.value.fields["code"] == "load_path_symlink_rejected"
