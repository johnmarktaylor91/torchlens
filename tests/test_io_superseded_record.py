"""A failed or killed overwrite leaves a record INSIDE the older copy, so the refusal travels.

The ``newer_save_unfinished`` sibling check reads a ``<name>.tmp.<hex>`` staging folder beside
the path, which stays on the machine that ran the save. Before an overwrite stages, every
overwriting saver now writes ``NEWER_SAVE_UNFINISHED.json`` inside the installed copy: a
completed overwrite replaces the copy and the record with it, a failed one restores the copy
with the record in it, and a killed one leaves it. A copy of the path (``shutil.copytree``,
fresh mtimes, no siblings) therefore still refuses on any machine until
``tl.io.cleanup_tmp(path)`` removes the record. No manifest lists it and no digest covers it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import _staging as staging_mod, bundle as bundle_mod, tlspec as tlspec_mod
from torchlens._io._staging import SUPERSEDED_RECORD
from torchlens.errors import ArtifactError, TorchLensWarning
from torchlens.intervention import _recipe_io
from torchlens.io import cleanup_tmp

CODE = "newer_save_unfinished"


def _trace(seed: int) -> tl.Trace:
    """A tiny capture whose saved output differs by ``seed``."""

    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    return tl.trace(model, torch.randn(2, 4), save="all")


def _out(trace: tl.Trace) -> torch.Tensor:
    """The model output payload of ``trace``."""

    return trace.output_ops[0].out


def _fail_payload_write(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
    """Make the next ``Trace.save`` fail while writing a tensor payload (mid-staging)."""

    def failing_write(*_args: object, **_kwargs: object) -> None:
        raise exc

    monkeypatch.setattr(bundle_mod, "_write_payload_blob", failing_write)


def _copy_elsewhere(path: Path, root: Path) -> Path:
    """Copy the artifact alone into a fresh directory, without preserving any mtime.

    ``shutil.copyfile`` copies bytes only, and every copied entry is then stamped with one
    old time, so neither a sibling staging folder nor the original mtimes come along.
    """

    destination = root / "elsewhere" / path.name
    destination.parent.mkdir(parents=True)
    shutil.copytree(path, destination, copy_function=shutil.copyfile)
    stamp = 1_000_000_000
    for entry in [destination, *destination.rglob("*")]:
        os.utime(entry, (stamp, stamp))
    assert not list(destination.parent.glob(f"{destination.name}.tmp.*"))
    return destination


def _digests(root: Path, *, skip_record: bool = True) -> dict[str, str]:
    """``{relpath: sha256}`` for every file under ``root`` (the record skipped by default)."""

    result: dict[str, str] = {}
    for entry in sorted(root.rglob("*")):
        if not entry.is_file():
            continue
        if skip_record and staging_mod.is_superseded_record_name(entry.name):
            continue
        result[entry.relative_to(root).as_posix()] = hashlib.sha256(entry.read_bytes()).hexdigest()
    return result


def _refusal(path: Path, **load_kwargs: object) -> ArtifactError:
    """Load ``path`` and return the ``newer_save_unfinished`` refusal it must raise."""

    with pytest.raises(ArtifactError) as info:
        tl.load(path, **load_kwargs)
    assert info.value.fields.get("code") == CODE
    return info.value


def _record(path: Path) -> dict[str, object]:
    """The parsed record inside ``path``."""

    data = json.loads((path / SUPERSEDED_RECORD).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_copy_of_failed_overwrite_refuses_until_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copied older copy (no sibling, fresh mtimes) refuses on the record; cleanup clears it."""

    path = tmp_path / "run.tlspec"
    older = _trace(0)
    older.save(path)
    _fail_payload_write(monkeypatch, OSError("disk full"))
    with pytest.raises(ArtifactError):
        _trace(1).save(path, overwrite=True)
    monkeypatch.undo()

    record = _record(path)
    assert record["state"] == "failed"
    assert record["reason"] == "OSError", "the scrubbed TYPE, never the message"
    assert isinstance(record["started_at"], str)
    assert "disk full" not in (path / SUPERSEDED_RECORD).read_text(encoding="utf-8")

    copy = _copy_elsewhere(path, tmp_path)
    error = _refusal(copy)
    assert SUPERSEDED_RECORD in str(error)
    assert "the save failed" in str(error)
    assert "file on disk is the older copy" in str(error)
    assert "tl.io.cleanup_tmp(path) removes it" in str(error)
    assert error.fields["record"] == str(copy / SUPERSEDED_RECORD)
    assert error.fields["staging"] is None
    assert error.fields["partial"] is True
    assert error.fields["reason"] == "OSError"
    _refusal(copy, lazy=True)

    assert cleanup_tmp(copy) == [copy / SUPERSEDED_RECORD]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = tl.load(copy)
    assert torch.equal(_out(loaded), _out(older))


def test_copy_of_killed_overwrite_refuses_until_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A save killed mid-staging leaves a ``started`` record that refuses in a copy."""

    path = tmp_path / "run.tlspec"
    older = _trace(0)
    older.save(path)
    # A SIGKILL runs no handler: no PARTIAL mark, no restore, no record rewrite.
    monkeypatch.setattr(bundle_mod, "_run_save_recovery", lambda *_a, **_k: None)
    _fail_payload_write(monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _trace(1).save(path, overwrite=True)
    monkeypatch.undo()
    assert _record(path)["state"] == "started"

    copy = _copy_elsewhere(path, tmp_path)
    error = _refusal(copy)
    assert "killed or is still writing" in str(error)
    assert error.fields["partial"] is False
    assert error.fields["reason"] is None

    assert cleanup_tmp(copy) == [copy / SUPERSEDED_RECORD]
    assert torch.equal(_out(tl.load(copy)), _out(older))

    # On the original machine the unmarked staging folder keeps both refusals until force.
    with pytest.warns(TorchLensWarning, match="force=True"):
        assert cleanup_tmp(path) == []
    assert (path / SUPERSEDED_RECORD).exists(), "kept while a non-PARTIAL folder remains"
    _refusal(path)
    removed = cleanup_tmp(path, force=True)
    assert path / SUPERSEDED_RECORD in removed
    assert torch.equal(_out(tl.load(path)), _out(older))


def test_copy_taken_before_the_failed_overwrite_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy made before the overwrite started carries no record and loads cleanly."""

    path = tmp_path / "run.tlspec"
    older = _trace(0)
    older.save(path)
    early = _copy_elsewhere(path, tmp_path)
    _fail_payload_write(monkeypatch, OSError("disk full"))
    with pytest.raises(ArtifactError):
        _trace(1).save(path, overwrite=True)
    monkeypatch.undo()

    assert not (early / SUPERSEDED_RECORD).exists()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = tl.load(early)
    assert torch.equal(_out(loaded), _out(older))


def test_successful_overwrite_leaves_no_record(tmp_path: Path) -> None:
    """A completed overwrite replaces the copy, and the record and its temp file with it."""

    path = tmp_path / "run.tlspec"
    _trace(0).save(path)
    newest = _trace(1)
    newest.save(path, overwrite=True)

    assert not [p for p in path.iterdir() if staging_mod.is_superseded_record_name(p.name)]
    assert not list(tmp_path.glob("run.tlspec.tmp.*"))
    assert not list(tmp_path.glob("run.tlspec.bak.*"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = tl.load(path)
    assert torch.equal(_out(loaded), _out(newest))


def test_record_leaves_the_older_copy_and_its_verification_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record is outside the manifest and the digests: every listed byte is unchanged."""

    from torchlens._io.format3 import contract, validate

    path = tmp_path / "run.tlspec"
    _trace(0).save(path)
    before = _digests(path, skip_record=False)
    manifest_bytes = (path / contract.MANIFEST_FILE).read_bytes()
    _fail_payload_write(monkeypatch, OSError("disk full"))
    with pytest.raises(ArtifactError):
        _trace(1).save(path, overwrite=True)
    monkeypatch.undo()

    assert (path / SUPERSEDED_RECORD).is_file()
    assert _digests(path) == before, "the older copy's files are byte-identical"
    assert (path / contract.MANIFEST_FILE).read_bytes() == manifest_bytes
    manifest = json.loads(manifest_bytes)
    assert SUPERSEDED_RECORD not in manifest["files"]
    # Verification of the copy, with the record present, passes and names nothing.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate.verify_listed_files(path, manifest)

    cleanup_tmp(path)
    assert _digests(path, skip_record=False) == before


def test_record_write_failure_fails_the_save_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the record cannot be written the overwrite fails first; the older copy is untouched."""

    path = tmp_path / "run.tlspec"
    older = _trace(0)
    older.save(path)
    before = _digests(path, skip_record=False)

    def refuse_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("read-only copy")

    monkeypatch.setattr(staging_mod, "_write_record", refuse_write)
    with pytest.raises(ArtifactError) as info:
        _trace(1).save(path, overwrite=True)
    monkeypatch.undo()

    assert info.value.fields.get("code") == "bundle_save_failed"
    assert not list(tmp_path.glob("run.tlspec.tmp.*")), "nothing was staged"
    assert _digests(path, skip_record=False) == before
    assert torch.equal(_out(tl.load(path)), _out(older))


def test_torn_record_write_is_ignored_and_swept(tmp_path: Path) -> None:
    """A crash before the record's rename leaves only its temp file: load serves the copy."""

    path = tmp_path / "run.tlspec"
    older = _trace(0)
    older.save(path)
    torn = path / f"{SUPERSEDED_RECORD}.tmp.0123abcd"
    torn.write_text('{"kind": "torchlens.newer', encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = tl.load(path)
    assert torch.equal(_out(loaded), _out(older))
    assert cleanup_tmp(path) == [torn]


@pytest.mark.parametrize("content", [b"", b"not json", b"[1, 2]"])
def test_damaged_record_still_refuses(tmp_path: Path, content: bytes) -> None:
    """An emptied or unparseable record fails closed as an unfinished save."""

    path = tmp_path / "run.tlspec"
    _trace(0).save(path)
    (path / SUPERSEDED_RECORD).write_bytes(content)

    error = _refusal(path)
    assert "killed or is still writing" in str(error)
    assert cleanup_tmp(path) == [path / SUPERSEDED_RECORD]
    tl.load(path)


def test_bundle_saver_copy_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``Bundle.save`` writes the record too; its copy refuses until cleanup."""

    path = tmp_path / "pair.tlspec"
    tl.Bundle({"m": _trace(0)}).save(path)
    real_write_json = tlspec_mod._TlSpecWriter.write_json

    def failing_write_json(target, payload):  # type: ignore[no-untyped-def]
        if Path(target).name == "manifest.json":
            raise RuntimeError("injected bundle manifest failure")
        return real_write_json(target, payload)

    monkeypatch.setattr(tlspec_mod._TlSpecWriter, "write_json", staticmethod(failing_write_json))
    with pytest.raises(RuntimeError, match="injected bundle manifest failure"):
        tl.Bundle({"m": _trace(1)}).save(path, overwrite=True)
    monkeypatch.undo()

    assert _record(path)["reason"] == "RuntimeError"
    copy = _copy_elsewhere(path, tmp_path)
    _refusal(copy)
    cleanup_tmp(copy)
    assert list(tl.load(copy).members) == ["m"]


def test_recipe_saver_copy_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``Recipe.save`` writes the record too; its copy refuses until cleanup."""

    path = tmp_path / "recipe.tlspec"
    older = tl.when(tl.func("relu"), tl.scale(0.5))
    older.save(path)
    real_write_tree = _recipe_io._write_recipe_tree

    def failing_write_tree(tmp, recipe, save_level):  # type: ignore[no-untyped-def]
        real_write_tree(tmp, recipe, save_level)
        raise RuntimeError("injected recipe failure")

    monkeypatch.setattr(_recipe_io, "_write_recipe_tree", failing_write_tree)
    with pytest.raises(RuntimeError, match="injected recipe failure"):
        tl.when(tl.func("relu"), tl.scale(0.25)).save(path, overwrite=True)
    monkeypatch.undo()

    copy = _copy_elsewhere(path, tmp_path)
    _refusal(copy)
    cleanup_tmp(copy)
    assert tl.load(copy).digest == older.digest


def test_intervention_spec_saver_copy_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``save_intervention`` restores the older spec with the record in it."""

    path = tmp_path / "spec.tlspec"
    torch.manual_seed(0)
    trace = tl.trace(
        nn.Linear(2, 2),
        torch.ones(1, 2),
        capture=tl.options.CaptureOptions(replayable=True),
    )
    trace.save_intervention(path, level="audit")
    original_rename = os.rename

    def fail_install(source: str | Path, target: str | Path) -> None:
        if Path(source).name.startswith(f"{path.name}.tmp.") and Path(target) == path:
            raise OSError("simulated intervention swap refusal")
        original_rename(source, target)

    monkeypatch.setattr("torchlens.intervention._save.os.rename", fail_install)
    with pytest.raises(OSError, match="swap refusal"):
        trace.save_intervention(path, level="audit", overwrite=True)
    monkeypatch.undo()

    assert _record(path)["reason"] == "OSError"
    copy = _copy_elsewhere(path, tmp_path)
    with pytest.raises(ArtifactError) as info:
        tl.io.load_intervention_spec(copy)
    assert info.value.fields.get("code") == CODE
    cleanup_tmp(copy)
    tl.io.load_intervention_spec(copy)
