"""``tl.load`` refuses a path whose newer save failed or never finished (``newer_save_unfinished``).

A failed overwrite restores the previous artifact and leaves its staging folder
``<name>.tmp.<hex>`` beside the path (marked ``PARTIAL`` with a ``REASON.txt``); a killed
save leaves the folder unmarked. Before this check a later reader got the older artifact
as a normal complete load with no sign of the failure. ``tl.io.cleanup_tmp(path)`` (with
``force=True`` for an unmarked folder) is how a caller accepts the older copy.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import bundle as bundle_mod, streaming as streaming_mod, tlspec as tlspec_mod
from torchlens._io._staging import refuse_stale_load
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


def _staging(path: Path) -> list[Path]:
    """The staging folders beside ``path``."""

    return sorted(path.parent.glob(f"{path.name}.tmp.*"))


def _fail_payload_write(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
    """Make the next ``Trace.save`` fail while writing a tensor payload."""

    def failing_write(*_args: object, **_kwargs: object) -> None:
        raise exc

    monkeypatch.setattr(bundle_mod, "_write_payload_blob", failing_write)


def _refusal(path: Path, **load_kwargs: object) -> ArtifactError:
    """Load ``path`` and return the ``newer_save_unfinished`` refusal it must raise."""

    with pytest.raises(ArtifactError) as info:
        tl.load(path, **load_kwargs)
    assert info.value.fields.get("code") == CODE
    return info.value


def test_failed_overwrite_refuses_until_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed overwrite makes load refuse, naming the folder and its REASON; cleanup clears it."""

    path = tmp_path / "run.tlspec"
    older = _trace(0)
    older.save(path)
    _fail_payload_write(monkeypatch, OSError("disk full"))
    with pytest.raises(ArtifactError):
        _trace(1).save(path, overwrite=True)
    monkeypatch.undo()

    (staging,) = _staging(path)
    error = _refusal(path)
    assert staging.name in str(error)
    assert "marked PARTIAL" in str(error)
    assert "OSError" in str(error), "the REASON text is quoted"
    assert "file on disk is the older copy" in str(error)
    assert "cleanup_tmp(path)" in str(error)
    assert error.fields["staging"] == str(staging)
    assert error.fields["partial"] is True
    assert error.fields["reason"] == "OSError"
    assert error.fields["remedy"]
    _refusal(path, lazy=True)

    assert staging in cleanup_tmp(path)
    assert torch.equal(_out(tl.load(path)), _out(older))


def test_killed_save_refuses_and_needs_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unmarked staging folder (a killed save) refuses; only ``force=True`` cleanup clears it."""

    path = tmp_path / "run.tlspec"
    older = _trace(0)
    older.save(path)
    # A SIGKILL runs no handler: no PARTIAL mark, no restore.
    monkeypatch.setattr(bundle_mod, "_run_save_recovery", lambda *_a, **_k: None)
    _fail_payload_write(monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _trace(1).save(path, overwrite=True)
    monkeypatch.undo()

    (staging,) = _staging(path)
    assert not (staging / "PARTIAL").exists()
    error = _refusal(path)
    assert "not marked PARTIAL" in str(error)
    assert "force=True" in str(error)
    assert error.fields["partial"] is False

    with pytest.warns(TorchLensWarning, match="force=True"):
        assert cleanup_tmp(path) == []
    assert staging.exists()
    _refusal(path)

    assert staging in cleanup_tmp(path, force=True)
    assert torch.equal(_out(tl.load(path)), _out(older))


def test_earlier_failure_then_successful_save_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leftover folder older than the installed copy warns once and does not refuse."""

    path = tmp_path / "run.tlspec"
    _trace(0).save(path)
    _fail_payload_write(monkeypatch, OSError("disk full"))
    with pytest.raises(ArtifactError):
        _trace(1).save(path, overwrite=True)
    monkeypatch.undo()
    (leftover,) = _staging(path)
    # Date the failure clearly before the next save, so a coarse filesystem clock
    # cannot give both the same timestamp.
    stamp = leftover.stat().st_mtime - 10
    os.utime(leftover, (stamp, stamp))
    newest = _trace(2)
    newest.save(path, overwrite=True)

    with pytest.warns(TorchLensWarning, match="cleanup_tmp") as record:
        loaded = tl.load(path)
    leftovers = [w.message for w in record if "Leftover staging folder" in str(w.message)]
    assert len(leftovers) == 1
    assert leftovers[0].fields["code"] == "older_staging_leftover"
    assert leftovers[0].fields["remedy"].startswith("run tl.io.cleanup_tmp(path)")
    assert torch.equal(_out(loaded), _out(newest))


def test_glob_metacharacters_do_not_reach_a_sibling(tmp_path: Path) -> None:
    """``job*`` must not pick up ``jobB``'s newer staging folder."""

    path = tmp_path / "job*"
    _trace(0).save(path)
    sibling_staging = tmp_path / "jobB.tmp.0123"
    sibling_staging.mkdir()
    (sibling_staging / "PARTIAL").write_text("", encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchLensWarning)
        tl.load(path)


def test_symlinked_staging_folder_refuses(tmp_path: Path) -> None:
    """A symlinked staging entry refuses at load, as ``cleanup_tmp`` refuses it."""

    path = tmp_path / "run.tlspec"
    _trace(0).save(path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "run.tlspec.tmp.link").symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(ArtifactError, match="symlink temp directory"):
        tl.load(path)
    with pytest.raises(ArtifactError, match="symlink temp directory"):
        cleanup_tmp(path)


def test_first_save_failure_keeps_the_not_found_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed first save leaves nothing at the path; load fails exactly as for a fresh path."""

    path = tmp_path / "run.tlspec"
    _fail_payload_write(monkeypatch, OSError("disk full"))
    with pytest.raises(ArtifactError):
        _trace(1).save(path)
    monkeypatch.undo()
    assert not path.exists()
    assert _staging(path), "the failed first save leaves its PARTIAL staging folder"

    fresh = tmp_path / "never.tlspec"
    with pytest.raises(Exception) as fresh_info:
        tl.load(fresh)
    with pytest.raises(Exception) as failed_info:
        tl.load(path)
    assert type(failed_info.value) is type(fresh_info.value)
    assert str(failed_info.value).replace(str(path), "<path>") == str(fresh_info.value).replace(
        str(fresh), "<path>"
    )


def test_bundle_saver_failed_overwrite_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Bundle.save`` keeps its staging folder PARTIAL after a failed overwrite."""

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

    (staging,) = _staging(path)
    assert (staging / "REASON.txt").read_text(encoding="utf-8") == "RuntimeError"
    _refusal(path)
    cleanup_tmp(path)
    assert list(tl.load(path).members) == ["m"]


def _replayable_trace() -> tl.Trace:
    """A replayable capture that can save an intervention spec."""

    torch.manual_seed(0)
    return tl.trace(
        nn.Linear(2, 2),
        torch.ones(1, 2),
        capture=tl.options.CaptureOptions(replayable=True),
    )


def test_intervention_spec_saver_failed_overwrite_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``save_intervention`` stages beside its target and keeps the folder after a failed overwrite."""

    path = tmp_path / "spec.tlspec"
    trace = _replayable_trace()
    trace.save_intervention(path, level="audit")
    original_spec = (path / "intervention.json").read_bytes()
    original_rename = os.rename

    def fail_install(source: str | Path, target: str | Path) -> None:
        if Path(source).name.startswith(f"{path.name}.tmp.") and Path(target) == path:
            raise OSError("simulated intervention swap refusal")
        original_rename(source, target)

    monkeypatch.setattr("torchlens.intervention._save.os.rename", fail_install)
    with pytest.raises(OSError, match="swap refusal"):
        trace.save_intervention(path, level="audit", overwrite=True)
    monkeypatch.undo()

    assert (path / "intervention.json").read_bytes() == original_spec
    (staging,) = _staging(path)
    assert (staging / "REASON.txt").read_text(encoding="utf-8") == "OSError"
    _refusal(path)
    with pytest.raises(ArtifactError) as info:
        tl.io.load_intervention_spec(path)
    assert info.value.fields.get("code") == CODE
    cleanup_tmp(path)
    tl.io.load_intervention_spec(path)  # refuses typed unless an InterventionSpec loads


def test_recipe_saver_failed_overwrite_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Recipe.save`` stages beside its target and keeps the folder after a failed overwrite."""

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

    (staging,) = _staging(path)
    _refusal(path)
    cleanup_tmp(path)
    assert tl.load(path).digest == older.digest


def test_recipe_saver_first_save_failure_leaves_no_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed first recipe save removes its staging folder as before."""

    path = tmp_path / "recipe.tlspec"
    real_write_tree = _recipe_io._write_recipe_tree

    def failing_write_tree(tmp, recipe, save_level):  # type: ignore[no-untyped-def]
        real_write_tree(tmp, recipe, save_level)
        raise RuntimeError("injected recipe failure")

    monkeypatch.setattr(_recipe_io, "_write_recipe_tree", failing_write_tree)
    with pytest.raises(RuntimeError):
        tl.when(tl.func("relu"), tl.scale(0.5)).save(path)
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_streaming_staging_folder_is_seen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed ``to_disk`` stream's staging folder is the one the load check reads."""

    path = tmp_path / "stream.tlspec"

    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(streaming_mod, "write_trace_envelope", interrupt)
    torch.manual_seed(0)
    with pytest.raises(KeyboardInterrupt):
        tl.trace(
            nn.Sequential(nn.Linear(4, 4), nn.ReLU()),
            torch.randn(2, 4),
            storage=tl.to_disk(path),
            save="all",
        )
    monkeypatch.undo()
    (staging,) = _staging(path)
    assert (staging / "PARTIAL").exists()

    stamp = staging.stat().st_mtime - 10
    os.utime(staging, (stamp, stamp))
    _trace(0).save(path)
    with pytest.warns(TorchLensWarning, match="Leftover staging folder"):
        tl.load(path)
    # A stream that started after the installed copy (another writer) refuses.
    future = path.stat().st_mtime + 10
    os.utime(staging, (future, future))
    error = _refusal(path)
    assert "finalize streaming bundle" in str(error), "the stream's REASON is quoted"


def test_member_loads_are_not_rechecked(tmp_path: Path) -> None:
    """The check runs on the top-level path only; a member path is never re-checked."""

    path = tmp_path / "run.tlspec"
    _trace(0).save(path)
    staging = tmp_path / "run.tlspec.tmp.0123"
    staging.mkdir()
    with pytest.raises(ArtifactError):
        refuse_stale_load(path)
    refuse_stale_load(path, member=True)
