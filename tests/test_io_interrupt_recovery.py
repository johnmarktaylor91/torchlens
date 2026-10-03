"""R38+R59+R63: staged-save debris is discoverable and recovery never masks interrupts.

Two converged findings, one interrupt lane:

- The bundle ``_TlSpecWriter.write_bundle`` named its staging/backup dirs
  target-independently (``tmp.<hex>`` / ``tmp.bak.<hex>``), so
  ``cleanup_tmp(target)`` -- which globs ``{name}.tmp.*`` / ``{name}.bak.*`` --
  could never find them. A SIGKILL between the two ``os.replace`` calls
  stranded the ONLY pre-overwrite copy in an undiscoverable directory, and the
  staging dir carried no PARTIAL sentinel (opus R38 + sol R59, merged).
- The bundle save's failure handlers ran their PARTIAL-mark / backup-restore
  bookkeeping unguarded, so a rollback-time failure masked the primary
  ``KeyboardInterrupt`` (reported as an ordinary I/O error) and skipped the
  backup restore (opus R63, measured).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import tlspec as tlspec_mod
from torchlens.io import cleanup_tmp


def _trace(seed: int = 0) -> tl.Trace:
    torch.manual_seed(seed)
    return tl.trace(nn.Linear(4, 4), torch.randn(2, 4), save=tl.func("linear"))


def test_tlspec_staging_dir_is_target_prefixed_and_partial_marked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-write strand leaves a ``{target}.tmp.*`` dir with a PARTIAL sentinel.

    SIGKILL simulation: the manifest write raises and the handler's rmtree is
    disabled, so the staging dir survives exactly as a hard kill would leave
    it. ``cleanup_tmp(target)`` must then sweep it WITHOUT ``force=True``.
    """

    target = tmp_path / "bundle.tlspec"
    real_write_json = tlspec_mod._TlSpecWriter.write_json

    def failing_write_json(path, payload):  # type: ignore[no-untyped-def]
        if Path(path).name == "manifest.json":
            raise RuntimeError("injected mid-write failure")
        return real_write_json(path, payload)

    monkeypatch.setattr(tlspec_mod._TlSpecWriter, "write_json", staticmethod(failing_write_json))
    monkeypatch.setattr(tlspec_mod.shutil, "rmtree", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="injected mid-write failure"):
        tl.Bundle({"m": _trace(1)}).save(target)

    stranded = sorted(tmp_path.glob(f"{target.name}.tmp.*"))
    assert stranded, "staging dir must carry the target-prefixed name cleanup_tmp globs for"
    assert (stranded[0] / "PARTIAL").exists(), "staging dir must be PARTIAL-marked from birth"

    monkeypatch.undo()
    removed = cleanup_tmp(target)
    assert stranded[0] in removed
    assert not stranded[0].exists()


def test_tlspec_stranded_backup_is_restored_by_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kill between the two publish renames leaves a recoverable backup.

    Fail-before: the backup was named ``tmp.bak.<hex>`` -- invisible to
    ``cleanup_tmp(target)`` -- so the ONLY pre-overwrite copy was permanently
    stranded.
    """

    target = tmp_path / "bundle.tlspec"
    tl.Bundle({"m": _trace(1)}).save(target)
    marker_bytes = (target / "manifest.json").read_bytes()

    real_replace = tlspec_mod.os.replace
    calls = {"n": 0}

    def killing_replace(src, dst):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            return real_replace(src, dst)  # target renamed aside to the backup
        raise KeyboardInterrupt  # simulated kill before/at the publish rename

    monkeypatch.setattr(tlspec_mod.os, "replace", killing_replace)
    # Disable the in-process handler's cleanup/restore too: a real SIGKILL
    # runs neither, and the on-disk debris must be recoverable anyway.
    monkeypatch.setattr(tlspec_mod.shutil, "rmtree", lambda *a, **k: None)

    with pytest.raises(KeyboardInterrupt):
        tl.Bundle({"m": _trace(2)}).save(target, overwrite=True)

    monkeypatch.undo()
    assert not target.exists(), "the kill window leaves no published target"
    backups = sorted(tmp_path.glob(f"{target.name}.bak.*"))
    assert backups, "the pre-overwrite copy must sit under a discoverable target-prefixed name"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        restored = cleanup_tmp(target)
    assert target in restored
    assert (target / "manifest.json").read_bytes() == marker_bytes


def test_bundle_save_recovery_failure_never_masks_the_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R63 F1: a rollback-time failure is disclosed, never raised over Ctrl-C.

    Fail-before: ``_mark_partial`` raising (e.g. ENOSPC) replaced the primary
    ``KeyboardInterrupt`` with its own ``OSError`` and skipped the backup
    restore entirely.
    """

    import torchlens._io.bundle as bundle_mod

    trace = _trace(3)
    target = tmp_path / "b.tl"
    tl.save(trace, target)
    before = sorted(p.name for p in target.iterdir())

    def interrupting_dump(state, handle):  # type: ignore[no-untyped-def]
        raise KeyboardInterrupt

    def failing_mark_partial(tmp, *, reason=None):  # type: ignore[no-untyped-def]
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(bundle_mod, "dump_canonical_metadata", interrupting_dump)
    monkeypatch.setattr(bundle_mod, "_mark_partial", failing_mark_partial)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(KeyboardInterrupt):
            tl.save(trace, target, overwrite=True)

    assert any("recovery step 'mark-partial' itself failed" in str(w.message) for w in caught)
    # The backup restore still ran: the pre-existing bundle is back in place.
    assert sorted(p.name for p in target.iterdir()) == before
