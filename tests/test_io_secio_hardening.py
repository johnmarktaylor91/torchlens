"""FW2-SECIO artifact-I/O hardening batch: perms, glob safety, ceilings, containment.

Each test proves a specific artifact-boundary guarantee and FAILS against the
pre-fix behavior:

* B8-10 -- ``metadata.pkl`` / ``manifest.json`` and the bundle directories are
  written private (0600 / 0700), matching the already-0600 safetensors blobs.
* B8-11 -- ``cleanup_tmp`` escapes glob metacharacters in the bundle basename, so a
  legal ``job*`` name cannot widen the sweep to sibling bundles.
* B8-12 -- the persisted PARTIAL failure reason is the exception TYPE name, not
  ``str(exc)`` (which can carry object reprs), and is length-bounded.
* B8-16 -- ``metadata.pkl`` load enforces a byte ceiling for parity with JSON.
* B8-9 -- a loaded ``visualizer_path`` is contained inside the bundle's own
  ``visualizers/`` directory, so a hostile bundle cannot disclose an arbitrary
  local ``.png`` through ``.draw()``.
* R27-3 -- a recursion blow-up during rehydration surfaces as a typed
  ``TorchLensIOError``, not a raw ``RecursionError`` escaping ``tl.load``.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import bundle as bundle_mod
from torchlens._io.bundle import _mark_partial, _reanchor_visualizer_paths, cleanup_tmp
from torchlens.errors import TorchLensIOError

pytestmark = pytest.mark.smoke


def _tiny() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU())


def _save(tmp_path: Path, name: str = "b.tlspec") -> Path:
    trace = tl.trace(
        _tiny().eval(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )
    spec = tmp_path / name
    tl.save(trace, str(spec))
    return spec


# --------------------------------------------------------------------------- #
# B8-10: private permissions                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits are the checked signal")
def test_bundle_sidecars_and_dirs_are_private(tmp_path: Path) -> None:
    """metadata.pkl/manifest.json are 0600 and the directories 0700.

    Fail-before: the sidecars carrying forward source and harvested attributes
    inherited the umask (0664 under umask 002) while the blobs were 0600.
    """

    prior = os.umask(0o022)
    try:
        spec = _save(tmp_path)
    finally:
        os.umask(prior)
    assert stat.S_IMODE((spec / "metadata.pkl").stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE((spec / "manifest.json").stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE(spec.stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE((spec / "blobs").stat().st_mode) & 0o077 == 0


# --------------------------------------------------------------------------- #
# B8-11: glob escaping in cleanup_tmp                                          #
# --------------------------------------------------------------------------- #


def test_cleanup_tmp_does_not_sweep_siblings_via_glob_metachars(tmp_path: Path) -> None:
    """A ``*`` in one bundle's name must not widen cleanup to a sibling bundle.

    Fail-before: ``f"{name}.tmp.*"`` was globbed, so ``cleanup_tmp(root/"job*")``
    removed ``jobA.tmp.*`` and ``jobB.tmp.*`` alike.
    """

    target_tmp = tmp_path / "job*.tmp.aaaa"
    sibling_tmp = tmp_path / "jobB.tmp.bbbb"
    for directory in (target_tmp, sibling_tmp):
        directory.mkdir()
        (directory / bundle_mod.PARTIAL_SENTINEL).write_text("", encoding="utf-8")

    removed = cleanup_tmp(tmp_path / "job*", force=True)

    assert sibling_tmp.exists(), "cleanup swept a sibling bundle through a glob metachar"
    assert sibling_tmp not in removed
    # The literal ``job*`` target itself is still swept (exact-name match).
    assert not target_tmp.exists()
    assert target_tmp in removed


# --------------------------------------------------------------------------- #
# B8-12: scrubbed, bounded partial reason                                      #
# --------------------------------------------------------------------------- #


def test_mark_partial_reason_is_length_bounded(tmp_path: Path) -> None:
    """A long reason is truncated so recovery debris cannot grow unbounded."""

    tmp_dir = tmp_path / "x.tmp.zzzz"
    tmp_dir.mkdir()
    _mark_partial(tmp_dir, reason="A" * 100_000)
    persisted = (tmp_dir / bundle_mod.REASON_SENTINEL).read_text()
    assert len(persisted) <= bundle_mod._MAX_PARTIAL_REASON_CHARS


def test_save_failure_persists_type_name_not_message(tmp_path: Path, monkeypatch) -> None:
    """A failed save records the exception TYPE, not a repr-bearing message.

    Fail-before: ``_mark_partial(..., reason=str(exc))`` embedded the full message,
    which for many exceptions carries object reprs / paths / values.
    """

    secret = "SENSITIVE-VALUE-should-not-persist"

    def _boom(*_args, **_kwargs):
        raise ValueError(secret)

    monkeypatch.setattr(bundle_mod, "_scrub_trace_for_bundle", _boom)
    trace = tl.trace(
        _tiny().eval(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )
    spec = tmp_path / "fail.tlspec"
    with pytest.raises(TorchLensIOError):
        tl.save(trace, str(spec))
    reasons = list(tmp_path.glob("fail.tlspec.tmp.*/" + bundle_mod.REASON_SENTINEL))
    assert reasons, "no PARTIAL reason sentinel was written"
    for reason_file in reasons:
        text = reason_file.read_text()
        assert secret not in text, "exception message leaked into recovery debris"
        assert text == "ValueError"


# --------------------------------------------------------------------------- #
# B8-16: metadata.pkl byte ceiling                                            #
# --------------------------------------------------------------------------- #


def test_metadata_pkl_byte_ceiling_refuses_oversize(tmp_path: Path, monkeypatch) -> None:
    """An implausibly large metadata.pkl is refused typed before it is unpickled."""

    spec = _save(tmp_path)
    monkeypatch.setattr(bundle_mod, "_MAX_METADATA_PKL_BYTES", 16)
    with pytest.raises(TorchLensIOError, match="ceiling"):
        tl.load(str(spec))


# --------------------------------------------------------------------------- #
# B8-9: visualizer_path containment on load                                    #
# --------------------------------------------------------------------------- #


def test_visualizer_path_outside_bundle_is_dropped(tmp_path: Path) -> None:
    """A hostile absolute ``visualizer_path`` is not honored on load.

    Fail-before: the field flowed verbatim into Graphviz ``image=`` with only a
    ``.png`` suffix check, disclosing any local ``.png`` via ``load(evil).draw()``.
    """

    victim_png = tmp_path / "victim_secret.png"
    victim_png.write_bytes(b"\x89PNG\r\n\x1a\n")
    spec = _save(tmp_path)
    trace = tl.load(str(spec))
    # Simulate a tampered field pointing at an out-of-bundle file.
    trace.layer_list[0].visualizer_path = str(victim_png)
    _reanchor_visualizer_paths(trace, spec)
    assert trace.layer_list[0].visualizer_path is None


def test_visualizer_path_inside_bundle_is_reanchored(tmp_path: Path) -> None:
    """A real thumbnail inside the bundle's visualizers/ dir is kept and re-anchored."""

    spec = _save(tmp_path)
    (spec / "visualizers").mkdir(exist_ok=True)
    thumb = spec / "visualizers" / "00000_layer.png"
    thumb.write_bytes(b"\x89PNG\r\n\x1a\n")
    trace = tl.load(str(spec))
    trace.layer_list[0].visualizer_path = "/somewhere/else/00000_layer.png"
    _reanchor_visualizer_paths(trace, spec)
    kept = trace.layer_list[0].visualizer_path
    assert kept is not None
    assert Path(kept).resolve() == thumb.resolve()


# --------------------------------------------------------------------------- #
# R21-1: persisted equivalence-class keys are canonically ordered              #
# --------------------------------------------------------------------------- #


class _NestedParamModel(nn.Module):
    """Nested module so op-level equivalence keys carry a module suffix (R21-1)."""

    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)


def test_persisted_equivalence_class_keys_are_canonically_ordered(tmp_path: Path) -> None:
    """Trace-level op_equivalence_classes keys sort their param_NNNNNN run.

    Fail-before (R21-1): weight-vs-bias order in a key was a per-capture coin flip
    (random raw barcodes). The op-level field was re-sorted at save but the
    trace-level dict keys of a NESTED module (whose op-level string carries a module
    suffix, so it misses the op-level remap) fell through to an order-preserving
    remap and stayed random -- breaking byte-reproducibility across processes. Many
    captures make the flip near-certain to appear if the canonicalization is absent.
    """

    import pickle
    import re

    param_token = re.compile(r"param_\d{6}")
    checked = 0
    for index in range(10):
        trace = tl.trace(
            _NestedParamModel().eval(),
            torch.randn(2, 4),
            capture=tl.options.CaptureOptions(layers_to_save="all"),
        )
        spec = tmp_path / f"nested_{index}.tlspec"
        tl.save(trace, str(spec))
        groups = pickle.loads((spec / "metadata.pkl").read_bytes()).get("op_equivalence_classes")
        for key in groups or {}:
            tokens = param_token.findall(key)
            if len(tokens) >= 2:
                checked += 1
                assert tokens == sorted(tokens), f"non-canonical equivalence key persisted: {key}"
    assert checked > 0, "no multi-param equivalence keys were exercised"


class _NestedMultiOutputParamModel(nn.Module):
    """Nested LSTM so multi-output param-group keys carry `_outindex{N}` (R21)."""

    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.LSTM(4, 4, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return out


def test_persisted_equivalence_keys_preserve_outindex_suffix(tmp_path: Path) -> None:
    """Multi-output param-op equivalence groups survive save with distinct keys.

    Fail-before (R21 fix-introduced regression, 6ff95fbc): canonical ordering of
    the `param_NNNNNN` run rebuilt the key as prefix + sorted tokens, DISCARDING
    the trailing `_outindex{N}` that disambiguates the outputs of a multi-output
    parameterized op (e.g. LSTM's output/h/c). All N keys collided into one and
    the dict rebuild silently kept only the last group.
    """

    import pickle
    import re

    trace = tl.trace(
        _NestedMultiOutputParamModel().eval(),
        torch.randn(2, 3, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )
    live_groups = dict(trace.op_equivalence_classes)
    live_outindex_keys = {key for key in live_groups if "_outindex" in key}
    assert len(live_outindex_keys) >= 2, "expected multi-output param groups in-memory"

    spec = tmp_path / "lstm_outindex.tlspec"
    tl.save(trace, str(spec))
    persisted = pickle.loads((spec / "metadata.pkl").read_bytes()).get("op_equivalence_classes")
    assert persisted is not None

    persisted_outindex_keys = {key for key in persisted if "_outindex" in key}
    suffix = re.compile(r"_outindex\d+$")
    live_suffixes = sorted(suffix.search(key).group(0) for key in live_outindex_keys)
    persisted_suffixes = sorted(
        match.group(0) for match in (suffix.search(key) for key in persisted_outindex_keys) if match
    )
    assert persisted_suffixes == live_suffixes, (
        "outindex-suffixed equivalence groups were collapsed at save: "
        f"live={live_suffixes} persisted={persisted_suffixes}"
    )
    assert len(persisted) == len(live_groups), (
        f"equivalence groups dropped at save: live={len(live_groups)} persisted={len(persisted)}"
    )
    param_token = re.compile(r"param_\d{6}")
    for key in persisted:
        tokens = param_token.findall(key)
        if len(tokens) >= 2:
            assert tokens == sorted(tokens), f"non-canonical equivalence key persisted: {key}"


# --------------------------------------------------------------------------- #
# B8-19: git-commit provenance follows include_source                          #
# --------------------------------------------------------------------------- #


def _manifest_git_hash(spec: Path) -> object:
    import json

    manifest = json.loads((spec / "manifest.json").read_text())
    return manifest.get("provenance", {}).get("git_commit_hash")


def test_git_commit_hash_is_dropped_when_source_excluded(tmp_path: Path) -> None:
    """include_source=False must not embed the cwd repo's HEAD commit.

    Fail-before: the manifest embedded the git commit of whatever repository
    contained the working directory at save time, with no opt-out or disclosure.
    """

    trace = tl.trace(
        _tiny().eval(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )
    excluded = tmp_path / "no_source.tlspec"
    tl.save(trace, str(excluded), include_source=False)
    assert _manifest_git_hash(excluded) is None

    included = tmp_path / "with_source.tlspec"
    tl.save(trace, str(included), include_source=True)
    # R21-2: when source is kept the field reflects TORCHLENS's own commit (or None
    # for a released wheel), never the working directory's unrelated repository.
    from torchlens._io.bundle import _git_commit_hash, _torchlens_package_dir

    expected = _git_commit_hash(_torchlens_package_dir())
    assert _manifest_git_hash(included) == expected


def test_git_commit_hash_is_independent_of_working_directory(tmp_path: Path, monkeypatch) -> None:
    """R21-2: the same capture saved from two directories records the same hash.

    Fail-before: the hash came from Path.cwd(), so saving from inside a different repo
    embedded that repo's HEAD -- an ambient-environment dependence that also leaked the
    user's unrelated repository commit.
    """

    trace = tl.trace(
        _tiny().eval(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )
    first = tmp_path / "a.tlspec"
    tl.save(trace, str(first), include_source=True)

    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    second = tmp_path / "b.tlspec"
    tl.save(trace, str(second), include_source=True)
    assert _manifest_git_hash(first) == _manifest_git_hash(second)


# --------------------------------------------------------------------------- #
# Atomic-save double-fault disclosure                                          #
# --------------------------------------------------------------------------- #


def test_double_fault_restore_discloses_stranded_backup(tmp_path: Path) -> None:
    """When a save fails AND the restore fails, the backup path is disclosed.

    Fail-before: the restore's ``except OSError`` silently passed, so the prior
    artifact was stranded under a hidden ``.bak.<uuid>`` name the error never named.
    """

    backup = tmp_path / "b.tlspec.bak.deadbeef"
    backup.mkdir()
    (backup / "marker").write_text("prior", encoding="utf-8")
    target = tmp_path / "b.tlspec"  # does not exist -> restore path is taken

    def _failing_rename(*_a, **_k):
        raise OSError("cross-device restore failed")

    import torchlens._io.bundle as b

    original = Path.rename
    try:
        Path.rename = _failing_rename  # type: ignore[method-assign]
        with pytest.warns(UserWarning, match=str(backup)):
            restored = b._restore_backup(backup, target)
    finally:
        Path.rename = original  # type: ignore[method-assign]
    assert restored is False
    assert backup.exists(), "the backup must remain recoverable"


def test_single_fault_restore_is_silent_and_succeeds(tmp_path: Path) -> None:
    """The normal single-fault path restores the backup with no warning."""

    import warnings as _warnings

    import torchlens._io.bundle as b

    backup = tmp_path / "b.tlspec.bak.cafe"
    backup.mkdir()
    target = tmp_path / "b.tlspec"
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        assert b._restore_backup(backup, target) is True
    assert target.exists()
    assert not backup.exists()


# --------------------------------------------------------------------------- #
# R27-3: typed recursion refusal during rehydration                            #
# --------------------------------------------------------------------------- #


def test_rehydrate_recursion_surfaces_typed(tmp_path: Path, monkeypatch) -> None:
    """A recursion blow-up during rehydration is a typed error, not a raw crash."""

    spec = _save(tmp_path)

    def _blow_up(*_args, **_kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(bundle_mod, "rehydrate_trace", _blow_up)
    with pytest.raises(TorchLensIOError, match="recursion"):
        tl.load(str(spec))
