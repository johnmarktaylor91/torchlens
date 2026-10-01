"""r58 A3a: ``trace(..., cache=True)`` must never unpickle unauthenticated bytes.

The capture cache stored a whole pickled ``Trace`` and read it back with a BARE
``pickle.load`` from a directory named by ``TORCHLENS_CACHE_DIR``. Point that variable
at any path a second principal can write -- a CI cache mount, a container volume, a
group-writable NFS home, a world-writable tmpdir -- and the next cache HIT was
arbitrary code execution inside the user's process, with no error, no warning, and a
returned Trace that looked normal (``capture_cache_hit=True``).

The cache cannot route through ``_io._safe_unpickle.SafeBundleUnpickler``: that
allowlist deliberately refuses tensor/storage CONSTRUCTION and every non-allowlisted
torchlens type, so a whole ``Trace`` fails it, and widening the allowlist to fit one
would disarm the ``.tlspec`` front door. The boundary is closed on AUTHENTICITY
instead: a private-directory precondition plus a per-entry HMAC-SHA256 tag keyed by a
0600 secret inside that directory. Bytes that do not authenticate are never handed to
``pickle``.
"""

from __future__ import annotations

import os
import pickle
import stat
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors import TorchLensIOError

_PWN_MARKER_NAME = "r58_capture_cache_pwned"


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU())


class _CodeExecPayload:
    """A ``__reduce__`` gadget: unpickling it writes a marker file."""

    def __init__(self, marker: Path) -> None:
        self._marker = marker

    def __reduce__(self) -> tuple:  # noqa: D105 - pickle protocol hook
        return (_write_marker, (str(self._marker),))


def _write_marker(path: str) -> str:
    """Stand-in for ``os.system``: proves the gadget ran without side effects."""

    Path(path).write_text("pwned", encoding="utf-8")
    return path


def _cache_entry(cache_root: Path) -> Path:
    """Return the single ``.pkl`` entry the capture cache just wrote."""

    entries = sorted(cache_root.glob("*.pkl"))
    assert len(entries) == 1, f"expected exactly one cache entry, found {entries}"
    return entries[0]


@pytest.fixture()
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the capture cache at a private per-test directory."""

    monkeypatch.setenv("TORCHLENS_CACHE_DIR", str(tmp_path / "tlcache"))
    return tmp_path / "tlcache" / "capture"


# --------------------------------------------------------------------------- #
# The cache still works                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_authenticated_cache_still_hits(cache_root: Path) -> None:
    """An entry this process wrote authenticates and is reused.

    Rebaselined for the atomic single-record format (grind-p3 T5.2): the tag
    is embedded in the entry's fixed header instead of a ``.hmac`` sidecar, so
    the assertion pins the self-authenticating record rather than the pair.
    """

    from torchlens.user_funcs import _CAPTURE_CACHE_MAGIC

    model, inputs = _tiny_model(), torch.rand(2, 4)
    first = tl.trace(
        model, inputs, capture=tl.options.CaptureOptions(layers_to_save="all", cache=True)
    )
    assert first.capture_cache_hit is False
    second = tl.trace(
        model, inputs, capture=tl.options.CaptureOptions(layers_to_save="all", cache=True)
    )
    assert second.capture_cache_hit is True
    entry = _cache_entry(cache_root)
    assert entry.read_bytes().startswith(_CAPTURE_CACHE_MAGIC)
    assert not entry.with_name(entry.name + ".hmac").exists()


@pytest.mark.smoke
def test_cache_secret_is_private(cache_root: Path) -> None:
    """The HMAC secret is created 0600, so its tags actually prove something."""

    tl.trace(
        _tiny_model(),
        torch.rand(2, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all", cache=True),
    )
    secret = cache_root / ".capture_cache_secret"
    assert secret.is_file()
    assert stat.S_IMODE(secret.stat().st_mode) == 0o600


# --------------------------------------------------------------------------- #
# The RCE itself                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_planted_code_exec_pickle_is_never_unpickled(cache_root: Path, tmp_path: Path) -> None:
    """A substituted entry does NOT execute and does NOT become a cache hit.

    Fail-before: the gadget ran at ``pickle.load`` time and its return value was
    handed back to the caller as a Trace.
    """

    model, inputs = _tiny_model(), torch.rand(2, 4)
    tl.trace(model, inputs, capture=tl.options.CaptureOptions(layers_to_save="all", cache=True))
    marker = tmp_path / _PWN_MARKER_NAME
    _cache_entry(cache_root).write_bytes(pickle.dumps(_CodeExecPayload(marker)))

    with pytest.warns(UserWarning, match="Ignoring TorchLens capture cache entry"):
        refreshed = tl.trace(
            model, inputs, capture=tl.options.CaptureOptions(layers_to_save="all", cache=True)
        )

    assert not marker.exists(), "the planted __reduce__ gadget executed"
    assert refreshed.capture_cache_hit is False
    assert isinstance(refreshed, tl.Trace)


@pytest.mark.smoke
def test_headerless_entry_is_a_miss_not_a_load(cache_root: Path, tmp_path: Path) -> None:
    """An entry without the authenticated header is refilled, not trusted.

    Rebaselined for the atomic single-record format (grind-p3 T5.2): the
    pre-upgrade-pair / bare-plant case is now a MISSING EMBEDDED HEADER
    instead of a missing ``.hmac`` sidecar. The guarantee is unchanged: bytes
    without an authenticating tag are never handed to ``pickle``.
    """

    model, inputs = _tiny_model(), torch.rand(2, 4)
    tl.trace(model, inputs, capture=tl.options.CaptureOptions(layers_to_save="all", cache=True))
    entry = _cache_entry(cache_root)
    marker = tmp_path / _PWN_MARKER_NAME
    entry.write_bytes(pickle.dumps(_CodeExecPayload(marker)))

    with pytest.warns(UserWarning, match="not a single-record authenticated"):
        refreshed = tl.trace(
            model, inputs, capture=tl.options.CaptureOptions(layers_to_save="all", cache=True)
        )

    assert not marker.exists()
    assert refreshed.capture_cache_hit is False
    # The rewrite re-commits an authenticated record, so the next run hits.
    assert (
        tl.trace(
            model, inputs, capture=tl.options.CaptureOptions(layers_to_save="all", cache=True)
        ).capture_cache_hit
        is True
    )


@pytest.mark.smoke
def test_tag_from_a_foreign_secret_does_not_authenticate(cache_root: Path, tmp_path: Path) -> None:
    """An attacker who can write the entry AND a tag still cannot forge one."""

    import hashlib
    import hmac

    from torchlens.user_funcs import _CAPTURE_CACHE_MAGIC

    model, inputs = _tiny_model(), torch.rand(2, 4)
    tl.trace(model, inputs, capture=tl.options.CaptureOptions(layers_to_save="all", cache=True))
    entry = _cache_entry(cache_root)
    marker = tmp_path / _PWN_MARKER_NAME
    payload = pickle.dumps(_CodeExecPayload(marker))
    # Rebaselined for the atomic single-record format (grind-p3 T5.2): the
    # attacker forges a well-formed record whose embedded tag is keyed by a
    # guessed secret. It must still refuse to authenticate.
    forged_tag = hmac.new(b"attacker-guessed-secret", payload, hashlib.sha256).hexdigest()
    entry.write_bytes(_CAPTURE_CACHE_MAGIC + forged_tag.encode("ascii") + b"\n" + payload)

    with pytest.warns(UserWarning, match="does not match its bytes"):
        tl.trace(model, inputs, capture=tl.options.CaptureOptions(layers_to_save="all", cache=True))

    assert not marker.exists()


@pytest.mark.smoke
def test_symlinked_entry_is_never_followed(cache_root: Path, tmp_path: Path) -> None:
    """A symlinked entry redirecting out of the cache is refused, not read."""

    model, inputs = _tiny_model(), torch.rand(2, 4)
    tl.trace(model, inputs, capture=tl.options.CaptureOptions(layers_to_save="all", cache=True))
    entry = _cache_entry(cache_root)
    marker = tmp_path / _PWN_MARKER_NAME
    elsewhere = tmp_path / "elsewhere.pkl"
    elsewhere.write_bytes(pickle.dumps(_CodeExecPayload(marker)))
    entry.unlink()
    entry.symlink_to(elsewhere)

    with pytest.warns(UserWarning, match="symlink"):
        tl.trace(model, inputs, capture=tl.options.CaptureOptions(layers_to_save="all", cache=True))

    assert not marker.exists()


# --------------------------------------------------------------------------- #
# The directory precondition                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits are the checked signal")
def test_cache_dirs_are_created_private() -> None:
    """Both torchlens-owned cache levels end up with no group/other write bits.

    ``mkdir(parents=True, mode=...)`` applies the mode to the LEAF only, so the
    configured directory itself kept the ambient umask permissions (0775 under the very
    common umask 002) and a second principal could replace the whole subtree.
    """

    cache_dir = Path(os.environ["TORCHLENS_CACHE_DIR"])
    tl.trace(
        _tiny_model(),
        torch.rand(2, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all", cache=True),
    )
    for directory in (cache_dir, cache_dir / "capture"):
        assert stat.S_IMODE(directory.stat().st_mode) & 0o022 == 0, directory


@pytest.mark.smoke
@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits are the checked signal")
def test_world_writable_cache_dir_is_tightened_not_trusted(cache_root: Path) -> None:
    """A world-writable cache directory is made private before any entry is read.

    Tightening rather than refusing is deliberate: the default cache path is created
    under the caller's umask, so a hard refusal would break the DEFAULT cache on any
    umask-002 install. The HMAC tag is the load-bearing guard.
    """

    model, inputs = _tiny_model(), torch.rand(2, 4)
    tl.trace(model, inputs, capture=tl.options.CaptureOptions(layers_to_save="all", cache=True))
    cache_root.chmod(0o777)
    tl.trace(model, inputs, capture=tl.options.CaptureOptions(layers_to_save="all", cache=True))
    assert stat.S_IMODE(cache_root.stat().st_mode) & 0o022 == 0


@pytest.mark.smoke
@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits are the checked signal")
def test_group_readable_secret_refuses_typed(cache_root: Path) -> None:
    """A secret other users can read cannot key a meaningful tag, so it refuses."""

    model, inputs = _tiny_model(), torch.rand(2, 4)
    tl.trace(model, inputs, capture=tl.options.CaptureOptions(layers_to_save="all", cache=True))
    secret = cache_root / ".capture_cache_secret"
    secret.chmod(0o644)
    try:
        with pytest.raises(TorchLensIOError, match="readable or writable"):
            tl.trace(
                model,
                torch.rand(2, 4),
                capture=tl.options.CaptureOptions(layers_to_save="all", cache=True),
            )
    finally:
        secret.chmod(0o600)


# --------------------------------------------------------------------------- #
# The double-read TOCTOU: authenticate one read, unpickle another               #
# --------------------------------------------------------------------------- #


class _OpenCountingSwapHook:
    """Audit hook that swaps the cache entry BETWEEN the verify and load reads.

    The vulnerable reader computed the HMAC by streaming one ``open`` of the
    entry, then performed a SECOND, independent ``open`` for ``pickle.load``.
    Those two reads resolved the same path at different times, so an attacker who
    can write the cache directory could substitute the payload after the tag
    verified over the benign bytes. This hook makes that race deterministic: it
    counts ``open`` events for the exact entry path and, on the SECOND one,
    overwrites the file with a code-execution gadget before the read proceeds.

    A correct single-read implementation opens the entry exactly ONCE (it
    authenticates and unpickles the same in-memory bytes), so the swap is never
    reachable and ``opens`` never climbs past 1.
    """

    def __init__(self, entry_path: Path, evil_bytes: bytes) -> None:
        self._entry_str = str(entry_path)
        self._entry_path = entry_path
        self._evil_bytes = evil_bytes
        self.opens = 0
        self.swapped = False
        self._swapping = False

    def __call__(self, event: str, args: tuple) -> None:  # noqa: D401 - audit hook
        if event != "open" or self._swapping:
            return
        raw_path = args[0]
        if raw_path is None:
            return
        try:
            candidate = os.fspath(raw_path)
        except TypeError:
            return
        if candidate != self._entry_str:
            return
        self.opens += 1
        if self.opens == 2 and not self.swapped:
            # Second open == the pickle.load read in the vulnerable path. Swap
            # the bytes now, before the read completes. Guard against the writes
            # we do here re-entering the hook.
            self._swapping = True
            try:
                with open(self._entry_path, "wb") as handle:
                    handle.write(self._evil_bytes)
                self.swapped = True
            finally:
                self._swapping = False


@pytest.mark.smoke
def test_authenticated_bytes_are_the_bytes_unpickled(cache_root: Path, tmp_path: Path) -> None:
    """The entry is read ONCE: what the tag authenticates is what gets unpickled.

    Fail-before (double-read TOCTOU): the HMAC verified the benign bytes on the
    first read, then a second read of the same path fed a substituted
    code-execution gadget straight into ``pickle.load``. The tag file was never
    touched, so the entry authenticated and the gadget still ran.
    """

    import sys

    model, inputs = _tiny_model(), torch.rand(2, 4)
    tl.trace(model, inputs, capture=tl.options.CaptureOptions(layers_to_save="all", cache=True))
    entry = _cache_entry(cache_root)
    marker = tmp_path / _PWN_MARKER_NAME
    evil = pickle.dumps(_CodeExecPayload(marker))

    hook = _OpenCountingSwapHook(entry, evil)
    sys.addaudithook(hook)

    refreshed = tl.trace(
        model, inputs, capture=tl.options.CaptureOptions(layers_to_save="all", cache=True)
    )

    assert not marker.exists(), (
        "the entry was authenticated on one read and unpickled from another: the "
        "swapped gadget executed"
    )
    assert hook.opens == 1, (
        "the cache entry was opened more than once during a single load; a "
        "correct reader authenticates and unpickles the SAME single read "
        f"(observed {hook.opens} opens)"
    )
    assert not hook.swapped, "the between-reads swap was reachable"
    # The benign bytes authenticated and loaded, so this is an ordinary hit.
    assert refreshed.capture_cache_hit is True
    assert isinstance(refreshed, tl.Trace)


@pytest.mark.smoke
def test_no_bare_pickle_read_in_the_package() -> None:
    """Standing gate: no bare ``pickle.load``/``loads`` outside the guarded readers.

    ``user_funcs`` held the ONLY raw ``pickle.load`` in the package while
    ``_safe_unpickle`` was imported in exactly one place. This scan keeps it that way,
    so a new artifact reader cannot reintroduce the class.
    """

    allowed = {
        # The guarded readers themselves.
        "_io/_safe_unpickle.py",
        # The authenticated capture-cache reader: verifies an HMAC tag over the exact
        # bytes BEFORE unpickling, and refuses a non-private cache directory. See this
        # module's docstring for why the SafeBundleUnpickler allowlist cannot apply.
        "user_funcs.py",
    }
    package_root = Path(tl.__file__).resolve().parent
    offenders: list[str] = []
    for source_path in sorted(package_root.rglob("*.py")):
        relative = source_path.relative_to(package_root).as_posix()
        if relative in allowed:
            continue
        for lineno, line in enumerate(source_path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "pickle.loads(pickle.dumps(" in line:
                # Same-expression in-process round-trip (Trace.__deepcopy__):
                # the loaded bytes are produced by the adjacent dumps of a live
                # object, so no external/attacker bytes can ever reach loads.
                continue
            if "pickle.load(" in line or "pickle.loads(" in line:
                offenders.append(f"{relative}:{lineno}: {stripped}")
    assert not offenders, (
        "bare pickle read outside the guarded readers (route through "
        f"_io._safe_unpickle, or authenticate the bytes first): {offenders}"
    )
