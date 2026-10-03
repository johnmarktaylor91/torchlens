"""grind-p3 T5.2: the capture-cache entry commits ATOMICALLY (payload + tag as one).

The pair format (``<key>.pkl`` payload committed by ``os.replace``, then a
SEPARATE ``.hmac`` sidecar written after it) had a torn-generation window: a
crash -- or a concurrent reader -- between the two writes observed a payload
against the wrong generation's tag. The mismatch demoted a legitimately
written entry to a permanent warn-and-miss until the next rewrite. The entry
is now ONE self-authenticating record (magic + embedded HMAC header + pickled
payload) committed by a single ``os.replace``, so a torn payload/tag state is
structurally unrepresentable.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.user_funcs as user_funcs


class _CacheModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


def _cache_capture(tmp_path):
    return tl.options.CaptureOptions(cache=True, cache_dir=tmp_path / "cache")


def _single_entry(tmp_path) -> Path:
    entries = sorted((tmp_path / "cache" / "capture").glob("*.pkl"))
    assert len(entries) == 1
    return entries[0]


def test_crash_at_the_legacy_tag_seam_never_leaves_a_torn_entry(tmp_path, monkeypatch) -> None:
    """A crash after the payload commit cannot strand a payload/tag mismatch.

    Pre-fix red: the simulated crash between ``os.replace`` (new payload) and
    the sidecar tag write left generation-2 bytes under generation-1's tag, so
    the next capture warned "Ignoring TorchLens capture cache entry ..." and
    missed. Post-fix the tag is embedded in the single committed record: the
    interrupted rewrite either fully commits or leaves the old generation
    intact, and the next capture is a clean authenticated hit either way.
    """

    model = _CacheModel()
    x = torch.randn(1, 4)
    first = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert first.capture_cache_hit is False
    entry = _single_entry(tmp_path)
    secret = user_funcs._capture_cache_secret(entry.parent)

    # A distinct trace object for the SAME key, guaranteed to pickle to
    # different bytes than generation 1.
    fresh = tl.trace(model, x)
    fresh.annotations["torn_window_nonce"] = "generation-two-bytes"

    def crash_before_tag_commit(*args, **kwargs):  # noqa: ANN002, ANN003
        raise OSError("simulated crash before the sidecar tag commit")

    # ``raising=False``: the fixed store has no sidecar-tag seam at all, so the
    # injection is vacuous there; against the historical pair-format code it
    # fires between the payload swap and the tag write (the red state).
    monkeypatch.setattr(user_funcs, "atomic_write_text", crash_before_tag_commit, raising=False)
    try:
        user_funcs._store_authenticated_capture_cache(fresh, entry, secret)
    except OSError:
        pass
    monkeypatch.undo()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        after = tl.trace(model, x, capture=_cache_capture(tmp_path))
    torn_warnings = [
        str(item.message)
        for item in caught
        if "Ignoring TorchLens capture cache entry" in str(item.message)
    ]
    assert torn_warnings == [], torn_warnings
    assert after.capture_cache_hit is True


def test_crash_at_the_single_commit_preserves_the_old_generation(tmp_path, monkeypatch) -> None:
    """A crash at the atomic commit itself leaves the prior entry fully intact."""

    model = _CacheModel()
    x = torch.randn(1, 4)
    tl.trace(model, x, capture=_cache_capture(tmp_path))
    entry = _single_entry(tmp_path)
    secret = user_funcs._capture_cache_secret(entry.parent)
    before = entry.read_bytes()

    fresh = tl.trace(model, x)

    import os as os_module

    real_replace = os_module.replace

    def crash_at_commit(src, dst, *args, **kwargs):  # noqa: ANN002, ANN003
        if Path(str(dst)) == entry:
            raise OSError("simulated crash at the atomic commit")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(user_funcs.os, "replace", crash_at_commit)
    # b6 R25: a disk failure degrades to warn-and-miss instead of raising out
    # of a capture that already succeeded; the commit must still not tear.
    with warnings.catch_warnings(record=True) as store_warnings:
        warnings.simplefilter("always")
        stored = user_funcs._store_authenticated_capture_cache(fresh, entry, secret)
    monkeypatch.undo()
    assert stored is False
    assert any("simulated crash at the atomic commit" in str(w.message) for w in store_warnings)

    assert entry.read_bytes() == before
    after = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert after.capture_cache_hit is True


@pytest.mark.smoke
def test_committed_entry_is_self_authenticating(tmp_path) -> None:
    """The committed record needs no sidecar: no ``.hmac`` file exists at all."""

    model = _CacheModel()
    x = torch.randn(1, 4)
    tl.trace(model, x, capture=_cache_capture(tmp_path))
    cache_root = tmp_path / "cache" / "capture"
    assert sorted(cache_root.glob("*.pkl")), "entry was written"
    assert not list(cache_root.glob("*.hmac")), (
        "the authenticated entry must be one atomic record, not a payload/tag pair"
    )
    second = tl.trace(model, x, capture=_cache_capture(tmp_path))
    assert second.capture_cache_hit is True


@pytest.mark.smoke
def test_hard_crash_temp_debris_is_swept_age_gated(tmp_path) -> None:
    """Orphaned mkstemp staging files are swept once stale (r3 T-CACHES).

    A hard crash (SIGKILL) between ``mkstemp`` and the atomic ``os.replace``
    strands ``.<name>.tmp.<rand>`` files that no ``*.pkl`` glob sees: they
    were invisible to eviction accounting AND ``clear_capture_cache``,
    accumulating without bound. Fresh temps (a possibly in-flight store)
    must survive the sweep.
    """

    import os
    import time as time_module

    model = _CacheModel()
    x = torch.randn(1, 4)
    tl.trace(model, x, capture=_cache_capture(tmp_path))
    cache_root = tmp_path / "cache" / "capture"
    entry = _single_entry(tmp_path)

    stale = cache_root / f".{entry.name}.tmp.stale123"
    fresh = cache_root / f".{entry.name}.tmp.fresh456"
    stale.write_bytes(b"debris")
    fresh.write_bytes(b"in-flight")
    two_hours_ago = time_module.time() - 7200
    os.utime(stale, (two_hours_ago, two_hours_ago))

    # A second store triggers eviction, which sweeps stale debris only.
    tl.trace(_CacheModel(), x, capture=_cache_capture(tmp_path))
    assert not stale.exists(), "stale mkstemp debris must be swept at eviction"
    assert fresh.exists(), "a fresh (possibly in-flight) temp must survive"

    # clear_capture_cache sweeps stale debris too (same age gate), and never
    # touches the secret file.
    stale.write_bytes(b"debris-again")
    os.utime(stale, (two_hours_ago, two_hours_ago))
    removed = tl.clear_capture_cache(tmp_path / "cache")
    assert removed >= 1
    assert not stale.exists()
    assert fresh.exists()
    assert (cache_root / ".capture_cache_secret").exists()
