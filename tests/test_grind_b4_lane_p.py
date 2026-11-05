"""Regression coverage for batch-4 Lane P cache, I/O, and perf fixes."""

from __future__ import annotations

import gc
import weakref
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens._capture.prepare.cache as prepare_cache
from torchlens import _user_funcs as user_funcs
from torchlens._capture_state_helpers import (
    _facet_recipe_cache_key,
    _snapshot_plain_attr_value,
)
from torchlens._io import ArtifactError
from torchlens._io.rehydrate import _rehydrate_object
from torchlens._io.scrub import _scrub_value, _ScrubOptions
from torchlens._io.state_keys import (
    static_class_attr,
)
from torchlens._trace_selector_helpers import _predicate_cache_key
from torchlens._utils.display import (
    cleanup_trace_visualizer_dir,
    ensure_trace_visualizer_dir,
)


class _NonPersistentBufferModel(nn.Module):
    """Tiny model whose output depends on non-persistent state."""

    def __init__(self) -> None:
        """Initialize one non-persistent buffer."""

        super().__init__()
        self.register_buffer("offset", torch.ones(1), persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Add the current non-persistent buffer value."""

        return value + self.offset


def _deep_list(depth: int) -> list[Any]:
    """Build a one-child list chain of ``depth`` levels."""

    root: list[Any] = []
    cursor = root
    for _ in range(depth):
        child: list[Any] = []
        cursor.append(child)
        cursor = child
    return root


def test_capture_cache_separates_training_and_nonpersistent_state(tmp_path: Path) -> None:
    """Cache keys include module mode flags and non-persistent buffers."""

    x = torch.ones(1)
    model = _NonPersistentBufferModel().eval()
    first = tl.trace(model, x, capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path))
    assert first.provenance.cache_hit is False
    assert (
        tl.trace(
            model, x, capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path)
        ).provenance.cache_hit
        is True
    )

    model.train()
    training = tl.trace(model, x, capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path))
    assert training.provenance.cache_hit is False

    model.offset.fill_(3)
    changed_buffer = tl.trace(
        model, x, capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path)
    )
    assert changed_buffer.provenance.cache_hit is False
    assert torch.equal(changed_buffer[changed_buffer.output_layers[0]].out, torch.tensor([4.0]))


def test_callable_cache_keys_include_code_and_are_stable() -> None:
    """Redefined nested functions separate by code while equivalent ones match."""

    def make_predicate(index: int) -> Any:
        """Build one closure-backed predicate."""

        return lambda ctx: ctx.raw_index == index

    first = make_predicate(1)
    same = make_predicate(1)
    changed = lambda ctx: ctx.raw_index != 1  # noqa: E731 - cache regression targets lambdas

    assert _predicate_cache_key(first) == _predicate_cache_key(same)
    assert _predicate_cache_key(first) != _predicate_cache_key(changed)
    assert _facet_recipe_cache_key([first]) != _facet_recipe_cache_key([changed])


def test_static_attr_cache_never_self_retains_its_weak_key() -> None:
    """A class attribute reaching the class must not pin the weak key (R37/b4:R39-6).

    ``answers[name] = resolved`` inside a ``WeakKeyDictionary`` value strongly
    reached the key when the attribute IS the class (``self_ref``), directly
    references it (``registry``), or is an instance of it (the fingerprint
    stored ``type(value)`` -- the class object itself). Eviction could then
    never start and the class lived forever.
    """

    class Ephemeral:
        marker = 1

    Ephemeral.self_ref = Ephemeral
    Ephemeral.registry = [Ephemeral]
    Ephemeral.default = Ephemeral()  # fingerprint half: type(value) is the class

    # Lookups still answer correctly (uncached for the self-reaching names).
    assert static_class_attr(Ephemeral, "self_ref") is Ephemeral
    assert static_class_attr(Ephemeral, "registry") == [Ephemeral]
    assert static_class_attr(Ephemeral, "marker") == 1
    assert static_class_attr(Ephemeral, "self_ref") is Ephemeral  # repeat read

    class_ref = weakref.ref(Ephemeral)
    del Ephemeral
    gc.collect()
    gc.collect()
    assert class_ref() is None, (
        "the static-attr memo self-retained its weak key through a "
        "class-reaching attribute value or the type-object fingerprint"
    )


def test_capture_cache_lru_and_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture cache enforces its entry cap and exposes bounded clearing."""

    monkeypatch.setattr(prepare_cache, "_CAPTURE_CACHE_MAX_ENTRIES", 2)
    x = torch.ones(1, 1)
    for value in (1.0, 2.0, 3.0):
        model = nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(value)
        tl.trace(model, x, capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path))

    cache_root = tmp_path / "capture"
    assert len(list(cache_root.glob("*.pkl"))) == 2
    assert user_funcs.clear_cache(tmp_path) == 2
    assert list(cache_root.glob("*.pkl")) == []
    assert (cache_root / ".capture_cache_secret").is_file()


def test_scrub_preserves_shared_and_cyclic_mutable_containers() -> None:
    """Portable scrub memoizes ordinary containers before descending."""

    shared = ["value"]
    root: list[Any] = [shared, shared]
    cycle: list[Any] = []
    cycle.append(cycle)
    root.append(cycle)
    scrubbed = _scrub_value(
        root,
        _ScrubOptions(False, False, False, False),
        {},
        [],
        [0],
    )

    assert scrubbed[0] is scrubbed[1]
    assert scrubbed[2][0] is scrubbed[2]


def test_portable_walks_refuse_excessive_depth_typed() -> None:
    """Save and load walkers turn hostile depth into ArtifactError."""

    deep = _deep_list(205)
    with pytest.raises(ArtifactError, match="maximum depth"):
        _scrub_value(
            deep,
            _ScrubOptions(False, False, False, False),
            {},
            [],
            [0],
        )

    with pytest.raises(ArtifactError, match="maximum depth"):
        _rehydrate_object(
            deep,
            {},
            Path("."),
            Path("."),
            False,
            "cpu",
            True,
            None,
            False,
            [],
            {},
        )


def test_plain_attr_snapshot_refuses_cycles_and_excessive_depth() -> None:
    """Validation fallback snapshots bound recursive user-owned state."""

    cycle: list[Any] = []
    cycle.append(cycle)
    with pytest.raises(RuntimeError, match="cyclic plain attribute"):
        _snapshot_plain_attr_value(cycle, "model.state")
    with pytest.raises(RuntimeError, match="container levels"):
        _snapshot_plain_attr_value(_deep_list(70), "model.state")


def test_visualizer_cleanup_helper_is_idempotent() -> None:
    """Scratch cleanup remains safe after the directory is already gone."""

    class Owner:
        """Weak-referenceable scratch owner."""

    owner = Owner()
    output_dir = ensure_trace_visualizer_dir(owner)
    cleanup_trace_visualizer_dir(owner)
    cleanup_trace_visualizer_dir(owner)
    assert not output_dir.exists()


def test_streamed_bundle_lazy_load_keeps_relation_labels_as_strings(tmp_path: Path) -> None:
    """Rehydrate memos must never serve a stale rebuilt value for a recycled id.

    Regression: the portable-walk memos keyed rebuilt containers by
    ``id(original)`` without pinning the originals. Replaced originals were
    freed mid-walk, CPython recycled their addresses, and later nodes hit the
    stale entries -- a lazy-loaded op's ``parents`` tuple came back holding the
    previous op's ``EdgeUseRecord`` payload instead of label strings.
    """

    class _TwoOpModel(nn.Module):
        """Conv-then-ReLU model exercising multi-op relation rehydration."""

        def __init__(self) -> None:
            """Initialize the two chained modules."""

            super().__init__()
            self.conv = nn.Conv2d(1, 2, 3)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run conv then relu."""

            return self.relu(self.conv(x))

    bundle_path = tmp_path / "streamed_relation_bundle.tl"
    tl.trace(
        _TwoOpModel(),
        torch.randn(1, 1, 8, 8),
        capture=tl.options.CaptureOptions(seed=0),
        save="all",
        storage=tl.options.StorageOptions(path=bundle_path),
    )
    lazy_log = tl.load(bundle_path, lazy=True)
    for op in lazy_log.ops:
        assert all(isinstance(parent, str) for parent in op.parents), op.label
        assert all(isinstance(child, str) for child in op.children), op.label


def test_over_ceiling_entry_is_refused_at_store_not_wiped_at_evict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry above the byte ceiling must never poison the cache (r2 F39-2).

    The store path had no size gate: an over-ceiling trace was fully pickled
    to disk on EVERY capture, the eviction pass then deleted every OTHER
    valid entry to satisfy the byte cap (the just-written ``keep`` is exempt
    but its bytes still count), and the read ceiling refused the entry on
    every later load -- one huge capture wiped the cache each run and could
    itself never hit. The store now refuses (with a warning) instead.
    """

    x = torch.ones(1, 1)
    for value in (1.0, 2.0):
        model = nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(value)
        tl.trace(model, x, capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path))
    cache_root = tmp_path / "capture"
    small_entries = sorted(path.name for path in cache_root.glob("*.pkl"))
    assert len(small_entries) == 2
    total_small = sum(path.stat().st_size for path in cache_root.glob("*.pkl"))

    # Ceiling above the two valid entries combined, below the big capture.
    monkeypatch.setattr(prepare_cache, "_CAPTURE_CACHE_MAX_BYTES", max(total_small + 4096, 200_000))

    big = nn.Linear(1, 1, bias=False)
    big.register_buffer("big_buffer", torch.arange(120_000, dtype=torch.float32))
    with pytest.warns(UserWarning, match="above the .*byte cache-entry ceiling"):
        first = tl.trace(big, x, capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path))
    assert first.provenance.cache_hit is False

    surviving = sorted(path.name for path in cache_root.glob("*.pkl"))
    assert surviving == small_entries, (
        "refusing the oversized store must leave every valid entry in place"
    )


def test_clear_cache_is_public(tmp_path: Path) -> None:
    """The round-1-agreed remedy tl.clear_cache() is reachable (r2 F39-3).

    The cache-clearing remedy shipped unexported: not in ``__all__`` and
    absent from the top-level namespace, so a user hitting the entry/byte cap
    or an oversized-entry refusal had no supported way to clear the cache.
    """

    assert "clear_cache" in tl.__all__
    x = torch.ones(1, 1)
    model = nn.Linear(1, 1, bias=False)
    tl.trace(model, x, capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path))
    assert len(list((tmp_path / "capture").glob("*.pkl"))) == 1
    assert tl.clear_cache(tmp_path) == 1
    assert list((tmp_path / "capture").glob("*.pkl")) == []


def test_predicate_keys_cover_keyword_only_defaults() -> None:
    """kwonly-default redefinition must change the key (r2 b4-fable R39-4).

    ``def p(ctx, *, thr=0.5)`` redefined with ``thr=0.9`` has identical
    co_code, an empty closure, and ``__defaults__ is None``: both selector
    key lanes collided the two definitions onto one key, serving the stale
    cached trace.
    """

    from torchlens._trace_selector_helpers import _stable_cache_fragment

    namespace_low: dict = {}
    namespace_high: dict = {}
    exec("def predicate(ctx, *, thr=0.5):\n    return ctx > thr\n", namespace_low)  # noqa: S102
    exec("def predicate(ctx, *, thr=0.9):\n    return ctx > thr\n", namespace_high)  # noqa: S102
    low, high = namespace_low["predicate"], namespace_high["predicate"]
    assert low.__code__.co_code == high.__code__.co_code
    assert low.__defaults__ is None and high.__defaults__ is None

    assert _predicate_cache_key(low) != _predicate_cache_key(high)
    assert _stable_cache_fragment(low) != _stable_cache_fragment(high)
    # Identical definitions still agree (no false misses).
    namespace_same: dict = {}
    exec("def predicate(ctx, *, thr=0.5):\n    return ctx > thr\n", namespace_same)  # noqa: S102
    assert _predicate_cache_key(low) == _predicate_cache_key(namespace_same["predicate"])
    assert _stable_cache_fragment(low) == _stable_cache_fragment(namespace_same["predicate"])
