"""Regression coverage for batch-4 Lane P cache, I/O, and perf fixes."""

from __future__ import annotations

import gc
import warnings
import weakref
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import user_funcs
from torchlens._capture_state_helpers import (
    _facet_recipe_cache_key,
    _snapshot_plain_attr_value,
)
from torchlens._io import TorchLensIOError
from torchlens._io.rehydrate import _rehydrate_object
from torchlens._io.scrub import _scrub_value, _ScrubOptions
from torchlens._io.state_keys import (
    _STATIC_ATTR_MEMO,
    invalidate_static_class_attr_cache,
    static_class_attr,
)
from torchlens._trace_selector_helpers import _predicate_cache_key
from torchlens.utils.display import cleanup_trace_visualizer_dir, ensure_trace_visualizer_dir
from torchlens.visualization import auto_collapse
from torchlens.visualization.auto_collapse import analyze_collapse

pytestmark = pytest.mark.smoke


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
    assert first.capture_cache_hit is False
    assert (
        tl.trace(
            model, x, capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path)
        ).capture_cache_hit
        is True
    )

    model.train()
    training = tl.trace(model, x, capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path))
    assert training.capture_cache_hit is False

    model.offset.fill_(3)
    changed_buffer = tl.trace(
        model, x, capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path)
    )
    assert changed_buffer.capture_cache_hit is False
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


def test_static_attr_cache_detects_replacement_and_does_not_pin_classes() -> None:
    """Definition fingerprints see same-size replacement and weak keys permit GC."""

    class Ephemeral:
        marker = 1

    assert static_class_attr(Ephemeral, "marker") == 1
    Ephemeral.marker = property(lambda self: 2)
    invalidate_static_class_attr_cache()
    assert isinstance(static_class_attr(Ephemeral, "marker"), property)

    class_ref = weakref.ref(Ephemeral)
    del Ephemeral
    gc.collect()
    assert class_ref() is None
    assert all(key is not class_ref() for key in _STATIC_ATTR_MEMO)


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

    monkeypatch.setattr(user_funcs, "_CAPTURE_CACHE_MAX_ENTRIES", 2)
    x = torch.ones(1, 1)
    for value in (1.0, 2.0, 3.0):
        model = nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(value)
        tl.trace(model, x, capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path))

    cache_root = tmp_path / "capture"
    assert len(list(cache_root.glob("*.pkl"))) == 2
    assert user_funcs.clear_capture_cache(tmp_path) == 2
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
    """Save and load walkers turn hostile depth into TorchLensIOError."""

    deep = _deep_list(205)
    with pytest.raises(TorchLensIOError, match="maximum depth"):
        _scrub_value(
            deep,
            _ScrubOptions(False, False, False, False),
            {},
            [],
            [0],
        )

    with pytest.raises(TorchLensIOError, match="maximum depth"):
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


def test_rehydrate_preserves_shared_and_cyclic_containers() -> None:
    """Portable rehydration memoizes rebuilt immutable and mutable containers."""

    shared = ("value",)
    cycle: list[Any] = []
    cycle.append(cycle)
    root: list[Any] = [shared, shared, cycle]
    rehydrated = _rehydrate_object(
        root,
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

    assert rehydrated[0] is rehydrated[1]
    assert rehydrated[2][0] is rehydrated[2]


def test_plain_attr_snapshot_refuses_cycles_and_excessive_depth() -> None:
    """Validation fallback snapshots bound recursive user-owned state."""

    cycle: list[Any] = []
    cycle.append(cycle)
    with pytest.raises(RuntimeError, match="cyclic plain attribute"):
        _snapshot_plain_attr_value(cycle, "model.state")
    with pytest.raises(RuntimeError, match="container levels"):
        _snapshot_plain_attr_value(_deep_list(70), "model.state")


def test_visualizer_scratch_is_removed_by_trace_cleanup() -> None:
    """Explicit trace cleanup removes all trace-owned visualizer artifacts."""

    trace = tl.trace(nn.Identity(), torch.ones(1))
    output_dir = ensure_trace_visualizer_dir(trace)
    (output_dir / "probe.png").write_bytes(b"probe")
    trace.cleanup()

    assert not output_dir.exists()
    assert getattr(trace, "_visualizer_dir", None) is None


def test_log_model_metadata_does_not_self_deprecate() -> None:
    """Canonical metadata helper uses grouped options internally."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.io.log_model_metadata(nn.Identity(), torch.ones(1))
    assert not [warning for warning in caught if warning.category is DeprecationWarning]


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
        capture=tl.options.CaptureOptions(layers_to_save="all", random_seed=0),
        streaming=tl.options.StreamingOptions(bundle_path=bundle_path),
    )
    lazy_log = tl.load(bundle_path, lazy=True)
    for op in lazy_log.ops:
        assert all(isinstance(parent, str) for parent in op.parents), op.label
        assert all(isinstance(child, str) for child in op.children), op.label


def test_collapse_analysis_cache_invalidates_after_equal_size_graph_edit() -> None:
    """Visualization caches fingerprint graph content rather than trace identity alone."""

    trace = tl.trace(nn.Sequential(nn.ReLU()), torch.ones(1))
    first = analyze_collapse(trace)
    relu = next(op for op in trace.ops if op.func_name == "relu")
    relu.func_name = "relu_cache_probe"

    second = analyze_collapse(trace)

    assert second is not first
    assert any("relu_cache_probe" in signal.own_func_names for signal in second.signals.values())


def test_collapse_analysis_fingerprints_once_per_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One analysis pass must validate the graph fingerprint O(1) times, not per edge.

    Regression: content-based cache validation recomputed the O(ops) graph
    fingerprint inside every per-edge relationship resolution, turning one
    ``analyze_collapse`` miss into O(edges x ops) work — minutes-long
    "hangs" on real torchvision graphs. The walk must validate once at entry
    and thread the validated adjacency index through the edge loops.
    """

    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 4, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, 10),
    )
    trace = tl.trace(model, torch.randn(1, 1, 8, 8))
    edge_count = sum(len(op.children) for op in trace.ops)
    assert edge_count >= 5

    calls = 0
    real_revision = auto_collapse._collapse_graph_revision

    def counting_revision(target: Any) -> tuple[object, ...]:
        """Count fingerprint computations while preserving behavior."""

        nonlocal calls
        calls += 1
        return real_revision(target)

    monkeypatch.setattr(auto_collapse, "_collapse_graph_revision", counting_revision)
    analyze_collapse(trace)
    assert calls <= 2, f"{calls} fingerprint walks for {edge_count} edges"


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
    monkeypatch.setattr(user_funcs, "_CAPTURE_CACHE_MAX_BYTES", max(total_small + 4096, 200_000))

    big = nn.Linear(1, 1, bias=False)
    big.register_buffer("big_buffer", torch.arange(120_000, dtype=torch.float32))
    with pytest.warns(UserWarning, match="above the .*byte cache-entry ceiling"):
        first = tl.trace(big, x, capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path))
    assert first.capture_cache_hit is False

    surviving = sorted(path.name for path in cache_root.glob("*.pkl"))
    assert surviving == small_entries, (
        "refusing the oversized store must leave every valid entry in place"
    )


def test_clear_capture_cache_is_public(tmp_path: Path) -> None:
    """The round-1-agreed remedy tl.clear_capture_cache() is reachable (r2 F39-3).

    user_funcs.clear_capture_cache shipped unexported: not in ``__all__`` and
    absent from the top-level namespace, so a user hitting the entry/byte cap
    or an oversized-entry refusal had no supported way to clear the cache.
    """

    assert "clear_capture_cache" in tl.__all__
    x = torch.ones(1, 1)
    model = nn.Linear(1, 1, bias=False)
    tl.trace(model, x, capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path))
    assert len(list((tmp_path / "capture").glob("*.pkl"))) == 1
    assert tl.clear_capture_cache(tmp_path) == 1
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


def test_accessor_caches_invalidate_on_rebind_resize_and_invalidator(tmp_path: Path) -> None:
    """Accessor-memo invalidation contract (r1 row 13, re-keyed in r3 R52-1).

    The r1 fix keyed the ``trace.ops`` / ``trace.layers`` memos BY VALUE
    (ordered label tuples) so even a direct equal-length in-place element
    swap auto-invalidated -- but that priced every memo HIT at O(n), turned
    hot ``trace.ops`` sweeps O(n^2), and measurably regressed ``tl.trace``
    itself (r3 R52-1/R28-1, the round-3 fixplan prescribes the identity+len
    key). The contract is now: REASSIGNMENT invalidates (identity), any
    add/remove invalidates (length), the internal rename/refresh paths call
    ``_invalidate_trace_op_layer_accessor_caches`` explicitly, and direct
    equal-length in-place mutation of the build products requires that same
    explicit invalidator.
    """

    from torchlens.data_classes._trace_accessors import (
        _invalidate_trace_op_layer_accessor_caches,
    )

    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    trace = tl.trace(model, torch.ones(1, 2))
    ops_before = trace.ops
    first_two = list(trace.layer_list[:2])

    # Unchanged containers re-serve the memo (identity-stable, O(1) key).
    assert trace.ops is ops_before

    # Equal-length in-place swap + the explicit invalidator (the supported
    # spelling for direct in-place edits of trace build products).
    trace.layer_list[0], trace.layer_list[1] = trace.layer_list[1], trace.layer_list[0]
    _invalidate_trace_op_layer_accessor_caches(trace)
    swapped = trace.ops
    assert swapped is not ops_before
    assert list(swapped)[:2] == [first_two[1], first_two[0]]

    # Equal-length reassignment auto-invalidates via container identity.
    layers_before = trace.layers
    items = list(trace.layer_logs.items())
    trace.layer_logs = dict(reversed(items))
    layers_after = trace.layers
    assert layers_after is not layers_before

    # A size change auto-invalidates via the length axis.
    ops_full = trace.ops
    removed = trace.layer_list.pop()
    try:
        assert trace.ops is not ops_full
    finally:
        trace.layer_list.append(removed)

    # Unchanged content still re-serves the memo (no per-access rebuilds).
    assert trace.layers is layers_after


def test_ops_memo_hit_is_o1_no_label_reads(tmp_path: Path) -> None:
    """A memo HIT must not touch per-op labels (r3 R52-1 O(n^2) kill).

    The r3 by-value key rebuilt the full O(n) label tuple on EVERY
    ``trace.ops`` / ``trace.layers`` access, so an n-op sweep that resolved
    ops through the accessor property cost O(n^2) label reads (measured
    exponent ~2.1, 9.6ms -> 138.5ms over 102 -> 402 ops).
    """

    from torchlens.data_classes.op import Op

    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
    trace = tl.trace(model, torch.ones(1, 2))
    warm = trace.ops
    assert trace.layers is trace.layers

    label_descriptor = Op.label
    reads = {"count": 0}

    def _counting_label(op_self):
        reads["count"] += 1
        return label_descriptor.__get__(op_self, Op)

    Op.label = property(_counting_label)  # type: ignore[assignment]
    try:
        for _ in range(32):
            assert trace.ops is warm
            trace.layers
    finally:
        Op.label = label_descriptor  # type: ignore[assignment]

    assert reads["count"] == 0, (
        f"memo hits must be O(1): {reads['count']} label reads across 32 accesses"
    )


def test_barcode_remap_scrubs_every_registered_barcode(tmp_path: Path) -> None:
    """Every live param barcode is remapped in persisted identity strings (R29).

    Pins the fast P-independent token scan that replaced the O(V x P)
    alternation regex: coverage must stay exact -- a raw capture barcode
    leaking into a persisted equivalence key would silently break
    cross-process byte reproducibility.
    """

    import torch.nn as nn

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    trace = tl.trace(model, torch.randn(2, 4))
    raw_barcodes = {
        barcode
        for op in trace.ops
        for barcode in (getattr(op, "_param_barcodes", ()) or ())
        if isinstance(barcode, str)
    }
    assert raw_barcodes, "expected live param barcodes on a param-bearing capture"

    tl.save(trace, str(tmp_path / "t.tlspec"))
    loaded = tl.load(str(tmp_path / "t.tlspec"))

    leaked: list[str] = []
    for op in loaded.ops:
        equivalence = getattr(op, "equivalence_class", None)
        if isinstance(equivalence, str):
            leaked.extend(b for b in raw_barcodes if b in equivalence)
    for key in getattr(loaded, "op_equivalence_classes", {}) or {}:
        if isinstance(key, str):
            leaked.extend(b for b in raw_barcodes if b in key)
    assert not leaked, f"raw capture barcodes leaked into persisted identity keys: {leaked!r}"
    remapped = [
        getattr(op, "equivalence_class", "")
        for op in loaded.ops
        if isinstance(getattr(op, "equivalence_class", None), str)
        and "param_" in getattr(op, "equivalence_class", "")
    ]
    assert remapped, "expected canonical param_NNNNNN identities on param ops"
