"""Regression tests for the per-draw collapse-rolling suffix memo.

Root cause (SPEEDENING wave 19): ``_collapsed_module_rolling_suffix`` is a pure
function of the captured graph, but the renderer re-entered it once per
collapsed-module NODE (``_render_nodes``, ``_render_dot``, ``render_ir``, and
``_single_op_module_should_keep_op_render``). Each entry re-walked EVERY rolled
layer and, through ``_call_groups_for_layer`` ->
``_same_layer_dependency_components`` -> ``_same_layer_reachability``, every pass
of every such layer. Cost was ``collapsed_nodes x layers x passes``: replaying the
unrolled renderer's access pattern on a 320-step ``nn.LSTMCell`` loop cost 41.8s of
CPU and grew ~8x per doubling of loop length.

Fix: restore a per-draw ContextVar memo scope -- the idiom this surface used before
the renderer decomposition -- and resolve every address from ONE pass over the
rolled layers. 41.8s -> 0.12s at 320 steps. Emitted DOT is byte-identical; these
tests pin the *shape* of the fix so it cannot silently regress:

* the rolled layers are walked O(1) times per draw, not once per collapsed node --
  a call-count assertion, no clock involved;
* the one-pass map agrees with the historical per-address scan on every module
  address, including addresses with a real split partition such as ``":1-2,3-5"``;
* the memo is scoped to one draw, so a second draw builds a fresh cache and nothing
  is retained once the draw returns.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.data_classes.layer import Layer
from torchlens.visualization import _render_leaf

os.environ.setdefault("MPLBACKEND", "Agg")

pytestmark = pytest.mark.smoke


class TwoDistinctLoops(nn.Module):
    """One reused block driven by two disjoint call runs -- a real split partition."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.block = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = x
        for _ in range(2):
            first = self.block(first)
        second = x * 2
        for _ in range(3):
            second = self.block(second)
        return first + second


class CellLoop(nn.Module):
    """A reused atomic cell -- one collapsed-module node per call when unrolled."""

    def __init__(self, hidden: int = 8, steps: int = 32) -> None:
        super().__init__()
        self.steps = steps
        self.cell = nn.LSTMCell(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = torch.zeros(x.shape[0], x.shape[-1])
        cx = torch.zeros(x.shape[0], x.shape[-1])
        for _ in range(self.steps):
            hx, cx = self.cell(x, (hx, cx))
        return hx


def _rolled_layer_count(trace: tl.Trace) -> int:
    """Number of layers whose passes the reachability walk can be asked about."""

    return sum(
        1
        for layer in trace.layer_logs.values()
        if isinstance(layer, Layer) and layer.num_passes > 1
    )


def _historical_dependency_components(layer_log: Layer) -> tuple[tuple[int, ...], ...]:
    """Compute pass components through the historical per-pass reachability walk."""

    reachability = _render_leaf._same_layer_reachability(layer_log)
    # OpAccessor iteration yields Ops (C02); .keys() is the pass-index view.
    adjacency = {pass_index: set() for pass_index in layer_log.ops.keys()}  # noqa: SIM118
    for source, targets in reachability.items():
        for target in targets:
            adjacency[source].add(target)
            adjacency[target].add(source)
    components: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for pass_index in sorted(layer_log.ops.keys()):
        if pass_index in seen:
            continue
        stack = [pass_index]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(adjacency[current] - seen)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda values: values[0]))


def _historical_rolling_suffix(trace: tl.Trace, address: str) -> str:
    """The pre-wave-19 per-address scan, verbatim, as the oracle."""

    candidate_groups: tuple[tuple[int, ...], ...] = ()
    for layer_log in trace.layer_logs.values():
        if not isinstance(layer_log, Layer) or layer_log.num_passes <= 1:
            continue
        layer_addresses = {
            parsed[0]
            for op in layer_log.ops.values()
            for module_call in op.modules
            if (parsed := _render_leaf._module_address_and_call(module_call)) is not None
        }
        if address not in layer_addresses:
            continue
        groups = _render_leaf._call_groups_for_layer_uncached(layer_log)
        if len(groups) > len(candidate_groups):
            candidate_groups = groups
    if not candidate_groups:
        return ""
    return f":{_render_leaf._format_call_groups(candidate_groups)}"


def _counted_draw(
    monkeypatch: pytest.MonkeyPatch,
    trace: tl.Trace,
    outpath: Path,
    **draw_kwargs: object,
) -> dict[str, int]:
    """Draw once, counting entries into the two rolled-layer sweeps.

    Both helpers are reached through the ``_render_leaf`` module globals, so the
    renderer's star-imported callers route into these counters.
    """

    counts: dict[str, int] = defaultdict(int)
    original_map = _render_leaf._collapsed_module_rolling_suffix_map
    original_reach = _render_leaf._same_layer_reachability
    original_suffix = _render_leaf._collapsed_module_rolling_suffix

    def counting_map(trace_arg: tl.Trace) -> dict[str, str]:
        counts["map"] += 1
        return original_map(trace_arg)

    def counting_reach(layer_log: Layer) -> dict[int, set[int]]:
        counts["reachability"] += 1
        return original_reach(layer_log)

    def counting_suffix(trace_arg: tl.Trace, address: str) -> str:
        counts["suffix"] += 1
        return original_suffix(trace_arg, address)

    monkeypatch.setattr(_render_leaf, "_collapsed_module_rolling_suffix_map", counting_map)
    monkeypatch.setattr(_render_leaf, "_same_layer_reachability", counting_reach)
    for module_name in ("_render_dot", "_render_nodes", "render_ir"):
        module = __import__(f"torchlens.visualization.{module_name}", fromlist=[module_name])
        if hasattr(module, "_collapsed_module_rolling_suffix"):
            monkeypatch.setattr(module, "_collapsed_module_rolling_suffix", counting_suffix)

    trace.draw(vis_save_only=True, vis_outpath=str(outpath), **draw_kwargs)
    return counts


@pytest.mark.parametrize("collapse", ["auto", "max"])
def test_unrolled_collapsed_nodes_share_one_rolled_layer_sweep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    collapse: str,
) -> None:
    """Many collapsed-module nodes, ONE pass over the rolled layers.

    A reused atomic cell renders one collapsed node per call, and every one of them
    asks for the module's rolling suffix. Before the fix each of those questions
    re-walked every rolled layer and every pass; a regression would show up here as
    the sweep count scaling with the node count.
    """

    trace = tl.trace(CellLoop(steps=32), torch.rand(2, 8))
    rolled_layers = _rolled_layer_count(trace)
    assert rolled_layers >= 1, "fixture must produce at least one rolled layer"

    counts = _counted_draw(
        monkeypatch,
        trace,
        tmp_path / "graph",
        vis_mode="unrolled",
        collapse=collapse,
    )

    # The fixture must actually exercise the re-entry the fix targets.
    assert counts["suffix"] > 8, f"expected many collapsed nodes, got {counts['suffix']}"
    # ... yet the expensive sweeps happen once, not once per node.
    assert counts["map"] == 1
    assert counts["reachability"] <= rolled_layers


@pytest.mark.parametrize("collapse", ["auto", "max"])
def test_rolled_view_also_shares_one_rolled_layer_sweep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    collapse: str,
) -> None:
    """The rolled view resolves suffixes for module clusters from the same one pass."""

    trace = tl.trace(CellLoop(steps=12), torch.rand(2, 8))
    rolled_layers = _rolled_layer_count(trace)

    counts = _counted_draw(
        monkeypatch,
        trace,
        tmp_path / "graph",
        vis_mode="rolled",
        collapse=collapse,
    )

    assert counts["suffix"] >= 1
    assert counts["map"] == 1
    assert counts["reachability"] <= rolled_layers


def test_one_pass_map_matches_the_historical_per_address_scan() -> None:
    """The map agrees with the pre-fix scan on every address, split ones included."""

    trace = tl.trace(TwoDistinctLoops(), torch.rand(2, 8))
    probes = sorted(trace.modules.keys()) + ["", "nonexistent"]
    expected = [_historical_rolling_suffix(trace, address) for address in probes]

    @_render_leaf._with_per_draw_collapse_cache
    def scoped() -> list[str]:
        return [_render_leaf._collapsed_module_rolling_suffix(trace, address) for address in probes]

    assert scoped() == expected
    # ... and outside any draw scope, where the memo is bypassed entirely.
    assert [
        _render_leaf._collapsed_module_rolling_suffix(trace, address) for address in probes
    ] == expected
    # The oracle must not pass vacuously: this fixture has a real split partition.
    assert any(value for value in expected), f"expected a split suffix, got {expected}"


def test_memo_does_not_outlive_one_draw() -> None:
    """Each scope starts empty and leaves nothing behind."""

    trace = tl.trace(TwoDistinctLoops(), torch.rand(2, 8))
    entry_states: list[bool] = []

    @_render_leaf._with_per_draw_collapse_cache
    def scoped() -> str:
        cache = _render_leaf._PER_DRAW_COLLAPSE_CACHE.get()
        assert cache is not None
        # A cache carried over from the previous scope would arrive populated.
        entry_states.append(cache.rolling_suffixes is None and not cache.call_groups)
        return _render_leaf._collapsed_module_rolling_suffix(trace, "block")

    first = scoped()
    second = scoped()
    assert first == second
    assert entry_states == [True, True], "each draw must start from an empty cache"
    # Outside every scope the ContextVar is clean, so no layer refs leak past a draw.
    assert _render_leaf._PER_DRAW_COLLAPSE_CACHE.get() is None


@pytest.mark.parametrize("steps", [5, 20, 80])
def test_union_find_components_match_historical_reachability(steps: int) -> None:
    """The linear interior union-find preserves the historical pass partition."""

    trace = tl.trace(CellLoop(steps=steps), torch.rand(2, 8))
    rolled_layers = [
        layer
        for layer in trace.layer_logs.values()
        if isinstance(layer, Layer) and layer.num_passes > 1
    ]
    assert rolled_layers
    for layer in rolled_layers:
        assert _render_leaf._same_layer_dependency_components(
            layer
        ) == _historical_dependency_components(layer)


def test_union_find_partition_is_children_order_invariant() -> None:
    """Permuting child tuples cannot change user-visible rolled pass groups."""

    trace = tl.trace(CellLoop(steps=20), torch.rand(2, 8))
    layer = max(
        (
            candidate
            for candidate in trace.layer_logs.values()
            if isinstance(candidate, Layer) and candidate.num_passes > 1
        ),
        key=lambda candidate: candidate.num_passes,
    )
    expected = _render_leaf._same_layer_dependency_components(layer)
    original_children = [(op, op.children) for op in trace.ops]
    try:
        for op, children in original_children:
            op.children = list(reversed(children))
        assert _render_leaf._same_layer_dependency_components(layer) == expected
    finally:
        for op, children in original_children:
            op.children = children
