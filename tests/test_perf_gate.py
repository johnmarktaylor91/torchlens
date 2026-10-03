"""Tests for P6 benchmark gate and generated performance snippets."""

from __future__ import annotations

import argparse
import copy
import cProfile
import gc
import statistics
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from benchmarks.generate_perf_numbers import render_numbers_markdown
from benchmarks.perf_gate import compare_gate_payloads, normalize_gate_payload
from benchmarks.perf_suite import (
    _assert_baseline_status_writable,
    _validate_baseline_status_args,
)
from torchlens.backends.torch.model_prep import _traverse_model_modules
from torchlens.data_classes._trace_accessors import (
    TraceModuleCallAccessor,
    _invalidate_trace_module_call_accessor_cache,
)
from torchlens.data_classes.trace import Trace
from torchlens.postprocess import ast_branches
from torchlens.postprocess.loop_grouping_adapter import FrontierNodes, _pop_frontier_node
from torchlens.utils.tensor_utils import get_memory_amount


def _row(
    model: str,
    device: str,
    operation: str,
    median_ms: float,
    *,
    iqr_ms: float = 1.0,
    status: str = "ok",
    cpu_median_ms: float | None = None,
    cpu_iqr_ms: float | None = None,
) -> dict[str, object]:
    """Build one synthetic benchmark row.

    Parameters
    ----------
    model:
        Model identifier.
    device:
        Device identifier.
    operation:
        Operation identifier.
    median_ms:
        Median timing.
    iqr_ms:
        Interquartile range timing.
    status:
        Row status.
    cpu_median_ms:
        Optional process-CPU median timing. The gate is CPU-authoritative
        for TorchLens-owned rows; omit only for rows exercising the
        wall-clock legacy fallback.
    cpu_iqr_ms:
        Optional process-CPU interquartile range timing.

    Returns
    -------
    dict[str, object]
        Benchmark row.
    """

    timing: dict[str, object] = {
        "median_ms": median_ms,
        "iqr_ms": iqr_ms,
    }
    if cpu_median_ms is not None:
        timing["cpu_median_ms"] = cpu_median_ms
    if cpu_iqr_ms is not None:
        timing["cpu_iqr_ms"] = cpu_iqr_ms
    return {
        "model": model,
        "device": device,
        "operation": operation,
        "label": operation,
        "status": status,
        "passes": {"timing": {"timing": timing}},
    }


def _payload(rows: list[dict[str, object]]) -> dict[str, object]:
    """Build a synthetic gate payload.

    Parameters
    ----------
    rows:
        Gate rows.

    Returns
    -------
    dict[str, object]
        Gate payload.
    """

    return normalize_gate_payload(
        {
            "date": "2026-06-12",
            "environment": {"torchlens_git_sha": "abc1234"},
            "rows": rows,
        }
    )


def _write_perf_source(tmp_path: Path, index: int) -> Path:
    """Write a distinct Python source file for AST cache tests.

    Parameters
    ----------
    tmp_path:
        Temporary directory for generated source files.
    index:
        Distinct numeric suffix and return value.

    Returns
    -------
    Path
        Path to the generated source file.
    """

    path = tmp_path / f"perf_cache_{index}.py"
    path.write_text(f"def forward():\n    return {index}\n", encoding="utf-8")
    return path


class _RepeatedModuleCallModel(nn.Module):
    """Model that repeatedly invokes one shared module."""

    def __init__(self, repeats: int) -> None:
        """Initialize the repeated-call model.

        Parameters
        ----------
        repeats:
            Number of calls to the shared module.
        """

        super().__init__()
        self.repeats = repeats
        self.shared = nn.ReLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Invoke the shared module repeatedly.

        Parameters
        ----------
        value:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Tensor produced by the final module call.
        """

        for _ in range(self.repeats):
            value = self.shared(value)
        return value


class _PinnedSmallCaptureChain(nn.Module):
    """Exact deterministic workload for the small-capture ratio gate."""

    def __init__(self, depth: int) -> None:
        """Store the number of eager ReLU calls."""

        super().__init__()
        self.depth = depth

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the pinned number of ReLU operations."""

        for _ in range(self.depth):
            value = torch.relu(value)
        return value


def _interleaved_median_capture_ms(models: list[nn.Module], value: torch.Tensor) -> list[float]:
    """Return per-model median capture times, measured ROUND-ROBIN.

    Process-CPU clock, not wall (r7 R30, sol b4 MED): the wall spelling
    demonstrated a red/green flip on the SAME tip purely from runner load
    (fixed_cost_ratio 0.556 loaded vs 0.352 isolated). The neighboring main
    perf gate is process-CPU-authoritative; the ratio gate follows the same
    discipline (threads are already pinned to 1 by the caller).

    Interleaved, not sequential (r7 fresh-env gate finding): measuring the
    three sizes in blocks let monotonic ambient drift (cold caches warming,
    background GC decaying, CPU frequency ramp) land disproportionately on
    whichever size ran FIRST -- a cold combined session measured the 1-op
    fixed cost at 22.9ms vs 13.4ms isolated and flipped the ratio red.
    Round-robin sampling spreads any drift evenly across all sizes, so the
    RATIO (the thing under test) stays drift-invariant.
    """

    for model in models:
        for _ in range(3):
            tl.trace(model, value).cleanup()
    gc.collect()
    samples: list[list[float]] = [[] for _ in models]
    for _ in range(9):
        for index, model in enumerate(models):
            start = time.process_time_ns()
            trace = tl.trace(model, value)
            samples[index].append(time.process_time_ns() - start)
            trace.cleanup()
    return [statistics.median(rows) / 1_000_000 for rows in samples]


@pytest.mark.heavy
@pytest.mark.serial
def test_pinned_small_capture_fixed_cost_ratio_gate() -> None:
    # heavy, not unmarked (b2 R41 round 5): the moment serial stopped being a
    # budget exemption, this 27-capture ratio gate measured 13.8s CPU in a
    # loaded session — squarely in the 5-20s heavy band the tier docs define.
    """A doubled fixed capture floor cannot hide behind large-model timings."""

    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        value = torch.ones(8)
        one_op_ms, sixteen_op_ms, sixty_four_op_ms = _interleaved_median_capture_ms(
            [
                _PinnedSmallCaptureChain(1),
                _PinnedSmallCaptureChain(16),
                _PinnedSmallCaptureChain(64),
            ],
            value,
        )
    finally:
        torch.set_num_threads(original_threads)

    fixed_cost_ratio = one_op_ms / sixteen_op_ms
    early_slope = (sixteen_op_ms - one_op_ms) / 15
    late_slope = (sixty_four_op_ms - sixteen_op_ms) / 48
    slope_ratio = late_slope / early_slope
    print(
        "pinned small-capture gate: "
        f"1={one_op_ms:.3f}ms 16={sixteen_op_ms:.3f}ms 64={sixty_four_op_ms:.3f}ms "
        f"fixed_ratio={fixed_cost_ratio:.3f} slope_ratio={slope_ratio:.3f}"
    )
    # 0.48 -> 0.55 (2026-08-16 fw7settle, bisected to e763548a r8 R39): the
    # wrong-trace cache trio deleted the module-namespace slot memo as
    # provably unsound (constant-length replacement kept a stale slot list
    # -- provenance laundering), so the session-end sweep pays one fresh
    # isinstance pass per capture: +~3ms fixed on this box (1-op 16.6ms ->
    # 18-19ms, 16/64-op unchanged; isolated ratio 0.43 -> 0.47, loaded
    # fw7c session 0.524). A conscious correctness-over-speed trade, not
    # creep; a DOUBLED fixed floor from the new baseline still reads ~0.63
    # and fails. Tracking exactly-stamped tensors instead of sweeping
    # namespaces would win the cost back -- follow-up, not a gate loosen.
    assert fixed_cost_ratio < 0.55
    assert 0.45 < slope_ratio < 2.25


def _fresh_module_calls(trace: Trace) -> TraceModuleCallAccessor:
    """Reconstruct the pre-cache module-call accessor.

    Parameters
    ----------
    trace:
        Trace whose ModuleCalls should be flattened.

    Returns
    -------
    TraceModuleCallAccessor
        Fresh accessor matching the original property implementation.
    """

    calls: OrderedDict[str, Any] = OrderedDict()
    for module in trace._module_logs:
        for call in module.calls.values():
            calls[call.call_label] = call
    return TraceModuleCallAccessor(calls)


def _profile_call_count(callback: Callable[[], object]) -> int:
    """Return deterministic Python call count for one callback.

    Parameters
    ----------
    callback:
        Workload to profile.

    Returns
    -------
    int
        Total profiled function calls.
    """

    profiler = cProfile.Profile()
    profiler.runcall(callback)
    return sum(entry.callcount for entry in profiler.getstats())


def test_module_calls_accessor_is_cached_with_profiled_rebuild_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ModuleCall flattening happens once and cuts profiled call count sharply."""

    trace = tl.trace(_RepeatedModuleCallModel(repeats=64), torch.ones(1))
    expected = _fresh_module_calls(trace)
    _invalidate_trace_module_call_accessor_cache(trace)

    from torchlens.data_classes import _trace_stats

    original_accessor = _trace_stats.TraceModuleCallAccessor
    rebuild_count = 0

    def counting_accessor(calls: OrderedDict[str, Any]) -> TraceModuleCallAccessor:
        """Count construction while preserving the production accessor type."""

        nonlocal rebuild_count
        rebuild_count += 1
        return original_accessor(calls)

    monkeypatch.setattr(_trace_stats, "TraceModuleCallAccessor", counting_accessor)
    cached = trace.module_calls
    for _ in range(128):
        assert trace.module_calls is cached

    assert rebuild_count == 1
    assert type(cached) is type(expected)
    assert cached.keys() == expected.keys()
    assert all(cached[key] is expected[key] for key in expected.keys())

    cold_calls = _profile_call_count(lambda: [_fresh_module_calls(trace) for _ in range(64)])
    hot_calls = _profile_call_count(lambda: [trace.module_calls for _ in range(64)])
    assert cold_calls * 2 >= hot_calls * 7


def test_module_calls_cache_invalidates_on_supported_call_mutation() -> None:
    """Mutating a scoped ModuleCall accessor refreshes the Trace-level cache."""

    trace = tl.trace(_RepeatedModuleCallModel(repeats=2), torch.ones(1))
    cached = trace.module_calls
    module = trace.modules["shared"]
    replacement = copy.copy(module.calls[0])
    replacement.call_label = "shared:replacement"

    module.calls[1] = replacement

    refreshed = trace.module_calls
    assert refreshed is not cached
    assert "shared:1" not in refreshed
    assert refreshed["shared:replacement"] is replacement


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        (torch.bfloat16, (2, 3)),
        (torch.float32, (4, 5)),
        (torch.float64, (1, 2, 3)),
        (torch.int8, (7,)),
    ],
)
def test_get_memory_amount_matches_tensor_method_byte_count(
    dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    """Tensor memory accounting matches direct Tensor method byte counts."""

    tensor = torch.ones(shape, dtype=dtype)
    expected = int(tensor.nelement() * tensor.element_size())

    assert get_memory_amount(tensor) == expected


def test_ast_file_cache_respects_lru_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AST file cache evicts least-recently-used entries beyond the configured cap."""

    cap = 3
    ast_branches.invalidate_cache()
    monkeypatch.setattr(ast_branches, "_FILE_CACHE_MAX_SIZE", cap)
    paths = [_write_perf_source(tmp_path, index) for index in range(cap + 2)]

    try:
        first_index = ast_branches.get_file_index(str(paths[0]))
        assert first_index is not None

        for path in paths[1:cap]:
            assert ast_branches.get_file_index(str(path)) is not None

        assert ast_branches.get_file_index(str(paths[0])) is first_index

        for path in paths[cap:]:
            assert ast_branches.get_file_index(str(path)) is not None

        assert len(ast_branches._file_cache) <= cap
        assert str(paths[0]) in ast_branches._file_cache
        assert str(paths[1]) not in ast_branches._file_cache
    finally:
        ast_branches.invalidate_cache()


def test_perf_gate_compare_passes_within_tolerance() -> None:
    """Regression gate accepts rows inside the baseline-derived tolerance.

    Rows carry process-CPU statistics because the gate is CPU-authoritative
    for TorchLens-owned rows: a wall-clock-only ``tl_trace`` row BLOCKS by
    default (covered in ``test_perf_gate_wiring.py``).
    """

    baseline = _payload(
        [
            _row(
                "resnet18",
                "cpu",
                "tl_trace",
                100.0,
                iqr_ms=3.0,
                cpu_median_ms=100.0,
                cpu_iqr_ms=3.0,
            )
        ]
    )
    current = _payload(
        [
            _row(
                "resnet18",
                "cpu",
                "tl_trace",
                109.0,
                iqr_ms=6.0,
                cpu_median_ms=109.0,
                cpu_iqr_ms=6.0,
            )
        ]
    )

    comparison = compare_gate_payloads(baseline, current)

    assert comparison["passed"] is True
    assert comparison["checks"][0]["metric"] == "process_cpu"
    assert comparison["checks"][0]["tolerance_ms"] == 10.0


def test_perf_gate_compare_fails_regression_beyond_tolerance() -> None:
    """Regression gate rejects median slowdowns beyond tolerance.

    CPU-authoritative rows, so the FAIL verdict comes from the tolerance
    math itself, not from the wall-clock-fallback block.
    """

    baseline = _payload(
        [
            _row(
                "resnet18",
                "cpu",
                "tl_trace",
                100.0,
                iqr_ms=1.0,
                cpu_median_ms=100.0,
                cpu_iqr_ms=1.0,
            )
        ]
    )
    current = _payload(
        [
            _row(
                "resnet18",
                "cpu",
                "tl_trace",
                130.0,
                iqr_ms=1.0,
                cpu_median_ms=130.0,
                cpu_iqr_ms=1.0,
            )
        ]
    )

    comparison = compare_gate_payloads(baseline, current)

    assert comparison["passed"] is False
    assert comparison["checks"][0]["metric"] == "process_cpu"
    assert comparison["wall_clock_fallback_blocking_rows"] == []
    assert comparison["regressions"][0]["delta_ms"] == 30.0


def test_perf_gate_requires_current_torchlens_rows_ok() -> None:
    """Regression gate rejects skipped or errored current TorchLens rows."""

    baseline = _payload([_row("resnet18", "cpu", "fastlog_halt_25", 10.0)])
    current = _payload([_row("resnet18", "cpu", "fastlog_halt_25", 9.0, status="skipped")])

    comparison = compare_gate_payloads(baseline, current)

    assert comparison["passed"] is False
    assert comparison["status_failures"] == [
        {"model": "resnet18", "device": "cpu", "operation": "fastlog_halt_25", "status": "skipped"}
    ]


def test_generated_numbers_print_halt_headline_only_when_sub_raw() -> None:
    """Generator prints the fractional-x halt headline only when measured sub-1.0x."""

    faster_payload = _payload(
        [
            _row("resnet18", "cpu", "raw_forward", 20.0),
            _row("resnet18", "cpu", "fastlog_halt_25", 18.0),
        ]
    )
    slower_payload = _payload(
        [
            _row("resnet18", "cpu", "raw_forward", 20.0),
            _row("resnet18", "cpu", "fastlog_halt_25", 21.0),
        ]
    )

    faster = render_numbers_markdown(faster_payload)
    slower = render_numbers_markdown(slower_payload)

    assert "Headline: measured `fastlog_halt_25` is 0.90x raw forward" in faster
    assert "Headline: measured `fastlog_halt_25`" not in slower
    assert "| resnet18 | cpu | fastlog_halt_25 | 21.0 | 1.05x | ok |" in slower


def test_generated_numbers_suppress_halt_headline_for_provisional_baseline() -> None:
    """Generator suppresses the halt headline for provisional baselines."""

    payload = _payload(
        [
            _row("resnet18", "cpu", "raw_forward", 20.0),
            _row("resnet18", "cpu", "fastlog_halt_25", 18.0),
        ]
    )
    payload["baseline_status"] = "provisional"

    markdown = render_numbers_markdown(payload)

    assert "Headline: measured `fastlog_halt_25`" not in markdown


def test_generated_numbers_emit_halt_headline_for_canonical_baseline() -> None:
    """Generator emits the halt headline for canonical baselines."""

    payload = _payload(
        [
            _row("resnet18", "cpu", "raw_forward", 20.0),
            _row("resnet18", "cpu", "fastlog_halt_25", 18.0),
        ]
    )
    payload["baseline_status"] = "canonical"

    markdown = render_numbers_markdown(payload)

    assert "Headline: measured `fastlog_halt_25` is 0.90x raw forward" in markdown


def test_canonical_baseline_rejects_smoke_mode() -> None:
    """Canonical baseline stamping rejects smoke runs."""

    args = argparse.Namespace(
        baseline_status="canonical",
        smoke=True,
        addendum_no_save=False,
    )

    with pytest.raises(SystemExit, match="requires a complete full run"):
        _validate_baseline_status_args(args)


def test_canonical_baseline_rejects_addendum_no_save_mode() -> None:
    """Canonical baseline stamping rejects addendum-only runs."""

    args = argparse.Namespace(
        baseline_status="canonical",
        smoke=False,
        addendum_no_save=True,
    )

    with pytest.raises(SystemExit, match="requires a complete full run"):
        _validate_baseline_status_args(args)


def test_partial_payload_is_not_writable_as_canonical() -> None:
    """Canonical baseline stamping rejects incomplete assembled payloads."""

    payload = {
        "baseline_status": "canonical",
        "status": "partial",
    }

    with pytest.raises(SystemExit, match="requires an ok run status"):
        _assert_baseline_status_writable(payload)


def test_emit_tensor_grad_event_populates_profile_buckets() -> None:
    """Tensor grad hooks record P6 profiling buckets for later optimization."""

    model = nn.Linear(3, 2)
    x = torch.randn(1, 3, requires_grad=True)
    trace = tl.trace(model, x, profile=True, capture=tl.options.CaptureOptions(backward_ready=True))

    trace.log_backward(trace[trace.output_layers[0]].out.sum())

    timings = trace._phase_timings
    assert timings["backward_grad_event:ensure_stream"]["count"] >= 1
    assert timings["backward_grad_event:ensure_pass"]["count"] >= 1
    assert timings["backward_grad_event:payload"]["count"] >= 1
    assert timings["backward_grad_event:append"]["count"] >= 1


def test_loop_frontier_uses_deque_fifo_order() -> None:
    """Frontier popping is direction-major FIFO with deque-backed candidates."""

    frontier_nodes: FrontierNodes = OrderedDict(
        [
            ("sg1", {"children": deque(["c1", "c2"]), "parents": deque(["p1"])}),
            ("sg2", {"children": deque(["c3"]), "parents": deque(["p2"])}),
        ]
    )

    assert isinstance(frontier_nodes["sg1"]["children"], deque)
    # Direction-major on purpose: every subgraph's children drain before any
    # parents so a loop-carried value shared between a child frontier and a
    # parent frontier is absorbed through its child position first.
    assert _pop_frontier_node(frontier_nodes) == ("c1", "children", "sg1")
    assert _pop_frontier_node(frontier_nodes) == ("c2", "children", "sg1")
    assert _pop_frontier_node(frontier_nodes) == ("c3", "children", "sg2")
    assert _pop_frontier_node(frontier_nodes) == ("p1", "parents", "sg1")
    assert _pop_frontier_node(frontier_nodes) == ("p2", "parents", "sg2")
    assert _pop_frontier_node(frontier_nodes) == (None, None, None)


def test_model_module_traversal_child_addresses_match_named_modules() -> None:
    """Incremental traversal addresses match PyTorch's dotted module names."""

    model = nn.Sequential(
        OrderedDict(
            [
                ("stem", nn.Linear(2, 2)),
                ("block", nn.Sequential(OrderedDict([("act", nn.ReLU())]))),
            ]
        )
    )
    visited_addresses: list[str] = []
    child_addresses: list[str] = []

    def _visitor(
        module: nn.Module,
        address: str,
        child_entries: list[tuple[str, nn.Module, str]],
        is_root: bool,
    ) -> None:
        """Collect traversal addresses for comparison with ``named_modules``."""

        del module, is_root
        visited_addresses.append(address)
        child_addresses.extend(child_address for _, _, child_address in child_entries)

    _traverse_model_modules(model, _visitor)

    expected_addresses = [name for name, _ in model.named_modules()]
    assert visited_addresses == expected_addresses
    assert child_addresses == expected_addresses[1:]
