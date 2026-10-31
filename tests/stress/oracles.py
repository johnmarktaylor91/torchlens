"""Cell execution and the six oracles.

``run_cell`` executes one cell exactly once and records everything the
oracles read into a ``CellResult``. Each oracle is a pure function from a
``CellResult`` to a ``Verdict``: ``pass``, ``fail`` (a finding: a real bug,
recorded by cell id, never papered over) or ``not_applicable`` (the cell
never reached the point the oracle judges, with the reason).

The oracles never loosen a product check: ``invariants_pass`` reads the
default ``tl.validate`` verdict, and the torch-clean checks are the same
process-global reads the suite's autouse leak fixtures make.
"""

from __future__ import annotations

import copy
import gc
import random
import resource
import sys
import time
import tracemalloc
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.modules import module as module_mod

import torchlens as tl
from torchlens import _state
from torchlens.errors import TorchLensError
from torchlens.options import CaptureOptions
from torchlens.validation import last_validation_failure
from torchlens.validation.diagnostics import CHECK_METADATA_INVARIANT

from .generator import Cell, Inputs

BUILD_SEED = 0
INPUT_SEED = 1
RUN_SEED = 0

#: Torch callables whose identity shows whether the lazy wrappers were
#: installed, re-installed or left half-swapped by a cell.
_WRAPPER_PROBES: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("torch.add", lambda: torch.add),
    ("torch.nn.functional.relu", lambda: torch.nn.functional.relu),
    ("torch.Tensor.add", lambda: torch.Tensor.add),
)

#: TorchLens process-global capture slots a finished call must release.
_CAPTURE_SLOTS = (
    "_active_trace",
    "_active_hook_plan",
    "_active_intervention_spec",
    "_capture_reserved_by",
)


@dataclass(frozen=True)
class Verdict:
    """What one oracle says about one cell."""

    status: str
    detail: str = ""

    @classmethod
    def passed(cls, detail: str = "") -> Verdict:
        """Build a passing verdict."""

        return cls("pass", detail)

    @classmethod
    def failed(cls, detail: str) -> Verdict:
        """Build a failing verdict (a finding)."""

        return cls("fail", detail)

    @classmethod
    def not_applicable(cls, detail: str) -> Verdict:
        """Build a not-applicable verdict."""

        return cls("not_applicable", detail)


@dataclass(frozen=True)
class TorchSnapshot:
    """Process-global torch and TorchLens state read before and after a cell."""

    global_forward_hooks: int
    global_forward_pre_hooks: int
    torch_function_stack: int
    held_slots: tuple[str, ...]
    logging_enabled: bool
    wrappers_installed: bool
    wrapper_ids: tuple[tuple[str, int], ...]


def rng_state() -> tuple[bytes, ...]:
    """Read the torch, ``random`` and NumPy global streams as comparable bytes."""

    numpy_state = np.random.get_state(legacy=True)
    return (
        torch.random.get_rng_state().numpy().tobytes(),
        repr(random.getstate()).encode("utf-8"),
        repr(numpy_state[0]).encode("utf-8") + numpy_state[1].tobytes(),
        repr(numpy_state[2:]).encode("utf-8"),
    )


def snapshot_torch() -> TorchSnapshot:
    """Read the process-global state the torch-clean oracle compares."""

    return TorchSnapshot(
        global_forward_hooks=len(module_mod._global_forward_hooks),
        global_forward_pre_hooks=len(module_mod._global_forward_pre_hooks),
        torch_function_stack=torch._C._len_torch_function_stack(),
        held_slots=tuple(name for name in _CAPTURE_SLOTS if getattr(_state, name) is not None),
        logging_enabled=bool(_state._logging_enabled),
        wrappers_installed=bool(_state._is_decorated),
        wrapper_ids=tuple((name, id(probe())) for name, probe in _WRAPPER_PROBES),
    )


RNG_STREAMS = ("torch", "random", "numpy", "numpy_tail")


def rng_difference(before: tuple[bytes, ...], after: tuple[bytes, ...]) -> list[str]:
    """Name the global streams whose state differs between two readings."""

    return [name for name, x, y in zip(RNG_STREAMS, before, after, strict=True) if x != y]


@dataclass
class CellResult:
    """Everything one cell execution recorded."""

    cell_id: str
    memory_budget_bytes: int
    before: TorchSnapshot
    after: TorchSnapshot | None = None
    rng_changed: dict[str, list[str]] = field(default_factory=dict)
    eager_error: BaseException | None = None
    trace_error: BaseException | None = None
    op_count: int | None = None
    validate_verdict: bool | None = None
    validate_error: BaseException | None = None
    validate_failure_check: str | None = None
    validate_failure_summary: str | None = None
    save_error: BaseException | None = None
    load_error: BaseException | None = None
    loaded_labels_equal: bool | None = None
    loaded_outs_compared: int | None = None
    loaded_outs_unequal: tuple[str, ...] = ()
    tracemalloc_peak_bytes: int | None = None
    maxrss_growth_bytes: int | None = None
    wall_seconds: float | None = None
    warnings_seen: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    def refusals(self) -> tuple[tuple[str, BaseException], ...]:
        """Return every exception TorchLens raised, with the stage that raised it."""

        stages = (
            ("trace", self.trace_error),
            ("validate", self.validate_error),
            ("save", self.save_error),
            ("load", self.load_error),
        )
        return tuple((stage, error) for stage, error in stages if error is not None)


def _maxrss_bytes() -> int:
    """Return the process high-water resident size in bytes."""

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def _trace_inputs(inputs: Inputs) -> tuple[Any, dict[str, Any] | None]:
    """Spell ``(args, kwargs)`` the way ``tl.trace`` and ``tl.validate`` take them."""

    args, kwargs = inputs
    return (list(args) if args else ()), (dict(kwargs) if kwargs else None)


def _to_device(value: Any, device: str) -> Any:
    """Move every tensor inside ``value`` to ``device``."""

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _tensors_identical(live: torch.Tensor, loaded: torch.Tensor) -> bool:
    """Return whether two tensors agree byte for byte, NaN positions included."""

    live = live.detach().cpu()
    loaded = loaded.detach().cpu()
    if live.shape != loaded.shape or live.dtype != loaded.dtype:
        return False
    if live.is_floating_point() or live.is_complex():
        if not torch.equal(torch.isnan(live), torch.isnan(loaded)):
            return False
        return torch.equal(live.nan_to_num(), loaded.nan_to_num())
    return torch.equal(live, loaded)


def _compare_loaded(result: CellResult, trace: Any, loaded: Any) -> None:
    """Record whether the loaded trace matches the live one op by op."""

    live_ops = list(trace.ops)
    loaded_ops = list(loaded.ops)
    result.loaded_labels_equal = [op.label for op in live_ops] == [op.label for op in loaded_ops]
    if not result.loaded_labels_equal:
        return
    compared = 0
    unequal = []
    for live_op, loaded_op in zip(live_ops, loaded_ops, strict=True):
        live_out = live_op.out
        if not isinstance(live_out, torch.Tensor):
            continue
        compared += 1
        loaded_out = loaded_op.out
        if not isinstance(loaded_out, torch.Tensor) or not _tensors_identical(live_out, loaded_out):
            unequal.append(live_op.label)
    result.loaded_outs_compared = compared
    result.loaded_outs_unequal = tuple(unequal)


def _run_save_load(result: CellResult, trace: Any, tmp_path: Path) -> None:
    """Save the trace portably, load it back and compare."""

    bundle = tmp_path / "cell.tlspec"
    rng_before = rng_state()
    try:
        tl.save(trace, bundle, level="portable")
    except Exception as error:
        result.save_error = error
        return
    finally:
        _note_rng(result, "save", rng_before)
    rng_before = rng_state()
    try:
        loaded = tl.load(bundle)
    except Exception as error:
        result.load_error = error
        return
    finally:
        _note_rng(result, "load", rng_before)
    try:
        _compare_loaded(result, trace, loaded)
    finally:
        del loaded


def _note_rng(result: CellResult, stage: str, before: tuple[bytes, ...]) -> None:
    """Record the global streams a TorchLens stage left changed."""

    changed = rng_difference(before, rng_state())
    if changed:
        result.rng_changed[stage] = changed


def _run_validate(result: CellResult, model: nn.Module, inputs: Inputs) -> None:
    """Run default forward validation (replay plus metadata invariants)."""

    args, kwargs = _trace_inputs(inputs)
    torch.manual_seed(RUN_SEED)
    rng_before = rng_state()
    try:
        verdict = tl.validate(model, args, kwargs, scope="forward", random_seed=RUN_SEED)
    except Exception as error:
        result.validate_error = error
        return
    finally:
        _note_rng(result, "validate", rng_before)
    result.validate_verdict = bool(verdict)
    if not verdict:
        failure = last_validation_failure()
        result.validate_failure_check = failure.check if failure is not None else None
        result.validate_failure_summary = (
            failure.summary() if failure is not None else "no failure record"
        )


def _execute(result: CellResult, cell: Cell, tmp_path: Path) -> None:
    """Build, run eagerly, capture, save/load and validate; record each stage."""

    torch.manual_seed(BUILD_SEED)
    model = cell.model.build()
    model.eval()
    torch.manual_seed(INPUT_SEED)
    inputs = cell.model.example_input(model)
    if cell.input.make is not None:
        inputs = cell.input.make(model)
    model, inputs = cell.dtype.cast(model, inputs)
    model = model.to(cell.device.device)
    inputs = _to_device(inputs, cell.device.device)

    args, kwargs = inputs
    torch.manual_seed(RUN_SEED)
    try:
        with torch.no_grad():
            model(*copy.deepcopy(args), **copy.deepcopy(kwargs))
    except Exception as error:
        result.eager_error = error
        return

    trace_args, trace_kwargs = _trace_inputs(inputs)
    capture = CaptureOptions(random_seed=RUN_SEED, **dict(cell.options.capture_kwargs))
    torch.manual_seed(RUN_SEED)
    rng_before = rng_state()
    try:
        trace = tl.trace(
            model, trace_args, trace_kwargs, capture=capture, **dict(cell.options.trace_kwargs)
        )
    except Exception as error:
        result.trace_error = error
        return
    finally:
        _note_rng(result, "trace", rng_before)
    try:
        result.op_count = len(trace.ops)
        _run_save_load(result, trace, tmp_path)
    finally:
        del trace
        gc.collect()

    # Validation re-traces from scratch; one plain forward first gives it the
    # model's steady state (a lazily built cache), as the classics corpus test does.
    with torch.no_grad():
        model(*copy.deepcopy(args), **copy.deepcopy(kwargs))
    _run_validate(result, model, inputs)


def run_cell(cell: Cell, tmp_path: Path) -> CellResult:
    """Execute one cell once and return everything the oracles read.

    Parameters
    ----------
    cell:
        The cell to run.
    tmp_path:
        Scratch directory for the save/load round trip.

    Returns
    -------
    CellResult
        The recorded stages, process-state snapshots and peak-memory readings.
    """

    gc.collect()
    result = CellResult(
        cell_id=cell.id,
        memory_budget_bytes=cell.model.memory_budget_mib * 1024 * 1024,
        before=snapshot_torch(),
    )
    rss_before = _maxrss_bytes()
    started = time.perf_counter()
    started_tracing = not tracemalloc.is_tracing()
    if started_tracing:
        tracemalloc.start()
    else:
        tracemalloc.reset_peak()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            _execute(result, cell, tmp_path)
    finally:
        _, peak = tracemalloc.get_traced_memory()
        if started_tracing:
            tracemalloc.stop()
        gc.collect()
        result.wall_seconds = time.perf_counter() - started
        result.tracemalloc_peak_bytes = peak
        result.maxrss_growth_bytes = max(0, _maxrss_bytes() - rss_before)
        result.after = snapshot_torch()
    result.warnings_seen = tuple(
        sorted({f"{w.category.__name__}: {str(w.message)[:120]}" for w in caught})
    )
    return result


def _describe(error: BaseException) -> str:
    """Return ``Type: message`` for an exception, bounded."""

    return f"{type(error).__name__}: {str(error)[:300]}"


def _before_capture(result: CellResult) -> Verdict | None:
    """Return the not-applicable verdict for a cell that produced no trace."""

    if result.eager_error is not None:
        return Verdict.not_applicable(
            f"eager forward itself raises; the row does not fit this dtype or input: "
            f"{_describe(result.eager_error)}"
        )
    if result.trace_error is not None:
        return Verdict.not_applicable(
            f"capture raised (judged by typed_refusal_names_condition): "
            f"{_describe(result.trace_error)}"
        )
    return None


def replay_equals_eager(result: CellResult) -> Verdict:
    """Forward replay of every captured op reproduces the eager outputs."""

    skipped = _before_capture(result)
    if skipped is not None:
        return skipped
    if result.validate_error is not None:
        return Verdict.not_applicable(
            f"validation raised (judged by typed_refusal_names_condition): "
            f"{_describe(result.validate_error)}"
        )
    if result.validate_verdict:
        return Verdict.passed()
    if result.validate_failure_check == CHECK_METADATA_INVARIANT:
        return Verdict.not_applicable(
            "validation stopped at a metadata invariant (judged by invariants_pass)"
        )
    return Verdict.failed(
        f"forward validation failed at check {result.validate_failure_check!r}: "
        f"{result.validate_failure_summary}"
    )


def invariants_pass(result: CellResult) -> Verdict:
    """Default validation's metadata invariants hold on the captured trace."""

    skipped = _before_capture(result)
    if skipped is not None:
        return skipped
    if result.validate_error is not None:
        return Verdict.not_applicable(
            f"validation raised (judged by typed_refusal_names_condition): "
            f"{_describe(result.validate_error)}"
        )
    if result.validate_verdict:
        return Verdict.passed()
    if result.validate_failure_check == CHECK_METADATA_INVARIANT:
        return Verdict.failed(f"metadata invariant failed: {result.validate_failure_summary}")
    return Verdict.not_applicable(
        f"validation stopped before the invariants at check "
        f"{result.validate_failure_check!r} (judged by replay_equals_eager)"
    )


def save_load_equals_live(result: CellResult) -> Verdict:
    """A portable save and load returns the live ops and their saved outputs."""

    skipped = _before_capture(result)
    if skipped is not None:
        return skipped
    if result.save_error is not None:
        return Verdict.not_applicable(
            f"save raised (judged by typed_refusal_names_condition): {_describe(result.save_error)}"
        )
    if result.load_error is not None:
        return Verdict.not_applicable(
            f"load raised (judged by typed_refusal_names_condition): {_describe(result.load_error)}"
        )
    if not result.loaded_labels_equal:
        return Verdict.failed("save/load changed the op labels")
    if result.loaded_outs_unequal:
        shown = ", ".join(result.loaded_outs_unequal[:5])
        return Verdict.failed(
            f"{len(result.loaded_outs_unequal)} of {result.loaded_outs_compared} saved outputs "
            f"differ after load: {shown}"
        )
    return Verdict.passed(f"{result.loaded_outs_compared} outputs compared")


def typed_refusal_names_condition(result: CellResult) -> Verdict:
    """Every error TorchLens raised is a typed error with a stable code and a message."""

    if result.eager_error is not None:
        return _before_capture(result) or Verdict.passed()
    refusals = result.refusals()
    if not refusals:
        return Verdict.passed("nothing refused")
    problems = []
    codes = []
    for stage, error in refusals:
        if not isinstance(error, TorchLensError):
            problems.append(f"{stage}: untyped {_describe(error)}")
            continue
        code = error.fields.get("code")
        if not isinstance(code, str) or not code:
            problems.append(f"{stage}: {type(error).__name__} carries no code: {str(error)[:200]}")
        elif not str(error).strip():
            problems.append(f"{stage}: {type(error).__name__} ({code}) has an empty message")
        else:
            codes.append(f"{stage}: {code}")
    if problems:
        return Verdict.failed("; ".join(problems))
    return Verdict.passed("; ".join(codes))


def _snapshot_differences(before: TorchSnapshot, after: TorchSnapshot) -> list[str]:
    """List the process-global state a cell left changed."""

    problems = []
    if after.held_slots:
        problems.append(f"capture slots still held: {list(after.held_slots)}")
    if after.logging_enabled:
        problems.append("_logging_enabled left True")
    hooks_before = (before.global_forward_hooks, before.global_forward_pre_hooks)
    hooks_after = (after.global_forward_hooks, after.global_forward_pre_hooks)
    if hooks_after != hooks_before:
        problems.append(f"global module hooks changed {hooks_before} to {hooks_after}")
    if after.torch_function_stack != before.torch_function_stack:
        problems.append(
            f"torch-function mode stack changed {before.torch_function_stack} "
            f"to {after.torch_function_stack}"
        )
    if before.wrappers_installed and not after.wrappers_installed:
        problems.append("torch wrappers were uninstalled by the cell")
    if before.wrappers_installed and after.wrapper_ids != before.wrapper_ids:
        problems.append("installed torch wrappers were swapped for new objects")
    return problems


def torch_clean_after_cell(result: CellResult) -> Verdict:
    """After the cell torch is clean: no held slots, hooks or modes; RNG restored.

    Capture documents that it snapshots the global RNG streams and restores
    them on every exit path, so every TorchLens stage (trace, save, load,
    validate) must leave the torch, ``random`` and NumPy streams exactly as it
    found them, pass or fail. Wrappers install lazily on the first capture and
    stay installed by design, so the oracle accepts an install but never an
    uninstall or a swap.
    """

    if result.after is None:
        return Verdict.not_applicable("cell never finished")
    problems = _snapshot_differences(result.before, result.after)
    for stage, streams in result.rng_changed.items():
        problems.append(f"{stage} changed the global RNG state of {streams}")
    if problems:
        return Verdict.failed("; ".join(problems))
    return Verdict.passed()


def peak_memory_within_budget(result: CellResult) -> Verdict:
    """Python-allocation peak and resident high-water growth stay under the row's budget."""

    if result.tracemalloc_peak_bytes is None or result.maxrss_growth_bytes is None:
        return Verdict.not_applicable("no memory reading")
    budget = result.memory_budget_bytes
    mib = 1024 * 1024
    problems = []
    if result.tracemalloc_peak_bytes > budget:
        problems.append(
            f"tracemalloc peak {result.tracemalloc_peak_bytes / mib:.0f} MiB over "
            f"{budget / mib:.0f} MiB"
        )
    if result.maxrss_growth_bytes > budget:
        problems.append(
            f"resident high-water grew {result.maxrss_growth_bytes / mib:.0f} MiB over "
            f"{budget / mib:.0f} MiB"
        )
    if problems:
        return Verdict.failed("; ".join(problems))
    return Verdict.passed(
        f"tracemalloc peak {result.tracemalloc_peak_bytes / mib:.1f} MiB, "
        f"rss growth {result.maxrss_growth_bytes / mib:.1f} MiB"
    )


ORACLES: tuple[tuple[str, Callable[[CellResult], Verdict]], ...] = (
    ("replay_equals_eager", replay_equals_eager),
    ("invariants_pass", invariants_pass),
    ("save_load_equals_live", save_load_equals_live),
    ("typed_refusal_names_condition", typed_refusal_names_condition),
    ("torch_clean_after_cell", torch_clean_after_cell),
    ("peak_memory_within_budget", peak_memory_within_budget),
)
"""The six oracles in report order; ``test_stress.py`` has one test per entry."""
