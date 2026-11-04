"""TorchLens's own tensor reads during capture skip the per-op wrapper.

Capture's bookkeeping (storage identity, element counts, layout flags) reads tensor
metadata under ``pause_logging``; each read used to re-enter the per-op wrapper only to
take its logging-off fast path. These tests pin that:

* every routed read name has a wrapper whose fast path is a pure pass-through (read
  from the installed wrapper's closure, the predicate the wrapper itself evaluates);
* during a capture no paused wrapper entry is spent on a routed read, and capture makes
  fewer Python calls per op than with the routing switched off;
* the routing honours every gate (logging on, a fast-run collector, an uncounted
  fingerprint call, a tensor subclass, an unwrapped class) by returning the old spelling;
* the logged ops are field-for-field identical with the routing on and off across the
  classics smoke corpus and the bound-generate toys (under a fake clock, so timing cells
  compare too).
"""

from __future__ import annotations

import hashlib
import itertools
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens._postprocess as postprocess_pkg
from torchlens import _state
from torchlens.backends.torch import wrappers

# The pure reads capture's bookkeeping makes under pause_logging (census of a GPT-2
# generate capture): kept literal here so the counter test fails on a tree without routing.
_EXPECTED_ROUTED = frozenset(
    {
        "data_ptr",
        "element_size",
        "is_conj",
        "is_contiguous",
        "is_inference",
        "is_neg",
        "nelement",
        "numel",
        "storage_offset",
        "untyped_storage",
    }
)
_ADDR = re.compile(r"0x[0-9a-fA-F]+")


class _NotATensor:
    """A receiver whose class is neither ``torch.Tensor`` nor ``torch.nn.Parameter``."""

    def numel(self) -> int:
        """Answer like a one-element tensor."""

        return 1


def _wrapped_code() -> Any:
    """Return the code object every ``torch_func_decorator`` wrapper shares."""

    for const in wrappers.torch_func_decorator.__code__.co_consts:
        if getattr(const, "co_name", None) == "wrapped_func":
            return const
    raise AssertionError("wrapped_func code object not found")


def _chain() -> nn.Module:
    """A 12-block Linear/ReLU chain (deterministic weights)."""

    torch.manual_seed(0)
    layers: list[nn.Module] = []
    for _ in range(12):
        layers.extend((nn.Linear(4, 4), nn.ReLU()))
    return nn.Sequential(*layers).eval()


def _capture_census(model: nn.Module, x: torch.Tensor) -> tuple[Counter[str], int, int]:
    """Trace ``model`` and count paused wrapper entries by name and Python calls in capture.

    Returns
    -------
    tuple[Counter[str], int, int]
        Paused wrapper entries per recorded name, Python calls during capture, op count.
    """

    code = _wrapped_code()
    state = {"on": False}
    paused: Counter[str] = Counter()
    calls = [0]
    real = postprocess_pkg.postprocess

    def hook(frame: Any, event: str, arg: Any) -> None:
        """Count capture-phase Python calls; classify wrapper entries."""

        if not state["on"] or event != "call":
            return
        calls[0] += 1
        if frame.f_code is code and not _state._logging_enabled:
            if _state._active_trace is not None:
                paused[frame.f_locals.get("func_name")] += 1

    def stop_at_postprocess(*args: Any, **kwargs: Any) -> Any:
        """Stop counting where postprocess begins."""

        state["on"] = False
        return real(*args, **kwargs)

    postprocess_pkg.postprocess = stop_at_postprocess
    try:
        state["on"] = True
        sys.setprofile(hook)
        try:
            trace = tl.trace(model, x)
        finally:
            sys.setprofile(None)
            state["on"] = False
    finally:
        postprocess_pkg.postprocess = real
    return paused, calls[0], len(trace.ops)


def test_routed_names_are_the_census_names() -> None:
    """The routing table holds exactly the census read names."""

    from torchlens.backends.torch._internal_originals import ROUTED_TENSOR_READS

    assert ROUTED_TENSOR_READS == _EXPECTED_ROUTED


@pytest.mark.parametrize("name", sorted(_EXPECTED_ROUTED))
def test_routed_read_wrapper_fast_path_is_a_pure_pass_through(name: str) -> None:
    """Each routed name's installed wrapper has no fast-path side effect."""

    tl.trace(_chain(), torch.zeros(2, 4))  # wraps torch
    wrapper = getattr(torch.Tensor, name)
    ledger = _state.wrap_epoch_ledgers()[0]
    assert id(wrapper) in ledger, f"Tensor.{name} is not a TorchLens wrapper"
    cells = dict(
        zip(
            wrapper.__code__.co_freevars,
            (cell.cell_contents for cell in wrapper.__closure__ or ()),
            strict=True,
        )
    )
    assert wrapper.__code__ is _wrapped_code()
    pure = {
        "mutates_receiver": False,
        "is_mutating_property_setter": False,
        "reconstructs_receiver_output": False,
        "inplace_param_index": None,
        "needs_device_injection": False,
        "_fast_path_subclass_cls_index": None,
        "is_detached_propagation_func": False,
    }
    assert {key: cells[key] for key in pure} == pure


def test_capture_spends_no_wrapper_entry_on_routed_reads() -> None:
    """No paused wrapper entry is spent on a routed read during capture."""

    paused, _calls, ops = _capture_census(_chain(), torch.zeros(2, 4))
    assert ops > 0
    routed = {name: count for name, count in paused.items() if name in _EXPECTED_ROUTED}
    assert routed == {}, f"paused wrapper entries on routed reads: {routed}"


def test_capture_python_calls_per_op_drop_with_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture makes fewer Python calls per op with the routing on than off."""

    from torchlens.backends.torch import _internal_originals

    paused_on, calls_on, ops_on = _capture_census(_chain(), torch.zeros(2, 4))
    monkeypatch.setattr(_internal_originals, "_route_open", lambda tensor: False)
    paused_off, calls_off, ops_off = _capture_census(_chain(), torch.zeros(2, 4))
    assert ops_on == ops_off
    skipped = sum(paused_off[name] for name in _EXPECTED_ROUTED)
    assert skipped > 0  # the closed routing re-enters the wrapper (not vacuous)
    # Every routed read no longer enters the wrapper: one Python call saved per entry.
    assert calls_on <= calls_off - skipped, (calls_on, calls_off, skipped)
    assert calls_on / ops_on < calls_off / ops_off


def test_gates_fall_back_to_the_method_call_spelling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every gate that would make the wrapper do more than pass through keeps it."""

    from torchlens.backends.torch import _internal_originals as io

    tl.trace(_chain(), torch.zeros(2, 4))  # wraps torch
    t = torch.zeros(3)
    original = _state.wrap_epoch_ledgers()[0][id(torch.Tensor.numel)]
    with _state.pause_logging():
        assert io.tensor_read(t, "numel") is original
        assert io.tensor_read(nn.Parameter(t), "numel") is original
        assert io.tensor_reads(t, "numel", "is_contiguous")[0] is original

        assert io.tensor_read(_NotATensor(), "numel") is io._SPELLED["numel"]
        monkeypatch.setattr(_state, "_active_fast_run_collector", object())
        assert io.tensor_read(t, "numel") is io._SPELLED["numel"]
        monkeypatch.setattr(_state, "_active_fast_run_collector", None)
        monkeypatch.setattr(_state, "_call_fingerprint", object())
        monkeypatch.setattr(_state, "_pause_depth", 0)
        assert io.tensor_reads(t, "numel")[0] is io._SPELLED["numel"]
        monkeypatch.setattr(_state, "_pause_depth", 1)
        assert io.tensor_read(t, "numel") is original
        monkeypatch.setattr(_state, "_call_fingerprint", None)
        monkeypatch.setattr(torch.Tensor, "numel", original)  # a class not wrapped
        assert io.tensor_read(t, "numel") is io._SPELLED["numel"]
    monkeypatch.undo()
    monkeypatch.setattr(_state, "_logging_enabled", True)
    assert io.tensor_read(t, "numel") is io._SPELLED["numel"]
    monkeypatch.undo()
    assert io._SPELLED["untyped_storage"](t).data_ptr() == t.untyped_storage().data_ptr()


def _canon(value: Any) -> str:
    """Return a run-independent text form of one store cell."""

    from torchlens._trace_core.op_store import PooledCell

    if value.__class__ is PooledCell:
        return "P:" + _canon(value.hydrate())
    if isinstance(value, torch.Tensor):
        try:
            data = value.detach().cpu().contiguous()
            digest = hashlib.sha1(data.numpy().tobytes()).hexdigest()[:16]
        except (RuntimeError, TypeError, ValueError) as exc:
            digest = f"nodata:{type(exc).__name__}"
        return f"T:{value.dtype}:{tuple(value.shape)}:{digest}"
    try:
        raw = repr(value)
    except Exception as exc:  # a released husk's repr reads freed fields
        raw = f"<unrepr {type(value).__name__} {type(exc).__name__}>"
    return f"{type(value).__name__}:{_ADDR.sub('0x', raw)}"


def _op_fields(build: Callable[[], tuple[Any, Any, dict[str, Any]]]) -> dict[str, list[str]]:
    """Trace one corpus entry and return each op-store field's cells in row order."""

    from torchlens._trace_core.op_store import _MISSING

    fn, inputs, kwargs = build()  # built before the fake clock
    with pytest.MonkeyPatch.context() as clock:
        _install_fake_clock(clock)
        torch.manual_seed(0)
        trace = tl.trace(fn, inputs, input_kwargs=kwargs or None)
        store = object.__getattribute__(trace.ops[0], "_core")
        rows = getattr(store, "_n_rows", None) or len(store)
        fields: dict[str, list[str]] = {}
        for fid, name in enumerate(store.layout.names):
            cells = []
            for row in range(rows):
                value = store.cell_get(row, fid)
                cells.append("-" if value is _MISSING else _canon(value))
            fields[name] = cells
        trace.cleanup()
    return fields


def _install_fake_clock(clock: pytest.MonkeyPatch) -> None:
    """Replace the ``time`` clocks with a fresh counter so timing cells are reproducible."""

    ticks = itertools.count(1)

    def tick() -> float:
        """Return the next fake second."""

        return 1_000_000.0 + next(ticks) * 0.001

    def tick_ns() -> int:
        """Return the next fake second in nanoseconds."""

        return int(tick() * 1e9)

    for name in ("time", "perf_counter", "monotonic", "process_time", "thread_time"):
        clock.setattr(time, name, tick)
    for name in ("time_ns", "perf_counter_ns", "monotonic_ns", "process_time_ns", "thread_time_ns"):
        clock.setattr(time, name, tick_ns)


def _corpus() -> list[tuple[str, Callable[[], tuple[Any, Any, dict[str, Any]]]]]:
    """The classics smoke corpus plus the bound-generate toys, as builders."""

    from classics_corpus._loader import build_entry, corpus_entries

    def classic(entry: Any) -> Callable[[], tuple[Any, Any, dict[str, Any]]]:
        """Builder for one classics entry."""

        def build() -> tuple[Any, Any, dict[str, Any]]:
            """Build the entry's model and inputs."""

            model, inputs = build_entry(entry)
            return model, inputs, {}

        return build

    entries = [(f"classic:{e.module}", classic(e)) for e in corpus_entries("smoke")]
    entries.append(("toy:cached_model", _cached_model))
    entries.append(("toy:generator", _toy_generator))
    return entries


def _cached_model() -> tuple[Any, Any, dict[str, Any]]:
    """The bound-generate test's cached model with a logging cache property."""

    import test_bound_generate_trace as toys

    return toys._CachedModel(), torch.zeros(1, 8), {"cache": toys._CacheWithLoggedProperty()}


def _toy_generator() -> tuple[Any, Any, dict[str, Any]]:
    """The bound-generate test's toy ``generate`` loop."""

    import test_bound_generate_trace as toys

    return toys._ToyGenerator().eval().generate, toys._ids(), {"n_new": toys.N_NEW}


@pytest.mark.parametrize("spec", [spec for spec, _ in _corpus()])
def test_logged_ops_identical_with_and_without_routing(
    spec: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every run-stable op-store field is identical with the routing on and off."""

    from torchlens.backends.torch import _internal_originals

    build = dict(_corpus())[spec]
    with monkeypatch.context() as closed:
        closed.setattr(_internal_originals, "_route_open", lambda tensor: False)
        base_a = _op_fields(build)
        base_b = _op_fields(build)
    routed = _op_fields(build)
    assert set(routed) == set(base_a)
    stable = [name for name in base_a if base_a[name] == base_b[name]]
    assert len(stable) >= len(base_a) // 2, "too few run-stable fields to compare"
    changed = [name for name in stable if routed[name] != base_a[name]]
    assert changed == [], f"{spec}: fields differ with routing on: {changed}"
