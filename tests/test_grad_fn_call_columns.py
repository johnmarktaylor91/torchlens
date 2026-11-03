"""Persisted grad_fn calls cost opcodes per owner, not per call (columnar ``GradFn.calls``).

A portable save pickles the ``GradFnCall`` records of one ``GradFn`` as one columnar table, and
each ``BackwardPass`` lists its calls as packed references into those tables, so the
``metadata.pkl`` opcode count no longer grows with grad_fns times backward passes (a Mamba-130m
trace with 96 Jacobian passes and 457,344 calls passed the loader's opcode ceiling).
"""

from __future__ import annotations

import pickletools
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.grad_fn import GradFnCallAccessor
from torchlens.options import CaptureOptions


def _mlp() -> nn.Module:
    """Return a seeded three-layer MLP."""

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))


def _trace_with_passes(num_passes: int) -> tl.Trace:
    """Capture the MLP with saved grads and run ``num_passes`` backward passes over it."""

    x = torch.randn(1, 8, requires_grad=True)
    trace = tl.trace(_mlp(), x, save="all", capture=CaptureOptions(save_grads="all"))
    out = trace[trace.output_layers[0]].out
    for _ in range(num_passes):
        torch.autograd.grad(out.sum(), x, retain_graph=True)
    return trace


def _metadata_opcodes(trace: tl.Trace, path: Path) -> int:
    """Save ``trace`` portable at ``path`` and count the opcodes of its ``metadata.pkl``."""

    trace.save(path, level="portable")
    return sum(1 for _ in pickletools.genops((path / "metadata.pkl").read_bytes()))


def _call_opcodes(trace: tl.Trace, directory: Path) -> tuple[int, int, int]:
    """Return the opcodes the grad_fn calls cost in ``metadata.pkl``, the calls and the grad_fns.

    The trace is saved with its calls and again after emptying every ``GradFn.calls`` and
    ``BackwardPass.grad_fn_calls``. Each pass's counted-flops record is pinned first, so both
    saves carry the same one; every other fact is in both saves, so the difference is the
    encoding of the calls alone.
    """

    grad_fns = list(trace._grad_fn_logs.values())
    passes = list(trace._backward_pass_logs.values())
    num_calls = sum(len(grad_fn.calls) for grad_fn in grad_fns)
    for backward_pass in passes:
        if backward_pass._executed_flops_record is None:
            backward_pass._executed_flops_record = backward_pass._census_record()
    with_calls = _metadata_opcodes(trace, directory / "with.tlspec")
    for grad_fn in grad_fns:
        grad_fn.calls = GradFnCallAccessor({}, grad_fn.label)
    for backward_pass in passes:
        backward_pass.grad_fn_calls = []
    without_calls = _metadata_opcodes(trace, directory / "without.tlspec")
    return with_calls - without_calls, num_calls, len(grad_fns)


def test_grad_fn_call_opcodes_scale_by_owner_not_by_call(tmp_path: Path) -> None:
    """Thirty-eight more backward passes add a constant per pass, not opcodes per call."""

    few, many = _trace_with_passes(2), _trace_with_passes(40)
    (tmp_path / "few").mkdir()
    (tmp_path / "many").mkdir()
    few_cost, few_calls, _ = _call_opcodes(few, tmp_path / "few")
    many_cost, many_calls, num_grad_fns = _call_opcodes(many, tmp_path / "many")
    extra_passes = 38
    assert many_calls - few_calls >= extra_passes * num_grad_fns, "every pass must fire the nodes"
    bound = 4 * num_grad_fns + 32 * extra_passes
    assert many_cost - few_cost < bound, (
        f"grad_fn-call opcodes grew {many_cost - few_cost} ({few_cost} -> {many_cost}) for "
        f"{many_calls - few_calls} more calls over {num_grad_fns} grad_fns; bound {bound}"
    )


def _materialized(value: Any) -> Any:
    """Return a saved payload with every lazy reference loaded."""

    if isinstance(value, tuple):
        return tuple(_materialized(item) for item in value)
    if isinstance(value, list):
        return [_materialized(item) for item in value]
    if hasattr(value, "materialize") and not isinstance(value, torch.Tensor):
        return value.materialize()
    return value


def _payload_equal(live: Any, loaded: Any) -> bool:
    """Compare a live saved payload with its loaded copy (tensors by ``torch.equal``)."""

    loaded = _materialized(loaded)
    if isinstance(live, torch.Tensor):
        return isinstance(loaded, torch.Tensor) and torch.equal(live, loaded)
    if isinstance(live, (tuple, list)):
        return (
            type(live) is type(loaded)
            and len(live) == len(loaded)
            and all(_payload_equal(a, b) for a, b in zip(live, loaded))
        )
    return live == loaded


_CALL_FACTS = (
    "call_index",
    "ordinal",
    "backward_pass_index",
    "grad_fn_label",
    "timestamp",
    "_time_started",
    "_time_finished",
    "fire_record",
)


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_grad_fn_calls_round_trip(tmp_path: Path, lazy: bool) -> None:
    """Every call's facts, saved gradients and shared identity survive a portable save."""

    trace = _trace_with_passes(3)
    path = tmp_path / "trace.tlspec"
    trace.save(path, level="portable")
    loaded = tl.load(path, lazy=lazy)
    live_calls = dict(trace.autograd.grad_fn_calls.items())
    loaded_calls = dict(loaded.autograd.grad_fn_calls.items())
    assert list(loaded_calls) == list(live_calls)
    saved = 0
    for label, live in live_calls.items():
        call = loaded_calls[label]
        for fact in _CALL_FACTS:
            assert getattr(call, fact) == getattr(live, fact), (label, fact)
        for payload in ("grad_inputs", "grad_outputs"):
            assert _payload_equal(getattr(live, payload), getattr(call, payload)), (label, payload)
        saved += live.is_saved
    assert saved > 0, "the round trip must cover saved gradient payloads"
    by_identity = {id(call) for call in loaded_calls.values()}
    owned = [call for grad_fn in loaded.autograd.grad_fns for call in grad_fn.calls.values()]
    assert {id(call) for call in owned} == by_identity
    for grad_fn in loaded.autograd.grad_fns:
        assert all(
            call is loaded_calls[f"{grad_fn.label}:{k}"] for k, call in grad_fn.calls.items()
        )
    for pass_index, live_pass in trace._backward_pass_logs.items():
        loaded_pass = loaded.autograd.backward_passes.for_pass(pass_index)
        assert [call.label for call in loaded_pass.grad_fn_calls] == [
            call.label for call in live_pass.grad_fn_calls
        ]
        assert all(call is loaded_calls[call.label] for call in loaded_pass.grad_fn_calls)


def test_trace_without_backward_never_names_the_tables(tmp_path: Path) -> None:
    """A trace without a backward pass persists its grad_fns exactly as before."""

    from torchlens._io._grad_fn_call_columns import reduce_with_grad_fn_call_columns

    trace = tl.trace(_mlp(), torch.randn(1, 8, requires_grad=True), save="all")
    for grad_fn in trace._grad_fn_logs.values():
        assert len(grad_fn.calls) == 0
        assert reduce_with_grad_fn_call_columns(None, grad_fn, type(grad_fn), 5) is NotImplemented
    path = tmp_path / "plain.tlspec"
    trace.save(path, level="portable")
    metadata = (path / "metadata.pkl").read_bytes()
    assert b"GradFnCallColumns" not in metadata
    assert b"GradFnCallRefs" not in metadata
    loaded = tl.load(path)
    assert all(len(grad_fn.calls) == 0 for grad_fn in loaded._grad_fn_logs.values())
