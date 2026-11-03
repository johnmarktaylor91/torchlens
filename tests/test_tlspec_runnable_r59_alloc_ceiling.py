"""r59 whole-class immunizer -- aggregate replay resource ceiling (allocation-DoS close).

r57 bounded per-op allocation BYTES via a ``FakeTensorMode`` projection. r58 found two
seams the byte budget is structurally blind to:

* free_1 -- ``tensor_split(x, N)`` drives N *empty* output tensors (~0 bytes) off ONE int
  literal; the projection itself builds the N-tensor fake tree, so the DEFENCE executes the
  attack (linear-in-N CPU/memory before any refusal).
* free_2 -- an ``expand``/``broadcast_to`` size literal inflated with an HONEST small
  recorded slot: the projection correctly charges a view 0 new bytes, but the framework's
  own bind-time ``value.detach().clone()`` re-materializes the huge logical view -> OOM.

Plus an ADJACENT accumulation seam: K honest, individually-feasible output slots whose SUM
of retained clone bytes exhausts the host (loaded replay retains an ``Op.out`` clone of every
taken-path op output for the whole run) -- a guaranteed mid-replay OOM every per-op gate passes.

r59 closes the class STRUCTURALLY with an aggregate resource admission -- three hard gates,
op-agnostic (no arity-estimator registry, which would be the r55 allowlist reborn):

1. output COUNT -- a count-instrumented ``FakeTensorMode`` caps realized fake outputs at
   ``max(recorded * 8, 4096)`` DURING fanout (so a huge N cannot self-DoS the projection),
   plus a bind-time aggregate accountant backstop. Refused typed at ``op_output_count_preflight``.
2. re-materialization BYTES -- every op-output snapshot clone routes through
   ``guarded_clone``, refusing typed at ``clone_allocation_preflight`` before it allocates.
3. run-prep RETENTION FLOOR -- the SUM of guaranteed-retained op-output clone bytes per device
   vs the never-under-estimating budget; refused typed at ``run_retention_preflight`` only when
   a guaranteed OOM is proved (zero false refusals).

All three reuse ``RUN_CAPABILITY_UNAVAILABLE`` with new ``detection_stage`` strings -- NO new
enum member. Over-trigger pins: legit many-output ops, decomposing ops (whose fake-count is
high), a genuinely huge honest view, ``nonzero`` fail-open, and a benign model all run clean.
"""

from __future__ import annotations

import json
import math
import resource
import time
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
import torchlens._runnable.call_arguments
import torchlens._runnable.call_outputs
import torchlens._runnable.state as rstate
from torchlens._runnable.call_arguments import _preflight_call_allocation
from torchlens._runnable.state import (
    _OUTPUT_COUNT_FLOOR,
    RunResourceCeiling,
    _torch_dtype,
)
from torchlens.errors import RunCapabilityUnavailableError

_CAPTURE = {"capture": tl.options.CaptureOptions(replayable=True)}


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #


def _build(
    tmp_path: Path, name: str, model: nn.Module, x: torch.Tensor, **trace_kwargs: Any
) -> Path:
    trace = tl.trace(model.eval(), x, **_CAPTURE, **trace_kwargs)
    bundle = tmp_path / name
    trace.save(str(bundle), level="runnable", include_weights=True)
    return bundle


def _tamper_literal(bundle: Path, kind: str, target: Any, replacement: Any) -> int:
    """Replace the first ``(kind, value == target)`` literal atom in any call's tree.

    Edits ONLY the plaintext literal; the recorded output slot shape is left honest --
    exactly the literal-only tamper the recorded-output bound cannot catch.
    """

    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    count = 0

    def _walk(node: Any) -> bool:
        nonlocal count
        if isinstance(node, dict):
            if node.get("kind") == kind and node.get("value") == target:
                node["value"] = replacement
                count += 1
                return True
            return any(_walk(value) for value in node.values())
        if isinstance(node, list):
            return any(_walk(item) for item in node)
        return False

    for call in manifest["run"]["calls"]:
        for literal in call.get("literal_arguments", []):
            if _walk(literal):
                break
    path.write_text(json.dumps(manifest))
    return count


@contextmanager
def _rlimit_cap(extra: int = 1 << 30) -> Iterator[None]:
    """Cap address space so an un-refused bomb raises MemoryError, not an OOM-kill.

    A refusal fires BEFORE any allocation, so under this cap a refused call raises the
    typed ``RunCapabilityUnavailableError`` while an un-refused bomb would surface as an
    allocator failure -- asserting the former (not a MemoryError) proves NO allocation.
    """

    with open("/proc/self/status", encoding="ascii") as handle:
        vmsize_kb = next(int(line.split()[1]) for line in handle if line.startswith("VmSize"))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (vmsize_kb * 1024 + extra, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def _stub_call(n_outputs: int = 1) -> Any:
    return types.SimpleNamespace(
        call_id="call:test",
        op_labels=tuple(f"op:{i}" for i in range(max(n_outputs, 1))),
        output_slot_ids=tuple(f"slot:{i}" for i in range(n_outputs)),
    )


def _descriptor(trace: Any) -> Any:
    return trace._runnable.descriptor


def _expected_retention_floor(descriptor: Any) -> int:
    """Recompute the retention floor from the descriptor (one clone per op label + output)."""

    slots = {slot.slot_id: slot for slot in descriptor.tensor_slots}
    total = 0
    for call in descriptor.calls:
        for slot_id in call.output_slot_ids:
            slot = slots.get(slot_id)
            if slot is not None:
                total += math.prod(slot.shape) * _torch_dtype(slot.dtype).itemsize
    for slot in descriptor.tensor_slots:
        if slot.role.value == "output":
            total += math.prod(slot.shape) * _torch_dtype(slot.dtype).itemsize
    return total


# --------------------------------------------------------------------------- #
# models                                                                        #
# --------------------------------------------------------------------------- #


class _TensorSplit(nn.Module):
    """A benign ``tensor_split(x, 2)`` -- one int literal drives the output count."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = torch.tensor_split(x, 2, dim=0)
        return parts[0].sum() + parts[1].sum() + x.sum()


class _ExpandView(nn.Module):
    """A benign ``expand`` view (logical > storage): free_2 tamper vector."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (64, 4)
        wide = x[:1].expand(64, 4)
        return wide.sum() + x.sum()


class _AddChain(nn.Module):
    """A deep single-output chain: many retained op-output clones, all small (gate-4 shape)."""

    def __init__(self, depth: int) -> None:
        super().__init__()
        self._depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self._depth):
            x = x + 1.0
        return x


class _AddChainSum(_AddChain):
    """The same chain reduced by a final ``sum``, so ``save=tl.func("sum")`` saves no add."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).sum()


class _Chunk(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (128, 4)
        parts = torch.chunk(x, 128, dim=0)
        return sum(p.sum() for p in parts) + x.sum()


class _Unbind(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (256, 4)
        parts = torch.unbind(x, dim=0)
        return sum(p.sum() for p in parts) + x.sum()


class _Interpolate(nn.Module):
    """A decomposing op: fake projection charges far more than 1 fake per recorded output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (1, 1, 8)
        import torch.nn.functional as F

        return F.interpolate(x, scale_factor=2.0, mode="nearest").sum() + x.sum()


class _Nonzero(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.nonzero(x > 0.0)
        return idx.float().sum() + x.sum()


class _HugeView(nn.Module):
    """A genuinely huge honest view whose materialized clone is still within budget."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (1, 4)
        wide = x.expand(10**7, 4)  # 4e7 elems ~= 160 MB clone; honest, in-budget
        return wide[0].sum() + x.sum()


# --------------------------------------------------------------------------- #
# (1) free_1 exact -- output-COUNT tamper refuses typed, bounded, no N-tree      #
# --------------------------------------------------------------------------- #


# Kept: sole branch coverage of _runnable/call_arguments.py, _runnable/state_context.py; the catch of its regression fault.
def test_tensor_split_count_bomb_refused_typed_and_bounded(tmp_path: Path) -> None:
    """A ``tensor_split`` count literal tamper (2 -> 3e5) refuses typed at
    ``op_output_count_preflight`` -- NOT ``PathDivergenceError`` -- fast (<< r58's 32.7s),
    building neither a real nor a full-fake N-tensor tree."""

    x = torch.randn(8)
    bundle = _build(tmp_path, "ts.tlspec", _TensorSplit(), x)
    assert _tamper_literal(bundle, "int", 2, 300000) >= 1
    loaded = tl.load(str(bundle))
    start = time.perf_counter()
    with pytest.raises(RunCapabilityUnavailableError) as caught:
        loaded.run(inputs=x.clone())
    elapsed = time.perf_counter() - start
    assert caught.value.fields.get("detection_stage") == "op_output_count_preflight"
    assert elapsed < 10.0, f"count refusal took {elapsed:.2f}s -- should abort at ceiling+1"


# --------------------------------------------------------------------------- #
# (2) axis-A aggregate accountant -- no literal to fire on -> bind backstop       #
# --------------------------------------------------------------------------- #


# Kept: sole branch coverage of _runnable/state.py.
def test_aggregate_accountant_refuses_realized_over_ceiling() -> None:
    """The bind-time accountant refuses typed when aggregate realized leaves exceed the
    ceiling -- the projection-skipped backstop (e.g. a fanout with no numeric literal)."""

    descriptor = types.SimpleNamespace(calls=[types.SimpleNamespace(output_slot_ids=("a",))])
    ceiling = RunResourceCeiling(descriptor)  # type: ignore[arg-type]
    assert ceiling.aggregate_output_count_ceiling() == _OUTPUT_COUNT_FLOOR
    call = _stub_call()
    with pytest.raises(RunCapabilityUnavailableError) as caught:
        ceiling.charge_realized_outputs(call, _OUTPUT_COUNT_FLOOR + 1)
    assert caught.value.fields.get("detection_stage") == "op_output_count_preflight"


# --------------------------------------------------------------------------- #
# (3) free_2 exact -- view literal tamper refuses at clone guard, no clone        #
# --------------------------------------------------------------------------- #


# Kept: sole branch coverage of _runnable/call_outputs.py, _runnable/transaction.py.
@pytest.mark.parametrize("sentinel,replacement", [(64, 10**11)])
def test_expand_view_tamper_refused_at_clone_guard(
    tmp_path: Path, sentinel: int, replacement: int
) -> None:
    """An ``expand`` size literal inflated (honest small slot) refuses typed at
    ``clone_allocation_preflight`` under an address-space cap -- NO clone materializes."""

    x = torch.randn(64, 4)
    bundle = _build(tmp_path, "expand.tlspec", _ExpandView(), x)
    assert _tamper_literal(bundle, "int", sentinel, replacement) >= 1
    with _rlimit_cap():
        loaded = tl.load(str(bundle))
        with pytest.raises(RunCapabilityUnavailableError) as caught:
            loaded.run(inputs=x.clone())
    assert caught.value.fields.get("detection_stage") == "clone_allocation_preflight"
    assert int(caught.value.fields["required_bytes"]) > int(caught.value.fields["available_bytes"])


# --------------------------------------------------------------------------- #
# (5) gate-4 retention floor -- P-A3 shape: per-slot/count/clone pass, SUM fails  #
# --------------------------------------------------------------------------- #


# Kept: sole branch coverage of _runnable/retention.py, _runnable/state.py.
def test_retention_floor_refuses_cumulative_over_budget_when_nothing_is_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same accumulation shape from a capture that saved none of the chain's ops still
    refuses at ``run_retention_preflight``: every op output stays alive in the run's slot
    values until the run ends, so the floor charges it whether or not it was saved."""

    x = torch.randn(1024)
    bundle = _build(tmp_path, "chain_unsaved.tlspec", _AddChainSum(30), x, save=tl.func("sum"))
    descriptor = _descriptor(tl.load(str(bundle)))
    floor = _expected_retention_floor(descriptor)
    per_slot_max = max(
        math.prod(s.shape) * _torch_dtype(s.dtype).itemsize
        for s in descriptor.tensor_slots
        if s.role.value in {"intermediate", "output"}
    )
    budget = floor - 1
    assert per_slot_max < budget < floor
    monkeypatch.setattr(rstate, "_allocation_budget_bytes", lambda device: budget)

    loaded = tl.load(str(bundle))
    with pytest.raises(RunCapabilityUnavailableError) as caught:
        loaded.run(inputs=x.clone())
    assert caught.value.fields.get("detection_stage") == "run_retention_preflight"
    assert int(caught.value.fields["required_bytes"]) == floor


# --------------------------------------------------------------------------- #
# (6) over-trigger pins -- legit ops / views / models must NOT be over-refused    #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# (8) section 2.4 -- allocation-classed projection fails CLOSED; others fail open #
# --------------------------------------------------------------------------- #


# Kept: sole branch coverage of _runnable/call_arguments.py, _runnable/state_context.py.
def test_projection_allocator_death_fails_closed_typed() -> None:
    """A projection that raises an allocation-classed error (the C++ prelude dying of
    ``std::bad_alloc`` at int-limit N) fails CLOSED at ``op_allocation_preflight`` -- failing
    open would just run the identical death twice."""

    def _mem_boom(n: int) -> torch.Tensor:
        raise MemoryError("boom")

    def _bad_alloc(n: int) -> torch.Tensor:
        raise RuntimeError("std::bad_alloc")

    for func in (_mem_boom, _bad_alloc):
        with pytest.raises(RunCapabilityUnavailableError) as caught:
            _preflight_call_allocation(None, func, [10], {}, _stub_call())
        assert caught.value.fields.get("detection_stage") == "op_allocation_preflight"


# --------------------------------------------------------------------------- #
# (9) source meta-tests -- no raw clone escapes the guard; sentinel ordering      #
# --------------------------------------------------------------------------- #
