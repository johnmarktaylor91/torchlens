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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest
import torch
import torch.nn as nn

import torchlens as tl
import torchlens._runnable_execution as rex
import torchlens._runnable_state as rstate
from torchlens._runnable_execution import (
    _ProjectionCountExceeded,
    _count_bounded_fake_tensor_mode_class,
    _preflight_call_allocation,
)
from torchlens._runnable_state import (
    _OUTPUT_COUNT_FLOOR,
    _OUTPUT_COUNT_MARGIN,
    RunResourceCeiling,
    _torch_dtype,
)
from torchlens.errors import RunCapabilityUnavailableError

pytestmark = pytest.mark.smoke

_CAPTURE = dict(intervention_ready=True)


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #


def _build(tmp_path: Path, name: str, model: nn.Module, x: torch.Tensor) -> Path:
    trace = tl.trace(model.eval(), x, **_CAPTURE)
    bundle = tmp_path / name
    tl.save(trace, str(bundle), level="runnable", include_weights=True)
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
    return trace.__dict__["_runnable_descriptor"]


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


def test_count_mode_aborts_during_fanout_not_post_return() -> None:
    """The count-bounded mode raises DURING dispatch fanout (not a post-return tree count),
    so a huge N aborts at ~ceiling fakes -- the projection cannot self-DoS."""

    mode_cls = _count_bounded_fake_tensor_mode_class()
    assert mode_cls is not None
    x = torch.randn(8)
    with pytest.raises(_ProjectionCountExceeded):
        with mode_cls(allow_non_fake_inputs=True, ceiling=4096) as mode:
            fx = mode.from_tensor(x)
            mode._tl_set_baseline()
            torch.tensor_split(fx, 10**7, dim=0)


# --------------------------------------------------------------------------- #
# (2) axis-A aggregate accountant -- no literal to fire on -> bind backstop       #
# --------------------------------------------------------------------------- #


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


def test_ceiling_constants_load_bearing_math() -> None:
    """FLOOR carries low-arity decompositions; MARGIN carries high-arity headroom."""

    assert _OUTPUT_COUNT_MARGIN == 8 and _OUTPUT_COUNT_FLOOR == 4096
    # high-arity: a legit unbind of 2048 gets a 16384 ceiling (MARGIN load-bearing)
    hi = _stub_call(2048)
    d = types.SimpleNamespace(calls=[hi])
    assert RunResourceCeiling(d).per_call_output_count_ceiling(hi) == 2048 * 8  # type: ignore[arg-type]
    # low-arity: FLOOR carries a 55-fake decomposition on a 1-output call
    lo = _stub_call(1)
    d2 = types.SimpleNamespace(calls=[lo])
    assert RunResourceCeiling(d2).per_call_output_count_ceiling(lo) == 4096  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# (3) free_2 exact -- view literal tamper refuses at clone guard, no clone        #
# --------------------------------------------------------------------------- #


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


def test_guarded_clone_refuses_over_budget_before_allocation() -> None:
    """``guarded_clone`` refuses typed BEFORE the materializing clone for an over-budget
    logical view, and clones an in-budget honest tensor normally."""

    d = types.SimpleNamespace(calls=[])
    ceiling = RunResourceCeiling(d)  # type: ignore[arg-type]
    base = torch.randn(4)
    with _rlimit_cap():
        huge_view = base[:1].expand(10**11, 4)  # 1.6TB logical, tiny storage
        with pytest.raises(RunCapabilityUnavailableError) as caught:
            ceiling.guarded_clone(huge_view, call_id="c", slot_id="s", affected_op_labels=())
        assert caught.value.fields.get("detection_stage") == "clone_allocation_preflight"
    small = ceiling.guarded_clone(base, call_id="c", slot_id="s", affected_op_labels=())
    assert torch.equal(small, base) and small.data_ptr() != base.data_ptr()


# --------------------------------------------------------------------------- #
# (5) gate-4 retention floor -- P-A3 shape: per-slot/count/clone pass, SUM fails  #
# --------------------------------------------------------------------------- #


def test_retention_floor_refuses_cumulative_over_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deep chain of honest, individually-feasible outputs whose SUM of retained clones
    exceeds a test-forced budget refuses typed at ``run_retention_preflight`` -- before any
    call executes -- while EACH single slot stays within budget (the P-A3 accumulation shape)."""

    x = torch.randn(1024)  # each op output = 4096 bytes
    bundle = _build(tmp_path, "chain.tlspec", _AddChain(30), x)
    loaded_clean = tl.load(str(bundle))
    assert loaded_clean.run(inputs=x.clone()).report.path_faithfulness.value == "verified"

    descriptor = _descriptor(tl.load(str(bundle)))
    floor = _expected_retention_floor(descriptor)
    per_slot_max = max(
        math.prod(s.shape) * _torch_dtype(s.dtype).itemsize
        for s in descriptor.tensor_slots
        if s.role.value in {"intermediate", "output"}
    )
    # budget above any single slot (per-slot + clone individually pass) but below the SUM.
    budget = floor - 1
    assert per_slot_max < budget < floor
    monkeypatch.setattr(rstate, "_allocation_budget_bytes", lambda device: budget)

    loaded = tl.load(str(bundle))
    with pytest.raises(RunCapabilityUnavailableError) as caught:
        loaded.run(inputs=x.clone())
    assert caught.value.fields.get("detection_stage") == "run_retention_preflight"
    assert int(caught.value.fields["required_bytes"]) == floor


def test_retention_floor_premise_every_op_label_retains_a_clone(tmp_path: Path) -> None:
    """Pin the gate-4 floor premise (P-A1): the descriptor charges one retained clone per
    taken-path op-output slot + one per model-output slot. If fork materialization ever
    stops retaining an op output, the floor proof re-opens here."""

    x = torch.randn(16)
    bundle = _build(tmp_path, "premise.tlspec", _AddChain(12), x)
    descriptor = _descriptor(tl.load(str(bundle)))
    op_output_slot_ids = [sid for call in descriptor.calls for sid in call.output_slot_ids]
    # every taken-path op label contributes exactly one recorded output slot to the floor
    assert len(op_output_slot_ids) >= 12
    slots = {s.slot_id: s for s in descriptor.tensor_slots}
    assert all(sid in slots for sid in op_output_slot_ids)
    assert _expected_retention_floor(descriptor) > 0


# --------------------------------------------------------------------------- #
# (6) over-trigger pins -- legit ops / views / models must NOT be over-refused    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,factory,x",
    [
        ("chunk128", _Chunk, torch.randn(128, 4)),
        ("unbind256", _Unbind, torch.randn(256, 4)),
        ("interpolate", _Interpolate, torch.randn(1, 1, 8)),
        ("tensor_split", _TensorSplit, torch.randn(8)),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_legit_many_output_and_decomposing_ops_run_verified(
    tmp_path: Path, name: str, factory: type[nn.Module], x: torch.Tensor
) -> None:
    """Legit many-output ops (chunk/unbind/tensor_split) and decomposing ops (interpolate,
    whose fake-count reaches ~55/recorded) replay VERIFIED -- no new-stage over-refusal."""

    bundle = _build(tmp_path, f"{name}.tlspec", factory(), x)
    result = tl.load(str(bundle)).run(inputs=x.clone())
    assert result.report.path_faithfulness.value == "verified"


def test_genuinely_huge_view_model_runs_verified(tmp_path: Path) -> None:
    """A real huge honest view (4e7-element ``expand``) whose materialized clone is within
    budget replays VERIFIED -- the clone guard does not over-refuse an in-budget view."""

    x = torch.randn(1, 4)
    bundle = _build(tmp_path, "hugeview.tlspec", _HugeView(), x)
    assert tl.load(str(bundle)).run(inputs=x.clone()).report.path_faithfulness.value == "verified"


def test_data_dependent_op_fails_open_and_runs(tmp_path: Path) -> None:
    """A legit ``nonzero`` (fake impl raises ``DynamicOutputShapeException``) still FAILS
    OPEN through the count-instrumented subclass and runs verified."""

    x = torch.randn(8)
    bundle = _build(tmp_path, "nonzero.tlspec", _Nonzero(), x)
    assert tl.load(str(bundle)).run(inputs=x.clone()).report.path_faithfulness.value == "verified"


def test_deep_model_within_budget_not_retention_refused(tmp_path: Path) -> None:
    """A legit deep model whose cumulative retention is within the real host budget runs
    VERIFIED -- the retention floor never fires on a completable run."""

    x = torch.randn(64)
    bundle = _build(tmp_path, "deep.tlspec", _AddChain(200), x)
    assert tl.load(str(bundle)).run(inputs=x.clone()).report.path_faithfulness.value == "verified"


# --------------------------------------------------------------------------- #
# (8) section 2.4 -- allocation-classed projection fails CLOSED; others fail open #
# --------------------------------------------------------------------------- #


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


def test_projection_non_allocation_error_fails_open() -> None:
    """A NON-allocation projection failure (a plain RuntimeError, or a data-dependent op)
    keeps failing open -- a legitimate op is never over-refused (r51 anti-pattern)."""

    def _plain_error(n: int) -> torch.Tensor:
        raise RuntimeError("some unrelated signature problem")

    # fail open -> returns None, no raise
    _preflight_call_allocation(None, _plain_error, [10], {}, _stub_call())
    _preflight_call_allocation(None, torch.nonzero, [torch.randn(4)], {}, _stub_call())


# --------------------------------------------------------------------------- #
# (9) source meta-tests -- no raw clone escapes the guard; sentinel ordering      #
# --------------------------------------------------------------------------- #


def test_no_raw_clone_on_op_output_bind_or_reconstruct_paths() -> None:
    """Every op-output snapshot clone in ``_bind_call_outputs`` / ``_reconstruct_output`` /
    ``_populate_source_slots`` routes through ``guarded_clone`` -- no raw ``detach().clone()``
    remains on those paths (a new unguarded clone site re-opens the free_2 class)."""

    import inspect

    for fn in (rex._bind_call_outputs, rex._reconstruct_output, rex._populate_source_slots):
        src = inspect.getsource(fn)
        assert "value.detach().clone()" not in src, (
            f"{fn.__name__} has a raw op-output clone outside guarded_clone"
        )
        assert "guarded_clone(" in src


def test_count_sentinel_precedes_generic_fail_open() -> None:
    """The count sentinel (and the allocation-classed carve-out) are handled BEFORE the
    generic fail-open in ``_preflight_call_allocation`` -- ordering is load-bearing."""

    import inspect

    src = inspect.getsource(_preflight_call_allocation)
    idx_sentinel = src.index("except _ProjectionCountExceeded")
    idx_alloc = src.index("_is_allocator_death")
    idx_return = src.rindex("return")
    assert idx_sentinel < idx_alloc, "count sentinel must precede the generic clause"
    # the fail-open `return` lives inside the generic clause, after the alloc carve-out
    assert idx_alloc < idx_return
