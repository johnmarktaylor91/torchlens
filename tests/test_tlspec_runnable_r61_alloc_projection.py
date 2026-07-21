"""r61 whole-class immunizer -- allocation projection completeness (free_1 + corr_2 close).

r57/r59 bounded per-op allocation via a ``FakeTensorMode`` projection, but GATED it on
``_has_numeric_literal``: a call with no int/float literal skipped the projection on the
premise "sizes coming only from live tensor shapes are already bounded". r60 falsified the
premise twice:

* free_1 -- no-literal SHAPE AMPLIFIERS (``outer``/``kron``/broadcast-``mul``/
  ``cartesian_prod``/``tensordot(dims=0)``/``einsum``/``diag``) produce outputs quadratically
  larger than their operands with NO literal anywhere; a tampered upstream literal makes the
  operands modest (each passing its own projection) and the amplifier's real allocation runs
  unguarded. The ZERO-NUMEL family (``mm``/``matmul``/``einsum`` on ``[N,0] @ [0,N]``) proves
  no numel/arity threshold can ever re-gate this: numel is a PRODUCT, so a 0 dim hides
  arbitrarily large sibling dims (149 GB output from two 0-byte operands at N=200k).
* corr_2 -- the accepted runtime INPUT MIRROR clone and the STATE re-materialization chain
  (binder staging, embedded/non-persistent bind, run-time state clone) materialize the
  LOGICAL extent of a view with no byte bound anywhere (user/embedded state never enters the
  run-prep representative sum; probe C: the strict binder accepts an expanded view).

r61 closes both STRUCTURALLY, no per-op patch:

1. The projection runs for EVERY taken-path call carrying a size source -- a numeric literal
   OR a tensor operand (``_projection_required_by_arguments``). The ONLY skip is a call with
   neither (no fake/meta kernel can size an output tree from it); every amplifier carries a
   tensor operand, so none can be pre-filtered out.
2. ONE ``RunResourceCeiling`` per transaction, constructed BEFORE input binding, and ONE
   module-level byte-guard core (``_byte_guarded_clone``) for every TorchLens-owned
   re-materialization clone: op-output snapshots, the runtime input mirror
   (``mirror_requires_grad=True`` keeps the r37 autograd-mirror rule), binder staging,
   embedded-state/non-persistent-buffer binds, and the run-time state clone. All refuse
   typed at the EXISTING ``clone_allocation_preflight`` stage -- NO new enum/error code.

Over-refusal pins: honest amplifier ops at feasible shapes stay VERIFIED, honest zero-numel
matmuls stay VERIFIED, honest expanded-view user state stays VERIFIED, honest differentiable
inputs keep ``requires_grad``.
"""

from __future__ import annotations

import inspect
import json
import resource
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
    _MAX_DECODE_NESTING_DEPTH,
    _has_tensor_operand,
    _preflight_call_allocation,
    _projection_required_by_arguments,
)
from torchlens._runnable_state import RunResourceCeiling, _byte_guarded_clone
from torchlens.errors import RunCapabilityUnavailableError
from torchlens.runnable import StateSource

pytestmark = pytest.mark.smoke

_CAPTURE = dict(intervention_ready=True)


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #


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


def _build(tmp_path: Path, name: str, model: nn.Module, x: torch.Tensor) -> Path:
    trace = tl.trace(model.eval(), x, **_CAPTURE)
    bundle = tmp_path / name
    tl.save(trace, str(bundle), level="runnable", include_weights=True)
    return bundle


def _tamper_literal(bundle: Path, kind: str, target: Any, replacement: Any) -> int:
    """Replace the first ``(kind, value == target)`` literal atom in any call's tree."""

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


def _tamper_input_slot_shape(bundle: Path, shape: list[int]) -> int:
    """Inflate every model_input tensor-slot shape (the r60 corr_2 vector)."""

    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    count = 0
    for slot in manifest["run"]["tensor_slots"]:
        if slot.get("role") == "model_input":
            slot["shape"] = list(shape)
            count += 1
    path.write_text(json.dumps(manifest))
    return count


def _tamper_slot_shapes(bundle: Path, old: list[int], new: list[int]) -> int:
    """Rewrite every tensor slot recording shape ``old`` to ``new`` (self-consistent
    tamper: the literal AND its downstream intermediate slots agree, so the per-call
    shape witness passes and only the amplifier's own projection can refuse)."""

    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    count = 0
    for slot in manifest["run"]["tensor_slots"]:
        if list(slot.get("shape", ())) == list(old):
            slot["shape"] = list(new)
            count += 1
    path.write_text(json.dumps(manifest))
    return count


def _stub_call(n_outputs: int = 1) -> Any:
    return types.SimpleNamespace(
        call_id="call:test",
        op_labels=tuple(f"op:{i}" for i in range(max(n_outputs, 1))),
        output_slot_ids=tuple(f"slot:{i}" for i in range(n_outputs)),
    )


_N = 32768  # dense amplifiers: operands ~128 KB, projected output 4+ GB
_Z = 200000  # zero-numel family: operands 0 bytes, projected output 160 GB


# --------------------------------------------------------------------------- #
# (1) free_1 UNIT mechanism -- every no-literal amplifier family projects and    #
#     refuses typed; NO numel/arity threshold gap (zero-numel included)          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,func,args",
    [
        ("outer", torch.outer, lambda: (torch.ones(_N), torch.ones(_N))),
        ("kron", torch.kron, lambda: (torch.ones(_N), torch.ones(_N))),
        ("broadcast_mul", torch.mul, lambda: (torch.ones(_N, 1), torch.ones(1, _N))),
        ("cartesian_prod", torch.cartesian_prod, lambda: (torch.ones(_N), torch.ones(_N))),
        (
            "tensordot_dims0",
            lambda a, b: torch.tensordot(a, b, dims=0),
            lambda: (torch.ones(_N), torch.ones(_N)),
        ),
        (
            "einsum_outer",
            lambda a, b: torch.einsum("i,j->ij", a, b),
            lambda: (torch.ones(_N), torch.ones(_N)),
        ),
        ("diag", torch.diag, lambda: (torch.ones(_N),)),
        ("mm_zero_numel", torch.mm, lambda: (torch.empty(_Z, 0), torch.empty(0, _Z))),
        ("matmul_zero_numel", torch.matmul, lambda: (torch.empty(_Z, 0), torch.empty(0, _Z))),
        (
            "einsum_zero_numel",
            lambda a, b: torch.einsum("ij,jk->ik", a, b),
            lambda: (torch.empty(_Z, 0), torch.empty(0, _Z)),
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_no_literal_amplifier_families_refused_typed(
    name: str, func: Any, args: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each no-literal amplifier form (incl. the zero-numel family every numel/arity
    threshold gaps on) is PROJECTED despite carrying no numeric literal, and refuses
    typed at ``op_allocation_preflight`` against a small budget -- with tiny/empty real
    operands and no real output allocation (fake tensors never allocate)."""

    monkeypatch.setattr(rex, "_allocation_budget_bytes", lambda device: 8 << 20)
    operands = args()
    with _rlimit_cap():
        with pytest.raises(RunCapabilityUnavailableError) as caught:
            _preflight_call_allocation(None, func, list(operands), {}, _stub_call())
    assert caught.value.fields.get("detection_stage") == "op_allocation_preflight"
    assert int(caught.value.fields["required_bytes"]) > int(caught.value.fields["available_bytes"])


def test_amplifier_ops_at_honest_shapes_not_refused() -> None:
    """The SAME amplifier ops at honest feasible shapes pass the projection silently
    (no over-refusal) under the REAL host budget."""

    small = torch.ones(8)
    for func, args in (
        (torch.outer, (small, small)),
        (torch.kron, (small, small)),
        (torch.mul, (torch.ones(8, 1), torch.ones(1, 8))),
        (torch.cartesian_prod, (small, small)),
        (torch.diag, (small,)),
        (torch.mm, (torch.empty(4, 0), torch.empty(0, 4))),
    ):
        _preflight_call_allocation(None, func, list(args), {}, _stub_call())


# --------------------------------------------------------------------------- #
# (2) free_1 E2E -- the exact r60 vector: tampered upstream literal feeding a     #
#     no-literal amplifier refuses typed BEFORE the real allocation               #
# --------------------------------------------------------------------------- #


class _ArangeOuter(nn.Module):
    """The r60 free_1 e2e shape: a literal-driven operand feeding a NO-literal outer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.arange(1000, device=x.device, dtype=torch.float32) + x[0] * 0
        return torch.outer(a, a).sum() + x.sum()


class _ZeroNumelMM(nn.Module):
    """Zero-numel amplifier: reshape literals drive ``[N,0] @ [0,N]`` with 0-byte operands."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (0,)
        a = x.reshape(4, 0)
        b = x.reshape(0, 4)
        return torch.mm(a, b).sum() + x.sum()


def test_e2e_no_literal_outer_amplifier_tamper_refused(tmp_path: Path) -> None:
    """The r60 free_1 repro, self-consistent: tamper the upstream ``arange`` literal
    (1000 -> 1e6) AND the matching intermediate slot shapes, so every upstream gate
    passes honestly (4 MB operand, in-budget clones, shape witnesses consistent) and the
    downstream NO-literal ``outer`` -- recorded output slot left honest/small -- would
    allocate 4 TB. r61 projects the outer call off its tensor operand and refuses typed
    at ``op_allocation_preflight``, never the raw allocator (address-space cap proves no
    allocation)."""

    x = torch.zeros(4)
    bundle = _build(tmp_path, "outer.tlspec", _ArangeOuter(), x)
    assert _tamper_literal(bundle, "int", 1000, 10**6) >= 1
    assert _tamper_slot_shapes(bundle, [1000], [10**6]) >= 1
    with _rlimit_cap():
        loaded = tl.load(str(bundle))
        with pytest.raises(RunCapabilityUnavailableError) as caught:
            loaded.run(inputs=x.clone())
    assert caught.value.fields.get("detection_stage") == "op_allocation_preflight"


def test_e2e_zero_numel_mm_tamper_refused(tmp_path: Path) -> None:
    """Zero-numel e2e, self-consistent: tamper the reshape literals (4 -> 1e7) AND the
    matching zero-numel slot shapes so ``mm`` on two 0-byte operands would allocate
    4e14 bytes. Every byte gate upstream passes honestly (views project 0 new bytes;
    0-numel slots weigh 0 bytes in every bound; the recorded mm output slot stays
    honest/small) -- only the r61 tensor-operand projection can refuse it."""

    x = torch.zeros(0)
    bundle = _build(tmp_path, "zeromm.tlspec", _ZeroNumelMM(), x)
    assert _tamper_literal(bundle, "int", 4, 10**7) >= 2
    assert _tamper_slot_shapes(bundle, [4, 0], [10**7, 0]) >= 1
    assert _tamper_slot_shapes(bundle, [0, 4], [0, 10**7]) >= 1
    with _rlimit_cap():
        loaded = tl.load(str(bundle))
        with pytest.raises(RunCapabilityUnavailableError) as caught:
            loaded.run(inputs=x.clone())
    assert caught.value.fields.get("detection_stage") == "op_allocation_preflight"


# --------------------------------------------------------------------------- #
# (3) over-refusal E2E pins -- honest amplifier zoo + honest zero-numel VERIFIED  #
# --------------------------------------------------------------------------- #


class _AmplifierZoo(nn.Module):
    """Every no-literal amplifier family at honest tiny shapes in ONE forward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (6,)
        col = x.unsqueeze(1)
        row = x.unsqueeze(0)
        return (
            torch.outer(x, x).sum()
            + torch.kron(x, x).sum()
            + (col * row).sum()
            + torch.cartesian_prod(x, x).sum()
            + torch.tensordot(x, x, dims=0).sum()
            + torch.einsum("i,j->ij", x, x).sum()
            + torch.diag(x).sum()
            + x.sum()
        )


def test_honest_amplifier_zoo_runs_verified(tmp_path: Path) -> None:
    """Honest no-literal amplifier ops replay VERIFIED -- projecting every
    tensor-operand call refuses nothing at feasible shapes."""

    x = torch.randn(6)
    bundle = _build(tmp_path, "zoo.tlspec", _AmplifierZoo(), x)
    result = tl.load(str(bundle)).run(inputs=x.clone())
    assert result.report.path_faithfulness.value == "verified"


def test_honest_zero_numel_model_runs_verified(tmp_path: Path) -> None:
    """An honest zero-numel matmul model replays VERIFIED (zero-numel operands are a
    legitimate shape, not an over-refusal channel)."""

    x = torch.zeros(0)
    bundle = _build(tmp_path, "zeromm_honest.tlspec", _ZeroNumelMM(), x)
    result = tl.load(str(bundle)).run(inputs=x.clone())
    assert result.report.path_faithfulness.value == "verified"


# --------------------------------------------------------------------------- #
# (4) corr_2 -- input mirror clone byte-guarded; honest requires_grad preserved   #
# --------------------------------------------------------------------------- #


class _Sum(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()


def test_e2e_tampered_input_slot_expand_view_refused_at_mirror(tmp_path: Path) -> None:
    """The r60 corr_2 repro: inflate the model_input slot shape ([1] -> [1e12]) and bind
    a cheap runtime ``expand`` view that passes the shape contract. The mirror clone
    would materialize 4 TB; r61 refuses typed at ``clone_allocation_preflight`` BEFORE
    the clone (address-space cap proves no allocation)."""

    x = torch.randn(1)
    bundle = _build(tmp_path, "mirror.tlspec", _Sum(), x)
    assert _tamper_input_slot_shape(bundle, [10**12]) >= 1
    with _rlimit_cap():
        loaded = tl.load(str(bundle))
        with pytest.raises(RunCapabilityUnavailableError) as caught:
            loaded.run(inputs=torch.randn(1).expand(10**12))
    assert caught.value.fields.get("detection_stage") == "clone_allocation_preflight"


def test_runtime_mirror_clone_routes_guard_and_mirrors_requires_grad() -> None:
    """The input mirror keeps the r37 autograd-mirror rule THROUGH the r61 guard: an
    honest differentiable leaf clones with ``requires_grad`` restored (attestation
    eligibility unchanged); a plain leaf clones without it; an over-budget logical view
    refuses typed before cloning."""

    ceiling = RunResourceCeiling(types.SimpleNamespace(calls=[]))  # type: ignore[arg-type]
    slot = types.SimpleNamespace(slot_id="slot:input_1")
    raw = torch.randn(3, requires_grad=True)
    clone = rex._runtime_mirror_clone(raw, ceiling, slot)  # type: ignore[arg-type]
    assert clone.requires_grad and clone.is_leaf
    assert clone.data_ptr() != raw.data_ptr()
    plain = rex._runtime_mirror_clone(torch.randn(3), ceiling, slot)  # type: ignore[arg-type]
    assert not plain.requires_grad
    with _rlimit_cap():
        huge = torch.randn(1).expand(10**12)
        with pytest.raises(RunCapabilityUnavailableError) as caught:
            rex._runtime_mirror_clone(huge, ceiling, slot)  # type: ignore[arg-type]
    assert caught.value.fields.get("detection_stage") == "clone_allocation_preflight"


# --------------------------------------------------------------------------- #
# (5) corr_2 state chain -- binder staging guarded from load_state_dict; honest   #
#     expanded-view user state stays VERIFIED; run-time state clone guarded       #
# --------------------------------------------------------------------------- #


class _LinearNoBias(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).sum()


def test_state_staging_clone_guards_logical_extent_from_load_state_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The binder staging clone (probe C's load-bearing site: the strict binder accepts
    an expanded view and materializes it with no byte bound) refuses typed at
    ``clone_allocation_preflight`` from ``load_state_dict``. The budget sits ABOVE the
    view's physical storage (16 B) and BELOW its logical extent (64 B), so the refusal
    proves the guard bounds the LOGICAL numel the clone would materialize."""

    x = torch.randn(2, 4)
    bundle = _build(tmp_path, "staging.tlspec", _LinearNoBias(), x)
    loaded = tl.load(str(bundle))
    expanded = torch.randn(1, 4).expand(4, 4)  # storage 16 B, logical 64 B
    monkeypatch.setattr(rstate, "_allocation_budget_bytes", lambda device: 32)
    with pytest.raises(RunCapabilityUnavailableError) as caught:
        loaded.load_state_dict({"lin.weight": expanded})
    assert caught.value.fields.get("detection_stage") == "clone_allocation_preflight"
    assert caught.value.fields.get("state_dict_name") == "lin.weight"


def test_honest_expanded_view_user_state_stays_verified(tmp_path: Path) -> None:
    """Honest expanded-view user state at recorded shapes binds, stages, clones, and
    replays VERIFIED (the probe C honest path, pinned against over-refusal)."""

    x = torch.randn(2, 4)
    bundle = _build(tmp_path, "expanded_state.tlspec", _LinearNoBias(), x)
    loaded = tl.load(str(bundle))
    loaded.load_state_dict({"lin.weight": torch.randn(1, 4).expand(4, 4)})
    result = loaded.run(inputs=x.clone())
    assert result.report.path_faithfulness.value == "verified"
    assert result.report.state_source is StateSource.USER_STATE_DICT


def test_byte_guarded_clone_core_units() -> None:
    """The module-level core refuses an over-budget logical view typed (with the state
    name in the diagnostic when given) and clones an in-budget tensor normally."""

    base = torch.randn(4)
    small = _byte_guarded_clone(base, state_dict_name="w")
    assert torch.equal(small, base) and small.data_ptr() != base.data_ptr()
    with _rlimit_cap():
        huge = base[:1].expand(10**12, 4)
        with pytest.raises(RunCapabilityUnavailableError) as caught:
            _byte_guarded_clone(huge, state_dict_name="w")
    assert caught.value.fields.get("detection_stage") == "clone_allocation_preflight"
    assert caught.value.fields.get("state_dict_name") == "w"
    assert "'w'" in str(caught.value)


def test_run_time_state_clone_routes_through_ceiling_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_clone_state_values`` refuses typed through the transaction ceiling while
    keeping the identity-keyed alias single-clone map on the honest path."""

    ceiling = RunResourceCeiling(types.SimpleNamespace(calls=[]))  # type: ignore[arg-type]
    descriptor = types.SimpleNamespace(tensor_slots=())
    shared = torch.randn(2)
    cloned = rex._clone_state_values(
        descriptor,  # type: ignore[arg-type]
        {"slot:a": shared, "slot:b": shared, "slot:c": torch.randn(2)},
        ceiling,
    )
    assert cloned["slot:a"] is cloned["slot:b"]  # alias group -> ONE clone
    assert cloned["slot:c"] is not cloned["slot:a"]
    monkeypatch.setattr(rstate, "_allocation_budget_bytes", lambda device: 4)
    with pytest.raises(RunCapabilityUnavailableError) as caught:
        rex._clone_state_values(descriptor, {"slot:a": torch.randn(8)}, ceiling)  # type: ignore[arg-type]
    assert caught.value.fields.get("detection_stage") == "clone_allocation_preflight"


# --------------------------------------------------------------------------- #
# (6) predicate truth table + source tripwires                                    #
# --------------------------------------------------------------------------- #


def test_projection_predicate_truth_table() -> None:
    """``_projection_required_by_arguments`` is exactly "any size source": tensor operand
    (including ZERO-numel) or numeric literal; nothing else."""

    assert _projection_required_by_arguments([torch.randn(2)], {})
    assert _projection_required_by_arguments([torch.empty(5, 0)], {})  # zero-numel counts
    assert _projection_required_by_arguments([2], {})
    assert _projection_required_by_arguments([], {"w": torch.randn(1)})
    assert _projection_required_by_arguments([[({"k": torch.randn(1)},)]], {})  # nested
    assert not _projection_required_by_arguments(["x"], {})
    assert not _projection_required_by_arguments([], {})
    assert not _projection_required_by_arguments([True, None, "y"], {"s": "z"})


def test_has_tensor_operand_is_isinstance_only_and_fails_closed() -> None:
    """The tensor-operand predicate reads NO tensor property (a subclass cannot route
    user code through it) and returns True over-depth (fail-closed: project)."""

    src = inspect.getsource(_has_tensor_operand)
    for prop in (".numel", ".shape", ".dtype", ".device"):
        assert prop not in src, f"predicate must not read tensor property {prop}"
    deep: Any = "leaf"
    for _ in range(_MAX_DECODE_NESTING_DEPTH + 2):
        deep = [deep]
    assert _has_tensor_operand(deep)  # over-depth -> True (project)
    assert not _has_tensor_operand(["x", 3, None])  # literals are the OTHER predicate's job


def test_source_tripwires_gate_and_clone_routing() -> None:
    """(a) ``_preflight_call_allocation`` no longer contains the has-literal-only early
    return; (b) every routed re-materialization site contains no raw ``.detach().clone()``
    and routes through the guard core; (c) the capture-side save-time snapshot stays
    deliberately UNROUTED (live-model values, no artifact amplification)."""

    gate_src = inspect.getsource(rex._preflight_call_allocation)
    assert "_projection_required_by_arguments(" in gate_src
    assert "if not (_has_numeric_literal" not in gate_src
    for fn in (rex._runtime_mirror_clone, rex._clone_state_values):
        src = inspect.getsource(fn)
        assert ".detach().clone()" not in src, f"{fn.__name__} has a raw unguarded clone"
        assert "guarded_clone(" in src
    for fn in (
        rstate._validate_named_slot_mapping,
        rstate.bind_embedded_trace_state,
        rstate.bind_embedded_nonpersistent_buffers,
    ):
        src = inspect.getsource(fn)
        assert ".detach().clone()" not in src, f"{fn.__name__} has a raw unguarded clone"
        assert "_byte_guarded_clone(" in src
    assert ".detach().clone()" in inspect.getsource(rstate.snapshot_capture_state)
