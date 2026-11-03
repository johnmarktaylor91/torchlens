"""Declared per-output dependence of known foreign kernels: the carve-out and its edges.

A known foreign kernel that returns several tensors is one ``autogradfunction`` op per
output, each carrying the call's full parent set. The kernel table declares, per Function
and output, which positional arguments that output depends on (ruling: declare per-output
dependence per kernel in a table); the perturbation check excuses a parent of one output
only at a position the table declares that output independent of. FlashAttention 4's
``FlashAttnFunc`` returns ``(out, lse)`` with autograd enabled and the log-sum-exp reads q
and k only, so its v edge is declared independent and nothing else is.

The stand-ins below have FlashAttention 4's shape (fresh ``torch.empty`` outputs filled below
the dispatcher, the interface module as ``__module__``), so the CPU suite exercises the real
capture, table and check. Regression: the declared kernel validates with autograd on (red at
the base, where the lse sibling's v edge failed). Fail-closed: an undeclared kernel with the
same unmoved sibling output still fails; output 0 keeps its full parent set; a declared
dependent argument that does not move the output still fails; a position outside the row,
a keyword, nested or unmapped parent and a missing output index stay strict.
"""

from __future__ import annotations

import array
import ctypes
import math
import warnings
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils import _torch_compat as tc
from torchlens.backends.torch._foreign_data import (
    FLASH_ATTENTION_4_EXIT,
    FLASH_ATTENTION_4_FUNCTION,
    FLASH_ATTENTION_4_INTERFACE_MODULE,
    OutputDependence,
    known_foreign_kernel_output_dependence,
)
from torchlens.backends.torch._function_boundary import FOREIGN_FUNCTION_FUNC_NAME
from torchlens.options import CaptureOptions
from torchlens.validation import last_validation_failure, validate_forward_pass
from torchlens.validation._foreign_kernel_dependence import (
    check_foreign_kernel_dependence_exempt,
    declared_output_dependence,
)
from torchlens.validation.core import (
    _check_perturbation_exemptions,
    _check_whether_func_on_saved_parents_yields_saved_tensor,
)
from torchlens.validation.exemptions import CUSTOM_EXEMPTION_CHECKS

_EXEMPT = ("exempted", "pre_perturbation_exemption")
_INSENSITIVE = ("failed", "perturbation_insensitive")
_CHANGED = ("validated", "perturbation_changed")
_NO_TRACE: Any = None  # the check ignores its trace argument


def _foreign_fill(shape: torch.Size, values: list[float]) -> torch.Tensor:
    """Return a fresh tensor filled through its data pointer (no op the dispatcher sees)."""

    out = torch.empty(shape, dtype=torch.float32)
    buffer = array.array("f", values)
    ctypes.memmove(out.data_ptr(), buffer.buffer_info()[0], len(buffer) * buffer.itemsize)
    return out


def _host_values(*tensors: torch.Tensor) -> list[list[float]]:
    return [t.detach().flatten().tolist() for t in tensors]


def _attention_out(q: list[float], k: list[float], v: list[float]) -> list[float]:
    return [a + b * c for a, b, c in zip(q, k, v, strict=True)]


def _log_sum_exp(q: list[float], k: list[float]) -> list[float]:
    return [math.log(math.exp(a) + math.exp(b)) for a, b in zip(q, k, strict=True)]


class FlashAttnFunc(torch.autograd.Function):
    """Stand-in with the declared kernel's name and shape: ``(out, lse)``; lse reads q and k."""

    __module__ = FLASH_ATTENTION_4_INTERFACE_MODULE

    @staticmethod
    def forward(ctx, q, k, v):  # type: ignore[override]
        qs, ks, vs = _host_values(q, k, v)
        return _foreign_fill(q.shape, _attention_out(qs, ks, vs)), _foreign_fill(
            q.shape, _log_sum_exp(qs, ks)
        )

    @staticmethod
    def backward(ctx, grad_out, grad_lse):  # type: ignore[override]
        return grad_out, grad_out, grad_out


class _UndeclaredTwoOutputKernel(torch.autograd.Function):
    """Same module, same shape, same unmoved sibling output; no row in the table."""

    __module__ = FLASH_ATTENTION_4_INTERFACE_MODULE

    @staticmethod
    def forward(ctx, q, k, v):  # type: ignore[override]
        qs, ks, vs = _host_values(q, k, v)
        return _foreign_fill(q.shape, _attention_out(qs, ks, vs)), _foreign_fill(
            q.shape, _log_sum_exp(qs, ks)
        )

    @staticmethod
    def backward(ctx, grad_out, grad_lse):  # type: ignore[override]
        return grad_out, grad_out, grad_out


class _LseIgnoresK(torch.autograd.Function):
    """The declared name, but output 1 ignores k, a declared-dependent argument."""

    __module__ = FLASH_ATTENTION_4_INTERFACE_MODULE
    __qualname__ = FLASH_ATTENTION_4_FUNCTION

    @staticmethod
    def forward(ctx, q, k, v):  # type: ignore[override]
        qs, ks, vs = _host_values(q, k, v)
        return _foreign_fill(q.shape, _attention_out(qs, ks, vs)), _foreign_fill(
            q.shape, _log_sum_exp(qs, qs)
        )

    @staticmethod
    def backward(ctx, grad_out, grad_lse):  # type: ignore[override]
        return grad_out, grad_out, grad_out


class _FourArgDeclaredName(torch.autograd.Function):
    """The declared name with a fourth tensor argument the row does not speak for."""

    __module__ = FLASH_ATTENTION_4_INTERFACE_MODULE
    __qualname__ = FLASH_ATTENTION_4_FUNCTION

    @staticmethod
    def forward(ctx, q, k, v, extra):  # type: ignore[override]
        qs, ks, vs, es = _host_values(q, k, v, extra)
        out = [a + e for a, e in zip(_attention_out(qs, ks, vs), es, strict=True)]
        return _foreign_fill(q.shape, out), _foreign_fill(q.shape, _log_sum_exp(qs, ks))

    @staticmethod
    def backward(ctx, grad_out, grad_lse):  # type: ignore[override]
        return grad_out, grad_out, grad_out, grad_out


class _KernelModel(nn.Module):
    """Three projections feeding a two-output kernel; only output 0 is consumed, as in FA4."""

    def __init__(self, kernel: type[torch.autograd.Function], extra: bool = False) -> None:
        super().__init__()
        self.kernel = kernel
        self.extra = extra
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.e_proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        args = [self.q_proj(x), self.k_proj(x), self.v_proj(x)]
        if self.extra:
            args.append(self.e_proj(x))
        out, _lse = self.kernel.apply(*args)
        return torch.relu(out)


@pytest.fixture(autouse=True)
def _known_kernel_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tc, "HAS_FLASH_ATTENTION_4_INTERFACE", True)


def _capture(model: nn.Module, x: torch.Tensor) -> Any:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tl.trace(model, x, capture=CaptureOptions(save_args=True, seed=0), save="all")


def _validates(model: nn.Module, x: torch.Tensor) -> bool:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(validate_forward_pass(model, [x], input_kwargs={}))


def _siblings(trace: Any) -> dict[int, Any]:
    """Return the kernel's boundary ops keyed by output index (exactly one call)."""

    ops = [op for op in trace._layer_list if op.func_name == FOREIGN_FUNCTION_FUNC_NAME]
    assert len({op.func_call_id for op in ops}) == 1, [op.label for op in ops]
    by_index = {op.multi_output_index: op for op in ops}
    assert set(by_index) == {0, 1}, by_index
    for op in ops:
        assert op.transform_config["exits"] == [FLASH_ATTENTION_4_EXIT]
    return by_index


def _decision(trace: Any, op: Any, parent: str) -> tuple[str, str]:
    result = _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, op.label, perturb=True, layers_to_perturb=[parent]
    )
    return result.decision, result.reason


def _freeze(op: Any) -> None:
    """Freeze ``op``'s replay to its saved output (the dropped-dependency simulation)."""

    saved = op.out.detach().clone()
    object.__setattr__(op, "func", lambda *args, **kwargs: saved.clone())


def _parent_at(op: Any, position: int) -> str:
    return op.parent_arg_positions["args"][position]


# ------------------------------------------------------------------ the table


def test_table_declares_the_flash_attention_4_log_sum_exp_row_only() -> None:
    """Exactly one row: output 1 of FlashAttnFunc depends on positions 0 and 1 of 0 to 2."""

    exits = [FLASH_ATTENTION_4_EXIT]
    row = known_foreign_kernel_output_dependence(
        FLASH_ATTENTION_4_INTERFACE_MODULE, FLASH_ATTENTION_4_FUNCTION, 1, exits
    )
    assert row == OutputDependence(
        function=FLASH_ATTENTION_4_FUNCTION,
        output_index=1,
        depends_on=frozenset({0, 1}),
        positions=frozenset({0, 1, 2}),
    )
    assert [p for p in range(4) if row.independent_of(p)] == [2]
    submodule = FLASH_ATTENTION_4_INTERFACE_MODULE + ".sm90"
    assert (
        known_foreign_kernel_output_dependence(submodule, FLASH_ATTENTION_4_FUNCTION, 1, exits)
        == row
    )
    strict = [
        (FLASH_ATTENTION_4_INTERFACE_MODULE, FLASH_ATTENTION_4_FUNCTION, 0, exits),
        (FLASH_ATTENTION_4_INTERFACE_MODULE, FLASH_ATTENTION_4_FUNCTION, 2, exits),
        (FLASH_ATTENTION_4_INTERFACE_MODULE, "FlashAttnVarlenFunc", 1, exits),
        ("flash_attn.flash_attn_interface", FLASH_ATTENTION_4_FUNCTION, 1, exits),
        (FLASH_ATTENTION_4_INTERFACE_MODULE, FLASH_ATTENTION_4_FUNCTION, 1, []),
        (FLASH_ATTENTION_4_INTERFACE_MODULE, FLASH_ATTENTION_4_FUNCTION, 1, ["numpy"]),
    ]
    for module, function, index, recorded_exits in strict:
        assert (
            known_foreign_kernel_output_dependence(module, function, index, recorded_exits) is None
        ), (module, function, index, recorded_exits)


def test_check_is_registered_under_autogradfunction() -> None:
    assert CUSTOM_EXEMPTION_CHECKS["autogradfunction"] is check_foreign_kernel_dependence_exempt


# ------------------------------------------------- the predicate on fake records


def _fake_op(
    *,
    args: dict[Any, str] | None = None,
    kwargs: dict[str, str] | None = None,
    output_index: Any = 1,
    function: str = FLASH_ATTENTION_4_FUNCTION,
    module: str = FLASH_ATTENTION_4_INTERFACE_MODULE,
    exits: Any = (FLASH_ATTENTION_4_EXIT,),
) -> SimpleNamespace:
    return SimpleNamespace(
        func_name=FOREIGN_FUNCTION_FUNC_NAME,
        multi_output_index=output_index,
        transform_config={"function": function, "function_module": module, "exits": exits},
        parent_arg_positions={
            "args": {0: "q", 1: "k", 2: "v"} if args is None else args,
            "kwargs": kwargs or {},
        },
    )


def test_predicate_exempts_exactly_the_declared_triple() -> None:
    """(FlashAttnFunc, output 1, position 2) is exempt; nothing else is."""

    op = _fake_op()
    assert check_foreign_kernel_dependence_exempt(_NO_TRACE, op, ["v"]) is True
    for perturbed in (["q"], ["k"], ["q", "v"], ["v", "k"], [], ["absent"]):
        assert check_foreign_kernel_dependence_exempt(_NO_TRACE, op, perturbed) is False, perturbed
    assert declared_output_dependence(_fake_op(output_index=0)) is None
    assert declared_output_dependence(_fake_op(output_index=None)) is None
    assert declared_output_dependence(_fake_op(function="FlashAttnVarlenFunc")) is None
    assert declared_output_dependence(_fake_op(module="flash_attn.flash_attn_interface")) is None
    assert declared_output_dependence(_fake_op(exits=["numpy"])) is None
    assert declared_output_dependence(_fake_op(exits=None)) is None
    assert declared_output_dependence(_fake_op(exits=["numpy", FLASH_ATTENTION_4_EXIT])) is not None


def test_predicate_stays_strict_off_the_declared_positions() -> None:
    """A keyword, nested, unmapped or out-of-row position, or a shared slot, is strict."""

    check = check_foreign_kernel_dependence_exempt
    assert check(_NO_TRACE, _fake_op(args={0: "q", 1: "k"}, kwargs={"v": "v"}), ["v"]) is False
    assert check(_NO_TRACE, _fake_op(args={0: "q", 1: "k", (2, 0): "v"}), ["v"]) is False
    assert check(_NO_TRACE, _fake_op(args={0: "q", 1: "k", 2: "v", 3: "e"}), ["e"]) is False
    assert check(_NO_TRACE, _fake_op(args={0: "q", 1: "k", 2: "v", 3: "e"}), ["v"]) is True
    # v also fills a dependent slot (q is v): the parent feeds the output and stays tested.
    assert check(_NO_TRACE, _fake_op(args={0: "v", 1: "k", 2: "v"}), ["v"]) is False
    assert check(_NO_TRACE, _fake_op(args={}), ["v"]) is False


# -------------------------------------------------- the captured stand-ins


def test_declared_kernel_validates_with_autograd_enabled() -> None:
    """Regression: the FlashAttention 4 shape validates with (out, lse) siblings (red at base)."""

    model = _KernelModel(FlashAttnFunc)
    x = torch.randn(3, 4)
    assert torch.is_grad_enabled()
    trace = _capture(model, x)
    try:
        siblings = _siblings(trace)
        assert siblings[0].children and not siblings[1].children
        for op in siblings.values():
            assert sorted(op.parent_arg_positions["args"]) == [0, 1, 2]
            assert len(op.parents) == 3
    finally:
        trace.cleanup()
    assert _validates(model, x) is True
    assert last_validation_failure() is None


def test_declared_edge_is_exempt_and_every_other_edge_is_checked() -> None:
    """Only the lse sibling's v edge is excused; q and k on lse and q, k, v on out are tested."""

    trace = _capture(_KernelModel(FlashAttnFunc), torch.randn(3, 4))
    try:
        siblings = _siblings(trace)
        out, lse = siblings[0], siblings[1]
        assert _decision(trace, lse, _parent_at(lse, 2)) == _EXEMPT
        assert _check_perturbation_exemptions(trace, lse, [_parent_at(lse, 2)]) is True
        for op, position in ((lse, 0), (lse, 1), (out, 0), (out, 1), (out, 2)):
            parent = _parent_at(op, position)
            assert _check_perturbation_exemptions(trace, op, [parent]) is False, (
                op.label,
                position,
            )
            assert _decision(trace, op, parent) == _CHANGED, (op.label, position)
    finally:
        trace.cleanup()


def test_output_zero_spurious_parent_still_fails() -> None:
    """Output 0 has no row: a v edge that does not move the attention output fails."""

    trace = _capture(_KernelModel(FlashAttnFunc), torch.randn(3, 4))
    try:
        out = _siblings(trace)[0]
        _freeze(out)
        assert _decision(trace, out, _parent_at(out, 2)) == _INSENSITIVE
        assert _decision(trace, out, _parent_at(out, 0)) == _INSENSITIVE
    finally:
        trace.cleanup()


def test_declared_dependent_argument_that_does_not_move_the_output_still_fails() -> None:
    """A frozen lse replay fails on q and k (declared dependent) while v stays exempt."""

    trace = _capture(_KernelModel(FlashAttnFunc), torch.randn(3, 4))
    try:
        lse = _siblings(trace)[1]
        _freeze(lse)
        assert _decision(trace, lse, _parent_at(lse, 0)) == _INSENSITIVE
        assert _decision(trace, lse, _parent_at(lse, 1)) == _INSENSITIVE
        assert _decision(trace, lse, _parent_at(lse, 2)) == _EXEMPT
    finally:
        trace.cleanup()


def test_declared_kernel_whose_lse_ignores_k_fails_validation() -> None:
    """A real kernel under the declared name whose lse ignores k fails on that edge."""

    model = _KernelModel(_LseIgnoresK)
    x = torch.randn(3, 4)
    assert _validates(model, x) is False
    failure = last_validation_failure()
    assert failure is not None
    assert failure.check_name == "perturbation", failure
    trace = _capture(model, x)
    try:
        lse = _siblings(trace)[1]
        assert failure.op_label == lse.label, failure
        assert failure.extra["perturbed_parents"] == [_parent_at(lse, 1)], failure
    finally:
        trace.cleanup()


def test_undeclared_kernel_unmoved_sibling_output_still_fails() -> None:
    """The same unmoved sibling output under an undeclared Function name fails on its v edge."""

    model = _KernelModel(_UndeclaredTwoOutputKernel)
    x = torch.randn(3, 4)
    assert _validates(model, x) is False
    failure = last_validation_failure()
    assert failure is not None and failure.check_name == "perturbation", failure
    trace = _capture(model, x)
    try:
        lse = _siblings(trace)[1]
        assert lse.transform_config["function"] == _UndeclaredTwoOutputKernel.__qualname__
        assert declared_output_dependence(lse) is None
        assert failure.op_label == lse.label, failure
        assert failure.extra["perturbed_parents"] == [_parent_at(lse, 2)], failure
        assert _decision(trace, lse, _parent_at(lse, 2)) == _INSENSITIVE
    finally:
        trace.cleanup()


def test_position_outside_the_row_stays_strict() -> None:
    """A fourth tensor argument is undeclared: its edge on lse is tested, v's stays exempt."""

    trace = _capture(_KernelModel(_FourArgDeclaredName, extra=True), torch.randn(3, 4))
    try:
        lse = _siblings(trace)[1]
        assert sorted(lse.parent_arg_positions["args"]) == [0, 1, 2, 3]
        assert _check_perturbation_exemptions(trace, lse, [_parent_at(lse, 3)]) is False
        assert _decision(trace, lse, _parent_at(lse, 2)) == _EXEMPT
        _freeze(lse)
        assert _decision(trace, lse, _parent_at(lse, 3)) == _INSENSITIVE
    finally:
        trace.cleanup()
