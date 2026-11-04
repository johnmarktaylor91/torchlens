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

FlashAttention 2 is a native custom op (``flash_attn::_flash_attn_forward``, four sibling
outputs, no boundary op). Its stand-in is a CPU operator registered under the same qualified
name and schema for the duration of one test (a scoped library, so a process that later loads
the real ``flash-attn`` is unaffected). Regression: it validates with an empty and with a
filled dropout mask. Fail-closed: the same operator under another namespace fails; the
attention output and q, k on the log-sum-exp keep failing when they do not move; a parent at
an undeclared position or a keyword stays strict; removing the rows makes it fail again.
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
from torchlens.backends.torch import _foreign_data
from torchlens.backends.torch._foreign_data import (
    FLASH_ATTENTION_2_CUSTOM_OP,
    FLASH_ATTENTION_2_FUNC_NAME,
    FLASH_ATTENTION_4_EXIT,
    FLASH_ATTENTION_4_FUNCTION,
    FLASH_ATTENTION_4_INTERFACE_MODULE,
    OutputDependence,
    known_foreign_kernel_output_dependence,
    known_native_custom_op_output_dependence,
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
    assert CUSTOM_EXEMPTION_CHECKS["_flash_attn_forward"] is check_foreign_kernel_dependence_exempt


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


# ------------------------------------------- FlashAttention 2: a native custom op

_FA2_SCHEMA = (
    "_flash_attn_forward(Tensor q, Tensor k, Tensor v, float dropout_p, float softmax_scale, "
    "bool causal, int window_size_left, int window_size_right, float softcap, "
    "Tensor? alibi_slopes, bool return_softmax) -> (Tensor, Tensor, Tensor, Tensor)"
)


def _fa2_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    return_softmax: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention 2's output shape in unpatchable C calls: (out, lse, S_dmask, rng_state)."""

    del dropout_p, softmax_scale, causal, window_size_left, window_size_right, softcap
    vf: Any = torch._C._VariableFunctions
    out = vf.add(q, vf.mul(k, v))
    if alibi_slopes is not None:
        out = vf.add(out, alibi_slopes)
    lse = vf.logaddexp(q, k)
    mask = vf.mul(q, k) if return_softmax else vf.empty((0,), dtype=q.dtype)
    rng_state = vf.full((2,), 1234, dtype=torch.int64)
    return out, lse, mask, rng_state


def _scoped_fa2_standin(namespace: str) -> Any:
    """Yield ``torch.ops.<namespace>._flash_attn_forward`` with the stand-in CPU kernel.

    The library is scoped: its definition and kernel are removed when the test ends. When
    the real ``flash-attn`` already registered the operator in this process, only the CPU
    kernel is added (the real schema is the one ``_FA2_SCHEMA`` spells).
    """

    with torch.library._scoped_library(namespace, "FRAGMENT") as lib:
        if not hasattr(getattr(torch.ops, namespace), "_flash_attn_forward"):
            lib.define(_FA2_SCHEMA)
        torch.library.register_kernel(
            f"{namespace}::_flash_attn_forward", "cpu", _fa2_kernel, lib=lib
        )
        yield getattr(torch.ops, namespace)._flash_attn_forward


@pytest.fixture
def fa2_op() -> Any:
    """Register the stand-in as ``flash_attn::_flash_attn_forward`` for one test."""

    yield from _scoped_fa2_standin("flash_attn")


@pytest.fixture
def undeclared_fa2_op() -> Any:
    """The same operator, name and schema under an undeclared namespace."""

    yield from _scoped_fa2_standin("tltest_fa2dep")


class _Fa2Model(nn.Module):
    """q, k, v projections into the four-output op; only the attention output is consumed."""

    def __init__(self, op: Any, *, return_softmax: bool = False, layout: str = "args") -> None:
        super().__init__()
        self.op = op
        self.return_softmax = return_softmax
        self.layout = layout
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.a_proj = nn.Linear(4, 4)

    def _fixed(self, alibi: torch.Tensor | None) -> dict[str, Any]:
        """The non-q, k, v arguments in schema order (no dropout, unit scale)."""

        return {
            "dropout_p": 0.0,
            "softmax_scale": 1.0,
            "causal": False,
            "window_size_left": -1,
            "window_size_right": -1,
            "softcap": 0.0,
            "alibi_slopes": alibi,
            "return_softmax": self.return_softmax,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        alibi = self.a_proj(x) if self.layout == "alibi" else None
        if self.layout == "kwarg":
            out, *_rest = self.op(q, k, v=v, **self._fixed(None))
        else:
            out, *_rest = self.op(q, k, v, *self._fixed(alibi).values())
        return torch.relu(out)


def _fa2_capture(model: nn.Module, x: torch.Tensor) -> Any:
    with torch.no_grad():
        return _capture(model, x)


def _fa2_validates(model: nn.Module, x: torch.Tensor) -> bool:
    with torch.no_grad():
        return _validates(model, x)


def _fa2_siblings(trace: Any) -> dict[int, Any]:
    """Return the four native sibling ops of the one ``_flash_attn_forward`` call."""

    ops = [op for op in trace._layer_list if op.func_name == FLASH_ATTENTION_2_FUNC_NAME]
    assert len({op.func_call_id for op in ops}) == 1, [op.label for op in ops]
    by_index = {op.multi_output_index: op for op in ops}
    assert set(by_index) == {0, 1, 2, 3}, by_index
    for op in ops:
        assert op.func_name != FOREIGN_FUNCTION_FUNC_NAME
        assert "function_module" not in op.transform_config, op.transform_config
    return by_index


def test_table_declares_the_flash_attention_2_rows_only() -> None:
    """Outputs 1 and 2 depend on q and k of q, k, v; output 3 on none; output 0 has no row."""

    lookup = known_native_custom_op_output_dependence
    qkv = frozenset({0, 1, 2})
    op, name = FLASH_ATTENTION_2_CUSTOM_OP, FLASH_ATTENTION_2_FUNC_NAME
    assert lookup(op, name, 1) == OutputDependence(op, 1, frozenset({0, 1}), qkv)
    assert lookup(op, name, 2) == OutputDependence(op, 2, frozenset({0, 1}), qkv)
    assert lookup(op, name, 3) == OutputDependence(op, 3, frozenset(), qkv)
    for index, positions in ((1, [2]), (2, [2]), (3, [0, 1, 2])):
        row = lookup(op, name, index)
        assert row is not None and [p for p in range(10) if row.independent_of(p)] == positions
    strict = (
        (op, name, 0),
        (op, name, 4),
        ("tltest_fa2dep::_flash_attn_forward", name, 1),
        (op, "flashattnforward", 1),
        ("flash_attn::_flash_attn_varlen_forward", name, 1),
    )
    for args in strict:
        assert lookup(*args) is None, args


def test_fa2_declared_op_validates_with_either_dropout_mask(fa2_op: Any) -> None:
    """Regression: four native siblings validate with an empty and a filled S_dmask."""

    x = torch.randn(3, 4)
    for return_softmax in (False, True):
        model = _Fa2Model(fa2_op, return_softmax=return_softmax)
        trace = _fa2_capture(model, x)
        try:
            siblings = _fa2_siblings(trace)
            assert siblings[2].out.numel() == (12 if return_softmax else 0)
            assert siblings[0].children and not any(siblings[i].children for i in (1, 2, 3))
            for op in siblings.values():
                assert declared_output_dependence(op) == known_native_custom_op_output_dependence(
                    FLASH_ATTENTION_2_CUSTOM_OP, FLASH_ATTENTION_2_FUNC_NAME, op.multi_output_index
                )
        finally:
            trace.cleanup()
        assert _fa2_validates(model, x) is True, return_softmax
        assert last_validation_failure() is None


def test_fa2_declared_edges_are_exempt_and_every_other_edge_is_checked(fa2_op: Any) -> None:
    """v on lse and S_dmask and q, k, v on rng_state are excused; q, k and output 0 are tested."""

    trace = _fa2_capture(_Fa2Model(fa2_op, return_softmax=True), torch.randn(3, 4))
    try:
        sib = _fa2_siblings(trace)
        exempt = ((1, 2), (2, 2), (3, 0), (3, 1), (3, 2))
        for index, position in exempt:
            assert _decision(trace, sib[index], _parent_at(sib[index], position)) == _EXEMPT
        for index, position in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)):
            parent = _parent_at(sib[index], position)
            assert _check_perturbation_exemptions(trace, sib[index], [parent]) is False
            assert _decision(trace, sib[index], parent) == _CHANGED, (index, position)
    finally:
        trace.cleanup()


def test_fa2_output_zero_and_dependent_edges_still_fail_when_unmoved(fa2_op: Any) -> None:
    """A frozen attention output fails on q and v; a frozen lse fails on q and k."""

    trace = _fa2_capture(_Fa2Model(fa2_op), torch.randn(3, 4))
    try:
        sib = _fa2_siblings(trace)
        _freeze(sib[0])
        _freeze(sib[1])
        for index, position in ((0, 0), (0, 2), (1, 0), (1, 1)):
            assert _decision(trace, sib[index], _parent_at(sib[index], position)) == _INSENSITIVE
        assert _decision(trace, sib[1], _parent_at(sib[1], 2)) == _EXEMPT
    finally:
        trace.cleanup()


def test_undeclared_custom_op_unmoved_sibling_output_still_fails(undeclared_fa2_op: Any) -> None:
    """The same op, name and shape under an undeclared namespace fails on the lse's v edge."""

    model = _Fa2Model(undeclared_fa2_op)
    x = torch.randn(3, 4)
    assert _fa2_validates(model, x) is False
    failure = last_validation_failure()
    assert failure is not None and failure.check_name == "perturbation", failure
    trace = _fa2_capture(model, x)
    try:
        lse = _fa2_siblings(trace)[1]
        assert declared_output_dependence(lse) is None
        assert failure.op_label == lse.label, failure
        assert failure.extra["perturbed_parents"] == [_parent_at(lse, 2)], failure
    finally:
        trace.cleanup()


def test_fa2_parent_at_an_undeclared_position_or_keyword_stays_strict(fa2_op: Any) -> None:
    """alibi_slopes (position 9) on rng_state, and v passed by keyword, are never excused."""

    for layout, label_of in (
        ("alibi", lambda op: _parent_at(op, 9)),
        ("kwarg", lambda op: op.parent_arg_positions["kwargs"]["v"]),
    ):
        trace = _fa2_capture(_Fa2Model(fa2_op, layout=layout), torch.randn(3, 4))
        try:
            sib = _fa2_siblings(trace)
            for index in (1, 3):
                parent = label_of(sib[index])
                assert _check_perturbation_exemptions(trace, sib[index], [parent]) is False
                assert _decision(trace, sib[index], parent) == _INSENSITIVE, (layout, index)
        finally:
            trace.cleanup()


def test_fa2_without_the_rows_fails_again(fa2_op: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation proof: with the native rows removed the declared op fails on the lse's v edge."""

    monkeypatch.setattr(_foreign_data, "_KNOWN_NATIVE_CUSTOM_OPS", ())
    model = _Fa2Model(fa2_op)
    assert _fa2_validates(model, torch.randn(3, 4)) is False
    failure = last_validation_failure()
    assert failure is not None and failure.check_name == "perturbation", failure
    assert failure.extra["perturbed_parents"], failure
