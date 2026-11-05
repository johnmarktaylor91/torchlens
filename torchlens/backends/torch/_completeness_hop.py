"""Higher-order-operator rules for the completeness witness's dispatch mode.

A ``HigherOrderOperator`` (``torch.cond``, ``while_loop``, ``map``, ``scan``,
``associative_scan``) that reaches the Python dispatch key while a
``TorchDispatchMode`` is active looks the mode's TYPE up in the operator's
``python_key_table`` and raises ``NotImplementedError`` when no rule is
registered. The witness mode had none, so ``tl.validate`` raised on every forward
that called one of these operators, although TorchLens captures the ops their
callables run.

Each rule here replays the operator's eager (``CompositeExplicitAutograd``)
semantics with the witness mode re-entered, so every aten op of the branch or
body that actually ran is censused exactly as in a plain forward: a dropped op
inside the subgraph is caught, and nothing is exempted. ``cond`` and ``scan``
carry a mode-free copy of their eager kernel because torch's own kernels refuse
to run while any dispatch mode is active. An operator without a rule here keeps
torch's ``NotImplementedError``: validation then fails loudly, never silently.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from ..._utils._hop_entry_compile import compile_disabled
from ..._utils._torch_compat import (
    get_cond_witness_operator,
    get_higher_order_eager_kernel,
    get_higher_order_operator,
    get_scan_eager_body,
)
from ._modes import _TorchLensDispatchMode

_HopRule = Callable[..., Any]

_DENSE_UNDER_MODE: tuple[tuple[str, str], ...] = (
    ("while_loop", "while_loop_op"),
    ("while_loop", "while_loop_stack_output_op"),
    ("map", "map_impl"),
    ("associative_scan", "associative_scan_op"),
)
"""``(module, attribute)`` of operators whose eager kernel runs as-is under the mode.

Their kernels only call the user callables and plain tensor ops, so re-entering
the witness mode around the kernel censuses the body ops and the stacking ops
TorchLens captured.
"""


def _cond_rule(
    mode: _TorchLensDispatchMode,
    pred: Any,
    true_fn: Callable[..., Any],
    false_fn: Callable[..., Any],
    operands: Any,
) -> Any:
    """Run the taken ``cond`` branch with ``mode`` active (eager ``cond`` semantics).

    The predicate is read under the mode, as torch's eager kernel reads it, so the
    data-dependent host read is observed like any Python ``if tensor:``.

    Parameters
    ----------
    mode:
        The witness mode the operator popped before calling this rule.
    pred:
        Boolean or single-element tensor predicate.
    true_fn, false_fn:
        Branch callables.
    operands:
        Branch operands.

    Returns
    -------
    Any
        The taken branch's output.
    """

    with mode:
        branch = true_fn if pred else false_fn
        return branch(*operands)


def _cond_rule_any_dispatch(*args: Any) -> Any:
    """Run the taken ``cond`` branch under either HOP dispatch convention.

    Current torch pops the mode and passes it first (``_cond_rule``). Torch 2.1,
    whose bare ``cond`` is the operator, calls the rule with the operator's own
    arguments and leaves the mode active, so the branch runs as-is and every op it
    dispatches still reaches the witness.

    Parameters
    ----------
    *args:
        ``(mode, pred, true_fn, false_fn, operands)`` or ``(pred, true_fn, false_fn, operands)``.

    Returns
    -------
    Any
        The taken branch's output.
    """

    if isinstance(args[0], _TorchLensDispatchMode):
        return _cond_rule(*args)
    pred, true_fn, false_fn, operands = args
    branch = true_fn if pred else false_fn
    return branch(*operands)


def _scan_rule_from(generic_scan: Callable[..., Any]) -> _HopRule:
    """Build the ``scan`` rule around torch's mode-free ``generic_scan``.

    Parameters
    ----------
    generic_scan:
        ``torch._higher_order_ops.scan.generic_scan`` (the body of torch's eager
        ``scan`` kernel without its no-mode assertion).

    Returns
    -------
    _HopRule
        The rule.
    """

    def _scan_rule(
        mode: _TorchLensDispatchMode,
        combine_fn: Callable[..., Any],
        init: Any,
        xs: Any,
        additional_inputs: Any,
    ) -> Any:
        """Run the eager scan with ``mode`` active.

        Parameters
        ----------
        mode:
            The witness mode the operator popped before calling this rule.
        combine_fn:
            Per-step callable.
        init:
            Initial carry.
        xs:
            Scanned inputs.
        additional_inputs:
            Extra inputs passed to every step.

        Returns
        -------
        Any
            Final carry and stacked step outputs.
        """

        with mode:
            return generic_scan(combine_fn, init, xs, additional_inputs=additional_inputs)

    return _scan_rule


def _dense_rule_from(dense: Callable[..., Any]) -> _HopRule:
    """Build a rule that runs an operator's eager kernel with the mode re-entered.

    Parameters
    ----------
    dense:
        The operator's ``CompositeExplicitAutograd`` kernel.

    Returns
    -------
    _HopRule
        The rule.
    """

    def _dense_rule(mode: _TorchLensDispatchMode, *args: Any, **kwargs: Any) -> Any:
        """Run the eager kernel with ``mode`` active.

        Parameters
        ----------
        mode:
            The witness mode the operator popped before calling this rule.
        *args, **kwargs:
            The operator's arguments.

        Returns
        -------
        Any
            The kernel's output.
        """

        with mode:
            return dense(*args, **kwargs)

    return _dense_rule


def witness_rule_table(op: Any) -> dict[Any, Any]:
    """Return the operator's per-mode rule table (torch 2.1 names it ``python_key_mode_table``).

    Parameters
    ----------
    op:
        Operator from ``get_higher_order_operator`` or ``get_cond_witness_operator``.

    Returns
    -------
    dict[Any, Any]
        Dispatch-mode type to rule.
    """

    table = getattr(op, "python_key_table", None)
    return table if isinstance(table, dict) else op.python_key_mode_table


def _witness_hop_rules() -> list[tuple[Any, _HopRule]]:
    """Return ``(operator, rule)`` pairs available on the running torch.

    Returns
    -------
    list[tuple[Any, _HopRule]]
        One pair per operator this torch provides.
    """

    rules: list[tuple[Any, _HopRule]] = []
    cond_op = get_cond_witness_operator()
    if cond_op is not None:
        rules.append((cond_op, _cond_rule_any_dispatch))
    scan_op = get_higher_order_operator("scan", "scan_op")
    generic_scan = None if scan_op is None else get_scan_eager_body()
    if scan_op is not None and generic_scan is not None:
        rules.append((scan_op, _scan_rule_from(generic_scan)))
    for module, attribute in _DENSE_UNDER_MODE:
        op = get_higher_order_operator(module, attribute)
        dense = None if op is None else get_higher_order_eager_kernel(op)
        if dense is not None:
            rules.append((op, _dense_rule_from(dense)))
    return rules


def register_witness_hop_rules(mode_type: type[_TorchLensDispatchMode]) -> tuple[str, ...]:
    """Register the HOP rules for ``mode_type`` (idempotent).

    Only operators whose ``torch._higher_order_ops`` module is already imported
    resolve (see ``get_higher_order_operator``): on torch 2.1, importing ``cond``
    imports ``torch._dynamo``, and an operator cannot run before its module is
    imported. While any operator is unresolved, the mode re-runs this registration
    on every entry (``_TorchLensDispatchMode.__enter__``), so an operator imported
    after TorchLens has its rule before a forward under the mode can call it.

    Parameters
    ----------
    mode_type:
        The witness dispatch-mode class.

    Returns
    -------
    tuple[str, ...]
        Names of the operators that now carry a rule for ``mode_type``.
    """

    registered: list[str] = []
    rules = _witness_hop_rules()
    for op, rule in rules:
        if mode_type not in witness_rule_table(op):
            # The operator pops the mode before the rule; inside a lifted entry compile
            # Dynamo would otherwise trace the rule's ``with mode:`` (_tracing_pass).
            # Without Dynamo imported nothing can trace it yet: wrap it at the first
            # entry compile instead (compile_disable_witness_hop_rules).
            dynamo_loaded = "torch._dynamo" in sys.modules
            op.py_impl(mode_type)(compile_disabled(rule) if dynamo_loaded else rule)
        registered.append(str(op.name()))
    unresolved = len(rules) < 2 + len(_DENSE_UNDER_MODE)
    mode_type._hop_rules_pending = register_witness_hop_rules if unresolved else None
    return tuple(registered)


def compile_disable_witness_hop_rules() -> None:
    """Put every rule registered before Dynamo was imported behind ``torch.compiler.disable``.

    Runs before each TorchLens-lifted operator entry compile (``hop_internal_compile_scope``),
    the first point at which Dynamo can trace a rule; a plain capture never gets here, so it
    never imports ``torch._dynamo`` (``torch.compiler.disable`` imports it, and FSDP with it).
    """

    for op, _rule in _witness_hop_rules():
        table = witness_rule_table(op)
        for mode_type, rule in list(table.items()):
            if _is_torchlens_mode(mode_type) and not getattr(rule, "_torchdynamo_disable", False):
                del table[mode_type]
                op.py_impl(mode_type)(compile_disabled(rule))


def _is_torchlens_mode(key: Any) -> bool:
    """Return whether rule-table key ``key`` is a TorchLens dispatch-mode class."""

    return isinstance(key, type) and issubclass(key, _TorchLensDispatchMode)
