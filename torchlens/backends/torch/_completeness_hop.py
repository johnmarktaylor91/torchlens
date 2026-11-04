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

import importlib
from collections.abc import Callable
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

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
    mode: TorchDispatchMode,
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
        mode: TorchDispatchMode,
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

    def _dense_rule(mode: TorchDispatchMode, *args: Any, **kwargs: Any) -> Any:
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


def _hop(module: str, attribute: str) -> Any:
    """Return a ``HigherOrderOperator`` from ``torch._higher_order_ops``, or ``None``.

    Parameters
    ----------
    module:
        Submodule name.
    attribute:
        Operator attribute.

    Returns
    -------
    Any
        The operator, or ``None`` when this torch lacks it or it has no rule table.
    """

    try:
        mod = importlib.import_module(f"torch._higher_order_ops.{module}")
    except ImportError:
        return None
    op = getattr(mod, attribute, None)
    if not isinstance(op, torch._ops.HigherOrderOperator):
        return None
    if not isinstance(getattr(op, "python_key_table", None), dict):
        return None
    return op


def _witness_hop_rules() -> list[tuple[Any, _HopRule]]:
    """Return ``(operator, rule)`` pairs available on the running torch.

    Returns
    -------
    list[tuple[Any, _HopRule]]
        One pair per operator this torch provides.
    """

    rules: list[tuple[Any, _HopRule]] = []
    cond_op = _hop("cond", "cond_op")
    if cond_op is not None:
        rules.append((cond_op, _cond_rule))
    scan_op = _hop("scan", "scan_op")
    if scan_op is not None:
        scan_module = importlib.import_module("torch._higher_order_ops.scan")
        generic_scan = getattr(scan_module, "generic_scan", None)
        if callable(generic_scan):
            rules.append((scan_op, _scan_rule_from(generic_scan)))
    dense_key = torch._C.DispatchKey.CompositeExplicitAutograd
    for module, attribute in _DENSE_UNDER_MODE:
        op = _hop(module, attribute)
        dense = None if op is None else op.py_kernels.get(dense_key)
        if dense is not None:
            rules.append((op, _dense_rule_from(dense)))
    return rules


def register_witness_hop_rules(mode_type: type[TorchDispatchMode]) -> tuple[str, ...]:
    """Register the HOP rules for ``mode_type`` (idempotent).

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
    for op, rule in _witness_hop_rules():
        if mode_type not in op.python_key_table:
            op.py_impl(mode_type)(rule)
        registered.append(str(op.name()))
    return tuple(registered)
