"""Recorded module returns: how a preview backend hands module-call outputs to the finalizer.

Torch records what each module call returned at module exit (``ModuleExitEvent``) and its
postprocess turns that into ``ModuleCall.output_ops``. Preview backends record module
membership only, so ``ModuleCall.output_ops`` stayed empty and ``Module.out`` raised. This
module is the one backend-agnostic seam that closes the gap; torch never calls it.

Contract (what a backend does)
------------------------------
1. Keep one :class:`ModuleReturnLedger` per capture, beside the backend's module tree.
2. At every module exit, while the call's frame is still known, call
   ``ledger.record(address, call_index, returned_value, is_leaf=..., label_of=...)``.
   ``address`` and ``call_index`` are the exact pair the backend writes into each op's
   ``modules`` record for that call (the normalized ``address:call_index`` label must match).
   ``is_leaf(value)`` says whether a value is a native array of this backend;
   ``label_of(leaf)`` returns the RAW op label that produced it, or ``None``.
   Record the value the module actually returns to its caller (after any intervention
   replacement), so the recorded leaves are the values downstream code consumes.
3. Build module logs through :func:`attach_object_module_logs_with_returns` instead of
   ``_finalize.attach_object_module_logs``, passing the ledger as ``returns=``.

How returns are represented after finalize (and after save and load)
--------------------------------------------------------------------
- Each recorded leaf whose raw label resolves to an op flags that op exactly as torch does:
  ``Op.module_context.is_module_output`` is ``True``, ``output_of_modules`` gains the
  address and ``output_of_module_calls`` gains the ``address:call_index`` label (in module
  exit order, so an inner call precedes the outer call that passes the leaf on).
- ``ModuleCall.output_ops`` and ``output_layers`` are then derived by the shared module-log
  builder, the same code torch uses: the flagged ops that ran inside the call, each once,
  in execution order. ``Module.out`` resolves through them.
- Duplicate leaves (``return x, x``): the op appears once in ``output_ops``;
  ``ModuleCall.output_paths`` keeps one path per returned occurrence.
- Returned inputs (a leaf produced before the call began, such as an identity module or a
  dropout layer in eval mode): the producing op is flagged with this call in
  ``output_of_module_calls`` and its path is in ``output_paths``, but it is NOT in
  ``output_ops``, because the module-call invariant keeps ``output_ops`` inside the call's
  own ops. Torch instead inserts an identity op inside the call. A pure pass-through call
  therefore has empty ``output_ops`` and ``Module.out`` raises; :func:`returned_op_labels`
  gives the full returned roster (inside and outside the call) for consumers that compare
  module outputs.
- Non-array leaves (Python scalars, strings, ``None``) and leaves with no op label are not
  represented: they appear in neither ``output_ops`` nor ``output_paths``.
- ``output_paths`` components are the typed path components torch uses (``TupleIndex``,
  ``DictKey``, ``NamedField``, ``DataclassField``); a bare returned array has path ``()``.
- ``ModuleCall.output_structure`` stays ``None`` on preview calls: there is no
  backend-neutral container spec for foreign leaves yet.
- The root call (address ``"self"``) is never recorded, matching torch, whose root outputs
  are the trace outputs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .._data_classes.trace import Trace
from .._ir.container import OutputPathComponent
from .._ir.container_registry import OUTPUT_TREE_MAX_DEPTH, _iter_container_children
from ._finalize import attach_object_module_logs

ReturnPath = tuple[OutputPathComponent, ...]

_ROOT_ADDRESS = "self"


@dataclass(frozen=True)
class ReturnedLeaf:
    """One array leaf a module call returned.

    Parameters
    ----------
    path:
        Typed path of the leaf inside the returned value (``()`` for a bare array).
    label_raw:
        Raw label of the op that produced the leaf, or ``None`` when it has none.
    """

    path: ReturnPath
    label_raw: str | None


@dataclass
class ModuleReturnLedger:
    """Capture-time record of the leaves each module call returned.

    Parameters
    ----------
    calls:
        Returned leaves keyed by ``(address, call_index)``, in module exit order.
    """

    calls: dict[tuple[str, int], tuple[ReturnedLeaf, ...]] = field(default_factory=dict)

    def record(
        self,
        address: str,
        call_index: int,
        value: object,
        *,
        is_leaf: Callable[[object], bool],
        label_of: Callable[[object], str | None],
    ) -> None:
        """Record the array leaves one module call returned.

        Parameters
        ----------
        address:
            Module address the backend writes into each op's ``modules`` record.
        call_index:
            One-based call index of this call of ``address``.
        value:
            The value the module returned to its caller.
        is_leaf:
            Predicate for this backend's native arrays.
        label_of:
            Raw op label of a leaf, or ``None``.
        """

        if address == _ROOT_ADDRESS:
            return
        self.calls[(address, call_index)] = tuple(
            ReturnedLeaf(path=path, label_raw=label_of(leaf))
            for path, leaf in iter_returned_leaves(value, is_leaf)
        )


def iter_returned_leaves(
    value: object,
    is_leaf: Callable[[object], bool],
    path: ReturnPath = (),
    depth: int = 0,
) -> Iterator[tuple[ReturnPath, object]]:
    """Yield each array leaf of a returned value with its typed path.

    Parameters
    ----------
    value:
        Returned value or a subtree of it.
    is_leaf:
        Predicate for this backend's native arrays.
    path:
        Path of ``value`` inside the root value.
    depth:
        Container depth of ``value``; deeper trees than torch walks are cut off.

    Yields
    ------
    tuple[ReturnPath, object]
        Leaf path and leaf, in container order.
    """

    if is_leaf(value):
        yield path, value
        return
    if depth >= OUTPUT_TREE_MAX_DEPTH:
        return
    for component, child in _iter_container_children(value):
        yield from iter_returned_leaves(child, is_leaf, (*path, component), depth + 1)


def attach_object_module_logs_with_returns(
    trace: Trace,
    tree: Any,
    *,
    returns: ModuleReturnLedger | None,
    **shared_options: Any,
) -> None:
    """Build object-module logs with module outputs taken from recorded returns.

    Parameters
    ----------
    trace:
        Trace whose ops are finalized and whose module logs are not yet built.
    tree:
        Backend module tree, passed through to the shared builder.
    returns:
        The capture's ledger, or ``None`` when the backend recorded no returns.
    **shared_options:
        Keyword options of ``_finalize.attach_object_module_logs``.
    """

    _flag_returned_ops(trace, returns)
    attach_object_module_logs(trace, tree, **shared_options)
    _attach_output_paths(trace, returns)


def returned_op_labels(trace: Trace, call_label: str) -> list[str]:
    """Return every op one module call returned, inside or outside the call.

    Parameters
    ----------
    trace:
        Finalized (or loaded) trace.
    call_label:
        Module call label, ``address:call_index``.

    Returns
    -------
    list[str]
        Final op labels in execution order, each once. Unlike ``ModuleCall.output_ops``
        this includes returned inputs.
    """

    return [
        op.label
        for op in trace._layer_list
        if call_label in (op._module_context_output_of_module_calls or ())
    ]


def _flag_returned_ops(trace: Trace, returns: ModuleReturnLedger | None) -> None:
    """Flag each recorded leaf's op as an output of the call that returned it."""

    if returns is None:
        return
    for (address, call_index), leaves in returns.calls.items():
        call_label = f"{address}:{call_index}"
        for leaf in leaves:
            op = _resolve_leaf_op(trace, leaf)
            if op is None or call_label in (op._module_context_output_of_module_calls or ()):
                continue
            op._module_context_is_module_output = True
            op._module_context_output_of_modules = [
                *(op._module_context_output_of_modules or ()),
                address,
            ]
            op._module_context_output_of_module_calls = [
                *(op._module_context_output_of_module_calls or ()),
                call_label,
            ]


def _attach_output_paths(trace: Trace, returns: ModuleReturnLedger | None) -> None:
    """Write each recorded call's leaf paths onto its built ModuleCall."""

    if returns is None:
        return
    pass_dict = getattr(trace._module_logs, "_pass_dict", {})
    for (address, call_index), leaves in returns.calls.items():
        module_call = pass_dict.get(f"{address}:{call_index}")
        if module_call is None:
            continue
        module_call.output_paths = tuple(
            leaf.path for leaf in leaves if _resolve_leaf_op(trace, leaf) is not None
        )
        module_call.output_structure = None


def _resolve_leaf_op(trace: Trace, leaf: ReturnedLeaf) -> Any | None:
    """Return the finalized op that produced a recorded leaf, if any."""

    if leaf.label_raw is None:
        return None
    return trace._layer_dict_all_keys.get(leaf.label_raw)
