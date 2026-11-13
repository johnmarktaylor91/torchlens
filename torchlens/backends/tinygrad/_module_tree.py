"""tinygrad object-module tree, call frames, and the module returns each call records.

``TinygradModuleTree`` and ``TinygradModuleFrame`` moved here from ``backend.py`` (file cap);
``backend.py`` re-imports both names.

Recorded module returns (the shared seam, :mod:`torchlens.backends._module_returns`): the
scoped ``__call__`` wrapper calls :func:`note_return` at every module exit, which keeps the
returned value and the UOp of each returned tensor as it was at exit (tinygrad repoints live
tensors at their buffers when it realizes, so the UOp must be read then). Once the capture has
labelled its recorded UOps, :func:`record_returns` resolves each returned UOp to its op label
and fills the tree's ledger. tinygrad is lazy: a returned tensor nothing downstream reads is
never computed, has no op, and is not represented (the seam's rule for unlabeled leaves).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from ..._errors import CaptureGapWarning
from .._module_returns import ModuleReturnLedger, iter_returned_leaves


@dataclass
class TinygradReturnLedger(ModuleReturnLedger):
    """Module-return ledger plus the returns noted at module exit.

    Parameters
    ----------
    pending:
        ``(address, call_index, value, uop_by_leaf_id)`` per module exit, in exit order.
    """

    pending: list[tuple[str, int, Any, dict[int, Any]]] = field(default_factory=list)


@dataclass(frozen=True)
class TinygradModuleFrame:
    """One observed tinygrad object-module stack frame.

    Parameters
    ----------
    address
        Primary TorchLens module address.
    call_index
        One-based call index for the primary address.
    module_type
        Runtime class name for the module object.
    """

    address: str
    call_index: int
    module_type: str


_ACTIVE_TINYGRAD_MODULE_STACK: list[TinygradModuleFrame] = []
"""Module frames of the tinygrad calls active right now, outermost first."""


@dataclass(frozen=True)
class TinygradModuleTree:
    """Discovered tinygrad object-module tree.

    Parameters
    ----------
    root
        Root callable object.
    metadata
        TorchLens module metadata keyed by primary address.
    address_by_id
        Primary module address keyed by object identity.
    modules_by_class
        Module instance addresses grouped by class for scoped ``__call__`` patching.
    param_owner_by_address
        Owning module address keyed by parameter address.
    param_address_by_uop_id
        Parameter address keyed by exact tensor UOp object identity.
    call_counts
        Per-primary-address call counts recorded during capture.
    forward_args_by_call
        Forward args keyed by ``(primary_address, call_index)``.
    module_returns
        Leaves each module call returned (the shared module-return seam).
    """

    root: Any
    metadata: dict[str, dict[str, Any]]
    address_by_id: dict[int, str]
    modules_by_class: dict[type[Any], dict[int, str]]
    param_owner_by_address: dict[str, str]
    param_address_by_uop_id: dict[int, str]
    call_counts: dict[str, int]
    forward_args_by_call: dict[tuple[str, int], tuple[tuple[Any, ...], dict[str, Any]]]
    module_returns: TinygradReturnLedger = field(default_factory=TinygradReturnLedger)


def note_return(tree: Any, address: str, call_index: int, value: Any) -> Any:
    """Note one module call's return and its tensors' UOps at exit; return the value."""

    ledger = getattr(tree, "module_returns", None)
    if isinstance(ledger, TinygradReturnLedger):
        uops = {id(t): cast(Any, t).uop for _p, t in iter_returned_leaves(value, _is_tensor)}
        ledger.pending.append((address, call_index, value, uops))
    return value


def record_returns(tree: Any, uop_labels: Mapping[int, str]) -> None:
    """Record every noted return with the raw label of each returned UOp's op."""

    ledger = getattr(tree, "module_returns", None)
    if not isinstance(ledger, TinygradReturnLedger):
        return
    for address, call_index, value, uops in ledger.pending:

        def label_of(leaf: object, uops: dict[int, Any] = uops) -> str | None:
            """Return the raw label of the op that produced a returned tensor, if any."""

            uop = uops.get(id(leaf))
            return None if uop is None else uop_labels.get(id(uop))

        ledger.record(address, call_index, value, is_leaf=_is_tensor, label_of=label_of)
        tensors = [leaf for _path, leaf in iter_returned_leaves(value, _is_tensor)]
        _disclose_unrecorded(f"{address}:{call_index}", sum(label_of(t) is None for t in tensors))
    ledger.pending = []


def _disclose_unrecorded(call_label: str, missing: int) -> None:
    """Warn that a call returned ``missing`` tensors no recorded op produced (never fabricated).

    tinygrad is lazy: a returned tensor nothing reads is never computed, so it has no op.
    TorchLens does not force the computation (that would change the user's execution); the
    tensor is left out of the call's ``output_ops`` and ``output_paths`` and disclosed here.
    """

    if missing:
        warnings.warn(
            CaptureGapWarning(
                f"tinygrad module call '{call_label}' returned {missing} tensor(s) with no "
                "recorded op (a lazy tensor nothing reads is never computed); they are left "
                "out of this call's output_ops, output_paths and Module.out. Remedy: read the "
                "returned tensor downstream if it should be recorded",
                code="module_return_unrecorded",
            ),
            stacklevel=2,
        )


def _is_tensor(value: object) -> bool:
    """Return whether ``value`` is a tinygrad Tensor."""

    from tinygrad import Tensor

    return isinstance(value, Tensor)
