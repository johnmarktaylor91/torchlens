"""Trace accessor helpers."""

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any
import weakref

from .._errors import AmbiguousOpLookupError
from ._accessor_base import Accessor
from .op import Op


class OrphanAccessor(Accessor[Op]):
    """Dict-like accessor for retained orphan ``Op`` records."""

    def __init__(self, _orphan_labels: Mapping[str, Op] | None = None) -> None:
        """Initialize from raw orphan labels.

        Parameters
        ----------
        _orphan_labels:
            Mapping from raw orphan labels to retained orphan operation logs.
        """

        super().__init__(_orphan_labels or {})

    def _resolve_substring(self, key: str) -> Op | None:
        """Resolve by any orphan label variant or unique substring.

        Parameters
        ----------
        key:
            Lookup key or substring.

        Returns
        -------
        Op | None
            Matching orphan operation, or ``None`` if not found or ambiguous.
        """

        exact_matches = [
            orphan
            for orphan in self._dict.values()
            if key
            in {
                orphan.layer_label,
                orphan.layer_label_short,
                orphan.label,
                orphan.label_short,
                orphan.layer_label,
                orphan.layer_label_short,
                orphan._label_raw,
            }
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]

        substring_matches = [
            orphan
            for orphan in self._dict.values()
            if any(
                label is not None and key.lower() in str(label).lower()
                for label in (
                    orphan.layer_label,
                    orphan.layer_label_short,
                    orphan.label,
                    orphan.label_short,
                    orphan.layer_label,
                    orphan.layer_label_short,
                    orphan._label_raw,
                )
            )
        ]
        if len(substring_matches) == 1:
            return substring_matches[0]
        return None

    @property
    def _item_kind(self) -> str:
        """Return display name used in generic ``KeyError`` messages."""

        return "orphan"


class TraceOpAccessor(Accessor[Op]):
    """Trace-level accessor for type-strict Op lookups."""

    def __init__(self, ops: Sequence[Op], layer_num_calls: Mapping[str, int]) -> None:
        """Initialize from ordered Op records.

        Parameters
        ----------
        ops:
            Ordered Op records.
        layer_num_calls:
            Mapping from parent Layer label to number of Op passes.
        """

        op_lookup: OrderedDict[str, Op] = OrderedDict()
        self._raw_index_lookup: dict[int, Op] = {}
        for op in ops:
            op_lookup[op.label] = op
            self._raw_index_lookup[op.raw_index] = op
        super().__init__(op_lookup, item_list=list(ops))
        self._layer_num_calls = dict(layer_num_calls)

    def by_raw_index(self, raw_index: int) -> Op:
        """Return an Op by its realtime raw capture index.

        Parameters
        ----------
        raw_index:
            One-based raw capture index stored on the Op.

        Returns
        -------
        Op
            Matching operation record.
        """

        try:
            return self._raw_index_lookup[raw_index]
        except KeyError as exc:
            raise KeyError(f"Op raw_index {raw_index} not found.") from exc

    def _resolve_pass_qualified(self, key: str) -> Op | None:
        """Resolve pass-qualified Op labels without returning parent Layers."""

        if key in self._dict:
            return self._dict[key]
        return None

    def _resolve_substring(self, key: str) -> Op | None:
        """Resolve exact long/short Op labels or unique bare parent labels."""

        for op in self._list:
            if key in {op.label, op.label_short, op._label_raw, op.raw_label}:
                return op
            if self._layer_num_calls.get(op.layer_label, 0) == 1 and key in {
                op.layer_label,
                op.layer_label_short,
            }:
                return op
        parent_matches = [op for op in self._list if key in {op.layer_label, op.layer_label_short}]
        if len(parent_matches) == 1:
            return parent_matches[0]
        if len(parent_matches) > 1:
            parent_label = parent_matches[0].layer_label
            qualified = ", ".join(op.label for op in parent_matches[:10])
            suffix = "..." if len(parent_matches) > 10 else ""
            raise AmbiguousOpLookupError(
                f"Layer '{parent_label}' has {len(parent_matches)} ops. Use a 0-based "
                "integer position or a pass-qualified Op label such as "
                f"{qualified}{suffix}."
            )
        return None


class TraceModuleCallAccessor(Accessor[Any]):
    """Trace-level accessor for type-strict ModuleCall lookups."""

    def __init__(self, calls: Mapping[str, Any]) -> None:
        """Initialize from call-label keyed ModuleCalls."""

        super().__init__(calls)

    def _resolve_substring(self, key: str) -> Any | None:
        """Resolve unique bare Module address to its only ModuleCall."""

        parent_matches = [call for call in self._list if key == getattr(call, "address", None)]
        if len(parent_matches) == 1:
            return parent_matches[0]
        if len(parent_matches) > 1:
            raise AmbiguousOpLookupError(
                f"Module '{key}' has {len(parent_matches)} calls. Use a 0-based integer "
                f"position or a call-qualified label like '{key}:1'."
            )
        return None


class TraceGradFnCallAccessor(Accessor[Any]):
    """Trace-level accessor for type-strict GradFnCall lookups."""

    def __init__(self, calls: Mapping[str, Any]) -> None:
        """Initialize from call-label keyed GradFnCalls."""

        super().__init__(calls)

    def _resolve_substring(self, key: str) -> Any | None:
        """Resolve unique bare GradFn label to its only GradFnCall."""

        parent_matches = [call for call in self._list if key == getattr(call, "label", None)]
        if len(parent_matches) == 1:
            return parent_matches[0]
        if len(parent_matches) > 1:
            raise AmbiguousOpLookupError(
                f"GradFn '{key}' has {len(parent_matches)} calls. Use a 0-based integer "
                f"position or a call-qualified label like '{key}:1'."
            )
        return None


_TRACE_OP_ACCESSOR_CACHE: weakref.WeakKeyDictionary[Any, tuple[int, TraceOpAccessor]] = (
    weakref.WeakKeyDictionary()
)
_TRACE_LAYER_ACCESSOR_CACHE: weakref.WeakKeyDictionary[Any, tuple[int, Any]] = (
    weakref.WeakKeyDictionary()
)
