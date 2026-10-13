"""Trace accessor helpers."""

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

from .._errors import AmbiguousOpLookupError
from ._accessor_base import Accessor
from .op import Op


def _group_by_attr(items: Sequence[Any], attr: str) -> dict[Any, list[Any]]:
    """Group items by one attribute value, preserving accessor list order.

    Parameters
    ----------
    items:
        Ordered accessor items.
    attr:
        Attribute name to group by; missing attributes group under ``None``.

    Returns
    -------
    dict[Any, list[Any]]
        Mapping from attribute value to items in list order.
    """

    index: dict[Any, list[Any]] = {}
    for item in items:
        index.setdefault(getattr(item, attr, None), []).append(item)
    return index


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
        # Lazily built reverse indexes for ``_resolve_substring``; see
        # ``_ensure_alias_indexes``. Lifetime is this accessor instance, whose
        # ``_list`` is immutable post-finalization and already rebuilt through the
        # existing ``_TRACE_OP_ACCESSOR_CACHE`` invalidation, so this adds no new
        # invalidation surface.
        self._alias_index: dict[Any, Op] | None = None
        self._layer_alias_index: dict[Any, list[Op]] | None = None
        self._alias_index_unavailable = False
        self._resolved_op_cache: dict[str, Op] = {}

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

    def resolve_all(self, key: str) -> list[Op]:
        """Resolve a label to every Op it denotes.

        A pass-qualified Op label -- or a single-pass Layer label, where the two
        coincide -- resolves to exactly one Op. A bare Layer label that spans
        multiple passes (a recurrent layer, as stored on the ROOT ``self:1``
        ModuleCall which aggregates by Layer) resolves to ALL of that layer's
        pass Ops. Returns an empty list when the label matches nothing.

        Unlike ``__getitem__``/``_resolve_substring``, this never raises
        ``AmbiguousOpLookupError``: multi-pass layer labels are the intended,
        fully-resolved case here.
        """

        direct = self._resolve_pass_qualified(key)
        if direct is not None:
            return [direct]
        return [op for op in self._list if key in {op.layer_label, op.layer_label_short}]

    def _ensure_alias_indexes(self) -> tuple[dict[Any, Op], dict[Any, list[Op]]] | None:
        """Build the alias reverse indexes once, or report them unavailable.

        The scanning form of ``_resolve_substring`` returns the FIRST Op in list
        order any of whose aliases equals the key, where the alias set is the union
        of the two per-Op match conditions. ``setdefault`` over both conditions in
        list order therefore stores exactly that minimum-index winner for every
        alias, and the second index reproduces the bare-parent-label match list in
        the same order -- so the indexed lookup is result-identical to the scan,
        ``AmbiguousOpLookupError`` message included.

        Building touches label attributes on every Op, which the early-returning
        scan may never have reached. If any read fails (e.g. an unfinished Op whose
        label slot is unset mid-capture), the indexes are abandoned and the scan is
        used instead, which reproduces the original raise-or-return behavior exactly.

        Returns
        -------
        tuple[dict[Any, Op], dict[Any, list[Op]]] | None
            The ``(alias -> Op, layer alias -> Ops)`` index pair, or ``None`` when
            the indexes could not be built and the scan must be used.
        """

        if self._alias_index is not None and self._layer_alias_index is not None:
            return self._alias_index, self._layer_alias_index
        if self._alias_index_unavailable:
            return None
        alias_index: dict[Any, Op] = {}
        layer_alias_index: dict[Any, list[Op]] = {}
        layer_num_calls = self._layer_num_calls
        try:
            for op in self._list:
                layer_label = op.layer_label
                layer_label_short = op.layer_label_short
                for alias in (op.label, op.label_short, op._label_raw, op.raw_label):
                    alias_index.setdefault(alias, op)
                if layer_num_calls.get(layer_label, 0) == 1:
                    for alias in (layer_label, layer_label_short):
                        alias_index.setdefault(alias, op)
                for alias in {layer_label, layer_label_short}:
                    layer_alias_index.setdefault(alias, []).append(op)
        except Exception:
            self._alias_index_unavailable = True
            return None
        self._alias_index = alias_index
        self._layer_alias_index = layer_alias_index
        return alias_index, layer_alias_index

    def _resolve_substring_by_scan(self, key: str) -> Op | None:
        """Resolve by linear scan, used when the alias indexes are unavailable."""

        for op in self._list:
            if key in {op.label, op.label_short, op._label_raw, op.raw_label}:
                return op
            if self._layer_num_calls.get(op.layer_label, 0) == 1 and key in {
                op.layer_label,
                op.layer_label_short,
            }:
                return op
        parent_matches = [op for op in self._list if key in {op.layer_label, op.layer_label_short}]
        return self._resolve_parent_matches(parent_matches)

    @staticmethod
    def _resolve_parent_matches(parent_matches: Sequence[Op]) -> Op | None:
        """Return the unique bare-parent-label match, or raise when ambiguous."""

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

    def _resolve_substring(self, key: str) -> Op | None:
        """Resolve exact long/short Op labels or unique bare parent labels."""

        indexes = self._ensure_alias_indexes()
        if indexes is None:
            return self._resolve_substring_by_scan(key)
        alias_index, layer_alias_index = indexes
        direct = alias_index.get(key)
        if direct is not None:
            return direct
        return self._resolve_parent_matches(layer_alias_index.get(key, ()))

    def _resolved_op(self, key: str) -> Op:
        """Return ``self[key]``, memoized per lookup key.

        Callers that resolve the same stored child/parent label once per graph edge
        (the ``_trace_stats`` edge counters) would otherwise repeat one full
        ``__getitem__`` per edge. Failed lookups are never memoized, so a missing or
        ambiguous key raises exactly as ``self[key]`` does on every call.
        """

        cached = self._resolved_op_cache.get(key)
        if cached is None:
            cached = self[key]
            self._resolved_op_cache[key] = cached
        return cached


class TraceModuleCallAccessor(Accessor[Any]):
    """Trace-level accessor for type-strict ModuleCall lookups."""

    def __init__(self, calls: Mapping[str, Any]) -> None:
        """Initialize from call-label keyed ModuleCalls."""

        super().__init__(calls)
        self._address_index: dict[Any, list[Any]] | None = None

    def _resolve_substring(self, key: str) -> Any | None:
        """Resolve unique bare Module address to its only ModuleCall."""

        if self._address_index is None:
            # Grouping by address in list order reproduces the per-lookup filter
            # exactly; lifetime is this accessor instance's immutable ``_list``.
            self._address_index = _group_by_attr(self._list, "address")
        parent_matches = self._address_index.get(key, ())
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
        self._label_index: dict[Any, list[Any]] | None = None

    def _resolve_substring(self, key: str) -> Any | None:
        """Resolve unique bare GradFn label to its only GradFnCall."""

        if self._label_index is None:
            # Grouping by label in list order reproduces the per-lookup filter
            # exactly; lifetime is this accessor instance's immutable ``_list``.
            self._label_index = _group_by_attr(self._list, "label")
        parent_matches = self._label_index.get(key, ())
        if len(parent_matches) == 1:
            return parent_matches[0]
        if len(parent_matches) > 1:
            raise AmbiguousOpLookupError(
                f"GradFn '{key}' has {len(parent_matches)} calls. Use a 0-based integer "
                f"position or a call-qualified label like '{key}:1'."
            )
        return None


# EVERY lazy Trace accessor is memoized on the owning Trace instance, NOT in
# a module-global weak-keyed dict. A record held by an accessor can reach its
# Trace strongly (``ModuleCall._source_trace``, and since M11 any fork record
# through ``OpStoreView.record_translator`` -> translated shells ->
# ``_source_trace_strong`` stamped by ``_rebind_fork_owner_refs``), so a
# module-global WeakKeyDictionary value would reach its own weak key and the
# Trace could never be collected. That class re-opened a THIRD time as the
# ``trace.run()`` fork leak (R37): the op-accessor cache entry populated
# during ``run()`` pinned the entire result fork, payloads included, for the
# process lifetime. An instance attribute makes the same edge an ordinary
# intra-object cycle that ``gc`` collects normally. All three attrs are
# registered ``FieldPolicy.DROP`` in ``Trace.PORTABLE_STATE_SPEC``, like the
# other lazy per-instance solutions. Do NOT reintroduce a module-global
# weak-keyed cache whose value holds records.
_TRACE_OP_ACCESSOR_ATTR = "_op_accessor_cache"
_TRACE_LAYER_ACCESSOR_ATTR = "_layer_accessor_cache"
_TRACE_MODULE_CALL_ACCESSOR_ATTR = "_module_call_accessor"


def _invalidate_trace_op_layer_accessor_caches(trace: Any) -> None:
    """Drop the cached op/layer accessors for one Trace.

    Parameters
    ----------
    trace:
        Trace whose layer or op population changed.
    """

    instance_dict = getattr(trace, "__dict__", None)
    if instance_dict is not None:
        instance_dict.pop(_TRACE_OP_ACCESSOR_ATTR, None)
        instance_dict.pop(_TRACE_LAYER_ACCESSOR_ATTR, None)


def _invalidate_trace_module_call_accessor_cache(trace: Any) -> None:
    """Drop the cached flattened ModuleCall accessor for one Trace.

    Parameters
    ----------
    trace:
        Trace whose module-call hierarchy changed.
    """

    instance_dict = getattr(trace, "__dict__", None)
    if instance_dict is not None:
        instance_dict.pop(_TRACE_MODULE_CALL_ACCESSOR_ATTR, None)
