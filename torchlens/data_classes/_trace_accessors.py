"""Trace accessor helpers."""

from collections import Counter, OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from .._errors import AmbiguousOpLookupError
from ._accessor_base import Accessor, attach_source_honesty
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


def _boundary_listing_passes(
    labels: Sequence[str], resolve: Callable[[str], Any]
) -> Iterator[list[Op]]:
    """Yield the Ops each boundary listing names, one list per listing.

    The one per-pass rule behind :func:`boundary_label_ops` and
    :func:`boundary_label_rows`: an Op entry names itself; a Layer listed as
    many times as it has passes names pass k at its k-th listing (the list and
    the Layer's passes are both in capture order); a Layer listed a different
    number of times names every pass at each listing.
    """

    listings = Counter(labels)
    taken: Counter[str] = Counter()
    for label in labels:
        entry = resolve(label)
        if isinstance(entry, Op):
            yield [entry]
            continue
        passes = sorted(entry.ops.values(), key=lambda op: op.pass_index)
        if listings[label] == len(passes):
            yield [passes[taken[label]]]
            taken[label] += 1
        else:
            yield passes


def boundary_label_ops(labels: Sequence[str], resolve: Callable[[str], Any]) -> list[Op]:
    """Resolve a boundary label list (``input_layers``, ``output_layers``) to its Ops.

    Torch stores these lists in layer space: the bare layer label of each
    boundary row, once per row. When recurrence grouping folds several
    boundary rows into one multi-pass Layer, its bare label is listed once
    per pass and resolves to that Layer, not to an Op. Both the list and the
    Layer's passes are in capture order, so the k-th listing names pass k; a
    Layer listed a different number of times than it has passes contributes
    every pass at its first listing. Each Op appears once, in list order.

    Parameters
    ----------
    labels:
        Ordered boundary labels.
    resolve:
        Label lookup (``Trace.__getitem__``) returning an Op or a Layer.

    Returns
    -------
    list[Op]
        The boundary Ops, one per row.
    """

    ops: dict[str, Op] = {}
    for chosen in _boundary_listing_passes(labels, resolve):
        for op in chosen:
            ops.setdefault(op.label, op)
    return list(ops.values())


def boundary_label_rows(labels: Sequence[str], resolve: Callable[[str], Any]) -> list[Op | None]:
    """Resolve a boundary label list to one Op per listing, aligned with ``labels``.

    The same per-pass rule as :func:`boundary_label_ops`, for positional
    readers that pair each listing with a runtime value. A listing the rule
    cannot pin to one pass (a multi-pass Layer listed a different number of
    times than it has passes) is ``None``.

    Parameters
    ----------
    labels:
        Ordered boundary labels.
    resolve:
        Label lookup (``Trace.__getitem__``) returning an Op or a Layer.

    Returns
    -------
    list[Op | None]
        One entry per listing, in list order.
    """

    return [
        chosen[0] if len(chosen) == 1 else None
        for chosen in _boundary_listing_passes(labels, resolve)
    ]


def _is_counted_op(op: Op) -> bool:
    """Whether an op row counts toward ``Trace.num_ops``.

    The count excludes the input, output and buffer rows, the definition in
    ``backends/_finalize.py``; those rows are aliases of a tensor, not ops.
    """

    return not (
        getattr(op, "is_input", False)
        or getattr(op, "is_output", False)
        or getattr(op, "is_buffer", False)
    )


class TraceOpAccessor(Accessor[Op]):
    """Trace-level accessor for type-strict Op lookups.

    The trace's own accessor (``Trace.ops``, built with ``counted_view=True``)
    counts and iterates the ops ``Trace.num_ops`` counts: ``len``,
    iteration, integer indexing, ``keys``/``values``/``items`` and
    ``to_pandas`` skip the input, output and buffer rows, so
    ``len(trace.ops) == trace.num_ops``. Label lookup still resolves EVERY
    row: ``trace.ops["input_1"]`` returns the input row and ``"input_1" in
    trace.ops`` is true although the label is not in ``keys()``. The
    alias-inclusive row sequence is ``trace.layer_list``. A derived accessor
    over a chosen row list (``input_ops``, ``output_ops``, ``saved_ops``, ...)
    keeps the default ``counted_view=False`` and lists exactly its rows.
    """

    def __init__(
        self,
        ops: Sequence[Op],
        layer_num_calls: Mapping[str, int],
        *,
        counted_view: bool = False,
    ) -> None:
        """Initialize from ordered Op records.

        Parameters
        ----------
        ops:
            Ordered Op records.
        layer_num_calls:
            Mapping from parent Layer label to number of Op passes.
        counted_view:
            Count and iterate only the ops ``Trace.num_ops`` counts (the
            ``Trace.ops`` view); ``False`` lists every row given.
        """

        op_lookup: OrderedDict[str, Op] = OrderedDict()
        self._raw_index_lookup: dict[int, Op] = {}
        for op in ops:
            op_lookup[op.label] = op
            self._raw_index_lookup[op.raw_index] = op
        super().__init__(op_lookup, item_list=list(ops))
        self._counted_cache: list[Op] | None = None
        self._counted_view = counted_view
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

    def _counted_ops(self) -> list[Op]:
        """The counted op rows, in capture order.

        Flags settle at finalization (``_tracing_finished`` is mirrored onto
        every op), so the list is memoized only once the last row is
        finished; mid-capture reads recompute it.
        """

        if not getattr(self, "_counted_view", False):
            return self._list
        cached = getattr(self, "_counted_cache", None)
        if cached is not None:
            return cached
        counted = [op for op in self._list if _is_counted_op(op)]
        if self._list and getattr(self._list[-1], "_tracing_finished", False):
            self._counted_cache = counted
        return counted

    def __getitem__(self, key: int | str) -> Op:
        """Return a counted op by ordinal, or any row by label."""

        if isinstance(key, int):
            return self._counted_ops()[key]
        return super().__getitem__(key)

    def __iter__(self) -> Iterator[Op]:
        """Iterate over the counted ops in capture order."""

        return iter(self._counted_ops())

    def __len__(self) -> int:
        """Return the counted op total (equals ``Trace.num_ops``)."""

        return len(self._counted_ops())

    def keys(self) -> list[Any]:
        """Return the labels of the counted ops."""

        return [op.label for op in self._counted_ops()]

    def values(self) -> list[Op]:
        """Return the counted ops."""

        return list(self._counted_ops())

    def items(self) -> list[tuple[Any, Op]]:
        """Return ``(label, op)`` pairs for the counted ops."""

        return [(op.label, op) for op in self._counted_ops()]

    def to_pandas(self) -> Any:
        """Return one row per counted op (the same rows ``len`` counts)."""

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        counted = self._counted_ops()
        frames = [op.to_pandas() for op in counted if hasattr(op, "to_pandas")]
        if not frames:
            return pd.DataFrame()
        return attach_source_honesty(pd.concat(frames, ignore_index=True), counted)

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
