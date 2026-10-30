"""Trace accessor helpers."""

from collections import Counter, OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .._errors import AmbiguousOpLookupError
from ..constants import is_suffixed_raw_label
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


@dataclass(frozen=True)
class _AliasIndexes:
    """Reverse indexes behind ``TraceOpAccessor._resolve_substring``.

    ``final`` maps each final-tier alias to its Op, ``collisions`` the final-tier
    aliases two Ops claim, ``raw`` each unsuffixed raw spelling to its Op, and
    ``layer`` each bare layer label to its Ops in list order.
    """

    final: dict[Any, Op]
    collisions: dict[Any, list[Op]]
    raw: dict[Any, Op]
    layer: dict[Any, list[Op]]


def _claim_alias(index: dict[Any, Op], collisions: dict[Any, list[Op]], alias: Any, op: Op) -> None:
    """Record ``alias -> op``; a second, different Op makes the alias a collision."""

    existing = index.setdefault(alias, op)
    if existing is op or not isinstance(alias, str):
        return
    claimants = collisions.setdefault(alias, [existing])
    if op not in claimants:
        claimants.append(op)


def _refuse_colliding_label(key: Any, claimants: Sequence[Op] | None) -> None:
    """Raise when one final label names several Ops, rather than guess one."""

    if not claimants:
        return
    names = ", ".join(f"{op.label!r} (raw {op._label_raw!r})" for op in claimants[:10])
    raise AmbiguousOpLookupError(
        f"Op label {key!r} is claimed by {len(claimants)} different ops: {names}. "
        "A final label must name exactly one op; this is corrupt labeling, not an "
        "ambiguous user key."
    )


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
        # Final Op labels name exactly one row. Two rows sharing one is corrupt
        # labeling; recording it here makes every lookup of that label refuse
        # instead of silently answering with whichever row came last.
        duplicate_labels: dict[str, list[Op]] = {}
        for op in ops:
            label = op.label
            existing = op_lookup.get(label)
            if isinstance(label, str) and existing is not None and existing is not op:
                duplicate_labels.setdefault(label, [existing]).append(op)
            op_lookup[label] = op
            self._raw_index_lookup[op.raw_index] = op
        super().__init__(op_lookup, item_list=list(ops))
        self._duplicate_labels = duplicate_labels
        self._counted_cache: list[Op] | None = None
        self._counted_view = counted_view
        self._layer_num_calls = dict(layer_num_calls)
        # Lazily built reverse indexes for ``_resolve_substring``; see
        # ``_ensure_alias_indexes``. Lifetime is this accessor instance, whose
        # ``_list`` is immutable post-finalization and already rebuilt through the
        # existing ``_TRACE_OP_ACCESSOR_CACHE`` invalidation, so this adds no new
        # invalidation surface.
        self._alias_indexes: _AliasIndexes | None = None
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
        _refuse_colliding_label(key, getattr(self, "_duplicate_labels", {}).get(key))
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

        _refuse_colliding_label(key, getattr(self, "_duplicate_labels", {}).get(key))
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

    def _ensure_alias_indexes(self) -> _AliasIndexes | None:
        """Build the alias reverse indexes once, or report them unavailable.

        Aliases come in two tiers, and a final-label alias always outranks a raw
        one. The final tier is what graph edges, ``Trace.__getitem__`` and users
        name: the pass-qualified label, its short form, the suffixed raw label
        (``x_raw`` cannot collide with a final label; an unsuffixed one is a raw
        spelling and joins the raw tier), and the bare layer label
        of a single-pass layer. The raw tier is the unsuffixed raw spelling
        (``Op.raw_label``), whose ``{type}_{type_idx}_{raw_idx}`` shape matches
        the final ``{type}_{type_idx}_{step_idx}`` shape. Once grouping folds
        passes into one layer the two numberings drift apart, and one Op's raw
        spelling can equal another Op's final label. Before the tiers, list
        order decided that tie, so a final label referenced by the graph could
        resolve to a different, earlier Op whose raw spelling happened to match
        it. A raw spelling shadowed that way stays reachable through its
        suffixed form or :meth:`by_raw_index`.

        A final-tier alias claimed by two different Ops is corrupt labeling and
        is kept in a collision table so the lookup refuses with
        ``AmbiguousOpLookupError`` instead of picking one. The third index
        reproduces the bare-parent-label match list in list order.

        Building touches label attributes on every Op, which the early-returning
        scan may never have reached. If any read fails (e.g. an unfinished Op whose
        label slot is unset mid-capture), the indexes are abandoned and the scan is
        used instead, which applies the same tiers.

        Returns
        -------
        _AliasIndexes | None
            The index bundle, or ``None`` when the indexes could not be built and
            the scan must be used.
        """

        if self._alias_indexes is not None:
            return self._alias_indexes
        if self._alias_index_unavailable:
            return None
        final_index: dict[Any, Op] = {}
        collisions: dict[Any, list[Op]] = {}
        raw_index: dict[Any, Op] = {}
        layer_alias_index: dict[Any, list[Op]] = {}
        try:
            for op in self._list:
                for alias in self._final_aliases(op):
                    _claim_alias(final_index, collisions, alias, op)
                raw_index.setdefault(op.raw_label, op)
                for alias in {op.layer_label, op.layer_label_short}:
                    layer_alias_index.setdefault(alias, []).append(op)
        except Exception:
            self._alias_index_unavailable = True
            return None
        self._alias_indexes = _AliasIndexes(final_index, collisions, raw_index, layer_alias_index)
        return self._alias_indexes

    def _final_aliases(self, op: Op) -> list[Any]:
        """Return the final-tier aliases of one Op (see ``_ensure_alias_indexes``)."""

        aliases = [op.label, op.label_short]
        label_raw = op._label_raw
        # Only a SUFFIXED raw label is final-tier: ``x_raw`` cannot be a final
        # label. An unsuffixed ``_label_raw`` has the raw-spelling shape and is
        # served by the raw tier (``Op.raw_label`` returns it unchanged).
        if is_suffixed_raw_label(label_raw):
            aliases.append(label_raw)
        if self._layer_num_calls.get(op.layer_label, 0) == 1:
            aliases.extend((op.layer_label, op.layer_label_short))
        return aliases

    def _resolve_substring_by_scan(self, key: str) -> Op | None:
        """Resolve by linear scan, used when the alias indexes are unavailable."""

        final_matches: list[Op] = []
        for op in self._list:
            if key in self._final_aliases(op) and op not in final_matches:
                final_matches.append(op)
        _refuse_colliding_label(key, final_matches if len(final_matches) > 1 else None)
        if final_matches:
            return final_matches[0]
        for op in self._list:
            if key == op.raw_label:
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
        _refuse_colliding_label(key, indexes.collisions.get(key))
        direct = indexes.final.get(key)
        if direct is None:
            direct = indexes.raw.get(key)
        if direct is not None:
            return direct
        return self._resolve_parent_matches(indexes.layer.get(key, ()))

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
