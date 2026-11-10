"""Trace accessor helpers."""

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from .._constants import _is_suffixed_raw_label
from .._errors import AmbiguousOpLookupError
from .._pass_tokens import PASS_TOKEN_ALL, split_pass_token
from .._utils._display_errors import lookup_message
from .._utils._display_keys import display_key
from ._accessor_base import Accessor, attach_source_honesty, first_source_trace
from ._layer_accessors import _LAYER_COUNTABLE_FIELDS
from ._lookup_kinds import converted, module_call_for_key, output_op
from ._nonfinite import nonfinite_op_labels
from ._op_door_keys import (
    OpRecordList,
    OpRecordMap,
    _AliasIndexes,
    _claim_alias,
    _group_by_layer_aliases,
    _is_counted_op,
    hold_ops,
)
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

    _COUNTABLE_FIELDS = _LAYER_COUNTABLE_FIELDS

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
                orphan.short_layer_label,
                orphan.label,
                orphan.short_label,
                orphan.layer_label,
                orphan.short_layer_label,
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
                    orphan.short_layer_label,
                    orphan.label,
                    orphan.short_label,
                    orphan.layer_label,
                    orphan.short_layer_label,
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
        passes = sorted(entry.ops.values(), key=lambda op: op.pass_number)
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


class TraceOpAccessor(Accessor[Op]):
    """Trace-level accessor for type-strict Op lookups.

    The trace's own accessor (``Trace.ops``, built with ``counted_view=True``)
    counts and iterates the ops the counted op total counts: ``len``,
    iteration, integer indexing, ``keys``/``values``/``items`` and
    ``to_pandas`` skip the input, output and buffer rows, so
    ``len(trace.ops)`` is that total. Label lookup still resolves EVERY
    row: ``trace.ops["input_1"]`` returns the input row and ``"input_1" in
    trace.ops`` is true although the label is not in ``keys()``. The
    alias-inclusive row sequence is ``trace._layer_list``. A derived accessor
    over a chosen row list (``input_ops``, ``output_ops``, ``saved_ops``, ...)
    keeps the default ``counted_view=False`` and lists exactly its rows.

    The door holds one row key per member and every index maps to row keys
    (``_op_door_keys``); the records stay cached in the trace core's side
    table. ``_list`` and ``_dict`` read the records back for code written
    against the containers the door used to hold.
    """

    _COUNTABLE_FIELDS = _LAYER_COUNTABLE_FIELDS

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
            Count and iterate only the ops the counted op total counts (the
            ``Trace.ops`` view); ``False`` lists every row given.
        """

        members = list(ops)
        self._table, self._keys = hold_ops(members, first_source_trace(members))
        label_keys: dict[str, int] = {}
        self._raw_index_lookup: dict[int, int] = {}
        # Final Op labels name exactly one row. Two rows sharing one is corrupt
        # labeling; recording it here makes every lookup of that label refuse
        # instead of silently answering with whichever row came last.
        duplicate_labels: dict[str, list[int]] = {}
        for held, op in zip(self._keys, members, strict=True):
            label = op.label
            if label in label_keys and label_keys[label] != held and isinstance(label, str):
                duplicate_labels.setdefault(label, [label_keys[label]]).append(held)
            label_keys[label] = held
            self._raw_index_lookup[op.raw_index] = held
        self._label_keys = label_keys
        self._duplicate_labels = duplicate_labels
        self._counted_cache: list[int] | None = None
        self._counted_view = counted_view
        self._layer_num_calls = dict(layer_num_calls)
        # Lazily built reverse indexes for ``_resolve_substring``; see
        # ``_ensure_alias_indexes``. Lifetime is this accessor instance, whose
        # ``_keys`` is immutable post-finalization and already rebuilt through the
        # existing ``_TRACE_OP_ACCESSOR_CACHE`` invalidation, so this adds no new
        # invalidation surface.
        self._alias_indexes: _AliasIndexes | None = None
        self._alias_index_unavailable = False
        self._resolved_op_cache: dict[str, int] = {}
        # Bare layer label -> its Ops' row keys in list order; see ``_layer_label_matches``.
        self._layer_label_index: dict[Any, list[int]] | None = None

    @property
    def _list(self) -> OpRecordList:  # type: ignore[override]
        """Every member record in order, read through the row keys."""

        return OpRecordList(self._table, self._keys)

    @property
    def _dict(self) -> OpRecordMap:  # type: ignore[override]
        """Label to member record, read through the row keys."""

        return OpRecordMap(self._table, self._label_keys)

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state holding this door's records, not the trace-wide side table."""

        state = super().__getstate__()
        state["_table"] = self._table.only(self._keys)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickle state; a door pickled while it held records is rebuilt from them."""

        if "_table" in state:
            super().__setstate__(state)
            return
        calls, counted = state.get("_layer_num_calls", {}), state.get("_counted_view", False)
        TraceOpAccessor.__init__(self, state["_list"], calls, counted_view=counted)

    def _ops(self, keys: Iterable[int]) -> list[Op]:
        """The records behind ``keys``, in order."""

        table = self._table
        return list(map(table.rows.__getitem__, keys) if table.plain else table.records(keys))

    def _counted_keys(self) -> list[int]:
        """The row keys of the counted op rows, in capture order.

        Flags settle at finalization (``_tracing_finished`` is mirrored onto
        every op), so the list is memoized only once the last row is
        finished; mid-capture reads recompute it.
        """

        keys = self._keys
        if not self._counted_view:
            return keys
        cached = self._counted_cache
        if cached is not None:
            return cached
        table = self._table
        members = zip(keys, table.records(keys), strict=True)
        counted = [held for held, op in members if _is_counted_op(op)]
        if keys and getattr(table.record(keys[-1]), "_tracing_finished", False):
            self._counted_cache = counted
        return counted

    def _counted_ops(self) -> list[Op]:
        """The counted op rows, in capture order."""

        return self._ops(self._counted_keys())

    def _refuse_duplicate_label(self, key: Any) -> None:
        """Raise when ``key`` is a member label two rows share."""

        claimed = self._duplicate_labels.get(key)
        if claimed:
            _refuse_colliding_label(key, self._ops(claimed))

    def __getitem__(self, key: int | str) -> Op:
        """Return a counted op by ordinal, or any row by label."""

        if isinstance(key, int):
            held = self._counted_keys()[key]
        else:
            if self._duplicate_labels:
                self._refuse_duplicate_label(key)
            label_keys = self._label_keys
            if key not in label_keys:
                return super().__getitem__(key)
            held = label_keys[key]
        table = self._table
        return table.rows[held] if held >= 0 else table.detached[held]

    def __iter__(self) -> Iterator[Op]:
        """Iterate over the counted ops in capture order."""

        return self._table.records(self._counted_keys())

    def __len__(self) -> int:
        """Return the counted op total (equals the counted op total)."""

        return len(self._counted_keys())

    def keys(self) -> list[Any]:
        """Return the labels of the counted ops."""

        return [op.label for op in self._counted_ops()]

    def values(self) -> list[Op]:
        """Return the counted ops."""

        return self._counted_ops()

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

    def _derived(self, ops: list[Op]) -> "TraceOpAccessor":
        """A collection listing exactly ``ops``, sharing this accessor's pass counts."""

        return TraceOpAccessor(ops, self._layer_num_calls)

    @property
    def compute(self) -> "TraceOpAccessor":
        """Access the compute ops: every member that is not an input, output or buffer row.

        The Op spelling of ``trace.layers.compute``; on ``trace.ops`` it lists the
        counted ops, on a derived collection (``trace.saved_ops.compute``) the
        members that pass the same filter.
        """

        return self._derived([op for op in self._table.records(self._keys) if _is_counted_op(op)])

    @property
    def nonfinite(self) -> "TraceOpAccessor":
        """Access the members whose output held NaN or Inf.

        The Op spelling of ``Trace.nonfinite_ops`` restricted to this collection;
        ``Trace.nonfinite_coverage`` names what the answer could not examine.
        Empty when no member still reaches its trace.
        """

        members = self._ops(self._keys)
        trace = first_source_trace(members)
        if trace is None:
            return self._derived([])
        flagged = set(nonfinite_op_labels(trace))
        return self._derived([op for op in members if op.label in flagged])

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
            held = self._raw_index_lookup[raw_index]
        except KeyError as exc:
            raise KeyError(f"Op raw_index {raw_index} not found.") from exc
        table = self._table
        return table.rows[held] if held >= 0 else table.detached[held]

    def _resolve_pass_qualified(self, key: str) -> Op | None:
        """Resolve pass-qualified Op labels without returning parent Layers."""

        if self._duplicate_labels:
            self._refuse_duplicate_label(key)
        label_keys = self._label_keys
        if key in label_keys:
            held, table = label_keys[key], self._table
            return table.rows[held] if held >= 0 else table.detached[held]
        token = split_pass_token(key)
        if token is None:
            return None
        passes = self._layer_label_matches(token[0])
        if not passes:
            return None
        if token[1] == PASS_TOKEN_ALL:
            raise KeyError(
                f"{key!r} names every pass of {token[0]!r}, which an Op collection cannot "
                f"return as one record; trace[{key!r}] gives the Selection and "
                f"trace.ops.resolve_all({token[0]!r}) the list of Ops."
            )
        return max(passes, key=lambda op: int(getattr(op, "pass_number", 1) or 1))

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
        return self._layer_label_matches(key)

    def _layer_label_matches(self, key: str) -> list[Op]:
        """Every Op whose layer label or short layer label is ``key``, in list order.

        ``trace[label]`` resolves through ``resolve_all`` for every string key, and
        the invariant checks look up one label per graph edge, so a scan of every
        row per lookup made validation quadratic in graph size. Once the rows are
        settled (the last row is finished, the predicate ``_counted_ops`` memoizes
        on) a reverse index built in one pass answers instead; it lives on this
        accessor instance, whose ``_keys`` is replaced (with the accessor) on any
        rebuild or rename. Before settlement the scan answers. A failed index build
        is not cached and the scan answers, raising exactly as it always did.
        """

        index = self._layer_label_index
        keys, table = self._keys, self._table
        if index is None and keys and getattr(table.record(keys[-1]), "_tracing_finished", False):
            try:
                index = _group_by_layer_aliases(keys, table.records(keys))
            except Exception:
                index = None
            self._layer_label_index = index
        if index is None:
            scanned = table.records(keys)
            return [op for op in scanned if key in {op.layer_label, op.short_layer_label}]
        held = index.get(key, ())
        return list(map(table.rows.__getitem__, held) if table.plain else table.records(held))

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
        it. The bare label of a multi-pass layer outranks the raw tier as well:
        it refuses as ambiguous rather than answering with the Op whose raw
        spelling it equals. A raw spelling shadowed that way stays reachable
        through its suffixed form or :meth:`by_raw_index`.

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
        final_index: dict[Any, int] = {}
        collisions: dict[Any, list[int]] = {}
        raw_index: dict[Any, int] = {}
        layer_alias_index: dict[Any, list[int]] = {}
        try:
            for held, op in zip(self._keys, self._table.records(self._keys), strict=True):
                for alias in self._final_aliases(op):
                    _claim_alias(final_index, collisions, alias, held)
                raw_index.setdefault(op.raw_label, held)
                for alias in {op.layer_label, op.short_layer_label}:
                    layer_alias_index.setdefault(alias, []).append(held)
        except Exception:
            self._alias_index_unavailable = True
            return None
        self._alias_indexes = _AliasIndexes(final_index, collisions, raw_index, layer_alias_index)
        return self._alias_indexes

    def _final_aliases(self, op: Op) -> list[Any]:
        """Return the final-tier aliases of one Op (see ``_ensure_alias_indexes``)."""

        aliases = [op.label, op.short_label]
        label_raw = op._label_raw
        # Only a SUFFIXED raw label is final-tier: ``x_raw`` cannot be a final
        # label. An unsuffixed ``_label_raw`` has the raw-spelling shape and is
        # served by the raw tier (``Op.raw_label`` returns it unchanged).
        if _is_suffixed_raw_label(label_raw):
            aliases.append(label_raw)
        if self._layer_num_calls.get(op.layer_label, 0) == 1:
            aliases.extend((op.layer_label, op.short_layer_label))
        return aliases

    def _resolve_substring_by_scan(self, key: str) -> Op | None:
        """Resolve by linear scan, used when the alias indexes are unavailable."""

        members = self._ops(self._keys)
        final_matches: list[Op] = []
        for op in members:
            if key in self._final_aliases(op) and op not in final_matches:
                final_matches.append(op)
        _refuse_colliding_label(key, final_matches if len(final_matches) > 1 else None)
        if final_matches:
            return final_matches[0]
        parent_matches = [op for op in members if key in {op.layer_label, op.short_layer_label}]
        resolved = self._resolve_parent_matches(parent_matches)
        if resolved is not None:
            return resolved
        for op in members:
            if key == op.raw_label:
                return op
        # A module, module-call or buffer key names its output Op here.
        return converted(self, key, output_op)

    @staticmethod
    def _resolve_parent_matches(parent_matches: Sequence[Op]) -> Op | None:
        """Return the unique bare-parent-label match, or raise when ambiguous."""

        if len(parent_matches) == 1:
            return parent_matches[0]
        if len(parent_matches) > 1:
            parent_label = parent_matches[0].short_layer_label
            raise AmbiguousOpLookupError(
                lookup_message(
                    f"Layer {parent_label!r} has {len(parent_matches)} ops.",
                    [display_key(op) for op in parent_matches],
                    "Use a 0-based integer position or a pass-qualified label.",
                    query=parent_label,
                )
            )
        return None

    def _resolve_substring(self, key: str) -> Op | None:
        """Resolve exact long/short Op labels, unique bare parent labels, or a record key's output Op."""

        indexes = self._ensure_alias_indexes()
        if indexes is None:
            return self._resolve_substring_by_scan(key)
        claimed = indexes.collisions.get(key)
        if claimed:
            _refuse_colliding_label(key, self._ops(claimed))
        direct = indexes.final.get(key)
        if direct is None:
            # A multi-pass layer's bare label is final naming too: it refuses as
            # ambiguous before another Op's raw spelling can answer for it.
            resolved = self._resolve_parent_matches(self._ops(indexes.layer.get(key, ())))
            if resolved is not None:
                return resolved
            direct = indexes.raw.get(key)
            if direct is None:
                return converted(self, key, output_op)
        table = self._table
        return table.rows[direct] if direct >= 0 else table.detached[direct]

    def _resolved_op(self, key: str) -> Op:
        """Return ``self[key]``, memoized per lookup key.

        Callers that resolve the same stored child/parent label once per graph edge
        (the ``_trace_views`` edge counters) would otherwise repeat one full
        ``__getitem__`` per edge. Failed lookups are never memoized, so a missing or
        ambiguous key raises exactly as ``self[key]`` does on every call.
        """

        held = self._resolved_op_cache.get(key)
        table = self._table
        if held is None:
            op = self[key]
            known = table.key_of(op)
            if known is not None:
                self._resolved_op_cache[key] = known
            return op
        return table.rows[held] if held >= 0 else table.detached[held]


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
        return converted(self, key, module_call_for_key)  # an op label names its call


class TraceGradFnCallAccessor(Accessor[Any]):
    """Trace-level accessor for type-strict GradFnCall lookups."""

    def __init__(self, calls: Mapping[str, Any]) -> None:
        """Initialize from call-label keyed GradFnCalls."""

        super().__init__(calls)
        self._label_index: dict[Any, list[Any]] | None = None

    def _resolve_substring(self, key: str) -> Any | None:
        """Resolve unique bare GradFn label to its only GradFnCall."""

        if self._label_index is None:
            # Grouping by the parent GradFn label in list order reproduces the per-lookup filter
            # exactly; lifetime is this accessor instance's immutable ``_list``.
            self._label_index = _group_by_attr(self._list, "grad_fn_label")
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
# a module-global weak-keyed dict. A record held by an accessor could reach
# its Trace (``ModuleCall._source_trace``, and since any fork record
# through ``OpStoreView.record_translator`` -> translated shells stamped by
# ``_rebind_fork_owner_refs``), so a module-global WeakKeyDictionary value
# would reach its own weak key and the Trace could never be collected. That
# class re-opened a THIRD time as the
# ``trace.run()`` fork leak: the op-accessor cache entry populated
# during ``run()`` pinned the entire result fork, payloads included, for the
# process lifetime. An instance attribute makes the same edge an ordinary
# intra-object cycle that ``gc`` collects normally. All three attrs are
# registered ``FieldPolicy.DROP`` in ``Trace._PORTABLE_STATE_SPEC``, like the
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
