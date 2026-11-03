"""Member-list coercion behind ``Bundle`` construction, ``add`` and ``remove``.

The functions normalise the construction shapes a bundle accepts (a mapping,
a sequence of traces, a sequence of ``(name, trace)`` pairs) into ordered
``(name, trace)`` pairs, refuse non-capture members typed, and resolve
member references (a name or the member object) to member names.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from ..errors._episode import BundleRelationError
from ..intervention.errors import BaselineUndeterminedError, BundleMemberError

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from . import Bundle


def parse_members(
    members: Mapping[str, Trace] | Sequence[Trace] | Sequence[tuple[str, Trace]],
    *,
    names: Sequence[str] | None,
) -> list[tuple[str, Trace]]:
    """Normalize supported construction shapes to ordered member pairs.

    Raises
    ------
    ValueError
        When ``names`` does not fit the members or member names repeat.
    BundleMemberError
        ``bundle_member_type_invalid`` for a member that is not a capture product.
    """

    if isinstance(members, Mapping):
        if names is not None:
            raise ValueError("names= is not accepted when Bundle members are a mapping.")
        pairs = [(str(name), log) for name, log in members.items()]
    else:
        values = list(members)
        if names is not None:
            if len(names) != len(values):
                raise ValueError("names length must match member count.")
            named_logs = cast("Sequence[Trace]", values)
            pairs = [(str(name), log) for name, log in zip(names, named_logs)]
        elif values and all(is_name_log_tuple(value) for value in values):
            tuple_values = cast("Sequence[tuple[str, Trace]]", values)
            pairs = [(str(member_name), log) for member_name, log in tuple_values]
        else:
            log_values = cast("Sequence[Trace]", values)
            pairs = dedupe_default_names(
                [
                    (derive_name(log, name=None, index=index), log)
                    for index, log in enumerate(log_values)
                ]
            )

    for member_name, log in pairs:
        require_trace(log, arg_name=f"member {member_name!r}")
    # O(n) duplicate detection (r8 R52): the prior per-member full
    # name-list rebuild + count was quadratic (measured 4.1s at 8k
    # members on the all-valid path).
    name_counts = Counter(name for name, _ in pairs)
    duplicate_names = sorted(name for name, count in name_counts.items() if count > 1)
    if duplicate_names:
        raise ValueError(f"Bundle member names must be unique; duplicates: {duplicate_names}")
    return pairs


def is_list_like(value: Any) -> bool:
    """Whether ``value`` is a non-string sequence (a list input)."""

    return isinstance(value, Sequence) and not isinstance(value, str)


def coerce_trace_list(value: Trace | Sequence[Trace], *, arg_name: str) -> list[Trace]:
    """Normalize a Trace-or-list input to a list of validated traces."""

    values = list(value) if is_list_like(value) else [cast("Trace", value)]
    for item in values:
        require_trace(item, arg_name=arg_name)
    return cast(list["Trace"], values)


def require_trace(value: Any, *, arg_name: str) -> None:
    """Refuse a non-Trace membership candidate, typed (ledger memo item 0).

    An experiment container with unvalidated membership cannot carry a
    provenance join (the measured defect: a dict silently became
    ``member_3``). Legitimate members are capture products: ``Trace``
    (live or loaded) and ``PartialTrace`` (failed-capture recovery — the
    episode status fold reads failed partial members).

    Raises
    ------
    BundleMemberError
        ``bundle_member_type_invalid`` for any other value.
    """

    from .._data_classes.trace import Trace as _Trace
    from .._partial import PartialTrace as _PartialTrace

    if not isinstance(value, (_Trace, _PartialTrace)):
        raise BundleMemberError(
            f"Bundle {arg_name} must be a Trace or PartialTrace, got "
            f"{type(value).__name__!r}. Bundle members are capture products "
            "(live, loaded, or failed-partial); wrap other values in a "
            "capture or keep them outside the bundle.",
            code="bundle_member_type_invalid",
            received_type=type(value).__name__,
        )


def coerce_optional_name_list(
    names: str | Sequence[str] | None,
    *,
    count: int,
) -> list[str | None]:
    """Normalize optional Bundle names to one entry per added log."""

    if names is None:
        return [None] * count
    if isinstance(names, str):
        if count != 1:
            raise ValueError("A single Bundle name can only be used with one Trace.")
        return [names]
    name_list: list[str | None] = [str(name) for name in names]
    if len(name_list) != count:
        raise ValueError("names length must match added log count.")
    return name_list


def coerce_member_name_list(
    bundle: Bundle,
    value: str | Trace | Sequence[str | Trace],
) -> list[str]:
    """Normalize member references (names, traces or a sequence of either) to names."""

    values = list(value) if is_list_like(value) else [value]
    # is_list_like narrows the runtime type (sequence -> its elements, scalar -> [value]),
    # but mypy can't follow that helper, so assert the element type the narrowing guarantees.
    return [coerce_member_name(bundle, cast("str | Trace", item)) for item in values]


def coerce_member_name(bundle: Bundle, value: str | Trace) -> str:
    """Resolve one member reference (a name or the member object) to its name."""

    if isinstance(value, str):
        return value
    for name, member in bundle._members.items():
        if member is value:
            return name
    raise KeyError("Trace is not a member of this Bundle.")


def dedupe_default_names(pairs: list[tuple[str, Trace]]) -> list[tuple[str, Trace]]:
    """Suffix repeated derived member names with ``_2``, ``_3``, ..."""

    seen: dict[str, int] = {}
    deduped: list[tuple[str, Trace]] = []
    for member_name, log in pairs:
        count = seen.get(member_name, 0) + 1
        seen[member_name] = count
        if count == 1:
            deduped.append((member_name, log))
        else:
            deduped.append((f"{member_name}_{count}", log))
    return deduped


def is_name_log_tuple(value: Any) -> bool:
    """Whether ``value`` is a two-item ``(name, log)`` tuple."""

    return isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str)


def derive_name(log: Trace, *, name: str | None, index: int) -> str:
    """Derive a member name from an explicit value, the trace label, or the index."""

    if name is not None:
        return str(name)
    log_name = getattr(log, "name", None)
    if log_name:
        return str(log_name)
    return f"member_{index}"


#: Per-kind name readers behind ``bundle.<kind>.shared`` / ``bundle.<kind>.divergent``.
_KIND_NAME_GETTERS: dict[str, Callable[[Any], Any]] = {
    "ops": lambda trace: getattr(trace, "op_labels", ()),
    "layers": lambda trace: getattr(trace, "layer_labels", ()),
    "modules": lambda trace: trace.modules.keys(),
    "params": lambda trace: trace.params.keys(),
    "buffers": lambda trace: trace.buffers.keys(),
    "grad_fns": lambda trace: trace.autograd.grad_fns.keys(),
}


def resolve_baseline_name(self: Bundle, baseline: str | Trace | None) -> str | None:
    """Resolve a baseline constructor argument to a member name."""

    if baseline is None:
        return self._auto_detect_baseline()
    if isinstance(baseline, str):
        if baseline not in self._members:
            raise KeyError(f"Unknown baseline member {baseline!r}.")
        return baseline
    for name, member in self._members.items():
        if member is baseline:
            return name
    raise KeyError("Baseline Trace is not a member of this Bundle.")


def auto_detect_baseline(self: Bundle) -> str | None:
    """Auto-detect a pristine baseline when exactly one candidate exists."""

    candidates = [
        name
        for name, member in self._members.items()
        if getattr(member, "is_pristine", None) is True
    ]
    return candidates[0] if len(candidates) == 1 else None


def baseline_or_raise(self: Bundle, baseline: str | Trace | None) -> str:
    """Return a baseline name or raise for ambiguity."""

    if baseline is not None:
        resolved = self._resolve_baseline_name(baseline)
        if resolved is not None:
            return resolved
    if self._baseline_name is not None:
        return self._baseline_name
    detected = self._auto_detect_baseline()
    if detected is not None:
        self._baseline_name = detected
        return detected
    raise BaselineUndeterminedError(
        "Bundle operation requires a baseline, but no unique pristine baseline exists."
    )


def set_capacity(self: Bundle, n: int | None) -> None:
    """Set member capacity with LRU-style eviction while preserving the baseline."""

    if n is None:
        self._capacity = None
        return
    if n < 1:
        raise ValueError("Bundle capacity must be at least 1.")
    self._capacity = n
    self._enforce_capacity()


def enforce_capacity(self: Bundle) -> None:
    """Evict oldest non-baseline members when above capacity."""

    if self._capacity is None:
        return
    while len(self._members) > self._capacity:
        evictable = next(
            (name for name in self._members if name != self._baseline_name),
            None,
        )
        if evictable is None:
            break
        if self._member_relations.rows_naming(evictable):
            # R5: an IMPLICIT eviction may neither orphan relation rows
            # nor cascade them silently — refuse typed, always.
            raise BundleRelationError(
                f"Bundle capacity eviction would orphan member-relation "
                f"rows naming {evictable!r}; remove the member explicitly "
                "(cascade_relations=True) or raise the capacity (S6 R5).",
                code="bundle_member_has_relations",
                operation="eviction",
                related_members=[evictable],
            )
        self._members.pop(evictable)
        self._member_construction.pop(evictable, None)
        self._supergraph = None


def add_members(
    self: Bundle,
    traces: Trace | Sequence[Trace],
    names: str | Sequence[str] | None = None,
) -> Bundle:
    """Add one or more member logs and invalidate the cached supergraph."""

    logs = coerce_trace_list(traces, arg_name="log_or_logs")
    member_names = coerce_optional_name_list(names, count=len(logs))
    for index, log in enumerate(logs):
        member_name = derive_name(
            log,
            name=member_names[index],
            index=len(self._members),
        )
        if member_name in self._members:
            raise ValueError(f"Bundle member names must be unique; duplicate {member_name!r}.")
        self._members[member_name] = log
        self._member_construction[member_name] = {"origin": "added"}
    self._supergraph = None
    self._enforce_capacity()
    return self


def remove_members(
    self: Bundle,
    names: str | Trace | Sequence[str | Trace],
    *,
    cascade_relations: bool = False,
) -> Trace | list[Trace]:
    """Remove and return one or more members by name or Trace object."""

    is_many = is_list_like(names)
    names = coerce_member_name_list(self, names)
    unknown = [name for name in names if name not in self._members]
    if unknown:
        raise KeyError(f"Unknown bundle member(s): {sorted(unknown)}")
    new_table = self._relation_table_for_removal(
        names, cascade_relations=cascade_relations, operation="remove"
    )
    removed: list[Trace] = []
    for name in names:
        log = self._members.pop(name)
        removed.append(log)
        self._member_construction.pop(name, None)
        if self._baseline_name == name:
            self._baseline_name = None
    if new_table is not None:
        self._member_relations = new_table
    self._supergraph = None
    return removed if is_many else removed[0]


def remove_except(
    self: Bundle,
    keep: str | Trace | Sequence[str | Trace],
    *,
    cascade_relations: bool = False,
) -> None:
    """Remove every member whose name is not listed in ``keep``."""

    keep_set = set(coerce_member_name_list(self, keep))
    unknown = keep_set - set(self._members)
    if unknown:
        raise KeyError(f"Unknown bundle member(s): {sorted(unknown)}")
    removed_names = [name for name in self._members if name not in keep_set]
    new_table = self._relation_table_for_removal(
        removed_names, cascade_relations=cascade_relations, operation="remove_except"
    )
    self._members = OrderedDict(
        (name, log) for name, log in self._members.items() if name in keep_set
    )
    if self._baseline_name is not None and self._baseline_name not in self._members:
        self._baseline_name = None
    if new_table is not None:
        self._member_relations = new_table
    self._supergraph = None


def clear_members(self: Bundle, *, cascade_relations: bool = False) -> None:
    """Remove all non-baseline members."""

    keep_baseline = self._baseline_name is not None and self._baseline_name in self._members
    removed_names = [
        name for name in self._members if not (keep_baseline and name == self._baseline_name)
    ]
    new_table = self._relation_table_for_removal(
        removed_names, cascade_relations=cascade_relations, operation="clear"
    )
    if keep_baseline:
        baseline_name = cast("str", self._baseline_name)
        baseline = self._members[baseline_name]
        self._members = OrderedDict([(baseline_name, baseline)])
    else:
        self._members.clear()
        self._baseline_name = None
    for removed_name in removed_names:
        self._member_construction.pop(removed_name, None)
    if new_table is not None:
        self._member_relations = new_table
    self._supergraph = None


def kind_names(self: Bundle, kind: str, *, shared: bool) -> list[str]:
    """Names of one kind present in every member, or in some but not all."""

    getter = _KIND_NAME_GETTERS[kind]
    return self._shared(getter) if shared else self._divergent(getter)


def shared_keys(self: Bundle, key_fn: Callable[[Trace], Sequence[Any]]) -> list[str]:
    """Keys common to every member, ordered by the first member."""

    member_keys = self._member_key_lists(key_fn)
    key_sets = [set(keys) for keys in member_keys] or [set()]
    shared = set.intersection(*key_sets)
    return [key for key in (member_keys or [[]])[0] if key in shared]


def divergent_keys(self: Bundle, key_fn: Callable[[Trace], Sequence[Any]]) -> list[str]:
    """Keys present in at least one member and absent from at least one member."""

    member_keys = self._member_key_lists(key_fn)
    key_sets = [set(keys) for keys in member_keys] or [set()]
    shared = set.intersection(*key_sets)
    union = set.union(*key_sets)
    divergent = union - shared
    ordered: list[str] = []
    seen: set[str] = set()
    for keys in member_keys:
        for key in keys:
            if key in divergent and key not in seen:
                ordered.append(key)
                seen.add(key)
    return ordered


def member_key_lists(self: Bundle, key_fn: Callable[[Trace], Sequence[Any]]) -> list[list[str]]:
    """Normalized per-member key lists with duplicates removed."""

    member_keys: list[list[str]] = []
    for member in self._members.values():
        keys: list[str] = []
        seen: set[str] = set()
        for raw_key in key_fn(member):
            if raw_key is None:
                continue
            key = str(raw_key)
            if key not in seen:
                keys.append(key)
                seen.add(key)
        member_keys.append(keys)
    return member_keys
