"""Member-list coercion behind ``Bundle`` construction, ``add`` and ``remove``.

The functions normalise the construction shapes a bundle accepts (a mapping,
a sequence of traces, a sequence of ``(name, trace)`` pairs) into ordered
``(name, trace)`` pairs, refuse non-capture members typed, and resolve
member references (a name or the member object) to member names.
``released_candidates`` names the effect-table candidates the bundle no
longer holds, which ``Bundle.map`` reports as ``"unmeasured"``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from ..intervention.errors import BundleMemberError

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


def released_candidates(bundle: Bundle) -> list[str]:
    """Effect-table candidates with no retained member, sorted, each once.

    Parameters
    ----------
    bundle:
        Bundle whose effect tables are read.

    Returns
    -------
    list[str]
        Candidate ids a table records whose member was released, refused or
        failed; the ``__baseline__`` row is never listed.
    """
    released: set[str] = set()
    for table in bundle._effect_tables.values():
        for row in table.rows:
            if row.candidate_id == "__baseline__":
                continue
            if row.member_name is None or row.member_name not in bundle._members:
                released.add(row.candidate_id)
    return sorted(released)
