"""Pair the module calls of two traces without looking at any value.

Pairing works top down from the root calls. Under each paired parent pair, the
children of both sides are matched by key, in this order of evidence:

1. ``module_map`` rewrites a side-a address prefix to a side-b prefix; a pair found
   this way is ``asserted`` (the user said so).
2. The same address on both sides is ``supported``.
3. The same address after dropping never-called container segments
   (:func:`~torchlens.debug._xb_calls.normalized_address`) is ``supported``, with
   the normalization disclosed. A normalized key shared by two different addresses
   on one side is ``ambiguous`` and pairs nothing.

Repeated calls under one key pair by observed call order; an extra call on one side
is ``absent`` (a coverage gap), never paired with something else. A call left
unpaired hands its children up to its parent's pool, so a wrapper module present on
one side only does not hide the modules inside it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ._xb_calls import CallInfo, normalized_address

SUPPORTED = "supported"
ASSERTED = "asserted"
AMBIGUOUS = "ambiguous"
ABSENT = "absent"


@dataclass(frozen=True)
class CallPair:
    """One row of module pairing: a call on each side, or one side only.

    Parameters
    ----------
    a, b:
        The paired calls; one of them is ``None`` for an unpaired call.
    alignment:
        ``supported``, ``asserted``, ``ambiguous`` or ``absent``.
    note:
        Disclosure of how the pair was made (``""`` for the same address).
    """

    a: CallInfo | None
    b: CallInfo | None
    alignment: str
    note: str


def _mapped(address: str, module_map: Mapping[str, str]) -> str | None:
    """Rewrite ``address`` through the longest matching ``module_map`` prefix."""

    best = None
    for source in module_map:
        if address == source or address.startswith(f"{source}."):
            if best is None or len(source) > len(best):
                best = source
    if best is None:
        return None
    return module_map[best] + address[len(best) :]


class _Side:
    """One trace's calls indexed for pairing."""

    def __init__(self, calls: list[CallInfo]) -> None:
        """Index calls by label and by parent, in call order."""

        self.by_label = {call.label: call for call in calls}
        self.children: dict[str | None, list[CallInfo]] = {}
        for call in sorted(calls, key=lambda info: (info.enter, info.depth)):
            self.children.setdefault(call.parent, []).append(call)
        called = frozenset(call.address for call in calls)
        self.norm = {call.address: normalized_address(call.address, called) for call in calls}


def _norm_groups(calls: list[CallInfo], side: _Side) -> dict[str, set[str]]:
    """Map each normalized key to the distinct addresses that produce it."""

    groups: dict[str, set[str]] = {}
    for call in calls:
        groups.setdefault(side.norm[call.address], set()).add(call.address)
    return groups


def _keys(
    pool_a: list[CallInfo],
    pool_b: list[CallInfo],
    sides: tuple[_Side, _Side],
    module_map: Mapping[str, str],
) -> tuple[dict[str, tuple[str, str]], set[str]]:
    """Return ``{a-address: (kind, b-address)}`` and the ambiguous a-addresses.

    Mapped and exact keys are taken first; a normalized key is taken only when it
    names exactly one unclaimed address on each side.
    """

    side_a, side_b = sides
    b_addresses = {call.address for call in pool_b}
    keys_a: dict[str, tuple[str, str]] = {}
    for call in pool_a:
        mapped = _mapped(call.address, module_map)
        if mapped is not None:
            keys_a[call.address] = (ASSERTED, mapped)
        elif call.address in b_addresses:
            keys_a[call.address] = (SUPPORTED, call.address)
    claimed = {target for _kind, target in keys_a.values()}
    open_a = [call for call in pool_a if call.address not in keys_a]
    open_b = [call for call in pool_b if call.address not in claimed]
    a_norm, b_norm = _norm_groups(open_a, side_a), _norm_groups(open_b, side_b)
    ambiguous: set[str] = set()
    for call in open_a:
        norm = side_a.norm[call.address]
        partners = b_norm.get(norm, set())
        if len(a_norm[norm]) > 1 or len(partners) > 1:
            ambiguous.add(call.address)
        elif partners:
            keys_a[call.address] = ("normalized", next(iter(partners)))
    return keys_a, ambiguous


def _zip_group(
    calls_a: list[CallInfo], calls_b: list[CallInfo], alignment: str, note: str
) -> tuple[list[CallPair], list[CallInfo]]:
    """Pair two same-key call lists by call order; return pairs and side-a leftovers.

    Side-b leftovers are found by the caller as the b calls no pair used.
    """

    count = min(len(calls_a), len(calls_b))
    pairs = [CallPair(calls_a[i], calls_b[i], alignment, note) for i in range(count)]
    return pairs, calls_a[count:]


def _match_pool(
    pool_a: list[CallInfo],
    pool_b: list[CallInfo],
    sides: tuple[_Side, _Side],
    module_map: Mapping[str, str],
) -> tuple[list[CallPair], list[CallInfo], list[CallInfo], list[CallInfo]]:
    """Match one pool; return pairs, unpaired a, unpaired b, ambiguous a."""

    keys_a, ambiguous = _keys(pool_a, pool_b, sides, module_map)
    by_target: dict[str, list[CallInfo]] = {}
    for call in pool_b:
        by_target.setdefault(call.address, []).append(call)
    groups: dict[tuple[str, str], list[CallInfo]] = {}
    for call in pool_a:
        if call.address in keys_a:
            groups.setdefault(keys_a[call.address], []).append(call)
    pairs: list[CallPair] = []
    left_a = [call for call in pool_a if call.address not in keys_a]
    for (kind, target), calls_a in groups.items():
        alignment = ASSERTED if kind == ASSERTED else SUPPORTED
        note = "" if kind == SUPPORTED else f"{kind}: {calls_a[0].address} ~ {target}"
        found, rest_a = _zip_group(calls_a, by_target.get(target, []), alignment, note)
        pairs.extend(found)
        left_a.extend(rest_a)
    used_b = {pair.b.label for pair in pairs if pair.b is not None}
    left_b = [call for call in pool_b if call.label not in used_b]
    ambiguous_calls = [call for call in left_a if call.address in ambiguous]
    left_a = [call for call in left_a if call.address not in ambiguous]
    return pairs, left_a, left_b, ambiguous_calls


def _pair_under(
    parent_a: str | None,
    parent_b: str | None,
    sides: tuple[_Side, _Side],
    module_map: Mapping[str, str],
) -> list[CallPair]:
    """Pair the children of one paired parent pair, lifting through unpaired calls."""

    side_a, side_b = sides
    pool_a = list(side_a.children.get(parent_a, []))
    pool_b = list(side_b.children.get(parent_b, []))
    rows: list[CallPair] = []
    while pool_a or pool_b:
        pairs, left_a, left_b, ambiguous = _match_pool(pool_a, pool_b, sides, module_map)
        for pair in pairs:
            rows.append(pair)
            label_a = pair.a.label if pair.a is not None else None
            label_b = pair.b.label if pair.b is not None else None
            rows.extend(_pair_under(label_a, label_b, sides, module_map))
        note = "normalized address matches several"
        rows.extend(CallPair(call, None, AMBIGUOUS, note) for call in ambiguous)
        rows.extend(CallPair(call, None, ABSENT, "") for call in left_a)
        rows.extend(CallPair(None, call, ABSENT, "") for call in left_b)
        unpaired_a = left_a + ambiguous
        lifted_a = [kid for call in unpaired_a for kid in side_a.children.get(call.label, [])]
        lifted_b = [kid for call in left_b for kid in side_b.children.get(call.label, [])]
        pool_a, pool_b = lifted_a, lifted_b
    return rows


def pair_calls(
    calls_a: list[CallInfo],
    calls_b: list[CallInfo],
    module_map: Mapping[str, str] | None = None,
) -> list[CallPair]:
    """Pair the module calls of two traces by address, never by value.

    Parameters
    ----------
    calls_a, calls_b:
        Calls from :func:`~torchlens.debug._xb_calls.read_calls`.
    module_map:
        Optional side-a to side-b address prefix map; mapped pairs are ``asserted``.

    Returns
    -------
    list[CallPair]
        Every call of both sides exactly once (paired, ambiguous or absent).
    """

    sides = (_Side(calls_a), _Side(calls_b))
    return _pair_under(None, None, sides, dict(module_map or {}))
