"""Payload eviction walk: replace every tensor reachable from a store cell with ``None``.

``OpRowStore._evict_tensor_payloads`` runs this walk over every cell once no
owning core is alive (a dropped trace dies by refcount, so it runs right at the
drop). The walk descends builtin containers only (list, tuple, dict, set,
frozenset) to a fixed depth; tensors inside custom objects are the disclosed
residual.

Most cells hold values the walk returns unchanged: builtin scalars, TorchLens
record objects, tuples of either. A sweep shares one ``_EvictTypes`` set: types
the walk returns unchanged by type alone. It starts as the builtin scalars and
learns each opaque type the first time the sweep meets it, so the container
sweeps skip later instances, and exact tuples made only of such members,
without a Python call.
"""

from __future__ import annotations

from typing import Any

from .._utils._display_bounded import RecordTuple

# Depth ceiling for the nested-container payload eviction walk.
_EVICT_WALK_DEPTH_CEILING = 8

#: Exact builtin types the eviction walk always returns unchanged: no instance
#: is a tensor or a container the walk descends, and none can override
#: ``__class__``, so an exact-type membership test answers what the walk's
#: ``isinstance`` chain would. Callers skip these values without a call (most
#: cells of a capture hold one: ``None``, labels, counts, the store sentinels).
#: Subclasses (``Bytes``, ``Flops``, ...) and record types take one per-value
#: call per sweep, which learns them (``_EvictTypes``).
_EVICT_INERT_TYPES = frozenset({type(None), bool, int, float, complex, str, bytes, object})

#: Every container family the walk descends; anything else that is not a
#: tensor is returned unchanged.
_EVICT_CONTAINER_TYPES = (list, tuple, dict, set, frozenset)


class _EvictTypes(set[type]):
    """One sweep's types the walk returns unchanged by type alone.

    Starts as the builtin scalars and grows with every opaque type the sweep
    meets (see ``_opaque_to_eviction``); ``refused`` remembers the types
    checked and found not opaque, so each type is checked once per sweep.
    ``whole`` holds the ids of the exact lists and dicts the sweep has walked as
    a cell's own value (depth 0): such a walk reaches everything any later walk
    of the same object could, so a container shared by many cells is walked
    once per sweep. Every cell stays alive during the sweep, so the ids are
    stable. Membership tests stay the builtin ``set`` lookup.
    """

    __slots__ = ("refused", "whole")

    def __init__(self) -> None:
        """Seed with the builtin scalar types; nothing refused or walked yet."""

        super().__init__(_EVICT_INERT_TYPES)
        self.refused: set[type] = set()
        self.whole: set[int] = set()


def _opaque_to_eviction(cls: type, tensor_cls: type) -> bool:
    """Return whether the walk returns every instance of ``cls`` unchanged, by type alone.

    True for a type that is neither a tensor nor a container the walk descends
    and whose instances cannot make ``isinstance`` answer anything but their
    real type: no class in its MRO below ``object`` defines ``__class__`` or
    ``__getattribute__``, and ``tensor_cls`` keeps the default instance check.
    Such values are returned unchanged whatever they hold, so a sweep may skip
    them without a call. Conservative: a C type that spells its default
    attribute lookup out (a ``__getattribute__`` slot in its dict) is refused
    and keeps the per-value path.
    """

    if type(tensor_cls).__instancecheck__ is not type.__instancecheck__:
        return False
    if issubclass(cls, tensor_cls) or issubclass(cls, _EVICT_CONTAINER_TYPES):
        return False
    for klass in cls.__mro__[:-1]:
        namespace = klass.__dict__
        if "__class__" in namespace or "__getattribute__" in namespace:
            return False
    return True


def _evict_payloads_in_cell(
    value: Any,
    tensor_cls: type,
    depth: int = 0,
    seen: set[int] | None = None,
    inert: _EvictTypes | None = None,
) -> Any:
    """Return ``value`` with every reachable tensor replaced by ``None``.

    A direct tensor becomes ``None`` (the payload-absent spelling); exact
    ``list``/``dict`` evict in place and exact tuples rebuild only when a
    member changed; cyclic containers terminate through ``seen``. Dispatch is
    by exact type first, so ``inert`` types return at once and exact
    containers skip the tensor check. A non-container that is not a tensor is
    returned unchanged, and its type is learned into ``inert`` when
    ``_opaque_to_eviction`` allows. Container subclasses take
    ``_evict_payloads_in_subclass``.
    """

    if inert is None:
        inert = _EvictTypes()
    cls = type(value)
    if cls in inert:
        return value
    if cls is tuple:
        if depth >= _EVICT_WALK_DEPTH_CEILING:
            return value
        return _evict_payloads_in_tuple(value, tensor_cls, depth + 1, seen, inert)
    if cls is list or cls is dict:
        if depth >= _EVICT_WALK_DEPTH_CEILING:
            return value
        key = id(value)
        if key in inert.whole or (seen is not None and key in seen):
            return value
        if seen is None:
            seen = set()
        if not depth:  # a cell's own value: walked whole, once per sweep
            inert.whole.add(key)
        seen.add(key)
        if cls is list:
            _evict_payloads_in_list(value, tensor_cls, depth + 1, seen, inert)
        else:
            _evict_payloads_in_mapping(value, tensor_cls, depth + 1, seen, inert)
        return value
    if isinstance(value, tensor_cls):
        return None
    if not isinstance(value, _EVICT_CONTAINER_TYPES):
        if cls not in inert.refused:
            if _opaque_to_eviction(cls, tensor_cls):
                inert.add(cls)
            else:
                inert.refused.add(cls)
        return value
    if cls.__dict__.get("_ROW_KEYS_ONLY") is True:
        # A record row form: row keys and records only, which the walk returns unchanged.
        inert.add(cls)
        return value
    if depth >= _EVICT_WALK_DEPTH_CEILING:
        return value
    return _evict_payloads_in_subclass(value, tensor_cls, depth, seen, inert)


def _evict_payloads_in_subclass(
    value: Any, tensor_cls: type, depth: int, seen: set[int] | None, inert: _EvictTypes
) -> Any:
    """Evict a container-subclass value (named tuples, ``torch.Size``, ``defaultdict``, ...).

    List and dict subclasses keep every read and write-back through their own
    item methods; tuple subclasses rebuild through their own type; sets and
    frozensets drop tensor members.
    """

    if isinstance(value, list):
        if seen is None:
            seen = set()
        key = id(value)
        if key in seen:
            return value
        seen.add(key)
        for index, item in enumerate(value):
            value[index] = _evict_payloads_in_cell(item, tensor_cls, depth + 1, seen, inert)
        return value
    if isinstance(value, tuple):
        return _evict_payloads_in_tuple(value, tensor_cls, depth + 1, seen, inert)
    if isinstance(value, dict):
        if seen is None:
            seen = set()
        key = id(value)
        if key in seen:
            return value
        seen.add(key)
        for item_key in list(value):
            value[item_key] = _evict_payloads_in_cell(
                value[item_key], tensor_cls, depth + 1, seen, inert
            )
        return value
    return _evict_payloads_in_set(value, tensor_cls)


# The three sweeps below skip, without a call, every member whose type is in
# ``inert`` and every exact tuple whose members all are (tensor shapes, label
# pairs, tuples of TorchLens records): the walk would return each unchanged.
# The skip is inlined in each loop on purpose (a helper would cost the call it
# saves); it raises their cyclomatic complexity (tuple sweep 15), kept for the hot path.


def _evict_payloads_in_list(
    values: list[Any],
    tensor_cls: type,
    depth: int = 0,
    seen: set[int] | None = None,
    inert: _EvictTypes | None = None,
) -> None:
    """Evict every member of ``values`` in place, writing back only changes.

    Used for list containers inside the walk and for the store's own cell
    lists (rows, object columns), where ``depth`` 0 and ``seen`` ``None``
    give each cell its own walk; ``inert`` is shared across a whole sweep.
    """

    if inert is None:
        inert = _EvictTypes()
    for index, item in enumerate(values):
        cls = type(item)
        if cls in inert:
            continue
        if cls is tuple:
            for member in item:
                if type(member) not in inert:
                    break
            else:
                continue
        evicted = _evict_payloads_in_cell(item, tensor_cls, depth, seen, inert)
        if evicted is not item:
            values[index] = evicted


def _evict_payloads_in_mapping(
    values: dict[Any, Any],
    tensor_cls: type,
    depth: int = 0,
    seen: set[int] | None = None,
    inert: _EvictTypes | None = None,
) -> None:
    """Evict every value of an exact ``dict`` in place, writing back only changes.

    Used for dict containers inside the walk and for the store overlays,
    where ``depth`` 0 and ``seen`` ``None`` give each cell its own walk;
    ``inert`` is shared across a whole sweep. Assigning an existing key does
    not resize the dict, so iterating its items meanwhile is safe.
    """

    if inert is None:
        inert = _EvictTypes()
    for key, item in values.items():
        cls = type(item)
        if cls in inert:
            continue
        if cls is tuple:
            for member in item:
                if type(member) not in inert:
                    break
            else:
                continue
        evicted = _evict_payloads_in_cell(item, tensor_cls, depth, seen, inert)
        if evicted is not item:
            values[key] = evicted


def _evict_payloads_in_tuple(
    value: tuple[Any, ...],
    tensor_cls: type,
    depth: int,
    seen: set[int] | None,
    inert: _EvictTypes | None = None,
) -> Any:
    """Return ``value`` evicted; rebuild only when a member changed."""

    if inert is None:
        inert = _EvictTypes()
    rebuilt: list[Any] | None = None
    for index, item in enumerate(value):
        cls = type(item)
        if cls in inert:
            continue
        if cls is tuple:
            for member in item:
                if type(member) not in inert:
                    break
            else:
                continue
        evicted = _evict_payloads_in_cell(item, tensor_cls, depth, seen, inert)
        if evicted is not item:
            if rebuilt is None:
                rebuilt = list(value)
            rebuilt[index] = evicted
    if rebuilt is None:
        return value
    if type(value) is tuple:
        return tuple(rebuilt)
    if type(value) is RecordTuple:
        return RecordTuple(rebuilt, value.title, value.noun, value.singular)
    try:
        # Named tuples and tuple subclasses reconstruct through their own
        # type; an unreconstructable subclass keeps the original (a
        # retained pin beats a corrupted container).
        return type(value)(*rebuilt)
    except Exception:
        try:
            return type(value)(rebuilt)
        except Exception:
            return value


def _evict_payloads_in_set(value: Any, tensor_cls: type) -> Any:
    """Return a set or frozenset with tensor members replaced by one ``None``."""

    if isinstance(value, (set, frozenset)):
        if not any(isinstance(item, tensor_cls) for item in value):
            return value
        survivors = {item for item in value if not isinstance(item, tensor_cls)}
        survivors.add(None)
        try:
            return type(value)(survivors)
        except Exception:
            return value
    return value
