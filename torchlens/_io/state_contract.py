"""The persisted-state contract service (ecosystem MEMO 3.3, megaplan P05).

ONE service shared by load, inspect, and digest generation: it enumerates the
known fields per record contract (through the field registry, which derives
them from the same ``FIELD_POLICY`` / ``PORTABLE_STATE_SPEC`` tables the
serializers use), partitions incoming state into known/unknown BEFORE any
object mutation, and refuses unknown fields TYPED and SYMMETRICALLY -- the
refusal fires whether the writer is newer than the reader (additive field)
or older (a field this reader deleted without an alias). The three storage
paths all route through the one chokepoint in
:func:`torchlens._io.read_tlspec_version`, which every portable
``__setstate__`` already calls first:

- ``__dict__``-backed classes (``Trace`` -- the path that never crashed and
  silently absorbed unknown keys),
- hook-restored columnar facades (``Op`` -- the path that crashed untyped),
- slotted/value classes (everything else).

The INERT inventory lives in :func:`inspect_state_contract`
(``torchlens.io``): a surrogate-unpickler scan of ``metadata.pkl`` that
builds NO live objects, imports NO foreign code, and reports unknown names
with their owning record types so a user can always see everything without
loading anything.

SCOPE: the contract governs ARTIFACT bytes only. The refusal fires inside
the governed-artifact-load window (:func:`governed_artifact_load`, armed by
the ``.tlspec`` loaders), where an unknown field can only mean writer/reader
contract skew -- the save path refuses to persist undeclared record fields,
so no governed artifact ever carries one legitimately. Plain session
pickling of live records stays OUTSIDE the window: records deliberately
round-trip user-set extra attributes through ``pickle`` (the dict-era
``__dict__`` semantics), and those extras never reach an artifact.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import Any

from . import TLSPEC_VERSION, UnknownPersistedFieldError

#: Cap on unknown names spelled out in one refusal/inventory row (the full
#: count always reports; a hostile million-key state must not build a
#: gigabyte message).
_MAX_NAMED_UNKNOWNS = 20

#: Transport-envelope keys the bundle writer places BESIDE the Trace state in
#: metadata.pkl and the loaders pop before ``__setstate__`` ever runs
#: (rehydrate pops the io accessor state; ``__setstate__`` pops the pickle
#: accessor state and the negative-verification row first thing). The LIVE
#: refusal path never sees them; only the root-dict INVENTORY must know them.
_TRACE_TRANSPORT_KEYS: frozenset[str] = frozenset(
    {
        "_io_module_accessor_state",
        "_pickle_module_accessor_state",
        "_capture_verification",
    }
)

#: Depth of the governed-artifact-load window (single-threaded by design,
#: like capture). Non-zero exactly while a ``.tlspec`` loader is restoring
#: governed bytes; the unknown-field refusal fires only inside the window.
_GOVERNED_LOAD_DEPTH = 0


@contextmanager
def governed_artifact_load() -> Iterator[None]:
    """Arm the unknown-field partition for one governed artifact load.

    Entered by the ``.tlspec`` loaders (the bundle metadata unpickler and the
    trace rehydrator) around the frames in which portable ``__setstate__``
    calls run on artifact bytes. Inside the window an unknown field is always
    writer/reader contract skew and refuses typed; outside it (plain session
    ``pickle`` of live records) user-set record extras keep their historical
    round-trip semantics -- the save path refuses to persist them, so they can
    never reach a governed artifact. Reentrant: nested member loads keep the
    window open.
    """

    global _GOVERNED_LOAD_DEPTH
    _GOVERNED_LOAD_DEPTH += 1
    try:
        yield
    finally:
        _GOVERNED_LOAD_DEPTH -= 1


def governed_load_active() -> bool:
    """True while at least one governed-artifact-load window is open."""

    return _GOVERNED_LOAD_DEPTH > 0


@cache
def _known_keys_for(cls: type) -> frozenset[str]:
    """Cached known-key set per record class (hot on the per-Op path)."""

    from .field_registry import known_state_keys

    return known_state_keys(cls)


def partition_state_keys(
    cls: type, state: dict[str, Any]
) -> tuple[frozenset[str], tuple[str, ...]]:
    """Partition ``state``'s keys into known/unknown for ``cls`` (pure).

    Parameters
    ----------
    cls:
        Portable record class whose contract governs the state.
    state:
        Incoming serialized state mapping. Never mutated.

    Returns
    -------
    tuple[frozenset[str], tuple[str, ...]]
        The known-key set for the class, and the sorted unknown keys found.
    """

    known = _known_keys_for(cls)
    return known, tuple(sorted(name for name in state if name not in known))


def _unknown_field_remedy(declared_version: int | None) -> str:
    """Derive the direction-aware remedy for an unknown-field refusal.

    The ledger-derived remedy service (MEMO 3.1 G6) will supersede these
    texts with exact release names once the compatibility ledger lands; the
    direction split is already the ledger's: newer writer -> upgrade, same
    stamp -> writer drift, older writer -> reader regression.
    """

    if declared_version is not None and declared_version > TLSPEC_VERSION:
        return (
            "The artifact was written by a newer torchlens; upgrade torchlens "
            "to the release that wrote it (or newer)."
        )
    if declared_version == TLSPEC_VERSION:
        return (
            "The artifact carries fields this same-stamp reader does not "
            "declare (same-stamp writer drift); load it under the exact "
            "release that wrote it, or inspect inertly with "
            "torchlens.io.inspect_state_contract."
        )
    return (
        "The artifact was written by an older torchlens whose field this "
        "reader no longer declares; load it under the writing release (or "
        "one that still declares the field) and re-save, or inspect inertly "
        "with torchlens.io.inspect_state_contract. A reader that dropped a "
        "persisted field must carry an alias (PORTABLE_STATE_ALIASES) -- "
        "report this refusal if the artifact is a governed release artifact."
    )


def enforce_known_state(
    cls: type,
    state: dict[str, Any],
    *,
    declared_version: int | None = None,
) -> None:
    """Refuse typed if ``state`` carries fields unknown to ``cls``'s contract.

    Runs BEFORE any object mutation (the caller is the version gate at the
    top of every portable ``__setstate__``). Absent fields stay tolerated
    (default-fill is unchanged); unknown fields refuse in both writer
    directions.

    Parameters
    ----------
    cls:
        Portable record class being restored.
    state:
        Incoming serialized state mapping. Never mutated here.
    declared_version:
        The state's declared ``tlspec_version`` when the caller already
        decoded it (drives the direction-aware remedy).

    Raises
    ------
    UnknownPersistedFieldError
        Naming the record type, the exact unknown field names, both declared
        and runtime schema versions, and the remedy.
    """

    _, unknown = partition_state_keys(cls, state)
    if not unknown:
        return
    named = ", ".join(unknown[:_MAX_NAMED_UNKNOWNS])
    overflow = (
        ""
        if len(unknown) <= _MAX_NAMED_UNKNOWNS
        else f" (+{len(unknown) - _MAX_NAMED_UNKNOWNS} more)"
    )
    remedy = _unknown_field_remedy(declared_version)
    raise UnknownPersistedFieldError(
        f"{cls.__name__} state carries {len(unknown)} field(s) unknown to this "
        f"reader's persisted-state contract: {named}{overflow}. Unknown fields "
        "refuse in both writer directions -- silently dropping them would "
        "discard captured evidence, and silently absorbing them would attest "
        f"an outcome the reader cannot vouch for. {remedy}",
        code="unknown_persisted_field",
        record_type=cls.__name__,
        unknown_fields=unknown,
        declared_tlspec_version=declared_version,
        runtime_tlspec_version=TLSPEC_VERSION,
        remedy=remedy,
    )


# ---------------------------------------------------------------------------
# Inert inventory: see everything without loading anything.
# ---------------------------------------------------------------------------


class _SurrogateRecord:
    """Skeleton stand-in for one pickled object: records identity and state keys.

    ``find_class`` returns per-qualname SUBCLASSES (the pickle VM's NEWOBJ
    opcode requires a real type), so instances exist for every construction
    protocol -- NEWOBJ, REDUCE-call, copyreg reconstructors -- while building
    nothing live: no torchlens ``__setstate__`` ever runs.
    """

    __slots__ = ("qualname", "state_keys")

    qualname: str
    state_keys: tuple[str, ...]

    _tl_qualname = "?"

    def __new__(cls, *args: Any, **kwargs: Any) -> _SurrogateRecord:
        obj = object.__new__(cls)
        obj.qualname = cls._tl_qualname
        obj.state_keys = ()
        return obj

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> _SurrogateRecord:
        return type(self)()

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.state_keys = tuple(sorted(str(key) for key in state))
        elif isinstance(state, tuple):
            # Slotted-object protocol: (dict_state, slots_state).
            keys: set[str] = set()
            for part in state:
                if isinstance(part, dict):
                    keys.update(str(key) for key in part)
            self.state_keys = tuple(sorted(keys))

    # The pickle VM may exercise container protocols on surrogates that
    # stand in for helper objects; stay inert under all of them.
    def __setitem__(self, key: Any, value: Any) -> None:
        pass

    def append(self, value: Any) -> None:
        """Absorb a pickle-VM list append inertly (nothing is stored)."""

    def extend(self, values: Any) -> None:
        """Absorb a pickle-VM batch append inertly (nothing is stored)."""


@cache
def _surrogate_class(qualname: str) -> type:
    """One cached surrogate SUBCLASS per pickled qualname."""

    return type("_Surrogate", (_SurrogateRecord,), {"_tl_qualname": qualname, "__slots__": ()})


def _iter_surrogates(root: Any) -> Iterator[_SurrogateRecord]:
    """Yield every ``_SurrogateRecord`` reachable from an unpickled skeleton."""

    seen: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, _SurrogateRecord):
            yield node
            continue
        if isinstance(node, dict):
            stack.extend(node.keys())
            stack.extend(node.values())
        elif isinstance(node, (list, tuple, set, frozenset)):
            stack.extend(node)


def inspect_state_contract(path: Any) -> dict[str, Any]:
    """Inertly inventory a bundle's persisted state against the contract.

    Reads ``metadata.pkl`` with a surrogate unpickler that builds skeleton
    records only: no torchlens record objects are constructed, no
    ``__setstate__`` runs, no module outside the pickle protocol is imported,
    and nothing refuses -- unknown fields are REPORTED with their owning
    record types (MEMO 3.3: a user can always see everything without loading
    anything).

    Parameters
    ----------
    path:
        Bundle directory containing ``metadata.pkl``.

    Returns
    -------
    dict[str, Any]
        ``{"record_types": {qualname: {"instances": n, "unknown_fields":
        [...]}}, "unknown_field_total": n, "governed_record_types": [...]}``.
        Record types outside the governed registry report their state keys
        uninterpreted (``"ungoverned": True``) rather than guessing.
    """

    import io as _stdlib_io
    import pickle
    from pathlib import Path

    from .field_registry import RECORD_CONTRACT_CLASSES

    metadata_path = Path(path) / "metadata.pkl"

    class _InventoryUnpickler(pickle.Unpickler):
        """Unpickler that resolves EVERY global to an inert surrogate class."""

        def find_class(self, module: str, name: str) -> Any:
            """Return the cached surrogate type instead of importing anything."""

            return _surrogate_class(f"{module}.{name}")

        def persistent_load(self, pid: Any) -> Any:
            """Stand in for persistent-id references with an inert surrogate."""

            return _surrogate_class("persistent_id")()

    with metadata_path.open("rb") as handle:
        skeleton = _InventoryUnpickler(_stdlib_io.BufferedReader(handle)).load()

    governed_by_qualname = {}
    for contract_key, (module_name, class_name) in RECORD_CONTRACT_CLASSES.items():
        governed_by_qualname[f"{module_name}.{class_name}"] = contract_key

    report: dict[str, dict[str, Any]] = {}

    # The metadata.pkl ROOT is the Trace's scrubbed state mapping itself (the
    # scrub path serializes the state dict, not a pickled Trace object), so
    # the root keys are attributed to the Trace contract directly.
    if isinstance(skeleton, dict):
        from importlib import import_module as _import_module

        trace_module, trace_class = RECORD_CONTRACT_CLASSES["trace"]
        trace_cls = getattr(_import_module(trace_module), trace_class)
        root_keys = {
            str(key): None
            for key in skeleton
            if isinstance(key, str) and key not in _TRACE_TRANSPORT_KEYS
        }
        _, root_unknown = partition_state_keys(trace_cls, root_keys)
        report[f"{trace_module}.{trace_class}"] = {
            "instances": 1,
            "unknown_fields": set(root_unknown),
            "ungoverned": False,
        }

    for surrogate in _iter_surrogates(skeleton):
        row = report.setdefault(
            surrogate.qualname,
            {"instances": 0, "unknown_fields": set(), "ungoverned": True},
        )
        row["instances"] += 1
        record_key = governed_by_qualname.get(surrogate.qualname)
        if record_key is None:
            continue
        row["ungoverned"] = False
        from importlib import import_module

        module_name, class_name = RECORD_CONTRACT_CLASSES[record_key]
        cls = getattr(import_module(module_name), class_name)
        _, unknown = partition_state_keys(cls, dict.fromkeys(surrogate.state_keys))
        row["unknown_fields"].update(unknown)

    unknown_total = sum(
        len(row["unknown_fields"]) for row in report.values() if not row["ungoverned"]
    )
    return {
        "record_types": {
            qualname: {
                "instances": row["instances"],
                "unknown_fields": sorted(row["unknown_fields"]),
                "ungoverned": row["ungoverned"],
            }
            for qualname, row in sorted(report.items())
        },
        "unknown_field_total": unknown_total,
        "governed_record_types": sorted(
            qualname for qualname, row in report.items() if not row["ungoverned"]
        ),
    }
