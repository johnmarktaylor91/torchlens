"""The persisted-state contract service (ecosystem MEMO 3.3, megaplan P05).

ONE service shared by load, inspect, and digest generation: it enumerates the
known fields per record contract (through the field registry, which derives
them from the same ``_FIELD_POLICY`` / ``_PORTABLE_STATE_SPEC`` tables the
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

The INERT inventory lives in :func:`inspect_state_contract` (private to
this module; release goldens use it): a surrogate-unpickler scan of ``metadata.pkl`` that
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

from .format_contract import TLSPEC_VERSION
from .format_errors import ArtifactError, UnknownPersistedFieldError

#: Cap on unknown names spelled out in one refusal/inventory row (the full
#: count always reports; a hostile million-key state must not build a
#: gigabyte message).
_MAX_NAMED_UNKNOWNS = 20

#: Byte ceiling on a legacy ``metadata.pkl`` the inert inventory reads, at parity with
#: the load path's own ceiling on the same file (``bundle._MAX_METADATA_PKL_BYTES``).
_MAX_INVENTORY_PKL_BYTES = 512 * 1024**2

#: Cap on distinct surrogate classes one inventory scan creates. Each distinct pickled
#: qualname costs one class object; a real 2.x trace names a few dozen, while a
#: hostile pickle could name one per few bytes. The surrogates live only for the scan.
_MAX_SURROGATE_CLASSES = 4096

_INVENTORY_REMEDY = "re-save the trace with trace.save()"

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

    The direction split: newer writer -> upgrade, same stamp -> writer drift,
    older writer -> reader regression.
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
            "release that wrote it."
        )
    return (
        "The artifact was written by an older torchlens whose field this "
        "reader no longer declares; load it under the writing release and "
        "re-save. If the artifact is a governed release artifact, report "
        "this refusal."
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


def _inventory_refusal(metadata_path: Any, reason: str) -> ArtifactError:
    """Return the typed refusal for a legacy ``metadata.pkl`` the inventory will not read.

    The same error and code the 3.0 inventory half raises for an unreadable record file.
    """

    from .format3 import contract

    return ArtifactError(
        f"Artifact metadata {metadata_path} could not be read: {reason}. "
        f"Remedy: {_INVENTORY_REMEDY}.",
        code=contract.REFUSAL_CODES["record_file_corrupt"],
        remedy=_INVENTORY_REMEDY,
        file_path=str(metadata_path),
    )


def _load_inventory_skeleton(metadata_path: Any) -> Any:
    """Inertly unpickle a legacy ``metadata.pkl`` into surrogate skeletons, under bounds.

    Every global resolves to a per-scan surrogate subclass (nothing is imported), the
    file must sit under the byte ceiling, and the scan creates at most
    ``_MAX_SURROGATE_CLASSES`` distinct surrogate classes.

    Raises
    ------
    ArtifactError
        When the file is over the byte ceiling or names too many distinct classes.
    """

    import io as _stdlib_io
    import os
    import pickle

    classes: dict[str, type] = {}

    def surrogate_class(qualname: str) -> type:
        cls = classes.get(qualname)
        if cls is None:
            if len(classes) >= _MAX_SURROGATE_CLASSES:
                raise _inventory_refusal(
                    metadata_path,
                    f"it names more than {_MAX_SURROGATE_CLASSES} distinct classes",
                )
            cls = type(
                "_Surrogate", (_SurrogateRecord,), {"_tl_qualname": qualname, "__slots__": ()}
            )
            classes[qualname] = cls
        return cls

    class _InventoryUnpickler(pickle.Unpickler):
        """Unpickler that resolves EVERY global to an inert surrogate class."""

        def find_class(self, module: str, name: str) -> Any:
            """Return the per-scan surrogate type instead of importing anything."""

            return surrogate_class(f"{module}.{name}")

        def persistent_load(self, pid: Any) -> Any:
            """Stand in for persistent-id references with an inert surrogate."""

            return surrogate_class("persistent_id")()

    with metadata_path.open("rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        if size > _MAX_INVENTORY_PKL_BYTES:
            raise _inventory_refusal(
                metadata_path,
                f"it is {size} bytes, above the {_MAX_INVENTORY_PKL_BYTES}-byte ceiling",
            )
        return _InventoryUnpickler(_stdlib_io.BufferedReader(handle)).load()


def inspect_state_contract(path: Any) -> dict[str, Any]:
    """Inertly inventory a bundle's persisted state against the contract.

    An artifact whose manifest names the 3.0 format, or that has no root
    ``metadata.pkl``, goes to :func:`.format3.inventory.inspect_record_tables`; a
    root ``metadata.pkl`` beside a 3.0 manifest is never unpickled and is reported
    under ``foreign_files`` by name, size and digest. A 2.x ``metadata.pkl`` is read
    by a bounded surrogate unpickler that builds skeleton records only (no record
    object, no ``__setstate__``, no import outside the pickle protocol). Nothing
    refuses an unknown field: unknown fields are REPORTED with their owning record
    types (MEMO 3.3).

    Parameters
    ----------
    path:
        Artifact directory (3.0 record tables, or a 2.x ``metadata.pkl``).

    Returns
    -------
    dict[str, Any]
        ``{"record_types": {qualname: {"instances": n, "unknown_fields":
        [...]}}, "unknown_field_total": n, "governed_record_types": [...]}``,
        plus ``foreign_files`` when a 3.0 artifact carries a root pickle.
        Record types outside the governed registry report their state keys
        uninterpreted (``"ungoverned": True``) rather than guessing.

    Raises
    ------
    ArtifactError
        When a file the inventory must read is unreadable, or a legacy
        ``metadata.pkl`` exceeds the inventory's byte or class bounds.
    """

    from pathlib import Path

    from .format3.inventory import foreign_file_note, inspect_record_tables, names_format3

    root = Path(path)
    metadata_path = root / "metadata.pkl"
    has_root_pickle = metadata_path.is_symlink() or metadata_path.exists()
    if not has_root_pickle or names_format3(root):
        report = inspect_record_tables(root)
        if has_root_pickle:
            report["foreign_files"] = [foreign_file_note(metadata_path)]
        return report
    return _legacy_skeleton_report(_load_inventory_skeleton(metadata_path))


def _trace_root_rows(skeleton: Any) -> dict[str, dict[str, Any]]:
    """Attribute a legacy ``metadata.pkl`` ROOT mapping's keys to the Trace contract.

    The root is the Trace's scrubbed state mapping itself (the scrub path
    serializes the state dict, not a pickled Trace object), so its keys, less
    the transport envelope, are checked against the Trace contract directly.
    """

    from importlib import import_module as _import_module

    from .field_registry import RECORD_CONTRACT_CLASSES

    if not isinstance(skeleton, dict):
        return {}
    trace_module, trace_class = RECORD_CONTRACT_CLASSES["trace"]
    trace_cls = getattr(_import_module(trace_module), trace_class)
    root_keys = {
        str(key): None
        for key in skeleton
        if isinstance(key, str) and key not in _TRACE_TRANSPORT_KEYS
    }
    _, root_unknown = partition_state_keys(trace_cls, root_keys)
    return {
        f"{trace_module}.{trace_class}": {
            "instances": 1,
            "unknown_fields": set(root_unknown),
            "ungoverned": False,
        }
    }


def _legacy_skeleton_report(skeleton: Any) -> dict[str, Any]:
    """Build the inventory report from a legacy ``metadata.pkl`` surrogate skeleton."""

    from importlib import import_module as _import_module

    from .field_registry import RECORD_CONTRACT_CLASSES

    governed_by_qualname = {
        f"{module_name}.{class_name}": contract_key
        for contract_key, (module_name, class_name) in RECORD_CONTRACT_CLASSES.items()
    }
    report = _trace_root_rows(skeleton)

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
        module_name, class_name = RECORD_CONTRACT_CLASSES[record_key]
        cls = getattr(_import_module(module_name), class_name)
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
