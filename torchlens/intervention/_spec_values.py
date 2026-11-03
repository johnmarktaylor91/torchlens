"""Value grammar of the persisted intervention spec: the JSON wrapper tags
(``__tensor_ref__``, ``__helper__``, ``__callable__``, ``__opaque_audit__``,
``__output_path_component__``, ``__dict_items__``, ``__tuple_key__``) and the
import-reference trust gate (``LazyImportRef``) that a ``__callable__`` resolves
through. One encoder (``_serialize_value``) and one decoder (``_deserialize_value``);
the spec-document layer in ``_spec_codec`` composes them. ``SaveLevel`` lives here
because it governs how each value encodes.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from .._io.manifest import TensorEntry
from .._ir.container import DataclassField, DictKey, HFKey, NamedField, TupleIndex
from ..intervention._resolver_registry import (
    resolve_import_ref,
)
from ._helpers import helper_from_serialized
from ._types import (
    Edit,
)
from .errors import (
    OpaqueCallableError,
    ReplayPreconditionError,
    SiteResolutionError,
    UnserializableDictKeyError,
    UntrustedCallableError,
)


class SaveLevel(str, Enum):
    """Supported intervention spec save levels."""

    AUDIT = "audit"
    EXECUTABLE_WITH_CALLABLES = "executable_with_callables"
    PORTABLE = "portable"


@dataclass(frozen=True)
class _SerializedState:
    """Internal state accumulated during spec serialization."""

    tensor_entries: list[TensorEntry]
    tensor_refs: dict[str, torch.Tensor]


@dataclass(frozen=True)
class LazyImportRef:
    """Callable import reference that resolves only at execution time.

    The trust context is captured from the load call that materialized this
    reference and carried until execution, so the deferred resolution enforces the
    SAME deny-by-default gate as ``resolve_function_registry_key``: a foreign module
    is never imported under the default ``trust_custom_callables=False`` load.
    """

    import_path: str
    trust_custom_callables: bool = False
    allowed_custom_callable_modules: tuple[str, ...] | None = None

    def __setstate__(self, state: object) -> None:
        """Force fail-closed trust on ANY unpickle reconstruction (self-defense).

        A ``LazyImportRef`` is materialized in-process from a load call with a trust
        context threaded through ``__init__``; it is NEVER a legitimate member of a
        pickled ``metadata.pkl`` graph. If some path (a forged bundle, a
        ``__reduce__`` gadget) reconstructs one via unpickle, its trust MUST NOT come
        from the attacker-controlled pickled fields, or a crafted
        ``LazyImportRef(import_path="os:system", trust_custom_callables=True)`` would
        REDUCE-invoke ``os.system``. Pickle restores frozen-dataclass state by
        bypassing ``__setattr__``, so this hook rebuilds the state with
        ``object.__setattr__`` and HARD-FORCES ``trust_custom_callables=False`` and
        ``allowed_custom_callable_modules=None`` regardless of the pickled values.
        The ``import_path`` is preserved (inert until ``__call__``), but resolution
        can now never be attacker-trusted -- the foreign resolver denies it.

        This is scoped to the UNPICKLE path only: the normal in-process load path
        constructs the reference through ``__init__`` (never ``__setstate__``), so a
        legitimately-trusted intervention-spec load is unaffected.

        Parameters
        ----------
        state:
            Pickled instance state (the frozen dataclass ``__dict__``, or a
            ``(dict, slots)`` 2-tuple), used only to recover ``import_path``.
        """

        payload: object = state
        if isinstance(state, tuple) and state and isinstance(state[0], dict):
            payload = state[0]
        import_path = payload.get("import_path", "") if isinstance(payload, dict) else ""
        object.__setattr__(self, "import_path", import_path if isinstance(import_path, str) else "")
        object.__setattr__(self, "trust_custom_callables", False)
        object.__setattr__(self, "allowed_custom_callable_modules", None)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Resolve and call the referenced object under its captured trust gate.

        Parameters
        ----------
        *args:
            Positional arguments forwarded to the imported callable.
        **kwargs:
            Keyword arguments forwarded to the imported callable.

        Returns
        -------
        Any
            Return value from the imported callable.
        """

        return _resolve_import_ref(
            self.import_path,
            trust_custom_callables=self.trust_custom_callables,
            allowed_custom_callable_modules=self.allowed_custom_callable_modules,
        )(*args, **kwargs)

    def __repr__(self) -> str:
        """Return a stable representation without importing the target.

        Returns
        -------
        str
            Import-reference representation.
        """

        return f"LazyImportRef({self.import_path!r})"


def _coerce_save_level(level: str | SaveLevel) -> SaveLevel:
    """Normalize a save-level input.

    Parameters
    ----------
    level:
        String or enum save level.

    Returns
    -------
    SaveLevel
        Normalized enum value.
    """

    return level if isinstance(level, SaveLevel) else SaveLevel(level)


# Reserved wrapper-dict keys used by ``_serialize_value`` / ``_deserialize_value`` to
# tag non-plain payloads. On load, ``_deserialize_value`` decides how to interpret a
# JSON object purely by testing for the *presence* of one of these keys, so a plain
# user dict whose key literally equals one of them would be misread (e.g. silently
# reconstructed as a ``Edit`` for ``__opaque_audit__``). To make that impossible,
# ANY dict with a key in this reserved namespace is escaped through the fully-general
# ``__dict_items__`` item-list encoding instead of the plain-object encoding, so a
# genuine user key can never be mistaken for a wrapper tag. The reserved namespace is
# every ``__dunder__`` string, which also future-proofs any wrapper tag added later.
_RESERVED_WRAPPER_KEYS = frozenset(
    {
        "__tensor_ref__",
        "__helper__",
        "__callable__",
        "__output_path_component__",
        "__opaque_audit__",
        "__dict_items__",
        "__tuple_key__",
    }
)


def _is_reserved_wrapper_key(key: Any) -> bool:
    """Return ``True`` if ``key`` lives in the reserved wrapper-dict namespace.

    Any string that both starts and ends with ``"__"`` (a classic dunder) is reserved
    so it can never collide with a serializer sentinel. This covers every current
    wrapper tag (see :data:`_RESERVED_WRAPPER_KEYS`) and any future one.

    Parameters
    ----------
    key:
        Candidate dict key.

    Returns
    -------
    bool
        Whether the key must be escaped rather than emitted as a plain JSON key.
    """

    if not isinstance(key, str):
        return False
    if key in _RESERVED_WRAPPER_KEYS:
        return True
    return len(key) >= 4 and key.startswith("__") and key.endswith("__")


def _serialize_value(value: Any, save_level: SaveLevel, state: _SerializedState) -> Any:
    """Serialize tensors, helpers, callables, and JSON-safe literals.

    Parameters
    ----------
    value:
        Runtime value.
    save_level:
        Requested save level.
    state:
        Serialization tensor state.

    Returns
    -------
    Any
        JSON-safe serialized value.
    """

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, torch.Tensor):
        tensor_id = f"value_{len(state.tensor_refs)}"
        state.tensor_refs[tensor_id] = value.detach().cpu()
        return {"__tensor_ref__": tensor_id}
    if isinstance(value, Edit):
        return {"__helper__": _serialize_helper(value, save_level, state)}
    if isinstance(value, TupleIndex | DictKey | HFKey | NamedField | DataclassField):
        return _serialize_output_path_component(value, save_level, state)
    if isinstance(value, tuple):
        return [_serialize_value(item, save_level, state) for item in value]
    if isinstance(value, list):
        return [_serialize_value(item, save_level, state) for item in value]
    if isinstance(value, dict):
        # All-``str`` keys round-trip losslessly as a plain JSON object (the common
        # case, on-disk format unchanged) -- UNLESS a key collides with a reserved
        # wrapper tag (``_is_reserved_wrapper_key``), in which case emitting a plain
        # object would let ``_deserialize_value`` misread the user dict as a wrapper
        # (silent corruption for ``__opaque_audit__``, loud crashes for the rest).
        # Any non-``str`` key (int/float/tuple/...) would also be silently corrupted
        # by ``str(key)`` -- JSON objects only allow string keys. Both cases are
        # encoded as an explicit item list that preserves each key's type and value
        # through load. TorchLens never silently stringifies or mis-tags keys
        # (mirrors ``annotate``'s reject-don't-coerce rule).
        if all(type(key) is str for key in value) and not any(
            _is_reserved_wrapper_key(key) for key in value
        ):
            return {key: _serialize_value(item, save_level, state) for key, item in value.items()}
        return {
            "__dict_items__": [
                [_serialize_dict_key(key), _serialize_value(item, save_level, state)]
                for key, item in value.items()
            ]
        }
    if callable(value):
        return {"__callable__": _serialize_callable(value, save_level)}
    return _serialize_opaque(value, save_level)


def _serialize_dict_key(key: Any) -> Any:
    """Serialize a dict key preserving its type through a spec round-trip.

    ``str``/``int``/``float``/``bool``/``None`` survive as native JSON scalars;
    tuples are tagged so they reload as hashable tuples. Any other key type raises
    rather than being silently stringified.

    Parameters
    ----------
    key:
        Runtime dict key.

    Returns
    -------
    Any
        JSON-safe, type-preserving key encoding.
    """

    if key is None or type(key) in (str, int, float, bool):
        return key
    if isinstance(key, tuple):
        return {"__tuple_key__": [_serialize_dict_key(item) for item in key]}
    raise UnserializableDictKeyError(
        f"dict key {key!r} of type {type(key).__name__!r} cannot be preserved through a "
        "spec save/load round-trip; TorchLens refuses to silently stringify it. Use "
        "str/int/float/bool/None keys, or a tuple of those."
    )


def _serialize_output_path_component(
    value: TupleIndex | DictKey | HFKey | NamedField | DataclassField,
    save_level: SaveLevel,
    state: _SerializedState,
) -> dict[str, Any]:
    """Serialize a portable output-container path component.

    Parameters
    ----------
    value:
        Output path component.
    save_level:
        Requested save level.
    state:
        Serialization tensor state.

    Returns
    -------
    dict[str, Any]
        JSON-safe component payload.
    """

    if isinstance(value, TupleIndex):
        return {"__output_path_component__": "tuple_index", "index": value.index}
    if isinstance(value, DictKey):
        return {
            "__output_path_component__": "dict_key",
            "key": _serialize_value(value._key, save_level, state),
        }
    if isinstance(value, HFKey):
        return {
            "__output_path_component__": "hf_key",
            "key": _serialize_value(value._key, save_level, state),
        }
    if isinstance(value, NamedField):
        return {"__output_path_component__": "named_field", "name": value._name}
    return {"__output_path_component__": "dataclass_field", "name": value._name}


def _jsonish_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe metadata dictionary.

    Parameters
    ----------
    metadata:
        Runtime metadata.

    Returns
    -------
    dict[str, Any]
        JSON-safe metadata.
    """

    return {str(key): _metadata_value(value) for key, value in metadata.items()}


def _metadata_value(value: Any) -> Any:
    """Serialize one metadata value conservatively.

    Parameters
    ----------
    value:
        Runtime metadata value.

    Returns
    -------
    Any
        JSON-safe value.
    """

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple | list):
        return [_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    return repr(value)


def _serialize_helper(
    helper: Edit,
    save_level: SaveLevel,
    state: _SerializedState,
) -> dict[str, Any]:
    """Serialize a helper spec with save-level enforcement.

    Parameters
    ----------
    helper:
        Helper spec.
    save_level:
        Requested save level.
    state:
        Serialization tensor state.

    Returns
    -------
    dict[str, Any]
        Serialized helper payload.
    """

    portability = helper.portability
    if save_level == SaveLevel.PORTABLE and portability != "builtin":
        raise OpaqueCallableError(
            f"Portable intervention specs cannot save {portability} helper {helper.name!r}."
        )
    if save_level == SaveLevel.EXECUTABLE_WITH_CALLABLES and portability == "opaque_audit":
        raise OpaqueCallableError(
            f"Executable intervention specs cannot save opaque helper {helper.name!r}."
        )
    if portability == "import_ref":
        import_path = dict(helper.metadata).get("import_path") or _import_path_for_callable(
            helper.factory
        )
        return {"portability": "import_ref", "name": helper.name, "import_path": import_path}
    if portability == "opaque_audit":
        return {
            "portability": "opaque_audit",
            "name": helper.name,
            "repr": repr(helper),
            "is_batch_independent": bool(helper.is_batch_independent),
            "is_compatible_with_append": bool(helper.is_compatible_with_append),
        }

    return {
        "portability": "builtin",
        "name": helper.name,
        "signal": helper.signal,
        "args": [_serialize_value(arg, save_level, state) for arg in helper.args],
        "kwargs": {key: _serialize_value(value, save_level, state) for key, value in helper.kwargs},
        "metadata": _jsonish_metadata(dict(helper.metadata)),
        "is_batch_independent": bool(helper.is_batch_independent),
        "is_compatible_with_append": bool(helper.is_compatible_with_append),
    }


def _serialize_callable(value: Callable[..., Any], save_level: SaveLevel) -> dict[str, Any]:
    """Serialize a callable as import-ref or audit-only repr.

    Parameters
    ----------
    value:
        Callable to serialize.
    save_level:
        Requested save level.

    Returns
    -------
    dict[str, Any]
        Callable payload.
    """

    if isinstance(value, LazyImportRef):
        # R10-12: load->resave round-trip. The loader mints LazyImportRef for
        # import-ref callables precisely so execution-time trust gating can
        # defer the import; the instance exposes no __module__/__qualname__,
        # so the generic path below read None and refused a spec that
        # legitimately saved executable. The ref IS its own import path --
        # resolving it here would perform the import the laziness exists to
        # avoid.
        if save_level == SaveLevel.PORTABLE:
            raise OpaqueCallableError(
                f"Portable intervention specs cannot save import-ref callable {value.import_path}."
            )
        return {
            "portability": "import_ref",
            "import_path": value.import_path,
            "repr": repr(value),
        }
    import_path = _import_path_for_callable(value)
    if import_path is not None and _callable_round_trips(value, import_path):
        if save_level == SaveLevel.PORTABLE:
            raise OpaqueCallableError(
                f"Portable intervention specs cannot save import-ref callable {import_path}."
            )
        return {"portability": "import_ref", "import_path": import_path, "repr": repr(value)}
    if save_level != SaveLevel.AUDIT:
        raise OpaqueCallableError(
            f"Callable {value!r} is opaque and can only be saved at audit level."
        )
    return {"portability": "opaque_audit", "repr": repr(value)}


def _serialize_opaque(value: Any, save_level: SaveLevel) -> dict[str, Any]:
    """Serialize a non-JSON object as audit-only data.

    Parameters
    ----------
    value:
        Opaque value.
    save_level:
        Requested save level.

    Returns
    -------
    dict[str, Any]
        Audit-only payload.
    """

    if save_level != SaveLevel.AUDIT:
        raise OpaqueCallableError(
            f"Value {value!r} is not portable and can only be saved at audit level."
        )
    return {"__opaque_audit__": {"type": type(value).__name__, "repr": repr(value)}}


def _deserialize_value(
    value: Any,
    tensors: dict[str, torch.Tensor],
    *,
    trust_custom_callables: bool = False,
    allowed_custom_callable_modules: Collection[str] | None = None,
) -> Any:
    """Deserialize a value from ``spec.json``.

    Any import-ref callable materialized here captures the load's trust context so
    its deferred resolution enforces the SAME deny-by-default gate as
    ``resolve_function_registry_key``. Defaults are fail-closed: an unthreaded caller
    gets an untrusted (no-foreign-import) reference.

    Parameters
    ----------
    value:
        JSON-decoded value.
    tensors:
        Loaded tensor refs.
    trust_custom_callables:
        Execution-time trust for foreign import-ref callables.
    allowed_custom_callable_modules:
        Optional foreign-module allowlist.

    Returns
    -------
    Any
        Runtime value.
    """

    allowed_tuple: tuple[str, ...] | None = (
        None if allowed_custom_callable_modules is None else tuple(allowed_custom_callable_modules)
    )

    def _decode(item: Any) -> Any:
        """Recurse under the same trust context."""

        return _deserialize_value(
            item,
            tensors,
            trust_custom_callables=trust_custom_callables,
            allowed_custom_callable_modules=allowed_custom_callable_modules,
        )

    def _trusted_import_resolver(import_path: str) -> Callable[..., Any]:
        """Resolve an import ref under this load's trust context."""

        return _resolve_import_ref(
            import_path,
            trust_custom_callables=trust_custom_callables,
            allowed_custom_callable_modules=allowed_custom_callable_modules,
        )

    if isinstance(value, dict) and "__tensor_ref__" in value:
        return tensors[str(value["__tensor_ref__"])]
    if isinstance(value, dict) and "__helper__" in value:
        # Decode a builtin helper's args/kwargs through THIS same full codec so the
        # decoder stays in lockstep with ``_serialize_value``. The narrow
        # ``_decode_jsonish`` fallback only understood ``__tensor_ref__`` and
        # silently returned every other wrapper (``__callable__``/``__opaque_audit__``/
        # ...) as a raw dict, corrupting callable/opaque helper arguments. The import
        # resolver is bound to the load's trust context so an import-ref helper cannot
        # import a foreign module under the default untrusted load.
        return helper_from_serialized(
            value["__helper__"],
            import_resolver=_trusted_import_resolver,
            value_decoder=_decode,
        )
    if isinstance(value, dict) and "__callable__" in value:
        callable_payload = value["__callable__"]
        if callable_payload["portability"] == "import_ref":
            return LazyImportRef(
                str(callable_payload["import_path"]),
                trust_custom_callables=trust_custom_callables,
                allowed_custom_callable_modules=allowed_tuple,
            )
        return Edit(
            name="opaque_audit",
            portability="opaque_audit",
            metadata=(("repr", callable_payload.get("repr", "")), ("executable", False)),
        )
    if isinstance(value, dict) and "__opaque_audit__" in value:
        payload = value["__opaque_audit__"]
        return Edit(
            name=str(payload.get("type", "opaque_audit")),
            portability="opaque_audit",
            metadata=(("repr", payload.get("repr", "")), ("executable", False)),
        )
    if isinstance(value, dict) and "__output_path_component__" in value:
        return _deserialize_output_path_component(
            value,
            tensors,
            trust_custom_callables=trust_custom_callables,
            allowed_custom_callable_modules=allowed_custom_callable_modules,
        )
    if isinstance(value, dict) and "__dict_items__" in value:
        return {_deserialize_dict_key(key): _decode(item) for key, item in value["__dict_items__"]}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    return value


def _deserialize_dict_key(key: Any) -> Any:
    """Reconstruct a dict key encoded by :func:`_serialize_dict_key`.

    Parameters
    ----------
    key:
        JSON-decoded key encoding.

    Returns
    -------
    Any
        Runtime, hashable dict key with its original type restored.
    """

    if isinstance(key, dict) and "__tuple_key__" in key:
        return tuple(_deserialize_dict_key(item) for item in key["__tuple_key__"])
    return key


def _deserialize_output_path_component(
    value: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    *,
    trust_custom_callables: bool = False,
    allowed_custom_callable_modules: Collection[str] | None = None,
) -> TupleIndex | DictKey | HFKey | NamedField | DataclassField:
    """Deserialize a portable output-container path component.

    Parameters
    ----------
    value:
        JSON-safe component payload.
    tensors:
        Loaded tensor refs.
    trust_custom_callables:
        Execution-time trust for foreign import-ref callables. Defaults fail-closed.
    allowed_custom_callable_modules:
        Optional foreign-module allowlist.

    Returns
    -------
    TupleIndex | DictKey | HFKey | NamedField | DataclassField
        Runtime path component.
    """

    def _decode(item: Any) -> Any:
        """Decode a nested key under the same trust context."""

        return _deserialize_value(
            item,
            tensors,
            trust_custom_callables=trust_custom_callables,
            allowed_custom_callable_modules=allowed_custom_callable_modules,
        )

    kind = str(value.get("__output_path_component__"))
    if kind == "tuple_index":
        return TupleIndex(int(value["index"]))
    if kind == "dict_key":
        return DictKey(_decode(value.get("key")))
    if kind == "hf_key":
        return HFKey(_decode(value.get("key")))
    if kind == "named_field":
        return NamedField(str(value["name"]))
    if kind == "dataclass_field":
        return DataclassField(str(value["name"]))
    raise SiteResolutionError(f"Unsupported output path component kind {kind!r}.")


def _import_path_for_callable(value: Any) -> str | None:
    """Return ``module:qualname`` for a callable when available.

    Parameters
    ----------
    value:
        Callable-like object.

    Returns
    -------
    str | None
        Import path or ``None``.
    """

    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not module or not qualname or "<locals>" in qualname:
        return None
    return f"{module}:{qualname}"


def _callable_round_trips(value: Callable[..., Any], import_path: str) -> bool:
    """Return whether an import path resolves to the same callable.

    Parameters
    ----------
    value:
        Callable being serialized.
    import_path:
        Candidate import path.

    Returns
    -------
    bool
        ``True`` when the import path resolves identically.
    """

    try:
        # Save time verifies the identity of the user's OWN in-memory callable, whose
        # module is already imported, so trust is granted here. A callable the shared
        # gate refuses as impure (e.g. a fixed-namespace side-effecting op) simply does
        # not round-trip as an import ref and falls back to opaque/audit serialization.
        return _resolve_import_ref(import_path, trust_custom_callables=True) is value
    except (
        AttributeError,
        ImportError,
        ValueError,
        TypeError,
        UntrustedCallableError,
        ReplayPreconditionError,
    ):
        return False


def _resolve_import_ref(
    import_path: str,
    *,
    trust_custom_callables: bool = False,
    allowed_custom_callable_modules: Collection[str] | None = None,
) -> Callable[..., Any]:
    """Resolve a ``module:qualname`` import reference through the shared trust gate.

    Delegates to :func:`torchlens.intervention._resolver_registry.resolve_import_ref` so a
    bundle-supplied import reference obeys the SAME deny-by-default contract as every
    other bundle-reachable callable resolution: fixed torch/operator namespaces and
    TorchLens-owned helpers always resolve (purity-gated), while a genuinely foreign
    module default-denies with :class:`UntrustedCallableError` and is never imported
    unless the caller opted into trust. Defaults are fail-closed.

    Parameters
    ----------
    import_path:
        Import reference in ``module:qualname`` form.
    trust_custom_callables:
        Explicit permission to import a foreign custom callable when no allowlist is
        supplied.
    allowed_custom_callable_modules:
        Optional allowlist of custom callable module names.

    Returns
    -------
    Callable[..., Any]
        Resolved callable.
    """

    return resolve_import_ref(
        import_path,
        trust_custom_callables=trust_custom_callables,
        allowed_custom_callable_modules=allowed_custom_callable_modules,
    )
