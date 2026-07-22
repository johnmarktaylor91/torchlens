"""Single-sourced model-input boundary traversal (r65 Cluster Y, the normative dispatch).

Every walker of the MODEL-INPUT boundary tree routes through
:func:`walk_input_boundary`:

* W1 -- capture literal-leaf walker
  (``torchlens.capture.trace._record_runnable_input_literal_leaves``),
* W2 -- capture metadata-read-site walker
  (``torchlens.capture.trace._record_runnable_input_tensor_sites``),
* W3 -- runtime literal-path walker
  (``torchlens._runnable_execution._runtime_nontensor_leaf_paths``).

One shared container dispatch is an HONESTY invariant, not a refactor: the walkers
witness the same physical tree for different fact families (literal values vs
metadata-read sites), so a container kind handled by one walker but missed by another
silently drops a whole fact family for every leaf beneath it. That is exactly the r64
Finding-1 false-VERIFIED: the metadata-site walker did not descend dataclass
containers, so ``box.x.is_contiguous()`` on a dataclass input field recorded no
metadata witness and a same-value non-contiguous twin replayed the captured branch as
``verified``. With the dispatch single-sourced here, a future container kind MUST be
added to this module and is picked up by every walker in lockstep; a private container
branch inside a walker body is forbidden by a source-scan meta-test
(``tests/test_tlspec_runnable_r65_walker_parity.py``).

Container-kind vocabulary (CLOSED SET -- the supported model-input container set):

    ``tensor`` (leaf) / ``empty`` (zero-length dict-list-tuple-namedtuple) /
    ``namedtuple`` / ``dataclass`` / ``mapping`` / ``sequence`` (list-tuple) /
    ``leaf`` (anything else, including genuinely-opaque objects)

Dispatch ORDER is fixed and load-bearing: tensor, empty container, namedtuple,
dataclass, mapping, sequence, leaf. Empty precedes namedtuple so a zero-field
namedtuple is witnessed as an EMPTY container by KIND; namedtuple precedes dataclass
and mapping so hybrid types keep their historical classification.

Dual mapping-key vocabulary (DECLARED here, residual R6)
--------------------------------------------------------
Two mapping-key path-component vocabularies coexist, both defined in THIS module and
only here:

* :func:`tagged_mapping_key_component` -- the persisted LITERAL vocabulary (W1/W3,
  decoded at run time by ``_value_at_path``): keys are gated by the frozen literal
  grammar (a non-representable key routes its whole child subtree to
  ``on_opaque_key_subtree``, downgrading witness coverage instead of silently dropping
  the subtree) and a bool key is tagged ``(BOOL_KEY_PATH_TAG, key)`` so ``{True: x}``
  stays type-distinct from ``{1: x}`` in the persisted leaf-path set (r29-C2 / F6).
* :func:`raw_mapping_key_component` -- the metadata-fact-site vocabulary (W2, and the
  historical binding/W4 path family): the raw key object, every key accepted. Raw
  bool/int keys CONFLATE (``{True: x}`` and ``{1: x}`` hash equal), which is honest
  today because (a) fact-site paths are in-memory attribution keys resolved through a
  mapping that conflates the two identically at capture and at run, and (b) the
  bool-vs-int input-tree swap itself diverges through the r33 ``_type_strict_path``
  symmetric belt on the tensor-leaf set contract. Re-vocabularying W2 to tagged keys
  would change persisted fact paths for zero honesty gain. Full single-vocabulary
  unification is residual R6 -- a deliberate future persisted-path migration round,
  never a silent drift.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

__all__ = [
    "INPUT_CONTAINER_KINDS",
    "InstanceStateInspection",
    "classify_input_container",
    "classify_scalar",
    "inspect_instance_state",
    "instance_state_names",
    "raw_mapping_key_component",
    "tagged_mapping_key_component",
    "walk_input_boundary",
]


INPUT_CONTAINER_KINDS: frozenset[str] = frozenset(
    {"tensor", "empty", "namedtuple", "dataclass", "mapping", "sequence", "registered", "leaf"}
)
"""The CLOSED container-kind vocabulary of the model-input boundary (see module doc).

r67 C2 adds ``registered`` (a ``tl.register_container`` type, descended through its own
``flatten`` with indexed children) and extends ``empty`` to zero-field dataclasses.
"""


def classify_input_container(value: Any) -> str:
    """Classify one boundary value into the closed container-kind vocabulary.

    This is THE single container dispatch (order is load-bearing; see the module
    docstring). :func:`walk_input_boundary` consumes it, so adding a container kind
    here extends every input-boundary walker in lockstep.
    """

    import torch

    from torchlens._io.runnable import empty_container_kind
    from torchlens.ir.container import get_registered_container

    if isinstance(value, torch.Tensor):
        return "tensor"
    # r67 C2 (corr1-3): a REGISTERED container type is the user's explicit declaration and
    # wins over the generic kinds -- capture and runtime both descend its ``flatten``
    # children, so a registered input can never be admitted at save and then fail
    # traversal at bind (advertise-then-fail).
    if not isinstance(value, type) and get_registered_container(type(value)) is not None:
        return "registered"
    if empty_container_kind(value) is not None:
        return "empty"
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return "namedtuple"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return "dataclass"
    if isinstance(value, Mapping):
        return "mapping"
    if isinstance(value, (list, tuple)):
        return "sequence"
    return "leaf"


def tagged_mapping_key_component(key: Any) -> Any:
    """Persisted-literal mapping-key vocabulary (W1/W3; decoded by ``_value_at_path``).

    Gates the key through the frozen literal grammar (``_encode_literal_key`` raises
    ``_UnsupportedLiteralError`` for a non-representable key -- enum, object, bytes,
    ... -- which :func:`walk_input_boundary` routes to ``on_opaque_key_subtree``) and
    tags a bool key as ``(BOOL_KEY_PATH_TAG, key)`` so it stays type-distinct from
    the equal-valued int key (r29-C2 / F6).
    """

    from torchlens._io.runnable import _encode_literal_key, input_path_key_component

    _encode_literal_key(key)
    return input_path_key_component(key)


def raw_mapping_key_component(key: Any) -> Any:
    """Metadata-fact-site mapping-key vocabulary (W2): the ONE codec, every key accepted.

    r67 C2 retires the dual vocabulary (residual R6): W2 components now come from the
    SAME type-strict codec as the persisted literal paths, so ``{True: x}`` and
    ``{1: x}`` fact sites stay distinct and every path family speaks one language. A
    NON-grammar key keeps the raw key object as an in-memory-only attribution component
    (W2 must accept every key; the literal walker's OPAQUE leaf independently ceilings
    such a subtree, so a dropped metadata fact under an exotic key can never yield a
    false VERIFIED).
    """

    try:
        return encode_mapping_key(key)
    except ValueError:
        return key


def walk_input_boundary(
    value: Any,
    path: tuple[Any, ...] = (),
    *,
    key_component: Callable[[Any], Any],
    on_tensor: Callable[[Any, tuple[Any, ...]], None] | None = None,
    on_leaf: Callable[[Any, tuple[Any, ...]], None] | None = None,
    on_empty_container: Callable[[str, tuple[Any, ...]], None] | None = None,
    on_opaque_key_subtree: Callable[[Any, tuple[Any, ...]], None] | None = None,
) -> None:
    """Traverse one model-input boundary value with the single container dispatch.

    Parameters
    ----------
    value:
        The boundary value to descend.
    path:
        Container path accumulated so far (empty at a top-level site).
    key_component:
        REQUIRED mapping-key vocabulary hook -- one of
        :func:`tagged_mapping_key_component` / :func:`raw_mapping_key_component`
        (every caller must consciously pick a declared vocabulary). It returns the
        path component for a mapping key, or raises ``_UnsupportedLiteralError`` to
        route the key's child subtree to ``on_opaque_key_subtree``.
    on_tensor:
        Called ``(tensor, path)`` for every tensor leaf. ``None`` skips tensors.
    on_leaf:
        Called ``(value, path)`` for every non-tensor, non-container leaf.
        ``None`` ignores such leaves.
    on_empty_container:
        Called ``(kind, path)`` for every EMPTY container, where ``kind`` is the
        ``empty_container_kind`` string. ``None`` ignores empty containers.
    on_opaque_key_subtree:
        Called ``(child, parent_path)`` once per mapping key rejected by
        ``key_component`` (the child subtree is NOT descended). ``None`` skips.
    """

    from torchlens._io.runnable import _UnsupportedLiteralError, empty_container_kind

    def _descend(value: Any, path: tuple[Any, ...]) -> None:
        """Dispatch one node through the closed container-kind vocabulary."""

        kind = classify_input_container(value)
        if kind == "tensor":
            if on_tensor is not None:
                on_tensor(value, path)
            return
        if kind == "empty":
            if on_empty_container is not None:
                empty_kind = empty_container_kind(value)
                assert empty_kind is not None  # classify_input_container said "empty"
                on_empty_container(empty_kind, path)
            return
        if kind == "registered":
            # r67 C2 (corr1-3): descend the registration's OWN flatten children with
            # indexed components; a throwing/nonconforming hook routes the node to the
            # opaque handler (fail closed: witness coverage downgrades / save refuses),
            # never a silent drop and never an advertise-then-fail at bind.
            from torchlens.ir.container import get_registered_container

            registration = get_registered_container(type(value))
            try:
                children = list(registration.flatten(value)[0]) if registration else None
            except Exception:
                children = None
            if children is None:
                if on_opaque_key_subtree is not None:
                    on_opaque_key_subtree(value, path)
                return
            for index, child in enumerate(children):
                _descend(child, (*path, index))
            return
        if kind == "namedtuple":
            for name in _instance_fields(value):
                _descend(getattr(value, name), (*path, str(name)))
            return
        if kind == "dataclass":
            for field in dataclasses.fields(value):
                _descend(getattr(value, field.name), (*path, field.name))
            return
        if kind == "mapping":
            for key, child in value.items():
                try:
                    component = key_component(key)
                except _UnsupportedLiteralError:
                    if on_opaque_key_subtree is not None:
                        on_opaque_key_subtree(child, path)
                    continue
                _descend(child, (*path, component))
            return
        if kind == "sequence":
            for index, child in enumerate(value):
                _descend(child, (*path, index))
            return
        if on_leaf is not None:
            on_leaf(value, path)

    _descend(value, path)


# --- r67 C2: the input-boundary SNAPSHOT spine -----------------------------------------------
#
# The r66 C2 findings (free-F2/F3, hon1-F1/F2a/F2b/F2c, corr1-3) share one root: the structure
# witness recorded ROOT kind only, so nested kinds, exact classes, empty-dataclass arity,
# non-str/int grammar keys, hidden instance state, and registered containers could be lost or
# treated inconsistently. One traversal per model-input site now produces an immutable
# per-node record list -- kind, exact (module, qualname) type, declared child schema, ordered
# type-strict-encoded mapping keys, and an instance-state/losslessness proof -- persisted by
# the runnable producer and re-derived at bind time by the SAME function, so capture and
# runtime can never diverge in vocabulary.

_KEY_CODEC_TAG = "\x00tlk:"
"""Reserved prefix of the canonical type-strict mapping-key codec (r67 C2).

Every non-``str``/``int`` grammar key (bool, float, None, safe tuple) -- and any string key
that could collide with an encoding -- is encoded into ONE tagged string component, so the
persisted path/key shape stays ``str | int`` while ``True != 1`` and ``1.0 != 1`` stay
type-distinct (``hash(True) == hash(1) == hash(1.0)`` would otherwise conflate them in path
sets and ordered-key facts).
"""


def _stock_numpy_scalar_types() -> "frozenset[type]":
    """Every CONCRETE stock NumPy scalar class, enumerated programmatically (r69 B).

    Identity-based membership from ``np.typecodes['All']`` -- never a name/module
    string check, so a user subclass can never enter the set regardless of spoofed
    ``__module__``/``__qualname__``.
    """

    global _STOCK_NP_SCALAR_CACHE
    if _STOCK_NP_SCALAR_CACHE is None:
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - numpy is a hard torchlens dependency
            _STOCK_NP_SCALAR_CACHE = (frozenset(), frozenset())
        else:
            all_types: set[type] = set()
            transparent: set[type] = set()
            for code in np.typecodes["All"]:
                try:
                    dtype = np.dtype(code)
                except TypeError:  # pragma: no cover - stale typecode entry
                    continue
                all_types.add(dtype.type)
                # Transparent VALUE wrappers: numeric/bool kinds whose ``.item()``
                # round-trips exactly (itemsize <= 8 excludes lossy longdouble).
                if dtype.kind in "biuf" and dtype.itemsize <= 8:
                    transparent.add(dtype.type)
            _STOCK_NP_SCALAR_CACHE = (frozenset(all_types), frozenset(transparent))
    return _STOCK_NP_SCALAR_CACHE[0]


def _stock_numpy_transparent_scalar_types() -> "frozenset[type]":
    """Exact stock NumPy numeric/bool wrapper classes admitted as VALUE-transparent."""

    _stock_numpy_scalar_types()
    assert _STOCK_NP_SCALAR_CACHE is not None
    return _STOCK_NP_SCALAR_CACHE[1]


_STOCK_NP_SCALAR_CACHE: "tuple[frozenset[type], frozenset[type]] | None" = None


def classify_scalar(value: Any) -> "tuple[str, Any]":
    """THE closed input-boundary scalar type-class lattice (r69 B/D).

    Every scalar type dispatch on the model-input boundary -- snapshot leaf
    disposition, literal encoding (``_encode_literal``), runtime literal comparison
    (``_literal_leaf_equal``), and mapping-key encoding (``encode_mapping_key``) --
    consumes this ONE classifier, so a scalar can never be classified two different
    ways by two different code paths (the r68 B/D conflation root). Classification
    order is load-bearing:

    1. ``("semantic", None)`` -- ``enum.Enum`` FIRST (``IntEnum`` / ``IntFlag`` /
       ``StrEnum`` / float-Enum / plain ``Enum``), before any numeric family, then
       (after the exact/stock checks below) every other ``int``/``float``/``str``
       subclass and every non-stock ``np.generic`` subclass. Type identity is
       application semantics (``isinstance``/``is Mode.FAST`` steering); such a
       VALUE refuses runnable save typed and such a runtime value diverges -- it
       never collapses into the builtin atom lane.
    2. ``("plain", value)`` -- ``type(value)`` is EXACTLY ``bool``/``int``/``float``/
       ``str`` (exact ``type() is``, never ``isinstance``).
    3. ``("singleton", value)`` -- ``None`` / ``Ellipsis`` (their existing grammar).
    4. ``("numpy", value.item())`` -- ``type(value)`` is EXACTLY a stock NumPy
       numeric/bool wrapper class (programmatic identity set; broad
       ``isinstance(value, np.generic)`` is forbidden for this admission) AND
       ``.item()`` lands on an exact builtin numeric/bool atom. The RATIFIED
       transparent-wrapper VALUE lane: ``np.float64(2.0) <-> 2.0`` verifies.
       A stock wrapper whose ``.item()`` is not an exact builtin numeric/bool
       falls to ``semantic`` (fail closed, platform ``longdouble`` edges).
    5. ``("opaque", None)`` -- everything else (bytes, complex, Decimal, datetime64,
       arbitrary objects, containers): the existing opaque/outside-grammar lane.
    """

    import enum as _enum

    if isinstance(value, _enum.Enum):
        return ("semantic", None)
    value_type = type(value)
    if value_type in (bool, int, float, str):
        return ("plain", value)
    if value is None or value is Ellipsis:
        return ("singleton", value)
    if value_type in _stock_numpy_transparent_scalar_types():
        try:
            item = value.item()
        except Exception:  # pragma: no cover - stock wrappers implement item()
            return ("semantic", None)
        if type(item) in (bool, int, float):
            return ("numpy", item)
        return ("semantic", None)
    if isinstance(value, (int, float, str)):
        # A builtin-scalar-family subclass outside the exact/stock sets (custom
        # int/float/str subclasses, np.str_, subclasses of np.float64, ...).
        return ("semantic", None)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard torchlens dependency
        return ("opaque", None)
    if isinstance(value, np.generic) and value_type not in _stock_numpy_scalar_types():
        # A non-stock np.generic subclass is user semantics, never a wrapper.
        return ("semantic", None)
    return ("opaque", None)


def encode_mapping_key(key: Any) -> "str | int":
    """Encode one grammar mapping key into the canonical type-strict component (r69 D).

    THE sole mapping-key identity authority: snapshot ordered-key facts, literal
    leaf paths, tensor binding paths, and runtime lookup all speak this token
    vocabulary, and runtime binding compares canonical tokens -- never decoded
    values. Admission is classifier-gated (exact builtin ``str``/``int``/``bool``/
    ``float``/``None`` and EXACT tuples whose members recursively satisfy the
    grammar); enums, NumPy scalar wrappers, builtin-scalar subclasses, and tuple
    subclasses raise ``ValueError`` and refuse typed at save (``opaque_mapping_key``)
    -- a key is never admitted and then advertised-then-failed later.

    Token codec: ``str`` keys pass through unless they collide with the codec prefix
    (then escaped); ``int`` keys pass through as ints; ``bool``/``None`` use tagged
    strings; finite floats use exact ``float.hex()`` (``-0.0`` distinct from
    ``+0.0``); a NaN key encodes its binary64 BIT PATTERN
    (``f:nan:<16 hex digits>``) so distinct NaN payloads stay distinct and an
    identical NaN key can bind itself. The codec is injective over every admitted
    mapping node (duplicate-token nodes refuse at save as ``ambiguous_mapping_key``).
    """

    kind, _payload = classify_scalar(key)
    if kind == "semantic":
        raise ValueError(
            f"Mapping key {key!r} carries semantic type identity (enum or scalar "
            "subclass) and is outside the input-boundary key grammar."
        )
    if kind == "numpy":
        raise ValueError(
            f"Mapping key {key!r} is a NumPy scalar wrapper; key identity is "
            "structural, so wrapper keys are outside the input-boundary key grammar."
        )
    if isinstance(key, bool):
        return f"{_KEY_CODEC_TAG}b:{'1' if key else '0'}"
    if type(key) is int:
        return key
    if type(key) is str:
        if key.startswith(_KEY_CODEC_TAG):
            return f"{_KEY_CODEC_TAG}s:{key[len(_KEY_CODEC_TAG) :]}"
        return key
    if key is None:
        return f"{_KEY_CODEC_TAG}n:"
    if type(key) is float:
        if key != key:  # NaN: float.hex() collapses every payload to "nan"
            import struct as _struct

            return f"{_KEY_CODEC_TAG}f:nan:{_struct.pack('>d', key).hex()}"
        return f"{_KEY_CODEC_TAG}f:{key.hex()}"
    if type(key) is tuple:
        parts = []
        for item in key:
            encoded = encode_mapping_key(item)
            text = f"i:{encoded}" if isinstance(encoded, int) else f"c:{encoded}"
            parts.append(text.replace("\\", "\\\\").replace("|", "\\|"))
        return f"{_KEY_CODEC_TAG}t:{'|'.join(parts)}"
    raise ValueError(f"Mapping key {key!r} is outside the input-boundary key grammar.")


def decode_mapping_key(component: "str | int") -> Any:
    """Decode one canonical component back to its mapping key (mapping nodes only).

    Retained for legitimate output-dict reconstruction and diagnostics; RUNTIME
    INPUT BINDING never decodes -- ``_value_at_path`` encodes each runtime candidate
    and compares exact canonical tokens (r69 D).
    """

    if isinstance(component, int) or not component.startswith(_KEY_CODEC_TAG):
        return component
    body = component[len(_KEY_CODEC_TAG) :]
    tag, _, payload = body.partition(":")
    if tag == "b":
        return payload == "1"
    if tag == "s":
        return _KEY_CODEC_TAG + payload
    if tag == "n":
        return None
    if tag == "f":
        if payload.startswith("nan:"):
            import struct as _struct

            return _struct.unpack(">d", bytes.fromhex(payload[len("nan:") :]))[0]
        return float.fromhex(payload)
    if tag == "t":
        items = []
        part = ""
        escaped = False
        parts: list[str] = []
        for ch in payload:
            if escaped:
                part += ch
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "|":
                parts.append(part)
                part = ""
            else:
                part += ch
        if part or payload:
            parts.append(part)
        for text in parts:
            kind, _, inner = text.partition(":")
            items.append(int(inner) if kind == "i" else decode_mapping_key(inner))
        return tuple(items)
    raise ValueError(f"Unknown key-codec component {component!r}.")


@dataclass(frozen=True, slots=True)
class InstanceStateInspection:
    """One inert instance-state inspection result (r71 C).

    ``complete`` is ``True`` only when the enumeration is PROVABLY total inertly;
    ``reason`` names the closed fail-closed cause otherwise. ``uncertainty is NEVER
    an empty set``: a blinded inspection reports ``complete=False`` with a non-empty
    reason, never ``names=frozenset()`` masquerading as "no extra state".
    """

    names: frozenset[str]
    complete: bool
    reason: str | None


def _raw_mro_attr(value: Any, attr: str) -> Any:
    """Resolve one attribute through raw class-dict MRO lookup (never ``getattr``).

    Returns the WINNING descriptor/object from ``type(value).__mro__`` without
    invoking any descriptor protocol, ``__getattribute__``, or ``__getattr__`` -- so
    a property/custom-descriptor shadow is OBSERVED, never executed.
    """

    for cls in type(value).__mro__:
        if attr in cls.__dict__:
            return cls.__dict__[attr]
    return None


def inspect_instance_state(value: Any) -> InstanceStateInspection:
    """Inertly enumerate the SET instance-state NAMES on ``value``, fail-closed (r71 C).

    THE single instance-state inspector for declared-schema containers. It NEVER routes the instance ``__dict__`` through ``getattr`` (r70 hon1-F1: a property/descriptor-shadowed
    ``__dict__`` executes user code returning attacker-chosen ``{}``, turning
    "untrustworthy" into "proved empty"). Instead:

    * The instance ``__dict__`` descriptor is resolved by RAW MRO lookup. Absence is
      valid (a genuinely slotted type). If present it must be EXACTLY
      ``types.GetSetDescriptorType`` (the standard instance-dict getset); it is
      invoked DIRECTLY and the result must be ``type(result) is dict``. A property,
      custom/replaced descriptor, dict-subclass/non-dict result, or an exception is
      INCOMPLETE (``instance_state_uninspectable``).
    * The all-MRO genuine ``MemberDescriptorType`` slot scan is unchanged (unset
      slots absent, mangled private names, falsey values counted -- inert).

    PRESENCE is state (a ``None``-valued extra counts). Returns ``complete=True`` with
    the actual storage names only when the enumeration is inertly total.
    """

    import types as _types

    names: set[str] = set()
    dict_descriptor = _raw_mro_attr(value, "__dict__")
    if dict_descriptor is not None:
        if type(dict_descriptor) is not _types.GetSetDescriptorType:
            # A property or non-standard descriptor shadows __dict__: reading it would
            # execute user code and could report attacker-chosen state. Fail closed.
            return InstanceStateInspection(frozenset(), False, "instance_state_uninspectable")
        try:
            instance_dict = dict_descriptor.__get__(value, type(value))
        except Exception:
            return InstanceStateInspection(frozenset(), False, "instance_state_uninspectable")
        if type(instance_dict) is not dict:
            # A dict SUBCLASS (or non-dict) result could carry a custom __iter__ /
            # __contains__ that hides keys: not inertly trustworthy.
            return InstanceStateInspection(frozenset(), False, "instance_state_uninspectable")
        names.update(str(key) for key in instance_dict)
    for cls in type(value).__mro__:
        raw_slots = cls.__dict__.get("__slots__")
        if raw_slots is None:
            continue
        slot_names = [raw_slots] if isinstance(raw_slots, str) else [str(s) for s in raw_slots]
        for raw_name in slot_names:
            if raw_name in {"__dict__", "__weakref__"}:
                continue
            name = raw_name
            if raw_name.startswith("__") and not raw_name.endswith("__"):
                # CPython mangles private slot names against the DECLARING class.
                stripped = cls.__name__.lstrip("_")
                if stripped:
                    name = f"_{stripped}{raw_name}"
            descriptor = cls.__dict__.get(name)
            if not isinstance(descriptor, _types.MemberDescriptorType):
                # A shadowed/replaced slot descriptor makes the slot storage
                # unreachable by ordinary attribute access; reading anything else
                # here could execute user code, so the name is skipped (inert).
                continue
            try:
                descriptor.__get__(value, cls)
            except AttributeError:
                continue  # unset slot: absent
            names.add(name)
    return InstanceStateInspection(frozenset(names), True, None)


def instance_state_names(value: Any) -> frozenset[str]:
    """Enumerate the SET instance-state names inertly (r69 C; r71 C fail-closed helper).

    Thin compatibility wrapper over :func:`inspect_instance_state`: an INCOMPLETE
    (uninspectable) inspection returns the empty set here for legacy callers, but the
    declared-schema gate consumes the full :class:`InstanceStateInspection` so a
    blinded inspection fails closed rather than reading as "no extra state".
    """

    return inspect_instance_state(value).names


def _declared_schema_uninspectable(value: Any) -> bool:
    """Return whether a custom attribute hook blinds the declared-field proof (r71 C).

    A dataclass/namedtuple with a non-standard ``__getattribute__`` or ANY MRO
    ``__getattr__`` can compute or hide declared-field values, so the walker cannot
    inertly prove field completeness WITHOUT invoking the untrusted hook. Raw-MRO
    resolution observes the hooks without executing them; any override short-circuits
    to uninspectable BEFORE any declared-field read. Namedtuples pass by default
    (``tuple`` does not override ``__getattribute__`` and declares no ``__getattr__``).
    """

    import types as _types

    getattribute = _raw_mro_attr(value, "__getattribute__")
    # A USER override is a Python callable (``FunctionType``); a builtin standard
    # ``__getattribute__`` (``object`` / ``tuple`` for namedtuples) is a C-level
    # ``wrapper_descriptor`` that cannot run arbitrary user code. Only a non-builtin
    # override blinds the proof.
    if getattribute is not object.__getattribute__ and not isinstance(
        getattribute, _types.WrapperDescriptorType
    ):
        return True
    # ``object`` / ``tuple`` declare NO ``__getattr__``, so any MRO ``__getattr__`` is
    # a user-added hook that can compute or hide declared-field values.
    for cls in type(value).__mro__:
        if "__getattr__" in cls.__dict__:
            return True
    return False


def undeclared_instance_state(value: Any, kind: str) -> bool:
    """Return whether a container INSTANCE carries state its DECLARED schema drops (r69 C).

    Scope: kinds with a declared FIELD schema. A dataclass or namedtuple subclass
    carrying live instance state outside its declared fields would silently lose that
    state on replay reconstruction -- and it can steer control flow the declared schema
    never witnesses (r66 R1: ``box.mode``; r68 hon1-F2: a ``__slots__`` subclass; r68
    hon1-F3: ``hasattr`` presence on a ``None``-valued extra). The live state set comes
    from the ONE inert inspector :func:`inspect_instance_state` (raw-MRO ``__dict__``
    descriptor + all-MRO set slots, presence counting regardless of value -- NO
    ``None`` carve-out), minus the declared field names.

    r71 C fail-closed: an UNINSPECTABLE inspection (property/descriptor-shadowed
    ``__dict__``, non-dict result, custom ``__getattribute__`` / ``__getattr__`` on
    the declared-schema container) reads as undeclared state PRESENT -- the walker
    can never inertly prove completeness, and it must never invoke the shadowing hook
    to try. Registered containers prove completeness through their explicit
    ``state_complete`` declaration instead (checked by the snapshot's registered
    branch).

    Mapping/sequence SUBCLASSES are deliberately NOT judged here (heavy-gate F7 ruling):
    their witnessed schema is the protocol view (ordered keys/children) plus the EXACT
    class identity the snapshot already records and compares, and a custom Mapping
    legitimately keeps its backing store in ``__dict__`` (``self.data = {...}``) --
    blanket-refusing it would reject every well-behaved custom Mapping input the
    incumbent F7 contract admits. A hidden non-protocol attribute steering control flow
    on such a subclass remains the documented residual (attribute reads are invisible
    to every Python-level net), never a false structural pass: the class swap itself
    still diverges through the exact-type node fact.
    """

    import dataclasses as _dc

    is_dataclass_kind = kind == "dataclass" or (kind == "empty" and _dc.is_dataclass(value))
    is_namedtuple_kind = kind == "namedtuple" or (
        kind == "empty" and isinstance(value, tuple) and hasattr(value, "_fields")
    )
    if not (is_dataclass_kind or is_namedtuple_kind):
        return False
    # r71 C: a custom attribute hook on the declared-schema container blinds the
    # field-completeness proof BEFORE any field is read -- fail closed.
    if _declared_schema_uninspectable(value):
        return True
    inspection = inspect_instance_state(value)
    if not inspection.complete:
        return True
    if is_dataclass_kind:
        declared = {field.name for field in _dc.fields(value)}
    else:
        declared = {str(name) for name in _instance_fields(value)}
    return bool(inspection.names - declared)


def _instance_fields(value: Any) -> tuple[str, ...]:
    """Read a namedtuple's ``_fields`` at TYPE level (r71 C; inert, no instance hook).

    Resolves ``_fields`` through the raw MRO so a shadowed instance ``_fields`` cannot
    forge the declared schema; requires a tuple of strings (else empty -> the
    all-state set is treated as undeclared, fail closed at the caller).
    """

    fields = _raw_mro_attr(value, "_fields")
    if isinstance(fields, tuple) and all(isinstance(name, str) for name in fields):
        return fields
    return ()


def snapshot_input_boundary(value: Any) -> dict[str, Any]:
    """Build ONE immutable per-node structure snapshot of a model-input site (r67 C2).

    Returns ``{"nodes": [...], "refusals": [...]}`` where every node record carries
    ``path`` (canonical components), ``kind`` (closed vocabulary + ``registered``),
    ``type`` (exact ``[module, qualname]``), and the declared child schema (``fields``
    for namedtuple/dataclass, ordered codec-encoded ``keys`` for mappings, ``size`` for
    sequences/registered, ``empty_kind`` for empty nodes, ``aux`` for registered
    containers). Refusals name the path and reason -- opaque keys, undeclared instance
    state, throwing/nonconforming or state-incomplete registrations -- and the runnable
    producer refuses the save on any of them (the existing
    ``missing_input_container_contract``); analysis captures are unaffected.
    """

    import dataclasses as _dc

    from torchlens._io.runnable import empty_container_kind
    from torchlens.ir.container import get_registered_container

    nodes: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []

    def _type_ref(item: Any) -> list[str]:
        cls = type(item)
        return [str(cls.__module__), str(cls.__qualname__)]

    def _descend(item: Any, path: tuple[Any, ...]) -> None:
        kind = classify_input_container(item)
        # r67 heavy-gate fix: only CONTAINER nodes carry the exact-class witness. A
        # tensor leaf's identity is the admission gate's domain, and a scalar LEAF
        # literal's semantics are the literal-witness VALUE contract (numeric equality
        # across float subclasses: a captured ``np.float64(2.0)`` re-run with plain
        # ``2.0`` must verify) -- an exact-type fact on leaves would structurally
        # diverge value-equal inputs the value contract admits.
        node: dict[str, Any] = {"path": list(path), "kind": kind}
        if kind not in {"tensor", "leaf"}:
            node["type"] = _type_ref(item)
        if kind == "tensor":
            nodes.append(node)
            return
        registration = None
        if kind not in {"tensor", "leaf"}:
            registration = get_registered_container(type(item))
        if registration is not None:
            node["kind"] = "registered"
            # r71 C adjacent closure: the registration's ``state_complete`` trust
            # rule reads the instance ``__dict__`` through the ONE inert inspector
            # (never a live ``getattr`` __dict__ read -- a property-shadowed __dict__
            # would blind or forge the state check). An uninspectable registered
            # container refuses fail-closed; the explicit ``state_complete=True``
            # trust is preserved.
            if not registration.state_complete:
                inspection = inspect_instance_state(item)
                if not inspection.complete:
                    refusals.append({"path": list(path), "reason": "instance_state_uninspectable"})
                elif inspection.names:
                    refusals.append(
                        {"path": list(path), "reason": "registered_state_not_declared_complete"}
                    )
            try:
                children, aux = registration.flatten(item)
                children = list(children)
            except Exception:
                refusals.append({"path": list(path), "reason": "registered_flatten_failed"})
                nodes.append(node)
                return
            try:
                node["aux"] = _safe_aux(aux)
            except ValueError:
                refusals.append({"path": list(path), "reason": "registered_aux_unsafe"})
                nodes.append(node)
                return
            node["size"] = len(children)
            nodes.append(node)
            for index, child in enumerate(children):
                _descend(child, (*path, index))
            return
        # r71 C: a declared-schema container whose attribute hooks / __dict__ cannot
        # be inspected inertly refuses with the DISTINCT ``instance_state_uninspectable``
        # reason BEFORE any declared field is read, so the shadowing hook never runs.
        if kind in {"dataclass", "namedtuple", "empty"} and _declared_schema_uninspectable(item):
            refusals.append({"path": list(path), "reason": "instance_state_uninspectable"})
            nodes.append(node)
            return
        if (
            kind in {"dataclass", "namedtuple", "empty"}
            and not inspect_instance_state(item).complete
        ):
            refusals.append({"path": list(path), "reason": "instance_state_uninspectable"})
            nodes.append(node)
            return
        if undeclared_instance_state(item, kind):
            refusals.append({"path": list(path), "reason": "undeclared_instance_state"})
        if kind == "empty":
            empty_kind = empty_container_kind(item)
            node["empty_kind"] = str(empty_kind)
            nodes.append(node)
            return
        if kind == "namedtuple":
            fields = _instance_fields(item)
            node["fields"] = [str(name) for name in fields]
            nodes.append(node)
            for name in fields:
                _descend(getattr(item, name), (*path, str(name)))
            return
        if kind == "dataclass":
            node["fields"] = [field.name for field in _dc.fields(item)]
            nodes.append(node)
            for field in _dc.fields(item):
                _descend(getattr(item, field.name), (*path, field.name))
            return
        if kind == "mapping":
            keys: list[Any] = []
            encodable = True
            for key in item.keys():
                try:
                    keys.append(encode_mapping_key(key))
                except ValueError:
                    refusals.append({"path": list(path), "reason": "opaque_mapping_key"})
                    encodable = False
                    break
            if encodable and len(set(keys)) != len(keys):
                # r69 D: two distinct key objects encoding to ONE canonical token
                # (e.g. multiple same-bit NaN objects -- hash-equal, eq-False, so
                # separate dict slots) make key identity ambiguous: ordered-key
                # facts self-collide and token binding would be first-match-wins.
                # Refuse typed at save; the symmetric runtime snapshot refusal
                # fails the structure check on a runtime-ambiguous mapping.
                refusals.append({"path": list(path), "reason": "ambiguous_mapping_key"})
                encodable = False
            node["keys"] = keys if encodable else []
            nodes.append(node)
            if encodable:
                for key, child in item.items():
                    _descend(child, (*path, encode_mapping_key(key)))
            return
        if kind == "sequence":
            node["size"] = len(item)
            nodes.append(node)
            for index, child in enumerate(item):
                _descend(child, (*path, index))
            return
        # Literal leaf: the VALUE is witnessed by the literal walker, but the scalar
        # TYPE-CLASS disposition is decided here by the ONE classifier (r69 B). A
        # semantic-typed scalar (enum member, builtin/np-scalar subclass) steers host
        # control flow through type identity that no persisted value fact can carry,
        # so it refuses the runnable save typed (``semantic_scalar_type``); the SAME
        # refusal fires in the runtime snapshot, so an admitted builtin/stock-numpy
        # capture replayed with a same-value semantic type diverges structurally.
        if classify_scalar(item)[0] == "semantic":
            refusals.append({"path": list(path), "reason": "semantic_scalar_type"})
        nodes.append(node)

    def _safe_aux(aux: Any) -> Any:
        if aux is None or isinstance(aux, (bool, int, float, str)):
            return aux
        if isinstance(aux, (list, tuple)):
            return [_safe_aux(item) for item in aux]
        raise ValueError(f"Registered container aux data {type(aux).__name__} is unsafe.")

    _descend(value, ())
    return {"nodes": nodes, "refusals": refusals}
