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
from typing import Any, Callable

__all__ = [
    "INPUT_CONTAINER_KINDS",
    "classify_input_container",
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
            for name in value._fields:
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


def encode_mapping_key(key: Any) -> "str | int":
    """Encode one grammar mapping key into the canonical type-strict component (r67 C2).

    ``str`` keys pass through unless they collide with the codec prefix (then escaped);
    ``int`` keys pass through as ints; ``bool`` / ``float`` / ``None`` / recursively safe
    tuples encode into tagged strings (floats via ``float.hex`` -- exact round-trip,
    ``1.0`` distinct from ``1``). A non-grammar key raises ``ValueError`` (the caller
    refuses or routes the subtree opaque -- never a silent drop).
    """

    if isinstance(key, bool):
        return f"{_KEY_CODEC_TAG}b:{'1' if key else '0'}"
    if isinstance(key, int):
        return key
    if isinstance(key, str):
        if key.startswith(_KEY_CODEC_TAG):
            return f"{_KEY_CODEC_TAG}s:{key[len(_KEY_CODEC_TAG) :]}"
        return key
    if key is None:
        return f"{_KEY_CODEC_TAG}n:"
    if isinstance(key, float):
        return f"{_KEY_CODEC_TAG}f:{key.hex()}"
    if isinstance(key, tuple):
        parts = []
        for item in key:
            encoded = encode_mapping_key(item)
            text = f"i:{encoded}" if isinstance(encoded, int) else f"c:{encoded}"
            parts.append(text.replace("\\", "\\\\").replace("|", "\\|"))
        return f"{_KEY_CODEC_TAG}t:{'|'.join(parts)}"
    raise ValueError(f"Mapping key {key!r} is outside the input-boundary key grammar.")


def decode_mapping_key(component: "str | int") -> Any:
    """Decode one canonical component back to its mapping key (mapping nodes only)."""

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


def undeclared_instance_state(value: Any, kind: str) -> bool:
    """Return whether a container INSTANCE carries state its declared schema drops (r67 C2).

    A dataclass with a non-field ``__dict__`` attribute, a namedtuple subclass with
    instance ``__dict__`` state, or a dict/list/tuple subclass carrying ``__dict__``
    attributes would silently lose that state on replay reconstruction -- and it can
    steer control flow the declared schema never witnesses (r66 R1: ``box.mode``).
    Registered containers prove completeness through their explicit ``state_complete``
    declaration instead (checked by the snapshot's registered branch).
    """

    import dataclasses as _dc

    if kind == "dataclass" or (kind == "empty" and _dc.is_dataclass(value)):
        field_names = {field.name for field in _dc.fields(value)}
        extra = {
            name
            for name, attr_value in getattr(value, "__dict__", {}).items()
            if name not in field_names and attr_value is not None
        }
        return bool(extra)
    if kind in {"namedtuple", "mapping", "sequence", "empty"}:
        extras = getattr(value, "__dict__", None)
        if not isinstance(extras, dict) or not extras:
            return False
        return any(item is not None for item in extras.values())
    return False


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
        node: dict[str, Any] = {"path": list(path), "kind": kind, "type": _type_ref(item)}
        if kind == "tensor":
            nodes.append(node)
            return
        registration = None
        if kind not in {"tensor", "leaf"}:
            registration = get_registered_container(type(item))
        if registration is not None:
            node["kind"] = "registered"
            if not registration.state_complete and getattr(item, "__dict__", None):
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
        if undeclared_instance_state(item, kind):
            refusals.append({"path": list(path), "reason": "undeclared_instance_state"})
        if kind == "empty":
            empty_kind = empty_container_kind(item)
            node["empty_kind"] = str(empty_kind)
            nodes.append(node)
            return
        if kind == "namedtuple":
            node["fields"] = [str(name) for name in item._fields]
            nodes.append(node)
            for name in item._fields:
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
        nodes.append(node)  # literal leaf: value witnessed by the literal walker

    def _safe_aux(aux: Any) -> Any:
        if aux is None or isinstance(aux, (bool, int, float, str)):
            return aux
        if isinstance(aux, (list, tuple)):
            return [_safe_aux(item) for item in aux]
        raise ValueError(f"Registered container aux data {type(aux).__name__} is unsafe.")

    _descend(value, ())
    return {"nodes": nodes, "refusals": refusals}
