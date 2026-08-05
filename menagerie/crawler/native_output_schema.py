"""Derive the checker's native structured-output schema from ``gate.v3`` itself.

The checker used to be handed a one-field ``result_json`` STRING transport as its
``--output-schema``. That made the provider's native structured-output constraint
enforce nothing about the gate object: the real content was an opaque string, so
the only channel carrying the gate's shape was prose in the frozen prompt. Prose
is always an incomplete restatement of a machine-readable schema that already
exists, and the observed failure mode was exactly that -- the checker invented
schema-shaped vocabulary (``rung_check.required``, ``integrity.findings``,
``terminal_disposition.arm``/``result_sha256``) for whichever closed block the
prompt had not yet enumerated, one nested block per rung.

This module removes the prose from that path. It LOWERS the repository's own
``menagerie.crawler.gate.v3`` item definition into the provider's strict
structured-output subset and hands THAT to the model, so an invented key is not
refused after the fact -- it is unrepresentable.

The lowering is deliberately lossy in one direction only. The one constraint the
strict subset cannot express -- the conditional correlating
``verified_hashes.code`` with ``code_manifest`` -- is DROPPED FROM THE NATIVE
SCHEMA ONLY and is still enforced, unchanged, by ``gate.v3`` after decoding. The
``if``/``then`` conditionals correlating ``gate_kind`` with ``fidelity`` and
``terminal_disposition`` need no conditional at all here, because their input is
machine-known at request time: the schema is derived PER GATE KIND and
``_narrow_for_gate_kind`` bakes each correlation in directly (rung 8: a checker
wrote a real ``fidelity.verdict`` on a metadata gate, ``gate.v3`` refused the
complete verdict after the fact, and both batch members died on the one slip).
The native schema is a strictly ADDITIONAL upstream constraint; it never
becomes the authority and it never widens the gate.

Constructs the subset cannot express are REFUSED, not silently ignored. Each one
that is deliberately deferred to ``gate.v3`` is named by exact JSON pointer in
``DEFERRED_APPLICATORS`` with its reason, so a NEW conditional appearing anywhere
in the gate schema raises at derivation time instead of quietly vanishing.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Mapping, Sequence, Union

from menagerie.crawler.constants import GATE_SCHEMA_VERSION_V3, GateKind
from menagerie.crawler.schema import load_schema, load_schema_resource

JsonObject = dict[str, Any]

#: ``$id`` of the version-neutral definitions document that ``gate-v3`` refs.
GATE_COMMON_ID = "https://torchlens.org/schemas/menagerie/crawler/gate-common.schema.json"

#: Filename of that document within the shipped schema resources.
GATE_COMMON_FILE = "gate-common.schema.json"

#: Sole property the checker is authorized to author. Everything else in a gate
#: is machine-owned scaffold stamped by the wrapper, so listing it here would be
#: actively harmful: the strict subset requires every declared property to be
#: PRESENT, which would turn "omitting the scaffold is always correct" into
#: "supplying a fabricated scaffold is mandatory".
AUTHORED_PROPERTY = "items"

#: Bookkeeping keywords that mean nothing to a constrained decoder. These are the
#: ONLY assertions dropped: the value constraints the subset does support
#: (``pattern``, ``minLength``, ``format``, item counts, numeric bounds) are kept,
#: so the decoder itself refuses a malformed digest rather than deferring it.
_DROPPED_ANNOTATIONS = frozenset({"$comment", "$id", "$schema", "default", "examples", "title"})

#: Applicators the strict subset cannot express. Reaching one that is not
#: allowlisted below is a derivation failure, never a silent drop.
_UNSUPPORTED_APPLICATORS = frozenset(
    {
        "allOf",
        "contains",
        "dependentRequired",
        "dependentSchemas",
        "else",
        "if",
        "not",
        "patternProperties",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)

#: Exact JSON pointers, within the resolved item definition, at which an
#: unsupported applicator is knowingly deferred to ``gate.v3``. The pointer is
#: recorded rather than the keyword alone so that the same keyword appearing at
#: a DIFFERENT site still raises.
DEFERRED_APPLICATORS: Mapping[str, str] = {
    "/properties/verified_hashes/allOf": (
        "gate.v3 makes verified_hashes.code_manifest required when code is a digest and "
        "forbidden when it is null. The strict subset has no conditionals, so the presence "
        "choice is lowered to a closed two-arm anyOf and the correlation with `code` stays "
        "with gate.v3."
    ),
}

#: Upper bound on optional properties in one object. The faithful lowering of an
#: optional property is an arm per presence combination, which is exponential; a
#: bound keeps a future schema edit from silently generating a huge schema.
_MAX_OPTIONAL_PROPERTIES = 1


class NativeOutputSchemaError(RuntimeError):
    """One gate construct that the strict structured-output subset cannot express."""


def _resolve_documents() -> tuple[JsonObject, JsonObject]:
    """Load the gate schema and the version-neutral definitions it references.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        The ``gate.v3`` document and the ``gate-common`` document.
    """

    return load_schema(GATE_SCHEMA_VERSION_V3), load_schema_resource(GATE_COMMON_FILE)


def _lookup(pointer: str, gate: Mapping[str, Any], common: Mapping[str, Any]) -> Any:
    """Resolve one ``$ref`` against the gate or the shared definitions document.

    Parameters
    ----------
    pointer:
        Exact ``$ref`` value, local (``#/$defs/x``) or absolute into gate-common.
    gate, common:
        Loaded documents.

    Returns
    -------
    Any
        Referenced subschema.

    Raises
    ------
    NativeOutputSchemaError
        If the reference names a document or definition that is not shipped.
    """

    if pointer.startswith(GATE_COMMON_ID):
        document: Mapping[str, Any] = common
        fragment = pointer[len(GATE_COMMON_ID) :]
    elif pointer.startswith("#"):
        document = gate
        fragment = pointer
    else:
        raise NativeOutputSchemaError(f"unresolvable gate schema reference: {pointer!r}")
    node: Any = document
    for token in fragment.lstrip("#").strip("/").split("/"):
        if not isinstance(node, Mapping) or token not in node:
            raise NativeOutputSchemaError(f"unresolvable gate schema reference: {pointer!r}")
        node = node[token]
    return node


def _inline(node: Any, gate: Mapping[str, Any], common: Mapping[str, Any], seen: tuple[str, ...]) -> Any:
    """Expand every reference so no ``$ref`` keeps sibling keywords.

    The strict subset rejects a ``$ref`` that carries any sibling keyword, and
    every gate property pairs its ``$ref`` with the ``description`` that tells the
    checker what the field means. Inlining keeps those descriptions -- they are
    the part of the prompt's shape prose worth preserving, and they now travel
    with the constraint instead of beside it.

    Parameters
    ----------
    node:
        Subschema under expansion.
    gate, common:
        Loaded documents.
    seen:
        Reference chain guarding against a cyclic definition.

    Returns
    -------
    Any
        Reference-free subschema.

    Raises
    ------
    NativeOutputSchemaError
        If the definitions are cyclic and cannot be inlined.
    """

    if isinstance(node, Mapping):
        reference = node.get("$ref")
        if isinstance(reference, str):
            if reference in seen:
                raise NativeOutputSchemaError(
                    f"cyclic gate schema reference cannot be inlined: {reference!r}"
                )
            expanded = _inline(
                _lookup(reference, gate, common), gate, common, seen + (reference,)
            )
            if not isinstance(expanded, dict):
                raise NativeOutputSchemaError(f"gate schema reference is not an object: {reference!r}")
            merged = dict(expanded)
            for key, value in node.items():
                if key != "$ref":
                    merged[key] = _inline(value, gate, common, seen)
            return merged
        return {key: _inline(value, gate, common, seen) for key, value in node.items()}
    if isinstance(node, list):
        return [_inline(value, gate, common, seen) for value in node]
    return node


def _lower(node: Any, pointer: str) -> Any:
    """Lower one reference-free subschema into the strict structured-output subset.

    Parameters
    ----------
    node:
        Reference-free subschema.
    pointer:
        JSON pointer of this node within the resolved item definition, used to
        name a refused construct and to match ``DEFERRED_APPLICATORS``.

    Returns
    -------
    Any
        Subschema expressed in the strict subset.

    Raises
    ------
    NativeOutputSchemaError
        If the node uses an applicator the subset cannot express and that is not
        explicitly deferred, or if it has more optional properties than the
        lowering will enumerate.
    """

    if isinstance(node, list):
        return [_lower(value, f"{pointer}/{index}") for index, value in enumerate(node)]
    if not isinstance(node, Mapping):
        return node

    lowered: JsonObject = {}
    for key, value in node.items():
        child = f"{pointer}/{key}"
        if key in _DROPPED_ANNOTATIONS:
            continue
        if key in _UNSUPPORTED_APPLICATORS:
            if child in DEFERRED_APPLICATORS:
                continue
            raise NativeOutputSchemaError(
                f"gate schema uses {key!r} at {child}, which the strict structured-output "
                "subset cannot express. Add an exact pointer entry to DEFERRED_APPLICATORS "
                "with the reason it is safe to leave to gate.v3, or express it in the subset."
            )
        if key == "oneOf":
            # ``anyOf`` is the subset's only union. It is strictly weaker than
            # ``oneOf`` (it does not demand exactly one match), and every gate
            # ``oneOf`` here is a disjoint nullable union, so no candidate the
            # subset admits is one gate.v3 would reject on that account.
            lowered["anyOf"] = _lower(value, child)
            continue
        if key == "const":
            lowered["enum"] = [value]
            continue
        lowered[key] = _lower(value, child)

    if lowered.get("type") != "object":
        return lowered
    return _close_object(lowered, pointer)


def _close_object(node: JsonObject, pointer: str) -> JsonObject:
    """Close one object and make every declared property required.

    The strict subset admits no optional property, so a property the gate schema
    declares but does not require is lowered into a closed arm per presence
    choice. That keeps the native schema from MANUFACTURING a key gate.v3
    forbids, which is the failure this whole module exists to prevent.

    Parameters
    ----------
    node:
        Lowered object subschema.
    pointer:
        JSON pointer used to name a refusal.

    Returns
    -------
    dict[str, Any]
        Closed object, or an ``anyOf`` over closed presence arms.

    Raises
    ------
    NativeOutputSchemaError
        If the object has more optional properties than will be enumerated.
    """

    properties: Mapping[str, Any] = node.get("properties") or {}
    required = list(node.get("required") or ())
    optional = sorted(set(properties) - set(required))
    if len(optional) > _MAX_OPTIONAL_PROPERTIES:
        raise NativeOutputSchemaError(
            f"gate schema object at {pointer} has {len(optional)} optional properties "
            f"({', '.join(optional)}); the strict subset has no optional property and the "
            f"lowering enumerates at most {_MAX_OPTIONAL_PROPERTIES}."
        )
    if not optional:
        closed = dict(node)
        closed["additionalProperties"] = False
        closed["required"] = sorted(properties)
        return closed
    arms = []
    for present in (True, False):
        arm_properties = {
            name: value
            for name, value in properties.items()
            if present or name not in optional
        }
        arm = dict(node)
        arm["properties"] = arm_properties
        arm["additionalProperties"] = False
        arm["required"] = sorted(arm_properties)
        arms.append(arm)
    return {"anyOf": arms, **{k: v for k, v in node.items() if k == "description"}}


def _narrow_for_gate_kind(item: JsonObject, gate_kind: GateKind) -> JsonObject:
    """Apply the machine-known gate-kind facts to one lowered item schema.

    ``gate.v3`` correlates ``gate_kind`` with ``fidelity`` and
    ``terminal_disposition`` through top-level conditionals the strict subset
    cannot express, so the kind-neutral lowering used to admit every value those
    conditionals forbid. The correlation's inputs are MACHINE-KNOWN at envelope
    build time -- the wrapper knows which gate kind it is running -- so the
    narrowing is applied here, per kind, and the forbidden values become
    unrepresentable instead of merely refused after a complete verdict was
    already produced. That refusal killed both members of the rung-8 batch
    ``metadata-c001cad863d95a67`` (m10517 + m9666): the checker wrote a real
    ``fidelity.verdict`` on a metadata gate and the whole batch exited as a
    permanent contract rejection.

    Per kind:

    - ``metadata_batch``: ``fidelity`` is MACHINE-OWNED. It is removed from the
      item schema entirely -- exactly like the gate scaffold, supplying it is not
      merely unnecessary, it is impossible -- and the wrapper stamps its typed
      not-applicable form. ``terminal_disposition`` is pinned to ``null``.
    - ``fidelity``: the block stays the checker's judgment, but ``required`` is
      pinned ``true`` (a machine fact of the lane, which the semantic validator
      already enforces) and ``terminal_disposition`` is pinned to ``null``.
    - ``terminal_disposition``: the terminal block is pinned to its closed object
      arm, matching gate.v3's ``then`` branch exactly.

    Parameters
    ----------
    item:
        Kind-neutral lowered item schema.
    gate_kind:
        Exact gate kind this schema will constrain.

    Returns
    -------
    dict[str, Any]
        Narrowed item schema. Every narrowing is toward ``gate.v3``'s own
        conditionals or the semantic validator; nothing the full validator would
        accept for this kind is refused.

    Raises
    ------
    NativeOutputSchemaError
        If the lowered item no longer has the shape this narrowing expects,
        so gate-schema drift raises at derivation time instead of silently
        producing a wrong constraint.
    """

    properties = item.get("properties")
    required = item.get("required")
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        raise NativeOutputSchemaError("lowered gate item is not a closed object")
    narrowed_properties = dict(properties)
    narrowed_required = list(required)

    terminal = narrowed_properties.get("terminal_disposition")
    if not isinstance(terminal, Mapping):
        raise NativeOutputSchemaError("lowered gate item lacks its terminal_disposition union")
    arms = terminal.get("anyOf")
    if not isinstance(arms, list) or len(arms) != 2:
        raise NativeOutputSchemaError(
            "lowered terminal_disposition is not the expected two-arm nullable union"
        )
    null_arms = [arm for arm in arms if isinstance(arm, Mapping) and arm.get("type") == "null"]
    object_arms = [arm for arm in arms if isinstance(arm, Mapping) and arm.get("type") == "object"]
    if len(null_arms) != 1 or len(object_arms) != 1:
        raise NativeOutputSchemaError(
            "lowered terminal_disposition union does not split into object and null arms"
        )
    description = {"description": terminal["description"]} if "description" in terminal else {}
    if gate_kind is GateKind.TERMINAL_DISPOSITION:
        narrowed_properties["terminal_disposition"] = {**object_arms[0], **description}
    else:
        narrowed_properties["terminal_disposition"] = {**null_arms[0], **description}

    if gate_kind is GateKind.METADATA_BATCH:
        if "fidelity" not in narrowed_properties:
            raise NativeOutputSchemaError("lowered gate item lacks its fidelity block")
        del narrowed_properties["fidelity"]
        narrowed_required = [name for name in narrowed_required if name != "fidelity"]
    elif gate_kind is GateKind.FIDELITY:
        fidelity = narrowed_properties.get("fidelity")
        if not isinstance(fidelity, Mapping):
            raise NativeOutputSchemaError("lowered gate item lacks its fidelity block")
        fidelity_properties = fidelity.get("properties")
        if not isinstance(fidelity_properties, Mapping):
            raise NativeOutputSchemaError("lowered fidelity block is not a closed object")
        required_property = fidelity_properties.get("required")
        if not isinstance(required_property, Mapping):
            raise NativeOutputSchemaError("lowered fidelity block lacks its required property")
        narrowed_properties["fidelity"] = {
            **fidelity,
            "properties": {
                **fidelity_properties,
                "required": {**required_property, "enum": [True]},
            },
        }

    return {**item, "properties": narrowed_properties, "required": narrowed_required}


@lru_cache(maxsize=None)
def _derived(gate_kind: GateKind) -> str:
    """Return the derived native schema as canonical JSON text for caching.

    Parameters
    ----------
    gate_kind:
        Exact gate kind the schema constrains.

    Returns
    -------
    str
        Serialized native structured-output schema.
    """

    gate, common = _resolve_documents()
    item = _inline({"$ref": "#/$defs/item"}, gate, common, ())
    lowered = _lower(item, "")
    if not isinstance(lowered, dict):
        raise NativeOutputSchemaError("lowered gate item is not an object schema")
    narrowed = _narrow_for_gate_kind(lowered, gate_kind)
    items_description = str(
        ((gate.get("properties") or {}).get(AUTHORED_PROPERTY) or {}).get("description")
        or "Gate verdict items."
    )
    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": [AUTHORED_PROPERTY],
        "properties": {
            AUTHORED_PROPERTY: {
                "type": "array",
                "description": (
                    f"{items_description} Exactly one item per envelope item, in envelope "
                    "order. Every other gate field is machine-owned and is stamped by the "
                    "wrapper."
                ),
                "items": narrowed,
            }
        },
    }
    return json.dumps(schema, sort_keys=True)


def native_output_schema(gate_kind: GateKind) -> JsonObject:
    """Return the checker's native structured-output schema for one gate kind.

    Parameters
    ----------
    gate_kind:
        Exact gate kind the request envelope declares. The kind is machine-known,
        so the per-kind conditionals ``gate.v3`` states at the top level are
        narrowed into the item schema itself instead of being deferred to a
        post-hoc refusal that discards a complete verdict.

    Returns
    -------
    dict[str, Any]
        Strict-subset schema constraining the checker to the exact
        ``menagerie.crawler.gate.v3`` item vocabulary for this gate kind.
        Machine-owned gate scaffold -- including the whole ``fidelity`` block on
        a ``metadata_batch`` item -- is absent by construction, so the model can
        neither invent a key nor be forced to fabricate a value it has no
        authority over.

    Raises
    ------
    NativeOutputSchemaError
        If the gate schema has grown a construct the strict subset cannot express.
    """

    loaded = json.loads(_derived(GateKind(gate_kind)))
    assert isinstance(loaded, dict)
    return loaded


def closed_key_sets(gate_kind: GateKind) -> dict[str, tuple[str, ...]]:
    """Return every closed object key set the native schema pins, by JSON pointer.

    This is the machine-readable answer to "which nested blocks are exposed to an
    invented key". It is derived from the same lowering the model is handed, so a
    test can assert the two agree with ``gate.v3`` instead of a prose list going
    stale one rung at a time.

    Parameters
    ----------
    gate_kind:
        Exact gate kind whose derived schema is walked.

    Returns
    -------
    dict[str, tuple[str, ...]]
        Pointer-to-sorted-key-tuple mapping for every closed object.
    """

    found: dict[str, tuple[str, ...]] = {}

    def walk(node: Any, pointer: str) -> None:
        """Record every closed object reachable from ``node``."""

        if isinstance(node, Mapping):
            if node.get("type") == "object" and node.get("additionalProperties") is False:
                found[pointer or "/"] = tuple(sorted(node.get("properties") or {}))
            for key, value in node.items():
                walk(value, f"{pointer}/{key}")
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for index, value in enumerate(node):
                walk(value, f"{pointer}/{index}")

    walk(native_output_schema(gate_kind), "")
    return found


def decode_native_result(payload: Union[Mapping[str, Any], Any]) -> JsonObject:
    """Return the checker's authored gate fragment from one native final message.

    Parameters
    ----------
    payload:
        Decoded provider final message.

    Returns
    -------
    dict[str, Any]
        Candidate gate carrying exactly the checker-authored ``items``.

    Raises
    ------
    NativeOutputSchemaError
        If the final message does not match the native transport.
    """

    if not isinstance(payload, Mapping) or set(payload) != {AUTHORED_PROPERTY}:
        raise NativeOutputSchemaError(
            "checker final message must be exactly one object whose sole key is "
            f"{AUTHORED_PROPERTY!r}"
        )
    items = payload[AUTHORED_PROPERTY]
    if not isinstance(items, list):
        raise NativeOutputSchemaError(f"checker {AUTHORED_PROPERTY} must be an array")
    return {AUTHORED_PROPERTY: list(items)}
