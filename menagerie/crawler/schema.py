"""JSON Schema loading and strict payload validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, MutableSet, Union

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry, Resource

from menagerie.crawler.constants import (
    ARTIFACT_EVENT_SCHEMA_VERSION,
    ATTEMPT_SCHEMA_VERSION,
    ATTEMPT_SCHEMA_VERSION_V3,
    AUTHOR_PROPOSAL_SCHEMA_VERSION,
    AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
    AUTHOR_RESULT_SCHEMA_VERSION,
    AUTHOR_RESULT_SCHEMA_VERSION_V3,
    GATE_SCHEMA_VERSION,
    GATE_SCHEMA_VERSION_V3,
    LEGACY_UNTRUSTED_SCHEMA_VERSIONS,
    MODEL_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION_V3,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
)

SCHEMA_DIRECTORY = Path(__file__).with_name("schemas")
SCHEMA_FILES = {
    MODEL_SCHEMA_VERSION: "model-v2.schema.json",
    ATTEMPT_SCHEMA_VERSION: "attempt-v2.schema.json",
    GATE_SCHEMA_VERSION: "gate-v2.schema.json",
    AUTHOR_PROPOSAL_SCHEMA_VERSION: "author-proposal-v2.schema.json",
    MODEL_SCHEMA_VERSION_V3: "model-v3.schema.json",
    ATTEMPT_SCHEMA_VERSION_V3: "attempt-v3.schema.json",
    GATE_SCHEMA_VERSION_V3: "gate-v3.schema.json",
    AUTHOR_PROPOSAL_SCHEMA_VERSION_V3: "author-proposal-v3.schema.json",
    AUTHOR_RESULT_SCHEMA_VERSION_V3: "author-result-v3.schema.json",
    AUTHOR_RESULT_SCHEMA_VERSION: "author-result-v4.schema.json",
    ARTIFACT_EVENT_SCHEMA_VERSION: "artifact-event-v1.schema.json",
    OPERATIONAL_EVENT_SCHEMA_VERSION: "operational-event-v1.schema.json",
}

SCHEMA_RESOURCE_FILES = ("model-common.schema.json",)

OWNERSHIP_ANNOTATED_SCHEMA_VERSIONS = frozenset(
    {
        MODEL_SCHEMA_VERSION_V3,
        ATTEMPT_SCHEMA_VERSION_V3,
        GATE_SCHEMA_VERSION_V3,
        AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
        AUTHOR_RESULT_SCHEMA_VERSION,
    }
)


class PayloadValidationError(ValueError):
    """Raised when a payload violates its executable schema."""


class SchemaOwnershipError(ValueError):
    """Raised when a v3 schema leaf has no single declared authority owner."""


class SchemaOwner(str, Enum):
    """Closed authority owners for normalized crawler schema leaves."""

    AUTHOR_GATED = "author-gated"
    WORKER_OBSERVED = "worker-observed"
    PARENT_OBSERVED = "parent-observed"
    REDUCER_DERIVED = "reducer-derived"
    TRUSTED_INTAKE = "trusted-intake"
    UNTRUSTED_HISTORY = "untrusted-history"


@dataclass(frozen=True)
class OwnedSchemaLeaf:
    """One normalized schema leaf and its sole authority owner.

    Parameters
    ----------
    path:
        JSON path with every collection item normalized to ``[]``.
    owner:
        Exact closed authority owner declared by schema metadata.
    """

    path: str
    owner: SchemaOwner


@lru_cache(maxsize=None)
def load_schema(schema_version: str) -> dict[str, Any]:
    """Load and cache one supported crawler schema.

    Parameters
    ----------
    schema_version:
        Exact schema-version constant embedded in a payload.

    Returns
    -------
    dict[str, Any]
        Parsed JSON Schema.

    Raises
    ------
    KeyError
        If the version is not supported.
    SchemaError
        If the bundled schema is invalid.
    """

    try:
        path = SCHEMA_DIRECTORY / SCHEMA_FILES[schema_version]
    except KeyError as exc:
        raise KeyError(f"unsupported crawler schema version: {schema_version!r}") from exc
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise SchemaError(f"schema root must be an object: {path}")
    Draft202012Validator.check_schema(loaded)
    return loaded


@lru_cache(maxsize=None)
def _load_schema_resource(filename: str) -> dict[str, Any]:
    """Load and cache one version-neutral bundled schema resource.

    Parameters
    ----------
    filename:
        Exact filename listed in ``SCHEMA_RESOURCE_FILES``.

    Returns
    -------
    dict[str, Any]
        Parsed JSON Schema resource.

    Raises
    ------
    KeyError
        If the filename is not a registered shared resource.
    SchemaError
        If the bundled resource is invalid.
    """

    if filename not in SCHEMA_RESOURCE_FILES:
        raise KeyError(f"unsupported crawler schema resource: {filename!r}")
    path = SCHEMA_DIRECTORY / filename
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise SchemaError(f"schema resource root must be an object: {path}")
    Draft202012Validator.check_schema(loaded)
    return loaded


@lru_cache(maxsize=None)
def get_validator(schema_version: str) -> Draft202012Validator:
    """Return a cached strict validator for a crawler schema.

    Parameters
    ----------
    schema_version:
        Exact supported schema version.

    Returns
    -------
    Draft202012Validator
        Validator configured with RFC date/time format checking.
    """

    resources: list[tuple[str, Resource[Any]]] = []
    for supported_version in SCHEMA_FILES:
        schema = load_schema(supported_version)
        resources.append((schema["$id"], Resource.from_contents(schema)))
    for filename in SCHEMA_RESOURCE_FILES:
        schema = _load_schema_resource(filename)
        resources.append((schema["$id"], Resource.from_contents(schema)))
    registry: Registry[Any] = Registry[Any]().with_resources(resources)
    return Draft202012Validator(
        load_schema(schema_version), format_checker=FormatChecker(), registry=registry
    )


def validate_payload(payload: Mapping[str, Any], schema_version: Union[str, None] = None) -> None:
    """Validate a payload and reject every schema or format violation.

    Parameters
    ----------
    payload:
        JSON-like mapping to validate.
    schema_version:
        Expected schema version. When omitted, the payload value is used.

    Raises
    ------
    PayloadValidationError
        If the schema version is absent/mismatched or validation fails.
    """

    actual = payload.get("schema_version")
    if not isinstance(actual, str):
        raise PayloadValidationError("schema_version must be present and be a string")
    expected = schema_version or actual
    if actual != expected:
        raise PayloadValidationError(
            f"schema_version mismatch: expected {expected!r}, received {actual!r}"
        )
    try:
        get_validator(expected).validate(dict(payload))
    except (KeyError, ValidationError) as exc:
        if isinstance(exc, ValidationError):
            location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
            message = f"{expected} validation failed at {location}: {exc.message}"
        else:
            message = str(exc)
        raise PayloadValidationError(message) from exc


def is_valid_payload(payload: Mapping[str, Any], schema_version: Union[str, None] = None) -> bool:
    """Return whether a payload passes strict schema validation.

    Parameters
    ----------
    payload:
        JSON-like mapping to validate.
    schema_version:
        Optional expected schema version.

    Returns
    -------
    bool
        True only for a fully valid payload.
    """

    try:
        validate_payload(payload, schema_version)
    except PayloadValidationError:
        return False
    return True


def is_legacy_untrusted(payload: Mapping[str, Any]) -> bool:
    """Return whether a historical row is categorically unusable as v3 authority.

    Parameters
    ----------
    payload:
        Persisted crawler row.

    Returns
    -------
    bool
        True for immutable v2 attempt/model/gate/proposal history. No adapter may
        synthesize missing v3 proof material for such a row.
    """

    return payload.get("schema_version") in LEGACY_UNTRUSTED_SCHEMA_VERSIONS


def schema_leaf_paths(schema: Mapping[str, Any]) -> frozenset[str]:
    """Return every allowed normalized leaf path reachable from a schema root.

    Parameters
    ----------
    schema:
        Loaded crawler JSON Schema, including optional external references.

    Returns
    -------
    frozenset[str]
        Exhaustive normalized leaf paths. Collection elements use ``[]``.
    """

    leaves: set[str] = set()
    _collect_schema_leaves(schema, schema, "$", leaves, set())
    return frozenset(leaves)


def _collect_schema_leaves(
    node: object,
    root: Mapping[str, Any],
    path: str,
    leaves: MutableSet[str],
    reference_stack: set[str],
) -> bool:
    """Recursively collect reachable leaf paths from one schema node.

    Parameters
    ----------
    node:
        Current JSON Schema node.
    root:
        Root schema used to resolve local references.
    path:
        Current normalized instance path.
    leaves:
        Mutable leaf accumulator.
    reference_stack:
        Active references used to stop malformed recursive cycles.

    Returns
    -------
    bool
        True when the node exposes at least one nested instance leaf.
    """

    if not isinstance(node, Mapping):
        return False
    nested = False
    reference = node.get("$ref")
    if isinstance(reference, str):
        if reference in reference_stack:
            return False
        resolved_root, resolved = _resolve_schema_reference(root, reference)
        nested |= _collect_schema_leaves(
            resolved,
            resolved_root,
            path,
            leaves,
            reference_stack | {reference},
        )
    properties = node.get("properties")
    if isinstance(properties, Mapping):
        for name, child in properties.items():
            child_path = f"{path}.{name}"
            child_nested = _collect_schema_leaves(child, root, child_path, leaves, reference_stack)
            if not child_nested:
                leaves.add(child_path)
            nested = True
    items = node.get("items")
    if isinstance(items, Mapping):
        item_path = f"{path}[]"
        item_nested = _collect_schema_leaves(items, root, item_path, leaves, reference_stack)
        if not item_nested:
            leaves.add(item_path)
        nested = True
    for keyword in ("oneOf", "anyOf", "allOf"):
        branches = node.get(keyword)
        if isinstance(branches, list):
            for branch in branches:
                nested |= _collect_schema_leaves(branch, root, path, leaves, reference_stack)
    return nested


def _resolve_schema_reference(
    root: Mapping[str, Any], reference: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Resolve one local or bundled crawler-schema reference.

    Parameters
    ----------
    root:
        Schema containing the reference.
    reference:
        Local fragment or bundled schema URI.

    Returns
    -------
    tuple[Mapping[str, Any], Mapping[str, Any]]
        Referenced schema root and target node.

    Raises
    ------
    SchemaOwnershipError
        If the reference is unsupported or cannot be resolved.
    """

    if reference.startswith("#"):
        target_root = root
        fragment = reference[1:]
    else:
        uri, separator, fragment = reference.partition("#")
        schema_version = next(
            (
                version
                for version in SCHEMA_FILES
                if load_schema(version).get("$id") == uri
                or SCHEMA_FILES[version] == uri
                or str(load_schema(version).get("$id", "")).endswith(f"/{uri}")
            ),
            None,
        )
        if schema_version is not None:
            target_root = load_schema(schema_version)
        else:
            resource_filename = next(
                (
                    filename
                    for filename in SCHEMA_RESOURCE_FILES
                    if _load_schema_resource(filename).get("$id") == uri
                    or filename == uri
                    or str(_load_schema_resource(filename).get("$id", "")).endswith(f"/{uri}")
                ),
                None,
            )
            if resource_filename is None:
                raise SchemaOwnershipError(f"unsupported schema reference: {reference!r}")
            target_root = _load_schema_resource(resource_filename)
        fragment = fragment if separator else ""
    target: object = target_root
    if fragment:
        if not fragment.startswith("/"):
            raise SchemaOwnershipError(f"unsupported schema fragment: {reference!r}")
        for raw_part in fragment.removeprefix("/").split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, Mapping) or part not in target:
                raise SchemaOwnershipError(f"unresolved schema reference: {reference!r}")
            target = target[part]
    if not isinstance(target, Mapping):
        raise SchemaOwnershipError(f"schema reference is not an object: {reference!r}")
    return target_root, target


def owned_schema_leaves(schema_version: str) -> tuple[OwnedSchemaLeaf, ...]:
    """Validate and return the exhaustive schema-derived ownership registry.

    Parameters
    ----------
    schema_version:
        Ownership-annotated v3 schema discriminator.

    Returns
    -------
    tuple[OwnedSchemaLeaf, ...]
        Stable path-sorted leaf registry.

    Raises
    ------
    SchemaOwnershipError
        If any leaf is missing, extraneous, or has an invalid/multiple owner.
    """

    if schema_version not in OWNERSHIP_ANNOTATED_SCHEMA_VERSIONS:
        raise SchemaOwnershipError(f"schema has no ownership contract: {schema_version!r}")
    return owned_schema_leaves_from_schema(load_schema(schema_version))


def owned_schema_leaves_from_schema(
    schema: Mapping[str, Any],
) -> tuple[OwnedSchemaLeaf, ...]:
    """Validate ownership metadata against an arbitrary schema mapping.

    Parameters
    ----------
    schema:
        Schema carrying an ``x-authority-ownership`` exact-path mapping.

    Returns
    -------
    tuple[OwnedSchemaLeaf, ...]
        Stable path-sorted ownership registry.

    Raises
    ------
    SchemaOwnershipError
        If coverage is not exact or an owner is outside the closed vocabulary.
    """

    declared = schema.get("x-authority-ownership")
    if not isinstance(declared, Mapping):
        raise SchemaOwnershipError("schema must declare x-authority-ownership")
    inherited_version = schema.get("x-authority-ownership-base")
    if inherited_version is not None:
        if not isinstance(inherited_version, str):
            raise SchemaOwnershipError("schema ownership base must be a schema version")
        try:
            inherited = load_schema(inherited_version).get("x-authority-ownership")
        except KeyError as exc:
            raise SchemaOwnershipError("schema ownership base is unsupported") from exc
        if not isinstance(inherited, Mapping):
            raise SchemaOwnershipError("schema ownership base has no ownership mapping")
        declared = {**inherited, **declared}
    prefix_copies = schema.get("x-authority-ownership-prefix-copies", {})
    if not isinstance(prefix_copies, Mapping) or any(
        not isinstance(source, str) or not isinstance(target, str)
        for source, target in prefix_copies.items()
    ):
        raise SchemaOwnershipError("schema ownership prefix copies must map string prefixes")
    copied: dict[str, object] = {}
    for source, target in prefix_copies.items():
        copied.update(
            {
                f"{target}{path.removeprefix(source)}": owner
                for path, owner in declared.items()
                if str(path).startswith(source)
            }
        )
    declared = {**declared, **copied}
    leaves = schema_leaf_paths(schema)
    declared_paths = frozenset(str(path) for path in declared)
    missing = sorted(leaves - declared_paths)
    extraneous = sorted(declared_paths - leaves)
    if missing or extraneous:
        raise SchemaOwnershipError(
            f"schema ownership coverage mismatch: missing={missing}, extraneous={extraneous}"
        )
    owned: list[OwnedSchemaLeaf] = []
    for path in sorted(leaves):
        raw_owner = declared[path]
        if not isinstance(raw_owner, str):
            raise SchemaOwnershipError(f"schema leaf {path} must have exactly one string owner")
        try:
            owner = SchemaOwner(raw_owner)
        except ValueError as exc:
            raise SchemaOwnershipError(
                f"schema leaf {path} has unknown authority owner {raw_owner!r}"
            ) from exc
        owned.append(OwnedSchemaLeaf(path=path, owner=owner))
    return tuple(owned)


def schema_owner_paths(schema_version: str, owner: SchemaOwner) -> frozenset[str]:
    """Return the schema-derived leaf policy for one authority owner.

    Parameters
    ----------
    schema_version:
        Ownership-annotated v3 schema discriminator.
    owner:
        Closed owner whose exact normalized leaves are requested.

    Returns
    -------
    frozenset[str]
        Exhaustive schema leaf paths assigned to ``owner``.
    """

    return frozenset(
        leaf.path for leaf in owned_schema_leaves(schema_version) if leaf.owner is owner
    )


def author_gated_schema_paths(schema_version: str) -> frozenset[str]:
    """Return the exhaustive schema-derived authored-leaf policy.

    Parameters
    ----------
    schema_version:
        Ownership-annotated author, model, attempt, or gate v3 schema.

    Returns
    -------
    frozenset[str]
        Exact normalized leaves that require independent author gating.
    """

    return schema_owner_paths(schema_version, SchemaOwner.AUTHOR_GATED)
