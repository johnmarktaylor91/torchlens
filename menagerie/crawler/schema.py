"""JSON Schema loading and strict payload validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
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

SCHEMA_RESOURCE_FILES = (
    "attempt-common.schema.json",
    "author-proposal-common.schema.json",
    "gate-common.schema.json",
    "model-common.schema.json",
)

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


class RequiredFieldProjection(str, Enum):
    """Closed mechanical required-field projections owned by Python policy."""

    MODEL_EXTERNAL_METADATA = "model-external-metadata"
    AUTHOR_EXTERNAL_METADATA = "author-external-metadata"
    AUTHOR_PROPOSAL_FACT_BLOCK = "author-proposal-fact-block"
    AUTHOR_PROPOSAL_VERIFIED_HASH = "author-proposal-verified-hash"
    GATE_ITEM_BINDING = "gate-item-binding"
    GATE_VERIFIED_HASH = "gate-verified-hash"
    MODEL_LEDGER_IDENTITY = "model-ledger-identity"
    ATTEMPT_LEDGER_IDENTITY = "attempt-ledger-identity"
    GATE_LEDGER_IDENTITY = "gate-ledger-identity"


@dataclass(frozen=True)
class RequiredField:
    """One Python-owned required field aligned with an executable schema leaf.

    Parameters
    ----------
    name:
        Consumer-facing field name within its projection.
    schema_path:
        Exact normalized leaf path in the independently executable JSON Schema.
    """

    name: str
    schema_path: str


@dataclass(frozen=True)
class RequiredFieldProjectionSpec:
    """One ordered required-field projection keyed by every authority owner.

    Parameters
    ----------
    schema_version:
        Executable schema whose leaves independently constrain the projection.
    fields_by_owner:
        Explicit, separately addressable field tuple for every ``SchemaOwner``.
    field_order:
        Original consumer validation order, retained independently of ownership
        grouping so consolidation cannot reorder checks.
    """

    schema_version: str
    fields_by_owner: Mapping[SchemaOwner, tuple[RequiredField, ...]]
    field_order: tuple[str, ...]

    def fields_for(self, owner: SchemaOwner) -> tuple[RequiredField, ...]:
        """Return the explicitly named required fields for one owner.

        Parameters
        ----------
        owner:
            Closed provenance owner whose projection is requested.

        Returns
        -------
        tuple[RequiredField, ...]
            Required fields owned by ``owner`` in their declared order.
        """

        return self.fields_by_owner[owner]

    def names_for(self, owner: SchemaOwner) -> tuple[str, ...]:
        """Return consumer-facing required names for one owner.

        Parameters
        ----------
        owner:
            Closed provenance owner whose field names are requested.

        Returns
        -------
        tuple[str, ...]
            Required field names owned by ``owner``.
        """

        return tuple(field.name for field in self.fields_for(owner))


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


_EXTERNAL_METADATA_FIELD_NAMES = (
    "modality",
    "architecture_class",
    "domain",
    "task",
    "field",
    "subfield",
    "paradigm",
    "lineage",
    "predecessors",
    "tags",
    "keywords",
    "venue",
    "family",
    "era",
    "year",
    "country",
    "authors",
    "institution",
    "citation",
    "license",
    "key_contribution",
    "description",
    "original_framework",
    "run_framework",
    "modes",
)
_AUTHOR_PROPOSAL_FACT_BLOCK_NAMES = (
    "identity",
    "taxonomy",
    "external_metadata",
    "website",
    "people_and_origin",
    "dates",
    "citation",
    "licenses",
    "source_resolution",
    "evidence",
    "implementation",
    "input_contract",
    "modes",
    "fidelity",
)
_AUTHOR_PROPOSAL_VERIFIED_HASH_NAMES = (
    "source_manifest",
    "evidence",
    "code",
    "source_to_code_map",
    "family_template",
    "code_manifest",
)
_GATE_ITEM_BINDING_FIELD_ORDER = (
    "work_id",
    "stable_id",
    "family_representative_id",
    "fidelity_identity",
    "vet_identity",
    "verified_hashes",
)


def _required_field_projection_spec(
    schema_version: str,
    *,
    author_gated: tuple[RequiredField, ...] = (),
    worker_observed: tuple[RequiredField, ...] = (),
    parent_observed: tuple[RequiredField, ...] = (),
    reducer_derived: tuple[RequiredField, ...] = (),
    trusted_intake: tuple[RequiredField, ...] = (),
    untrusted_history: tuple[RequiredField, ...] = (),
    field_order: tuple[str, ...],
) -> RequiredFieldProjectionSpec:
    """Build one immutable owner-explicit required-field projection.

    Parameters
    ----------
    schema_version:
        Independently executable schema aligned with the Python policy.
    author_gated, worker_observed, parent_observed, reducer_derived:
        Explicit required fields for the four active provenance owners.
    trusted_intake, untrusted_history:
        Explicit required fields for retained intake and categorical legacy history.
    field_order:
        Historical consumer validation order.

    Returns
    -------
    RequiredFieldProjectionSpec
        Immutable typed projection with every ``SchemaOwner`` key present.

    Raises
    ------
    RuntimeError
        If a field is duplicated or the ordered names do not exactly cover the
        owner-specific fields.
    """

    fields_by_owner = {
        SchemaOwner.AUTHOR_GATED: author_gated,
        SchemaOwner.WORKER_OBSERVED: worker_observed,
        SchemaOwner.PARENT_OBSERVED: parent_observed,
        SchemaOwner.REDUCER_DERIVED: reducer_derived,
        SchemaOwner.TRUSTED_INTAKE: trusted_intake,
        SchemaOwner.UNTRUSTED_HISTORY: untrusted_history,
    }
    names = tuple(field.name for owner in SchemaOwner for field in fields_by_owner[owner])
    if len(names) != len(set(names)):
        raise RuntimeError(f"{schema_version} required-field projection duplicates names")
    if len(field_order) != len(set(field_order)) or set(field_order) != set(names):
        raise RuntimeError(f"{schema_version} required-field projection order is incomplete")
    return RequiredFieldProjectionSpec(
        schema_version=schema_version,
        fields_by_owner=MappingProxyType(fields_by_owner),
        field_order=field_order,
    )


_MODEL_EXTERNAL_METADATA_FIELDS = tuple(
    RequiredField(name, f"$.external_metadata.{name}") for name in _EXTERNAL_METADATA_FIELD_NAMES
)
_AUTHOR_EXTERNAL_METADATA_FIELDS = tuple(
    RequiredField(name, f"$.proposed_facts.external_metadata.{name}")
    for name in _EXTERNAL_METADATA_FIELD_NAMES
)
_AUTHOR_PROPOSAL_FACT_BLOCK_FIELDS = tuple(
    RequiredField(name, f"$.proposed_facts.{name}") for name in _AUTHOR_PROPOSAL_FACT_BLOCK_NAMES
)
_AUTHOR_PROPOSAL_VERIFIED_HASH_FIELDS = tuple(
    RequiredField(name, f"$.verified_hashes.{name}")
    for name in _AUTHOR_PROPOSAL_VERIFIED_HASH_NAMES
)
_GATE_ITEM_REDUCER_BINDING_FIELDS = (
    RequiredField("work_id", "$.items[].work_id"),
    RequiredField("fidelity_identity", "$.items[].fidelity_identity"),
    RequiredField("vet_identity", "$.items[].vet_identity"),
    RequiredField("verified_hashes", "$.items[].verified_hashes.code"),
)
_GATE_ITEM_TRUSTED_BINDING_FIELDS = (
    RequiredField("stable_id", "$.items[].stable_id"),
    RequiredField("family_representative_id", "$.items[].family_representative_id"),
)
_GATE_VERIFIED_HASH_FIELDS = tuple(
    RequiredField(name, f"$.items[].verified_hashes.{name}")
    for name in (*_AUTHOR_PROPOSAL_VERIFIED_HASH_NAMES, "proposal")
)

REQUIRED_FIELD_PROJECTION_SPECS: Mapping[RequiredFieldProjection, RequiredFieldProjectionSpec] = (
    MappingProxyType(
        {
            RequiredFieldProjection.MODEL_EXTERNAL_METADATA: _required_field_projection_spec(
                MODEL_SCHEMA_VERSION_V3,
                author_gated=_MODEL_EXTERNAL_METADATA_FIELDS,
                field_order=_EXTERNAL_METADATA_FIELD_NAMES,
            ),
            RequiredFieldProjection.AUTHOR_EXTERNAL_METADATA: _required_field_projection_spec(
                AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
                author_gated=_AUTHOR_EXTERNAL_METADATA_FIELDS,
                field_order=_EXTERNAL_METADATA_FIELD_NAMES,
            ),
            RequiredFieldProjection.AUTHOR_PROPOSAL_FACT_BLOCK: _required_field_projection_spec(
                AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
                author_gated=_AUTHOR_PROPOSAL_FACT_BLOCK_FIELDS,
                field_order=_AUTHOR_PROPOSAL_FACT_BLOCK_NAMES,
            ),
            RequiredFieldProjection.AUTHOR_PROPOSAL_VERIFIED_HASH: _required_field_projection_spec(
                AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
                reducer_derived=_AUTHOR_PROPOSAL_VERIFIED_HASH_FIELDS,
                field_order=_AUTHOR_PROPOSAL_VERIFIED_HASH_NAMES,
            ),
            RequiredFieldProjection.GATE_ITEM_BINDING: _required_field_projection_spec(
                GATE_SCHEMA_VERSION_V3,
                reducer_derived=_GATE_ITEM_REDUCER_BINDING_FIELDS,
                trusted_intake=_GATE_ITEM_TRUSTED_BINDING_FIELDS,
                field_order=_GATE_ITEM_BINDING_FIELD_ORDER,
            ),
            RequiredFieldProjection.GATE_VERIFIED_HASH: _required_field_projection_spec(
                GATE_SCHEMA_VERSION_V3,
                reducer_derived=_GATE_VERIFIED_HASH_FIELDS,
                field_order=(*_AUTHOR_PROPOSAL_VERIFIED_HASH_NAMES, "proposal"),
            ),
            RequiredFieldProjection.MODEL_LEDGER_IDENTITY: _required_field_projection_spec(
                MODEL_SCHEMA_VERSION_V3,
                reducer_derived=(RequiredField("record_revision", "$.record_revision"),),
                trusted_intake=(RequiredField("stable_id", "$.stable_id"),),
                field_order=("stable_id", "record_revision"),
            ),
            RequiredFieldProjection.ATTEMPT_LEDGER_IDENTITY: _required_field_projection_spec(
                ATTEMPT_SCHEMA_VERSION_V3,
                reducer_derived=(RequiredField("attempt_id", "$.attempt_id"),),
                field_order=("attempt_id",),
            ),
            RequiredFieldProjection.GATE_LEDGER_IDENTITY: _required_field_projection_spec(
                GATE_SCHEMA_VERSION_V3,
                reducer_derived=(RequiredField("gate_id", "$.gate_id"),),
                field_order=("gate_id",),
            ),
        }
    )
)


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


def required_field_projection_spec(
    projection: RequiredFieldProjection,
) -> RequiredFieldProjectionSpec:
    """Return one Python-owned typed required-field projection.

    Parameters
    ----------
    projection:
        Closed projection name.

    Returns
    -------
    RequiredFieldProjectionSpec
        Immutable spec retaining every owner-specific field set and validation order.
    """

    return REQUIRED_FIELD_PROJECTION_SPECS[projection]


def _schema_path_selects(required_path: str, leaf_path: str) -> bool:
    """Return whether one required property selects a normalized schema leaf.

    Parameters
    ----------
    required_path:
        Exact property path named by Python projection policy.
    leaf_path:
        Normalized executable-schema leaf path.

    Returns
    -------
    bool
        True for the property itself or one of its object/collection descendants.
    """

    return (
        leaf_path == required_path
        or leaf_path.startswith(f"{required_path}.")
        or leaf_path.startswith(f"{required_path}[]")
    )


def validate_required_field_projection_specs() -> None:
    """Require every Python-owned projection field to match schema ownership.

    The executable schemas remain independent structural validators. This check
    compares their inert ownership annotations with the Python policy literals;
    it never constructs or changes the Python spec from schema contents.

    Raises
    ------
    SchemaOwnershipError
        If a required Python field is absent from the schema or assigned to a
        different provenance owner.
    """

    for projection, spec in REQUIRED_FIELD_PROJECTION_SPECS.items():
        paths_by_owner = {
            owner: schema_owner_paths(spec.schema_version, owner) for owner in SchemaOwner
        }
        for owner in SchemaOwner:
            mismatched: list[str] = []
            for field in spec.fields_for(owner):
                selected_by_owner = {
                    candidate_owner
                    for candidate_owner, candidate_paths in paths_by_owner.items()
                    if any(
                        _schema_path_selects(field.schema_path, candidate_path)
                        for candidate_path in candidate_paths
                    )
                }
                if selected_by_owner != {owner}:
                    mismatched.append(field.schema_path)
            if mismatched:
                raise SchemaOwnershipError(
                    f"{projection.value} required-field ownership mismatch for "
                    f"{owner.value}: {mismatched}"
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
