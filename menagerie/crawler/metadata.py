"""Authored-fact gating, accepted identities, and input-receipt validation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.constants import (
    AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
    MODEL_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION_V3,
    AccuracyVerdict,
)
from menagerie.crawler.identity import (
    compute_evidence_identity,
    compute_fidelity_identity,
    compute_recipe_revision,
    compute_source_identity,
    compute_vet_identity,
    stable_hash,
)
from menagerie.crawler.schema import (
    OwnedSchemaLeaf,
    author_gated_schema_paths,
    load_schema,
    owned_schema_leaves,
)


def _required_external_fields(schema_version: str) -> tuple[str, ...]:
    """Return schema-required top-level external-metadata fields.

    Parameters
    ----------
    schema_version:
        Crawler schema whose ``external_metadata`` definition is inspected.

    Returns
    -------
    tuple[str, ...]
        Required external-metadata fields in schema order.
    """

    schema = load_schema(schema_version)
    definitions = schema.get("$defs")
    external_metadata = (
        definitions.get("external_metadata") if isinstance(definitions, Mapping) else None
    )
    required = external_metadata.get("required") if isinstance(external_metadata, Mapping) else None
    if (
        not isinstance(required, list)
        or not required
        or not all(isinstance(field, str) and field for field in required)
    ):
        raise RuntimeError(f"{schema_version} does not define required external_metadata fields")
    if len(required) != len(set(required)):
        raise RuntimeError(f"{schema_version} duplicates required external_metadata fields")
    return tuple(required)


_MODEL_EXTERNAL_FIELDS = _required_external_fields(MODEL_SCHEMA_VERSION_V3)
_AUTHOR_EXTERNAL_FIELDS = _required_external_fields(AUTHOR_PROPOSAL_SCHEMA_VERSION_V3)
if _MODEL_EXTERNAL_FIELDS != _AUTHOR_EXTERNAL_FIELDS:
    raise RuntimeError("model-v3 and author-proposal-v3 external metadata requirements diverge")

MANDATORY_EXTERNAL_FIELDS = _MODEL_EXTERNAL_FIELDS

# This is the frozen demarcation from PLAN.md. These names are mechanically derivable
# only below the ``observed`` root; an authored leaf with the same name remains gated.
TORCHLENS_DERIVABLE_FIELDS = frozenset(
    {
        "parameter_count_total",
        "parameter_count_trainable",
        "input_shapes",
        "output_shapes",
        "op_types",
        "layer_types",
        "flops",
        "graph",
        "dtype",
        "device",
    }
)

_REQUIRED_NONEMPTY_ARRAYS = frozenset(
    {
        "modality",
        "architecture_class",
        "domain",
        "task",
        "paradigm",
        "authors",
        "institution",
        "keywords",
    }
)
_ARRAY_FIELDS = _REQUIRED_NONEMPTY_ARRAYS | {"lineage", "predecessors", "tags", "keywords"}
_REQUIRED_NONEMPTY_STRINGS = frozenset(
    {"family", "era", "key_contribution", "description", "original_framework", "run_framework"}
)
_CANONICAL_MODE_ORDER = ("train", "eval")
_INDEX_PATTERN = re.compile(r"\[[0-9]+\]")


class MetadataValidationError(ValueError):
    """Raised when an authored fact is missing, malformed, or ungated."""


@dataclass(frozen=True)
class MetadataValidationReport:
    """Authored metadata validation summary.

    Parameters
    ----------
    present_fields:
        Required external-metadata fields present in the object.
    gated_fields:
        Recursively identified authored leaves independently gated accurate.
    derivable_fields_present:
        Optional structural observations present incidentally.
    """

    present_fields: frozenset[str]
    gated_fields: frozenset[str]
    derivable_fields_present: frozenset[str]


@dataclass(frozen=True)
class AcceptedIdentities:
    """Identities recomputed from accepted fact bytes and current checker bytes."""

    source: str
    evidence: str
    recipe: str
    vet: str
    fidelity: Optional[str]


def canonical_meaningful_modes(value: Any, *, field: str) -> list[str]:
    """Return one validated meaningful-mode set in canonical order.

    Parameters
    ----------
    value:
        Candidate JSON meaningful-mode array.
    field:
        Dotted field name used in validation errors.

    Returns
    -------
    list[str]
        Unique declared modes ordered as ``train``, then ``eval``.
    """

    if (
        not isinstance(value, list)
        or not value
        or not all(mode in _CANONICAL_MODE_ORDER for mode in value)
        or len(value) != len(set(value))
    ):
        raise MetadataValidationError(f"{field} must contain unique train/eval modes")
    declared = set(value)
    return [mode for mode in _CANONICAL_MODE_ORDER if mode in declared]


def authored_fact_leaves(
    facts: Mapping[str, Any], *, schema_version: str = MODEL_SCHEMA_VERSION
) -> Mapping[str, Any]:
    """Return canonical model-owned author-gated fact leaves in deterministic order.

    The v3 model schema is the canonical write-policy source. This intentionally
    excludes proposal-time top-level mode observations that the model schema assigns
    to the reducer, while retaining declared meaningful modes and external mode
    claims. Collection items use normalized ``[]`` paths and repeated concrete values
    are collected in traversal order under their single schema leaf.

    Parameters
    ----------
    facts:
        Complete ``proposed_facts`` or corresponding canonical fact mapping.
    schema_version:
        Ownership policy to apply. V3 uses the frozen generated registry. The v2
        default exists only until the Phase-2 hubs switch atomically.

    Returns
    -------
    Mapping[str, Any]
        Dotted/indexed paths mapped to their exact JSON-compatible values.
    """

    if schema_version == MODEL_SCHEMA_VERSION:
        return _legacy_v2_authored_fact_leaves(facts)
    if schema_version != MODEL_SCHEMA_VERSION_V3:
        raise MetadataValidationError(f"unsupported authored ownership schema: {schema_version}")
    return _owned_instance_leaves(facts, schema_version=schema_version, schema_prefix="$")


def _legacy_v2_authored_fact_leaves(facts: Mapping[str, Any]) -> Mapping[str, Any]:
    """Retain the unwired v2 hub's historical authored projection.

    This path is categorically legacy-untrusted and removable once reducer/driver
    pass ``model.v3`` explicitly. It cannot be selected for v3 facts.
    """

    leaves: dict[str, Any] = {}

    def visit(value: Any, path: str) -> None:
        """Collect one historical concrete v2 leaf."""

        if isinstance(value, Mapping):
            if not value:
                leaves[path] = {}
                return
            for key in sorted(value, key=str):
                child = f"{path}.{key}" if path else str(key)
                visit(value[key], child)
            return
        if isinstance(value, list):
            if not value:
                leaves[path] = []
                return
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
            return
        leaves[path] = value

    for root in sorted(facts):
        if root in {"observed", "fidelity"}:
            continue
        if root == "modes":
            modes = facts[root]
            if isinstance(modes, Mapping):
                visit(modes.get("meaningful_modes"), "modes.meaningful_modes")
            continue
        visit(facts[root], str(root))
    return dict(sorted(leaves.items()))


def authored_model_leaves(model: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return schema-owned author-gated leaves from a canonical v3 model.

    Parameters
    ----------
    model:
        Complete canonical model mapping.

    Returns
    -------
    Mapping[str, Any]
        Exact model paths and values owned by ``author-gated`` schema policy.
    """

    return _owned_instance_leaves(
        model,
        schema_version=MODEL_SCHEMA_VERSION_V3,
        schema_prefix="$",
    )


def proposal_fact_block(model: Mapping[str, Any]) -> Mapping[str, Any]:
    """Select the proposal-fact block from a model using the v3 schema contract.

    Parameters
    ----------
    model:
        Canonical model or proposal-shaped fact source.

    Returns
    -------
    Mapping[str, Any]
        Exact fields required by ``author-proposal.v3`` ``proposed_facts``.
    """

    schema = load_schema(AUTHOR_PROPOSAL_SCHEMA_VERSION_V3)
    definitions = schema.get("$defs")
    proposed = definitions.get("proposed_facts") if isinstance(definitions, Mapping) else None
    required = proposed.get("required") if isinstance(proposed, Mapping) else None
    if not isinstance(required, list) or not all(isinstance(field, str) for field in required):
        raise MetadataValidationError("author-proposal.v3 proposed_facts contract is incomplete")
    missing = [field for field in required if field not in model]
    if missing:
        raise MetadataValidationError(f"model lacks proposal fact blocks: {missing}")
    return {field: model[field] for field in required}


def _owned_instance_leaves(
    value: Mapping[str, Any],
    *,
    schema_version: str,
    schema_prefix: str,
) -> Mapping[str, Any]:
    """Project concrete instance leaves selected by schema ownership metadata.

    Parameters
    ----------
    value:
        Instance mapping rooted at ``schema_prefix``.
    schema_version:
        Ownership-annotated schema discriminator.
    schema_prefix:
        Exact schema path corresponding to ``value``.

    Returns
    -------
    Mapping[str, Any]
        Normalized paths relative to ``schema_prefix`` and their exact values.

    Raises
    ------
    MetadataValidationError
        If the instance exposes a leaf not present in the frozen schema registry.
    """

    owned = author_gated_schema_paths(schema_version)
    all_paths = frozenset(leaf.path for leaf in _owned_schema_registry(schema_version))
    selected: dict[str, Any] = {}

    def relative_path(path: str) -> str:
        """Return a normalized instance path relative to the supplied schema root."""

        path = _INDEX_PATTERN.sub("[]", path)
        if schema_prefix == "$":
            return path.removeprefix("$.")
        return path.removeprefix(f"{schema_prefix}.")

    def select(path: str, item: Any) -> None:
        """Collect one concrete value under its normalized owned schema leaf."""

        relative = relative_path(path)
        if "[]" not in relative:
            selected[relative] = item
            return
        existing = selected.setdefault(relative, [])
        if not isinstance(existing, list):
            raise MetadataValidationError(f"normalized collection leaf conflicts at {relative}")
        existing.append(item)

    def visit(item: Any, path: str) -> None:
        """Visit one concrete instance node using normalized schema paths."""

        normalized = _INDEX_PATTERN.sub("[]", path)
        if normalized in all_paths:
            if normalized in owned:
                select(path, item)
            return
        if isinstance(item, Mapping):
            if not item:
                return
            for key in sorted(item, key=str):
                visit(item[key], f"{path}.{key}")
            return
        if isinstance(item, list):
            if not item:
                return
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        descendants = frozenset(
            candidate for candidate in all_paths if candidate.startswith(f"{normalized}.")
        )
        if descendants:
            return
        raise MetadataValidationError(f"unowned schema leaf at {relative_path(path)}")

    for key in sorted(value, key=str):
        visit(value[key], f"{schema_prefix}.{key}")
    return dict(sorted(selected.items()))


def _owned_schema_registry(schema_version: str) -> tuple[OwnedSchemaLeaf, ...]:
    """Return the cached validated ownership registry without duplicating policy."""

    return owned_schema_leaves(schema_version)


def _evidence_references(facts: Mapping[str, Any]) -> Mapping[str, Any]:
    """Derive exact field-to-literal-evidence references from accepted facts."""

    references: dict[str, list[str]] = {}
    evidence = facts.get("evidence")
    excerpts = evidence.get("excerpts", []) if isinstance(evidence, Mapping) else []
    for excerpt in excerpts:
        if not isinstance(excerpt, Mapping):
            continue
        evidence_id = excerpt.get("evidence_id")
        if not isinstance(evidence_id, str):
            continue
        for supported in excerpt.get("supports", []):
            if isinstance(supported, str):
                references.setdefault(supported, []).append(evidence_id)
    return {path: sorted(set(values)) for path, values in sorted(references.items())}


def recompute_accepted_identities(
    facts: Mapping[str, Any],
    *,
    checker_prompt_hash: str,
    checker_model: str,
    checker_version: str,
    schema_version: str = MODEL_SCHEMA_VERSION,
) -> AcceptedIdentities:
    """Recompute all accepted identities from fact bytes and checker prompt bytes.

    Parameters
    ----------
    facts:
        Complete accepted model/proposal facts.
    checker_prompt_hash, checker_model, checker_version:
        Current frozen checker prompt byte hash and exact checker identity.
    schema_version:
        Exact model ownership policy used for vet identity derivation.

    Returns
    -------
    AcceptedIdentities
        Source, evidence, recipe, vet, and optional fidelity identities.
    """

    source_resolution = _mapping(facts.get("source_resolution"), "source_resolution")
    evidence_block = _mapping(facts.get("evidence"), "evidence")
    implementation = _mapping(facts.get("implementation"), "implementation")
    input_contract = _mapping(facts.get("input_contract"), "input_contract")
    modes = _mapping(facts.get("modes"), "modes")
    sources = source_resolution.get("sources")
    excerpts = evidence_block.get("excerpts")
    if not isinstance(sources, list) or not isinstance(excerpts, list):
        raise MetadataValidationError("source/evidence facts are incomplete")
    source = compute_source_identity(
        [value for value in sources if isinstance(value, Mapping)],
        _mapping(source_resolution.get("search_report"), "source_resolution.search_report"),
    )
    evidence = compute_evidence_identity(
        [value for value in excerpts if isinstance(value, Mapping)]
    )
    meaningful_modes = canonical_meaningful_modes(
        modes.get("meaningful_modes"), field="modes.meaningful_modes"
    )
    recipe_facts = {
        "implementation": {
            key: value for key, value in implementation.items() if key != "recipe_revision"
        },
        "input_contract": input_contract,
        "modes": {"meaningful_modes": meaningful_modes},
    }
    recipe = compute_recipe_revision(recipe_facts, source)
    authored = authored_fact_leaves(facts, schema_version=schema_version)
    vet = compute_vet_identity(
        authored_metadata=authored,
        evidence_references=_evidence_references(facts),
        source_identity=source,
        evidence_identity=evidence,
        prompt_hash=checker_prompt_hash,
        checker_model=checker_model,
        checker_version=checker_version,
    )
    fidelity_block = facts.get("fidelity")
    rung = source_resolution.get("rung")
    fidelity_required = bool(
        isinstance(fidelity_block, Mapping) and fidelity_block.get("required")
    ) or rung in {"R3_PORT", "R4_REIMPLEMENT"}
    fidelity = None
    if fidelity_required:
        source_to_code_map = implementation.get("source_to_code_map")
        if not isinstance(source_to_code_map, list):
            raise MetadataValidationError("implementation.source_to_code_map is incomplete")
        implementation_hash = implementation.get("code_sha256")
        if implementation_hash is None:
            implementation_hash = stable_hash(recipe_facts["implementation"])
        fidelity = compute_fidelity_identity(
            source_identity=source,
            evidence_identity=evidence,
            implementation_hash=str(implementation_hash),
            source_to_code_map=[
                value for value in source_to_code_map if isinstance(value, Mapping)
            ],
            prompt_hash=checker_prompt_hash,
            checker_model=checker_model,
            checker_version=checker_version,
        )
    return AcceptedIdentities(source, evidence, recipe, vet, fidelity)


def validate_external_metadata(
    metadata: Mapping[str, Any],
    *,
    field_checks: Optional[Sequence[Mapping[str, Any]]] = None,
) -> MetadataValidationReport:
    """Validate external metadata under the source-read/derivable demarcation."""

    missing = [field for field in MANDATORY_EXTERNAL_FIELDS if field not in metadata]
    if missing:
        raise MetadataValidationError(f"missing mandatory external metadata: {missing}")
    for field in _ARRAY_FIELDS:
        value = metadata[field]
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise MetadataValidationError(f"external_metadata.{field} must be a string array")
        if field in _REQUIRED_NONEMPTY_ARRAYS and not value:
            raise MetadataValidationError(f"external_metadata.{field} must be non-empty")
    for field in _REQUIRED_NONEMPTY_STRINGS:
        value = metadata[field]
        if not isinstance(value, str) or not value.strip():
            raise MetadataValidationError(f"external_metadata.{field} must be non-empty")
    if not isinstance(metadata["citation"], Mapping):
        raise MetadataValidationError("external_metadata.citation must be an object")
    modes = metadata["modes"]
    if not isinstance(modes, Mapping):
        raise MetadataValidationError("external_metadata.modes must be an object")
    canonical_meaningful_modes(
        modes.get("meaningful_modes"), field="external_metadata.modes.meaningful_modes"
    )
    if modes.get("train_eval_divergence") not in {"none", "statistical", "structural"}:
        raise MetadataValidationError(
            "external_metadata.modes.train_eval_divergence is not canonical"
        )
    gated = (
        _validate_external_field_checks(field_checks) if field_checks is not None else frozenset()
    )
    derivable = frozenset(field for field in TORCHLENS_DERIVABLE_FIELDS if field in metadata)
    return MetadataValidationReport(
        present_fields=frozenset(MANDATORY_EXTERNAL_FIELDS),
        gated_fields=gated,
        derivable_fields_present=derivable,
    )


def validate_external_metadata_for_write(
    metadata: Mapping[str, Any], gate_item: Mapping[str, Any]
) -> MetadataValidationReport:
    """Retain the legacy external-metadata-only write validation API."""

    _validate_gate_header(gate_item)
    checks = gate_item.get("field_checks")
    if not isinstance(checks, list):
        raise MetadataValidationError("metadata gate has no per-field checks")
    return validate_external_metadata(metadata, field_checks=checks)


def validate_authored_facts_for_write(
    facts: Mapping[str, Any], gate_item: Mapping[str, Any]
) -> MetadataValidationReport:
    """Require one unique accurate evidence/relevance check for every authored leaf.

    Parameters
    ----------
    facts:
        Entire proposed/canonical fact tree copied at canonical write.
    gate_item:
        Independently bound metadata checker item.

    Returns
    -------
    MetadataValidationReport
        Exhaustive authored-leaf gate coverage.
    """

    _validate_gate_header(gate_item)
    external = _mapping(facts.get("external_metadata"), "external_metadata")
    validate_external_metadata(external)
    checks = gate_item.get("field_checks")
    if not isinstance(checks, list):
        raise MetadataValidationError("metadata gate has no per-field checks")
    strict_v3 = "terminal_disposition" in gate_item
    required = authored_fact_leaves(
        facts,
        schema_version=MODEL_SCHEMA_VERSION_V3 if strict_v3 else MODEL_SCHEMA_VERSION,
    )
    source_ids, evidence_by_id = _authored_reference_indexes(facts)
    verdicts: dict[str, str] = {}
    for check in checks:
        if not isinstance(check, Mapping):
            raise MetadataValidationError("metadata field check must be an object")
        raw_field = check.get("field")
        if not isinstance(raw_field, str):
            raise MetadataValidationError("metadata field check has no field name")
        field = raw_field.removeprefix("proposed_facts.")
        if field not in required:
            if strict_v3:
                raise MetadataValidationError(f"extraneous authored field check: {field}")
            continue
        if field in verdicts:
            raise MetadataValidationError(f"duplicate authored field check: {field}")
        checked_source_ids = check.get("checked_source_ids")
        if not isinstance(checked_source_ids, list) or not checked_source_ids:
            raise MetadataValidationError(
                f"authored field check lacks checked source context: {field}"
            )
        evidence_ids = check.get("evidence_ids")
        if not isinstance(evidence_ids, list):
            raise MetadataValidationError(f"authored field check has invalid evidence_ids: {field}")
        if not _is_keyword_leaf(field) and not evidence_ids:
            raise MetadataValidationError(
                f"authored field check lacks verified evidence support: {field}"
            )
        if strict_v3:
            _validate_field_check_references(
                field,
                checked_source_ids=checked_source_ids,
                evidence_ids=evidence_ids,
                source_ids=source_ids,
                evidence_by_id=evidence_by_id,
            )
        verdicts[field] = str(check.get("verdict"))
    missing = set(required) - set(verdicts)
    if missing:
        raise MetadataValidationError(f"ungated authored facts: {sorted(missing)}")
    failed = {
        field: verdict
        for field, verdict in verdicts.items()
        if verdict != AccuracyVerdict.ACCURATE.value
    }
    if failed:
        raise MetadataValidationError(f"non-accurate authored facts: {failed}")
    if strict_v3:
        _validate_rung_and_search_attestation(facts, gate_item)
    return MetadataValidationReport(
        present_fields=frozenset(MANDATORY_EXTERNAL_FIELDS),
        gated_fields=frozenset(verdicts),
        derivable_fields_present=frozenset(),
    )


def _is_keyword_leaf(field: str) -> bool:
    """Return whether an authored leaf is a user-search keyword.

    Parameters
    ----------
    field:
        Dotted/indexed authored fact path.

    Returns
    -------
    bool
        True for individual ``external_metadata.keywords`` entries.
    """

    return field in {"external_metadata.keywords", "external_metadata.keywords[]"}


def _authored_reference_indexes(
    facts: Mapping[str, Any],
) -> tuple[frozenset[str], Mapping[str, Mapping[str, Any]]]:
    """Build exact source/evidence indexes for v3 gate reference validation.

    Parameters
    ----------
    facts:
        Complete proposed fact tree.

    Returns
    -------
    tuple[frozenset[str], Mapping[str, Mapping[str, Any]]]
        Exact source-ID set and evidence excerpts keyed by evidence ID.
    """

    resolution = _mapping(facts.get("source_resolution"), "source_resolution")
    raw_sources = resolution.get("sources")
    evidence = _mapping(facts.get("evidence"), "evidence")
    raw_excerpts = evidence.get("excerpts")
    if not isinstance(raw_sources, list) or not isinstance(raw_excerpts, list):
        raise MetadataValidationError("authored source/evidence references are incomplete")
    source_values = [
        source.get("source_id") for source in raw_sources if isinstance(source, Mapping)
    ]
    if (
        not all(isinstance(source_id, str) and source_id for source_id in source_values)
        or len(source_values) != len(raw_sources)
        or len(source_values) != len(set(source_values))
    ):
        raise MetadataValidationError("authored source IDs must be complete and unique")
    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    for excerpt in raw_excerpts:
        if not isinstance(excerpt, Mapping):
            raise MetadataValidationError("authored evidence excerpt must be an object")
        evidence_id = excerpt.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id or evidence_id in evidence_by_id:
            raise MetadataValidationError("authored evidence IDs must be complete and unique")
        if excerpt.get("source_id") not in source_values:
            raise MetadataValidationError(
                f"evidence {evidence_id} references a source outside the proposal"
            )
        evidence_by_id[evidence_id] = excerpt
    return frozenset(str(source_id) for source_id in source_values), evidence_by_id


def _validate_field_check_references(
    field: str,
    *,
    checked_source_ids: Sequence[Any],
    evidence_ids: Sequence[Any],
    source_ids: frozenset[str],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    """Resolve one exact v3 checker field check to proposal sources and excerpts."""

    if not all(isinstance(source_id, str) and source_id for source_id in checked_source_ids) or len(
        checked_source_ids
    ) != len(set(checked_source_ids)):
        raise MetadataValidationError(
            f"authored field check source IDs are invalid or duplicated: {field}"
        )
    extraneous_sources = set(checked_source_ids) - source_ids
    if extraneous_sources:
        raise MetadataValidationError(
            f"authored field check references extraneous sources for {field}: "
            f"{sorted(extraneous_sources)}"
        )
    if not all(isinstance(evidence_id, str) and evidence_id for evidence_id in evidence_ids) or len(
        evidence_ids
    ) != len(set(evidence_ids)):
        raise MetadataValidationError(
            f"authored field check evidence IDs are invalid or duplicated: {field}"
        )
    for evidence_id in evidence_ids:
        excerpt = evidence_by_id.get(str(evidence_id))
        if excerpt is None:
            raise MetadataValidationError(
                f"authored field check references fabricated evidence for {field}: {evidence_id}"
            )
        if excerpt.get("source_id") not in checked_source_ids:
            raise MetadataValidationError(
                f"authored field evidence source was not checked for {field}: {evidence_id}"
            )
        supports = excerpt.get("supports")
        accepted_supports = {field, f"proposed_facts.{field}"}
        if not isinstance(supports, list) or not accepted_supports.intersection(supports):
            raise MetadataValidationError(
                f"authored evidence {evidence_id} does not support exact field {field}"
            )


def _validate_rung_and_search_attestation(
    facts: Mapping[str, Any], gate_item: Mapping[str, Any]
) -> None:
    """Consume exact rung equality and closed R4 search-attestation findings."""

    resolution = _mapping(facts.get("source_resolution"), "source_resolution")
    rung = resolution.get("rung")
    rung_check = _mapping(gate_item.get("rung_check"), "rung_check")
    if rung_check.get("selected_rung") != rung:
        raise MetadataValidationError("checker selected_rung does not match proposed source rung")
    if rung_check.get("verdict") == AccuracyVerdict.ACCURATE.value and (
        rung_check.get("highest_applicable") != rung_check.get("selected_rung")
    ):
        raise MetadataValidationError(
            "accurate rung check requires highest_applicable == selected_rung"
        )
    if rung != "R4_REIMPLEMENT":
        return
    search_report = _mapping(resolution.get("search_report"), "source_resolution.search_report")
    links = search_report.get("links_checked")
    findings = rung_check.get("findings")
    if not isinstance(links, list) or not isinstance(findings, list):
        raise MetadataValidationError("R4 rung check lacks typed search attestations")
    attested = {
        finding.removeprefix("search-attested:")
        for finding in findings
        if isinstance(finding, str) and finding.startswith("search-attested:")
    }
    cannot_verify = any(
        isinstance(finding, str) and finding.startswith("search-cannot-verify:")
        for finding in findings
    )
    if rung_check.get("verdict") == AccuracyVerdict.ACCURATE.value:
        missing = set(str(link) for link in links) - attested
        if missing:
            raise MetadataValidationError(
                f"accurate R4 rung check lacks re-executed search attestations: {sorted(missing)}"
            )
        if cannot_verify:
            raise MetadataValidationError(
                "accurate R4 rung check cannot carry search-cannot-verify"
            )
    elif not cannot_verify:
        raise MetadataValidationError(
            "non-accurate R4 rung check requires typed search-cannot-verify"
        )


def input_signature_matches_contract(signature: Any, input_contract: Mapping[str, Any]) -> bool:
    """Return whether a receipt fully and exactly describes the accepted dummy call.

    Parameters
    ----------
    signature:
        Worker input pytree signature.
    input_contract:
        Accepted typed args/kwargs and non-tensor values.

    Returns
    -------
    bool
        True only for a non-null complete signature with exact leaf coverage.
    """

    if not isinstance(signature, Mapping) or "tree" not in signature:
        return False
    leaves = signature.get("leaves")
    if not isinstance(leaves, list) or not leaves:
        return False
    by_path: dict[str, Mapping[str, Any]] = {}
    for leaf in leaves:
        if not isinstance(leaf, Mapping) or not isinstance(leaf.get("path"), str):
            return False
        path = str(leaf["path"]).removeprefix("input.")
        if path in by_path:
            return False
        by_path[path] = leaf
    expected: dict[str, Mapping[str, Any]] = {}
    for collection in ("args", "kwargs"):
        values = input_contract.get(collection)
        if not isinstance(values, list):
            return False
        for value in values:
            if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
                return False
            expected[str(value["path"])] = value
    non_tensor = input_contract.get("non_tensor_values")
    if not isinstance(non_tensor, list):
        return False
    for value in non_tensor:
        if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
            return False
        expected[str(value["path"])] = value
    if not expected or set(by_path) != set(expected):
        return False
    for path, contract_leaf in expected.items():
        observed = by_path[path]
        if "shape" in contract_leaf:
            expected_shape = contract_leaf.get("shape")
            observed_shape = observed.get("shape")
            if observed.get("kind") != "tensor" or not isinstance(observed_shape, list):
                return False
            if not isinstance(expected_shape, list) or len(expected_shape) != len(observed_shape):
                return False
            if any(
                isinstance(dimension, int) and dimension != observed_shape[index]
                for index, dimension in enumerate(expected_shape)
            ):
                return False
            expected_dtype = str(contract_leaf.get("dtype", "")).lower().replace("torch.", "")
            observed_dtype = str(observed.get("dtype", "")).lower().replace("torch.", "")
            if expected_dtype != observed_dtype:
                return False
        else:
            if observed.get("kind") != "python":
                return False
            if observed.get("value_sha256") != stable_hash(contract_leaf.get("value")):
                return False
    return True


def _validate_gate_header(gate_item: Mapping[str, Any]) -> None:
    """Validate the item-wide accurate verdict and integrity result."""

    if gate_item.get("verdict") != AccuracyVerdict.ACCURATE.value:
        raise MetadataValidationError("metadata gate is not accurate; canonical write is blocked")
    integrity = gate_item.get("integrity")
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("verdict") != AccuracyVerdict.ACCURATE.value
    ):
        raise MetadataValidationError(
            "metadata integrity is not accurate; canonical write is blocked"
        )


def _validate_external_field_checks(
    field_checks: Sequence[Mapping[str, Any]],
) -> frozenset[str]:
    """Validate legacy top-level external-metadata field coverage."""

    verdicts: dict[str, str] = {}
    for check in field_checks:
        raw_field = check.get("field")
        if not isinstance(raw_field, str):
            raise MetadataValidationError("metadata field check has no field name")
        field = raw_field.removeprefix("external_metadata.")
        if field not in MANDATORY_EXTERNAL_FIELDS:
            continue
        if field in verdicts:
            raise MetadataValidationError(f"duplicate metadata field check: {field}")
        verdicts[field] = str(check.get("verdict"))
    missing = set(MANDATORY_EXTERNAL_FIELDS) - set(verdicts)
    if missing:
        raise MetadataValidationError(f"ungated mandatory external metadata: {sorted(missing)}")
    failed = {
        field: verdict
        for field, verdict in verdicts.items()
        if verdict != AccuracyVerdict.ACCURATE.value
    }
    if failed:
        raise MetadataValidationError(f"non-accurate external metadata fields: {failed}")
    return frozenset(verdicts)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Return an object mapping or raise a metadata validation error."""

    if not isinstance(value, Mapping):
        raise MetadataValidationError(f"{name} must be an object")
    return value
