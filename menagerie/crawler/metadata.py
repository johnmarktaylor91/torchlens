"""Authored-fact gating, accepted identities, and input-receipt validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.constants import (
    AUTHOR_PROPOSAL_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
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
from menagerie.crawler.schema import load_schema


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


_MODEL_EXTERNAL_FIELDS = _required_external_fields(MODEL_SCHEMA_VERSION)
_AUTHOR_EXTERNAL_FIELDS = _required_external_fields(AUTHOR_PROPOSAL_SCHEMA_VERSION)
if _MODEL_EXTERNAL_FIELDS != _AUTHOR_EXTERNAL_FIELDS:
    raise RuntimeError("model-v2 and author-proposal-v2 external metadata requirements diverge")

MANDATORY_EXTERNAL_FIELDS = _MODEL_EXTERNAL_FIELDS

# This is the frozen demarcation from PLAN.md. These are the only model facts excluded
# from the recursively derived source-read/human-judgment write gate.
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
    {"modality", "architecture_class", "domain", "task", "paradigm", "authors", "institution"}
)
_ARRAY_FIELDS = _REQUIRED_NONEMPTY_ARRAYS | {"lineage", "predecessors", "tags", "keywords"}
_REQUIRED_NONEMPTY_STRINGS = frozenset(
    {"family", "era", "key_contribution", "description", "original_framework", "run_framework"}
)
_CANONICAL_MODE_ORDER = ("train", "eval")


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


def authored_fact_leaves(facts: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return every recursively gated authored leaf in deterministic path order.

    Empty arrays and objects are leaves because their emptiness is itself an authored
    judgment. Only the frozen TorchLens-derivable field names are excluded.

    Parameters
    ----------
    facts:
        Complete ``proposed_facts`` or corresponding canonical fact mapping.

    Returns
    -------
    Mapping[str, Any]
        Dotted/indexed paths mapped to their exact JSON-compatible values.
    """

    leaves: dict[str, Any] = {}

    def visit(value: Any, path: str) -> None:
        """Collect one value recursively unless it is mechanically derivable."""

        final_name = path.rsplit(".", 1)[-1].split("[", 1)[0]
        if final_name in TORCHLENS_DERIVABLE_FIELDS:
            return
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
            for index, child_value in enumerate(value):
                visit(child_value, f"{path}[{index}]")
            return
        leaves[path] = value

    for root in sorted(facts):
        if root == "fidelity":
            continue
        if root == "modes":
            modes = facts[root]
            if isinstance(modes, Mapping):
                visit(modes.get("meaningful_modes"), "modes.meaningful_modes")
            continue
        visit(facts[root], str(root))
    return dict(sorted(leaves.items()))


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
) -> AcceptedIdentities:
    """Recompute all accepted identities from fact bytes and checker prompt bytes.

    Parameters
    ----------
    facts:
        Complete accepted model/proposal facts.
    checker_prompt_hash, checker_model, checker_version:
        Current frozen checker prompt byte hash and exact checker identity.

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
    authored = authored_fact_leaves(facts)
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
    """Require one unique accurate evidence-backed check for every authored leaf.

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
    required = authored_fact_leaves(facts)
    checks = gate_item.get("field_checks")
    if not isinstance(checks, list):
        raise MetadataValidationError("metadata gate has no per-field checks")
    verdicts: dict[str, str] = {}
    for check in checks:
        raw_field = check.get("field")
        if not isinstance(raw_field, str):
            raise MetadataValidationError("metadata field check has no field name")
        field = raw_field.removeprefix("proposed_facts.")
        if field not in required:
            continue
        if field in verdicts:
            raise MetadataValidationError(f"duplicate authored field check: {field}")
        if not check.get("evidence_ids") or not check.get("checked_source_ids"):
            raise MetadataValidationError(
                f"authored field check lacks verified evidence support: {field}"
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
    return MetadataValidationReport(
        present_fields=frozenset(MANDATORY_EXTERNAL_FIELDS),
        gated_fields=frozenset(verdicts),
        derivable_fields_present=frozenset(),
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
