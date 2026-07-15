"""Staged author-proposal validation and deterministic anti-slop gates."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from menagerie.crawler.constants import AUTHOR_PROPOSAL_SCHEMA_VERSION, SourceRung
from menagerie.crawler.evidence import EvidenceValidationError, evidence_ids, validate_evidence
from menagerie.crawler.fetcher import cas_path as source_cas_path
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.metadata import MANDATORY_EXTERNAL_FIELDS
from menagerie.crawler.schema import PayloadValidationError, validate_payload

DEFAULT_GATED_CLAIMS = frozenset(
    {f"external_metadata.{field}" for field in MANDATORY_EXTERNAL_FIELDS}
    | {
        "external_metadata.description",
        "source_resolution.rung",
        "taxonomy",
        "input_contract",
    }
)
_FORBIDDEN_CALLS = frozenset({"eval", "exec", "compile"})
_SLOP_PATTERNS = (
    r"\bcompact\s+(?:stand[- ]?in|substitute|approximation|version)\b",
    r"\bgeneric\s+(?:stand[- ]?in|substitute|implementation|version)\b",
    r"\b(?:knowingly\s+)?simplif(?:ied|ication)\b",
    r"\b(?:rough(?:ly)?\s+)?approximat(?:e|ed|ion)\b",
    r"\btoy\s+(?:model|implementation|version|replica|example)\b",
    r"\b(?:stand[- ]?in|placeholder|mock|surrogate|proxy)\b",
    r"\b(?:minimal|lightweight|reduced)\s+(?:facsimile|imitation|replica|substitute)\b",
    r"\brepresentative\s+(?:approximation|substitute|implementation)\b",
)
_SUPPORT_ALIASES = {
    "citation": "external_metadata.citation",
    "country": "external_metadata.country",
    "license": "external_metadata.license",
    "year": "external_metadata.year",
}
_GENERIC_MODEL_CALLS = frozenset(
    {
        "AdaptiveAvgPool1d",
        "AdaptiveAvgPool2d",
        "AvgPool1d",
        "AvgPool2d",
        "BatchNorm1d",
        "BatchNorm2d",
        "Conv1d",
        "Conv2d",
        "Dropout",
        "Flatten",
        "GELU",
        "LayerNorm",
        "Linear",
        "MaxPool1d",
        "MaxPool2d",
        "ReLU",
        "Sequential",
        "Sigmoid",
        "Softmax",
        "Tanh",
    }
)
_GENERIC_FAMILY_NAMES = frozenset(
    {"feedforward", "mlp", "multilayer perceptron", "sequential", "simple neural network"}
)
_SUPPORT_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "also",
        "architecture",
        "based",
        "from",
        "into",
        "model",
        "network",
        "source",
        "that",
        "their",
        "this",
        "using",
        "with",
    }
)
_WRITE_METHODS = frozenset(
    {"write_text", "write_bytes", "touch", "mkdir", "rename", "replace", "unlink", "rmdir"}
)
_IMPLEMENTATION_SOURCE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cu",
        ".cuh",
        ".cxx",
        ".go",
        ".h",
        ".hh",
        ".hpp",
        ".hxx",
        ".ipynb",
        ".java",
        ".jl",
        ".js",
        ".kt",
        ".lua",
        ".m",
        ".mm",
        ".py",
        ".pyx",
        ".r",
        ".rs",
        ".scala",
        ".swift",
        ".ts",
    }
)
_NON_IMPLEMENTATION_CODE_NAMES = frozenset(
    {
        "__init__.py",
        "conftest.py",
        "setup.py",
        "version.py",
    }
)
_NON_IMPLEMENTATION_SOURCE_DIRS = frozenset({".github", "ci", "doc", "docs", "test", "tests"})
_MAX_TEXT_INVENTORY_BYTES = 8 * 1024**2


class ProposalValidationError(ValueError):
    """Raised when a staged proposal fails schema, grounding, or anti-slop checks."""


@dataclass(frozen=True)
class ProposalValidationReport:
    """Summary of a fully validated staged proposal.

    Parameters
    ----------
    stable_id:
        Validated model identity.
    rung:
        Validated source-ladder rung.
    code_path:
        Validated staged code path, if applicable.
    supported_claims:
        Claim categories backed by literal excerpts.
    """

    stable_id: str
    rung: SourceRung
    code_path: Optional[Path]
    supported_claims: frozenset[str]


def validate_author_proposal(
    proposal: Mapping[str, Any],
    *,
    allowed_model_dir: Union[str, Path],
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    required_claims: Optional[Iterable[str]] = None,
    cas_root: Union[str, Path, None] = None,
) -> ProposalValidationReport:
    """Validate one complete author proposal without modifying it.

    Parameters
    ----------
    proposal:
        Complete ``author-proposal.v2`` object.
    allowed_model_dir:
        Only directory in which staged typed code may reside or write.
    source_manifest:
        Exact controlled-fetch source manifests.
    required_claims:
        Optional gated claim categories. The default enforces the plan's core
        externally-authored categories.
    cas_root:
        Optional source CAS root.

    Returns
    -------
    ProposalValidationReport
        Immutable validation summary.

    Raises
    ------
    ProposalValidationError
        If schema, evidence, code, rung, link, or anti-slop validation fails.
    """

    try:
        validate_payload(proposal, AUTHOR_PROPOSAL_SCHEMA_VERSION)
    except PayloadValidationError as exc:
        raise ProposalValidationError(str(exc)) from exc
    facts = _mapping(proposal.get("proposed_facts"), "proposed_facts")
    resolution = _mapping(facts.get("source_resolution"), "source_resolution")
    try:
        rung = SourceRung(str(resolution.get("rung")))
    except ValueError as exc:
        raise ProposalValidationError("source_resolution.rung is not canonical") from exc
    _validate_mandatory_source_link(resolution)
    _validate_description(facts)
    evidence = _mapping(facts.get("evidence"), "evidence")
    claims = set(required_claims if required_claims is not None else DEFAULT_GATED_CLAIMS)
    if _citation_is_present(facts):
        claims.add("external_metadata.citation")
    try:
        evidence_report = validate_evidence(
            evidence,
            source_manifest,
            claims,
            cas_root=cas_root,
            require_family_grounding=True,
        )
    except EvidenceValidationError as exc:
        raise ProposalValidationError(str(exc)) from exc
    _validate_claim_support(facts, evidence, claims)
    known_evidence = evidence_ids(evidence)
    _validate_citation(facts, known_evidence)
    _validate_citation_consistency(facts)
    implementation = _mapping(facts.get("implementation"), "implementation")
    allowed_dir = Path(allowed_model_dir).resolve()
    code_path = _validate_code(implementation, rung, allowed_dir)
    _validate_source_ladder(
        rung,
        resolution,
        implementation,
        evidence,
        known_evidence,
        source_manifest,
        cas_root,
    )
    _validate_structural_slop(facts, code_path)
    _validate_anti_slop(facts)
    return ProposalValidationReport(
        stable_id=str(proposal["stable_id"]),
        rung=rung,
        code_path=code_path,
        supported_claims=evidence_report.supported_claims,
    )


def _validate_mandatory_source_link(resolution: Mapping[str, Any]) -> None:
    """Validate the public primary-link invariant.

    Parameters
    ----------
    resolution:
        Source-resolution block.

    Raises
    ------
    ProposalValidationError
        If the primary source link is absent or inconsistent.
    """

    if resolution.get("mandatory_link_status") != "ok":
        raise ProposalValidationError("mandatory source link is not satisfied")
    primary_id = resolution.get("primary_source_id")
    sources = resolution.get("sources")
    if not isinstance(primary_id, str) or not primary_id or not isinstance(sources, list):
        raise ProposalValidationError("primary source identity is missing")
    primary = next(
        (
            source
            for source in sources
            if isinstance(source, Mapping) and source.get("source_id") == primary_id
        ),
        None,
    )
    if primary is None:
        raise ProposalValidationError("primary_source_id does not name a declared source")
    url = primary.get("url")
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        raise ProposalValidationError("primary source must have an exact public URL")


def _validate_description(facts: Mapping[str, Any]) -> None:
    """Reject absent or whitespace-only authored descriptions.

    Parameters
    ----------
    facts:
        Proposed fact tree.

    Raises
    ------
    ProposalValidationError
        If external or website description is empty.
    """

    metadata = _mapping(facts.get("external_metadata"), "external_metadata")
    website = _mapping(facts.get("website"), "website")
    for field, value in (
        ("external_metadata.description", metadata.get("description")),
        ("website.description", website.get("description")),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ProposalValidationError(f"{field} must be non-empty")


def _citation_is_present(facts: Mapping[str, Any]) -> bool:
    """Return whether the proposal asserts an introducing citation.

    Parameters
    ----------
    facts:
        Proposed fact tree.

    Returns
    -------
    bool
        True for a present citation.
    """

    citation = facts.get("citation")
    return isinstance(citation, Mapping) and citation.get("status") == "present"


def _validate_citation(facts: Mapping[str, Any], known_evidence: frozenset[str]) -> None:
    """Reject fabricated or evidence-free citation claims.

    Parameters
    ----------
    facts:
        Proposed fact tree.
    known_evidence:
        Literal evidence identifiers.

    Raises
    ------
    ProposalValidationError
        If a present citation lacks a source, identity, or valid evidence.
    """

    citation = _mapping(facts.get("citation"), "citation")
    if citation.get("status") != "present":
        return
    if not all(
        isinstance(citation.get(field), str) and str(citation[field]).strip()
        for field in ("title", "url")
    ):
        raise ProposalValidationError("present citation must name a title and public URL")
    cited_ids = citation.get("source_evidence_ids")
    if not isinstance(cited_ids, list) or not cited_ids or not set(cited_ids) <= known_evidence:
        raise ProposalValidationError("citation references missing or fabricated evidence")


def _validate_citation_consistency(facts: Mapping[str, Any]) -> None:
    """Require the public citation to equal the accuracy-gated metadata citation.

    Parameters
    ----------
    facts:
        Proposed fact tree.

    Raises
    ------
    ProposalValidationError
        If the top-level and external-metadata citation blocks diverge.
    """

    metadata = _mapping(facts.get("external_metadata"), "external_metadata")
    external_citation = _mapping(metadata.get("citation"), "external_metadata.citation")
    citation = _mapping(facts.get("citation"), "citation")
    if external_citation != citation:
        raise ProposalValidationError(
            "top-level citation differs from accuracy-checked external_metadata.citation"
        )


def _validate_claim_support(
    facts: Mapping[str, Any], evidence: Mapping[str, Any], required_claims: Iterable[str]
) -> None:
    """Verify that each evidence label is substantively bound to its proposed value.

    Parameters
    ----------
    facts:
        Complete proposed fact tree.
    evidence:
        Literal evidence block already verified against source bytes.
    required_claims:
        Claim paths requiring deterministic content support.

    Raises
    ------
    ProposalValidationError
        If a required label has no excerpt whose text supports the proposed value.
    """

    excerpts = evidence.get("excerpts")
    if not isinstance(excerpts, list):
        raise ProposalValidationError("evidence.excerpts must be a list")
    support_texts: dict[str, list[str]] = {}
    for excerpt in excerpts:
        if not isinstance(excerpt, Mapping):
            continue
        text = excerpt.get("text")
        supports = excerpt.get("supports")
        if not isinstance(text, str) or not isinstance(supports, list):
            continue
        for support in supports:
            if isinstance(support, str):
                canonical = _SUPPORT_ALIASES.get(support, support)
                support_texts.setdefault(canonical, []).append(text)

    unsupported: list[str] = []
    for claim in required_claims:
        canonical = _SUPPORT_ALIASES.get(claim, claim)
        texts = support_texts.get(canonical, [])
        value = _claim_value(facts, canonical)
        if not texts or not _text_supports_claim(canonical, value, "\n".join(texts)):
            unsupported.append(canonical)
    if unsupported:
        raise ProposalValidationError(
            "evidence excerpts do not substantively support claimed values: "
            f"{sorted(set(unsupported))}"
        )


def _claim_value(facts: Mapping[str, Any], claim: str) -> object:
    """Resolve a supported claim path into the proposed fact tree.

    Parameters
    ----------
    facts:
        Proposed fact tree.
    claim:
        Dot-separated claim path or supported aggregate category.

    Returns
    -------
    object
        Proposed value for deterministic excerpt comparison.

    Raises
    ------
    ProposalValidationError
        If the claim does not name a proposed value.
    """

    value: object = facts
    for part in claim.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ProposalValidationError(f"gated evidence claim does not name a fact: {claim}")
        value = value[part]
    return value


def _text_supports_claim(claim: str, value: object, text: str) -> bool:
    """Return whether literal excerpt text supports a proposed claim value.

    Parameters
    ----------
    claim:
        Canonical claim path.
    value:
        Proposed value at that path.
    text:
        Combined literal excerpts explicitly bound to the claim.

    Returns
    -------
    bool
        True when deterministic value-bearing tokens occur in the excerpt text.
    """

    normalized_text = _normalize_support_text(text)
    if claim == "source_resolution.rung":
        rung_terms = {
            SourceRung.LIBRARY.value: ("library", "package", "registry", "official"),
            SourceRung.VENDOR.value: ("upstream", "repository", "official", "source code"),
            SourceRung.PORT.value: ("port", "translation", "source code"),
            SourceRung.REIMPLEMENT.value: ("layer", "equation", "architecture", "forward"),
            SourceRung.SKIP.value: ("insufficient", "unavailable", "not found", "search"),
        }
        return any(term in normalized_text for term in rung_terms.get(str(value), ()))
    if value is None or value == []:
        return True
    if claim.endswith((".description", ".key_contribution")):
        expected = _significant_tokens(str(value))
        overlap = expected & set(normalized_text.split())
        return len(overlap) >= min(2, len(expected)) if expected else False
    if claim == "taxonomy":
        family = value.get("family") if isinstance(value, Mapping) else None
        if isinstance(family, str) and _normalize_support_text(family) not in normalized_text:
            return False
        return _matches_any_scalar(value, normalized_text, excluded={family})
    if claim == "input_contract":
        if not isinstance(value, Mapping):
            return False
        semantic_values = [
            value.get("semantic_description"),
            value.get("expected_output_semantics"),
            *(
                item.get("semantic_role")
                for key in ("args", "kwargs", "non_tensor_values")
                for item in value.get(key, [])
                if isinstance(item, Mapping)
            ),
        ]
        return _matches_any_scalar(semantic_values, normalized_text)
    if claim.endswith(".citation") and isinstance(value, Mapping):
        title = value.get("title")
        year = value.get("year")
        return _scalar_matches(title, normalized_text) and _scalar_matches(year, normalized_text)
    if claim.endswith(".modes") and isinstance(value, Mapping):
        return _matches_any_scalar(value.get("meaningful_modes"), normalized_text)
    scalars = _positive_scalars(value)
    return all(_scalar_matches(scalar, normalized_text) for scalar in scalars)


def _matches_any_scalar(
    value: object, normalized_text: str, *, excluded: set[object] | None = None
) -> bool:
    """Return whether any positive scalar value occurs in normalized excerpt text.

    Parameters
    ----------
    value:
        Nested value whose positive scalar leaves are candidates.
    normalized_text:
        Lowercase whitespace-normalized excerpt text.
    excluded:
        Optional scalar values not considered for the match.

    Returns
    -------
    bool
        True when at least one candidate scalar is represented.
    """

    excluded_values = excluded or set()
    scalars = [scalar for scalar in _positive_scalars(value) if scalar not in excluded_values]
    return not scalars or any(_scalar_matches(scalar, normalized_text) for scalar in scalars)


def _positive_scalars(value: object) -> list[object]:
    """Flatten positive JSON-like leaves used for evidence comparison.

    Parameters
    ----------
    value:
        Proposed scalar, sequence, or mapping.

    Returns
    -------
    list[object]
        Non-null, non-empty string and numeric leaves.
    """

    if isinstance(value, Mapping):
        return [scalar for child in value.values() for scalar in _positive_scalars(child)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [scalar for child in value for scalar in _positive_scalars(child)]
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return [value] if str(value).strip() else []
    return []


def _scalar_matches(value: object, normalized_text: str) -> bool:
    """Return whether one proposed scalar is represented in excerpt text.

    Parameters
    ----------
    value:
        Proposed scalar value.
    normalized_text:
        Lowercase whitespace-normalized excerpt text.

    Returns
    -------
    bool
        True when the scalar or its significant tokens occur.
    """

    if value is None:
        return True
    normalized_value = _normalize_support_text(str(value))
    if not normalized_value:
        return True
    if f" {normalized_value} " in f" {normalized_text} ":
        return True
    tokens = _significant_tokens(normalized_value)
    return bool(tokens) and tokens <= set(normalized_text.split())


def _significant_tokens(value: str) -> set[str]:
    """Return distinctive lowercase tokens suitable for evidence matching.

    Parameters
    ----------
    value:
        Proposed or excerpt text.

    Returns
    -------
    set[str]
        Tokens longer than three characters after generic-word removal.
    """

    return {
        token
        for token in _normalize_support_text(value).split()
        if (len(token) > 3 or token.isdigit()) and token not in _SUPPORT_STOPWORDS
    }


def _normalize_support_text(value: str) -> str:
    """Normalize text for conservative deterministic token comparison.

    Parameters
    ----------
    value:
        Text to normalize.

    Returns
    -------
    str
        Lowercase alphanumeric tokens separated by single spaces.
    """

    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _validate_code(
    implementation: Mapping[str, Any], rung: SourceRung, allowed_dir: Path
) -> Optional[Path]:
    """Validate staged typed code, path isolation, and forbidden execution APIs.

    Parameters
    ----------
    implementation:
        Proposed implementation block.
    rung:
        Selected source rung.
    allowed_dir:
        Resolved model sandbox directory.

    Returns
    -------
    pathlib.Path | None
        Resolved code path for typed-code rungs.

    Raises
    ------
    ProposalValidationError
        If code is missing, outside the sandbox, untyped, or unsafe.
    """

    code_value = implementation.get("code_path")
    if rung in {SourceRung.LIBRARY, SourceRung.SKIP}:
        if rung is SourceRung.LIBRARY and code_value is not None:
            raise ProposalValidationError(
                "R1_LIBRARY must use a declarative recipe, not staged code"
            )
        return None
    if not isinstance(code_value, str) or not code_value.strip():
        raise ProposalValidationError(f"{rung.value} requires a staged code_path")
    candidate = Path(code_value)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (allowed_dir / candidate).resolve()
    )
    if not resolved.is_relative_to(allowed_dir):
        raise ProposalValidationError("implementation.code_path escapes the model sandbox")
    try:
        code = resolved.read_bytes()
    except OSError as exc:
        raise ProposalValidationError(f"cannot read staged code_path {resolved}: {exc}") from exc
    expected_hash = implementation.get("code_sha256")
    if expected_hash != hash_bytes(code):
        raise ProposalValidationError("implementation.code_sha256 does not match staged bytes")
    try:
        tree = ast.parse(code.decode("utf-8"), filename=str(resolved))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ProposalValidationError(f"staged code is not valid UTF-8 Python: {exc}") from exc
    _validate_typed_functions(tree)
    _validate_calls_and_writes(tree, allowed_dir)
    return resolved


def _validate_typed_functions(tree: ast.AST) -> None:
    """Require annotations on every staged function and method.

    Parameters
    ----------
    tree:
        Parsed staged Python module.

    Raises
    ------
    ProposalValidationError
        If a function argument or return is untyped.
    """

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        arguments = [argument for argument in arguments if argument.arg not in {"self", "cls"}]
        if node.args.vararg is not None:
            arguments.append(node.args.vararg)
        if node.args.kwarg is not None:
            arguments.append(node.args.kwarg)
        if node.returns is None or any(argument.annotation is None for argument in arguments):
            raise ProposalValidationError(f"staged function {node.name!r} must be fully typed")


def _validate_calls_and_writes(tree: ast.AST, allowed_dir: Path) -> None:
    """Reject dynamic execution and statically unsafe writes.

    Parameters
    ----------
    tree:
        Parsed staged Python module.
    allowed_dir:
        Resolved write sandbox.

    Raises
    ------
    ProposalValidationError
        If forbidden execution or an unsafe write is found.
    """

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node.func)
        if call_name in _FORBIDDEN_CALLS or call_name.rsplit(".", 1)[-1] in _FORBIDDEN_CALLS:
            raise ProposalValidationError(f"forbidden dynamic execution call: {call_name}")
        if call_name == "open" and _open_writes(node):
            _validate_literal_write_target(node, allowed_dir, call_name)
        elif call_name.rsplit(".", 1)[-1] in _WRITE_METHODS:
            _validate_literal_write_target(node, allowed_dir, call_name)


def _call_name(function: ast.expr) -> str:
    """Return a dotted static call name when available.

    Parameters
    ----------
    function:
        Call target expression.

    Returns
    -------
    str
        Dotted call name or an empty string.
    """

    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        prefix = _call_name(function.value)
        return f"{prefix}.{function.attr}" if prefix else function.attr
    if isinstance(function, ast.Call):
        return _call_name(function.func)
    return ""


def _open_writes(node: ast.Call) -> bool:
    """Return whether a built-in ``open`` call may write.

    Parameters
    ----------
    node:
        Static ``open`` call.

    Returns
    -------
    bool
        True for write/append/create/update modes or dynamic mode values.
    """

    mode_node: Optional[ast.expr] = node.args[1] if len(node.args) > 1 else None
    for keyword in node.keywords:
        if keyword.arg == "mode":
            mode_node = keyword.value
    if mode_node is None:
        return False
    if not isinstance(mode_node, ast.Constant) or not isinstance(mode_node.value, str):
        return True
    return any(character in mode_node.value for character in "wax+")


def _validate_literal_write_target(node: ast.Call, allowed_dir: Path, call_name: str) -> None:
    """Require write targets to be literal paths inside the model sandbox.

    Parameters
    ----------
    node:
        Static write call.
    allowed_dir:
        Resolved model sandbox.
    call_name:
        Call name used in error reporting.

    Raises
    ------
    ProposalValidationError
        If the target is dynamic or outside the sandbox.
    """

    target_node: Optional[ast.expr] = None
    if call_name == "open" and node.args:
        target_node = node.args[0]
    elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call):
        path_call = node.func.value
        if _call_name(path_call.func).endswith("Path") and path_call.args:
            target_node = path_call.args[0]
    if not isinstance(target_node, ast.Constant) or not isinstance(target_node.value, str):
        raise ProposalValidationError(f"{call_name} has a dynamic or unverifiable write target")
    candidate = Path(target_node.value)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (allowed_dir / candidate).resolve()
    )
    if not resolved.is_relative_to(allowed_dir):
        raise ProposalValidationError(f"{call_name} writes outside the model sandbox")


def _validate_source_ladder(
    rung: SourceRung,
    resolution: Mapping[str, Any],
    implementation: Mapping[str, Any],
    evidence: Mapping[str, Any],
    known_evidence: frozenset[str],
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    cas_root: Union[str, Path, None],
) -> None:
    """Enforce rung-specific source-ladder honesty.

    Parameters
    ----------
    rung:
        Selected rung.
    resolution, implementation, evidence:
        Proposal source and implementation blocks.
    known_evidence:
        Valid literal evidence identifiers.
    source_manifest:
        Exact fetched sources.
    cas_root:
        Optional source CAS root for source inventory inspection.

    Raises
    ------
    ProposalValidationError
        If the chosen rung contradicts its evidence or requirements.
    """

    attempted = resolution.get("attempted_rungs")
    if not isinstance(attempted, list) or not attempted:
        raise ProposalValidationError("source ladder must record attempted rungs")
    attempted_values = [item.get("rung") for item in attempted if isinstance(item, Mapping)]
    rung_order = [member.value for member in SourceRung]
    selected_index = rung_order.index(rung.value)
    if any(required not in attempted_values for required in rung_order[: selected_index + 1]):
        raise ProposalValidationError("selected rung does not document every higher rung")
    if rung is SourceRung.LIBRARY:
        recipe = implementation.get("library_recipe")
        required = ("distribution", "version", "artifact_sha256", "module", "symbol")
        if implementation.get("recipe_type") != "declarative-library" or not isinstance(
            recipe, Mapping
        ):
            raise ProposalValidationError("R1_LIBRARY requires a declarative library recipe")
        if any(
            not isinstance(recipe.get(field), str) or not str(recipe[field]).strip()
            for field in required
        ):
            raise ProposalValidationError("R1_LIBRARY recipe is incomplete")
        if not recipe.get("pretrained_disable_fields"):
            raise ProposalValidationError("R1_LIBRARY must explicitly disable pretrained fields")
    if rung is SourceRung.REIMPLEMENT and _implementation_source_available(
        source_manifest, cas_root=cas_root
    ):
        raise ProposalValidationError("R4_REIMPLEMENT is forbidden when source code is available")
    if rung is SourceRung.VENDOR:
        _validate_r2_source_binding(implementation, resolution, evidence, source_manifest)
    if rung in {SourceRung.VENDOR, SourceRung.PORT, SourceRung.REIMPLEMENT}:
        source_map = implementation.get("source_to_code_map")
        if not isinstance(source_map, list) or not source_map:
            raise ProposalValidationError(f"{rung.value} requires a material source-to-code map")
        cited = {
            evidence_id
            for item in source_map
            if isinstance(item, Mapping)
            for evidence_id in item.get("evidence_ids", [])
            if isinstance(evidence_id, str)
        }
        if not cited or not cited <= known_evidence:
            raise ProposalValidationError(
                f"{rung.value} source map lacks literal descriptive evidence"
            )
        excerpts = evidence.get("excerpts", [])
        descriptive_ids = {
            item.get("evidence_id")
            for item in excerpts
            if isinstance(item, Mapping)
            and any(
                token in str(support).lower()
                for support in item.get("supports", [])
                for token in ("architecture", "implementation", "input_contract", "fidelity")
            )
        }
        if not cited & descriptive_ids:
            raise ProposalValidationError(f"{rung.value} did not cite transcribed descriptive text")


def _validate_r2_source_binding(
    implementation: Mapping[str, Any],
    resolution: Mapping[str, Any],
    evidence: Mapping[str, Any],
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
) -> None:
    """Bind an R2 adapter to exact mirrored upstream bytes and mapped code.

    Parameters
    ----------
    implementation:
        R2 implementation block.
    resolution:
        Source-resolution block classifying authoritative implementation sources.
    evidence:
        Literal excerpt block already verified against controlled source bytes.
    source_manifest:
        Controlled-fetch manifest containing exact source hashes.

    Raises
    ------
    ProposalValidationError
        If upstream files or source-map rows do not bind to exact fetched bytes.
    """

    sources = _source_manifest_index(source_manifest)
    declared_sources = resolution.get("sources")
    if not isinstance(declared_sources, list):
        raise ProposalValidationError("R2_VENDOR has no declared implementation sources")
    declared_by_id = {
        source.get("source_id"): source
        for source in declared_sources
        if isinstance(source, Mapping) and isinstance(source.get("source_id"), str)
    }
    upstream_files = implementation.get("upstream_files")
    if not isinstance(upstream_files, list) or not upstream_files:
        raise ProposalValidationError("R2_VENDOR requires exact mirrored upstream files")
    upstream_source_ids: set[str] = set()
    for upstream in upstream_files:
        if not isinstance(upstream, Mapping):
            raise ProposalValidationError("R2_VENDOR upstream file binding must be an object")
        source_id = upstream.get("source_id")
        source = sources.get(source_id) if isinstance(source_id, str) else None
        if source is None:
            raise ProposalValidationError("R2_VENDOR upstream file references an unfetched source")
        declared = declared_by_id.get(source_id)
        if not isinstance(declared, Mapping) or declared.get("role") != "implementation":
            raise ProposalValidationError(
                "R2_VENDOR upstream bytes are not classified as implementation source"
            )
        if upstream.get("sha256") != source.get("content_sha256"):
            raise ProposalValidationError(
                "R2_VENDOR upstream file hash does not match exact source bytes"
            )
        if declared.get("content_sha256") != source.get("content_sha256"):
            raise ProposalValidationError(
                "R2_VENDOR declared source does not match controlled source bytes"
            )
        upstream_source_ids.add(str(source_id))
    source_map = implementation.get("source_to_code_map")
    if not isinstance(source_map, list) or not source_map:
        raise ProposalValidationError("R2_VENDOR requires an exact source-to-code map")
    excerpts = evidence.get("excerpts")
    if not isinstance(excerpts, list):
        raise ProposalValidationError("R2_VENDOR source map has no literal evidence")
    excerpt_bindings = {
        (excerpt.get("evidence_id"), excerpt.get("source_id"), excerpt.get("locator"))
        for excerpt in excerpts
        if isinstance(excerpt, Mapping)
    }
    for mapping in source_map:
        if not isinstance(mapping, Mapping):
            raise ProposalValidationError("R2_VENDOR source-to-code binding must be an object")
        source_id = mapping.get("source_id")
        locator = mapping.get("source_locator")
        if not isinstance(source_id, str) or source_id not in upstream_source_ids:
            raise ProposalValidationError(
                "R2_VENDOR source map does not reference bound upstream bytes"
            )
        if not isinstance(locator, str) or not locator.strip():
            raise ProposalValidationError("R2_VENDOR source map lacks an exact source locator")
        mapping_evidence = mapping.get("evidence_ids")
        if not isinstance(mapping_evidence, list) or not any(
            (evidence_id, source_id, locator) in excerpt_bindings
            for evidence_id in mapping_evidence
        ):
            raise ProposalValidationError(
                "R2_VENDOR source map locator is not bound to a verified exact excerpt"
            )


def _source_manifest_index(
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
) -> dict[str, Mapping[str, Any]]:
    """Index exact controlled-fetch rows by source identifier.

    Parameters
    ----------
    source_manifest:
        Manifest wrapper or direct source sequence.

    Returns
    -------
    dict[str, Mapping[str, Any]]
        Exact source rows keyed by source ID.

    Raises
    ------
    ProposalValidationError
        If the manifest is malformed or duplicates an identifier.
    """

    raw_sources: object
    if isinstance(source_manifest, Mapping):
        raw_sources = source_manifest.get("sources")
        if raw_sources is None and "source_id" in source_manifest:
            raw_sources = [source_manifest]
    else:
        raw_sources = source_manifest
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        raise ProposalValidationError("source manifest must contain a source list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for source in raw_sources:
        if not isinstance(source, Mapping):
            raise ProposalValidationError("every source manifest must be an object")
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ProposalValidationError("source manifest row has no source_id")
        if source_id in indexed:
            raise ProposalValidationError(f"duplicate source_id: {source_id}")
        indexed[source_id] = source
    return indexed


def _implementation_source_available(
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    *,
    cas_root: Union[str, Path, None],
) -> bool:
    """Return whether exact fetched CAS bytes expose implementation source.

    Parameters
    ----------
    source_manifest:
        Controlled-fetch manifest wrapper or rows.
    cas_root:
        Optional CAS root for manifests without an explicit object path.

    Returns
    -------
    bool
        True when a fetched object inventory contains usable source code.
    """

    sources = _source_manifest_index(source_manifest)
    return any(
        _source_cas_contains_implementation(source, cas_root=cas_root)
        for source in sources.values()
    )


def _source_cas_contains_implementation(
    source: Mapping[str, Any], *, cas_root: Union[str, Path, None]
) -> bool:
    """Inspect one hash-bound CAS object for usable implementation code.

    Parameters
    ----------
    source:
        Controlled-fetch manifest row.
    cas_root:
        Optional CAS root for manifests without an explicit object path.

    Returns
    -------
    bool
        True when archive names, a byte manifest, or raw source bytes expose code.

    Raises
    ------
    ProposalValidationError
        If fetched bytes are absent or no longer match their declared digest.
    """

    path_value = source.get("cas_path")
    digest = source.get("content_sha256")
    if not isinstance(digest, str):
        raise ProposalValidationError("fetched source manifest has no content_sha256")
    if isinstance(path_value, str) and path_value:
        path = Path(path_value)
    elif cas_root is not None:
        path = source_cas_path(cas_root, digest)
    else:
        raise ProposalValidationError("R4 source inventory has no inspectable CAS path")
    if not _cas_object_matches_digest(path, digest):
        raise ProposalValidationError(f"R4 source inventory CAS object does not match {digest}")
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                return any(
                    not member.is_dir()
                    and member.file_size > 0
                    and _inventory_name_is_implementation(member.filename)
                    for member in archive.infolist()
                )
        if tarfile.is_tarfile(path):
            with tarfile.open(path, mode="r:*") as archive:
                return any(
                    member.isfile()
                    and member.size > 0
                    and _inventory_name_is_implementation(member.name)
                    for member in archive
                )
        with path.open("rb") as handle:
            content = handle.read(_MAX_TEXT_INVENTORY_BYTES + 1)
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise ProposalValidationError(
            f"cannot inventory fetched source CAS object {path}: {exc}"
        ) from exc
    if len(content) > _MAX_TEXT_INVENTORY_BYTES:
        return False
    return _byte_manifest_has_implementation(content) or _raw_python_is_implementation(content)


def _cas_object_matches_digest(path: Path, digest: str) -> bool:
    """Return whether a CAS file exists and matches its prefixed SHA-256 digest.

    Parameters
    ----------
    path:
        Candidate exact CAS object.
    digest:
        Declared prefixed SHA-256 digest.

    Returns
    -------
    bool
        True only for a byte-identical regular file.
    """

    if not path.is_file() or not digest.startswith("sha256:"):
        return False
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024**2), b""):
                hasher.update(chunk)
    except OSError:
        return False
    return f"sha256:{hasher.hexdigest()}" == digest


def _inventory_name_is_implementation(value: str) -> bool:
    """Return whether an archive-internal path denotes usable source code.

    Parameters
    ----------
    value:
        Archive member or byte-manifest path.

    Returns
    -------
    bool
        True for a non-packaging code file outside documentation/test trees.
    """

    normalized = value.replace("\\", "/").strip("/")
    path = Path(normalized)
    lowered_parts = {part.lower() for part in path.parts[:-1]}
    name = path.name.lower()
    return (
        bool(name)
        and name not in _NON_IMPLEMENTATION_CODE_NAMES
        and not lowered_parts & _NON_IMPLEMENTATION_SOURCE_DIRS
        and path.suffix.lower() in _IMPLEMENTATION_SOURCE_SUFFIXES
    )


def _byte_manifest_has_implementation(content: bytes) -> bool:
    """Inspect JSON manifest bytes for embedded source-code paths.

    Parameters
    ----------
    content:
        Small non-archive CAS object bytes.

    Returns
    -------
    bool
        True when any manifest string names a usable code file.
    """

    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False

    def strings(item: object) -> Iterable[str]:
        """Yield every string key and value from a JSON-like manifest.

        Parameters
        ----------
        item:
            Current JSON value.

        Yields
        ------
        str
            String keys and scalar string values.
        """

        if isinstance(item, Mapping):
            for key, child in item.items():
                if isinstance(key, str):
                    yield key
                yield from strings(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for child in item:
                yield from strings(child)
        elif isinstance(item, str):
            yield item

    return any(_inventory_name_is_implementation(value) for value in strings(value))


def _raw_python_is_implementation(content: bytes) -> bool:
    """Recognize extensionless raw Python implementation bytes structurally.

    Parameters
    ----------
    content:
        Small non-archive CAS object bytes.

    Returns
    -------
    bool
        True when parseable Python defines executable model-like structure.
    """

    try:
        tree = ast.parse(content.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError):
        return False
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef))
    ]
    return bool(definitions) and any(
        isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)) for node in tree.body
    )


def _validate_structural_slop(facts: Mapping[str, Any], code_path: Optional[Path]) -> None:
    """Trip on a plain Sequential/MLP staged as a named exotic family.

    Parameters
    ----------
    facts:
        Proposed fact tree.
    code_path:
        Validated staged adapter path, if the selected rung uses one.

    Raises
    ------
    ProposalValidationError
        If staged structure is a generic stand-in for an exotic family claim.
    """

    if code_path is None:
        return
    try:
        tree = ast.parse(code_path.read_text(encoding="utf-8"), filename=str(code_path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise ProposalValidationError(f"cannot inspect staged structure: {exc}") from exc
    defined_classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    module_calls = [
        _call_name(node.func).rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            _call_name(node.func).startswith(("nn.", "torch.nn."))
            or _call_name(node.func).rsplit(".", 1)[-1] in _GENERIC_MODEL_CALLS
        )
        and _call_name(node.func).rsplit(".", 1)[-1] not in defined_classes
    ]
    if not module_calls:
        return
    generic_structure = ("Sequential" in module_calls or module_calls.count("Linear") >= 2) and set(
        module_calls
    ) <= _GENERIC_MODEL_CALLS
    if not generic_structure or not _claims_exotic_family(facts):
        return
    raise ProposalValidationError(
        "structural slop tripwire: generic Sequential/MLP staged as an exotic named family"
    )


def _claims_exotic_family(facts: Mapping[str, Any]) -> bool:
    """Return whether authored identity claims more than a generic MLP family.

    Parameters
    ----------
    facts:
        Proposed fact tree.

    Returns
    -------
    bool
        True when family/name/class claims are not explicitly generic feed-forward terms.
    """

    identity = _mapping(facts.get("identity"), "identity")
    metadata = _mapping(facts.get("external_metadata"), "external_metadata")
    architecture_classes = metadata.get("architecture_class")
    names = [
        identity.get("canonical_name"),
        metadata.get("family"),
        *(architecture_classes if isinstance(architecture_classes, list) else []),
    ]
    normalized = {
        _normalize_support_text(str(name))
        for name in names
        if isinstance(name, str) and name.strip()
    }
    return bool(normalized) and not normalized <= _GENERIC_FAMILY_NAMES


def _validate_anti_slop(facts: Mapping[str, Any]) -> None:
    """Reject explicit approximation language in authored implementation claims.

    Parameters
    ----------
    facts:
        Proposed fact tree.

    Raises
    ------
    ProposalValidationError
        If authored text admits a generic or simplified stand-in.
    """

    texts = _authored_implementation_texts(facts)
    lowered = " ".join(texts).lower()
    matched = sorted(
        {match.group(0) for pattern in _SLOP_PATTERNS if (match := re.search(pattern, lowered))}
    )
    if matched:
        raise ProposalValidationError(
            f"proposal contains forbidden approximation language: {matched}"
        )


def _authored_implementation_texts(facts: Mapping[str, Any]) -> list[str]:
    """Collect authored prose surfaces that can admit approximation or slop.

    Parameters
    ----------
    facts:
        Proposed fact tree.

    Returns
    -------
    list[str]
        Authored descriptions, decisions, rationales, and fidelity notes.
    """

    metadata = _mapping(facts.get("external_metadata"), "external_metadata")
    website = _mapping(facts.get("website"), "website")
    resolution = _mapping(facts.get("source_resolution"), "source_resolution")
    implementation = _mapping(facts.get("implementation"), "implementation")
    fidelity = _mapping(facts.get("fidelity"), "fidelity")
    texts = [
        metadata.get("description"),
        metadata.get("key_contribution"),
        website.get("tagline"),
        website.get("description"),
        website.get("key_contribution"),
        resolution.get("decision"),
        _mapping(resolution.get("search_report"), "source_resolution.search_report").get(
            "conclusion"
        ),
        fidelity.get("reason"),
    ]
    for collection_name in ("patches", "declared_choices"):
        collection = implementation.get(collection_name)
        if isinstance(collection, list):
            texts.extend(item.get("rationale") for item in collection if isinstance(item, Mapping))
    deviations = fidelity.get("deviations")
    if isinstance(deviations, list):
        texts.extend(deviations)
    return [text for text in texts if isinstance(text, str)]


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    """Return a required mapping.

    Parameters
    ----------
    value:
        Candidate object.
    field:
        Field name used in errors.

    Returns
    -------
    Mapping[str, Any]
        Valid mapping.

    Raises
    ------
    ProposalValidationError
        If the value is not an object.
    """

    if not isinstance(value, Mapping):
        raise ProposalValidationError(f"{field} must be an object")
    return value
