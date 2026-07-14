"""Staged author-proposal validation and deterministic anti-slop gates."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from menagerie.crawler.constants import AUTHOR_PROPOSAL_SCHEMA_VERSION, SourceRung
from menagerie.crawler.evidence import EvidenceValidationError, evidence_ids, validate_evidence
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.schema import PayloadValidationError, validate_payload

DEFAULT_GATED_CLAIMS = frozenset(
    {
        "external_metadata.description",
        "source_resolution.rung",
        "taxonomy",
        "input_contract",
        "license",
        "year",
        "country",
    }
)
_FORBIDDEN_CALLS = frozenset({"eval", "exec", "compile"})
_SLOP_TERMS = frozenset(
    {
        "compact stand-in",
        "generic stand-in",
        "simplified substitute",
        "representative approximation",
    }
)
_WRITE_METHODS = frozenset(
    {"write_text", "write_bytes", "touch", "mkdir", "rename", "replace", "unlink", "rmdir"}
)


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
        claims.add("citation")
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
    known_evidence = evidence_ids(evidence)
    _validate_citation(facts, known_evidence)
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
    )
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
        resolution, source_manifest
    ):
        raise ProposalValidationError("R4_REIMPLEMENT is forbidden when source code is available")
    if rung in {SourceRung.PORT, SourceRung.REIMPLEMENT}:
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


def _implementation_source_available(
    resolution: Mapping[str, Any],
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
) -> bool:
    """Return whether fetched evidence exposes usable implementation source.

    Parameters
    ----------
    resolution:
        Proposal source-resolution block.
    source_manifest:
        Controlled-fetch manifest wrapper or rows.

    Returns
    -------
    bool
        True when any exact fetched source is classified as implementation.
    """

    candidates: list[Mapping[str, Any]] = []
    resolution_sources = resolution.get("sources", [])
    if isinstance(resolution_sources, list):
        candidates.extend(item for item in resolution_sources if isinstance(item, Mapping))
    if isinstance(source_manifest, Mapping):
        manifest_sources = source_manifest.get("sources", [source_manifest])
    else:
        manifest_sources = source_manifest
    if isinstance(manifest_sources, Sequence) and not isinstance(manifest_sources, (str, bytes)):
        candidates.extend(item for item in manifest_sources if isinstance(item, Mapping))
    return any(
        source.get("role") == "implementation"
        or source.get("source_code_available") is True
        or source.get("content_kind") == "source-code"
        for source in candidates
    )


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

    metadata = _mapping(facts.get("external_metadata"), "external_metadata")
    website = _mapping(facts.get("website"), "website")
    texts = [
        str(metadata.get("description", "")),
        str(website.get("description", "")),
        str(_mapping(facts.get("source_resolution"), "source_resolution").get("decision", "")),
    ]
    lowered = " ".join(texts).lower()
    matched = sorted(term for term in _SLOP_TERMS if term in lowered)
    if matched:
        raise ProposalValidationError(
            f"proposal contains forbidden approximation language: {matched}"
        )


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
