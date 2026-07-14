"""Deterministic one-model author request and result-envelope handling."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Union

from menagerie.crawler.constants import AUTHOR_PROPOSAL_SCHEMA_VERSION, AUTHOR_PROMPT_NAME
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.models import JsonObject
from menagerie.crawler.proposal import ProposalValidationReport, validate_author_proposal

PROMPT_PATH = Path(__file__).with_name("prompts") / f"{AUTHOR_PROMPT_NAME}.txt"


class AuthorDispatchError(ValueError):
    """Raised when an author envelope or result does not bind completely."""


def build_author_envelope(
    *,
    work_id: str,
    stable_id: str,
    untrusted_hints: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    allowed_model_dir: Union[str, Path],
    output_path: Union[str, Path],
    author_model: str,
    author_version: str,
) -> JsonObject:
    """Build one deterministic rich-session author task packet.

    Parameters
    ----------
    work_id, stable_id:
        Exact work and model identities.
    untrusted_hints:
        Preserved legacy/discovery hints, explicitly marked untrusted.
    source_manifest:
        Frozen controlled-fetch manifest.
    allowed_model_dir:
        Staged-code path sandbox for this model.
    output_path:
        Exact atomic result path the author must write.
    author_model, author_version:
        Expected author identity.

    Returns
    -------
    dict[str, Any]
        Hash-bound author envelope.
    """

    prompt_sha256 = hash_bytes(_read_prompt())
    source_manifest_sha256 = str(
        source_manifest.get("manifest_sha256") or stable_hash(source_manifest.get("sources", []))
    )
    body: JsonObject = {
        "envelope_version": "menagerie.crawler.author-envelope.v2",
        "work_id": work_id,
        "stable_id": stable_id,
        "untrusted_hints": dict(untrusted_hints),
        "legacy_claims_untrusted": True,
        "source_manifest": dict(source_manifest),
        "source_manifest_sha256": source_manifest_sha256,
        "prompt": {
            "name": AUTHOR_PROMPT_NAME,
            "path": str(PROMPT_PATH),
            "sha256": prompt_sha256,
        },
        "author": {"provider": "anthropic", "model": author_model, "version": author_version},
        "allowed_model_dir": str(Path(allowed_model_dir).resolve()),
        "allowed_output_root": str(Path(output_path).resolve().parent),
        "required_output_path": str(Path(output_path).resolve()),
        "required_result_schema": AUTHOR_PROPOSAL_SCHEMA_VERSION,
    }
    return {**body, "envelope_sha256": stable_hash(body)}


def validate_author_result(
    result_path: Union[str, Path],
    envelope: Mapping[str, Any],
    *,
    cas_root: Union[str, Path, None] = None,
) -> tuple[JsonObject, ProposalValidationReport]:
    """Validate a complete atomic author result against its exact request.

    Parameters
    ----------
    result_path:
        Expected final ``result.json`` path.
    envelope:
        Author envelope returned by :func:`build_author_envelope`.
    cas_root:
        Optional source CAS root used for evidence verification.

    Returns
    -------
    tuple[dict[str, Any], ProposalValidationReport]
        Parsed proposal and deterministic validation report.

    Raises
    ------
    AuthorDispatchError
        If the file is partial, misplaced, mismatched, or invalid.
    """

    _validate_envelope_hash(envelope)
    path = Path(result_path).resolve()
    expected_path = Path(str(envelope.get("required_output_path"))).resolve()
    if path != expected_path or path.name != "result.json":
        raise AuthorDispatchError("author result is not at the envelope's exact atomic path")
    proposal = _read_json_object(path)
    for field in ("work_id", "stable_id"):
        if proposal.get(field) != envelope.get(field):
            raise AuthorDispatchError(f"author result {field} does not match its envelope")
    author = proposal.get("author")
    expected_author = envelope.get("author")
    if not isinstance(author, Mapping) or not isinstance(expected_author, Mapping):
        raise AuthorDispatchError("author identity is missing")
    for field in ("provider", "model", "version"):
        if author.get(field) != expected_author.get(field):
            raise AuthorDispatchError(f"author result {field} does not match its envelope")
    prompt = envelope.get("prompt")
    if not isinstance(prompt, Mapping) or author.get("prompt_sha256") != prompt.get("sha256"):
        raise AuthorDispatchError("author prompt hash does not match its envelope")
    verified_hashes = proposal.get("verified_hashes")
    if not isinstance(verified_hashes, Mapping) or verified_hashes.get(
        "source_manifest"
    ) != envelope.get("source_manifest_sha256"):
        raise AuthorDispatchError("proposal source manifest hash does not match its envelope")
    expected_proposal_hash = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    if proposal.get("proposal_sha256") != expected_proposal_hash:
        raise AuthorDispatchError("proposal_sha256 does not bind the complete result")
    try:
        report = validate_author_proposal(
            proposal,
            allowed_model_dir=str(envelope["allowed_model_dir"]),
            source_manifest=_required_mapping(envelope.get("source_manifest"), "source_manifest"),
            cas_root=cas_root,
        )
    except ValueError as exc:
        raise AuthorDispatchError(str(exc)) from exc
    return proposal, report


def write_envelope_atomic(envelope: Mapping[str, Any], path: Union[str, Path]) -> Path:
    """Atomically persist one deterministic work envelope.

    Parameters
    ----------
    envelope:
        Complete envelope object.
    path:
        Destination path.

    Returns
    -------
    pathlib.Path
        Published envelope path.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    data = json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _read_prompt() -> bytes:
    """Read the exact frozen author prompt bytes.

    Returns
    -------
    bytes
        Frozen prompt bytes.

    Raises
    ------
    AuthorDispatchError
        If the frozen prompt is absent.
    """

    try:
        return PROMPT_PATH.read_bytes()
    except OSError as exc:
        raise AuthorDispatchError(f"cannot read frozen author prompt: {exc}") from exc


def _validate_envelope_hash(envelope: Mapping[str, Any]) -> None:
    """Validate an author envelope's self-hash.

    Parameters
    ----------
    envelope:
        Candidate envelope.

    Raises
    ------
    AuthorDispatchError
        If its canonical hash is stale or absent.
    """

    expected = stable_hash(
        {key: value for key, value in envelope.items() if key != "envelope_sha256"}
    )
    if envelope.get("envelope_sha256") != expected:
        raise AuthorDispatchError("author envelope hash mismatch")


def _read_json_object(path: Path) -> JsonObject:
    """Read one complete UTF-8 JSON object.

    Parameters
    ----------
    path:
        Final result path.

    Returns
    -------
    dict[str, Any]
        Parsed object.

    Raises
    ------
    AuthorDispatchError
        If the final file is absent, partial, or not one object.
    """

    try:
        raw = path.read_bytes()
        if not raw or path.is_symlink():
            raise AuthorDispatchError("author result must be a non-empty regular file")
        parsed = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorDispatchError(f"partial or invalid author result: {exc}") from exc
    if not isinstance(parsed, dict):
        raise AuthorDispatchError("author result must contain exactly one JSON object")
    return parsed


def _required_mapping(value: object, field: str) -> Mapping[str, Any]:
    """Return a required mapping from an envelope.

    Parameters
    ----------
    value:
        Candidate mapping.
    field:
        Field name used in errors.

    Returns
    -------
    Mapping[str, Any]
        Valid mapping.

    Raises
    ------
    AuthorDispatchError
        If the field is absent or not an object.
    """

    if not isinstance(value, Mapping):
        raise AuthorDispatchError(f"author envelope {field} must be an object")
    return value


def _fsync_directory(path: Path) -> None:
    """Synchronize a directory after atomic envelope publication.

    Parameters
    ----------
    path:
        Directory to synchronize.
    """

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
