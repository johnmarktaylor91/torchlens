"""Canonical identities and exact dependency-staleness propagation."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from urllib.parse import SplitResult, urlsplit, urlunsplit

from menagerie.crawler.constants import STABLE_ID_DIGEST_CHARS
from menagerie.crawler.models import StalenessReport

_HASH_PREFIX = "sha256:"


def utc_now() -> str:
    """Return the current RFC 3339 UTC timestamp.

    Returns
    -------
    str
        Current timestamp ending in ``Z``.
    """

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def is_sha256(value: str) -> bool:
    """Return whether a value is a canonical prefixed SHA-256 digest.

    Parameters
    ----------
    value:
        Candidate digest.

    Returns
    -------
    bool
        Whether the digest uses lowercase ``sha256:<64 hex>`` syntax.
    """

    return (
        len(value) == 71
        and value.startswith(_HASH_PREFIX)
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def fsync_directory(path: Path) -> None:
    """Synchronize a directory after changing one of its entries.

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


def atomic_replace_bytes(path: Path, data: bytes) -> None:
    """Replace a file with fsynced bytes and then fsync its parent directory.

    Parameters
    ----------
    path:
        Destination path.
    data:
        Exact bytes to persist.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value into stable UTF-8 bytes.

    Parameters
    ----------
    value:
        JSON-compatible value.

    Returns
    -------
    bytes
        Deterministic serialization with sorted object keys.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def hash_bytes(data: bytes) -> str:
    """Return a prefixed SHA-256 digest for bytes.

    Parameters
    ----------
    data:
        Bytes to hash.

    Returns
    -------
    str
        ``sha256:`` followed by 64 lowercase hexadecimal characters.
    """

    return f"{_HASH_PREFIX}{hashlib.sha256(data).hexdigest()}"


def stable_hash(value: Any) -> str:
    """Hash a JSON-compatible value using canonical serialization.

    Parameters
    ----------
    value:
        JSON-compatible value.

    Returns
    -------
    str
        Canonical prefixed SHA-256 digest.
    """

    return hash_bytes(canonical_json_bytes(value))


def payload_hash(payload: Mapping[str, Any]) -> str:
    """Hash a ledger payload without its self-referential hash fields.

    Parameters
    ----------
    payload:
        Ledger payload.

    Returns
    -------
    str
        Canonical payload digest.
    """

    unhashed = {
        key: value
        for key, value in payload.items()
        if key not in {"payload_sha256", "record_revision"}
    }
    return stable_hash(unhashed)


def assign_stable_id(namespace: str, natural_key: str) -> tuple[str, str]:
    """Assign a deterministic stable ID and retain its full digest.

    Parameters
    ----------
    namespace:
        Immutable identity namespace.
    natural_key:
        Immutable normalized natural key.

    Returns
    -------
    tuple[str, str]
        ``m_`` ID using the first 20 base32 characters and full SHA-256 digest.
    """

    digest = hashlib.sha256(canonical_json_bytes([namespace, natural_key])).digest()
    base32 = base64.b32encode(digest).decode("ascii").lower().rstrip("=")
    return f"m_{base32[:STABLE_ID_DIGEST_CHARS]}", f"{_HASH_PREFIX}{digest.hex()}"


def normalize_url(url: str) -> str:
    """Normalize a source URL without changing path/query semantics.

    Parameters
    ----------
    url:
        Absolute public source URL.

    Returns
    -------
    str
        URL with lowercase scheme/host, no fragment, and normalized default port.
    """

    split = urlsplit(url.strip())
    hostname = (split.hostname or "").lower()
    port = split.port
    if port is not None and not (
        (split.scheme.lower() == "http" and port == 80)
        or (split.scheme.lower() == "https" and port == 443)
    ):
        hostname = f"{hostname}:{port}"
    normalized = SplitResult(
        split.scheme.lower(),
        hostname,
        split.path or "/",
        split.query,
        "",
    )
    return urlunsplit(normalized)


def compute_source_identity(
    sources: Sequence[Mapping[str, Any]], search_report: Mapping[str, Any]
) -> str:
    """Compute source identity over normalized sources and search report.

    Parameters
    ----------
    sources:
        Source rows containing URLs, revisions, locators, and content hashes.
    search_report:
        Complete source-search report.

    Returns
    -------
    str
        Source identity.
    """

    normalized_sources = []
    for source in sources:
        selected = {
            "url": normalize_url(str(source["url"])),
            "revision_kind": source.get("revision_kind"),
            "revision": source.get("revision"),
            "locator": source.get("locator"),
            "content_sha256": source.get("content_sha256"),
        }
        normalized_sources.append(selected)
    normalized_sources.sort(key=lambda item: canonical_json_bytes(item))
    return stable_hash({"sources": normalized_sources, "search_report": search_report})


def compute_evidence_identity(excerpts: Sequence[Mapping[str, Any]]) -> str:
    """Compute ordered literal-evidence identity.

    Parameters
    ----------
    excerpts:
        Ordered excerpts with locators, supports, and content hashes.

    Returns
    -------
    str
        Evidence identity.
    """

    selected = [
        {
            "source_id": excerpt.get("source_id"),
            "locator": excerpt.get("locator"),
            "text": excerpt.get("text"),
            "text_sha256": excerpt.get("text_sha256"),
            "supports": excerpt.get("supports"),
            "source_content_sha256": excerpt.get("source_content_sha256"),
        }
        for excerpt in excerpts
    ]
    return stable_hash(selected)


def compute_recipe_revision(
    recipe: Mapping[str, Any], source_identity: str, *, adapter_bytes: Optional[bytes] = None
) -> str:
    """Compute a recipe revision bound to exact source identity.

    Parameters
    ----------
    recipe:
        Declarative constructor/adapter choices and policies.
    source_identity:
        Current source identity.
    adapter_bytes:
        Optional exact typed-adapter module bytes.

    Returns
    -------
    str
        Recipe revision.
    """

    return stable_hash(
        {
            "recipe": recipe,
            "adapter_sha256": hash_bytes(adapter_bytes) if adapter_bytes is not None else None,
            "source_identity": source_identity,
        }
    )


def compute_env_generation(
    intent: Mapping[str, Any],
    exact_lock: Any,
    resolved_export: Any,
    platform_facts: Mapping[str, Any],
    probe_results: Sequence[Mapping[str, Any]],
) -> str:
    """Compute an exact target environment generation.

    Parameters
    ----------
    intent:
        Environment intent.
    exact_lock:
        Exact lock data including artifact hashes.
    resolved_export:
        Actual resolved export.
    platform_facts:
        Compiler and SDK facts.
    probe_results:
        Ordered probe results.

    Returns
    -------
    str
        Environment generation.
    """

    return stable_hash(
        {
            "intent": intent,
            "exact_lock": exact_lock,
            "resolved_export": resolved_export,
            "platform_facts": platform_facts,
            "probe_results": probe_results,
        }
    )


def compute_fidelity_identity(
    *,
    source_identity: str,
    evidence_identity: str,
    implementation_hash: Optional[str],
    source_to_code_map: Sequence[Mapping[str, Any]],
    prompt_hash: str,
    checker_model: str,
    checker_version: str,
) -> str:
    """Compute a fidelity-gate identity.

    Parameters
    ----------
    source_identity, evidence_identity:
        Current source and evidence identities.
    implementation_hash:
        Exact accepted implementation/code hash.
    source_to_code_map:
        Material source-to-code mapping.
    prompt_hash:
        Frozen fidelity prompt hash.
    checker_model, checker_version:
        Exact checker identity.

    Returns
    -------
    str
        Fidelity identity.
    """

    return stable_hash(locals())


def compute_vet_identity(
    *,
    authored_metadata: Mapping[str, Any],
    evidence_references: Mapping[str, Any],
    source_identity: str,
    evidence_identity: str,
    prompt_hash: str,
    checker_model: str,
    checker_version: str,
) -> str:
    """Compute an authored-metadata vet identity.

    Parameters
    ----------
    authored_metadata:
        Every authored metadata field.
    evidence_references:
        Field-to-evidence references.
    source_identity, evidence_identity:
        Current source and evidence identities.
    prompt_hash:
        Frozen vet prompt hash.
    checker_model, checker_version:
        Exact checker identity.

    Returns
    -------
    str
        Vet identity.
    """

    return stable_hash(locals())


def compute_execution_identity(
    *,
    stable_id: str,
    recipe_revision: str,
    env_generation: str,
    runner_version: str,
    target: str,
    machine_class: str,
    seed_policy: Mapping[str, Any],
    framework_adapter: Mapping[str, Any],
    device: str,
) -> str:
    """Compute a controlled execution identity.

    Parameters
    ----------
    stable_id, recipe_revision, env_generation:
        Model, recipe, and environment identities.
    runner_version, target, machine_class:
        Exact runner and platform facts.
    seed_policy, framework_adapter, device:
        Execution policy inputs.

    Returns
    -------
    str
        Execution identity.
    """

    return stable_hash(locals())


def stale_dependencies(changed_inputs: Iterable[str]) -> StalenessReport:
    """Return dependent products stale for named byte-level input changes.

    Parameters
    ----------
    changed_inputs:
        Names such as source, evidence, code, lock, runner, or checker_prompt.

    Returns
    -------
    StalenessReport
        Transitive dependency invalidations.
    """

    graph = {
        "source": {"source", "recipe", "fidelity", "vet"},
        "evidence": {"evidence", "fidelity", "vet"},
        "code": {"recipe", "fidelity"},
        "source_to_code_map": {"fidelity"},
        "authored_metadata": {"vet"},
        "lock": {"environment"},
        "resolved_export": {"environment"},
        "probe": {"environment"},
        "runner": {"execution"},
        "adapter": {"recipe", "execution"},
        "input": {"recipe", "execution"},
        "author_prompt": {"vet"},
        "checker_prompt": {"fidelity", "vet"},
    }
    stale: set[str] = set()
    for changed in changed_inputs:
        stale.update(graph.get(changed, {changed}))
    if stale & {"recipe", "environment"}:
        stale.add("execution")
    if "execution" in stale:
        stale.update({"attempts", "run_status"})
    if "fidelity" in stale:
        stale.add("fidelity_verdict")
    if "vet" in stale:
        stale.add("accuracy_gate")
    return StalenessReport(frozenset(stale))
