"""Closed shapes of the optional manifest summaries (``model_fingerprint``,
``backward_summary``). The 2.x unified manifest carried both; a 3.0 writer may. When one is
present the reader checks it against the tables in :mod:`.contract` and refuses
``manifest_schema_invalid`` on a violation; absent or null summaries pass. Ruled 2 Nov 2026.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Final

from . import contract

__all__ = ["summary_schema_errors"]

_SHA256_HEX: Final = re.compile(r"^[0-9a-f]{64}$")
#: The closed key sets: a summary carries exactly these keys and no other.
_FINGERPRINT_FIELDS: Final = (
    *contract.MODEL_FINGERPRINT_HASH_FIELDS,
    "class_qualname",
    "extra_metadata",
)
_BACKEND_FINGERPRINT_FIELDS: Final = ("backend_fingerprint",)
_BACKWARD_SUMMARY_FIELDS: Final = (
    "has_backward_pass",
    *contract.BACKWARD_SUMMARY_COUNT_FIELDS,
    "gradient_blob_kinds",
    "old_bundle_policy",
)


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _unsupported(name: str, value: Mapping[str, Any], allowed: tuple[str, ...]) -> list[str]:
    """The closed-shape violation for keys of ``value`` outside ``allowed``, if any."""

    extra = sorted(str(key) for key in set(value) - set(allowed))
    return [f"{name} has unsupported fields {extra}"] if extra else []


def summary_schema_errors(manifest: Mapping[str, Any]) -> list[str]:
    """Violations of the closed shapes of the optional ``model_fingerprint`` and
    ``backward_summary`` summaries; an absent (or null) summary is fine."""

    errors: list[str] = []
    fingerprint = manifest.get("model_fingerprint")
    if fingerprint is not None:
        errors.extend(_fingerprint_errors(fingerprint))
    summary = manifest.get("backward_summary")
    if summary is not None:
        errors.extend(_backward_summary_errors(summary))
    backend = _fingerprint_backend(fingerprint)
    if backend is not None and backend != "torch" and summary is not None:
        errors.append(f"backward_summary must be null for a {backend!r} backend artifact")
    return errors


def _fingerprint_backend(fingerprint: Any) -> str | None:
    """The backend a backend-shaped fingerprint names, else ``None``."""

    inner = fingerprint.get("backend_fingerprint") if isinstance(fingerprint, Mapping) else None
    backend = inner.get("backend") if isinstance(inner, Mapping) else None
    return backend if isinstance(backend, str) else None


def _fingerprint_errors(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["model_fingerprint must be an object"]
    if "backend_fingerprint" in value:
        inner = value["backend_fingerprint"]
        if not isinstance(inner, Mapping) or set(inner) != set(contract.BACKEND_FINGERPRINT_FIELDS):
            return [
                "model_fingerprint.backend_fingerprint must be an object with exactly "
                f"{contract.BACKEND_FINGERPRINT_FIELDS}"
            ]
        if not isinstance(inner["backend"], str) or not inner["backend"]:
            return ["model_fingerprint.backend_fingerprint.backend must be a non-empty string"]
        return _unsupported("model_fingerprint", value, _BACKEND_FINGERPRINT_FIELDS)
    errors = _unsupported("model_fingerprint", value, _FINGERPRINT_FIELDS)
    errors += [
        f"model_fingerprint.{name} must be a SHA-256 hex digest"
        for name in contract.MODEL_FINGERPRINT_HASH_FIELDS
        if not isinstance(value.get(name), str) or _SHA256_HEX.match(value[name]) is None
    ]
    qualname = value.get("class_qualname")
    if not isinstance(qualname, str) or not qualname:
        errors.append("model_fingerprint.class_qualname must be a non-empty string")
    if not isinstance(value.get("extra_metadata"), Mapping):
        errors.append("model_fingerprint.extra_metadata must be an object")
    return errors


def _backward_summary_errors(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["backward_summary must be an object"]
    errors = _unsupported("backward_summary", value, _BACKWARD_SUMMARY_FIELDS)
    if not isinstance(value.get("has_backward_pass"), bool):
        errors.append("backward_summary.has_backward_pass must be a bool")
    for name in contract.BACKWARD_SUMMARY_COUNT_FIELDS:
        count = value.get(name)
        if not _is_count(count):
            errors.append(f"backward_summary.{name} must be a non-negative int")
    kinds = value.get("gradient_blob_kinds")
    if (
        not isinstance(kinds, list)
        or len(set(kinds)) != len(kinds)
        or any(kind not in contract.GRADIENT_BLOB_KINDS for kind in kinds)
    ):
        errors.append(
            "backward_summary.gradient_blob_kinds must list distinct kinds from "
            f"{contract.GRADIENT_BLOB_KINDS}"
        )
    policy = value.get("old_bundle_policy")
    if not isinstance(policy, str) or not policy:
        errors.append("backward_summary.old_bundle_policy must be a non-empty string")
    return errors
