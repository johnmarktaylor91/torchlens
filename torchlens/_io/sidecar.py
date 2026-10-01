"""Namespaced typed SIDECAR families: out-of-tree persistence at L1.

The persistence half of out-of-tree equal citizenship (architecture memo 6.3
seam 1, C01 item 14): an appliance -- in-tree or out-of-tree -- persists its
own data on a Trace WITHOUT editing ``Trace``/``Op`` schemas, through one
typed door. Each family declares a schema ID, version, owner, validator, and
size budget at registration; payloads ride the persisting
``Trace.annotations`` mapping under the reserved ``"sidecar"`` namespace as
self-describing envelopes.

Missing-provider behavior is ANALYSIS-ONLY, declared at the registry domain:
loading an artifact whose sidecar family is not registered in this process
returns an :class:`AnalysisOnlySidecar` view (raw envelope, no validation
claim) and never imports provider code (memo 6.4: artifact load never
imports a provider).

Persistence status: the ``"sidecar"`` annotations sub-key persists plainly
as of the coordinated tlspec v9 write (lane C07; the C01-era pre-release
gating retired at the bump). Loads validate the namespace and envelope
shape fail-closed (:mod:`torchlens._io.forgery_validation`); payload
semantics stay with the owning family's validator at read time.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .._registry import TORCHLENS_PROVIDER, ProviderInfo, create_registry
from ..errors._base import ConfigurationError

__tl_layer__ = "L1"

#: Reserved ``Trace.annotations`` namespace every sidecar envelope lives under.
SIDECAR_ANNOTATIONS_KEY = "sidecar"

#: Default per-family payload budget. Sidecars are metadata-sized records,
#: not activation stores; families that need more declare it at registration.
DEFAULT_SIDECAR_BUDGET_BYTES = 1_048_576

#: Family ids are namespaced ``<owner_ns>.<name>`` (at least one dot), each
#: segment a lowercase identifier: the namespace is what makes two providers'
#: families collision-free by construction.
_FAMILY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")


class SidecarError(ConfigurationError):
    """Raised for sidecar-family registration and payload refusals."""


@dataclass(frozen=True)
class SidecarFamily:
    """One registered sidecar family's declared contract.

    Parameters
    ----------
    family_id:
        Namespaced stable id (``"myorg.saliency"``).
    schema_id:
        Stable payload-schema identifier recorded in every envelope. Schema
        IDs, never Python paths (org rule 9).
    version:
        Positive integer payload-schema version.
    owner:
        Human-readable owning party (package or team).
    size_budget_bytes:
        Maximum canonical-JSON payload size accepted at attach time.
    """

    family_id: str
    schema_id: str
    version: int
    owner: str
    size_budget_bytes: int = DEFAULT_SIDECAR_BUDGET_BYTES

    def validate_payload(self, payload: Any) -> None:
        """Family-specific payload validation hook (default: no extra checks).

        Out-of-tree families subclass and override; the door always runs the
        JSON-portability and budget checks regardless, so an override can
        only tighten, never loosen.
        """


@dataclass(frozen=True)
class AnalysisOnlySidecar:
    """Unvalidated view of a persisted sidecar whose provider is absent.

    The declared missing-provider behavior of the sidecar domain: the raw
    envelope stays inspectable (analysis-only), no validation is claimed,
    and no provider code is imported.
    """

    family_id: str
    schema_id: str | None
    version: int | None
    payload: Any
    analysis_only: bool = True


_SIDECAR_REGISTRY = create_registry(
    "sidecar_families",
    kind_label="sidecar family",
    missing_provider_behavior="analysis_only",
)

# The reserved namespace persists plainly as of the coordinated tlspec v9
# write (C07): the C01-era pre-release gating retired at the bump, and loads
# validate the envelope shape fail-closed (_io/forgery_validation.py).


def register_sidecar_family(
    family: SidecarFamily,
    *,
    provider: ProviderInfo | None = None,
    capabilities: dict[str, Any] | None = None,
    replace: bool = False,
) -> None:
    """Register one sidecar family through the public door.

    Parameters
    ----------
    family:
        The family contract. ``family_id`` must be namespaced
        (``"<owner_ns>.<name>"``), ``schema_id`` non-empty, ``version`` a
        positive int, ``size_budget_bytes`` positive.
    provider:
        Stable provider identity; defaults to the TorchLens builtin row (an
        out-of-tree provider passes its own).
    capabilities:
        Optional extra capability rows; the door always declares the payload
        encoding and budget.
    replace:
        Explicit replacement opt-in (kernel collision refusal otherwise).
    """

    if not isinstance(family, SidecarFamily):
        raise SidecarError(
            f"register_sidecar_family expects a SidecarFamily; got {type(family).__name__!r}.",
            code="sidecar_family_type_invalid",
            remedy="Construct torchlens.io.SidecarFamily(...) (or a subclass) and pass it.",
        )
    if not _FAMILY_ID_PATTERN.match(family.family_id or ""):
        raise SidecarError(
            f"Sidecar family id {family.family_id!r} is not namespaced. Family ids "
            "are '<owner_ns>.<name>' (lowercase identifiers, at least one dot), "
            "so two providers can never collide by construction.",
            code="sidecar_family_id_invalid",
            family_id=str(family.family_id),
            remedy="Use an id like 'myorg.saliency'.",
        )
    if not family.schema_id or not isinstance(family.schema_id, str):
        raise SidecarError(
            f"Sidecar family {family.family_id!r} declares no schema_id; envelopes "
            "are self-describing and artifacts store schema IDs, never Python "
            "paths.",
            code="sidecar_schema_invalid",
            family_id=family.family_id,
            remedy="Declare a stable non-empty schema_id string.",
        )
    if not isinstance(family.version, int) or family.version < 1:
        raise SidecarError(
            f"Sidecar family {family.family_id!r} declares invalid version "
            f"{family.version!r}; versions are positive integers.",
            code="sidecar_schema_invalid",
            family_id=family.family_id,
            remedy="Declare version >= 1.",
        )
    if not isinstance(family.size_budget_bytes, int) or family.size_budget_bytes < 1:
        raise SidecarError(
            f"Sidecar family {family.family_id!r} declares invalid size budget "
            f"{family.size_budget_bytes!r}.",
            code="sidecar_schema_invalid",
            family_id=family.family_id,
            remedy="Declare a positive size_budget_bytes.",
        )
    rows: dict[str, Any] = {
        "payload_encoding": "json",
        "size_budget_bytes": family.size_budget_bytes,
        "schema_id": family.schema_id,
        "schema_version": family.version,
    }
    if capabilities:
        rows.update(capabilities)
    _SIDECAR_REGISTRY.register(
        family.family_id,
        family,
        capabilities=rows,
        provider=provider or TORCHLENS_PROVIDER,
        replace=replace,
    )


def unregister_sidecar_family(family_id: str) -> None:
    """Remove one sidecar family registration (tests and provider teardown)."""

    _SIDECAR_REGISTRY.unregister(family_id)


def list_sidecar_families() -> tuple[str, ...]:
    """Return the registered sidecar family ids."""

    return _SIDECAR_REGISTRY.list_ids()


def _canonical_payload_bytes(family_id: str, payload: Any) -> int:
    """Measure one payload's canonical-JSON size, refusing unportable data."""

    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise SidecarError(
            f"Sidecar payload for family {family_id!r} is not JSON-portable: {exc}. "
            "Sidecar envelopes persist inside Trace.annotations, so payloads are "
            "JSON data (convert tensors/arrays to lists or store digests).",
            code="sidecar_payload_invalid",
            family_id=family_id,
            remedy="Pass JSON-serializable payload data.",
        ) from exc
    return len(encoded.encode("utf-8"))


def _annotations_mapping(trace: Any, family_id: str) -> dict[str, Any]:
    """Return ``trace.annotations`` or refuse typed."""

    annotations = getattr(trace, "annotations", None)
    if not isinstance(annotations, dict):
        raise SidecarError(
            f"Sidecar attach/read for family {family_id!r} needs a Trace with an "
            f"annotations mapping; got {type(trace).__name__!r}.",
            code="sidecar_trace_invalid",
            family_id=family_id,
            remedy="Pass a captured or loaded torchlens Trace.",
        )
    return annotations


def attach_sidecar(trace: Any, family_id: str, payload: Any) -> None:
    """Validate and attach one sidecar payload to a Trace.

    The write path requires the family to be REGISTERED (writes are a
    provider act; only reads degrade analysis-only). The payload is validated
    (JSON portability, size budget, then the family's own validator) and
    stored as a self-describing envelope under
    ``trace.annotations["sidecar"][family_id]``.
    """

    family = _SIDECAR_REGISTRY.get(family_id)
    annotations = _annotations_mapping(trace, family_id)
    size = _canonical_payload_bytes(family_id, payload)
    if size > family.size_budget_bytes:
        raise SidecarError(
            f"Sidecar payload for family {family_id!r} is {size} bytes, over the "
            f"family's declared budget of {family.size_budget_bytes} bytes.",
            code="sidecar_budget_exceeded",
            family_id=family_id,
            payload_bytes=size,
            budget_bytes=family.size_budget_bytes,
            remedy=(
                "Store less data (digests/references instead of values), or "
                "register the family with a larger size_budget_bytes."
            ),
        )
    family.validate_payload(payload)
    envelope = {
        "schema_id": family.schema_id,
        "version": family.version,
        "owner": family.owner,
        "payload": payload,
    }
    annotations.setdefault(SIDECAR_ANNOTATIONS_KEY, {})[family_id] = envelope


def read_sidecar(trace: Any, family_id: str) -> Any | AnalysisOnlySidecar:
    """Read one sidecar payload; provider-absent reads degrade analysis-only.

    Returns
    -------
    Any | AnalysisOnlySidecar
        The validated payload when the family is registered; an
        :class:`AnalysisOnlySidecar` (raw envelope, no validation claim) when
        it is not. Never imports provider code.
    """

    annotations = _annotations_mapping(trace, family_id)
    namespace = annotations.get(SIDECAR_ANNOTATIONS_KEY)
    if not isinstance(namespace, dict) or family_id not in namespace:
        raise SidecarError(
            f"Trace carries no sidecar for family {family_id!r}. Present: "
            f"{sorted(namespace) if isinstance(namespace, dict) else 'none'}.",
            code="sidecar_absent",
            family_id=family_id,
            remedy="Attach the sidecar first, or read one of the present families.",
        )
    envelope = namespace[family_id]
    if not isinstance(envelope, dict) or "payload" not in envelope:
        raise SidecarError(
            f"Sidecar envelope for family {family_id!r} is malformed "
            "(not a self-describing mapping with a 'payload' key).",
            code="sidecar_envelope_invalid",
            family_id=family_id,
            remedy="Re-attach through attach_sidecar(); do not hand-edit envelopes.",
        )
    if family_id not in _SIDECAR_REGISTRY.list_ids():
        return AnalysisOnlySidecar(
            family_id=family_id,
            schema_id=envelope.get("schema_id"),
            version=envelope.get("version"),
            payload=envelope.get("payload"),
        )
    family = _SIDECAR_REGISTRY.get(family_id)
    declared_version = envelope.get("version")
    if not isinstance(declared_version, int) or declared_version > family.version:
        raise SidecarError(
            f"Sidecar envelope for family {family_id!r} declares version "
            f"{declared_version!r}, newer than the registered family version "
            f"{family.version}. The artifact is valid; this provider is too old.",
            code="sidecar_version_unsupported",
            family_id=family_id,
            envelope_version=declared_version,
            registered_version=family.version,
            remedy="Upgrade the provider that owns this sidecar family.",
        )
    payload = envelope.get("payload")
    family.validate_payload(payload)
    return payload
