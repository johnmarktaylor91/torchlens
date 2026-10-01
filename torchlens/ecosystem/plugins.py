"""Entry-point provider mechanics (ecosystem MEMO section 5, build item B3).

Consent law (v1): DISCOVERY IS METADATA-ONLY; ACTIVATION IS EXPLICIT AND
DISTRIBUTION-SCOPED. Discovery parses installed dist-info text and never
calls ``EntryPoint.load()``; activation is one explicit call naming a
distribution. There is NO default-all, NO autoload-on-name, NO cwd trust
grant, NO attribute-triggered execution: data never triggers code. The
shipped counter-example this design corrects: the historical recipes
autoloader executed installed third-party code on an ordinary ``tl.trace()``
with no opt-in (fixed in gate G2; this module generalizes that posture to
every group).

Exactly the FIVE architecture-approved groups exist (the signed architecture
memo owns the seam inventory); the ``torchlens.*`` entry-point namespace
belongs to the project, torchlens consults only the groups listed here, and
declaring an unlisted ``torchlens.*`` group has no effect and no contract.
Each activated entry point must be a ZERO-ARG FACTORY returning frozen data
for its domain door; import side effects do not count as registration, and
the honest transaction limit is stated plainly: a rolled-back activation
cannot undo arbitrary side effects performed during Python import.

Security posture, stated plainly: enabling a provider grants ordinary Python
process authority; there is no sandbox claim; conformance evidence
(``torchlens.conformance``) is correctness evidence, never a security
review. ``TORCHLENS_PLUGINS=none`` beats everything.

Every spelling is DOCUMENTED-UNSTABLE pending naming-session ratification.
The per-user trust ledger is deferred (MEMO 8.1); its one v1 obligation
ships here: :func:`activate_configured` computes and REPORTS the digest of
the declaration it consumed, so future grants can key on (source, digest).
"""

from __future__ import annotations

import hashlib
import logging
import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from ..errors import ConfigurationError, TorchLensWarning

__tl_layer__ = "L8"

LOGGER = logging.getLogger(__name__)

#: Environment kill switch / allowlist. ``"none"`` refuses every activation;
#: a comma-separated distribution list is the CI allowlist consumed by
#: ``activate_configured(source="env")``. Scanning it is inert -- the value
#: only ever supplies NAMES to the explicit activation operation.
PLUGINS_ENV_VAR = "TORCHLENS_PLUGINS"

#: The five architecture-approved entry-point groups (MEMO 5.1). Values name
#: the domain door each factory result is committed through.
ENTRY_POINT_GROUPS: tuple[str, ...] = (
    "torchlens.backends",
    "torchlens.transforms",
    "torchlens.export_targets",
    "torchlens.appliances",
    "torchlens.recipes",
)


class PluginLoadWarning(TorchLensWarning):
    """Categorized warning for soft (non-strict) provider-load failures."""


@dataclass(frozen=True)
class ProviderCandidate:
    """Metadata-only description of one installed entry point.

    Parameters
    ----------
    group:
        Entry-point group (one of :data:`ENTRY_POINT_GROUPS`).
    name:
        Entry-point name inside the group.
    value:
        The dotted target string from dist-info (never imported here).
    distribution:
        Installed distribution (PyPI project) declaring the entry point.
    version:
        Declared distribution version.
    """

    group: str
    name: str
    value: str
    distribution: str
    version: str


@dataclass(frozen=True)
class PluginRecord:
    """One four-state listing row (installed / requested / loaded / failed).

    Parameters
    ----------
    candidate:
        The metadata-only candidate row.
    state:
        ``"installed"`` (discovered, never requested), ``"requested"``
        (activation began), ``"loaded"`` (committed through its door), or
        ``"failed"`` (activation raised; detail carries the cause).
    detail:
        Human-readable failure or commit detail.
    """

    candidate: ProviderCandidate
    state: str
    detail: str = ""


@dataclass(frozen=True)
class ActivationReport:
    """Result of one explicit distribution-scoped activation.

    Parameters
    ----------
    distribution:
        The activated distribution name.
    loaded:
        Candidates committed through their domain doors.
    failed:
        Candidates whose activation failed (also warned or raised).
    declaration_digest:
        SHA-256 of the consumed declaration for configured activations
        (the v2 trust-ledger upgrade hook, MEMO 8.1), else ``None``.
    """

    distribution: str
    loaded: tuple[ProviderCandidate, ...]
    failed: tuple[ProviderCandidate, ...]
    declaration_digest: str | None = None


#: Session activation ledger keyed by (group, name, distribution).
_RECORDS: dict[tuple[str, str, str], PluginRecord] = {}


def _entry_points_for_group(group: str) -> tuple[metadata.EntryPoint, ...]:
    """Enumerate installed entry points for one group, metadata-only."""

    try:
        return tuple(metadata.entry_points().select(group=group))
    except Exception as exc:  # noqa: BLE001 - dist-info parse failures are disclosed, never fatal
        warnings.warn(
            PluginLoadWarning(
                f"Could not inspect {group} entry points: {exc}. Remedy: check "
                "the broken installed distribution's metadata",
                code="plugin_metadata_unreadable",
            ),
            stacklevel=3,
        )
        return ()


def _candidate(group: str, entry_point: metadata.EntryPoint) -> ProviderCandidate:
    """Build the metadata row for one entry point without importing it."""

    dist = getattr(entry_point, "dist", None)
    return ProviderCandidate(
        group=group,
        name=entry_point.name,
        value=entry_point.value,
        distribution=getattr(dist, "name", "") or "",
        version=getattr(dist, "version", "") or "",
    )


def discover(group: str | None = None) -> tuple[ProviderCandidate, ...]:
    """Discover installed providers WITHOUT importing any of them.

    Parameters
    ----------
    group:
        One of :data:`ENTRY_POINT_GROUPS`, or ``None`` for all five.

    Returns
    -------
    tuple[ProviderCandidate, ...]
        Metadata-only rows (group, name, value, distribution, version).
        Nothing is imported and nothing executes; the rows may improve an
        unknown-name refusal with the exact activation spelling (a
        signpost, not a wall) but can claim no capability or conformance.

    Raises
    ------
    ConfigurationError
        ``plugin_group_unknown`` for a group outside the approved five --
        the ``torchlens.*`` entry-point namespace belongs to the project
        and unlisted groups have no contract.
    """

    if group is not None and group not in ENTRY_POINT_GROUPS:
        raise ConfigurationError(
            f"Unknown TorchLens entry-point group {group!r}; the approved groups "
            f"are {list(ENTRY_POINT_GROUPS)}. Declaring an unlisted torchlens.* "
            "group has no effect and no contract. Remedy: use one of the "
            "approved groups.",
            code="plugin_group_unknown",
            remedy="use one of the approved groups",
            requested_group=group,
        )
    groups = ENTRY_POINT_GROUPS if group is None else (group,)
    rows: list[ProviderCandidate] = []
    for group_name in groups:
        for entry_point in _entry_points_for_group(group_name):
            candidate = _candidate(group_name, entry_point)
            rows.append(candidate)
            key = (candidate.group, candidate.name, candidate.distribution)
            _RECORDS.setdefault(key, PluginRecord(candidate=candidate, state="installed"))
    return tuple(rows)


def plugin_status() -> tuple[PluginRecord, ...]:
    """Four-state read-only listing (installed / requested / loaded / failed).

    Returns
    -------
    tuple[PluginRecord, ...]
        Every candidate observed this session, refreshed with any
        newly-installed metadata rows first. Reading the listing never
        imports provider code.
    """

    discover()
    return tuple(_RECORDS[key] for key in sorted(_RECORDS))


def _plugins_env_disabled() -> bool:
    """``TORCHLENS_PLUGINS=none`` beats everything."""

    return os.environ.get(PLUGINS_ENV_VAR, "").strip().lower() == "none"


def _commit_backends(result: Any, provider: ProviderCandidate) -> None:
    """Commit one backend provider spec through the backend door."""

    from ..backends.registry import BackendSpec, register_backend_spec

    if not isinstance(result, BackendSpec):
        raise ConfigurationError(
            f"torchlens.backends factory {provider.name!r} returned "
            f"{type(result).__name__}, not a BackendSpec. Remedy: return the "
            "frozen BackendSpec from a zero-arg factory.",
            code="plugin_activation_invalid",
            remedy="return the frozen BackendSpec from a zero-arg factory",
        )
    register_backend_spec(result)


def _commit_transforms(result: Any, provider: ProviderCandidate) -> None:
    """Commit one transform definition through the transforms door."""

    from .._registry import ProviderInfo
    from ..transforms._registry import register_transform

    register_transform(
        result,
        provider=ProviderInfo(
            provider_id=provider.distribution or provider.name,
            distribution=provider.distribution or None,
            version=provider.version or None,
        ),
    )


def _commit_export_targets(result: Any, provider: ProviderCandidate) -> None:
    """Commit one export-target row through the export door."""

    from .._registry import ProviderInfo
    from ..export._registry import register_export_target

    if not isinstance(result, dict) or not {"name", "fn", "tier"} <= set(result):
        raise ConfigurationError(
            f"torchlens.export_targets factory {provider.name!r} must return a "
            "dict with at least 'name', 'fn', and 'tier'. Remedy: return the "
            "frozen export-target row from a zero-arg factory.",
            code="plugin_activation_invalid",
            remedy="return the frozen export-target row from a zero-arg factory",
        )
    register_export_target(
        result["name"],
        result["fn"],
        tier=result["tier"],
        capabilities=result.get("capabilities"),
        provider=ProviderInfo(
            provider_id=provider.distribution or provider.name,
            distribution=provider.distribution or None,
            version=provider.version or None,
        ),
    )


#: Domain-door committers per group. ``torchlens.recipes`` is special-cased
#: in ``_activate_one`` (its door owns the load); ``torchlens.appliances``
#: refuses before any import (its architecture door has not landed).
_DOOR_COMMITTERS: dict[str, Callable[[Any, ProviderCandidate], None]] = {
    "torchlens.backends": _commit_backends,
    "torchlens.transforms": _commit_transforms,
    "torchlens.export_targets": _commit_export_targets,
}


def _activate_one(entry_point: metadata.EntryPoint, candidate: ProviderCandidate) -> None:
    """Load and commit ONE entry point through its domain door.

    The loader announces distribution/group/name BEFORE import (so a hang or
    crash during a hostile import is attributable), requires a zero-arg
    factory, and commits the frozen result through the door.
    """

    if candidate.group == "torchlens.appliances":
        raise ConfigurationError(
            "The torchlens.appliances provider door has not landed (the "
            "architecture memo reserves the seam); activation refuses BEFORE "
            "importing the provider so no code runs for a doomed activation. "
            "Remedy: track the appliance door's landing; the group row exists "
            "so the eventual door is one typed committer, not a loader "
            "redesign.",
            code="plugin_group_door_unavailable",
            remedy="wait for the appliance provider door to land",
            group=candidate.group,
        )
    LOGGER.info(
        "torchlens plugin activation: loading %s:%s from distribution %s==%s",
        candidate.group,
        candidate.name,
        candidate.distribution,
        candidate.version,
    )
    if candidate.group == "torchlens.recipes":
        from ..semantic.recipes import activate_entrypoint_recipes

        activate_entrypoint_recipes([candidate.name])
        return
    factory = entry_point.load()
    if not callable(factory):
        raise ConfigurationError(
            f"Entry point {candidate.group}:{candidate.name} is not callable; "
            "every provider target must be a zero-arg factory returning frozen "
            "data (import side effects do not count as registration). Remedy: "
            "point the entry point at a zero-arg factory.",
            code="plugin_activation_invalid",
            remedy="point the entry point at a zero-arg factory",
        )
    result = factory()
    _DOOR_COMMITTERS[candidate.group](result, candidate)


def _handle_activation_failure(
    candidate: ProviderCandidate, distribution: str, exc: Exception, *, strict: bool
) -> None:
    """Escalate one provider-activation failure per the strict/soft contract.

    Strict mode raises the typed ``plugin_strict_load_failed`` refusal with
    the provider's own exception chained; soft mode warns
    :class:`PluginLoadWarning` (code ``plugin_provider_load_failed``) and
    returns so the batch continues. The caller has already recorded the
    ``failed`` ledger state.

    Parameters
    ----------
    candidate:
        The provider whose activation raised.
    distribution:
        Distribution name named in the message.
    exc:
        The provider's activation exception.
    strict:
        Whether the failure stops the batch.

    Raises
    ------
    ConfigurationError
        With ``code="plugin_strict_load_failed"`` when ``strict`` is true.
    """

    if strict:
        raise ConfigurationError(
            f"Strict activation of {candidate.group}:{candidate.name} "
            f"from {distribution!r} failed: {exc}. Remedy: fix or "
            "unpin the provider, or pass strict=False to continue past "
            "broken providers with a warning.",
            code="plugin_strict_load_failed",
            remedy="fix the provider or pass strict=False",
            group=candidate.group,
            provider_name=candidate.name,
        ) from exc
    warnings.warn(
        PluginLoadWarning(
            f"Skipping broken provider {candidate.group}:{candidate.name} "
            f"from {distribution!r}: {exc}. Remedy: fix or unpin the "
            "provider distribution",
            code="plugin_provider_load_failed",
        ),
        stacklevel=3,
    )


def activate(
    distribution: str,
    *,
    groups: tuple[str, ...] | None = None,
    strict: bool = True,
    _declaration_digest: str | None = None,
) -> ActivationReport:
    """Explicitly activate every provider one DISTRIBUTION declares.

    Activation is the one act that imports provider code, and it is scoped
    to a named installed distribution -- never "everything installed", never
    a name read from an artifact. ``TORCHLENS_PLUGINS=none`` refuses.

    Parameters
    ----------
    distribution:
        Installed distribution (PyPI project) whose entry points activate.
    groups:
        Optional subset of :data:`ENTRY_POINT_GROUPS` to activate.
    strict:
        When ``True`` (default) a provider failure raises typed after
        recording the failure; when ``False`` failures warn
        :class:`PluginLoadWarning` and the batch continues.
    _declaration_digest:
        Internal: digest passthrough from :func:`activate_configured`.

    Returns
    -------
    ActivationReport
        Loaded and failed candidates plus the declaration digest when the
        activation came from a configured declaration.

    Raises
    ------
    ConfigurationError
        ``plugin_activation_disabled`` under ``TORCHLENS_PLUGINS=none``;
        ``plugin_distribution_not_installed`` when nothing is installed
        under the name; ``plugin_strict_load_failed`` for a strict-mode
        provider failure (chaining the cause).
    """

    if _plugins_env_disabled():
        raise ConfigurationError(
            f"{PLUGINS_ENV_VAR}=none disables every plugin activation in this "
            "process (the environment kill switch beats every other consent "
            "path). Remedy: unset the variable to activate providers.",
            code="plugin_activation_disabled",
            remedy="unset TORCHLENS_PLUGINS to activate providers",
        )
    selected_groups = ENTRY_POINT_GROUPS if groups is None else groups
    for group in selected_groups:
        if group not in ENTRY_POINT_GROUPS:
            raise ConfigurationError(
                f"Unknown TorchLens entry-point group {group!r}; the approved "
                f"groups are {list(ENTRY_POINT_GROUPS)}. Remedy: use approved "
                "groups only.",
                code="plugin_group_unknown",
                remedy="use approved groups only",
                requested_group=group,
            )
    matched: list[tuple[metadata.EntryPoint, ProviderCandidate]] = []
    for group in selected_groups:
        for entry_point in _entry_points_for_group(group):
            candidate = _candidate(group, entry_point)
            if candidate.distribution == distribution:
                matched.append((entry_point, candidate))
    if not matched:
        installed = sorted({c.distribution for c in discover() if c.distribution})
        raise ConfigurationError(
            f"No installed distribution {distribution!r} declares TorchLens "
            f"entry points in groups {list(selected_groups)}. Installed "
            f"provider distributions: {installed}. Remedy: install the "
            "provider distribution, then call activate() with its exact name.",
            code="plugin_distribution_not_installed",
            remedy="install the provider distribution and re-run activate()",
            requested_distribution=distribution,
        )
    loaded: list[ProviderCandidate] = []
    failed: list[ProviderCandidate] = []
    for entry_point, candidate in matched:
        key = (candidate.group, candidate.name, candidate.distribution)
        _RECORDS[key] = PluginRecord(candidate=candidate, state="requested")
        try:
            _activate_one(entry_point, candidate)
        except Exception as exc:  # noqa: BLE001 - any provider failure lands in the ledger typed
            _RECORDS[key] = PluginRecord(candidate=candidate, state="failed", detail=str(exc))
            failed.append(candidate)
            _handle_activation_failure(candidate, distribution, exc, strict=strict)
            continue
        _RECORDS[key] = PluginRecord(candidate=candidate, state="loaded")
        loaded.append(candidate)
        LOGGER.info(
            "torchlens plugin activation: loaded %s:%s from %s==%s",
            candidate.group,
            candidate.name,
            candidate.distribution,
            candidate.version,
        )
    return ActivationReport(
        distribution=distribution,
        loaded=tuple(loaded),
        failed=tuple(failed),
        declaration_digest=_declaration_digest,
    )


def activate_configured(
    names: tuple[str, ...] | list[str] | None = None,
    *,
    source: str = "env",
    strict: bool = True,
) -> tuple[ActivationReport, ...]:
    """Activate distributions NAMED by a declaration, reporting its digest.

    Config files and environment variables supply NAMES as arguments to this
    explicit operation; scanning them is inert (a checked-out project
    declaring plugins activates NOTHING until this call). The activation
    notice computes and reports the SHA-256 DIGEST of the declaration it
    consumed from day one -- the entire v2 trust-ledger upgrade path keys
    future grants on (source, digest), and without the v1 digest every
    pre-existing grant would be retroactively unverifiable (MEMO 8.1).

    Parameters
    ----------
    names:
        Distribution names to activate, or ``None`` with ``source="env"``
        to consume the ``TORCHLENS_PLUGINS`` comma-list.
    source:
        Declaration provenance label recorded with the digest (``"env"``
        or a caller-supplied config-file path string).
    strict:
        Passed through to :func:`activate`.

    Returns
    -------
    tuple[ActivationReport, ...]
        One report per activated distribution, each carrying the
        declaration digest.

    Raises
    ------
    ConfigurationError
        ``plugin_activation_disabled`` when the declaration is ``"none"``
        or the kill switch is set; the :func:`activate` refusals otherwise.
    """

    if names is None:
        declaration = os.environ.get(PLUGINS_ENV_VAR, "")
        source = f"env:{PLUGINS_ENV_VAR}"
        parsed = [part.strip() for part in declaration.split(",") if part.strip()]
    else:
        parsed = [str(name).strip() for name in names if str(name).strip()]
        declaration = ",".join(parsed)
    digest = hashlib.sha256(declaration.encode("utf-8")).hexdigest()
    LOGGER.info(
        "torchlens configured activation: source=%s declaration_digest=%s names=%s",
        source,
        digest,
        parsed,
    )
    if [name.lower() for name in parsed] == ["none"] or _plugins_env_disabled():
        raise ConfigurationError(
            f"The consumed plugin declaration from {source} is 'none' (or the "
            f"{PLUGINS_ENV_VAR} kill switch is set): activation is disabled. "
            f"Declaration digest: {digest}. Remedy: supply distribution names "
            "to activate providers.",
            code="plugin_activation_disabled",
            remedy="supply distribution names to activate providers",
            declaration_digest=digest,
        )
    return tuple(activate(name, strict=strict, _declaration_digest=digest) for name in parsed)
