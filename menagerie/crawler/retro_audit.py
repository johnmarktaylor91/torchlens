"""Total-authority retro-audit waves over explicitly untrusted legacy hints."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.constants import FidelityVerdict, TERMINAL_STATUS_CODES


class CampaignWave(str, Enum):
    """Frozen ruled retro-audit campaign order."""

    CALIBRATION = "calibration"
    KNOWN_SLOP = "known-slop"
    PRESUMPTIVE_SLOP = "presumptive-slop"
    FAITHFUL_CLAIMERS = "faithful-claimers"
    CLASSICS = "classics"
    OFFICIAL_LIBRARIES = "official-libraries"
    REPOSITORY_RESEARCH_TAIL = "repository-research-tail"
    DEFERRED_DISCOVERY_ONLY = "deferred-discovery-only"


@dataclass(frozen=True)
class LegacyAuditItem:
    """One legacy roster item used only for conservative audit routing."""

    stable_id: str
    flags: frozenset[str]
    discovery_sources: frozenset[str]
    legacy_untrusted: bool = True
    inherited_award: bool = False

    def __post_init__(self) -> None:
        """Forbid legacy authority or inherited awards.

        Raises
        ------
        ValueError
            If the item is unidentified, trusted, or pre-awarded.
        """

        if not self.stable_id.strip():
            raise ValueError("retro-audit stable_id must be non-empty")
        if not self.legacy_untrusted or self.inherited_award:
            raise ValueError("legacy hints must remain untrusted and never inherit awards")

    @classmethod
    def from_migration(cls, payload: Mapping[str, Any]) -> "LegacyAuditItem":
        """Convert a ``migrate_legacy`` payload into an untrusted audit item.

        Parameters
        ----------
        payload:
            Hash-only legacy migration mapping.

        Returns
        -------
        LegacyAuditItem
            Conservative routing-only item.

        Raises
        ------
        ValueError
            If migration invariants do not prove the claims untrusted.
        """

        intake = payload.get("intake")
        inherited = payload.get("inherited_claims")
        if not isinstance(intake, Mapping) or not isinstance(inherited, Mapping):
            raise ValueError("retro audit requires a complete legacy migration payload")
        if intake.get("legacy_claims_untrusted") is not True:
            raise ValueError("legacy migration did not mark claims untrusted")
        if any(
            inherited.get(field) not in {False, None}
            for field in ("runs", "rung", "source_verified", "fidelity", "recipe_accepted")
        ):
            raise ValueError("legacy migration contains an inherited award")
        raw_flags = intake.get("preserved_legacy_flags", [])
        raw_sources = intake.get("discovery_sources", [])
        if not isinstance(raw_flags, list) or not isinstance(raw_sources, list):
            raise ValueError("legacy flags and discovery_sources must be lists")
        return cls(
            stable_id=str(payload.get("stable_id", "")),
            flags=frozenset(str(flag) for flag in raw_flags),
            discovery_sources=frozenset(str(source) for source in raw_sources),
        )


@dataclass(frozen=True)
class ReearnedRecord:
    """Current authoritative evidence used by wave completion gates."""

    stable_id: str
    terminal_status: str
    execution_reearned: bool
    current: bool
    fidelity_verdict: Optional[FidelityVerdict] = None
    fidelity_current: bool = False

    def __post_init__(self) -> None:
        """Validate a re-earned completion receipt.

        Raises
        ------
        ValueError
            If identity/status or fidelity state is contradictory.
        """

        if not self.stable_id.strip() or self.terminal_status not in TERMINAL_STATUS_CODES:
            raise ValueError("re-earned record requires a stable ID and canonical terminal status")
        if self.fidelity_current != (self.fidelity_verdict is not None):
            raise ValueError("current fidelity must carry exactly one five-way verdict")


@dataclass(frozen=True)
class WaveDefinition:
    """Deterministic pool and ruled fidelity gate for one campaign wave."""

    wave: CampaignWave
    stable_ids: tuple[str, ...]
    requires_fidelity: bool

    def completion(self, records: Iterable[ReearnedRecord]) -> "WaveCompletion":
        """Evaluate full pool coverage without awarding any run.

        Parameters
        ----------
        records:
            Current reducer-authorized re-earned records.

        Returns
        -------
        WaveCompletion
            Exact uncovered and missing-verdict sets.
        """

        current = {
            record.stable_id: record
            for record in records
            if record.current and record.execution_reearned
        }
        uncovered = tuple(stable_id for stable_id in self.stable_ids if stable_id not in current)
        missing_fidelity = tuple(
            stable_id
            for stable_id in self.stable_ids
            if stable_id in current
            and self.requires_fidelity
            and not current[stable_id].fidelity_current
        )
        return WaveCompletion(self.wave, uncovered, missing_fidelity)


@dataclass(frozen=True)
class WaveCompletion:
    """Exact result of one retro-audit wave completion gate."""

    wave: CampaignWave
    uncovered_stable_ids: tuple[str, ...]
    missing_fidelity_stable_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        """Return whether every legacy item re-earned all ruled facts.

        Returns
        -------
        bool
            True only with full pool coverage and required verdicts.
        """

        return not self.uncovered_stable_ids and not self.missing_fidelity_stable_ids


@dataclass(frozen=True)
class RetroAuditCampaign:
    """Ordered, disjoint campaign waves over the full legacy roster."""

    waves: tuple[WaveDefinition, ...]

    def wave(self, wave: CampaignWave) -> WaveDefinition:
        """Return one named campaign wave.

        Parameters
        ----------
        wave:
            Frozen wave name.

        Returns
        -------
        WaveDefinition
            Matching definition.

        Raises
        ------
        KeyError
            If the campaign omitted the wave.
        """

        for definition in self.waves:
            if definition.wave is wave:
                return definition
        raise KeyError(wave.value)


def define_campaign(
    items: Sequence[LegacyAuditItem], *, calibration_ids: Iterable[str]
) -> RetroAuditCampaign:
    """Select and flag all frozen retro-audit pools deterministically.

    Wave membership is disjoint in campaign order. Calibration membership is an
    explicit ruled sample; all other routing uses preserved risk flags and roster
    source names only, never inherited success/fidelity claims.

    Parameters
    ----------
    items:
        Complete untrusted legacy roster.
    calibration_ids:
        Explicit balanced calibration sample selected by the campaign planner.

    Returns
    -------
    RetroAuditCampaign
        Ordered wave definitions covering every supplied item exactly once.

    Raises
    ------
    ValueError
        If IDs are duplicated or calibration references an unknown item.
    """

    indexed: dict[str, LegacyAuditItem] = {}
    for item in items:
        if item.stable_id in indexed:
            raise ValueError(f"duplicate retro-audit stable ID: {item.stable_id}")
        indexed[item.stable_id] = item
    calibration = set(calibration_ids)
    unknown = calibration - indexed.keys()
    if unknown:
        raise ValueError(f"calibration references unknown stable IDs: {sorted(unknown)}")
    pools: dict[CampaignWave, list[str]] = {wave: [] for wave in CampaignWave}
    for stable_id in sorted(indexed):
        item = indexed[stable_id]
        selected = _select_wave(item, stable_id in calibration)
        pools[selected].append(stable_id)
    verdict_waves = {
        CampaignWave.CALIBRATION,
        CampaignWave.KNOWN_SLOP,
        CampaignWave.PRESUMPTIVE_SLOP,
        CampaignWave.FAITHFUL_CLAIMERS,
        CampaignWave.CLASSICS,
    }
    return RetroAuditCampaign(
        tuple(
            WaveDefinition(wave, tuple(pools[wave]), wave in verdict_waves) for wave in CampaignWave
        )
    )


def _select_wave(item: LegacyAuditItem, calibration: bool) -> CampaignWave:
    """Select one ruled wave from untrusted routing hints.

    Parameters
    ----------
    item:
        Legacy routing-only item.
    calibration:
        Whether the explicit calibration sample contains it.

    Returns
    -------
    CampaignWave
        First applicable frozen wave.
    """

    if calibration:
        return CampaignWave.CALIBRATION
    flags = item.flags
    sources = {source.lower() for source in item.discovery_sources}
    if "slop-detected-r1-audit" in flags or "legacy-known-slop" in flags:
        return CampaignWave.KNOWN_SLOP
    if flags & {
        "legacy-presumptive-slop",
        "legacy-compact-recipe",
        "legacy-local-recipe",
        "legacy-unregistered-recipe",
    }:
        return CampaignWave.PRESUMPTIVE_SLOP
    if "legacy-fidelity-claim" in flags or "legacy-faithful-claimer" in flags:
        return CampaignWave.FAITHFUL_CLAIMERS
    if "classic" in sources or "classics" in sources or "legacy-classic" in flags:
        return CampaignWave.CLASSICS
    if "legacy-official-library" in flags or "official-library" in sources:
        return CampaignWave.OFFICIAL_LIBRARIES
    if "legacy-deferred" in flags or sources & {"deferred", "discovery-only"}:
        return CampaignWave.DEFERRED_DISCOVERY_ONLY
    return CampaignWave.REPOSITORY_RESEARCH_TAIL
