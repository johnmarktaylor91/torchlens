"""Durable recurring wake episodes for usage-limit pauses.

The append-only operational ledger is the authority for episode state. Scheduler
files, OS registrations, and fire-intent files are disposable projections that
this module verifies and repairs from that authority.
"""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from menagerie.crawler.authority import WakeEpisode
from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.identity import canonical_json_bytes, fsync_directory, stable_hash
from menagerie.crawler.models import AppendResult, JsonObject
from menagerie.crawler.recordio import JsonlLedger

DEFAULT_RETRY_INTERVAL_SECONDS = 15 * 60
MIN_RETRY_INTERVAL_SECONDS = 5 * 60
MAX_RETRY_INTERVAL_SECONDS = 15 * 60
DEFAULT_HEALTH_THRESHOLD_FIRES = 96
RESET_OBSERVATIONS = frozenset({"observed", "guessed"})


class WakeupBackend(str, Enum):
    """Supported recurring OS scheduler mechanisms."""

    LAUNCHD = "launchd"
    SYSTEMD_TIMER = "systemd-timer"
    CRON = "cron"


class WakeupError(RuntimeError):
    """Base class for wake episode failures."""


class WakeupConfigurationError(WakeupError):
    """Raised when a recurring wake episode cannot be represented or verified."""


@dataclass(frozen=True)
class OperationalContext:
    """Common immutable fields carried by every operational event."""

    run_id: str
    machine_id: str
    queued_work_counts: Mapping[str, int]
    current_environment: Optional[str]

    def __post_init__(self) -> None:
        """Validate common event context.

        Raises
        ------
        ValueError
            If an identity or queue count is invalid.
        """

        if not self.run_id.strip() or not self.machine_id.strip():
            raise ValueError("run_id and machine_id must be non-empty")
        if any(not key.strip() or value < 0 for key, value in self.queued_work_counts.items()):
            raise ValueError("queued work counts require non-empty keys and non-negative values")


@dataclass(frozen=True)
class RenderedWakeupDefinition:
    """Disposable backend projection of one frozen wake episode."""

    episode: WakeEpisode
    backend: WakeupBackend
    definition_path: Path
    scheduler_definition: str
    service_path: Optional[Path]
    service_definition: Optional[str]


@dataclass(frozen=True)
class WakeEpisodeState:
    """Reducer-derived operational state for one frozen wake episode."""

    episode: WakeEpisode
    disposition: str
    terminal_event_id: Optional[str]
    fire_buckets: tuple[int, ...]
    install_verified: bool
    health_degraded: bool
    last_event_id: str
    backend: Optional[WakeupBackend]

    @property
    def active(self) -> bool:
        """Return whether recurrence must remain installed.

        Returns
        -------
        bool
            True until a durable resolving or superseding fact exists.
        """

        return self.disposition == "active"


@dataclass(frozen=True)
class WakeEpisodeProjection:
    """Pure operational-ledger projection of every wake episode."""

    episodes: Mapping[str, WakeEpisodeState]
    active_episode_ids: tuple[str, ...]

    @property
    def active_episodes(self) -> tuple[WakeEpisodeState, ...]:
        """Return active episode states in deterministic opening order.

        Returns
        -------
        tuple[WakeEpisodeState, ...]
            Active reducer-derived states.
        """

        return tuple(self.episodes[episode_id] for episode_id in self.active_episode_ids)


@dataclass(frozen=True)
class WakeupScheduleResult:
    """Result of opening and verifying one recurring wake episode."""

    episode: WakeEpisode
    definition: RenderedWakeupDefinition
    created: bool
    repaired: bool
    verified: bool
    pause_event: AppendResult
    scheduler_event: AppendResult


@dataclass(frozen=True)
class WakeupCallbackResult:
    """Result of one callback fire handled under the driver lock."""

    episode_id: str
    time_bucket: int
    appended: bool
    guarded_not_before: bool
    should_resume: bool
    deactivated: bool


@dataclass(frozen=True)
class WakeupReconciliationResult:
    """Summary of scheduler repair/deactivation work."""

    active_episode_ids: tuple[str, ...]
    installed_episode_ids: tuple[str, ...]
    repaired_episode_ids: tuple[str, ...]
    deactivated_episode_ids: tuple[str, ...]
    failures: Mapping[str, str]


@dataclass(frozen=True)
class WakeupInspection:
    """Read-only comparison of active authority and scheduler projections."""

    backend: WakeupBackend
    active_episode_ids: tuple[str, ...]
    missing_or_mismatched_episode_ids: tuple[str, ...]
    unverified_episode_ids: tuple[str, ...]

    @property
    def healthy(self) -> bool:
        """Return whether every active episode has an exact live projection.

        Returns
        -------
        bool
            True when no active projection needs repair.
        """

        return not self.missing_or_mismatched_episode_ids and not self.unverified_episode_ids


CommandExists = Callable[[str], bool]
DefinitionAction = Callable[[RenderedWakeupDefinition], None]
DefinitionVerifier = Callable[[RenderedWakeupDefinition], bool]


def detect_wakeup_backend(
    *,
    platform_name: Optional[str] = None,
    command_exists: Optional[CommandExists] = None,
) -> WakeupBackend:
    """Feature-detect launchd, systemd timer, or cron.

    Parameters
    ----------
    platform_name:
        Optional test override for ``platform.system()``.
    command_exists:
        Optional command-presence probe.

    Returns
    -------
    WakeupBackend
        launchd on macOS, systemd when available elsewhere, otherwise cron.

    Raises
    ------
    WakeupConfigurationError
        If no supported recurring scheduler exists.
    """

    system = (platform_name or platform.system()).lower()
    exists = command_exists or (lambda name: shutil.which(name) is not None)
    if system == "darwin":
        if not exists("launchctl"):
            raise WakeupConfigurationError("macOS wake episodes require launchctl")
        return WakeupBackend.LAUNCHD
    if exists("systemctl"):
        return WakeupBackend.SYSTEMD_TIMER
    if exists("crontab"):
        return WakeupBackend.CRON
    raise WakeupConfigurationError("no launchd, systemd, or cron wakeup mechanism found")


def build_wake_episode(
    *,
    provider: str,
    reset_at: str,
    reset_observation: str,
    callback_argv: Sequence[str],
    not_before: Optional[str] = None,
    retry_interval_seconds: int = DEFAULT_RETRY_INTERVAL_SECONDS,
    supersedes_episode_id: Optional[str] = None,
) -> WakeEpisode:
    """Build the frozen durable episode identity and callback argv.

    Parameters
    ----------
    provider:
        Usage-limited provider, ``anthropic`` or ``openai``.
    reset_at:
        Provider reset timestamp, whether observed or guessed.
    reset_observation:
        Closed evidence class, ``observed`` or ``guessed``.
    callback_argv:
        Exact base crawler-resume argv. The episode identity argument is added.
    not_before:
        Callback guard; defaults to ``reset_at``.
    retry_interval_seconds:
        Fixed recurring cadence in the inclusive five-to-fifteen-minute bound.
    supersedes_episode_id:
        Optional earlier episode replaced by improved reset evidence.

    Returns
    -------
    WakeEpisode
        Exact Phase-0 frozen episode value.

    Raises
    ------
    WakeupConfigurationError
        If a frozen episode invariant is invalid.
    """

    _validate_provider(provider)
    _validate_reset_observation(reset_observation)
    _parse_utc(reset_at)
    guarded_at = not_before or reset_at
    _parse_utc(guarded_at)
    _validate_retry_interval(retry_interval_seconds)
    base_argv = tuple(callback_argv)
    if not base_argv or any(not argument for argument in base_argv):
        raise WakeupConfigurationError("callback argv must contain non-empty arguments")
    if "--wake-episode-id" in base_argv:
        raise WakeupConfigurationError("base callback argv must not predeclare an episode ID")
    callback_identity = stable_hash({"callback_argv": list(base_argv)})
    identity = stable_hash(
        {
            "provider": provider,
            "reset_at": reset_at,
            "reset_observation": reset_observation,
            "not_before": guarded_at,
            "retry_interval_seconds": retry_interval_seconds,
            "callback_identity": callback_identity,
            "supersedes_episode_id": supersedes_episode_id,
        }
    ).removeprefix("sha256:")
    episode_id = f"wake-episode-{identity[:32]}"
    if supersedes_episode_id == episode_id:
        raise WakeupConfigurationError("a wake episode cannot supersede itself")
    opened_event_id = f"usage-pause-{episode_id}"
    return WakeEpisode(
        episode_id=episode_id,
        provider=provider,
        reset_at=reset_at,
        reset_observation=reset_observation,
        not_before=guarded_at,
        retry_interval_seconds=retry_interval_seconds,
        callback_identity=callback_identity,
        callback_argv=(*base_argv, "--wake-episode-id", episode_id),
        opened_event_id=opened_event_id,
        supersedes_episode_id=supersedes_episode_id,
    )


def build_usage_pause_event(
    *,
    episode: WakeEpisode,
    observed_response: str,
    context: OperationalContext,
    created_at: str,
) -> JsonObject:
    """Construct the authoritative opening event for one wake episode.

    Parameters
    ----------
    episode:
        Frozen episode being opened.
    observed_response:
        Exact redacted provider limit observation.
    context:
        Common campaign context.
    created_at:
        Exact event timestamp.

    Returns
    -------
    dict[str, object]
        Logical operational event ready for append.
    """

    _validate_episode(episode)
    _parse_utc(created_at)
    return _base_event(
        event_id=episode.opened_event_id,
        created_at=created_at,
        event_kind=OperationalEventKind.USAGE_PAUSE,
        status=OperationalEventStatus.USAGE_PAUSED,
        context=context,
        provider=episode.provider,
        observed_response=observed_response,
        reset_at=episode.reset_at,
        details={
            "episode": _episode_payload(episode),
            "reset_exact": episode.reset_observation == "observed",
        },
    )


def record_operational_event(ledger: JsonlLedger, event: Mapping[str, object]) -> AppendResult:
    """Append one operational event through the canonical strict ledger.

    Parameters
    ----------
    ledger:
        Operational-event ledger.
    event:
        Complete logical event without sequence/hash fields.

    Returns
    -------
    AppendResult
        Idempotent append result.
    """

    if ledger.schema_version != OPERATIONAL_EVENT_SCHEMA_VERSION:
        raise ValueError("record_operational_event requires an operational-event ledger")
    return ledger.append(event)


def reduce_wake_episodes(events: Sequence[Mapping[str, object]]) -> WakeEpisodeProjection:
    """Derive current wake episode state solely from operational events.

    Parameters
    ----------
    events:
        Persisted or logical operational events in ledger order.

    Returns
    -------
    WakeEpisodeProjection
        Immutable derived episode view.

    Raises
    ------
    WakeupConfigurationError
        If immutable episode identity is conflicting or malformed.
    """

    episodes: dict[str, WakeEpisode] = {}
    opening_order: list[str] = []
    dispositions: dict[str, str] = {}
    terminal_events: dict[str, Optional[str]] = {}
    fire_buckets: dict[str, set[int]] = {}
    install_verified: dict[str, bool] = {}
    health_degraded: dict[str, bool] = {}
    last_event_ids: dict[str, str] = {}
    backends: dict[str, Optional[WakeupBackend]] = {}

    for event in events:
        event_kind = event.get("event_kind")
        details = event.get("details")
        if not isinstance(details, Mapping):
            continue
        event_id = str(event.get("event_id", ""))
        if event_kind == OperationalEventKind.USAGE_PAUSE.value and isinstance(
            details.get("episode"), Mapping
        ):
            episode = _episode_from_payload(details["episode"])
            existing = episodes.get(episode.episode_id)
            if existing is not None and existing != episode:
                raise WakeupConfigurationError(
                    f"conflicting operational episode identity: {episode.episode_id}"
                )
            if existing is None:
                episodes[episode.episode_id] = episode
                opening_order.append(episode.episode_id)
                dispositions[episode.episode_id] = "active"
                terminal_events[episode.episode_id] = None
                fire_buckets[episode.episode_id] = set()
                install_verified[episode.episode_id] = False
                health_degraded[episode.episode_id] = False
                backends[episode.episode_id] = None
            last_event_ids[episode.episode_id] = event_id
            superseded = episode.supersedes_episode_id
            if superseded in episodes and dispositions[superseded] == "active":
                dispositions[superseded] = "superseded"
                terminal_events[superseded] = episode.opened_event_id
            continue

        episode_id = details.get("episode_id")
        if event_kind in {
            OperationalEventKind.CAMPAIGN_COMPLETED.value,
            OperationalEventKind.OPERATOR_CANCELLED.value,
        }:
            targets = (
                [str(episode_id)] if isinstance(episode_id, str) and episode_id else list(episodes)
            )
            disposition = (
                "completed"
                if event_kind == OperationalEventKind.CAMPAIGN_COMPLETED.value
                else "cancelled"
            )
            for target in targets:
                if target in episodes:
                    dispositions[target] = disposition
                    terminal_events[target] = event_id
                    last_event_ids[target] = event_id
            continue
        if not isinstance(episode_id, str) or episode_id not in episodes:
            continue
        last_event_ids[episode_id] = event_id
        if event_kind == OperationalEventKind.USAGE_RESUME.value:
            dispositions[episode_id] = "resumed"
            terminal_events[episode_id] = event_id
        elif event_kind in {
            OperationalEventKind.WAKEUP_INSTALLED.value,
            OperationalEventKind.WAKEUP_REPAIRED.value,
        }:
            install_verified[episode_id] = details.get("verified") is True
            backend_value = details.get("backend")
            if isinstance(backend_value, str):
                try:
                    backends[episode_id] = WakeupBackend(backend_value)
                except ValueError as exc:
                    raise WakeupConfigurationError(
                        f"unknown wakeup backend in event {event_id}: {backend_value!r}"
                    ) from exc
        elif event_kind == OperationalEventKind.WAKEUP_FAILED.value:
            install_verified[episode_id] = False
        elif event_kind in {
            OperationalEventKind.WAKEUP_FIRED.value,
            OperationalEventKind.WAKE_NOOP_ALREADY_RUNNING.value,
        }:
            bucket = details.get("time_bucket")
            if isinstance(bucket, int) and bucket >= 0:
                fire_buckets[episode_id].add(bucket)
        elif event_kind == OperationalEventKind.WAKEUP_HEALTH_DEGRADED.value:
            health_degraded[episode_id] = True

    states: dict[str, WakeEpisodeState] = {}
    for episode_id in opening_order:
        states[episode_id] = WakeEpisodeState(
            episode=episodes[episode_id],
            disposition=dispositions[episode_id],
            terminal_event_id=terminal_events[episode_id],
            fire_buckets=tuple(sorted(fire_buckets[episode_id])),
            install_verified=install_verified[episode_id],
            health_degraded=health_degraded[episode_id],
            last_event_id=last_event_ids[episode_id],
            backend=backends[episode_id],
        )
    active_ids = tuple(episode_id for episode_id in opening_order if states[episode_id].active)
    return WakeEpisodeProjection(states, active_ids)


def render_wakeup_status(projection: WakeEpisodeProjection) -> JsonObject:
    """Render operational episode history and active health for CLI/status consumers.

    Parameters
    ----------
    projection:
        Reducer-derived wake state.

    Returns
    -------
    dict[str, object]
        JSON-safe wake status summary.
    """

    episodes: list[JsonObject] = []
    for state in projection.episodes.values():
        episode = state.episode
        episodes.append(
            {
                "episode_id": episode.episode_id,
                "provider": episode.provider,
                "reset_at": episode.reset_at,
                "reset_observation": episode.reset_observation,
                "not_before": episode.not_before,
                "retry_interval_seconds": episode.retry_interval_seconds,
                "disposition": state.disposition,
                "active": state.active,
                "install_verified": state.install_verified,
                "backend": state.backend.value if state.backend is not None else None,
                "fire_count": len(state.fire_buckets),
                "last_time_bucket": state.fire_buckets[-1] if state.fire_buckets else None,
                "health_degraded": state.health_degraded,
                "terminal_event_id": state.terminal_event_id,
                "supersedes_episode_id": episode.supersedes_episode_id,
            }
        )
    return {
        "active_count": len(projection.active_episode_ids),
        "active_episode_ids": list(projection.active_episode_ids),
        "healthy": all(
            state.install_verified and not state.health_degraded
            for state in projection.active_episodes
        ),
        "episodes": episodes,
    }


def inspect_wakeup_definitions(
    wakeup_root: Path,
    events: Sequence[Mapping[str, object]],
    backend: WakeupBackend,
    *,
    verifier: Optional[DefinitionVerifier] = None,
) -> WakeupInspection:
    """Compare active operational authority with local and OS definitions.

    Parameters
    ----------
    wakeup_root:
        Gitignored scheduler projection root.
    events:
        Canonical operational events.
    backend:
        Host scheduler backend.
    verifier:
        Optional injected OS registration verifier.

    Returns
    -------
    WakeupInspection
        Exact missing/mismatched and unverified active episode IDs.
    """

    projection = reduce_wake_episodes(events)
    active_ids = projection.active_episode_ids
    missing: list[str] = []
    unverified: list[str] = []
    check = verifier or _verify_wakeup
    for state in projection.active_episodes:
        definition = render_wakeup_definition(wakeup_root, state.episode, backend)
        if not _definition_matches(definition):
            missing.append(state.episode.episode_id)
            continue
        if not check(definition):
            unverified.append(state.episode.episode_id)
    return WakeupInspection(backend, active_ids, tuple(missing), tuple(unverified))


class WakeupManager:
    """Reconcile recurring scheduler projections with operational episode authority."""

    def __init__(
        self,
        wakeup_root: Path,
        ledger: JsonlLedger,
        callback_argv: Sequence[str],
        *,
        backend: Optional[WakeupBackend] = None,
        platform_name: Optional[str] = None,
        command_exists: Optional[CommandExists] = None,
        installer: Optional[DefinitionAction] = None,
        verifier: Optional[DefinitionVerifier] = None,
        deactivator: Optional[DefinitionAction] = None,
        health_threshold_fires: int = DEFAULT_HEALTH_THRESHOLD_FIRES,
    ) -> None:
        """Bind scheduler effects and the canonical operational writer.

        Parameters
        ----------
        wakeup_root:
            Gitignored runtime root for definitions and fire intents.
        ledger:
            Canonical operational-event writer held under the driver lock.
        callback_argv:
            Base callback argv used when opening a new episode.
        backend, platform_name, command_exists:
            Explicit backend or feature-detection controls.
        installer, verifier, deactivator:
            Injectable OS scheduler operations.
        health_threshold_fires:
            Fire count that emits one visible degraded-health fact.
        """

        if not callback_argv or any(not argument for argument in callback_argv):
            raise WakeupConfigurationError("callback argv must contain non-empty arguments")
        if health_threshold_fires <= 0:
            raise WakeupConfigurationError("health threshold must be positive")
        self.root = wakeup_root
        self.ledger = ledger
        self.callback_argv = tuple(callback_argv)
        self.backend = backend or detect_wakeup_backend(
            platform_name=platform_name, command_exists=command_exists
        )
        self.installer = installer or _install_wakeup
        self.verifier = verifier or _verify_wakeup
        self.deactivator = deactivator or _deactivate_wakeup
        self.health_threshold_fires = health_threshold_fires

    @property
    def projection(self) -> WakeEpisodeProjection:
        """Return current event-derived episode state.

        Returns
        -------
        WakeEpisodeProjection
            Current pure operational projection.
        """

        return reduce_wake_episodes(self.ledger.records)

    def record_pause_and_schedule(
        self,
        *,
        provider: str,
        observed_response: str,
        reset_at: str,
        context: OperationalContext,
        created_at: str,
        reset_observation: str = "observed",
        not_before: Optional[str] = None,
        retry_interval_seconds: int = DEFAULT_RETRY_INTERVAL_SECONDS,
    ) -> WakeupScheduleResult:
        """Open an episode and verify its recurring scheduler projection.

        A newer pause for the same provider supersedes the currently active
        episode. Replaying the exact pause is append- and install-idempotent.

        Parameters
        ----------
        provider, observed_response, reset_at:
            Exact provider observation and reset timestamp.
        context:
            Common campaign event context.
        created_at:
            Exact pause timestamp.
        reset_observation:
            ``observed`` or ``guessed``.
        not_before:
            Optional callback guard, defaulting to the reset timestamp.
        retry_interval_seconds:
            Recurrence in the frozen five-to-fifteen-minute range.

        Returns
        -------
        WakeupScheduleResult
            Frozen episode and verified disposable projection.

        Raises
        ------
        WakeupConfigurationError
            If installation cannot be durably verified.
        """

        projection = self.projection
        prior_for_provider = [
            state.episode
            for state in projection.episodes.values()
            if state.episode.provider == provider
        ]
        supersedes = prior_for_provider[-1].episode_id if prior_for_provider else None
        candidate = build_wake_episode(
            provider=provider,
            reset_at=reset_at,
            reset_observation=reset_observation,
            callback_argv=self.callback_argv,
            not_before=not_before,
            retry_interval_seconds=retry_interval_seconds,
            supersedes_episode_id=supersedes,
        )
        if supersedes is not None and projection.episodes[supersedes].active:
            prior = projection.episodes[supersedes].episode
            if _same_episode_request(prior, candidate):
                candidate = prior
        existing_open = next(
            (
                event
                for event in self.ledger.records
                if event.get("event_id") == candidate.opened_event_id
            ),
            None,
        )
        if (
            existing_open is not None
            and existing_open.get("observed_response") == observed_response
        ):
            pause = AppendResult(existing_open, appended=False)
        else:
            pause = record_operational_event(
                self.ledger,
                build_usage_pause_event(
                    episode=candidate,
                    observed_response=observed_response,
                    context=context,
                    created_at=created_at,
                ),
            )
        definition = render_wakeup_definition(self.root, candidate, self.backend)
        created, repaired, verified, scheduler_event = self._ensure_installed(
            definition, context=context, created_at=created_at
        )
        if supersedes is not None and candidate.episode_id != supersedes:
            prior = projection.episodes[supersedes].episode
            prior_definition = render_wakeup_definition(self.root, prior, self.backend)
            if self._projection_or_registration_exists(prior_definition):
                self._deactivate(
                    prior_definition,
                    context=context,
                    created_at=created_at,
                )
        return WakeupScheduleResult(
            candidate,
            definition,
            created,
            repaired,
            verified,
            pause,
            scheduler_event,
        )

    def reconcile(
        self,
        *,
        context: OperationalContext,
        created_at: str,
    ) -> WakeupReconciliationResult:
        """Repair active definitions and deactivate every resolved projection.

        Parameters
        ----------
        context:
            Common campaign event context.
        created_at:
            Exact reconciliation timestamp.

        Returns
        -------
        WakeupReconciliationResult
            Installed, repaired, deactivated, and failed episode IDs.
        """

        _parse_utc(created_at)
        projection = self.projection
        installed: list[str] = []
        repaired: list[str] = []
        deactivated: list[str] = []
        failures: dict[str, str] = {}
        for state in projection.episodes.values():
            definition = render_wakeup_definition(self.root, state.episode, self.backend)
            if state.active:
                try:
                    created, was_repaired, _, _ = self._ensure_installed(
                        definition, context=context, created_at=created_at
                    )
                except WakeupConfigurationError as exc:
                    failures[state.episode.episode_id] = str(exc)
                    continue
                if created:
                    installed.append(state.episode.episode_id)
                elif was_repaired:
                    repaired.append(state.episode.episode_id)
            elif self._projection_or_registration_exists(definition):
                try:
                    self._deactivate(definition, context=context, created_at=created_at)
                except WakeupConfigurationError as exc:
                    failures[state.episode.episode_id] = str(exc)
                else:
                    deactivated.append(state.episode.episode_id)
        return WakeupReconciliationResult(
            self.projection.active_episode_ids,
            tuple(installed),
            tuple(repaired),
            tuple(deactivated),
            failures,
        )

    def handle_fire(
        self,
        episode_id: str,
        *,
        fired_at: str,
        context: OperationalContext,
    ) -> WakeupCallbackResult:
        """Handle one callback fire while the caller holds ``driver.lock``.

        Parameters
        ----------
        episode_id:
            Exact active or recently resolved episode.
        fired_at:
            Actual callback timestamp used for the idempotency bucket.
        context:
            Common campaign event context.

        Returns
        -------
        WakeupCallbackResult
            Guard, resume, append, and deactivation decisions.

        Raises
        ------
        WakeupConfigurationError
            If the episode is unknown or scheduler deactivation fails.
        """

        fired = _parse_utc(fired_at)
        projection = self.projection
        try:
            state = projection.episodes[episode_id]
        except KeyError as exc:
            raise WakeupConfigurationError(f"unknown wake episode: {episode_id}") from exc
        episode = state.episode
        bucket = _time_bucket(fired, episode.retry_interval_seconds)
        guarded = fired < _parse_utc(episode.not_before)
        if bucket in state.fire_buckets:
            return WakeupCallbackResult(episode_id, bucket, False, guarded, False, False)
        fire_event = _episode_event(
            episode,
            event_kind=OperationalEventKind.WAKEUP_FIRED,
            status=OperationalEventStatus.WAKEUP_FIRED,
            context=context,
            created_at=fired_at,
            identity={"time_bucket": bucket},
            details={
                "time_bucket": bucket,
                "guarded_not_before": guarded,
                "source": "scheduler-callback",
            },
        )
        fire_result = record_operational_event(self.ledger, fire_event)
        self._record_health_if_needed(episode, context=context, created_at=fired_at)
        state = self.projection.episodes[episode_id]
        if guarded:
            return WakeupCallbackResult(
                episode_id, bucket, fire_result.appended, True, False, False
            )
        if not state.active:
            definition = render_wakeup_definition(self.root, episode, self.backend)
            deactivated = self._deactivate(
                definition, context=context, created_at=fired_at, time_bucket=bucket
            )
            return WakeupCallbackResult(
                episode_id, bucket, fire_result.appended, False, False, deactivated
            )
        resume_event = _episode_event(
            episode,
            event_kind=OperationalEventKind.USAGE_RESUME,
            status=OperationalEventStatus.USAGE_RESUMED,
            context=context,
            created_at=fired_at,
            identity={"time_bucket": bucket},
            details={"time_bucket": bucket, "source": "scheduler-callback"},
        )
        record_operational_event(self.ledger, resume_event)
        definition = render_wakeup_definition(self.root, episode, self.backend)
        deactivated = self._deactivate(
            definition, context=context, created_at=fired_at, time_bucket=bucket
        )
        return WakeupCallbackResult(
            episode_id, bucket, fire_result.appended, False, True, deactivated
        )

    def resolve_episode(
        self,
        episode_id: str,
        *,
        resolution: str,
        context: OperationalContext,
        created_at: str,
    ) -> bool:
        """Durably resolve an episode before removing its scheduler projection.

        Parameters
        ----------
        episode_id:
            Exact frozen episode identity.
        resolution:
            ``manual-resume``, ``campaign-completed``, or ``operator-cancelled``.
        context:
            Common campaign operational context.
        created_at:
            Exact resolution timestamp.

        Returns
        -------
        bool
            True when a new deactivation fact was appended.

        Raises
        ------
        WakeupConfigurationError
            If the episode or resolution is unknown.
        """

        projection = self.projection
        try:
            episode = projection.episodes[episode_id].episode
        except KeyError as exc:
            raise WakeupConfigurationError(f"unknown wake episode: {episode_id}") from exc
        event_contract = {
            "manual-resume": (
                OperationalEventKind.USAGE_RESUME,
                OperationalEventStatus.USAGE_RESUMED,
            ),
            "campaign-completed": (
                OperationalEventKind.CAMPAIGN_COMPLETED,
                OperationalEventStatus.CAMPAIGN_COMPLETED,
            ),
            "operator-cancelled": (
                OperationalEventKind.OPERATOR_CANCELLED,
                OperationalEventStatus.OPERATOR_CANCELLED,
            ),
        }
        try:
            event_kind, status = event_contract[resolution]
        except KeyError as exc:
            raise WakeupConfigurationError(f"unknown wake resolution: {resolution}") from exc
        event = _episode_event(
            episode,
            event_kind=event_kind,
            status=status,
            context=context,
            created_at=created_at,
            identity={"resolution": resolution},
            details={"source": resolution},
        )
        record_operational_event(self.ledger, event)
        definition = render_wakeup_definition(self.root, episode, self.backend)
        return self._deactivate(definition, context=context, created_at=created_at)

    def ingest_fire_intents(
        self,
        *,
        context: OperationalContext,
        created_at: str,
    ) -> int:
        """Ingest fsynced lock-contention intents under the live driver lock.

        Parameters
        ----------
        context:
            Current live-driver operational context.
        created_at:
            Ingestion timestamp.

        Returns
        -------
        int
            Number of distinct durable intent files consumed.
        """

        _parse_utc(created_at)
        consumed = 0
        intent_root = self.root / "fire-intents"
        for path in sorted(intent_root.glob("*/*.json")) if intent_root.exists() else ():
            payload = _load_fire_intent(path)
            episode_id = str(payload["episode_id"])
            projection = self.projection
            if episode_id not in projection.episodes:
                raise WakeupConfigurationError(f"fire intent names unknown episode: {episode_id}")
            episode = projection.episodes[episode_id].episode
            bucket = int(payload["time_bucket"])
            fired_at = str(payload["fired_at"])
            if bucket in projection.episodes[episode_id].fire_buckets:
                path.unlink()
                fsync_directory(path.parent)
                consumed += 1
                continue
            event = _episode_event(
                episode,
                event_kind=OperationalEventKind.WAKEUP_FIRED,
                status=OperationalEventStatus.WAKEUP_FIRED,
                context=context,
                created_at=fired_at,
                identity={"time_bucket": bucket},
                details={
                    "time_bucket": bucket,
                    "guarded_not_before": _parse_utc(fired_at) < _parse_utc(episode.not_before),
                    "source": "lock-contention-intent",
                },
            )
            record_operational_event(self.ledger, event)
            noop = _episode_event(
                episode,
                event_kind=OperationalEventKind.WAKE_NOOP_ALREADY_RUNNING,
                status=OperationalEventStatus.WAKE_NOOP_ALREADY_RUNNING,
                context=context,
                created_at=fired_at,
                identity={"time_bucket": bucket},
                details={
                    "time_bucket": bucket,
                    "authoritative_driver_lock_live": True,
                    "source": "lock-contention-intent",
                },
            )
            record_operational_event(self.ledger, noop)
            self._record_health_if_needed(episode, context=context, created_at=created_at)
            path.unlink()
            fsync_directory(path.parent)
            consumed += 1
        return consumed

    def _ensure_installed(
        self,
        definition: RenderedWakeupDefinition,
        *,
        context: OperationalContext,
        created_at: str,
    ) -> tuple[bool, bool, bool, AppendResult]:
        """Write, install, verify, and visibly record one projection."""

        _parse_utc(created_at)
        projection_matches = _definition_matches(definition)
        registration_matches = projection_matches and self.verifier(definition)
        episode_id = definition.episode.episode_id
        previously_projected = any(
            event.get("event_kind")
            in {
                OperationalEventKind.WAKEUP_INSTALLED.value,
                OperationalEventKind.WAKEUP_REPAIRED.value,
            }
            and isinstance(event.get("details"), Mapping)
            and event["details"].get("episode_id") == episode_id
            for event in self.ledger.records
        )
        created = not previously_projected
        repaired = previously_projected and not registration_matches
        if registration_matches and previously_projected:
            persisted = next(
                event
                for event in reversed(self.ledger.records)
                if event.get("event_kind")
                in {
                    OperationalEventKind.WAKEUP_INSTALLED.value,
                    OperationalEventKind.WAKEUP_REPAIRED.value,
                }
                and isinstance(event.get("details"), Mapping)
                and event["details"].get("episode_id") == episode_id
            )
            return False, False, True, AppendResult(persisted, appended=False)
        if not registration_matches:
            _write_definition(definition)
            try:
                self.installer(definition)
                verified = self.verifier(definition)
            except Exception as exc:
                self._record_install_failure(
                    definition.episode,
                    context=context,
                    created_at=created_at,
                    reason=str(exc),
                )
                raise WakeupConfigurationError(
                    f"cannot install recurring wake episode {definition.episode.episode_id}: {exc}"
                ) from exc
            if not verified:
                reason = "scheduler verification returned false after install"
                self._record_install_failure(
                    definition.episode,
                    context=context,
                    created_at=created_at,
                    reason=reason,
                )
                raise WakeupConfigurationError(
                    f"cannot verify recurring wake episode {definition.episode.episode_id}"
                )
        else:
            verified = True
        kind = (
            OperationalEventKind.WAKEUP_REPAIRED
            if repaired
            else OperationalEventKind.WAKEUP_INSTALLED
        )
        status = (
            OperationalEventStatus.WAKEUP_REPAIRED
            if repaired
            else OperationalEventStatus.WAKEUP_INSTALLED
        )
        event = _episode_event(
            definition.episode,
            event_kind=kind,
            status=status,
            context=context,
            created_at=created_at,
            identity={
                "backend": definition.backend.value,
                "definition_sha256": stable_hash(definition.scheduler_definition),
                "repair_at": created_at if repaired else None,
            },
            details={
                "backend": definition.backend.value,
                "definition_path": str(definition.definition_path),
                "definition_sha256": stable_hash(definition.scheduler_definition),
                "verified": verified,
                "recurring": True,
                "retry_interval_seconds": definition.episode.retry_interval_seconds,
            },
        )
        return created, repaired, verified, record_operational_event(self.ledger, event)

    def _record_install_failure(
        self,
        episode: WakeEpisode,
        *,
        context: OperationalContext,
        created_at: str,
        reason: str,
    ) -> None:
        """Append one visible fail-closed scheduler installation fact."""

        event = _episode_event(
            episode,
            event_kind=OperationalEventKind.WAKEUP_FAILED,
            status=OperationalEventStatus.RUNNER_FAILED,
            context=context,
            created_at=created_at,
            identity={"reason": reason, "failed_at": created_at},
            details={"backend": self.backend.value, "reason": reason, "verified": False},
        )
        record_operational_event(self.ledger, event)

    def _record_health_if_needed(
        self,
        episode: WakeEpisode,
        *,
        context: OperationalContext,
        created_at: str,
    ) -> None:
        """Emit one health fact without changing recurrence or episode state."""

        state = self.projection.episodes[episode.episode_id]
        if len(state.fire_buckets) < self.health_threshold_fires or state.health_degraded:
            return
        event = _episode_event(
            episode,
            event_kind=OperationalEventKind.WAKEUP_HEALTH_DEGRADED,
            status=OperationalEventStatus.WAKEUP_HEALTH_DEGRADED,
            context=context,
            created_at=created_at,
            identity={"threshold": self.health_threshold_fires},
            details={
                "fire_count": len(state.fire_buckets),
                "threshold": self.health_threshold_fires,
                "recurrence_retained": True,
            },
        )
        record_operational_event(self.ledger, event)

    def _deactivate(
        self,
        definition: RenderedWakeupDefinition,
        *,
        context: OperationalContext,
        created_at: str,
        time_bucket: Optional[int] = None,
    ) -> bool:
        """Deactivate only after reducer state has a durable resolving fact."""

        state = self.projection.episodes[definition.episode.episode_id]
        if state.active:
            raise WakeupConfigurationError(
                f"refusing to deactivate active episode {definition.episode.episode_id}"
            )
        try:
            self.deactivator(definition)
        except Exception as exc:
            self._record_install_failure(
                definition.episode,
                context=context,
                created_at=created_at,
                reason=f"deactivation failed: {exc}",
            )
            raise WakeupConfigurationError(
                f"cannot deactivate wake episode {definition.episode.episode_id}: {exc}"
            ) from exc
        _remove_definition_projection(definition)
        identity: JsonObject = {"terminal_event_id": state.terminal_event_id}
        if time_bucket is not None:
            identity["time_bucket"] = time_bucket
        event = _episode_event(
            definition.episode,
            event_kind=OperationalEventKind.WAKEUP_DEACTIVATED,
            status=OperationalEventStatus.WAKEUP_DEACTIVATED,
            context=context,
            created_at=created_at,
            identity=identity,
            details={
                "backend": definition.backend.value,
                "terminal_event_id": state.terminal_event_id,
                "time_bucket": time_bucket,
            },
        )
        return record_operational_event(self.ledger, event).appended

    def _projection_or_registration_exists(self, definition: RenderedWakeupDefinition) -> bool:
        """Return whether local or OS disposable state still exists."""

        return definition.definition_path.exists() or self.verifier(definition)


def render_wakeup_definition(
    wakeup_root: Path,
    episode: WakeEpisode,
    backend: WakeupBackend,
) -> RenderedWakeupDefinition:
    """Render an equivalent recurring backend projection.

    Parameters
    ----------
    wakeup_root:
        Gitignored definition root.
    episode:
        Frozen episode authority.
    backend:
        OS scheduler backend.

    Returns
    -------
    RenderedWakeupDefinition
        Deterministic disposable scheduler files.
    """

    _validate_episode(episode)
    suffix = {
        WakeupBackend.LAUNCHD: "plist",
        WakeupBackend.SYSTEMD_TIMER: "timer",
        WakeupBackend.CRON: "cron",
    }[backend]
    path = wakeup_root / f"{episode.episode_id}.{suffix}"
    if backend is WakeupBackend.LAUNCHD:
        args = "".join(f"<string>{_xml_escape(arg)}</string>" for arg in episode.callback_argv)
        content = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            '<plist version="1.0"><dict>'
            f"<key>Label</key><string>{_launchd_label(episode)}</string>"
            f"<key>ProgramArguments</key><array>{args}</array>"
            f"<key>StartInterval</key><integer>{episode.retry_interval_seconds}</integer>"
            "<key>RunAtLoad</key><true/></dict></plist>\n"
        )
        return RenderedWakeupDefinition(episode, backend, path, content, None, None)
    if backend is WakeupBackend.SYSTEMD_TIMER:
        service_path = path.with_suffix(".service")
        timer = (
            "[Unit]\nDescription=TorchLens crawler recurring usage-limit wake\n"
            "[Timer]\n"
            f"OnUnitInactiveSec={episode.retry_interval_seconds}s\n"
            "Persistent=true\n"
            f"Unit={service_path.name}\n"
            "[Install]\nWantedBy=timers.target\n"
        )
        service = (
            "[Unit]\nDescription=TorchLens crawler wake callback\n"
            "[Service]\nType=oneshot\n"
            f"ExecStart={shlex.join(episode.callback_argv)}\n"
        )
        return RenderedWakeupDefinition(episode, backend, path, timer, service_path, service)
    minutes = episode.retry_interval_seconds // 60
    command = shlex.join(episode.callback_argv)
    marker = _cron_marker(episode)
    content = f"{marker} begin\nCRON_TZ=UTC\n*/{minutes} * * * * {command}\n{marker} end\n"
    return RenderedWakeupDefinition(episode, backend, path, content, None, None)


def write_fire_intent(
    wakeup_root: Path,
    episode: WakeEpisode,
    *,
    fired_at: str,
) -> Path:
    """Atomically fsync one lock-contention callback intent per time bucket.

    Parameters
    ----------
    wakeup_root:
        Gitignored wake runtime root.
    episode:
        Frozen episode read from canonical operational history.
    fired_at:
        Actual callback timestamp.

    Returns
    -------
    Path
        Durable intent path. Replays return the same path.

    Raises
    ------
    WakeupConfigurationError
        If a same-bucket intent has conflicting bytes.
    """

    fired = _parse_utc(fired_at)
    bucket = _time_bucket(fired, episode.retry_interval_seconds)
    payload: JsonObject = {
        "episode_id": episode.episode_id,
        "time_bucket": bucket,
        "fired_at": fired_at,
        "callback_identity": episode.callback_identity,
    }
    path = wakeup_root / "fire-intents" / episode.episode_id / f"{bucket}.json"
    encoded = canonical_json_bytes(payload) + b"\n"
    if path.exists():
        existing = _load_fire_intent(path)
        if (
            existing["episode_id"] != episode.episode_id
            or existing["time_bucket"] != bucket
            or existing["callback_identity"] != episode.callback_identity
        ):
            raise WakeupConfigurationError(f"conflicting fire intent: {path}")
        return path
    _write_bytes_atomic(path, encoded)
    return path


def _same_episode_request(prior: WakeEpisode, candidate: WakeEpisode) -> bool:
    """Return whether a candidate is an exact replay apart from supersession."""

    return bool(
        prior.provider == candidate.provider
        and prior.reset_at == candidate.reset_at
        and prior.reset_observation == candidate.reset_observation
        and prior.not_before == candidate.not_before
        and prior.retry_interval_seconds == candidate.retry_interval_seconds
        and prior.callback_identity == candidate.callback_identity
    )


def _episode_payload(episode: WakeEpisode) -> JsonObject:
    """Return the JSON-safe exact frozen episode payload."""

    payload = asdict(episode)
    payload["callback_argv"] = list(episode.callback_argv)
    return payload


def _episode_from_payload(payload: Mapping[str, object]) -> WakeEpisode:
    """Parse and validate an exact frozen episode payload."""

    expected = {
        "episode_id",
        "provider",
        "reset_at",
        "reset_observation",
        "not_before",
        "retry_interval_seconds",
        "callback_identity",
        "callback_argv",
        "opened_event_id",
        "supersedes_episode_id",
    }
    if set(payload) != expected:
        raise WakeupConfigurationError(
            f"wake episode payload fields differ: expected={sorted(expected)}, got={sorted(payload)}"
        )
    argv = payload["callback_argv"]
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise WakeupConfigurationError("wake episode callback_argv must be a string list")
    supersedes = payload["supersedes_episode_id"]
    if supersedes is not None and not isinstance(supersedes, str):
        raise WakeupConfigurationError("wake episode supersedes ID must be a string or null")
    retry_interval = payload["retry_interval_seconds"]
    if not isinstance(retry_interval, int):
        raise WakeupConfigurationError("wake episode retry interval must be an integer")
    episode = WakeEpisode(
        episode_id=str(payload["episode_id"]),
        provider=str(payload["provider"]),
        reset_at=str(payload["reset_at"]),
        reset_observation=str(payload["reset_observation"]),
        not_before=str(payload["not_before"]),
        retry_interval_seconds=retry_interval,
        callback_identity=str(payload["callback_identity"]),
        callback_argv=tuple(argv),
        opened_event_id=str(payload["opened_event_id"]),
        supersedes_episode_id=supersedes,
    )
    _validate_episode(episode)
    return episode


def _validate_episode(episode: WakeEpisode) -> None:
    """Validate frozen episode fields without creating another authority type."""

    _validate_provider(episode.provider)
    _validate_reset_observation(episode.reset_observation)
    _parse_utc(episode.reset_at)
    _parse_utc(episode.not_before)
    _validate_retry_interval(episode.retry_interval_seconds)
    if not episode.episode_id or episode.opened_event_id != f"usage-pause-{episode.episode_id}":
        raise WakeupConfigurationError("wake episode/opening event identity is invalid")
    if not episode.callback_argv or any(not argument for argument in episode.callback_argv):
        raise WakeupConfigurationError("wake episode callback argv is invalid")
    try:
        flag_index = episode.callback_argv.index("--wake-episode-id")
    except ValueError as exc:
        raise WakeupConfigurationError("wake episode callback omits its identity") from exc
    if (
        flag_index + 1 >= len(episode.callback_argv)
        or episode.callback_argv[flag_index + 1] != episode.episode_id
    ):
        raise WakeupConfigurationError("wake episode callback identity argument is invalid")
    base_argv = episode.callback_argv[:flag_index] + episode.callback_argv[flag_index + 2 :]
    if stable_hash({"callback_argv": list(base_argv)}) != episode.callback_identity:
        raise WakeupConfigurationError("wake episode callback identity hash is invalid")
    if episode.supersedes_episode_id == episode.episode_id:
        raise WakeupConfigurationError("wake episode cannot supersede itself")


def _validate_provider(provider: str) -> None:
    """Validate the closed provider vocabulary."""

    if provider not in {"anthropic", "openai"}:
        raise WakeupConfigurationError(f"unsupported usage-limit provider: {provider!r}")


def _validate_reset_observation(reset_observation: str) -> None:
    """Validate observed-versus-guessed reset evidence."""

    if reset_observation not in RESET_OBSERVATIONS:
        raise WakeupConfigurationError(
            f"reset_observation must be one of {sorted(RESET_OBSERVATIONS)}"
        )


def _validate_retry_interval(retry_interval_seconds: int) -> None:
    """Validate the frozen recurring cadence and cron parity."""

    if not MIN_RETRY_INTERVAL_SECONDS <= retry_interval_seconds <= MAX_RETRY_INTERVAL_SECONDS:
        raise WakeupConfigurationError("retry cadence must be between 300 and 900 seconds")
    if retry_interval_seconds % 60 != 0:
        raise WakeupConfigurationError("retry cadence must be an exact whole minute")
    if 3600 % retry_interval_seconds != 0:
        raise WakeupConfigurationError(
            "retry cadence must divide one hour for exact recurring cron parity"
        )


def _episode_event(
    episode: WakeEpisode,
    *,
    event_kind: OperationalEventKind,
    status: OperationalEventStatus,
    context: OperationalContext,
    created_at: str,
    identity: Mapping[str, object],
    details: JsonObject,
) -> JsonObject:
    """Build one deterministic episode-linked operational event."""

    _parse_utc(created_at)
    event_identity = stable_hash(
        {
            "episode_id": episode.episode_id,
            "event_kind": event_kind.value,
            **dict(identity),
        }
    ).removeprefix("sha256:")[:24]
    return _base_event(
        event_id=f"{event_kind.value}-{event_identity}",
        created_at=created_at,
        event_kind=event_kind,
        status=status,
        context=context,
        provider=episode.provider,
        observed_response=None,
        reset_at=episode.reset_at,
        details={"episode_id": episode.episode_id, **details},
    )


def _base_event(
    *,
    event_id: str,
    created_at: str,
    event_kind: OperationalEventKind,
    status: OperationalEventStatus,
    context: OperationalContext,
    provider: Optional[str],
    observed_response: Optional[str],
    reset_at: Optional[str],
    details: JsonObject,
) -> JsonObject:
    """Build the strict common operational-event envelope."""

    return {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "event_id": event_id,
        "created_at": created_at,
        "event_kind": event_kind.value,
        "status": status.value,
        "provider": provider,
        "observed_response": observed_response,
        "reset_at": reset_at,
        "queued_work_counts": dict(context.queued_work_counts),
        "current_environment": context.current_environment,
        "run_id": context.run_id,
        "machine_id": context.machine_id,
        "details": details,
    }


def _definition_matches(definition: RenderedWakeupDefinition) -> bool:
    """Return whether every local projection byte matches the renderer."""

    if not definition.definition_path.is_file():
        return False
    if definition.definition_path.read_text(encoding="utf-8") != definition.scheduler_definition:
        return False
    if definition.service_path is None:
        return True
    return bool(
        definition.service_path.is_file()
        and definition.service_path.read_text(encoding="utf-8") == definition.service_definition
    )


def _write_definition(definition: RenderedWakeupDefinition) -> None:
    """Atomically replace disposable definition projections and fsync them."""

    _write_bytes_atomic(definition.definition_path, definition.scheduler_definition.encode("utf-8"))
    if definition.service_path is not None and definition.service_definition is not None:
        _write_bytes_atomic(definition.service_path, definition.service_definition.encode("utf-8"))


def _remove_definition_projection(definition: RenderedWakeupDefinition) -> None:
    """Remove local projections only after scheduler deactivation succeeds."""

    definition.definition_path.unlink(missing_ok=True)
    if definition.service_path is not None:
        definition.service_path.unlink(missing_ok=True)
    if definition.definition_path.parent.exists():
        fsync_directory(definition.definition_path.parent)


def _install_wakeup(definition: RenderedWakeupDefinition) -> None:
    """Install or repair one recurring backend definition."""

    if definition.backend is WakeupBackend.LAUNCHD:
        domain = f"gui/{os.getuid()}"
        label = _launchd_label(definition.episode)
        subprocess.run(
            ["launchctl", "bootout", f"{domain}/{label}"],
            check=False,
            capture_output=True,
            text=True,
        )
        _run_checked(["launchctl", "bootstrap", domain, str(definition.definition_path)])
        return
    if definition.backend is WakeupBackend.SYSTEMD_TIMER:
        if definition.service_path is None:
            raise WakeupConfigurationError("systemd timer projection omits its service")
        _run_checked(
            [
                "systemctl",
                "--user",
                "link",
                str(definition.service_path),
                str(definition.definition_path),
            ]
        )
        _run_checked(["systemctl", "--user", "daemon-reload"])
        _run_checked(["systemctl", "--user", "enable", "--now", definition.definition_path.name])
        return
    current = _read_crontab()
    stripped = _remove_cron_block(current, definition.episode)
    if stripped and not stripped.endswith("\n"):
        stripped += "\n"
    _write_crontab(stripped + definition.scheduler_definition)


def _verify_wakeup(definition: RenderedWakeupDefinition) -> bool:
    """Return whether the OS scheduler has the exact recurring projection active."""

    if not _definition_matches(definition):
        return False
    if definition.backend is WakeupBackend.LAUNCHD:
        domain = f"gui/{os.getuid()}"
        result = subprocess.run(
            ["launchctl", "print", f"{domain}/{_launchd_label(definition.episode)}"],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    if definition.backend is WakeupBackend.SYSTEMD_TIMER:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", definition.definition_path.name],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and result.stdout.strip() == "active"
    return definition.scheduler_definition in _read_crontab()


def _deactivate_wakeup(definition: RenderedWakeupDefinition) -> None:
    """Remove one recurring backend definition idempotently."""

    if definition.backend is WakeupBackend.LAUNCHD:
        domain = f"gui/{os.getuid()}"
        result = subprocess.run(
            ["launchctl", "bootout", f"{domain}/{_launchd_label(definition.episode)}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode not in {0, 3, 113}:
            raise WakeupConfigurationError(f"launchd deactivation failed: {result.stderr.strip()}")
        return
    if definition.backend is WakeupBackend.SYSTEMD_TIMER:
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "disable",
                "--now",
                definition.definition_path.name,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 and "not loaded" not in result.stderr.lower():
            raise WakeupConfigurationError(f"systemd deactivation failed: {result.stderr.strip()}")
        return
    current = _read_crontab()
    updated = _remove_cron_block(current, definition.episode)
    if updated != current:
        _write_crontab(updated)


def _launchd_label(episode: WakeEpisode) -> str:
    """Return the deterministic launchd label for an episode."""

    return f"org.torchlens.crawler.{episode.episode_id}"


def _cron_marker(episode: WakeEpisode) -> str:
    """Return the unique cron projection marker."""

    return f"# torchlens-wake-episode {episode.episode_id}"


def _read_crontab() -> str:
    """Read the current user crontab, treating absence as empty."""

    current = subprocess.run(["crontab", "-l"], check=False, capture_output=True, text=True)
    if current.returncode not in {0, 1}:
        raise WakeupConfigurationError(f"cannot inspect current crontab: {current.stderr.strip()}")
    return current.stdout if current.returncode == 0 else ""


def _write_crontab(content: str) -> None:
    """Replace the user crontab without invoking a shell."""

    installed = subprocess.run(
        ["crontab", "-"], input=content, check=False, capture_output=True, text=True
    )
    if installed.returncode != 0:
        raise WakeupConfigurationError(
            f"cannot install recurring cron wakeup: {installed.stderr.strip()}"
        )


def _remove_cron_block(content: str, episode: WakeEpisode) -> str:
    """Remove exactly one marked cron episode block."""

    begin = f"{_cron_marker(episode)} begin"
    end = f"{_cron_marker(episode)} end"
    output: list[str] = []
    skipping = False
    for line in content.splitlines(keepends=True):
        if line.rstrip("\n") == begin:
            skipping = True
            continue
        if skipping and line.rstrip("\n") == end:
            skipping = False
            continue
        if not skipping:
            output.append(line)
    if skipping:
        raise WakeupConfigurationError(f"unterminated cron wake block for {episode.episode_id}")
    return "".join(output)


def _run_checked(argv: Sequence[str]) -> None:
    """Run one scheduler argv and raise a typed error on failure."""

    completed = subprocess.run(list(argv), check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise WakeupConfigurationError(
            f"scheduler command {list(argv)!r} failed: {completed.stderr.strip()}"
        )


def _load_fire_intent(path: Path) -> JsonObject:
    """Load and validate one exact local fire-intent object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WakeupConfigurationError(f"invalid fire intent {path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "episode_id",
        "time_bucket",
        "fired_at",
        "callback_identity",
    }:
        raise WakeupConfigurationError(f"invalid fire intent fields: {path}")
    if not isinstance(payload["time_bucket"], int) or payload["time_bucket"] < 0:
        raise WakeupConfigurationError(f"invalid fire intent bucket: {path}")
    _parse_utc(str(payload["fired_at"]))
    return payload


def _time_bucket(instant: datetime, interval_seconds: int) -> int:
    """Return the deterministic UTC recurrence bucket."""

    return int(instant.timestamp()) // interval_seconds


def _parse_utc(value: str) -> datetime:
    """Parse an exact RFC 3339 UTC timestamp."""

    if not value.endswith("Z"):
        raise WakeupConfigurationError("wakeup timestamps must be RFC 3339 UTC ending in Z")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise WakeupConfigurationError(f"invalid wakeup timestamp {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise WakeupConfigurationError("wakeup timestamps must use exact UTC")
    return parsed


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    """Atomically write and fsync one disposable runtime file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _xml_escape(value: str) -> str:
    """Escape an argv leaf for a launchd plist."""

    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
