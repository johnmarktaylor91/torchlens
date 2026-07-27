"""Typed command-line interface for the resumable menagerie crawler."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import shlex
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

from menagerie.crawler.authority import AuthorityContext, build_authority_context
from menagerie.crawler.checkpoint import (
    CheckpointError,
    append_canonical_requeue_grant,
    build_canonical_requeue_grant,
    canonical_operational_ledger_path,
    create_canonical_checkpoint,
)
from menagerie.crawler.constants import (
    InvocationOrigin,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventStatus,
)
from menagerie.crawler.doctor import DoctorConfig, DoctorError, DoctorProbes, run_doctor
from menagerie.crawler.driver import (
    CommandAuthorLane,
    CommandCheckerLane,
    CommandNotifier,
    CrawlerDriver,
    DriverConfig,
    DriverDependencies,
    DriverError,
    DriverLock,
    DriverLockError,
    DriverPaths,
    DriverResult,
    SupervisedForwardLane,
    build_command_environment_lane,
    default_driver_paths,
    utc_now,
)
from menagerie.crawler.envs import load_environment_registry
from menagerie.crawler.intake import IntakeSnapshot, create_intake_snapshot, load_intake_snapshot
from menagerie.crawler.partitioner import (
    DEFAULT_CAMPAIGN_MANIFEST,
    CampaignBinding,
    emit_campaign_partitions,
    find_campaign_binding,
)
from menagerie.crawler.recordio import SingleWriterError
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.reducer import materialize_current
from menagerie.crawler.routing import phase_routes, route_model
from menagerie.crawler.status import funnel_counts, partition_report, wakeup_status
from menagerie.crawler.wakeup import (
    OperationalContext,
    WakeupManager,
    reduce_wake_episodes,
    render_wakeup_status,
    write_fire_intent,
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_LOCKED = 3
EXIT_PAUSED = 4


class DriverFactory(Protocol):
    """CLI injection point for fully configured live or fake driver lanes."""

    def __call__(self, args: argparse.Namespace) -> CrawlerDriver:
        """Build a driver for parsed run/resume/handoff arguments."""

        ...


def build_parser() -> argparse.ArgumentParser:
    """Build the complete typed crawler argument parser."""

    parser = argparse.ArgumentParser(prog="python -m menagerie.crawler", description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path.cwd(), help="TorchLens worktree root"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="run the strict campaign preflight")
    doctor.add_argument("--target", choices=("osx-arm64", "linux-x86_64-cuda"), required=True)
    doctor.add_argument(
        "--strict", action="store_true", help="enforce every frozen preflight check"
    )
    doctor.add_argument("--minimum-disk-gib", type=int, default=100)

    intake = subparsers.add_parser("intake", help="create an immutable intake snapshot")
    intake.add_argument("--all-existing", action="store_true", help="use menagerie/data catalogs")
    intake.add_argument("--master", type=Path)
    intake.add_argument("--deferred", type=Path)
    intake.add_argument("--stable-ids", type=Path)
    intake.add_argument("--output-root", type=Path)
    intake.add_argument("--snapshot-date", help="operator label retained by surrounding automation")

    campaigns = subparsers.add_parser(
        "partition-campaigns",
        help="emit the four tier-aligned campaign partitions and intake snapshots",
    )
    campaigns.add_argument("--roster", type=Path)
    campaigns.add_argument("--stable-ids", type=Path)
    campaigns.add_argument("--output-root", type=Path)

    plan = subparsers.add_parser("plan", help="print deterministic phase/intent assignments")
    _add_intake_argument(plan)
    _add_target_phase_arguments(plan)

    run = subparsers.add_parser("run", help="run or resume the single-writer scheduler")
    _add_intake_argument(run, required=False)
    _add_target_phase_arguments(run)
    _add_driver_config_arguments(run)
    _add_dry_run_arguments(run)
    run.add_argument("--resume", action="store_true", help="resume durable unsatisfied identities")
    run.add_argument(
        "--sequential-envs",
        action="store_true",
        default=True,
        help="required one-at-a-time environment lifecycle",
    )

    status = subparsers.add_parser("status", help="print the current terminal funnel")
    _add_intake_argument(status)
    status.add_argument("--records-root", type=Path)
    status.add_argument("--full", action="store_true")
    status.add_argument("--verify-partition", action="store_true")

    wake = subparsers.add_parser("wake", help="inspect or cancel durable wake episodes")
    wake.add_argument("--records-root", type=Path)
    wake_actions = wake.add_subparsers(dest="wake_command", required=True)
    wake_actions.add_parser("status", help="print durable wake episode state")
    wake_cancel = wake_actions.add_parser("cancel", help="cancel one exact wake episode")
    wake_cancel.add_argument("episode_id")

    checkpoint = subparsers.add_parser("checkpoint", help="verify ledgers and disposable views")
    _add_intake_argument(checkpoint)
    checkpoint.add_argument("--records-root", type=Path)
    checkpoint.add_argument("--verify-ledgers", action="store_true")
    checkpoint.add_argument("--verify-views", action="store_true")

    teardown = subparsers.add_parser(
        "teardown", help="remove only disposable crawler runtime state"
    )
    teardown.add_argument("--target", choices=("osx-arm64", "linux-x86_64-cuda"), required=True)
    teardown.add_argument("--active-env", action="store_true")
    teardown.add_argument("--verify-disk", action="store_true")

    requeue = subparsers.add_parser("requeue", help="record an explicit bounded effort grant")
    requeue.add_argument("stable_id")
    requeue.add_argument("--reason", required=True)
    requeue.add_argument("--grant", required=True, type=int)
    requeue.add_argument("--stage", required=True)
    requeue.add_argument("--granted-by", default="operator")
    requeue.add_argument(
        "--intent",
        dest="target_intent",
        help="dependency-evidenced corrected environment intent",
    )

    handoff = subparsers.add_parser(
        "handoff", aliases=["handoff-linux"], help="run Linux deferred sweep"
    )
    _add_intake_argument(handoff)
    _add_driver_config_arguments(handoff)
    handoff.add_argument("--resume", action="store_true")
    handoff.add_argument("--only-status", default="deferred:*")
    handoff.set_defaults(target="linux-x86_64-cuda", phase=None)

    resume = subparsers.add_parser("resume", help="resume a paused campaign")
    _add_intake_argument(resume, required=False)
    _add_target_phase_arguments(resume)
    _add_driver_config_arguments(resume)
    _add_dry_run_arguments(resume)
    resume.add_argument(
        "--after-review",
        action="store_true",
        help="record human sign-off for the one-shot review checkpoint",
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    driver_factory: Optional[DriverFactory] = None,
    doctor_probes: Optional[DoctorProbes] = None,
) -> int:
    """Parse and execute one crawler command with stable exit codes."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor_command(args, doctor_probes)
        if args.command == "intake":
            return _intake_command(args)
        if args.command == "partition-campaigns":
            return _partition_campaigns_command(args)
        if args.command == "plan":
            return _plan_command(args)
        if args.command == "status":
            return _status_command(args)
        if args.command == "wake":
            return _wake_command(args)
        if args.command == "checkpoint":
            return _checkpoint_command(args)
        if args.command == "teardown":
            return _teardown_command(args)
        if args.command == "requeue":
            return _requeue_command(args)
        if args.command in {"run", "resume", "handoff", "handoff-linux"}:
            return _driver_command(args, driver_factory or _default_driver_factory)
    except DoctorError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ERROR
    except DriverLockError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_LOCKED
    except (CheckpointError, DriverError, OSError, ValueError) as exc:
        print(f"crawler error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_USAGE


def _add_intake_argument(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    """Add the shared required intake snapshot path."""

    parser.add_argument("--intake", type=Path, required=required, help="intake snapshot directory")


def _add_dry_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the opt-in fixture-corpus acceptance harness arguments."""

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the tiny-model acceptance corpus with deterministic agent lanes and a real prefix",
    )
    parser.add_argument(
        "--dry-run-root",
        type=Path,
        help="isolated dry-run campaign root (defaults under .crawl-local)",
    )
    parser.add_argument(
        "--dry-run-environment-prefix",
        type=Path,
        help="real materialized fixture prefix required by the acceptance dry-run",
    )


def _add_target_phase_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared target and ordered phase arguments."""

    parser.add_argument("--target", choices=("osx-arm64", "linux-x86_64-cuda"), default="osx-arm64")
    parser.add_argument("--phase", choices=("pytorch", "native-tail"))


def _add_driver_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared checkpoint, milestone, and notification configuration."""

    parser.add_argument("--review-checkpoint-at", type=int, default=1000)
    parser.add_argument(
        "--progress-milestones",
        type=_positive_csv,
        default=(2000, 3000, 5000, 10000, 15000, 20000),
        metavar="N,N,...",
    )
    parser.add_argument("--notify-command")
    parser.add_argument("--run-id", default="crawler-run")
    parser.add_argument("--wake-episode-id", help=argparse.SUPPRESS)
    parser.add_argument(
        "--author-command",
        default=os.environ.get("MENAGERIE_AUTHOR_COMMAND"),
        help="Claude Code wrapper command; defaults to MENAGERIE_AUTHOR_COMMAND",
    )
    parser.add_argument(
        "--checker-command",
        default=os.environ.get("MENAGERIE_CHECKER_COMMAND"),
        help="Codex wrapper command; defaults to MENAGERIE_CHECKER_COMMAND",
    )
    parser.add_argument(
        "--environment-command",
        default=os.environ.get("MENAGERIE_ENVIRONMENT_COMMAND"),
        help="exact-lock lifecycle wrapper; defaults to MENAGERIE_ENVIRONMENT_COMMAND",
    )


def _doctor_command(args: argparse.Namespace, probes: Optional[DoctorProbes]) -> int:
    """Run and print the strict doctor report."""

    config = DoctorConfig(
        repo_root=args.repo_root.resolve(),
        runtime_root=(args.repo_root / ".crawl-local").resolve(),
        target=args.target,
        minimum_disk_bytes=args.minimum_disk_gib * 1024**3,
        strict=args.strict,
    )
    report = run_doctor(config, probes)
    print(json.dumps({"passed": report.passed, "checks": report.checks}, sort_keys=True))
    return EXIT_OK


def _intake_command(args: argparse.Namespace) -> int:
    """Create an immutable intake snapshot from explicit or conventional inputs."""

    data_root = args.repo_root / "menagerie" / "data"
    if args.master is not None:
        master = args.master
    elif args.all_existing:
        # Prefer the full crawl roster (every found model: built + classics +
        # to-build) when it is present; fall back to the built-only catalog.
        roster = data_root / "crawl_roster.jsonl"
        master = roster if roster.exists() else data_root / "master_catalog.jsonl"
    else:
        master = None
    deferred = args.deferred or (data_root / "deferred.jsonl" if args.all_existing else None)
    if master is None or deferred is None:
        raise ValueError("intake requires --master/--deferred or --all-existing")
    output = args.output_root or args.repo_root / "menagerie" / "crawler" / "records" / "intake"
    stable_ids = args.stable_ids or data_root / "stable_ids.jsonl"
    snapshot = create_intake_snapshot(master, deferred, output, stable_ids_path=stable_ids)
    print(
        json.dumps(
            {
                "snapshot_id": snapshot.snapshot_id,
                "items": len(snapshot.items),
                "created": snapshot.created,
                "path": str(snapshot.root),
            },
            sort_keys=True,
        )
    )
    return EXIT_OK


def _partition_campaigns_command(args: argparse.Namespace) -> int:
    """Emit the four immutable tier-aligned campaign inputs."""

    data_root = args.repo_root / "menagerie" / "data"
    records_root = args.repo_root / "menagerie" / "crawler" / "records"
    bindings = emit_campaign_partitions(
        args.roster or data_root / "crawl_roster.jsonl",
        args.output_root or records_root,
        stable_ids_path=args.stable_ids or data_root / "stable_ids.jsonl",
    )
    print(
        json.dumps(
            {
                "campaigns": {
                    binding.spec.campaign_id: {
                        "author_model": binding.spec.author_model,
                        "checker_model": binding.spec.checker_model,
                        "items": binding.row_count,
                        "intake": binding.intake_path,
                    }
                    for binding in bindings
                }
            },
            sort_keys=True,
        )
    )
    return EXIT_OK


def _plan_command(args: argparse.Namespace) -> int:
    """Print deterministic phase-first environment assignments."""

    snapshot = load_intake_snapshot(args.intake)
    registry = load_environment_registry(target=args.target)
    routes = phase_routes(
        route_model(item.to_model_requirements(_framework(item.name, item.zoo)))
        for item in snapshot.items
    )
    if args.phase is not None:
        routes = tuple(route for route in routes if route.phase.value == args.phase)
    payload = {
        "target": args.target,
        "phase_order": [phase.value for phase in registry.phase_order],
        "assignments": [
            {"stable_id": route.stable_id, "intent": route.intent, "phase": route.phase.value}
            for route in routes
        ],
    }
    print(json.dumps(payload, sort_keys=True))
    return EXIT_OK


def _driver_command(args: argparse.Namespace, factory: DriverFactory) -> int:
    """Run a fully configured driver and map pauses to a stable exit code."""

    driver = factory(args)
    after_review = bool(getattr(args, "after_review", False))
    raw_episode_id = getattr(args, "wake_episode_id", None)
    episode_id = raw_episode_id if isinstance(raw_episode_id, str) else None
    origin = (
        InvocationOrigin.WAKE_CALLBACK
        if episode_id is not None
        else InvocationOrigin.MANUAL_RESUME
        if args.command == "resume" or bool(getattr(args, "resume", False))
        else InvocationOrigin.ORDINARY_RUN
    )
    driver.config = replace(
        driver.config,
        invocation_origin=origin,
        wake_episode_id=episode_id,
    )
    try:
        result: DriverResult = driver.run(after_review=after_review)
    except DriverLockError:
        if origin is not InvocationOrigin.WAKE_CALLBACK:
            raise
        if not isinstance(episode_id, str):
            raise AssertionError("validated wake episode identity was lost")
        fired_at = utc_now()
        _record_scheduled_wake_intent(driver, episode_id, fired_at)
        print(
            json.dumps(
                {
                    "status": OperationalEventStatus.WAKE_NOOP_ALREADY_RUNNING.value,
                    "episode_id": episode_id,
                },
                sort_keys=True,
            )
        )
        return EXIT_OK
    output: dict[str, object] = dict(result.__dict__)
    if result.shutdown_interruption is not None:
        output["shutdown_interruption"] = asdict(result.shutdown_interruption)
    if bool(getattr(args, "dry_run", False)):
        snapshot = load_intake_snapshot(driver.paths.intake_root)
        attempts = scan_jsonl(driver.paths.ledgers.attempts)
        context = _snapshot_authority_context(snapshot, driver.config)
        context = replace(
            context,
            environment_generations=_persisted_environment_generations(attempts),
        )
        current = materialize_current(
            driver.paths.ledgers,
            context=context,
        )
        acceptance = _dry_run_acceptance(
            snapshot,
            current,
            attempts,
            paused=result.paused_reason is not None,
        )
        output.update(
            {
                "dry_run": True,
                "funnel": funnel_counts(current),
                "acceptance": acceptance,
            }
        )
    print(json.dumps(output, sort_keys=True))
    if result.paused_reason is not None:
        return EXIT_PAUSED
    if bool(getattr(args, "dry_run", False)) and acceptance["status"] != "passed":
        return EXIT_ERROR
    return EXIT_OK


def _record_scheduled_wake_intent(driver: CrawlerDriver, episode_id: str, fired_at: str) -> None:
    """Fsync one idempotent fire intent when the live driver owns its lock.

    Parameters
    ----------
    driver:
        Lock-contended driver exposing canonical and gitignored runtime paths.
    episode_id, fired_at:
        Exact frozen episode identity and actual callback timestamp.
    """

    path = canonical_operational_ledger_path(driver.paths.ledgers.models)
    projection = reduce_wake_episodes(scan_jsonl(path))
    try:
        episode = projection.episodes[episode_id].episode
    except KeyError as exc:
        raise ValueError(f"scheduled callback names unknown wake episode: {episode_id}") from exc
    write_fire_intent(driver.paths.wakeup_root, episode, fired_at=fired_at)


def _default_driver_factory(args: argparse.Namespace) -> CrawlerDriver:
    """Build production command lanes from explicit flags or supervisor environment."""

    if bool(getattr(args, "dry_run", False)):
        from menagerie.crawler.tests.dry_run_support import build_dry_run_driver

        dry_run_root = args.dry_run_root or args.repo_root / ".crawl-local" / "dry-run"
        environment_prefix = getattr(args, "dry_run_environment_prefix", None)
        if not isinstance(environment_prefix, Path):
            if os.environ.get("MENAGERIE_RELEASE_GATE") == "1":
                raise ValueError("unmet-release-gate: --dry-run-environment-prefix is required")
            raise ValueError("dry-run requires --dry-run-environment-prefix")
        return build_dry_run_driver(
            args.repo_root.resolve(),
            dry_run_root.resolve(),
            environment_prefix.resolve(),
            review_checkpoint_at=args.review_checkpoint_at,
            progress_milestones=tuple(args.progress_milestones),
            run_id=args.run_id,
        )
    if getattr(args, "dry_run_environment_prefix", None) is not None:
        raise ValueError("--dry-run-environment-prefix requires --dry-run")
    if args.intake is None:
        raise ValueError("run/resume requires --intake unless --dry-run is selected")
    author_command = _required_command(args.author_command, "author")
    checker_command = _required_command(args.checker_command, "checker")
    environment_command = _required_command(args.environment_command, "environment")
    paths = default_driver_paths(args.repo_root.resolve(), args.intake.resolve())
    snapshot = load_intake_snapshot(args.intake.resolve())
    campaign_binding = _optional_campaign_binding(args.repo_root.resolve(), snapshot)
    default_config = DriverConfig()
    config = DriverConfig(
        target=args.target,
        phase=args.phase,
        run_id=args.run_id,
        review_checkpoint_at=args.review_checkpoint_at,
        progress_milestones=tuple(args.progress_milestones),
        notify_command=args.notify_command,
        author_model=(
            campaign_binding.spec.author_model
            if campaign_binding is not None
            else default_config.author_model
        ),
        checker_model=(
            campaign_binding.spec.checker_model
            if campaign_binding is not None
            else default_config.checker_model
        ),
        only_status=getattr(args, "only_status", None),
        invocation_origin=(
            InvocationOrigin.WAKE_CALLBACK
            if getattr(args, "wake_episode_id", None) is not None
            else InvocationOrigin.MANUAL_RESUME
            if args.command == "resume" or bool(getattr(args, "resume", False))
            else InvocationOrigin.ORDINARY_RUN
        ),
        wake_episode_id=(
            str(args.wake_episode_id)
            if getattr(args, "wake_episode_id", None) is not None
            else None
        ),
    )
    dependencies = DriverDependencies(
        author=CommandAuthorLane(author_command),
        checker=CommandCheckerLane(checker_command),
        forward=SupervisedForwardLane(cwd=args.repo_root.resolve()),
        environments=build_command_environment_lane(environment_command, paths.runtime_root),
        notifier=CommandNotifier(config.notify_command),
        clock=utc_now,
    )
    return CrawlerDriver(paths, config, dependencies)


def _dry_run_acceptance(
    snapshot: IntakeSnapshot,
    current: Mapping[str, Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    *,
    paused: bool,
) -> dict[str, object]:
    """Evaluate semantic success for the tiny-model acceptance gate.

    Parameters
    ----------
    snapshot, current, attempts:
        Expected tiny intake, canonical current projection, and authenticated attempt ledger.
    paused:
        Whether the driver intentionally stopped at a resumable checkpoint.

    Returns
    -------
    dict[str, object]
        Structured pending, passed, or acceptance-failed result.
    """

    expected_ids = {item.stable_id for item in snapshot.items}
    runs_records = {
        stable_id: record
        for stable_id, record in current.items()
        if stable_id in expected_ids and record.get("status", {}).get("code") == "runs"
    }
    authenticated_attempt_ids: dict[str, set[str]] = {
        stable_id: set() for stable_id in expected_ids
    }
    for attempt in attempts:
        stable_id = attempt.get("stable_id")
        if not isinstance(stable_id, str) or stable_id not in expected_ids:
            continue
        raw_digest = attempt.get("raw_award_receipt_sha256")
        parent = attempt.get("parent_attestation")
        if (
            attempt.get("result") == "succeeded"
            and isinstance(attempt.get("raw_award_receipt"), Mapping)
            and isinstance(raw_digest, str)
            and isinstance(parent, Mapping)
            and parent.get("named_raw_award_receipt_sha256") == raw_digest
            and isinstance(attempt.get("attempt_id"), str)
        ):
            authenticated_attempt_ids[stable_id].add(str(attempt["attempt_id"]))
    authenticated_models = {
        stable_id
        for stable_id, record in runs_records.items()
        if authenticated_attempt_ids[stable_id]
        and set(record.get("execution", {}).get("accepted_attempt_ids", ()))
        == authenticated_attempt_ids[stable_id]
    }
    counts = {
        "expected_runnable_models": len(expected_ids),
        "runs_revisions": len(runs_records),
        "authenticated_attempt_models": len(authenticated_models),
    }
    if paused:
        return {"status": "pending", "reason_code": "driver-paused", **counts}
    if set(runs_records) == expected_ids and authenticated_models == expected_ids:
        return {"status": "passed", **counts}
    unexpected_statuses = Counter(
        str(record.get("status", {}).get("code", "missing-status"))
        for stable_id, record in current.items()
        if stable_id in expected_ids and stable_id not in runs_records
    )
    missing_records = expected_ids - set(current)
    if missing_records:
        unexpected_statuses["missing"] += len(missing_records)
    all_source_failure = (
        len(current) == len(expected_ids)
        and set(current) == expected_ids
        and set(unexpected_statuses) == {"failed:source"}
    )
    return {
        "status": "acceptance-failed",
        "reason_code": "all-source-failure" if all_source_failure else "expected-runs-missing",
        **counts,
        "unexpected_statuses": dict(sorted(unexpected_statuses.items())),
    }


def _persisted_environment_generations(
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Recover exact environment generations already authenticated by attempt rows.

    Parameters
    ----------
    attempts:
        Canonical attempt-ledger rows for the dry-run campaign.

    Returns
    -------
    dict[str, str]
        Environment-family to generation bindings needed for current projection.
    """

    generations: dict[str, str] = {}
    for attempt in attempts:
        environment = attempt.get("environment")
        if not isinstance(environment, Mapping):
            continue
        family = environment.get("family")
        identities = attempt.get("identities")
        generation = identities.get("environment") if isinstance(identities, Mapping) else None
        if not isinstance(family, str) or not isinstance(generation, str):
            continue
        existing = generations.get(family)
        if existing is not None and existing != generation:
            raise ValueError(f"dry-run attempts contain conflicting generations for {family}")
        generations[family] = generation
    return generations


def _required_command(value: Optional[str], lane: str) -> tuple[str, ...]:
    """Parse one required non-shell lane command with a clear configuration error."""

    if value is None or not value.strip():
        environment_name = f"MENAGERIE_{lane.upper()}_COMMAND"
        raise ValueError(f"{lane} lane requires --{lane}-command or {environment_name}")
    command = tuple(shlex.split(value))
    if not command:
        raise ValueError(f"{lane} lane command cannot be empty")
    return command


def _snapshot_authority_context(
    snapshot: IntakeSnapshot,
    config: DriverConfig,
) -> AuthorityContext:
    """Build current projection authority for a loaded intake snapshot."""

    return build_authority_context(
        active_intake_snapshot_id=snapshot.snapshot_id,
        active_intake_snapshot_sha256=snapshot.snapshot_sha256,
        intake_rows=(item.to_dict() for item in snapshot.items),
        author_model=config.author_model,
        author_version=config.author_version,
        checker_model=config.checker_model,
        checker_version=config.checker_version,
    )


def _optional_campaign_binding(
    repo_root: Path,
    snapshot: IntakeSnapshot,
) -> Optional[CampaignBinding]:
    """Return a campaign binding when the canonical manifest owns the snapshot."""

    manifest_path = repo_root / DEFAULT_CAMPAIGN_MANIFEST
    if not manifest_path.is_file():
        return None
    return find_campaign_binding(manifest_path, snapshot)


def _snapshot_driver_config(repo_root: Path, snapshot: IntakeSnapshot) -> DriverConfig:
    """Return exact campaign provenance labels for one intake snapshot."""

    binding = _optional_campaign_binding(repo_root, snapshot)
    if binding is None:
        return DriverConfig()
    return DriverConfig(
        review_checkpoint_at=binding.spec.review_checkpoint_at,
        author_model=binding.spec.author_model,
        checker_model=binding.spec.checker_model,
    )


def _status_command(args: argparse.Namespace) -> int:
    """Print current funnel and optional exact partition diagnostics."""

    snapshot = load_intake_snapshot(args.intake)
    records_root = args.records_root or args.repo_root / "menagerie" / "crawler" / "records"
    paths = default_driver_paths(args.repo_root, args.intake).ledgers
    if records_root != args.repo_root / "menagerie" / "crawler" / "records":
        paths = DriverPaths(args.repo_root / ".crawl-local", args.intake, paths).ledgers
        paths = type(paths)(
            records_root / "models" / "current-shard.jsonl",
            records_root / "attempts" / "local.jsonl",
            records_root / "gates" / "current-shard.jsonl",
        )
    current = materialize_current(
        paths,
        context=_snapshot_authority_context(
            snapshot,
            _snapshot_driver_config(args.repo_root, snapshot),
        ),
    )
    operational_path = canonical_operational_ledger_path(paths.models)
    wakeups = wakeup_status(scan_jsonl(operational_path))
    output: dict[str, object] = {
        "funnel": funnel_counts(current),
        "terminal": len(current),
        "wakeups": wakeups,
    }
    if args.verify_partition:
        report = partition_report((item.stable_id for item in snapshot.items), current)
        output["partition_valid"] = report.valid
        output["missing"] = sorted(report.missing_ids)
    if args.full:
        output["current"] = current
    print(json.dumps(output, sort_keys=True))
    return EXIT_OK


def _wake_command(args: argparse.Namespace) -> int:
    """Inspect or durably cancel one exact wake episode under the driver lock."""

    records_root = args.records_root or args.repo_root / "menagerie" / "crawler" / "records"
    operational_path = records_root / "operational" / "events.jsonl"
    if args.wake_command == "status":
        print(json.dumps(render_wakeup_status(reduce_wake_episodes(scan_jsonl(operational_path)))))
        return EXIT_OK

    runtime = args.repo_root / ".crawl-local"
    owner = {
        "pid": os.getpid(),
        "run_id": "wake-cancel",
        "target": "wake-control",
        "created_at": utc_now(),
        "episode_id": args.episode_id,
    }
    with DriverLock(runtime / "locks" / "driver.lock", owner):
        with JsonlLedger(operational_path, OPERATIONAL_EVENT_SCHEMA_VERSION) as operational:
            projection = reduce_wake_episodes(operational.records)
            try:
                state = projection.episodes[args.episode_id]
            except KeyError as exc:
                raise ValueError(f"unknown wake episode: {args.episode_id}") from exc
            opening = next(
                (
                    event
                    for event in operational.records
                    if isinstance(event.get("details"), dict)
                    and isinstance(event["details"].get("episode"), dict)
                    and event["details"]["episode"].get("episode_id") == args.episode_id
                ),
                None,
            )
            if opening is None:
                raise ValueError("wake episode lacks its owning campaign event")
            context = OperationalContext(
                run_id=str(opening["run_id"]),
                machine_id=str(opening["machine_id"]),
                queued_work_counts=dict(opening["queued_work_counts"]),
                current_environment=(
                    str(opening["current_environment"])
                    if opening["current_environment"] is not None
                    else None
                ),
            )
            manager = WakeupManager(
                runtime / "wakeups",
                operational,
                state.episode.callback_argv,
                backend=state.backend,
            )
            manager.resolve_episode(
                args.episode_id,
                resolution="operator-cancelled",
                context=context,
                created_at=utc_now(),
            )
            manager.reconcile(context=context, created_at=utc_now())
            output = render_wakeup_status(manager.projection)
    print(json.dumps(output, sort_keys=True))
    return EXIT_OK


def _checkpoint_command(args: argparse.Namespace) -> int:
    """Run and report one complete fail-closed canonical checkpoint transaction."""

    del args.verify_ledgers, args.verify_views
    owner = {
        "pid": os.getpid(),
        "run_id": "checkpoint",
        "target": "canonical-checkpoint",
        "created_at": utc_now(),
        "command": list(sys.argv),
    }
    try:
        with DriverLock(args.repo_root / ".crawl-local" / "locks" / "driver.lock", owner):
            snapshot = load_intake_snapshot(args.intake)
            defaults = _snapshot_driver_config(args.repo_root, snapshot)
            context = build_authority_context(
                active_intake_snapshot_id=snapshot.snapshot_id,
                active_intake_snapshot_sha256=snapshot.snapshot_sha256,
                intake_rows=(item.to_dict() for item in snapshot.items),
                author_model=defaults.author_model,
                author_version=defaults.author_version,
                checker_model=defaults.checker_model,
                checker_version=defaults.checker_version,
            )
            result = create_canonical_checkpoint(
                args.repo_root,
                args.intake,
                records_root=args.records_root,
                authority_context=context,
            )
    except SingleWriterError as exc:
        raise DriverLockError(str(exc)) from exc
    print(
        json.dumps(
            {
                "verified": True,
                "branch": result.branch,
                "staged_paths": [path.as_posix() for path in result.paths],
                "license_report": result.license_report.to_dict(),
            },
            sort_keys=True,
        )
    )
    return EXIT_OK


def _teardown_command(args: argparse.Namespace) -> int:
    """Lock and remove only active disposable environment/cache/scratch state."""

    runtime = args.repo_root / ".crawl-local"
    before = shutil.disk_usage(runtime if runtime.exists() else args.repo_root).free
    owner = {
        "pid": os.getpid(),
        "run_id": "teardown",
        "target": args.target,
        "created_at": utc_now(),
    }
    with DriverLock(runtime / "locks" / "driver.lock", owner):
        names = ["caches", "scratch"]
        if args.active_env:
            names.append("envs")
        for name in names:
            shutil.rmtree(runtime / name, ignore_errors=True)
    after = shutil.disk_usage(runtime if runtime.exists() else args.repo_root).free
    if args.verify_disk and after < before:
        raise ValueError("teardown did not recover disk capacity")
    print(json.dumps({"teardown": "complete", "bytes_recovered": max(0, after - before)}))
    return EXIT_OK


def _requeue_command(args: argparse.Namespace) -> int:
    """Lock and append one validated grant directly to canonical history."""

    runtime = args.repo_root / ".crawl-local"
    path = (
        args.repo_root
        / "menagerie"
        / "crawler"
        / "records"
        / "operational"
        / "requeue-grants.jsonl"
    )
    owner = {
        "pid": os.getpid(),
        "run_id": "requeue",
        "target": None,
        "created_at": utc_now(),
    }
    with DriverLock(runtime / "locks" / "driver.lock", owner):
        payload = build_canonical_requeue_grant(
            path,
            stable_id=args.stable_id,
            stage=args.stage,
            reason=args.reason,
            attempts=args.grant,
            granted_by=args.granted_by,
            target_intent=args.target_intent,
        )
        append_canonical_requeue_grant(path, payload)
    print(json.dumps(payload, sort_keys=True))
    return EXIT_OK


def _positive_csv(value: str) -> tuple[int, ...]:
    """Parse a comma-separated unique positive integer tuple; empty disables milestones."""

    if not value.strip():
        return ()
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("milestones must be comma-separated integers") from exc
    if any(item < 1 for item in parsed) or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("milestones must be positive and unique")
    return parsed


def _framework(name: str, zoo: str) -> str:
    """Infer only explicit native framework markers for CLI planning."""

    lowered = f"{name} {zoo}".lower()
    for marker in ("tensorflow", "keras", "jax", "flax", "paddle"):
        if marker in lowered:
            return marker
    return "pytorch"
