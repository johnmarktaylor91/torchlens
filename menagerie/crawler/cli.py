"""Typed command-line interface for the resumable menagerie crawler."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Optional, Protocol, Sequence

from menagerie.crawler.checkpoint import (
    CheckpointError,
    append_canonical_requeue_grant,
    build_canonical_requeue_grant,
    create_canonical_checkpoint,
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
from menagerie.crawler.intake import create_intake_snapshot, load_intake_snapshot
from menagerie.crawler.recordio import SingleWriterError
from menagerie.crawler.reducer import materialize_current
from menagerie.crawler.routing import ModelRequirements, phase_routes, route_model
from menagerie.crawler.status import funnel_counts, partition_report

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
        if args.command == "plan":
            return _plan_command(args)
        if args.command == "status":
            return _status_command(args)
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
        help="run the test-only tiny-model corpus with fake LLM and environment lanes",
    )
    parser.add_argument(
        "--dry-run-root",
        type=Path,
        help="isolated dry-run campaign root (defaults under .crawl-local)",
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
    master = args.master or (data_root / "master_catalog.jsonl" if args.all_existing else None)
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


def _plan_command(args: argparse.Namespace) -> int:
    """Print deterministic phase-first environment assignments."""

    snapshot = load_intake_snapshot(args.intake)
    registry = load_environment_registry(target=args.target)
    routes = phase_routes(
        route_model(ModelRequirements(item.stable_id, _framework(item.name, item.zoo)))
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
    result: DriverResult = driver.run(after_review=after_review)
    output: dict[str, object] = dict(result.__dict__)
    if bool(getattr(args, "dry_run", False)):
        current = materialize_current(driver.paths.ledgers)
        output.update({"dry_run": True, "funnel": funnel_counts(current)})
    print(json.dumps(output, sort_keys=True))
    return EXIT_PAUSED if result.paused_reason is not None else EXIT_OK


def _default_driver_factory(args: argparse.Namespace) -> CrawlerDriver:
    """Build production command lanes from explicit flags or supervisor environment."""

    if bool(getattr(args, "dry_run", False)):
        from menagerie.crawler.tests.dry_run_support import build_dry_run_driver

        dry_run_root = args.dry_run_root or args.repo_root / ".crawl-local" / "dry-run"
        return build_dry_run_driver(
            args.repo_root.resolve(),
            dry_run_root.resolve(),
            review_checkpoint_at=args.review_checkpoint_at,
            progress_milestones=tuple(args.progress_milestones),
            run_id=args.run_id,
        )
    if args.intake is None:
        raise ValueError("run/resume requires --intake unless --dry-run is selected")
    author_command = _required_command(args.author_command, "author")
    checker_command = _required_command(args.checker_command, "checker")
    environment_command = _required_command(args.environment_command, "environment")
    paths = default_driver_paths(args.repo_root.resolve(), args.intake.resolve())
    config = DriverConfig(
        target=args.target,
        phase=args.phase,
        run_id=args.run_id,
        review_checkpoint_at=args.review_checkpoint_at,
        progress_milestones=tuple(args.progress_milestones),
        notify_command=args.notify_command,
        only_status=getattr(args, "only_status", None),
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


def _required_command(value: Optional[str], lane: str) -> tuple[str, ...]:
    """Parse one required non-shell lane command with a clear configuration error."""

    if value is None or not value.strip():
        environment_name = f"MENAGERIE_{lane.upper()}_COMMAND"
        raise ValueError(f"{lane} lane requires --{lane}-command or {environment_name}")
    command = tuple(shlex.split(value))
    if not command:
        raise ValueError(f"{lane} lane command cannot be empty")
    return command


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
    current = materialize_current(paths)
    output: dict[str, object] = {"funnel": funnel_counts(current), "terminal": len(current)}
    if args.verify_partition:
        report = partition_report((item.stable_id for item in snapshot.items), current)
        output["partition_valid"] = report.valid
        output["missing"] = sorted(report.missing_ids)
    if args.full:
        output["current"] = current
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
            result = create_canonical_checkpoint(
                args.repo_root,
                args.intake,
                records_root=args.records_root,
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
