"""Author operator wrapper: the subprocess contract, served by the subagent pool.

The engine's author boundary is argv-only: run a command with **one absolute request
path**, and read the result from the request's exact ``required_output_path``. This module
is that command. It does not author anything itself -- it publishes the request as a job on
the campaign's author queue, blocks until the managing session's subagent pool answers,
and translates the answer into the operator protocol's exit codes.

That indirection is what lets the doctor's capability probe be a *real* probe. The doctor
mints a nonce and invokes ``MENAGERIE_AUTHOR_COMMAND`` exactly as the engine would, so the
nonce travels the production author path end to end -- wrapper, queue, live pool, real
subagent, real web tools -- and the receipt it returns attests the path that will actually
do the campaign's research, not a proxy for it.

Exit codes (``PLAN_RECONCILED`` section 3.0)::

    0   result published at the required path, receipt committed
    64  permanent contract rejection
    75  retryable infrastructure failure
    76  provider rate/quota pause
    78  service unavailable
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Optional, Sequence
import uuid

from menagerie.crawler.author_queue import (
    DEFAULT_EFFORT_GRANT,
    EXIT_OK,
    EXIT_PERMANENT,
    EXIT_QUOTA,
    EXIT_RETRYABLE,
    EXIT_UNAVAILABLE,
    AuthorQueueError,
    EffortGrant,
    QueueJob,
    build_job_descriptor as build_queue_job_descriptor,
    ensure_author_queue,
    job_paths,
    matches_attempt,
    read_json,
    write_json_atomic,
)
from menagerie.crawler.capability_probe import CAPABILITY_PROBE_FORMAT
from menagerie.crawler.constants import TIER_CAMPAIGN_ENV, TIER_CAMPAIGN_IDS
from menagerie.crawler.identity import utc_now
from menagerie.crawler.models import JsonObject

WRAPPER_VERSION = "menagerie-author-operator 1.0.0"

#: Outer deadline. A managing session that dies must look like a stalled queue -- retryable
#: infrastructure -- never like a failed model.
DEFAULT_STALL_SECONDS = 45 * 60.0
DEFAULT_POLL_SECONDS = 2.0

_SOURCE_REQUEST_VERSION = "menagerie.crawler.author-source-request.v1"
_AUTHOR_ENVELOPE_PREFIX = "menagerie.crawler.author-envelope"


class OperatorAuthorError(RuntimeError):
    """Raised when a request cannot be turned into a well-formed queue job."""


def classify_request(request: Mapping[str, Any]) -> str:
    """Return the queue job kind one request envelope asks for.

    Parameters
    ----------
    request:
        Parsed request envelope.

    Returns
    -------
    str
        ``capability-probe``, ``source-request``, or ``author``.

    Raises
    ------
    OperatorAuthorError
        When the envelope is not one the author path serves. Guessing would enqueue a job
        the pool has no brief for.
    """

    if str(request.get("format", "")) == CAPABILITY_PROBE_FORMAT:
        return "capability-probe"
    version = str(request.get("envelope_version", ""))
    if version == _SOURCE_REQUEST_VERSION:
        return "source-request"
    if version.startswith(_AUTHOR_ENVELOPE_PREFIX):
        return "author"
    raise OperatorAuthorError(
        f"author request is neither a capability probe nor an author envelope: "
        f"format={request.get('format')!r} envelope_version={version!r}"
    )


def build_job_descriptor(
    request_path: Path,
    request: Mapping[str, Any],
    *,
    queue_root: Path,
    effort_grant: EffortGrant,
    stall_timeout_seconds: float,
    attempt_nonce: Optional[str] = None,
    author_model: Optional[str] = None,
    tier_campaign_id: Optional[str] = None,
) -> JsonObject:
    """Build the pending descriptor for one request.

    The shape is the lane's own: the pool cannot tell whether a job was enqueued by
    ``QueueAuthorLane`` or by this wrapper, and must not need to.

    Parameters
    ----------
    request_path:
        Absolute request envelope path.
    request:
        Parsed request envelope.
    queue_root:
        Campaign-local author-queue root.
    effort_grant:
        Per-session budget published to the pool.
    stall_timeout_seconds:
        Outer deadline published to the pool.
    attempt_nonce:
        Per-attempt nonce; generated when absent.
    author_model:
        The tier campaign's frozen author model, when configured.
    tier_campaign_id:
        Frozen tier campaign this wrapper serves (``--campaign``). It is the run's tier
        and is published on its own descriptor key; it never overwrites the envelope's
        ``campaign_id``, which is the driver's per-item repair scope.

    Returns
    -------
    dict[str, Any]
        Job descriptor ready to publish.

    Raises
    ------
    OperatorAuthorError
        When the request names no output path, or when the configured tier campaign is
        outside the partitioner's closed four.
    """

    kind = classify_request(request)
    output = request.get("required_output_path")
    if not output:
        raise OperatorAuthorError("author request names no required_output_path")
    if tier_campaign_id is not None and tier_campaign_id not in TIER_CAMPAIGN_IDS:
        raise OperatorAuthorError(
            f"configured tier campaign {tier_campaign_id!r} is not one of "
            f"{sorted(TIER_CAMPAIGN_IDS)}"
        )
    nonce = str(request.get("nonce", ""))
    stable_id = str(request.get("stable_id") or f"capability-probe-{nonce[:16]}")
    work_id = str(request.get("work_id") or f"probe-{nonce[:16]}")
    return build_queue_job_descriptor(
        queue_root=queue_root,
        kind=kind,
        stable_id=stable_id,
        work_id=work_id,
        request_path=request_path,
        required_output_path=Path(str(output)),
        effort_grant=effort_grant,
        stall_timeout_seconds=stall_timeout_seconds,
        attempt_nonce=attempt_nonce or uuid.uuid4().hex,
        author_model=author_model,
        # The envelope's ``campaign_id`` is the driver's per-item repair scope and is the
        # only thing that may fill this slot. The wrapper's ``--campaign`` is the run's
        # frozen tier and travels on its own key; letting it overwrite the repair scope is
        # exactly the conflation that made the pool read a tier out of a repair lineage.
        repair_campaign_id=(
            None
            if request.get("campaign_id") is None
            else str(request.get("campaign_id"))
        ),
        tier_campaign_id=tier_campaign_id,
        enqueued_at=utc_now(),
    )


def _await_answer(
    job: QueueJob,
    *,
    poll_seconds: float,
    stall_seconds: float,
    monotonic: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> tuple[int, str]:
    """Block until the pool answers this exact attempt.

    Parameters
    ----------
    job:
        Published job.
    poll_seconds:
        Polling cadence.
    stall_seconds:
        Outer deadline.
    monotonic, sleep:
        Injectable time sources.

    Returns
    -------
    tuple[int, str]
        Operator exit status and a diagnostic line. A superseded attempt's file is never
        accepted: only a nonce-matched payload ends the wait.
    """

    deadline = monotonic() + stall_seconds
    while True:
        backoff = read_json(job.backoff_path)
        if matches_attempt(backoff, job) and backoff is not None:
            return (
                EXIT_QUOTA,
                "author provider pause "
                f"({backoff.get('provider')}/{backoff.get('reason')}) reset_at="
                f"{backoff.get('reset_at')}: {backoff.get('response_excerpt', '')[:500]}",
            )
        failure = read_json(job.failure_path)
        if matches_attempt(failure, job) and failure is not None:
            retryable = failure.get("retryable")
            if not isinstance(retryable, bool):
                return (
                    EXIT_PERMANENT,
                    f"author pool failure omitted a boolean retryable: {failure.get('reason')}",
                )
            code = EXIT_RETRYABLE if retryable else EXIT_PERMANENT
            return (
                code,
                f"author pool failure ({failure.get('reason')}): "
                f"{str(failure.get('detail', ''))[:1000]}",
            )
        receipt = read_json(job.receipt_path)
        if matches_attempt(receipt, job) and receipt is not None:
            if not job.required_output_path.is_file():
                return (
                    EXIT_PERMANENT,
                    "author pool committed a receipt without a result at "
                    f"{job.required_output_path}",
                )
            return (EXIT_OK, f"author job {job.job_id} completed")
        if monotonic() >= deadline:
            claimed = job.claim_path.is_file()
            return (
                EXIT_RETRYABLE,
                f"author job {job.job_id} produced no result within {stall_seconds:g}s "
                f"({'claimed but unfinished' if claimed else 'never claimed'})",
            )
        sleep(poll_seconds)


def serve_request(
    request_path: Path,
    *,
    queue_root: Path,
    effort_grant: Optional[EffortGrant] = None,
    stall_seconds: float = DEFAULT_STALL_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    author_model: Optional[str] = None,
    tier_campaign_id: Optional[str] = None,
    monotonic: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> tuple[int, str]:
    """Publish one request as a queue job and wait for the pool's answer.

    Parameters
    ----------
    request_path:
        Absolute request envelope path.
    queue_root:
        Campaign-local author-queue root.
    effort_grant:
        Per-session budget; defaults to the LP-13.2 ceiling.
    stall_seconds, poll_seconds:
        Outer deadline and polling cadence.
    author_model, tier_campaign_id:
        Frozen tier-campaign bindings recorded on the job. The job's per-item repair scope
        comes from the request envelope and is never taken from here.
    monotonic, sleep:
        Injectable time sources.

    Returns
    -------
    tuple[int, str]
        Operator exit status and diagnostic line.
    """

    request = read_json(request_path)
    if request is None:
        return (EXIT_PERMANENT, f"author request at {request_path} is not a JSON object")
    ensure_author_queue(queue_root)
    descriptor = build_job_descriptor(
        request_path,
        request,
        queue_root=queue_root,
        effort_grant=effort_grant or DEFAULT_EFFORT_GRANT,
        stall_timeout_seconds=stall_seconds,
        author_model=author_model,
        tier_campaign_id=tier_campaign_id,
    )
    job = QueueJob.from_mapping(descriptor)
    if job.kind == "capability-probe":
        # The doctor's freshness window, not the author stall guard, bounds a probe: a
        # receipt that arrives after the window is refused anyway, so waiting longer only
        # burns the doctor's own subprocess timeout.
        stall_seconds = min(stall_seconds, float(request.get("deadline_seconds", 120) or 120))
        poll_seconds = min(poll_seconds, 0.5)
        descriptor["stall_timeout_seconds"] = stall_seconds
        job = QueueJob.from_mapping(descriptor)
    # A previous attempt's answer must never satisfy this one. Clearing the control files
    # for this job id before publishing keeps the nonce check from having to be the only
    # line of defence.
    for name in ("claim", "receipt", "backoff", "failure"):
        job_paths(queue_root, job.job_id)[name].unlink(missing_ok=True)
    write_json_atomic(job_paths(queue_root, job.job_id)["pending"], descriptor)
    try:
        return _await_answer(
            job,
            poll_seconds=poll_seconds,
            stall_seconds=stall_seconds,
            monotonic=monotonic,
            sleep=sleep,
        )
    finally:
        job_paths(queue_root, job.job_id)["pending"].unlink(missing_ok=True)


def _resolve_queue_root(explicit: Optional[str]) -> Path:
    """Resolve the campaign author-queue root.

    Parameters
    ----------
    explicit:
        Command-line override.

    Returns
    -------
    Path
        Absolute queue root.

    Raises
    ------
    OperatorAuthorError
        When no queue root is configured. Silently inventing one would enqueue jobs no
        pool is watching.
    """

    raw = explicit or os.environ.get("MENAGERIE_AUTHOR_QUEUE")
    if not raw:
        raise OperatorAuthorError(
            "no author queue configured; set MENAGERIE_AUTHOR_QUEUE or pass --queue"
        )
    return Path(raw).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    """Build the operator wrapper CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(
        prog="python -m menagerie.crawler.operator_author",
        description="Serve one author request through the in-session subagent pool.",
    )
    parser.add_argument("--version", action="store_true", help="print the wrapper version")
    parser.add_argument("--queue", default=None, help="campaign author-queue root")
    parser.add_argument(
        "--campaign",
        dest="tier_campaign_id",
        default=None,
        help=(
            "frozen tier campaign this wrapper serves, one of "
            f"{sorted(TIER_CAMPAIGN_IDS)}; defaults to {TIER_CAMPAIGN_ENV}. Recorded as "
            "the job's tier_campaign_id; the job's per-item repair scope always comes "
            "from the request envelope"
        ),
    )
    parser.add_argument("--author-model", default=None, help="tier campaign author model")
    parser.add_argument(
        "--stall-seconds", type=float, default=DEFAULT_STALL_SECONDS, help="outer deadline"
    )
    parser.add_argument(
        "--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS, help="polling cadence"
    )
    parser.add_argument("request", nargs="?", default=None, help="absolute request path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the author operator wrapper.

    Parameters
    ----------
    argv:
        Command-line arguments.

    Returns
    -------
    int
        Operator protocol exit status.
    """

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.version:
        print(WRAPPER_VERSION)
        return EXIT_OK
    if args.request is None:
        print("author operator requires exactly one request path", file=sys.stderr)
        return EXIT_PERMANENT
    try:
        queue_root = _resolve_queue_root(args.queue)
        code, detail = serve_request(
            Path(args.request).expanduser().resolve(),
            queue_root=queue_root,
            stall_seconds=args.stall_seconds,
            poll_seconds=args.poll_seconds,
            author_model=args.author_model or os.environ.get("MENAGERIE_AUTHOR_MODEL"),
            tier_campaign_id=args.tier_campaign_id or os.environ.get(TIER_CAMPAIGN_ENV),
        )
    except (OperatorAuthorError, AuthorQueueError) as exc:
        print(f"author operator error: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except OSError as exc:
        print(f"author operator io error: {exc}", file=sys.stderr)
        return EXIT_RETRYABLE
    stream = sys.stdout if code == EXIT_OK else sys.stderr
    print(detail, file=stream)
    return code


def wrapper_command(queue_root: Path, *, tier_campaign_id: Optional[str] = None) -> str:
    """Return the shell-quoted command the operator should export.

    Parameters
    ----------
    queue_root:
        Campaign-local author-queue root.
    tier_campaign_id:
        Frozen tier campaign to bind.

    Returns
    -------
    str
        Value for ``MENAGERIE_AUTHOR_COMMAND``.
    """

    argv = [sys.executable, "-m", "menagerie.crawler.operator_author", "--queue", str(queue_root)]
    if tier_campaign_id:
        argv.extend(["--campaign", tier_campaign_id])
    return shlex.join(argv)


if __name__ == "__main__":  # pragma: no cover -- operator entry point
    raise SystemExit(main())
