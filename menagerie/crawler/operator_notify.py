"""Notifier shim that turns "we ran a script" into "the message was delivered".

The crawler's unattended operation depends on notifications: quota pauses, review
checkpoints, milestone crossings, and stall alarms all reach the operator through one
delivery script. Before this shim, nothing verified that script existed, ran, or
succeeded -- a campaign could pause for a day with the operator never told.

The doctor now demands proof. It invokes the notifier with a fresh nonce in
``MENAGERIE_NOTIFICATION_IDEMPOTENCY_KEY`` and a path in
``MENAGERIE_NOTIFICATION_RECEIPT_PATH``, and requires a receipt at that path echoing the
nonce. This module writes that receipt -- and writes it **only after the transport exits
zero**. There is no arm that reports success without a successful delivery: a missing
transport, a nonzero exit, and a timeout all leave no receipt, so the strict doctor fails
exactly when notifications would silently vanish.

The receipt is evidence, not a flag: it records the resolved transport, its exit status,
the delivery instant, and the digest of the exact bytes sent, so an operator can match a
receipt to a message that actually arrived on the phone.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional, Sequence

from menagerie.crawler.identity import atomic_replace_bytes, canonical_json_bytes, utc_now

NOTIFY_RECEIPT_FORMAT = "menagerie.crawler.notification-receipt.v1"
NOTIFIER_VERSION = "menagerie-notify 1.0.0"
DEFAULT_TIMEOUT_SECONDS = 120.0

#: Search order for the underlying delivery transport, unchanged from the historical
#: notifier resolution so this shim only adds a receipt, never moves the transport.
_TRANSPORT_NAME = "send-to-jmt.sh"


def resolve_transport(explicit: Optional[str] = None) -> Optional[tuple[str, ...]]:
    """Resolve the underlying delivery transport.

    Parameters
    ----------
    explicit:
        Operator override; also read from ``MENAGERIE_NOTIFY_TRANSPORT``.

    Returns
    -------
    tuple[str, ...] | None
        Transport argv, or ``None`` when no transport is installed. ``None`` is a genuine
        failure state, not a fallback to pretending: the caller writes no receipt.
    """

    raw = explicit or os.environ.get("MENAGERIE_NOTIFY_TRANSPORT")
    if raw:
        import shlex

        parsed = tuple(shlex.split(raw))
        return parsed or None
    found = shutil.which(_TRANSPORT_NAME)
    if found is not None:
        return (found,)
    for candidate in (
        Path.home() / "scripts" / _TRANSPORT_NAME,
        Path.home() / "bin" / _TRANSPORT_NAME,
        Path.home() / ".claude" / "scripts" / _TRANSPORT_NAME,
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return (str(candidate),)
    return None


def deliver(
    message: str,
    *,
    transport: Optional[Sequence[str]] = None,
    receipt_path: Optional[Path] = None,
    nonce: Optional[str] = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> int:
    """Deliver one message and, on success, write its nonce receipt.

    Parameters
    ----------
    message:
        Message body to deliver.
    transport:
        Resolved transport argv; resolved from the environment when absent.
    receipt_path:
        Destination for the delivery receipt.
    nonce:
        Idempotency key that the receipt must echo.
    timeout_seconds:
        Transport timeout.

    Returns
    -------
    int
        Transport exit status, or a nonzero status when no transport is installed or the
        transport timed out. A receipt is written on -- and only on -- exit zero.
    """

    resolved = tuple(transport) if transport else resolve_transport()
    if resolved is None:
        print(f"no {_TRANSPORT_NAME} transport is installed", file=sys.stderr)
        return 69
    try:
        completed = subprocess.run(
            [*resolved, message],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        print(f"notifier transport timed out after {timeout_seconds:g}s", file=sys.stderr)
        return 75
    except OSError as exc:
        print(f"notifier transport failed to start: {exc}", file=sys.stderr)
        return 69
    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        return completed.returncode
    if receipt_path is not None and nonce:
        _write_receipt(receipt_path, nonce=nonce, transport=resolved, message=message)
    return 0


def _write_receipt(
    path: Path, *, nonce: str, transport: Sequence[str], message: str
) -> None:
    """Write one delivery receipt atomically.

    Parameters
    ----------
    path:
        Receipt destination.
    nonce:
        Idempotency key the doctor matches on.
    transport:
        Transport argv that delivered the message.
    message:
        Delivered body, retained by digest and bounded prefix only.
    """

    import hashlib

    payload = {
        "format": NOTIFY_RECEIPT_FORMAT,
        "nonce": nonce,
        "delivered_at": utc_now(),
        "transport": list(transport),
        "transport_exit_code": 0,
        "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
        "message_head": message[:200],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace_bytes(path, canonical_json_bytes(payload) + b"\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the notifier shim CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(
        prog="python -m menagerie.crawler.operator_notify",
        description="Deliver one crawler notification and attest the delivery.",
    )
    parser.add_argument("--version", action="store_true", help="print the notifier version")
    parser.add_argument("--transport", default=None, help="explicit delivery transport argv")
    parser.add_argument(
        "--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="transport timeout"
    )
    parser.add_argument("message", nargs="?", default=None, help="message body")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the notifier shim.

    Parameters
    ----------
    argv:
        Command-line arguments.

    Returns
    -------
    int
        Process exit status mirroring the transport.
    """

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.version:
        print(NOTIFIER_VERSION)
        return 0
    if args.message is None:
        print("notifier requires exactly one message argument", file=sys.stderr)
        return 64
    raw_receipt = os.environ.get("MENAGERIE_NOTIFICATION_RECEIPT_PATH")
    return deliver(
        args.message,
        transport=None if args.transport is None else _split(args.transport),
        receipt_path=None if not raw_receipt else Path(raw_receipt),
        nonce=os.environ.get("MENAGERIE_NOTIFICATION_IDEMPOTENCY_KEY"),
        timeout_seconds=args.timeout_seconds,
    )


def _split(value: str) -> tuple[str, ...]:
    """Split one shell-quoted transport specification.

    Parameters
    ----------
    value:
        Transport specification.

    Returns
    -------
    tuple[str, ...]
        Transport argv.
    """

    import shlex

    return tuple(shlex.split(value))


def read_receipt(path: Path) -> Optional[dict]:
    """Read one delivery receipt.

    Parameters
    ----------
    path:
        Receipt path.

    Returns
    -------
    dict | None
        Parsed receipt, or ``None`` when absent or unreadable.
    """

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


if __name__ == "__main__":  # pragma: no cover -- operator entry point
    raise SystemExit(main())
