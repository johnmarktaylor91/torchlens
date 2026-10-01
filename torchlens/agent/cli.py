"""The TorchLens CLI: one registry, tiered process shapes (agent memo 3.11).

Tier 0 (torch-free, target < 0.2 s): ``info`` / ``overview --manifest-only``,
``schema``, ``guide``, ``version``, ``ls`` -- manifest and package data only;
a subprocess purity test asserts torch stays out of ``sys.modules``. Tier 1
(one question, one process): every registry verb, honest in ``--help`` about
the import cost. Machine mode (``--json``) emits ONLY canonical envelope JSON
on stdout; diagnostics go to stderr; no ANSI in pipes.

Exit codes (closed set): 0 ok; 1 CI gate tripped (``--fail-on``); 2
usage/schema error; 3 artifact unreadable / wrong kind; 4 typed analysis
refusal.

``--fail-on`` gates READ the emitted record, never recompute: ``unverified``
(any ``capture.capture_verified is False``), ``incomplete`` (envelope status
not ``ok``, ``structure_only``, or ``capture_status != "complete"`` -- halted,
aborted, failed, unattested, unknown), ``nonfinite`` (audit non-finite labels
or a ``nonfinite_ops`` anomaly), ``mismatch`` (compare: changed sites or a
fingerprint miss), ``truncation`` (any truncation disclosure). overview,
dump, explain, and diff all carry the blocks the gates read.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

#: Exit-code vocabulary (closed).
EXIT_OK = 0
EXIT_GATE = 1
EXIT_USAGE = 2
EXIT_ARTIFACT = 3
EXIT_REFUSAL = 4

#: ``--fail-on`` checks: each READS an existing result record, never recomputes.
FAIL_ON_CHECKS = ("unverified", "incomplete", "nonfinite", "mismatch", "truncation")

#: Verbs that must stay torch-free (tier 0).
TIER0_VERBS = ("info", "schema", "guide", "version", "ls")

#: Error codes that mean "artifact unreadable / wrong kind" (exit 3).
_ARTIFACT_CODES = (
    "agent_artifact_unreadable",
    "agent_artifact_load_refused",
    "agent_artifact_kind_unsupported",
    # The loader's own integrity refusal (a corrupted metadata.pkl) is an
    # unreadable artifact too, not an analysis refusal.
    "bundle_metadata_integrity_refused",
)


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    """Write one result to stdout (canonical JSON or plain key lines)."""

    from ._envelope import canonical_dumps, json_safe

    if as_json:
        sys.stdout.write(canonical_dumps(json_safe(payload)) + "\n")
        return
    data = payload.get("data", payload)
    for key, value in sorted(data.items()) if isinstance(data, dict) else [("result", data)]:
        rendered = json.dumps(json_safe(value)) if isinstance(value, dict | list) else str(value)
        if len(rendered) > 200:
            rendered = rendered[:197] + "..."
        sys.stdout.write(f"{key}: {rendered}\n")


def _tier0_info(path_arg: str, as_json: bool) -> int:
    """Torch-free manifest preflight (``info`` verb)."""

    from ._artifacts import artifact_block, read_manifest, resolve_artifact_path, resolve_digest
    from ._envelope import build_envelope
    from ._overview import OVERVIEW_MANIFEST_SCHEMA, manifest_overview

    path = resolve_artifact_path(path_arg)
    digest, _ = resolve_digest(path)
    manifest, _ = read_manifest(path)
    envelope = build_envelope(
        schema=OVERVIEW_MANIFEST_SCHEMA,
        data=manifest_overview(manifest),
        artifact=artifact_block(path, digest, manifest),
        request={"path": path_arg, "mode": "manifest"},
    )
    _emit(envelope, as_json)
    return EXIT_OK


def _tier0_ls(dir_arg: str, as_json: bool) -> int:
    """List .tlspec artifacts under one directory (tier 0; memo D5 majority)."""

    from pathlib import Path

    root = Path(dir_arg).expanduser()
    if not root.is_dir():
        sys.stderr.write(f"not a directory: {dir_arg}\n")
        return EXIT_ARTIFACT
    rows = sorted(str(item.name) for item in root.iterdir() if item.name.endswith(".tlspec"))
    _emit({"data": {"dir": dir_arg, "artifacts": rows}}, as_json)
    return EXIT_OK


def _tier0(verb: str, args: argparse.Namespace) -> int:
    """Dispatch one tier-0 verb without importing torch or loading bodies."""

    as_json = bool(args.json)
    if verb == "version":
        from ._envelope import _torchlens_version

        _emit({"data": {"torchlens_version": _torchlens_version()}}, as_json)
        return EXIT_OK
    if verb == "guide":
        from ._guide_text import guide

        sys.stdout.write(guide())
        return EXIT_OK
    if verb == "schema":
        from ._envelope import build_envelope
        from ._schemas import SCHEMA_TOOL_SCHEMA, load_schema, schema_index

        name = getattr(args, "name", None)
        envelope = build_envelope(
            schema=SCHEMA_TOOL_SCHEMA,
            data={"index": schema_index(), "document": load_schema(name) if name else None},
            request={"name": name} if name else {},
        )
        _emit(envelope, as_json)
        return EXIT_OK
    if verb == "ls":
        return _tier0_ls(args.dir, as_json)
    return _tier0_info(args.path, as_json)


def _capture_blocks(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    """Every capture honesty block a record carries (one, or one per side).

    overview/explain/dump carry ``data.capture`` directly; compare carries
    ``data.capture = {"reference": ..., "subject": ...}``. A gate reads them
    all: a diff over one unverified side is an unverified diff.
    """

    capture = (envelope.get("data") or {}).get("capture")
    if not isinstance(capture, dict):
        return []
    if "capture_status" in capture:
        return [capture]
    return [side for side in capture.values() if isinstance(side, dict)]


def _audit_blocks(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    """Every audit block a record carries (one, or one per compare side)."""

    audit = (envelope.get("data") or {}).get("audit")
    if not isinstance(audit, dict):
        return []
    if "nonfinite" in audit or "health" in audit:
        return [audit]
    return [side for side in audit.values() if isinstance(side, dict)]


def _check_unverified(envelope: dict[str, Any]) -> bool:
    """Whether any capture block records an explicit verification ceiling.

    ``capture_verified`` is TRI-STATE: ``None`` means no ceiling recorded (a
    healthy capture) and never trips.
    """

    return any(block.get("capture_verified") is False for block in _capture_blocks(envelope))


def _check_incomplete(envelope: dict[str, Any]) -> bool:
    """Whether the record is anything short of a settled COMPLETE capture.

    Trips on a degraded envelope status, a structure-only capture, or a
    ``capture_status`` other than ``complete`` -- HALTED, ABORTED_NONFINITE,
    FAILED, UNATTESTED (legacy, never blessed complete), and UNKNOWN all trip
    (AUD-CODE 3.11a: the gate used to ignore HALTED). A record without a
    capture block cannot prove completeness and trips too.
    """

    if envelope.get("status") != "ok":
        return True
    blocks = _capture_blocks(envelope)
    if not blocks:
        return True
    return any(
        bool(block.get("structure_only")) or block.get("capture_status") != "complete"
        for block in blocks
    )


def _check_nonfinite(envelope: dict[str, Any]) -> bool:
    """Whether the record's audit block(s) or anomalies carry non-finite evidence."""

    data = envelope.get("data") or {}
    if any(
        (audit.get("nonfinite") or {}).get("n_labels", 0) > 0 for audit in _audit_blocks(envelope)
    ):
        return True
    return any(anomaly.get("kind") == "nonfinite_ops" for anomaly in data.get("anomalies") or [])


def _check_mismatch(envelope: dict[str, Any]) -> bool:
    """Whether a compare record carries changed sites or a fingerprint miss."""

    data = envelope.get("data") or {}
    coverage = data.get("coverage") or {}
    return coverage.get("sites_changed", 0) > 0 or data.get("fingerprint_match") is False


#: --fail-on check -> record predicate (each READS, never recomputes).
_FAIL_ON_PREDICATES: dict[str, Any] = {
    "unverified": _check_unverified,
    "incomplete": _check_incomplete,
    "nonfinite": _check_nonfinite,
    "mismatch": _check_mismatch,
    "truncation": lambda envelope: envelope.get("truncation") is not None,
}


def _apply_fail_on(envelope: dict[str, Any], checks: list[str]) -> int:
    """Evaluate ``--fail-on`` gates against one result record (read, never recompute)."""

    tripped = [check for check in checks if _FAIL_ON_PREDICATES[check](envelope)]
    if tripped:
        sys.stderr.write(f"--fail-on tripped: {', '.join(tripped)}\n")
        return EXIT_GATE
    return EXIT_OK


def _tier1(verb: str, args: argparse.Namespace) -> int:
    """Dispatch one registry verb in-process (tier 1)."""

    from ._envelope import canonical_dumps, json_safe
    from ._registry import call_tool, tool_specs

    tool_by_verb = {spec.cli_verb: spec for spec in tool_specs() if spec.cli_verb}
    spec = tool_by_verb[verb]
    request: dict[str, Any] = {}
    for key in spec.input_schema.get("properties", {}):
        value = getattr(args, key.replace("-", "_"), None)
        if value is not None:
            request[key] = value
    try:
        envelope = call_tool(spec.name, request)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: canonical error JSON + closed exit code
        from ._envelope import build_error_envelope

        error = build_error_envelope(exc)
        sys.stderr.write(canonical_dumps(json_safe(error)) + "\n")
        code = (error.get("error") or {}).get("code")
        return EXIT_ARTIFACT if code in _ARTIFACT_CODES else EXIT_REFUSAL
    _emit(envelope, bool(args.json))
    checks = [check for check in (getattr(args, "fail_on", None) or "").split(",") if check]
    if checks:
        unknown = [check for check in checks if check not in FAIL_ON_CHECKS]
        if unknown:
            sys.stderr.write(f"unknown --fail-on checks: {', '.join(unknown)}\n")
            return EXIT_USAGE
        return _apply_fail_on(envelope, checks)
    return EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    """Build the tiered argument parser."""

    parser = argparse.ArgumentParser(
        prog="python -m torchlens",
        description=(
            "Read-only TorchLens inspector over saved .tlspec artifacts. "
            "Tier-0 verbs (info/schema/guide/version/ls) never import torch; "
            "other verbs pay the full torch import (~4 s)."
        ),
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    def _add(verb: str, help_text: str, *, path: bool = False) -> argparse.ArgumentParser:
        """Register one verb subparser with the shared --json flag."""

        verb_parser = sub.add_parser(verb, help=help_text)
        verb_parser.add_argument(
            "--json", action="store_true", help="Emit the canonical envelope JSON."
        )
        if path:
            verb_parser.add_argument("path", help="Path to a .tlspec artifact.")
        return verb_parser

    _add("info", "Torch-free manifest preflight (never unpickles).", path=True)
    schema_parser = _add("schema", "Schema index or one document (torch-free).")
    schema_parser.add_argument("name", nargs="?", help="Schema id.")
    _add("guide", "Print the agent guide (torch-free).")
    _add("version", "Print the torchlens version (torch-free).")
    ls_parser = _add("ls", "List .tlspec artifacts in a directory (torch-free).")
    ls_parser.add_argument("dir", help="Directory to list.")

    _add("doctor", "Environment health check (imports torch).")
    api_parser = _add("api-map", "Public-surface index (imports torch).")
    api_parser.add_argument("--name", help="Detail mode for one root name.")
    overview_parser = _add("overview", "Structural overview of an artifact.", path=True)
    overview_parser.add_argument("--mode", choices=["manifest", "folded"], default=None)
    dump_parser = _add("dump", "Paged structural dump views.", path=True)
    dump_parser.add_argument("--view", choices=["overview", "graph", "full"], default=None)
    dump_parser.add_argument("--max-rows", type=int, dest="max_rows")
    dump_parser.add_argument("--class-id", dest="class_id")
    dump_parser.add_argument("--max-tokens", type=int, dest="max_tokens")
    explain_parser = _add("explain", "Plain-language report.", path=True)
    explain_parser.add_argument("--max-tokens", type=int, dest="max_tokens")
    explain_parser.add_argument("--audience", choices=["researcher", "practitioner", "auto"])
    query_parser = _add("query", "Structured site discovery.", path=True)
    query_parser.add_argument(
        "--query", type=json.loads, help="torchlens.agent_query.v1 AST as JSON."
    )
    query_parser.add_argument("--max-rows", type=int, dest="max_rows")
    stats_parser = _add("stats", "Bounded payload statistics.", path=True)
    stats_parser.add_argument("--labels", nargs="*", default=None)
    stats_parser.add_argument("--target", choices=["out", "grad"])
    stats_parser.add_argument("--max-rows", type=int, dest="max_rows")
    diff_parser = sub.add_parser("diff", help="Compare two artifacts (subject - reference).")
    diff_parser.add_argument("reference")
    diff_parser.add_argument("subject")
    diff_parser.add_argument("--json", action="store_true")
    diff_parser.add_argument("--max-rows", type=int, dest="max_rows")
    diff_parser.add_argument(
        "--fail-on",
        dest="fail_on",
        help=f"Comma-joined CI gates from: {', '.join(FAIL_ON_CHECKS)}.",
    )
    for gated in (overview_parser, dump_parser, explain_parser):
        gated.add_argument(
            "--fail-on",
            dest="fail_on",
            help=f"Comma-joined CI gates from: {', '.join(FAIL_ON_CHECKS)}.",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv:
        Argument vector (``None`` = ``sys.argv[1:]``).

    Returns
    -------
    int
        Closed-set exit code.
    """

    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code not in (0, None) else EXIT_OK
    if args.verb in TIER0_VERBS:
        try:
            return _tier0(args.verb, args)
        except Exception as exc:  # noqa: BLE001 - CLI boundary: message to stderr + closed exit code
            code = (getattr(exc, "fields", {}) or {}).get("code")
            sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
            return EXIT_ARTIFACT if code in _ARTIFACT_CODES else EXIT_REFUSAL
    return _tier1(args.verb, args)


if __name__ == "__main__":
    sys.exit(main())
