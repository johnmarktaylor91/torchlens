"""Enforce the crawler public/private byte boundary and emit a license report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.identity import canonical_json_bytes
from menagerie.crawler.licenses import (
    LicenseDecision,
    LicensedArtifact,
    LicenseSweepReport,
    PublicMergeRejected,
    RedistributionClass,
    pre_public_merge_sweep,
)
from menagerie.crawler.mirrors import ArtifactManifest, MirrorStore
from menagerie.crawler.recordio import scan_jsonl


def build_parser() -> argparse.ArgumentParser:
    """Build the license-sweep argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for staged artifact manifests and separated mirror roots.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def _licensed_artifact(payload: Mapping[str, Any]) -> LicensedArtifact:
    """Parse one staged artifact row.

    Parameters
    ----------
    payload:
        JSONL row with ``staged_path``, ``manifest``, and ``decision`` fields.

    Returns
    -------
    LicensedArtifact
        Typed staged artifact for the existing license gate.
    """

    manifest_raw = payload["manifest"]
    decision_raw = payload["decision"]
    if not isinstance(manifest_raw, Mapping) or not isinstance(decision_raw, Mapping):
        raise ValueError("artifact manifest and decision must be objects")
    manifest = ArtifactManifest.from_dict(manifest_raw)
    decision = LicenseDecision(
        content_sha256=str(decision_raw["content_sha256"]),
        redistribution_class=RedistributionClass(str(decision_raw["redistribution_class"])),
        evidence_ids=tuple(str(item) for item in decision_raw.get("evidence_ids", [])),
        rationale=str(decision_raw["rationale"]),
    )
    return LicensedArtifact(Path(str(payload["staged_path"])), manifest, decision)


def _write_report(path: Path, report: LicenseSweepReport) -> None:
    """Atomically write one deterministic license report.

    Parameters
    ----------
    path:
        Report destination.
    report:
        Passing or rejected complete sweep report.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_json_bytes(report.to_dict()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the existing license gate and persist its full report.

    Parameters
    ----------
    argv:
        Optional command arguments, excluding the executable name.

    Returns
    -------
    int
        Zero only when every staged artifact is public-compatible.
    """

    args = build_parser().parse_args(argv)
    try:
        artifacts = [_licensed_artifact(row) for row in scan_jsonl(args.artifacts, validate=False)]
        mirrors = MirrorStore(args.public_root, args.private_root, args.local_root)
        try:
            report = pre_public_merge_sweep(artifacts, mirrors)
        except PublicMergeRejected as exc:
            report = exc.report
            _write_report(args.report, report)
            print(json.dumps(report.to_dict(), sort_keys=True))
            return 1
        _write_report(args.report, report)
        print(json.dumps(report.to_dict(), sort_keys=True))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"license sweep failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
