"""Fail-closed materialization of committed SHA-256 release locks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence

from menagerie.crawler.env_lifecycle import (
    installed_package_inventory_bytes,
    parse_exact_lock,
    parse_resolved_export,
)


class ReleaseGateError(RuntimeError):
    """Raised when a committed release lock cannot be materialized exactly."""


def _sha256_file(path: Path) -> str:
    """Return the canonical SHA-256 digest of one file.

    Parameters
    ----------
    path:
        File whose bytes must be hashed.

    Returns
    -------
    str
        ``sha256:``-prefixed lowercase digest.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _metadata_rows(prefix: Path) -> tuple[Mapping[str, object], ...]:
    """Load every installed conda metadata row.

    Parameters
    ----------
    prefix:
        Freshly materialized conda prefix.

    Returns
    -------
    tuple[collections.abc.Mapping[str, object], ...]
        Installed rows in metadata-filename order.
    """

    rows: list[Mapping[str, object]] = []
    for path in sorted((prefix / "conda-meta").glob("*.json")):
        value = json.loads(path.read_bytes())
        if not isinstance(value, dict):
            raise ReleaseGateError(f"installed package metadata is not an object: {path}")
        rows.append(value)
    if not rows:
        raise ReleaseGateError(f"installed package metadata is absent below {prefix}")
    return tuple(rows)


def _verify_materialized_artifacts(
    prefix: Path,
    expected: frozenset[tuple[str, str]],
) -> None:
    """Verify every downloaded artifact against the committed SHA-256 lock.

    Parameters
    ----------
    prefix:
        Fresh prefix created with a dedicated package cache.
    expected:
        Exact ``(URL, sha256:...)`` rows parsed from the committed lock.
    """

    observed: set[tuple[str, str]] = set()
    for row in _metadata_rows(prefix):
        url = row.get("url")
        declared = row.get("sha256")
        archive_value = row.get("package_tarball_full_path")
        if not all(isinstance(value, str) and value for value in (url, declared, archive_value)):
            raise ReleaseGateError("installed metadata lacks URL, SHA-256, or package archive")
        archive = Path(str(archive_value))
        if not archive.is_file():
            raise ReleaseGateError(f"materialized package archive is unavailable: {archive}")
        canonical_declared = f"sha256:{str(declared).removeprefix('sha256:')}"
        observed_digest = _sha256_file(archive)
        if observed_digest != canonical_declared:
            raise ReleaseGateError(f"materialized package SHA-256 mismatch: {archive}")
        observed.add((str(url), observed_digest))
    if frozenset(observed) != expected or len(observed) != len(expected):
        raise ReleaseGateError("materialized package artifacts differ from the committed lock")


def create_from_committed_lock(
    lock_path: Path,
    export_path: Path,
    prefix: Path,
    *,
    conda_command: str = "conda",
) -> None:
    """Create and verify a fresh prefix solely from a committed SHA-256 lock.

    Conda releases before native SHA-256 explicit-fragment support receive a temporary
    URL-only explicit projection. The committed SHA-256 lock remains authoritative:
    every downloaded archive and the complete installed inventory are checked against it
    before the prefix is exposed.

    Parameters
    ----------
    lock_path, export_path:
        Committed explicit lock and canonical resolved export.
    prefix:
        Nonexistent destination prefix.
    conda_command:
        Conda executable selected by the release host.
    """

    if not lock_path.is_file() or not export_path.is_file():
        raise ReleaseGateError("committed lock or resolved export is unavailable")
    if prefix.exists():
        raise ReleaseGateError(f"release prefix already exists: {prefix}")
    lock_bytes = lock_path.read_bytes()
    export_bytes = export_path.read_bytes()
    receipts = parse_exact_lock(lock_bytes)
    if parse_resolved_export(export_bytes) != export_bytes:
        raise ReleaseGateError("committed resolved export is not canonical")
    expected = frozenset((receipt.url, receipt.sha256) for receipt in receipts)
    if len(expected) != len(receipts):
        raise ReleaseGateError("committed lock contains duplicate artifact identities")

    prefix.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".round21-lock-create-", dir=prefix.parent) as temporary:
        temporary_root = Path(temporary)
        projected_lock = temporary_root / "urls-only-explicit.lock"
        projected_lock.write_text(
            "@EXPLICIT\n" + "".join(f"{receipt.url}\n" for receipt in receipts),
            encoding="utf-8",
        )
        package_cache = temporary_root / "package-cache"
        environment = dict(os.environ)
        environment["CONDA_PKGS_DIRS"] = str(package_cache)
        completed = subprocess.run(
            (
                conda_command,
                "create",
                "--yes",
                "--prefix",
                str(prefix),
                "--file",
                str(projected_lock),
            ),
            check=False,
            env=environment,
            text=True,
        )
        if completed.returncode != 0:
            raise ReleaseGateError(
                f"conda failed to materialize the committed lock: exit {completed.returncode}"
            )
        installed = installed_package_inventory_bytes(prefix)
        if installed != export_bytes:
            raise ReleaseGateError(
                "created-prefix package inventory differs from the committed resolved export"
            )
        _verify_materialized_artifacts(prefix, expected)


def _parser() -> argparse.ArgumentParser:
    """Build the release-lock command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the single fail-closed create operation.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--resolved-export", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--conda", default="conda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Materialize one committed release lock or emit an unmet-gate error.

    Parameters
    ----------
    argv:
        Optional command-line arguments excluding the executable name.

    Returns
    -------
    int
        Zero after exact creation; one after a release-gate failure.
    """

    arguments = _parser().parse_args(argv)
    try:
        create_from_committed_lock(
            arguments.lock,
            arguments.resolved_export,
            arguments.prefix,
            conda_command=str(arguments.conda),
        )
    except (OSError, ValueError, subprocess.SubprocessError, ReleaseGateError) as exc:
        print(f"unmet-release-gate: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
