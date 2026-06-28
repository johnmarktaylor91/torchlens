"""Build distributable menagerie artifact bundles."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from menagerie.csv_export import DATASET_SCHEMA_VERSION


DATA_DIR = Path(__file__).resolve().parent / "data"
MENAGERIE_DIR = Path(__file__).resolve().parent
DEFAULT_DIST_DIR = DATA_DIR / "dist"
DEFAULT_MAX_COMBINED_GB = 5.0
FIXED_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
COMPRESSION_LEVEL = 9
CATALOG_FILES = (
    DATA_DIR / "master_catalog.jsonl",
    DATA_DIR / "catalog_canonical.tsv",
    DATA_DIR / "stable_ids.jsonl",
    DATA_DIR / "routing_manifest.tsv",
    MENAGERIE_DIR / "README.md",
    MENAGERIE_DIR / "METHODOLOGY.md",
)
CSV_FILES = (
    "menagerie.csv",
    "trace_metrics.parquet",
    "trace_histograms.jsonl",
    "DATA_DICTIONARY.md",
)
VISUAL_SUFFIXES = frozenset({".svg", ".png", ".pdf"})


@dataclass(frozen=True)
class BundleStats:
    """Summary for one written zip bundle.

    Parameters
    ----------
    name:
        Zip file name.
    path:
        Zip path.
    file_count:
        Number of file entries.
    uncompressed_bytes:
        Sum of source file sizes.
    compressed_bytes:
        Zip file size.
    sha256:
        SHA-256 digest of the zip file.
    skipped:
        Whether the bundle was intentionally skipped.
    reason:
        Skip reason.
    """

    name: str
    path: Path | None
    file_count: int
    uncompressed_bytes: int
    compressed_bytes: int
    sha256: str
    skipped: bool = False
    reason: str = ""


@dataclass(frozen=True)
class ZipEntry:
    """One source file mapped into a deterministic zip entry.

    Parameters
    ----------
    source:
        Source file path.
    arcname:
        POSIX zip entry path.
    """

    source: Path
    arcname: str


def _log(message: str) -> None:
    """Print one bundling log message.

    Parameters
    ----------
    message:
        Human-readable message.
    """

    print(f"[menagerie bundle] {message}", flush=True)


def _git_commit() -> str:
    """Return the current Git commit hash when available.

    Returns
    -------
    str
        Commit hash or an empty string.
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=MENAGERIE_DIR.parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    """Return a SHA-256 digest for one file.

    Parameters
    ----------
    path:
        File path.

    Returns
    -------
    str
        Hexadecimal digest.
    """

    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry_sort_key(entry: ZipEntry) -> str:
    """Return the deterministic sort key for a zip entry.

    Parameters
    ----------
    entry:
        Zip entry.

    Returns
    -------
    str
        Sort key.
    """

    return entry.arcname


def _file_entries(
    root: Path, *, prefix: str, suffixes: frozenset[str] | None = None
) -> list[ZipEntry]:
    """Collect file entries below a root directory.

    Parameters
    ----------
    root:
        Source root.
    prefix:
        Zip entry prefix.
    suffixes:
        Optional lowercase file suffix filter.

    Returns
    -------
    list[ZipEntry]
        Sorted file entries.
    """

    if not root.exists():
        return []
    entries: list[ZipEntry] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        arcname = (Path(prefix) / path.relative_to(root)).as_posix()
        entries.append(ZipEntry(path, arcname))
    return sorted(entries, key=_entry_sort_key)


def _selected_file_entries(files: Iterable[Path], *, prefix: str = "") -> list[ZipEntry]:
    """Collect entries for an explicit file list.

    Parameters
    ----------
    files:
        Source file paths.
    prefix:
        Optional zip entry prefix.

    Returns
    -------
    list[ZipEntry]
        Sorted existing file entries.
    """

    entries: list[ZipEntry] = []
    for path in files:
        if not path.exists() or not path.is_file():
            continue
        arcname = (Path(prefix) / path.name).as_posix() if prefix else path.name
        entries.append(ZipEntry(path, arcname))
    return sorted(entries, key=_entry_sort_key)


def _uncompressed_size(entries: Sequence[ZipEntry]) -> int:
    """Return the summed source size for entries.

    Parameters
    ----------
    entries:
        Zip entries.

    Returns
    -------
    int
        Total source bytes.
    """

    return sum(entry.source.stat().st_size for entry in entries)


def _assert_free_space(target_dir: Path, required_bytes: int) -> None:
    """Raise when the target filesystem lacks required free bytes.

    Parameters
    ----------
    target_dir:
        Target directory.
    required_bytes:
        Required free bytes.
    """

    target_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(target_dir).free
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"free disk below bundle requirement: {free_bytes} bytes < {required_bytes} bytes"
        )


def _write_zip(zip_path: Path, entries: Sequence[ZipEntry], *, as_of: date | None) -> BundleStats:
    """Write one deterministic zip file.

    Parameters
    ----------
    zip_path:
        Final zip path.
    entries:
        Source entries.
    as_of:
        Optional date used as the fixed zip member timestamp.

    Returns
    -------
    BundleStats
        Written bundle statistics.
    """

    sorted_entries = sorted(entries, key=_entry_sort_key)
    uncompressed_bytes = _uncompressed_size(sorted_entries)
    _assert_free_space(zip_path.parent, uncompressed_bytes + 64 * 1024 * 1024)
    date_time = (
        FIXED_ZIP_DATE_TIME if as_of is None else (as_of.year, as_of.month, as_of.day, 0, 0, 0)
    )
    fd, tmp_name = tempfile.mkstemp(prefix=f".{zip_path.name}.", suffix=".tmp", dir=zip_path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(
            tmp_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=COMPRESSION_LEVEL,
        ) as archive:
            for entry in sorted_entries:
                info = zipfile.ZipInfo(entry.arcname, date_time)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 0
                info.external_attr = 0o644 << 16
                with entry.source.open("rb") as source, archive.open(info, "w") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        os.replace(tmp_path, zip_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    compressed_bytes = zip_path.stat().st_size
    return BundleStats(
        name=zip_path.name,
        path=zip_path,
        file_count=len(sorted_entries),
        uncompressed_bytes=uncompressed_bytes,
        compressed_bytes=compressed_bytes,
        sha256=_sha256_file(zip_path),
    )


def _stats_json(stats: BundleStats) -> dict[str, Any]:
    """Return a JSON-ready bundle stats object.

    Parameters
    ----------
    stats:
        Bundle stats.

    Returns
    -------
    dict[str, Any]
        JSON-ready stats.
    """

    return {
        "name": stats.name,
        "path": "" if stats.path is None else stats.path.name,
        "file_count": stats.file_count,
        "uncompressed_bytes": stats.uncompressed_bytes,
        "compressed_bytes": stats.compressed_bytes,
        "sha256": stats.sha256,
        "skipped": stats.skipped,
        "reason": stats.reason,
    }


def _write_readme(dist_dir: Path, manifest: dict[str, Any]) -> None:
    """Write the top-level distribution README.

    Parameters
    ----------
    dist_dir:
        Distribution directory.
    manifest:
        Manifest payload.
    """

    bundle_lines = [
        f"- {item['name']}: {item['file_count']} files, sha256={item['sha256']}"
        for item in manifest["bundles"]
        if not item["skipped"]
    ]
    if manifest.get("full_bundle", {}).get("skipped"):
        bundle_lines.append(f"- menagerie_full.zip: skipped ({manifest['full_bundle']['reason']})")
    lines = [
        "# TorchLens Menagerie Distribution",
        "",
        f"- Dataset schema version: {manifest['dataset_schema_version']}",
        f"- Dataset as-of date: {manifest['dataset_as_of_date']}",
        f"- Git commit: {manifest['git_commit_sha']}",
        "",
        "## Bundles",
        "",
        *bundle_lines,
        "",
    ]
    (dist_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_bundles(
    *,
    dist_dir: Path = DEFAULT_DIST_DIR,
    tlspec_dir: Path | None = None,
    visuals_dir: Path | None = None,
    csv_dir: Path | None = None,
    catalog_files: Sequence[Path] = CATALOG_FILES,
    max_combined_gb: float = DEFAULT_MAX_COMBINED_GB,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Build all menagerie distribution bundles.

    Parameters
    ----------
    dist_dir:
        Output distribution directory.
    tlspec_dir:
        Source portable trace artifact directory.
    visuals_dir:
        Source gallery directory.
    csv_dir:
        Source CSV export directory.
    catalog_files:
        Source catalog files.
    max_combined_gb:
        Maximum uncompressed GiB allowed for ``menagerie_full.zip``.
    as_of:
        Optional date used for manifest and zip timestamps.

    Returns
    -------
    dict[str, Any]
        Top-level manifest payload.
    """

    dist_dir.mkdir(parents=True, exist_ok=True)
    dataset_as_of = (as_of or datetime.now(UTC).date()).isoformat()
    tlspec_entries = [] if tlspec_dir is None else _file_entries(tlspec_dir, prefix="tlspecs")
    visual_entries = (
        []
        if visuals_dir is None
        else _file_entries(visuals_dir, prefix="visuals", suffixes=VISUAL_SUFFIXES)
    )
    csv_entries = (
        []
        if csv_dir is None
        else _selected_file_entries((csv_dir / name for name in CSV_FILES), prefix="csv")
    )
    catalog_entries = _selected_file_entries(catalog_files, prefix="catalog")

    bundle_inputs = (
        ("menagerie_tlspecs.zip", tlspec_entries),
        ("menagerie_visuals.zip", visual_entries),
        ("menagerie_csv.zip", csv_entries),
        ("menagerie_catalog.zip", catalog_entries),
    )
    stats: list[BundleStats] = []
    for name, entries in bundle_inputs:
        _log(f"writing {name} entries={len(entries)}")
        stats.append(_write_zip(dist_dir / name, entries, as_of=as_of))

    total_uncompressed = sum(item.uncompressed_bytes for item in stats)
    max_combined_bytes = int(max_combined_gb * (1024**3))
    full_stats: BundleStats
    if total_uncompressed <= max_combined_bytes:
        full_entries = (
            [
                ZipEntry(entry.source, f"tlspecs/{entry.arcname.removeprefix('tlspecs/')}")
                for entry in tlspec_entries
            ]
            + [
                ZipEntry(entry.source, f"visuals/{entry.arcname.removeprefix('visuals/')}")
                for entry in visual_entries
            ]
            + [
                ZipEntry(entry.source, f"csv/{entry.arcname.removeprefix('csv/')}")
                for entry in csv_entries
            ]
            + [
                ZipEntry(entry.source, f"catalog/{entry.arcname.removeprefix('catalog/')}")
                for entry in catalog_entries
            ]
        )
        _log(f"writing menagerie_full.zip entries={len(full_entries)}")
        full_stats = _write_zip(dist_dir / "menagerie_full.zip", full_entries, as_of=as_of)
    else:
        reason = (
            f"total uncompressed bytes {total_uncompressed} exceeds threshold "
            f"{max_combined_bytes}; use per-type bundles"
        )
        _log(f"skipping menagerie_full.zip: {reason}")
        full_stats = BundleStats(
            name="menagerie_full.zip",
            path=None,
            file_count=0,
            uncompressed_bytes=total_uncompressed,
            compressed_bytes=0,
            sha256="",
            skipped=True,
            reason=reason,
        )

    manifest = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "dataset_as_of_date": dataset_as_of,
        "git_commit_sha": _git_commit(),
        "bundles": [_stats_json(item) for item in stats],
        "full_bundle": _stats_json(full_stats),
        "download_set": (
            [item.name for item in stats] if full_stats.skipped else [full_stats.name]
        ),
    }
    (dist_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_readme(dist_dir, manifest)
    return manifest


def _parse_as_of(value: str | None) -> date | None:
    """Parse an optional ISO date.

    Parameters
    ----------
    value:
        Date string.

    Returns
    -------
    date | None
        Parsed date or ``None``.
    """

    if value is None:
        return None
    return date.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    """Build the bundler CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=DEFAULT_DIST_DIR)
    parser.add_argument("--tlspec-dir", type=Path)
    parser.add_argument("--visuals-dir", type=Path)
    parser.add_argument("--csv-dir", type=Path)
    parser.add_argument("--max-combined-gb", type=float, default=DEFAULT_MAX_COMBINED_GB)
    parser.add_argument("--as-of", help="ISO date used for dataset_as_of_date and zip mtimes")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bundler CLI.

    Parameters
    ----------
    argv:
        Optional CLI arguments.

    Returns
    -------
    int
        Process exit status.
    """

    args = build_parser().parse_args(argv)
    build_bundles(
        dist_dir=args.dist_dir,
        tlspec_dir=args.tlspec_dir,
        visuals_dir=args.visuals_dir,
        csv_dir=args.csv_dir,
        max_combined_gb=args.max_combined_gb,
        as_of=_parse_as_of(args.as_of),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
