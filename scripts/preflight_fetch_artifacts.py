"""Preflight fetcher for the real-model artifact cache (testing MEMO 4.3, A2).

No pytest body reaches the network, ever. This script runs BEFORE pytest:
it fetches exactly the registry blobs for the requested venue/rows into an
isolated cache, verifies size and SHA-256 for every file (including every
cache-restored file: verify-on-hit, because the CI cache key is exact and
prefix fallback is forbidden), enforces the byte-budget cap by lstat-walking
``blobs/`` (never following the HF snapshot symlinks -- the measured
double-count trap), and prints the offline env pytest must run under.

A missing or corrupt core artifact FAILS the preflight loudly with a nonzero
exit; it never becomes a per-test skip.

Usage::

    python scripts/preflight_fetch_artifacts.py fetch --venue pr_telemetry --cache-dir .artifact-cache
    python scripts/preflight_fetch_artifacts.py verify --venue nightly --cache-dir .artifact-cache
    python scripts/preflight_fetch_artifacts.py cache-key
    python scripts/preflight_fetch_artifacts.py lint-bytes --cache-dir .artifact-cache
    python scripts/preflight_fetch_artifacts.py print-env --cache-dir .artifact-cache
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.real_model.registry import (  # noqa: E402
    ArtifactRow,
    Registry,
    load_registry,
)

USER_AGENT = "torchlens-preflight"
TORCHVISION_BASE = "https://download.pytorch.org/models/"


def _hf_home(cache_dir: Path) -> Path:
    return cache_dir / "hf"


def _torch_home(cache_dir: Path) -> Path:
    return cache_dir / "torch"


def offline_env(cache_dir: Path) -> dict[str, str]:
    """The exact env pytest runs under after a green preflight."""

    return {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(_hf_home(cache_dir).resolve()),
        "TORCH_HOME": str(_torch_home(cache_dir).resolve()),
        "TORCHLENS_REALMODEL_CACHE": str(cache_dir.resolve()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response, dest.open("wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)


def _hub_dir(row: ArtifactRow, cache_dir: Path) -> Path:
    flat = row.model_id.replace("/", "--")
    return _hf_home(cache_dir) / "hub" / f"models--{flat}"


def _hub_paths(row: ArtifactRow, cache_dir: Path) -> list[tuple[Path, str, int | None]]:
    """(snapshot path, expected sha256, expected size) per blob of a hub row."""

    base = _hub_dir(row, cache_dir) / "snapshots" / str(row.revision)
    return [(base / blob.filename, blob.sha256 or "", blob.size_bytes) for blob in row.blobs]


def _torchvision_paths(row: ArtifactRow, cache_dir: Path) -> list[tuple[Path, str, int | None]]:
    base = _torch_home(cache_dir) / "hub" / "checkpoints"
    return [(base / blob.filename, blob.sha256 or "", blob.size_bytes) for blob in row.blobs]


def _verify_file(path: Path, sha256: str, size: int | None, failures: list[str]) -> None:
    if not path.exists():
        failures.append(f"MISSING {path}")
        return
    actual_size = path.stat().st_size
    if size is not None and actual_size != size:
        failures.append(f"SIZE {path}: {actual_size} != registry {size}")
        return
    if sha256:
        actual = _sha256(path)
        if actual != sha256:
            failures.append(f"SHA256 {path}: {actual[:16]}... != registry {sha256[:16]}...")


def _rows_for(registry: Registry, venue: str | None, ids: list[str]) -> list[ArtifactRow]:
    if ids:
        rows = [registry.get(artifact_id) for artifact_id in ids]
    elif venue:
        rows = list(registry.rows_for_venue(venue))
    else:
        raise SystemExit("pass --venue or --rows")
    return [row for row in rows if row.kind in ("hf_hub", "torchvision_weights") and row.blobs]


def cmd_fetch(registry: Registry, args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)
    failures: list[str] = []
    fetched = verified = 0
    for row in _rows_for(registry, args.venue, args.rows):
        if row.kind == "hf_hub":
            paths = _hub_paths(row, cache_dir)

            def url_for(blob_name: str, r: ArtifactRow = row) -> str:
                return f"https://huggingface.co/{r.model_id}/resolve/{r.revision}/{blob_name}"
        else:
            paths = _torchvision_paths(row, cache_dir)

            def url_for(blob_name: str, r: ArtifactRow = row) -> str:
                return TORCHVISION_BASE + blob_name

        for (path, sha256, size), blob in zip(paths, row.blobs, strict=True):
            if path.exists():
                # verify-on-hit: every restored blob is rehashed; a stale or
                # truncated cache entry is refetched, never trusted.
                pre: list[str] = []
                _verify_file(path, sha256, size, pre)
                if not pre:
                    verified += 1
                    continue
                print(f"preflight: cache hit failed verification, refetching: {pre[0]}")
                path.unlink()
            _download(url_for(blob.filename), path)
            fetched += 1
            if not sha256:
                # A row can enter the registry without a full digest only
                # until first measured; an unpinned blob NEVER verifies
                # silently green -- the fetch fails with the measurement to pin.
                failures.append(
                    f"UNPINNED {row.artifact_id}/{blob.filename}: measured"
                    f" sha256={_sha256(path)} size={path.stat().st_size};"
                    " pin these in artifact_registry.jsonl"
                )
                continue
            post: list[str] = []
            _verify_file(path, sha256, size, post)
            failures.extend(post)
    if failures:
        print("preflight FETCH FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"preflight fetch ok: {fetched} fetched, {verified} verified-on-hit")
    return cmd_lint_bytes(registry, args)


def cmd_verify(registry: Registry, args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)
    failures: list[str] = []
    count = 0
    for row in _rows_for(registry, args.venue, args.rows):
        paths = (
            _hub_paths(row, cache_dir)
            if row.kind == "hf_hub"
            else _torchvision_paths(row, cache_dir)
        )
        for path, sha256, size in paths:
            if not sha256:
                failures.append(f"UNPINNED {row.artifact_id}: {path.name} has no registry sha256")
                continue
            _verify_file(path, sha256, size, failures)
            count += 1
    if failures:
        print("preflight VERIFY FAILED (missing core artifacts fail loudly;", file=sys.stderr)
        print("they never become per-test skips):", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"preflight verify ok: {count} blobs")
    return 0


def _lstat_bytes(root: Path) -> int:
    """Sum file bytes under ``root`` with lstat, never following symlinks.

    The HF cache's ``snapshots/`` tree symlinks into ``blobs/``; following
    the links double-counts every blob (the measured trap the memo names).
    """

    total = 0
    if not root.exists():
        return 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            info = os.lstat(os.path.join(dirpath, filename))
            if not os.path.islink(os.path.join(dirpath, filename)):
                total += info.st_size
    return total


def cmd_lint_bytes(registry: Registry, args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)
    cap = registry.nightly_cap_bytes()
    total = _lstat_bytes(cache_dir)
    if total > cap:
        print(
            f"byte-budget lint FAILED: cache holds {total / 1e6:.0f} MB >"
            f" cap {cap / 1e6:.0f} MB (nightly set + 20%, memo 4.3). Remedy:"
            " drop rows from the venue, or re-derive the cap after a conscious"
            " registry change.",
            file=sys.stderr,
        )
        return 1
    print(f"byte-budget lint ok: {total / 1e6:.0f} MB <= cap {cap / 1e6:.0f} MB")
    return 0


def cmd_cache_key(registry: Registry, _args: argparse.Namespace) -> int:
    print(registry.cache_key())
    return 0


def cmd_print_env(_registry: Registry, args: argparse.Namespace) -> int:
    for key, value in offline_env(Path(args.cache_dir)).items():
        print(f"{key}={value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["fetch", "verify", "cache-key", "lint-bytes", "print-env"]
    )
    parser.add_argument("--venue", default=None, help="registry venue (e.g. pr_telemetry)")
    parser.add_argument(
        "--rows", default="", help="comma-separated artifact ids (overrides --venue)"
    )
    parser.add_argument("--cache-dir", default=".artifact-cache")
    parser.add_argument("--registry", default=None, help="alternate registry path (tests)")
    args = parser.parse_args(argv)
    args.rows = [row for row in args.rows.split(",") if row]
    registry = load_registry(Path(args.registry)) if args.registry else load_registry()
    handler = {
        "fetch": cmd_fetch,
        "verify": cmd_verify,
        "cache-key": cmd_cache_key,
        "lint-bytes": cmd_lint_bytes,
        "print-env": cmd_print_env,
    }[args.command]
    return handler(registry, args)


if __name__ == "__main__":
    raise SystemExit(main())
