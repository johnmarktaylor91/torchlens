"""Check Total Audit coverage changes against the last committed manifest."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from check_audit_coverage import collect_coverage, flatten_manifest

MANIFEST_PATH = Path("notebooks/total_audit/_coverage_manifest.json")


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk.

    Parameters
    ----------
    path:
        JSON path.

    Returns
    -------
    dict[str, Any]
        Parsed JSON object.
    """

    return json.loads(path.read_text())


def _load_baseline_manifest() -> dict[str, Any] | None:
    """Load the manifest from ``HEAD`` if one is already tracked.

    Returns
    -------
    dict[str, Any] | None
        Baseline manifest, or ``None`` when this is the first manifest.
    """

    result = subprocess.run(
        ["git", "show", f"HEAD:{MANIFEST_PATH.as_posix()}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def _compat_alias_names(manifest: dict[str, Any]) -> set[str]:
    """Return deprecated alias names from a manifest.

    Parameters
    ----------
    manifest:
        Coverage manifest.

    Returns
    -------
    set[str]
        Alias names prefixed as coverage items.
    """

    return {f"tl.{entry['name']}" for entry in manifest.get("compat_aliases", [])}


def check_delta() -> int:
    """Run the coverage delta check.

    Returns
    -------
    int
        Process exit code.
    """

    current = _load_json(MANIFEST_PATH)
    baseline = _load_baseline_manifest()
    if baseline is None:
        print("No committed coverage manifest found; treating current manifest as baseline.")
        baseline = current

    baseline_public = set(flatten_manifest(baseline).keys())
    current_public = set(flatten_manifest(current).keys())
    new_public = sorted(current_public - baseline_public)

    statuses = collect_coverage(current)
    new_uncovered = [item for item in new_public if item in statuses and not statuses[item].called]

    baseline_aliases = _compat_alias_names(baseline)
    current_aliases = _compat_alias_names(current)
    removed_aliases = sorted(baseline_aliases - current_aliases)

    failed = False
    if new_uncovered:
        failed = True
        print(f"New public items without called audit coverage: {len(new_uncovered)}")
        for item in new_uncovered:
            print(f"  - {item}")
    if removed_aliases:
        failed = True
        print(f"Compat aliases removed from manifest: {len(removed_aliases)}")
        for item in removed_aliases:
            print(f"  - {item}")
    if failed:
        return 1

    print(f"Coverage delta OK: {len(new_public)} new public items, 0 uncovered.")
    print("Compat alias deletion check OK.")
    return 0


def main() -> None:
    """Run the command-line entry point."""

    raise SystemExit(check_delta())


if __name__ == "__main__":
    main()
