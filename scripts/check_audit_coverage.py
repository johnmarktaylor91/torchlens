"""Check Total Audit notebook coverage against the generated manifest."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path("notebooks/total_audit/_coverage_manifest.json")
NOTEBOOK_DIR = Path("notebooks/total_audit")
MATRIX_PATH = NOTEBOOK_DIR / "_coverage_matrix.md"
EXCEPTIONS_PATH = NOTEBOOK_DIR / "_coverage_exceptions.txt"


@dataclass
class CoverageStatus:
    """Coverage status for one manifest item.

    Attributes
    ----------
    item:
        Canonical coverage item name.
    category:
        Manifest category for the item.
    mentioned:
        Notebook files whose source mentions the item.
    called:
        Notebook files whose executable coverage metadata calls the item.
    expected_failure:
        Notebook files whose metadata records an expected failure for the item.
    """

    item: str
    category: str
    mentioned: set[str] = field(default_factory=set)
    called: set[str] = field(default_factory=set)
    expected_failure: set[str] = field(default_factory=set)


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk.

    Parameters
    ----------
    path:
        JSON file path.

    Returns
    -------
    dict[str, Any]
        Parsed JSON object.
    """

    return json.loads(path.read_text())


def flatten_manifest(
    manifest: dict[str, Any], *, include_compat_aliases: bool = False
) -> dict[str, str]:
    """Flatten manifest records into canonical coverage item names.

    Parameters
    ----------
    manifest:
        Generated coverage manifest.
    include_compat_aliases:
        Whether deprecated top-level aliases should be included.

    Returns
    -------
    dict[str, str]
        Mapping of coverage item name to category.
    """

    items: dict[str, str] = {}
    for entry in manifest.get("top_level", []):
        items[f"tl.{entry['name']}"] = "top_level"
    for submodule_name, submodule_entry in manifest.get("submodules", {}).items():
        items[f"tl.{submodule_name}"] = "submodule"
        for name in submodule_entry.get("names", []):
            items[f"tl.{submodule_name}.{name}"] = "submodule"
    for class_name, member_entries in manifest.get("classes", {}).items():
        items[f"tl.{class_name}"] = "class"
        for entry in member_entries:
            items[f"tl.{class_name}.{entry['name']}"] = "class_member"
    if include_compat_aliases:
        for entry in manifest.get("compat_aliases", []):
            items[f"tl.{entry['name']}"] = "compat_alias"
    return dict(sorted(items.items()))


def _notebook_paths() -> list[Path]:
    """Return Total Audit notebook paths in execution order.

    Returns
    -------
    list[pathlib.Path]
        Notebook paths.
    """

    return sorted(path for path in NOTEBOOK_DIR.glob("*.ipynb") if path.name[:2].isdigit())


def _load_exception_items(path: Path = EXCEPTIONS_PATH) -> set[str]:
    """Load strict-mode coverage exceptions.

    Parameters
    ----------
    path:
        Exception file path.

    Returns
    -------
    set[str]
        Exact coverage item names exempted from strict called coverage.
    """

    if not path.exists():
        return set()
    exceptions: set[str] = set()
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            exceptions.add(line.split("#", 1)[0].strip())
    return exceptions


def _cell_source(cell: dict[str, Any]) -> str:
    """Return a notebook cell source as a string.

    Parameters
    ----------
    cell:
        Notebook cell dictionary.

    Returns
    -------
    str
        Cell source.
    """

    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def _metadata_list(cell: dict[str, Any], key: str) -> list[str]:
    """Return a string list from notebook cell metadata.

    Parameters
    ----------
    cell:
        Notebook cell dictionary.
    key:
        Metadata key.

    Returns
    -------
    list[str]
        Metadata values coerced to strings.
    """

    raw_value = cell.get("metadata", {}).get(key, [])
    if not isinstance(raw_value, list):
        raise TypeError(f"Cell metadata {key!r} must be a list, got {type(raw_value).__name__}.")
    return [str(value) for value in raw_value]


def collect_coverage(
    manifest: dict[str, Any],
    *,
    include_compat_aliases: bool = False,
) -> dict[str, CoverageStatus]:
    """Collect mentioned, called, and expected-failure coverage from notebooks.

    Parameters
    ----------
    manifest:
        Generated coverage manifest.
    include_compat_aliases:
        Whether deprecated aliases should be tracked.

    Returns
    -------
    dict[str, CoverageStatus]
        Coverage status by item.
    """

    flattened = flatten_manifest(manifest, include_compat_aliases=include_compat_aliases)
    statuses = {
        item: CoverageStatus(item=item, category=category) for item, category in flattened.items()
    }
    item_patterns = {
        item: re.compile(rf"(?<![\w.]){re.escape(item)}(?![\w.])") for item in statuses
    }

    for path in _notebook_paths():
        notebook = _load_json(path)
        notebook_name = path.name
        for cell in notebook.get("cells", []):
            source = _cell_source(cell)
            if source:
                for item, pattern in item_patterns.items():
                    if pattern.search(source):
                        statuses[item].mentioned.add(notebook_name)

            if cell.get("cell_type") != "code":
                continue

            for item in _metadata_list(cell, "coverage_calls"):
                if item in statuses:
                    statuses[item].called.add(notebook_name)
                    statuses[item].mentioned.add(notebook_name)
            for item in _metadata_list(cell, "coverage_expected_failure"):
                if item in statuses:
                    statuses[item].expected_failure.add(notebook_name)
                    statuses[item].mentioned.add(notebook_name)
    return statuses


def write_matrix(statuses: dict[str, CoverageStatus], path: Path = MATRIX_PATH) -> None:
    """Write a markdown coverage matrix.

    Parameters
    ----------
    statuses:
        Coverage status by item.
    path:
        Destination markdown path.
    """

    lines = [
        "# TorchLens Total Audit Coverage Matrix",
        "",
        "| Item | Category | Mentioned | Called | Expected failure |",
        "|---|---|---|---|---|",
    ]
    for item, status in sorted(statuses.items()):
        mentioned = ", ".join(sorted(status.mentioned)) or "-"
        called = ", ".join(sorted(status.called)) or "-"
        expected_failure = ", ".join(sorted(status.expected_failure)) or "-"
        lines.append(
            f"| `{item}` | {status.category} | {mentioned} | {called} | {expected_failure} |"
        )
    path.write_text("\n".join(lines) + "\n")


def check(strict: bool = False, *, include_compat_aliases: bool = False) -> int:
    """Run the coverage check.

    Parameters
    ----------
    strict:
        Whether every non-excepted public item must reach called coverage.
    include_compat_aliases:
        Whether deprecated aliases should be included in the matrix.

    Returns
    -------
    int
        Process exit code.
    """

    manifest = _load_json(MANIFEST_PATH)
    statuses = collect_coverage(manifest, include_compat_aliases=include_compat_aliases)
    write_matrix(statuses)

    exceptions = _load_exception_items()
    missing_called = [
        item
        for item, status in statuses.items()
        if status.category != "compat_alias" and item not in exceptions and not status.called
    ]
    if strict and missing_called:
        print(f"Strict coverage failed: {len(missing_called)} items lack called coverage.")
        for item in missing_called[:50]:
            print(f"  - {item}")
        if len(missing_called) > 50:
            print(f"  ... {len(missing_called) - 50} more")
        print(f"Wrote {MATRIX_PATH}")
        return 1

    print(f"Wrote {MATRIX_PATH}")
    print(f"Coverage gaps: {len(missing_called)}")
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Argument parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Fail unless every item is called.")
    parser.add_argument(
        "--include-compat-aliases",
        action="store_true",
        help="Include deprecated aliases in the generated matrix.",
    )
    return parser


def main() -> None:
    """Run the command-line entry point."""

    args = _parser().parse_args()
    raise SystemExit(check(strict=args.strict, include_compat_aliases=args.include_compat_aliases))


if __name__ == "__main__":
    main()
