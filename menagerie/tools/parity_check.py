"""Phase-1a menagerie catalog parity checks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from menagerie.catalog import CatalogRow, build_canonical_rows
from menagerie.classics import CLASSIC_ZOO, CLASSICS
from menagerie.generate_menagerie import parse_shape, unrenderable_reason


DEFAULT_BASELINE = Path(".research/menagerie-redesign/phase0_baseline.json")
RenderableKey = tuple[str, str, str, str, str]


def _renderable_keys(rows: Sequence[CatalogRow]) -> set[RenderableKey]:
    """Return renderable row keys using the renderer's current input gates.

    Parameters
    ----------
    rows:
        Canonical catalog rows.

    Returns
    -------
    set[RenderableKey]
        Renderable row identity tuples.
    """

    renderable = set()
    for row in rows:
        reason = unrenderable_reason(row)
        if reason is not None:
            continue
        if row.source == "classics":
            renderable.add(
                (row.name, row.zoo, row.constructor_call, row.input_shape, row.input_dtype)
            )
            continue
        try:
            parse_shape(row.input_shape)
        except Exception:  # noqa: BLE001 - parity reports the row as non-renderable.
            continue
        renderable.add((row.name, row.zoo, row.constructor_call, row.input_shape, row.input_dtype))
    return renderable


def _load_baseline(path: Path) -> dict[str, Any]:
    """Load the Step-1 baseline JSON.

    Parameters
    ----------
    path:
        Baseline JSON path.

    Returns
    -------
    dict[str, Any]
        Parsed baseline.
    """

    return json.loads(path.read_text())


def _baseline_floor(baseline: dict[str, Any]) -> set[RenderableKey]:
    """Return the baseline renderable floor after intended classics dedup.

    Parameters
    ----------
    baseline:
        Parsed Step-1 baseline.

    Returns
    -------
    set[RenderableKey]
        Baseline renderable rows that must still be present exactly.
    """

    floor = set()
    for raw_key in baseline["renderable_rows"]:
        key = tuple(str(part) for part in raw_key)
        if len(key) != 5:
            raise ValueError(f"baseline renderable key has {len(key)} fields: {raw_key!r}")
        name, zoo, _constructor_call, _input_shape, _input_dtype = key
        if zoo == CLASSIC_ZOO and name in CLASSICS:
            continue
        floor.add(key)
    return floor


def _check_classics_count(rows: Sequence[CatalogRow], errors: list[str]) -> None:
    """Check registry classics are counted exactly once.

    Parameters
    ----------
    rows:
        Current canonical rows.
    errors:
        Mutable error accumulator.
    """

    classics_rows = [row for row in rows if row.source == "classics"]
    if len(classics_rows) != len(CLASSICS):
        errors.append(
            f"classics count mismatch: source=classics rows {len(classics_rows)} != "
            f"len(CLASSICS) {len(CLASSICS)}"
        )
    classic_names = {row.name for row in classics_rows}
    if classic_names != set(CLASSICS):
        missing = sorted(set(CLASSICS) - classic_names)
        extra = sorted(classic_names - set(CLASSICS))
        errors.append(
            f"classics registry name mismatch: missing={missing[:10]!r} extra={extra[:10]!r}"
        )


def _check_natural_keys(rows: Sequence[CatalogRow], errors: list[str]) -> None:
    """Check ``(name, zoo, variant)`` uniqueness.

    Parameters
    ----------
    rows:
        Current canonical rows.
    errors:
        Mutable error accumulator.
    """

    duplicates = [
        key
        for key, count in Counter((row.name, row.zoo, row.variant) for row in rows).items()
        if count > 1
    ]
    if duplicates:
        errors.append(f"duplicate natural keys: {duplicates[:20]!r}")


def _check_renderable_floor(
    baseline: dict[str, Any],
    rows: Sequence[CatalogRow],
    errors: list[str],
) -> None:
    """Check no non-classics renderable baseline row disappeared.

    Parameters
    ----------
    baseline:
        Parsed Step-1 baseline.
    rows:
        Current canonical rows.
    errors:
        Mutable error accumulator.
    """

    floor = _baseline_floor(baseline)
    current = _renderable_keys(rows)
    missing = sorted(floor - current)
    if missing:
        errors.append(f"non-classics renderable rows disappeared: {missing[:20]!r}")

    current_classic_names = {row.name for row in rows if row.source == "classics"}
    if current_classic_names != set(CLASSICS):
        missing_names = sorted(set(CLASSICS) - current_classic_names)
        extra_names = sorted(current_classic_names - set(CLASSICS))
        errors.append(
            f"classics names changed: missing={missing_names[:20]!r} extra={extra_names[:20]!r}"
        )


def _check_stable_id_determinism(errors: list[str]) -> None:
    """Check stable-ID assignment is identical across consecutive builds.

    Parameters
    ----------
    errors:
        Mutable error accumulator.
    """

    first = build_canonical_rows()
    second = build_canonical_rows()
    first_ids = {(row.name, row.zoo, row.variant): row.stable_id for row in first}
    second_ids = {(row.name, row.zoo, row.variant): row.stable_id for row in second}
    if first_ids != second_ids:
        changed = sorted(
            key
            for key in set(first_ids) | set(second_ids)
            if first_ids.get(key) != second_ids.get(key)
        )
        errors.append(f"stable ID assignment changed across two builds: {changed[:20]!r}")


def run_parity_check(baseline_path: Path) -> int:
    """Run Phase-1a parity checks.

    Parameters
    ----------
    baseline_path:
        Step-1 baseline JSON path.

    Returns
    -------
    int
        Process exit code.
    """

    baseline = _load_baseline(baseline_path)
    rows = build_canonical_rows()
    errors: list[str] = []
    _check_classics_count(rows, errors)
    _check_natural_keys(rows, errors)
    _check_renderable_floor(baseline, rows, errors)
    _check_stable_id_determinism(errors)
    if errors:
        print("Phase-1a parity check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Phase-1a parity check passed")
    print(f"canonical_rows={len(rows)}")
    print(f"classics_rows={sum(row.source == 'classics' for row in rows)}")
    print(f"len_CLASSICS={len(CLASSICS)}")
    print(f"baseline_floor={len(_baseline_floor(baseline))}")
    print(f"current_renderable={len(_renderable_keys(rows))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the parity-check CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the parity-check CLI.

    Parameters
    ----------
    argv:
        Optional argument vector.

    Returns
    -------
    int
        Process exit code.
    """

    args = build_parser().parse_args(argv)
    return run_parity_check(args.baseline)


if __name__ == "__main__":
    raise SystemExit(main())
