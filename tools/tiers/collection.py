"""Publish the test-tier collection artifact (``tests/tiers/collection.json``).

The tier counts quoted in ``test-tiers.json``, ``AGENTS.md`` and ``tests/AGENTS.md`` come
from this one artifact, so they are measured instead of remembered. The script runs no
tests and no collection itself: it reads one ``pytest --collect-only -q`` output file per
tier (produced in the pinned smoke environment, see ``TIERS`` for the exact commands) and a
per-test durations CSV, and writes per tier the collected count, a digest of the sorted
test ids, the collect command, the source commit and a wall estimate.

Generate::

    python tools/tiers/collection.py --tier smoke=smoke.txt --tier mid=mid.txt \\
        --tier not_slow=notslow.txt --tier all=all.txt \\
        --durations durations.csv --source-commit 1a2b3c4d

Check the committed artifact against a fresh collect output (exit 1 on drift)::

    python tools/tiers/collection.py --check --tier smoke=smoke.txt [--tier ...]

``--durations`` takes either a per-test CSV (``test_id,seconds,...``, as written by a
``--durations=0`` run) or the committed per-file summary (``file,tests,seconds``). A
per-test CSV is also summarized per file into ``tests/tiers/durations.csv``; with the
summary alone, each tier's estimate is proportional to its share of each file's tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "tests" / "tiers" / "collection.json"
DURATIONS_SUMMARY = REPO_ROOT / "tests" / "tiers" / "durations.csv"

#: Every collect runs in the pinned smoke environment with this command prefix.
COLLECT_PREFIX = (
    "CI=true OMP_NUM_THREADS=1 TORCHLENS_ORACLE_ENFORCE=1 "
    "python -m pytest --collect-only -q -o addopts= -p no:randomly -p no:cacheprovider"
)

#: Tier key -> marker expression (``None`` collects everything, rare included).
#: ``unmarked`` is the cost-tier census row of ``tests/AGENTS.md`` (``serial`` and the
#: selection markers are orthogonal to cost tiers, so they do not count as marked).
TIERS: dict[str, str | None] = {
    "smoke": "smoke",
    "mid": "not rare and not slow and not heavy",
    "not_slow": "not rare and not slow",
    "all": None,
    "unmarked": "not smoke and not heavy and not slow and not rare",
}

#: Workers the wall estimate divides the serial seconds across (CI's smoke ``-n``).
ESTIMATE_WORKERS = 4

_SUMMARY_RE = re.compile(r"^(?:(\d+)/)?(\d+) tests? collected")
_ERROR_RE = re.compile(r"\berrors?\b")


@dataclass(frozen=True)
class Collected:
    """The test ids one collect-only output file lists."""

    ids: tuple[str, ...]

    @property
    def per_file(self) -> Counter[str]:
        """Return the collected count per test file."""

        return Counter(node_id.split("::", 1)[0] for node_id in self.ids)


def collect_command(tier: str) -> str:
    """Return the exact collect command a tier's output file must come from."""

    marker = TIERS[tier]
    selector = "" if marker is None else f' -m "{marker}"'
    return f"{COLLECT_PREFIX}{selector} tests/"


def id_digest(ids: tuple[str, ...]) -> int:
    """Return the first 48 bits of the sha256 of the sorted ids, one per line.

    An integer, not a hex string: the repo's secret scan flags long hex strings in JSON.
    """

    payload = "\n".join(sorted(ids)).encode()
    return int(hashlib.sha256(payload).hexdigest()[:12], 16)


def parse_collect_output(text: str) -> Collected:
    """Parse ``pytest --collect-only -q`` output; refuse errors or a count mismatch."""

    ids: list[str] = []
    reported: int | None = None
    for line in text.splitlines():
        if line.startswith("tests/") and "::" in line:
            ids.append(line.strip())
            continue
        match = _SUMMARY_RE.match(line.strip())
        if match:
            reported = int(match.group(1) or match.group(2))
            if _ERROR_RE.search(line):
                raise ValueError(f"collection reported errors: {line.strip()!r}")
    if reported is None:
        raise ValueError("no 'N tests collected' summary line; not a collect-only -q output")
    if reported != len(ids):
        raise ValueError(f"summary says {reported} tests but {len(ids)} ids were listed")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate test ids in the collect output")
    return Collected(tuple(ids))


def read_durations(path: Path) -> tuple[dict[str, float], dict[str, tuple[int, float]]]:
    """Return ``(per_test_seconds, per_file_summary)``; one of them may be empty."""

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "test_id" in rows[0]:
        per_test = {row["test_id"]: float(row["seconds"]) for row in rows}
        return per_test, summarize_per_file(per_test)
    if rows and {"file", "tests", "seconds"} <= set(rows[0]):
        summary = {row["file"]: (int(row["tests"]), float(row["seconds"])) for row in rows}
        return {}, summary
    raise ValueError(f"{path}: expected test_id,seconds or file,tests,seconds columns")


def summarize_per_file(per_test: dict[str, float]) -> dict[str, tuple[int, float]]:
    """Fold per-test seconds into ``file -> (tests, seconds)``."""

    summary: dict[str, tuple[int, float]] = {}
    for test_id, seconds in per_test.items():
        file = test_id.split("::", 1)[0]
        count, total = summary.get(file, (0, 0.0))
        summary[file] = (count + 1, total + seconds)
    return summary


def write_durations_summary(summary: dict[str, tuple[int, float]], path: Path) -> None:
    """Write the per-file summary, sorted by file, seconds rounded to 0.01."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["file", "tests", "seconds"])
        for file in sorted(summary):
            count, seconds = summary[file]
            writer.writerow([file, count, f"{seconds:.2f}"])


def wall_estimate(
    collected: Collected,
    per_test: dict[str, float],
    summary: dict[str, tuple[int, float]],
) -> dict[str, Any]:
    """Estimate a tier's test seconds from per-test durations, else per-file means."""

    if per_test:
        known = [per_test[node_id] for node_id in collected.ids if node_id in per_test]
        serial, with_duration, method = sum(known), len(known), "per_test"
    else:
        serial, with_duration = 0.0, 0
        for file, count in collected.per_file.items():
            if file in summary and summary[file][0]:
                tests, seconds = summary[file]
                serial += seconds * count / tests
                with_duration += count
        method = "per_file_mean"
    return {
        "method": method,
        "tests_with_duration": with_duration,
        "tests_without_duration": len(collected.ids) - with_duration,
        "serial_seconds": round(serial, 1),
        f"n{ESTIMATE_WORKERS}_seconds": round(serial / ESTIMATE_WORKERS, 1),
    }


def tier_entry(
    tier: str,
    collected: Collected,
    per_test: dict[str, float],
    summary: dict[str, tuple[int, float]],
) -> dict[str, Any]:
    """Return one tier's artifact entry."""

    return {
        "count": len(collected.ids),
        "files": len(collected.per_file),
        "id_digest": id_digest(collected.ids),
        "collect_command": collect_command(tier),
        "wall_estimate": wall_estimate(collected, per_test, summary),
    }


def build_artifact(
    tiers: dict[str, Collected],
    per_test: dict[str, float],
    summary: dict[str, tuple[int, float]],
    *,
    source_commit: str,
    durations_source: str,
) -> dict[str, Any]:
    """Assemble the artifact dict from parsed tiers and durations."""

    return {
        "schema": 1,
        "source_commit": source_commit,
        "environment": "pinned smoke row: python 3.10, torch 2.13.0+cpu",
        "durations_source": durations_source,
        "estimate_workers": ESTIMATE_WORKERS,
        "tiers": {
            tier: tier_entry(tier, tiers[tier], per_test, summary)
            for tier in TIERS
            if tier in tiers
        },
    }


def check_artifact(artifact: dict[str, Any], tiers: dict[str, Collected]) -> list[str]:
    """Return one line per tier whose fresh collect differs from the artifact."""

    problems: list[str] = []
    for tier, collected in tiers.items():
        entry = artifact["tiers"].get(tier)
        if entry is None:
            problems.append(f"{tier}: not in the artifact")
            continue
        if entry["count"] != len(collected.ids):
            problems.append(f"{tier}: artifact count {entry['count']}, fresh {len(collected.ids)}")
        if entry["id_digest"] != id_digest(collected.ids):
            problems.append(f"{tier}: test ids differ from the artifact (digest mismatch)")
    return problems


def _parse_tier_args(values: list[str]) -> dict[str, Collected]:
    tiers: dict[str, Collected] = {}
    for value in values:
        tier, sep, path = value.partition("=")
        if not sep or tier not in TIERS:
            raise SystemExit(f"--tier expects NAME=PATH with NAME in {sorted(TIERS)}: {value!r}")
        tiers[tier] = parse_collect_output(Path(path).read_text())
    return tiers


def main(argv: list[str] | None = None) -> int:
    """Generate or check the artifact; returns the process exit code."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tier", action="append", default=[], help="NAME=collect-output path")
    parser.add_argument("--durations", type=Path, help="per-test CSV or per-file summary")
    parser.add_argument("--source-commit", help="short commit the collect ran at")
    parser.add_argument("--durations-note", help="where the per-test durations came from")
    parser.add_argument("--out", type=Path, default=ARTIFACT)
    parser.add_argument("--durations-out", type=Path, default=DURATIONS_SUMMARY)
    parser.add_argument("--check", action="store_true", help="compare, do not write")
    args = parser.parse_args(argv)
    tiers = _parse_tier_args(args.tier)
    if not tiers:
        parser.error("at least one --tier NAME=PATH is required")
    if args.check:
        problems = check_artifact(json.loads(args.out.read_text()), tiers)
        for line in problems:
            print(f"tier drift: {line}")
        print("tier artifact: " + ("STALE" if problems else f"current for {sorted(tiers)}"))
        return 1 if problems else 0
    if missing := set(TIERS) - set(tiers):
        parser.error(f"generation needs every tier; missing {sorted(missing)}")
    if args.durations is None or not args.source_commit:
        parser.error("generation needs --durations and --source-commit")
    per_test, summary = read_durations(args.durations)
    if per_test:
        write_durations_summary(summary, args.durations_out)
    artifact = build_artifact(
        tiers,
        per_test,
        summary,
        source_commit=args.source_commit,
        durations_source=args.durations_note or "tests/tiers/durations.csv",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=False) + "\n")
    print(f"wrote {args.out} ({', '.join(f'{t}={len(c.ids)}' for t, c in tiers.items())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
