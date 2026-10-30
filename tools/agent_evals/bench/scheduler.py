"""Randomized, interleaved run order, so quota windows and drift hit every arm equally.

The plan is built in blocks: block ``r`` holds every cell (task, arm, tier) once, as repeat
``r``, in a seeded random order. Running the blocks in sequence means any stretch of time (a
quota window, a busy worker, a provider change) touches every arm and tier about equally, and a
stopped study still has balanced complete blocks. Cells where a task does not apply to an arm are
kept in the plan as ``not_applicable`` rows so they are reported, not silently dropped.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlannedRun:
    """One row of the plan."""

    order: int
    block: int
    task: str
    arm: str
    tier: str
    repeat: int
    applicable: bool

    @property
    def run_key(self) -> str:
        """A stable id for the row (task, arm, tier, repeat)."""

        return f"{self.task}-{self.arm}-{self.tier}-r{self.repeat}"


def plan(
    cells: Iterable[tuple[str, str, str]],
    repeats: int,
    seed: int,
    applicable: Mapping[tuple[str, str], bool] | None = None,
) -> list[PlannedRun]:
    """Return the run order for ``cells`` (task, arm, tier) over ``repeats`` blocks.

    Parameters
    ----------
    cells:
        The distinct (task, arm, tier) triples to run.
    repeats:
        Number of blocks; each block runs every cell once.
    seed:
        Seed for the order, recorded with the plan so it can be rebuilt.
    applicable:
        ``(task, arm) -> bool``; missing pairs count as applicable.
    """

    distinct = sorted(set(cells))
    rng = random.Random(seed)
    rows: list[PlannedRun] = []
    for block in range(repeats):
        order = distinct[:]
        rng.shuffle(order)
        for task, arm, tier in order:
            ok = True if applicable is None else applicable.get((task, arm), True)
            rows.append(PlannedRun(len(rows), block, task, arm, tier, block, ok))
    return rows


def pilot_cells() -> list[tuple[str, str, str]]:
    """The pilot (FINAL-PLAN 6.2): T1, T7, T12 in arms A and B at small and mid; C on T1.

    With three repeats this is 36 plus 6 runs, 42 in all.
    """

    cells = [
        (task, arm, tier)
        for task in ("T1", "T7", "T12")
        for arm in ("A", "B")
        for tier in ("small", "mid")
    ]
    return cells + [("T1", "C", tier) for tier in ("small", "mid")]


def balance(rows: Sequence[PlannedRun], window: int) -> dict[str, dict[str, int]]:
    """Count runs per arm in consecutive windows of ``window`` rows (a check on interleaving)."""

    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        key = f"w{row.order // window}"
        counts.setdefault(key, {})
        counts[key][row.arm] = counts[key].get(row.arm, 0) + 1
    return counts


def write(rows: Sequence[PlannedRun], path: Path, seed: int) -> Path:
    """Write the plan as JSON lines, with a header line naming the seed."""

    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"plan": "agent-bench", "seed": seed, "rows": len(rows)}) + "\n")
        for row in rows:
            handle.write(json.dumps({**asdict(row), "run_key": row.run_key}) + "\n")
    return path


def pending(rows: Sequence[PlannedRun], results_dir: Path) -> list[PlannedRun]:
    """Return the rows with no record yet under ``results_dir/<run_key>/record.json``."""

    return [row for row in rows if not (results_dir / row.run_key / "record.json").exists()]
