"""Harvestability spike (row 0.9): count taught paths + claim strings NOW.

The taught-path and prose-claim universes stay OPEN until their parsers
exist; this spike proves the harvest is feasible and pins TODAY's counts so
the sweeps written against them at wave A start from a measured baseline,
never a guess. Fallback, stated in the ledger: declared allowlists with
weaker closure.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.compo]

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"

_FENCE_PATTERN = re.compile(r"^```python\s*$", re.MULTILINE)
_CLAIM_PATTERN = re.compile(
    r"\b(not supported|unsupported|not tracked|refuses?d? typed|deferred)\b", re.IGNORECASE
)


def _doc_files() -> tuple[Path, ...]:
    """The taught-surface corpus: docs/** markdown plus the README.

    ``docs/agent-reference/`` is excluded on purpose: the 2026-10-01 docs move
    ("docs: move agent reference material out of startup instructions" / "docs:
    reserve nested instruction budget for module rules") relocated AGENTS.md's
    own working-notes content under ``docs/``, but that material is AGENT
    reference, not the user-taught surface this spike measures (prose claims
    and python fences aimed at a human reader of the shipped docs). Counting
    it would inflate the baseline with a directory move that taught nothing
    new to a user, not a genuine documentation change.
    """

    files = sorted(
        path
        for path in (REPO_ROOT / "docs").rglob("*.md")
        if "__pycache__" not in path.parts and "agent-reference" not in path.parts
    )
    return (REPO_ROOT / "README.md", *files)


def harvest_counts() -> dict[str, int]:
    """The spike's emitted counts (documents, python fences, claim strings)."""

    fence_count = 0
    claim_count = 0
    files = _doc_files()
    for path in files:
        text = path.read_text(encoding="utf-8")
        fence_count += len(_FENCE_PATTERN.findall(text))
        claim_count += len(_CLAIM_PATTERN.findall(text))
    return {
        "harvest_doc_files": len(files),
        "harvest_python_fences": fence_count,
        "harvest_claim_strings": claim_count,
    }


def _baseline_counts() -> dict[str, int]:
    rows: dict[str, int] = {}
    for line in (DATA_DIR / "harvest_counts.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("metric\t"):
            continue
        metric, _, count = line.partition("\t")
        rows[metric.strip()] = int(count.strip())
    return rows


def test_harvest_counts_print_and_lockstep() -> None:
    """The harvested counts match the committed baseline exactly.

    Teaching: a docs change that adds/removes python fences or support-claim
    strings updates ``data/harvest_counts.tsv`` in the same change -- the
    diff is the taught-surface change review, and the wave-A sweeps consume
    these counts as their denominators.
    """

    live = harvest_counts()
    baseline = _baseline_counts()
    print("\nharvest spike counts:")
    for metric, count in live.items():
        print(f"  {metric}\t{count}\t(baseline {baseline.get(metric)})")
    assert live == baseline, (
        "harvest counts drifted (docs taught surface changed). Review the delta, "
        "then update tests/composition_expectations/data/harvest_counts.tsv in "
        f"this change. live={live} baseline={baseline}"
    )


def test_harvest_is_feasible() -> None:
    """The spike's verdict: the taught surface IS harvestable (non-trivial counts)."""

    counts = harvest_counts()
    assert counts["harvest_doc_files"] > 10, "docs corpus vanished"
    assert counts["harvest_python_fences"] > 50, (
        "python-fence harvest collapsed; if the docs moved, re-point _doc_files() -- "
        "the taught-path universe depends on this feed"
    )
    assert counts["harvest_claim_strings"] > 10, "claim-string harvest collapsed"
