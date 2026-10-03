"""Memory-timeline parity oracle + frozen torch-2.13 golden (W0.7, DEADLINE).

Torch's still-alive PRIVATE categorizer (``prof._memory_profile()``) is the
one oracle the rebuilt categorized timeline (observe, W3.2) can ever be
checked against; after upstream deletes it the oracle can never be built
again. This suite is the test-oracle exception to the no-private-APIs rule:
it feature-detects through the ``HAS_MEMORY_PROFILE`` compat flag, SKIPS
WITH THAT NAMED REASON on absence, and freezes the categorizer's behavior
on a pinned deterministic scenario as a torch-2.13 golden fixture
(``tests/data/torchnative/memory_parity_torch213.json``).

The 6.3 category contract is enforced besides: every torch category the
scenario produces is either EMITTED (1:1 observed record fact) or
NEVER-EMITTED (heuristic-only) -- an unmapped category is the review
trigger, red until the migration table learns it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from torchlens.observability._memory_parity import (
    EMITTED_CATEGORIES,
    NEVER_EMITTED_CATEGORIES,
    categorized_key_counts,
    category_vocabulary_report,
    pinned_parity_scenario,
)

_GOLDEN_PATH = Path(__file__).parent / "data" / "torchnative" / "memory_parity_torch213.json"
_GOLDEN = json.loads(_GOLDEN_PATH.read_text())


@pytest.fixture(scope="module")
def parity_counts() -> dict[str, int]:
    """Run the pinned scenario once; skip with the named capability reason."""

    profiler = pinned_parity_scenario()
    counts = categorized_key_counts(profiler)
    if counts is None:
        pytest.skip(
            "HAS_MEMORY_PROFILE is False: torch removed the private "
            "categorizer; the frozen torch-2.13 golden fixture is the "
            "remaining authority (torchnative W0.7)"
        )
    return counts


@pytest.mark.heavy
def test_parity_against_frozen_golden(parity_counts: dict[str, int]) -> None:
    """On the golden's torch minor, the categorizer must match it exactly.

    On other torch versions the exact counts may legitimately drift; the
    vocabulary test below still runs. A mismatch ON 2.13 means either the
    pinned scenario changed (regenerate consciously, in this lane's owner
    review) or torch patched the categorizer under the same minor.
    """

    golden_minor = ".".join(str(_GOLDEN["torch_version"]).split(".")[:2])
    runtime_minor = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    if runtime_minor != golden_minor:
        pytest.skip(
            f"golden frozen on torch {golden_minor}, runtime is {runtime_minor}; "
            "exact-count parity applies only on the frozen minor"
        )
    assert parity_counts == _GOLDEN["category_counts"]


@pytest.mark.heavy
def test_category_vocabulary_is_fully_mapped(parity_counts: dict[str, int]) -> None:
    """Every observed torch category is contractually mapped (6.3).

    ``unmapped`` non-empty = torch grew a category the migration table does
    not know -- the review trigger fires as a red test, never silently.
    """

    report = category_vocabulary_report(parity_counts)
    assert not report["unmapped"], (
        f"torch categorizer produced unmapped categories {report['unmapped']}; "
        "extend the 6.3 migration contract consciously"
    )
    # The pinned scenario exercises both contract sides.
    assert set(report["emitted"]) >= {"PARAMETER", "GRADIENT", "INPUT", "ACTIVATION"}
    assert set(report["never_emitted"]) == set(NEVER_EMITTED_CATEGORIES)


def test_contract_tables_are_disjoint_and_documented() -> None:
    """The emitted / never-emitted tables partition cleanly with reasons."""

    assert not set(EMITTED_CATEGORIES) & set(NEVER_EMITTED_CATEGORIES)
    for reason in {**EMITTED_CATEGORIES, **NEVER_EMITTED_CATEGORIES}.values():
        assert reason.strip()


def test_golden_fixture_shape() -> None:
    """The frozen fixture stays self-describing and version-stamped."""

    assert _GOLDEN["schema"] == "torchlens.memory_parity_golden.v1"
    assert _GOLDEN["torch_version"].startswith("2.13")
    assert _GOLDEN["category_counts"]
    assert all(
        isinstance(count, int) and count > 0 for count in _GOLDEN["category_counts"].values()
    )
