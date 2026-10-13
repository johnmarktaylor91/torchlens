"""Named scale profiles and their budgets (U40.3; oracles memo D23/H6).

A scale profile re-runs quantity oracles at a size where scale-only
wrongness could show. Each profile names its discriminating property, the
venue it runs at, and its own budget: CPU-minutes per run, wall seconds and
resident-memory growth. The budgets sit under two global ceilings so an expensive profile
cannot quietly turn itself off by outgrowing its leg:

- ``NIGHTLY_CEILING_CPU_MIN``: the nightly profiles together, per night.
- ``WEEKLY_CEILING_CPU_MIN``: the whole scale load of one week, which is
  the weekly profiles plus seven runs of every nightly profile.

``tests/real_model/test_scale_profiles.py`` holds the ceilings, checks that
every profile is fully named, and runs the built profiles under their own
budgets. An unbuilt profile keeps its budget reserved and says so.
"""

from __future__ import annotations

from dataclasses import dataclass

NIGHTLY_CEILING_CPU_MIN = 10.0
WEEKLY_CEILING_CPU_MIN = 30.0
NIGHTS_PER_WEEK = 7
VENUES = ("nightly", "weekly")
STATUSES = ("built", "unbuilt")


@dataclass(frozen=True)
class ScaleProfile:
    """One named scale profile.

    Parameters
    ----------
    profile_id:
        Stable id (``SP<n>-<slug>``), the ``scale_profile_id`` an oracle row
        names.
    discriminating_property:
        What this size exposes that the small fixtures cannot.
    venue:
        ``nightly`` or ``weekly``.
    cpu_minutes:
        Budgeted CPU-minutes (user + system) for one run of the profile.
    wall_budget_s:
        Budgeted wall-clock seconds for one run.
    rss_budget_mib:
        Budgeted resident-memory growth of the run while its capture is held,
        in MiB (a pytest worker's lifetime peak belongs to earlier tests).
    owner:
        Who keeps the profile running; ``unassigned`` is stated, never blank.
    test_node:
        The pytest node that runs the profile; empty while ``unbuilt``.
    status:
        ``built`` (the node exists and enforces this budget) or ``unbuilt``.
    """

    profile_id: str
    discriminating_property: str
    venue: str
    cpu_minutes: float
    wall_budget_s: float
    rss_budget_mib: int
    owner: str
    test_node: str
    status: str


SCALE_PROFILES: tuple[ScaleProfile, ...] = (
    ScaleProfile(
        profile_id="SP1-wide-output-linear",
        discriminating_property=(
            "a Linear(8, 50000) head: an output wide enough that per-element"
            " alias and byte double-counting (the gpt2 +31% shape) moves the"
            " parameter, byte and MAC totals far past rounding, at near-zero"
            " model cost"
        ),
        venue="nightly",
        cpu_minutes=1.0,
        wall_budget_s=60.0,
        rss_budget_mib=2048,
        owner="real-model lane",
        test_node="tests/real_model/test_scale_profiles.py::test_sp1_wide_output_linear_quantities",
        status="built",
    ),
    ScaleProfile(
        profile_id="SP2-qwen2.5-0.5b-512",
        discriminating_property=(
            "transformer aggregation and truncation disclosure at 512 tokens on"
            " the real Qwen2.5-0.5B checkpoint (registry row"
            " r2-qwen2.5-0.5b-instruct)"
        ),
        venue="weekly",
        cpu_minutes=12.0,
        wall_budget_s=600.0,
        rss_budget_mib=8192,
        owner="unassigned",
        test_node="",
        status="unbuilt",
    ),
    ScaleProfile(
        profile_id="SP3-chunked-extraction",
        discriminating_property=(
            "count, order and conservation across chunks of a real extraction"
            " under a registered save budget, against an unchunked baseline"
        ),
        venue="weekly",
        cpu_minutes=8.0,
        wall_budget_s=480.0,
        rss_budget_mib=6144,
        owner="unassigned",
        test_node="",
        status="unbuilt",
    ),
)

SCALE_PROFILES_BY_ID = {profile.profile_id: profile for profile in SCALE_PROFILES}


def nightly_cpu_minutes() -> float:
    """CPU-minutes the nightly profiles spend in one night."""

    return sum(p.cpu_minutes for p in SCALE_PROFILES if p.venue == "nightly")


def weekly_cpu_minutes() -> float:
    """CPU-minutes of one week: weekly profiles plus seven nightly runs."""

    weekly = sum(p.cpu_minutes for p in SCALE_PROFILES if p.venue == "weekly")
    return weekly + NIGHTS_PER_WEEK * nightly_cpu_minutes()
