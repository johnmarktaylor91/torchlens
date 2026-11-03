"""Coverage-floor governance: the tripwire floors must be LOCKED, not YAML literals.

R72 (b9-fable/opus, round 3+4): the nightly coverage floor existed ONLY as a
workflow literal with a "never lower it (tripwire)" comment; no test pinned it,
so deleting the job or editing the floor down to 1 kept every gate green.

Two gates, two tiers (Lead 3 ruling, 2026-11-03; the tests.5 shape applied to the
aggregate floor):

1. the nightly ``coverage`` job instruments the SMOKE tier and enforces the
   per-package floors of ``scripts/check_package_coverage_floors.py`` on it; it
   never applies the aggregate floor (smoke is a 250 to 400 test sample that
   leaves unfloored packages out);
2. the aggregate floor (``COVERAGE_FLOOR_BASELINE``, never lowered) is enforced by
   ``coverage-aggregate`` on the combined data of ``coverage-mid``, which
   instruments the MID tier (``not rare and not slow and not heavy``) in shards;
3. ``[tool.coverage.report] fail_under`` carries the same authority in pyproject
   so a local ``coverage report`` reaches the same verdict as CI (R72 sol).

pyproject parsing is line/regex-based on purpose: ``tomllib`` only exists on
3.11+ and the suite still runs a 3.10 leg (the test_order_isolation_infra
precedent). The workflow side parses real YAML (PyYAML is a declared test dep)
so the assertions bind to the actual job commands instead of whole-file
substrings (r7 R72).
"""

from __future__ import annotations

import copy
import re
import shlex
from pathlib import Path
from typing import Any

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Committed coverage-floor baseline (percent). NEVER lower this to make a
#: gate pass — that is the exact silent-disarm this file exists to prevent.
#: Raise it as measured coverage grows (measured 64% on 2026-08-15; the 55
#: floor's 9-point slack is a known, deliberately conservative first gate).
COVERAGE_FLOOR_BASELINE = 55

#: The mid tier's marker clauses (the nightly mid tier, tests.5's shape).
MID_TIER_CLAUSES = frozenset({"not rare", "not slow", "not heavy"})

#: Jobs: the smoke per-package gate, the mid-tier shards, the aggregate floor.
SMOKE_JOB = "coverage"
MID_JOB = "coverage-mid"
AGGREGATE_JOB = "coverage-aggregate"


def _nightly_text() -> str:
    """Return the nightly workflow text, or skip when the tree ships none.

    Returns
    -------
    str
        Raw text of ``.github/workflows/nightly.yml``.
    """

    path = _PROJECT_ROOT / ".github" / "workflows" / "nightly.yml"
    if not path.exists():
        pytest.skip("no nightly workflow in this tree")
    return path.read_text(encoding="utf-8")


def _nightly_workflow() -> dict[str, Any]:
    """Return the parsed nightly workflow."""

    import yaml

    return yaml.safe_load(_nightly_text())


def _instrumented_commands(job: dict[str, Any]) -> list[str]:
    """Return the job's pytest commands that run under coverage."""

    return [
        step.get("run", "")
        for step in job.get("steps", [])
        if "pytest" in step.get("run", "") and "--cov" in step.get("run", "")
    ]


def _marker_clauses(command: str) -> frozenset[str]:
    """Return the ``-m`` expression's ``and`` clauses (empty when none is given)."""

    words = shlex.split(command.replace("\\\n", " "))
    for index, word in enumerate(words[:-1]):
        if word == "-m":
            return frozenset(c.strip() for c in words[index + 1].split(" and ") if c.strip())
    return frozenset()


def _fail_unders(text: str, flag: str) -> list[int]:
    """Return every ``FLAG=N`` value in ``text``."""

    return [int(value) for value in re.findall(rf"{re.escape(flag)}=(\d+)", text)]


def _coverage_job_command() -> str:
    """Return the nightly smoke ``coverage`` job's pytest command, job-scoped.

    r7 R72 (sol MED): the old checks searched the WHOLE workflow text for
    ``--cov=torchlens`` / floor literals, so an emptied coverage job plus
    those strings anywhere else satisfied every predicate. Parse the actual
    job so the assertions bind to the command that runs.
    """

    job = _nightly_workflow().get("jobs", {}).get(SMOKE_JOB)
    assert job is not None, (
        "the nightly workflow no longer declares the `coverage` job, the smoke-tier "
        "per-package coverage gate (R72/SF-19). Restore it; do not delete a coverage tripwire."
    )
    commands = _instrumented_commands(job)
    assert len(commands) == 1, (
        f"expected exactly ONE instrumented pytest command in jobs.coverage, got {len(commands)}"
    )
    return commands[0]


def aggregate_floor_problems(workflow: dict[str, Any]) -> list[str]:
    """Return every way the workflow stops enforcing the aggregate floor on the mid tier.

    Parameters
    ----------
    workflow:
        The parsed nightly workflow.

    Returns
    -------
    list[str]
        One line per problem; empty when the floor is enforced as ruled.
    """

    jobs = workflow.get("jobs", {})
    mid, aggregate = jobs.get(MID_JOB), jobs.get(AGGREGATE_JOB)
    if mid is None or aggregate is None:
        return [f"job {name} is missing" for name in (MID_JOB, AGGREGATE_JOB) if name not in jobs]
    problems: list[str] = []
    commands = _instrumented_commands(mid)
    shards = mid.get("strategy", {}).get("matrix", {}).get("shard", [])
    if not shards or shards != list(range(1, len(shards) + 1)):
        problems.append(f"{MID_JOB} matrix shards are not 1..N: {shards}")
    halves = set()
    for command in commands:
        clauses = _marker_clauses(command)
        halves.add("serial" in clauses)
        if clauses - {"not serial", "serial"} != MID_TIER_CLAUSES:
            problems.append(f"{MID_JOB} selects {sorted(clauses)}, not the mid tier")
        if "--cov=torchlens" not in command or "--cov-branch" not in command:
            problems.append(f"{MID_JOB} no longer instruments torchlens under branch coverage")
        if "--deselect" in command:
            problems.append(f"{MID_JOB} deselects tests by name")
        if any(floor != 0 for floor in _fail_unders(command, "--cov-fail-under")):
            problems.append(f"{MID_JOB} applies a floor to a shard, which holds part of the tier")
        if "serial" not in clauses and f'/{len(shards)}"' not in command:
            problems.append(f"{MID_JOB} does not shard by --tier-shard INDEX/{len(shards)}")
    if halves != {False, True}:
        problems.append(f"{MID_JOB} lost its xdist or its serial half of the mid tier")
    if MID_JOB not in aggregate.get("needs", []):
        problems.append(f"{AGGREGATE_JOB} no longer waits for {MID_JOB}")
    runs = "\n".join(step.get("run", "") for step in aggregate.get("steps", []))
    if "coverage combine" not in runs or f"-eq {len(shards) + 1}" not in runs:
        problems.append(f"{AGGREGATE_JOB} does not combine every shard's data file")
    floors = [
        int(value) for value in re.findall(r"coverage report\b[^\n]*--fail-under=(\d+)", runs)
    ]
    if not floors:
        problems.append(f"{AGGREGATE_JOB} lost its `coverage report --fail-under` floor")
    problems.extend(
        f"aggregate floor {floor} sits below the committed baseline {COVERAGE_FLOOR_BASELINE}"
        for floor in floors
        if floor < COVERAGE_FLOOR_BASELINE
    )
    return problems


def test_nightly_coverage_job_exists_and_instruments_the_smoke_tier() -> None:
    """Deleting or hollowing the nightly smoke coverage job must go red here."""

    command = _coverage_job_command()
    assert _marker_clauses(command) == {"smoke"}, (
        "the coverage job no longer selects the smoke tier"
    )
    assert "--cov=torchlens" in command and "--cov-branch" in command, (
        "the nightly coverage job no longer instruments torchlens under branch "
        "coverage — the floor would be measuring nothing"
    )
    # The instrumented leg must not deselect budget tests by name: duration
    # budgets are a repository check over tests/tiers/durations.csv, and a
    # coverage-instrumented measurement is judged at the slowdown factor (a
    # renamed test id once turned a deselect into a silent no-op while
    # claiming the exemption).
    assert "--deselect" not in command, (
        "the coverage job deselects tests by name again — instrumented runs are "
        "allowed for in scripts/tier_durations.py (MAX_SLOWDOWN_FACTOR), where a "
        "rename cannot silently void it"
    )
    from scripts.tier_durations import MAX_SLOWDOWN_FACTOR, budget_factor

    assert budget_factor({"instrumented": True, "workers": 1}) == MAX_SLOWDOWN_FACTOR, (
        "scripts/tier_durations.py lost the instrumented-measurement allowance that "
        "replaced the coverage job's brittle --deselect"
    )


def test_smoke_tier_run_never_applies_the_aggregate_floor() -> None:
    """The smoke run overrides pyproject's fail_under with 0, never the aggregate floor.

    pytest-cov inherits ``[tool.coverage.report] fail_under`` when no
    ``--cov-fail-under`` is given, so the override must be explicit.
    """

    floors = _fail_unders(_coverage_job_command(), "--cov-fail-under")
    assert floors == [0], (
        f"the smoke coverage run applies --cov-fail-under {floors}: the aggregate floor "
        "is measured on the mid tier (coverage-aggregate), never on the smoke sample; "
        "keep --cov-fail-under=0 so pytest-cov does not inherit pyproject's fail_under"
    )


def test_nightly_aggregate_floor_is_enforced_on_the_mid_tier_and_never_lowered() -> None:
    """The aggregate floor runs on the combined mid-tier shards at >= the baseline."""

    problems = aggregate_floor_problems(_nightly_workflow())
    assert not problems, (
        "the nightly aggregate coverage floor is no longer enforced on the mid tier: "
        f"{problems}. The floor is a tripwire: never lower, skip or move it to pass "
        "(raising COVERAGE_FLOOR_BASELINE alongside a real coverage gain is the only "
        "sanctioned edit)."
    )


def _lower_the_floor(workflow: dict[str, Any]) -> None:
    for step in workflow["jobs"][AGGREGATE_JOB]["steps"]:
        if "run" in step:
            step["run"] = step["run"].replace("--fail-under=55", "--fail-under=12")


def _remove_the_aggregate_job(workflow: dict[str, Any]) -> None:
    del workflow["jobs"][AGGREGATE_JOB]


def _remove_the_mid_job(workflow: dict[str, Any]) -> None:
    del workflow["jobs"][MID_JOB]


def _select_smoke(workflow: dict[str, Any]) -> None:
    for step in workflow["jobs"][MID_JOB]["steps"]:
        if "run" in step:
            step["run"] = re.sub(r'-m "[^"]*"', "-m smoke", step["run"])


@pytest.mark.parametrize(
    "mutate",
    [_lower_the_floor, _remove_the_aggregate_job, _remove_the_mid_job, _select_smoke],
    ids=["lowered", "aggregate-job-removed", "mid-job-removed", "smoke-selector"],
)
def test_aggregate_floor_lock_is_red_capable(mutate) -> None:
    """Lowering the value, removing a job or selecting smoke each goes red."""

    workflow = copy.deepcopy(_nightly_workflow())
    assert not aggregate_floor_problems(workflow)
    mutate(workflow)
    assert aggregate_floor_problems(workflow)


def test_floor_lock_is_red_capable() -> None:
    """The floor extraction flags a lowered literal (red-capability self-test)."""

    planted = "run: |\n  coverage report --fail-under=12"
    floors = _fail_unders(planted, "--fail-under")
    assert floors == [12]
    assert [floor for floor in floors if floor < COVERAGE_FLOOR_BASELINE] == [12]


def test_pyproject_carries_the_same_coverage_floor() -> None:
    """[tool.coverage.report] fail_under must match the committed baseline.

    Keeps local ``coverage report`` verdicts aligned with CI's flag-passed
    floor: the flag overrides config, so the two can only disagree when the
    config is silently missing — which is exactly how the floor stayed
    unlocked for four review passes.
    """

    pyproject = (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"^fail_under\s*=\s*(\d+)", pyproject, flags=re.MULTILINE)
    assert match, (
        "[tool.coverage.report] lost its fail_under — local coverage runs no "
        f"longer enforce any floor. Restore fail_under = {COVERAGE_FLOOR_BASELINE}."
    )
    assert int(match.group(1)) >= COVERAGE_FLOOR_BASELINE, (
        f"pyproject fail_under = {match.group(1)} sits below the committed "
        f"baseline {COVERAGE_FLOOR_BASELINE} — the floor is a tripwire, never "
        "lower it to pass"
    )


#: Shrink-forbidden per-package floor baselines (r7 R72). The script is the
#: single runtime authority; this mirror refuses a silent floor cut there.
PACKAGE_FLOOR_BASELINES: dict[str, float] = {
    "torchlens/_capture": 74.0,
    "torchlens/_postprocess": 80.0,
    "torchlens/validation": 54.0,
    "torchlens/backends": 48.0,
    "torchlens/_data_classes": 65.0,
    "torchlens/_io": 63.0,
    "torchlens/intervention": 59.0,
    "torchlens/_utils": 59.0,
    "torchlens/_merged": 80.0,
    "torchlens/_trace_core": 80.0,
    "torchlens/_visualization": 57.0,
}


def _load_floor_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_pkg_floor_script", _PROJECT_ROOT / "scripts" / "check_package_coverage_floors.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_per_package_floors_are_enforced_and_never_lowered() -> None:
    """r7 R72 (sol MED): verdict-critical packages are bound individually.

    The nightly coverage job must invoke the floor script on the json its
    smoke-tier run just produced, and the script's floors may rise but never
    drop below the committed baselines (the aggregate-floor doctrine, per
    package). The floors are measured on the smoke tier; only the aggregate
    floor moved to the mid tier.
    """

    assert _marker_clauses(_coverage_job_command()) == {"smoke"}, (
        "the per-package floors are measured on the smoke tier; the coverage job "
        "no longer selects it"
    )
    job = _nightly_workflow()["jobs"][SMOKE_JOB]
    runs = "\n".join(step.get("run", "") for step in job.get("steps", []))
    assert "scripts/check_package_coverage_floors.py" in runs, (
        "the nightly coverage job no longer enforces per-package floors"
    )
    assert "--cov-report=json" in runs, (
        "the coverage job stopped producing the json the floor script reads"
    )
    script = _load_floor_script()
    assert set(script.PACKAGE_FLOORS) >= set(PACKAGE_FLOOR_BASELINES), (
        "per-package floor row(s) deleted from the script: "
        f"{sorted(set(PACKAGE_FLOOR_BASELINES) - set(script.PACKAGE_FLOORS))}"
    )
    lowered = {
        prefix: (script.PACKAGE_FLOORS[prefix], baseline)
        for prefix, baseline in PACKAGE_FLOOR_BASELINES.items()
        if script.PACKAGE_FLOORS.get(prefix, 0) < baseline
    }
    assert not lowered, (
        f"per-package floors lowered below their committed baselines: {lowered} "
        "— the floor is a tripwire, never lower it to pass"
    )


def test_every_package_floor_key_names_an_existing_package() -> None:
    """A floor keyed by a moved package's old path measures nothing.

    Round-1 package moves (``postprocess`` to ``_postprocess`` and three more)
    left the floors keyed by the old paths: the script then finds no measured
    file for the key, so the floor no longer guards the package it names.
    Every key in the script and in the baseline mirror must be a real package
    directory holding Python source.
    """

    script = _load_floor_script()
    keys = sorted(set(script.PACKAGE_FLOORS) | set(PACKAGE_FLOOR_BASELINES))
    missing = [
        key
        for key in keys
        if not (_PROJECT_ROOT / key).is_dir() or not any((_PROJECT_ROOT / key).rglob("*.py"))
    ]
    assert not missing, (
        f"per-package coverage floor key(s) name no package directory: {missing}. "
        "Re-key them to the package's current path with the SAME floor value "
        "(script and the PACKAGE_FLOOR_BASELINES mirror together); never drop or lower one."
    )


def test_package_percentage_aggregation_is_red_capable() -> None:
    """The aggregation flags a hollowed package (unit red-capability)."""

    script = _load_floor_script()
    payload = {
        "files": {
            "torchlens/_capture/trace.py": {
                "summary": {
                    "covered_lines": 10,
                    "num_statements": 100,
                    "covered_branches": 0,
                    "num_branches": 0,
                }
            }
        }
    }
    measured = script.package_percentages(payload)
    assert measured["torchlens/_capture"] == 10.0
    assert measured["torchlens/_capture"] < script.PACKAGE_FLOORS["torchlens/_capture"]


#: No-growth ceiling on `# pragma: no cover` sites in torchlens/ (r7 R72-F2:
#: this was the ONE exemption channel with no bound while every sibling
#: mechanism — ruff ceilings, blanket-noqa ceiling, grow-only mypy flags,
#: docstring ledger, mutation-deselect expiry, size ledgers — carries one).
#: Excluded lines leave the coverage denominator entirely, so unbounded
#: growth is invisible to every floor above. Measured 2026-08-16: 57 sites,
#: all guard-line scoped with inline reasons. SHRINK-ONLY. Re-measured
#: 2026-08-16 (wave-0 governance sweep): 50 — the nine redundant
#: ``if TYPE_CHECKING:`` pragmas were deleted (the 193-site majority
#: convention carries no pragma; the guard line itself executes at import),
#: and the one REACHABLE pragma'd raise (_runnable_transaction until=
#: substring token) lost its pragma and gained a covering test. The one
#: surviving wave-0 addition is the _encoding.py import-time two-row
#: classification guard, unreachable while the completeness pin holds.
#: 51 -> 57 (2026-08-28 F27): six uncoverable-on-CPU-CI seams -- the two
#: kernel_profile CUDA legs (device activities + synchronize), the wrappers/
#: backward observability-import failure belts (the substrate always imports
#: in CI; the belt exists for torn installs), and the no-loadavg platform
#: branch in the overhead harness, plus the compute-truth join lookup's
#: torn-install ImportError belt. Each is a disclosure/degradation path,
#: never a semantics branch.
#: 50 -> 51 (2026-08-19 post-tour sprint): debug/_params.py's both-entries-None
#: guard in the param-diff walk. UNREACHABLE by construction -- ``names`` is the
#: UNION of both state keysets, so a name cannot be absent from both -- which is
#: also why it cannot gain a covering test the way the wave-0 reachable raise did.
#: Raised rather than deleted: it is another module's defensive guard, and trading
#: a guard for a lint point is the worse side of that bargain.
#: 57 -> 60 (2026-08-28 F29 agent surface, re-tiered on the T80 merged tree):
#: (a) agent/_envelope.py's PackageNotFoundError fallback fires only when the
#: torchlens distribution metadata is absent (a non-installed source tree; no
#: test environment can honestly provoke it without uninstalling itself);
#: (b) bridge/mcp.py's resource-registration AttributeError guard fires only
#: on an older mcp package surface than the pinned extra installs; (c)
#: agent/_artifacts.py's spec_from_file_location None guard fires only on a
#: corrupted install (missing _io/_json.py file). All three are environment
#: guards, not logic branches.
# 60 -> 61 (2026-09-01 FLAKEHUNT rng-channels split): the split module carries a
# typing-only TYPE_CHECKING import guard and rng.py a non-POSIX ImportError
# guard -- both structurally unreachable in the Linux CI environment, both
# inline-justified. Conscious raise per this test's own remedy.
_PRAGMA_NO_COVER_CEILING = 61


def test_pragma_no_cover_census_never_grows() -> None:
    """The coverage-exemption channel stays bounded and never excludes a body."""

    package_root = _PROJECT_ROOT / "torchlens"
    sites: list[str] = []
    body_exclusions: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        rel = path.relative_to(_PROJECT_ROOT).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "pragma: no cover" not in line:
                continue
            sites.append(f"{rel}:{lineno}")
            stripped = line.lstrip()
            if stripped.startswith(("def ", "class ", "async def ")):
                body_exclusions.append(f"{rel}:{lineno}")
    assert not body_exclusions, (
        f"pragma: no cover on a def/class line excludes a WHOLE BODY from the "
        f"denominator — scope it to the guard line instead: {body_exclusions}"
    )
    assert len(sites) <= _PRAGMA_NO_COVER_CEILING, (
        f"pragma: no cover sites grew to {len(sites)} (ceiling "
        f"{_PRAGMA_NO_COVER_CEILING}): coverage exemptions leave the "
        "denominator invisibly; justify the new site and raise the ceiling "
        "in the same reviewed change, or cover the path"
    )
