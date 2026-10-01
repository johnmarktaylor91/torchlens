"""Coverage-floor governance: the tripwire floor must be LOCKED, not a YAML literal.

R72 (b9-fable/opus, round 3+4): the nightly coverage job's ``--cov-fail-under``
floor existed ONLY as a workflow literal with a "never lower it (tripwire)"
comment — no test pinned it, so deleting the job or editing the floor down to 1
kept every gate green. These checks make the claim mechanical:

1. the nightly ``coverage`` job EXISTS and instruments the smoke tier;
2. its ``--cov-fail-under`` value never drops below the committed baseline
   (shrink-forbidden ratchet: raising it is welcome, lowering goes red here);
3. ``[tool.coverage.report] fail_under`` carries the same authority in
   pyproject so a local ``coverage report`` reaches the same verdict as CI
   (R72 sol: without it, local and CI coverage verdicts silently differed).

pyproject parsing is line/regex-based on purpose: ``tomllib`` only exists on
3.11+ and the suite still runs a 3.10 leg (the test_order_isolation_infra
precedent). The workflow side parses real YAML (PyYAML is a declared test
dep) so the assertions bind to the actual ``jobs.coverage`` command instead
of whole-file substrings (r7 R72).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Committed coverage-floor baseline (percent). NEVER lower this to make a
#: gate pass — that is the exact silent-disarm this file exists to prevent.
#: Raise it as measured coverage grows (measured 64% on 2026-08-15; the 55
#: floor's 9-point slack is a known, deliberately conservative first gate).
COVERAGE_FLOOR_BASELINE = 55


def _nightly_text() -> str:
    """Return the nightly workflow text, or skip when the tree ships none.

    Returns
    -------
    str
        Raw text of ``.github/workflows/nightly.yml``.
    """

    path = _PROJECT_ROOT / ".github" / "workflows" / "nightly.yml"
    if not path.exists():
        pytest.skip("no nightly workflow in this tree (sdist/test-only layout)")
    return path.read_text(encoding="utf-8")


def _coverage_job_command() -> str:
    """Return the nightly ``coverage`` job's pytest command, job-scoped.

    r7 R72 (sol MED): the old checks searched the WHOLE workflow text for
    ``--cov=torchlens`` / floor literals, so an emptied coverage job plus
    those strings anywhere else satisfied every predicate. Parse the actual
    job so the assertions bind to the command that runs.
    """

    import yaml

    workflow = yaml.safe_load(_nightly_text())
    job = workflow.get("jobs", {}).get("coverage")
    assert job is not None, (
        "the nightly workflow no longer declares the `coverage` job — the only "
        "measured coverage gate in the repo (R72/SF-19). Restore it; do not "
        "delete the sole coverage tripwire."
    )
    commands = [
        step.get("run", "")
        for step in job.get("steps", [])
        if "pytest" in step.get("run", "") and "--cov" in step.get("run", "")
    ]
    assert len(commands) == 1, (
        f"expected exactly ONE instrumented pytest command in jobs.coverage, got {len(commands)}"
    )
    return commands[0]


def test_nightly_coverage_job_exists_and_instruments_the_smoke_tier() -> None:
    """Deleting or hollowing the nightly coverage job must go red here."""

    command = _coverage_job_command()
    assert "-m smoke" in command, "the coverage job no longer selects the smoke tier"
    assert "--cov=torchlens" in command and "--cov-branch" in command, (
        "the nightly coverage job no longer instruments torchlens under branch "
        "coverage — the floor would be measuring nothing"
    )
    # The instrumented leg must not deselect budget tests by name: the
    # carve-out lives in the budget machinery itself (a renamed test id once
    # turned the deselect into a silent no-op while claiming the exemption).
    assert "--deselect" not in command, (
        "the coverage job deselects tests by name again — the instrumentation "
        "carve-out belongs in tests/conftest.py (cov_source check), where a "
        "rename cannot silently void it"
    )
    conftest_text = (_PROJECT_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert 'getattr(item.config.option, "cov_source", None)' in conftest_text, (
        "tests/conftest.py lost the instrumented-session budget carve-out that "
        "replaced the coverage job's brittle --deselect"
    )


def test_nightly_coverage_floor_is_never_lowered() -> None:
    """The job's --cov-fail-under may rise but never drop below baseline."""

    command = _coverage_job_command()
    floors = [int(value) for value in re.findall(r"--cov-fail-under=(\d+)", command)]
    assert floors, (
        "the nightly coverage job lost its --cov-fail-under floor entirely; "
        f"restore at least --cov-fail-under={COVERAGE_FLOOR_BASELINE}"
    )
    lowered = [floor for floor in floors if floor < COVERAGE_FLOOR_BASELINE]
    assert not lowered, (
        f"nightly --cov-fail-under {lowered} sits below the committed baseline "
        f"{COVERAGE_FLOOR_BASELINE}. The floor is a tripwire: never lower it to "
        "pass — root-cause the coverage loss instead. (Raising the baseline "
        "constant in this file alongside a real coverage gain is the only "
        "sanctioned edit.)"
    )


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


def test_floor_lock_is_red_capable() -> None:
    """The floor extraction flags a lowered literal (red-capability self-test)."""

    planted = "run: |\n  pytest tests/ --cov=torchlens --cov-fail-under=12 --tb=short"
    floors = [int(value) for value in re.findall(r"--cov-fail-under=(\d+)", planted)]
    assert floors == [12]
    assert [floor for floor in floors if floor < COVERAGE_FLOOR_BASELINE] == [12]


#: Shrink-forbidden per-package floor baselines (r7 R72). The script is the
#: single runtime authority; this mirror refuses a silent floor cut there.
PACKAGE_FLOOR_BASELINES: dict[str, float] = {
    "torchlens/capture": 74.0,
    "torchlens/postprocess": 80.0,
    "torchlens/validation": 54.0,
    "torchlens/backends": 48.0,
    "torchlens/data_classes": 65.0,
    "torchlens/_io": 63.0,
    "torchlens/intervention": 59.0,
    "torchlens/utils": 59.0,
    "torchlens/merged": 80.0,
    "torchlens/_trace_core": 80.0,
    "torchlens/visualization": 57.0,
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

    The nightly coverage job must invoke the floor script on the json it
    just produced, and the script's floors may rise but never drop below the
    committed baselines (the aggregate-floor doctrine, per package).
    """

    import yaml

    workflow = yaml.safe_load(_nightly_text())
    job = workflow.get("jobs", {}).get("coverage")
    assert job is not None
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


def test_package_percentage_aggregation_is_red_capable() -> None:
    """The aggregation flags a hollowed package (unit red-capability)."""

    script = _load_floor_script()
    payload = {
        "files": {
            "torchlens/capture/trace.py": {
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
    assert measured["torchlens/capture"] == 10.0
    assert measured["torchlens/capture"] < script.PACKAGE_FLOORS["torchlens/capture"]


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
#: native_profile CUDA legs (device activities + synchronize), the wrappers/
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
_PRAGMA_NO_COVER_CEILING = 60


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
