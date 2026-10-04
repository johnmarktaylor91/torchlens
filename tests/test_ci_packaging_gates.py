"""CI / packaging governance gates (grind-p3 T13 fix lane).

Each test here pins a CI-plumbing invariant that regressed silently at least
once: pre-commit and the lint gate disagreeing on ruff order, oracle legs
skipping every golden while staying green, the byte oracles never executing
on any CI leg, unscoped release credentials, and an ungoverned sdist. These
are parse/lint assertions over the checked-in config — the runtime halves
live in the workflows themselves.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOWS = _PROJECT_ROOT / ".github" / "workflows"


def _load_yaml(path: Path) -> dict:
    """Parse one YAML config file."""

    return yaml.safe_load(path.read_text())


def _smoke_job() -> dict:
    """Return the smoke job of the Tests workflow."""

    return _load_yaml(_WORKFLOWS / "tests.yml")["jobs"]["smoke"]


def test_exactly_one_smoke_row_enforces_the_byte_oracle_goldens() -> None:
    """One smoke row declares oracle enforcement and matches the ENV markers.

    The byte-oracle goldens enforce only where the environment fingerprint
    matches the committed ``ENV`` marker; every other CI leg legitimately
    skips them. Without a declared enforcing row, a matrix torch bump moves
    that one leg off-canonical and the golden families enforce on NO leg at
    all while every row stays green (T13.1). This pins the lockstep between
    the matrix row and the markers, so bumping either alone goes red.
    """

    rows = [
        row for row in _smoke_job()["strategy"]["matrix"]["include"] if row.get("scope") == "smoke"
    ]
    assert len(rows) >= 6, "the PR-blocking smoke matrix shrank unexpectedly"
    enforcing = [row for row in rows if str(row.get("oracle_enforce", "")) == "1"]
    assert len(enforcing) == 1, (
        "exactly one smoke row must declare oracle_enforce so the byte-oracle "
        "goldens are guaranteed to enforce on one CI leg"
    )
    row = enforcing[0]
    row_fingerprint = f"py{row['python']}-torch{str(row['torch']).split('+', 1)[0]}"
    for goldens_dir in ("surface_oracle", "godobject_oracle"):
        marker = (_PROJECT_ROOT / "tests" / goldens_dir / "goldens" / "ENV").read_text().strip()
        assert row_fingerprint == marker, (
            f"the enforcing smoke row ({row_fingerprint}) no longer matches the "
            f"committed {goldens_dir} ENV marker ({marker}); rebaseline the "
            "goldens deliberately or fix the matrix row — do not let them drift"
        )


def test_local_ci_smoke_script_matches_the_enforcing_row() -> None:
    """``scripts/smoke_ci_parity.py`` pins the same environment as the enforcing row.

    The local commit gate only answers like CI when it runs CI's enforcing
    environment: same python, torch, torchvision, transformers and numpy, the
    same extras and byte-emitter pins, and the same smoke-step env and floor.
    Bumping the workflow without the script (or the reverse) goes red here.
    """

    spec = importlib.util.spec_from_file_location(
        "smoke_ci_parity", _PROJECT_ROOT / "scripts" / "smoke_ci_parity.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    job = _smoke_job()
    row = next(
        row for row in job["strategy"]["matrix"]["include"] if row.get("oracle_enforce") == "1"
    )
    assert (
        str(row["python"]),
        str(row["torch"]),
        str(row["torchvision"]),
    ) == (script.PYTHON, script.TORCH, script.TORCHVISION)
    assert str(row["transformers"]) == script.TRANSFORMERS
    assert str(row["numpy"]) == script.NUMPY_SPEC

    install_step = next(step for step in job["steps"] if "uv pip install" in step.get("run", ""))
    for token in (f'"{script.EXTRAS}"', *(f'"{pin}"' for pin in script.EXTRA_PINS)):
        assert token in install_step["run"], f"CI install no longer carries {token}"

    run_step = next(step for step in job["steps"] if step.get("name", "").startswith("Run smoke"))
    assert run_step["env"]["OMP_NUM_THREADS"] == script.SMOKE_ENV["OMP_NUM_THREADS"]
    assert script.SMOKE_ENV["TORCHLENS_ORACLE_ENFORCE"] == "1"
    assert "-m smoke -n 4" in run_step["run"]
    floor_step = next(step for step in job["steps"] if "check_ci_executed_tests.py" in step["run"])
    assert f"{script.EXECUTED_FLOOR} {script.SKIP_FRACTION}" in floor_step["run"]


def test_every_smoke_row_runs_the_executed_floor_attestation() -> None:
    """All smoke rows emit junit XML and enforce an executed-test floor.

    A leg whose suite hollows out (collection drift, conftest guards, mass
    importorskip or oracle skipping) exits 0 and stays green; the floor turns
    an under-executed run into a failure. Previously only the nightly preview
    legs carried this attestation (T13.1).
    """

    steps = _smoke_job()["steps"]
    run_step = next(step for step in steps if step.get("name", "").startswith("Run smoke tests"))
    assert "--junitxml" in run_step["run"], "smoke run must emit junit XML"
    assert run_step["env"]["TORCHLENS_ORACLE_ENFORCE"] == "${{ matrix.oracle_enforce }}", (
        "the enforce declaration must reach the test process"
    )
    floor_step = next(
        (step for step in steps if "check_ci_executed_tests.py" in step.get("run", "")),
        None,
    )
    assert floor_step is not None, "smoke rows lost the executed-floor attestation"
    assert floor_step.get("if") == run_step.get("if"), (
        "the floor check must run on every row the smoke suite runs on"
    )


def test_precommit_ruff_fix_runs_before_ruff_format() -> None:
    """pre-commit applies lint fixes BEFORE formatting, matching the CI gate.

    ``ruff check --fix`` can rewrite code into an unformatted shape; running it
    AFTER ``ruff-format`` therefore commits bytes the Lint workflow's
    format-then-lint check rejects. Ruff's own pre-commit guidance orders the
    ``ruff`` (fix) hook before ``ruff-format`` so the formatter has the last
    word locally, exactly like ``ruff format --check`` has the last word in CI.
    """

    config = _load_yaml(_PROJECT_ROOT / ".pre-commit-config.yaml")
    ruff_repo = next(repo for repo in config["repos"] if "ruff-pre-commit" in repo["repo"])
    hook_ids = [hook["id"] for hook in ruff_repo["hooks"]]
    assert hook_ids.index("ruff") < hook_ids.index("ruff-format"), (
        "pre-commit must run the ruff (--fix) hook BEFORE ruff-format; the "
        "reverse order lets a lint autofix produce unformatted code that the "
        "CI `ruff format --check` gate rejects"
    )


def test_render_byte_oracle_executes_on_a_ci_leg() -> None:
    """A dedicated step actually RUNS the heavy render byte-oracle test.

    The byte half of the render-identity oracle is heavy-marked, so setting
    ``TORCHLENS_RENDER_BYTE_ORACLE`` on a row whose selection is ``-m smoke``
    was dead config: DOT-byte identity was enforced on no CI leg while the
    flag looked wired (T13.2). The dedicated step must select the heavy
    marker and attest execution through the junit floor.
    """

    smoke = _smoke_job()
    rows = [
        row
        for row in smoke["strategy"]["matrix"]["include"]
        if str(row.get("render_byte_oracle", "")) == "1"
    ]
    assert len(rows) == 1, "exactly one smoke row must carry the render byte oracle"
    step = next(
        (
            step
            for step in smoke["steps"]
            if "test_viz_render_identity_oracle.py" in step.get("run", "")
        ),
        None,
    )
    assert step is not None, "no smoke step executes the render byte-oracle test"
    assert step.get("if") == "matrix.render_byte_oracle == '1'"
    assert step["env"]["TORCHLENS_RENDER_BYTE_ORACLE"] == "1"
    assert "-m heavy" in step["run"], (
        "the byte-oracle consumer is heavy-marked; without selecting the "
        "heavy marker the step executes nothing"
    )
    assert "check_ci_executed_tests.py" in step["run"], (
        "the byte-oracle step must attest the test EXECUTED rather than skipped"
    )


def test_surface_byte_oracle_executes_on_the_enforcing_leg() -> None:
    """A dedicated step actually RUNS the heavy surface byte-oracle family.

    The surface-oracle byte tests are heavy-marked, so the enforcing row's
    ``-m smoke`` selection collects only the fact pins, and every scheduled
    leg that selects heavy runs off-canonical and skips through
    ``require_env_golden`` — byte identity enforced on NO CI leg, the T13.2
    class recurring one family over (b10 R78 round-4). The dedicated step
    must run on the enforcing row, select the heavy marker, arm enforcement
    (so an off-canonical drift is a hard failure, not a skip), and attest at
    least the six model-axis byte tests EXECUTED.
    """

    smoke = _smoke_job()
    step = next(
        (step for step in smoke["steps"] if "tests/surface_oracle/" in step.get("run", "")),
        None,
    )
    assert step is not None, "no smoke step executes the surface byte-oracle family"
    assert step.get("if") == "matrix.oracle_enforce == '1'", (
        "the surface byte family must run exactly on the ONE enforcing row"
    )
    assert step["env"]["TORCHLENS_ORACLE_ENFORCE"] == "1"
    assert "-m heavy" in step["run"], (
        "the surface byte tests are heavy-marked; without selecting the heavy "
        "marker the step executes nothing"
    )
    assert "check_ci_executed_tests.py" in step["run"], (
        "the surface byte step must attest the tests EXECUTED rather than skipped"
    )
    floor = int(step["run"].rsplit(None, 1)[-1])
    assert floor >= 6, "the executed floor must cover all six model-axis byte tests"


def test_capture_oracle_matrix_enforces_on_a_nightly_leg() -> None:
    """Nightly runs the slow capture-characterization matrix with a floor."""

    nightly = _load_yaml(_WORKFLOWS / "nightly.yml")["jobs"]
    job = nightly.get("capture-byte-oracle")
    assert job is not None, "nightly lost the capture-byte-oracle job (T13.2)"
    runs = "\n".join(step.get("run", "") for step in job["steps"])
    assert "tests/capture_oracle/" in runs and "-m slow" in runs
    assert "check_ci_executed_tests.py" in runs, (
        "the capture-oracle leg must attest executed tests: the version gate "
        "skips the whole matrix on any non-recording torch, which is exactly "
        "the silent-green this leg exists to prevent"
    )


def test_capture_oracle_version_gate_strips_the_build_tag() -> None:
    """The golden version gate compares torch SOURCE versions, not build tags.

    A ``+cu130``-recorded golden must enforce on a ``+cpu`` CI runtime of the
    same torch version; comparing full build strings made every CI leg skip
    the capture-characterization matrix forever (T13.2).
    """

    from capture_oracle.test_capture_oracle import _recording_torch_matches

    assert _recording_torch_matches("2.13.0+cu130", "2.13.0+cpu")
    assert _recording_torch_matches("2.13.0", "2.13.0+cpu")
    assert not _recording_torch_matches("2.12.0+cpu", "2.13.0+cpu")
    assert not _recording_torch_matches(None, "2.13.0+cpu")


def test_release_app_token_is_permission_scoped() -> None:
    """The minted GitHub App token carries an explicit minimal permission set.

    Without a ``permission-*`` input the token inherits the App
    installation's FULL permissions and checkout persists it to disk for the
    whole release job, including third-party pip installs (zizmor
    ``github-app`` HIGH, T13.3). Contents write is everything the job needs.
    """

    release = _load_yaml(_WORKFLOWS / "release.yml")["jobs"]["release"]
    token_step = next(
        step for step in release["steps"] if "create-github-app-token" in step.get("uses", "")
    )
    scoped = [key for key in token_step["with"] if key.startswith("permission-")]
    assert scoped == ["permission-contents"], (
        "the release App token must be minted with exactly the minimal permission-contents scope"
    )
    assert token_step["with"]["permission-contents"] == "write"


def test_release_job_bounds_the_app_token_hold() -> None:
    """The release job declares a timeout so the App token's life is bounded.

    Without ``timeout-minutes`` the job inherits GitHub's 6-hour default,
    and a hung pip resolve or PyPI upload keeps a live repo-write credential
    on the runner for all of it (r7 R82). A healthy release finishes in well
    under 30 minutes; anything longer is a failure worth killing.
    """

    release = _load_yaml(_WORKFLOWS / "release.yml")["jobs"]["release"]
    timeout = release.get("timeout-minutes")
    assert isinstance(timeout, int), (
        "the release job must declare timeout-minutes; the 6-hour default "
        "is a 6-hour repo-write App-token hold"
    )
    assert timeout <= 60, f"release timeout-minutes {timeout} exceeds the 1-hour token-hold budget"


def test_mutation_dispatch_inputs_are_validated_whole_string() -> None:
    """The slot step validates arm_shard with case patterns, never per-line grep.

    ``grep -qE '^...$'`` matches PER LINE: ``1/1\\nforged=x`` passed on its
    first line and the embedded newline reached ``$GITHUB_OUTPUT`` as a
    forged output row (r7 R82, empirically reproduced). Shell ``case``
    patterns match the entire string, newlines included, so the validation
    must stay case-only.
    """

    mutation = _load_yaml(_WORKFLOWS / "mutation.yml")["jobs"]
    steps = [step for job in mutation.values() for step in job.get("steps", [])]
    slot = next(step for step in steps if "INPUT_ARM_SHARD" in str(step.get("env", {})))
    script = "\n".join(
        line for line in slot["run"].splitlines() if not line.lstrip().startswith("#")
    )
    assert "grep" not in script, (
        "arm_shard validation must not use grep: it matches per line and a "
        "newline-embedding input forges $GITHUB_OUTPUT rows"
    )
    assert "*[!0-9/]*" in script, (
        "arm_shard validation lost the whole-string character-class case "
        "pattern that rejects newlines and shell metacharacters"
    )


def test_non_release_checkouts_do_not_persist_credentials() -> None:
    """Every checkout that never pushes sets ``persist-credentials: false``.

    checkout persists its token into ``.git/config`` for ALL later steps by
    default (zizmor ``artipacked``). Only the release job's checkout may
    persist — semantic-release pushes the version commit and tag through it.
    """

    offenders = []
    for workflow in sorted(_WORKFLOWS.glob("*.yml")):
        for job_name, job in _load_yaml(workflow)["jobs"].items():
            for step in job.get("steps", ()):
                if "actions/checkout@" not in step.get("uses", ""):
                    continue
                with_block = step.get("with", {})
                if workflow.name == "release.yml" and job_name == "release":
                    assert "token" in with_block, (
                        "the release checkout must authenticate as the App "
                        "(its push path relies on the persisted scoped token)"
                    )
                    continue
                if with_block.get("persist-credentials") is not False:
                    offenders.append(f"{workflow.name}:{job_name}")
    assert not offenders, (
        f"checkout steps persisting credentials without needing to push: {offenders}"
    )


def test_a_smoke_row_covers_the_sys_monitoring_python() -> None:
    """At least one PR-blocking row runs Python >= 3.12.

    Advertised 3.12/3.13 support selects the ``sys.monitoring``
    escape-detection path, which 3.10/3.11 rows can never execute: without a
    >=3.12 row that whole path ships untested (its only >=3.12-gated unit
    test had already rotted unnoticed, T13.4).
    """

    rows = [
        row for row in _smoke_job()["strategy"]["matrix"]["include"] if row.get("scope") == "smoke"
    ]
    versions = [tuple(int(part) for part in str(row["python"]).split(".")) for row in rows]
    assert any(version >= (3, 12) for version in versions), (
        "no smoke row runs Python >= 3.12; the sys.monitoring escape-detection "
        "path has no CI coverage"
    )


def test_every_advertised_python_classifier_has_a_smoke_row() -> None:
    """Each `Programming Language :: Python :: X.Y` classifier is CI-real.

    Python 3.13 was advertised (and even test-locked) while its only CI
    appearance was nightly's resolution-only ``uv pip compile`` loop, which
    imports nothing — a classifier enforced in metadata while unenforceable
    in CI (grind r5, b10 R85-1, SF-16). Every advertised interpreter must
    have a PR-blocking smoke row; drop the classifier or add the row.
    """

    import re

    # Regex on purpose: tomllib is 3.11+ and the suite's floor row runs 3.10.
    pyproject = (_PROJECT_ROOT / "pyproject.toml").read_text()
    advertised = set(re.findall(r'"Programming Language :: Python :: (3\.\d+)"', pyproject))
    assert advertised, "no python-version classifiers found in pyproject.toml"
    rows = [
        row for row in _smoke_job()["strategy"]["matrix"]["include"] if row.get("scope") == "smoke"
    ]
    executed = {str(row["python"]) for row in rows}
    unbacked = sorted(advertised - executed)
    assert not unbacked, (
        f"python classifiers with no PR-blocking smoke row: {unbacked} — "
        "resolution-only coverage is not execution"
    )


def test_packaging_tripwires_run_on_the_nightly_wheel_leg() -> None:
    """Nightly builds run the wheel-diet and sdist manifest tripwires.

    The nightly wheel job used to build and smoke-install a wheel WITHOUT
    running the diet manifest test (slow-marked, weekly-only), so a packaging
    regression could ship for up to a week before the tripwire fired (T13.5).
    """

    wheel_job = _load_yaml(_WORKFLOWS / "nightly.yml")["jobs"]["wheel"]
    runs = "\n".join(step.get("run", "") for step in wheel_job["steps"])
    assert "test_built_wheel_manifest_is_diet" in runs, (
        "the nightly wheel job must run the wheel-diet tripwire, not just build"
    )
    assert "test_built_sdist_manifest_is_governed" in runs, (
        "the nightly wheel job must run the sdist manifest tripwire"
    )


def test_release_job_python_satisfies_requires_python() -> None:
    """The release job builds the shipped artifacts on a supported Python.

    semantic-release's build_command builds the published sdist/wheel on the
    release job's interpreter; a 3.9 builder produced release artifacts on a
    Python the package's own requires-python (>=3.10) refuses (T13.7).
    """

    import re

    pyproject = (_PROJECT_ROOT / "pyproject.toml").read_text()
    match = re.search(r'requires-python\s*=\s*">=([0-9]+)\.([0-9]+)"', pyproject)
    assert match is not None, "pyproject must declare a parseable requires-python floor"
    floor = (int(match.group(1)), int(match.group(2)))

    release = _load_yaml(_WORKFLOWS / "release.yml")["jobs"]["release"]
    setup = next(step for step in release["steps"] if "setup-python" in step.get("uses", ""))
    version = tuple(int(part) for part in str(setup["with"]["python-version"]).split("."))
    assert version >= floor, (
        f"the release job builds on Python {version} but the package requires >= {floor}"
    )


@pytest.mark.slow
def test_built_sdist_manifest_is_governed(tmp_path: Path) -> None:
    """Assert the built sdist's manifest: package + metadata in, half-suites OUT.

    With no MANIFEST.in the default manifest shipped every ``tests/*.py``
    file WITHOUT its goldens, ENV markers, or data — an unrunnable half-suite
    — and nothing pinned the sdist at all (T13.5).
    """

    import subprocess
    import sys
    import tarfile

    sdist_dir = tmp_path / "sdist"
    sdist_dir.mkdir()
    # Build from a NEUTRAL cwd with an explicit srcdir, never cwd=_PROJECT_ROOT.
    # `python -m build` puts the cwd on sys.path[0], and a setuptools `build/`
    # directory in the repo root then shadows the installed `build` MODULE:
    # "No module named build.__main__; 'build' is a package and cannot be
    # directly executed". That made this test order-dependent -- it passed alone
    # and failed in CI right after the wheel-diet test, whose own build created
    # the ./build/ directory that broke it (2026-08-19).
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--outdir",
            str(sdist_dir),
            str(_PROJECT_ROOT),
        ],
        cwd=tmp_path,
        check=True,
    )
    archives = sorted(sdist_dir.glob("torchlens-*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0]) as archive:
        members = [name.split("/", 1)[1] for name in archive.getnames() if "/" in name]

    for required in (
        "LICENSE",
        "NOTICE",
        "README.md",
        "pyproject.toml",
        "torchlens/__init__.py",
        "torchlens/py.typed",
    ):
        assert required in members, f"sdist lost required member {required}"
    assert any(m.startswith("torchlens/schemas/") and m.endswith(".json") for m in members), (
        "sdist must ship the torchlens schema data files"
    )
    # r7 R84-2: ban ALL the trees MANIFEST.in prunes, plus the two private
    # gitignored roots no `prune` can cover — not just the original five.
    for banned_prefix in (
        "tests/",
        "docs/",
        "examples/",
        "notebooks/",
        "benchmarks/",
        "scripts/",
        "tools/",
        "templates/",
        ".research/",
        ".project-context/",
    ):
        offenders = [m for m in members if m.startswith(banned_prefix)]
        assert not offenders, (
            f"sdist ships {len(offenders)} member(s) under {banned_prefix} — the sdist "
            "is the wheel's source, not a repo snapshot (half-shipped suites are "
            "unrunnable; use a checkout)"
        )
    # r7 R84-1: the internal agent docs must never ship in EITHER artifact;
    # MANIFEST.in's recursive-exclude comment names this test as its belt.
    agent_docs = [m for m in members if m.rsplit("/", 1)[-1] in ("CLAUDE.md", "AGENTS.md")]
    assert not agent_docs, (
        f"sdist ships internal agent docs: {agent_docs} — MANIFEST.in's "
        "recursive-exclude belt regressed on a PUBLIC repo"
    )
