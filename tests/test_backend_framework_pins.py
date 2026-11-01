"""Framework pins: protocol requirements, pyproject extras and CI install lines agree.

Every backend states what it needs from its framework once, in the
``FrameworkRequirement`` of its protocol shell
(``torchlens/backends/_protocols/<backend>.py``): the supported range
``[minimum, below)`` minus ``excluded``, and the ``pinned`` version the
per-change CI leg installs. Three other places repeat those facts and must
not drift from them:

- the pyproject extra named by ``extra`` (the base ``torch`` dependency for
  torch) declares exactly the supported range;
- the ``backend-protocol`` legs in ``.github/workflows/pr.yml`` install
  exactly ``pinned`` (the declared range while a backend has no pin yet), and
  every pull-request torch install equals torch's pin;
- the nightly ``preview-backends`` legs install the declared range or the pin,
  with the same marker, import check and executed-test floor as the
  per-change leg.

The per-change legs run only when a change touches their backend; the
path rules live in ``scripts/backend_ci_changes.py`` and are pinned here too.
Runs in a torch-only environment: nothing here imports a preview framework.
"""

from __future__ import annotations

import importlib.util
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

import torchlens as tl
from torchlens.backends import FrameworkRequirement
from torchlens.backends.registry import get_backend_spec, registered_backend_specs

if sys.version_info >= (3, 11):
    import tomllib
else:  # pytest itself requires tomli below 3.11
    import tomli as tomllib

_REPO_ROOT = Path(tl.__file__).resolve().parent.parent
_PR_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "pr.yml"
_NIGHTLY_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "nightly.yml"
_CHANGES_SCRIPT = _REPO_ROOT / "scripts" / "backend_ci_changes.py"

ALL_BACKENDS = ("torch", "tf", "jax", "mlx", "paddle", "tinygrad")
PREVIEW_BACKENDS = ("tf", "jax", "mlx", "paddle", "tinygrad")

#: A specifier set compared by meaning: ``>=0.13`` equals ``>=0.13.0``.
SpecKey = frozenset[tuple[str, Version]]


def _framework(backend: str) -> FrameworkRequirement:
    """The registered protocol's framework requirement for ``backend``."""

    spec = get_backend_spec(backend)
    assert spec.protocol is not None, f"{backend} registers no protocol factory"
    return spec.protocol().framework


def _spec_key(requirement: Requirement) -> SpecKey:
    """The requirement's specifier as (operator, version) pairs."""

    return frozenset((spec.operator, Version(spec.version)) for spec in requirement.specifier)


def _declared_range(framework: FrameworkRequirement) -> SpecKey:
    """The supported range ``[minimum, below)`` minus ``excluded``, as a key."""

    pairs = {(">=", Version(framework.minimum))}
    if framework.below is not None:
        pairs.add(("<", Version(framework.below)))
    pairs.update(("!=", Version(version)) for version in framework.excluded)
    return frozenset(pairs)


def _pin_key(pinned: str) -> SpecKey:
    return frozenset({("==", Version(pinned))})


def _framework_requirement_in(
    requirements: list[str], framework: FrameworkRequirement
) -> Requirement:
    """The one requirement in ``requirements`` naming a framework distribution."""

    names = {canonicalize_name(name) for name in framework.distributions}
    matches = [
        parsed
        for parsed in (Requirement(text) for text in requirements)
        if canonicalize_name(parsed.name) in names
    ]
    assert len(matches) == 1, f"expected one of {sorted(names)} in {requirements}, got {matches}"
    return matches[0]


def _pyproject() -> dict[str, Any]:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _workflow(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _legs(path: Path, job: str) -> dict[str, dict[str, Any]]:
    """The matrix ``include`` rows of ``job`` keyed by backend."""

    rows = _workflow(path)["jobs"][job]["strategy"]["matrix"]["include"]
    legs = {row["backend"]: row for row in rows}
    assert len(legs) == len(rows), f"{path.name} {job} repeats a backend"
    return legs


def _install_requirements(leg: dict[str, Any]) -> list[str]:
    return shlex.split(str(leg["install"]))


def _changes_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_backend_ci_changes", _CHANGES_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_registered_backend_states_a_framework_requirement() -> None:
    """The six backends register protocols; previews are everything but torch."""

    names = {spec.name for spec in registered_backend_specs() if spec.protocol is not None}
    assert set(ALL_BACKENDS) <= names
    for backend in ALL_BACKENDS:
        assert isinstance(_framework(backend), FrameworkRequirement)


@pytest.mark.parametrize("backend", PREVIEW_BACKENDS)
def test_preview_extra_declares_exactly_the_supported_range(backend: str) -> None:
    """The pyproject extra's framework specifier equals ``[minimum, below)`` minus excluded."""

    framework = _framework(backend)
    extras = _pyproject()["project"]["optional-dependencies"]
    assert framework.extra in extras, f"{backend}: extra {framework.extra!r} missing"
    requirement = _framework_requirement_in(extras[framework.extra], framework)
    assert _spec_key(requirement) == _declared_range(framework), (
        f"{backend}: extra {framework.extra!r} declares {requirement.specifier}, "
        f"the protocol declares {sorted(_declared_range(framework))}"
    )


def test_torch_base_dependency_matches_the_declared_floor() -> None:
    """Torch is a base dependency: its lowest floor is ``minimum`` and no ceiling is set."""

    framework = _framework("torch")
    assert framework.extra == ""
    torch_lines = [
        Requirement(text)
        for text in _pyproject()["project"]["dependencies"]
        if canonicalize_name(Requirement(text).name) == "torch"
    ]
    assert torch_lines, "torch is not a base dependency"
    floors = [
        Version(s.version) for line in torch_lines for s in line.specifier if s.operator == ">="
    ]
    ceilings = [s for line in torch_lines for s in line.specifier if s.operator in ("<", "<=")]
    assert min(floors) == Version(framework.minimum)
    if framework.below is None:
        assert not ceilings, f"torch declares no ceiling but pyproject sets {ceilings}"


def test_pr_backend_protocol_legs_cover_exactly_the_previews() -> None:
    """One per-change leg per preview backend, matching the change-detection map."""

    legs = _legs(_PR_WORKFLOW, "backend-protocol")
    assert sorted(legs) == sorted(PREVIEW_BACKENDS)
    assert sorted(_changes_module().BACKEND_TOKENS) == sorted(PREVIEW_BACKENDS)


@pytest.mark.parametrize("backend", PREVIEW_BACKENDS)
def test_pr_leg_installs_the_pinned_framework(backend: str) -> None:
    """The per-change leg installs ``pinned`` exactly, or the declared range while unpinned."""

    framework = _framework(backend)
    leg = _legs(_PR_WORKFLOW, "backend-protocol")[backend]
    requirement = _framework_requirement_in(_install_requirements(leg), framework)
    expected = (
        _declared_range(framework) if framework.pinned is None else _pin_key(framework.pinned)
    )
    assert _spec_key(requirement) == expected, (
        f"{backend}: pr.yml installs {requirement}, the protocol pins {framework.pinned!r} "
        f"in {sorted(_declared_range(framework))}"
    )


def test_pull_request_torch_installs_equal_the_torch_pin() -> None:
    """Every exact torch install in pr.yml is torch's pin (local ``+cpu`` label aside).

    A CPU torch from the PyTorch index needs the matching CPU torchvision beside
    it, so every exact torchvision install equals the pinned smoke row's.
    """

    pinned = _framework("torch").pinned
    assert pinned is not None, "torch's per-change environment must be pinned"
    text = _PR_WORKFLOW.read_text(encoding="utf-8")
    installs = re.findall(r'"torch==([0-9][^"]*)"', text)
    smoke_rows = _workflow(_PR_WORKFLOW)["jobs"]["smoke"]["strategy"]["matrix"]["include"]
    versions = installs + [str(row["torch"]) for row in smoke_rows]
    assert versions, "pr.yml installs no exact torch"
    assert {Version(version).public for version in versions} == {pinned}, versions
    torchvision = re.findall(r'"torchvision==([0-9][^"]*)"', text)
    smoke_torchvision = {str(row["torchvision"]) for row in smoke_rows}
    assert torchvision and set(torchvision) <= smoke_torchvision, (torchvision, smoke_torchvision)


@pytest.mark.parametrize("backend", PREVIEW_BACKENDS)
def test_nightly_leg_installs_the_declared_range_or_the_pin(backend: str) -> None:
    """Nightly tracks the newest release in range (or the pin), never outside it."""

    framework = _framework(backend)
    leg = _legs(_NIGHTLY_WORKFLOW, "preview-backends")[backend]
    requirement = _framework_requirement_in(_install_requirements(leg), framework)
    allowed = {_declared_range(framework)}
    if framework.pinned is not None:
        allowed.add(_pin_key(framework.pinned))
    assert _spec_key(requirement) in allowed, f"{backend}: nightly installs {requirement}"


@pytest.mark.parametrize("backend", PREVIEW_BACKENDS)
def test_pr_and_nightly_legs_agree_on_suite_and_floor(backend: str) -> None:
    """Both legs run the same marker suite, import check and executed-test floor."""

    pr_leg = _legs(_PR_WORKFLOW, "backend-protocol")[backend]
    nightly_leg = _legs(_NIGHTLY_WORKFLOW, "preview-backends")[backend]
    for key in ("marker", "import_check", "executed_floor"):
        assert pr_leg[key] == nightly_leg[key], f"{backend}: {key} differs"
    assert pr_leg.get("dev") == nightly_leg.get("dev")
    assert int(pr_leg["executed_floor"]) >= 10


def test_pr_backend_protocol_job_gates_floor_and_scorecard() -> None:
    """Each leg runs on pull requests, declares its floor and checks the shipped scorecard."""

    workflow = _workflow(_PR_WORKFLOW)
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict) and "pull_request" in triggers
    job = workflow["jobs"]["backend-protocol"]
    runs = "\n".join(str(step.get("run", "")) for step in job["steps"])
    assert "scripts/backend_ci_changes.py" in runs
    assert 'pytest tests/ -m "$MARKER and not slow"' in runs
    assert "scripts/check_ci_executed_tests.py --legs" in runs
    assert "$EXECUTED_FLOOR" in runs
    assert 'python -m torchlens.conformance.scorecard --backend "$BACKEND" --check' in runs
    gated = [step for step in job["steps"] if "if" in step]
    assert all("steps.changes.outputs.run == 'true'" in str(step["if"]) for step in gated)


def test_torch_scorecard_is_checked_in_its_pinned_smoke_row() -> None:
    """Torch has no preview leg; its pinned smoke row checks the shipped torch scorecard."""

    job = _workflow(_PR_WORKFLOW)["jobs"]["smoke"]
    runs = [str(step.get("run", "")) for step in job["steps"]]
    assert "python -m torchlens.conformance.scorecard --backend torch --check" in runs


def test_nightly_runs_the_ecosystem_rosters() -> None:
    """The nightly preview matrix runs each backend's ecosystem roster with networked weights."""

    job = _workflow(_NIGHTLY_WORKFLOW)["jobs"]["preview-backends"]
    ecosystem = [
        step
        for step in job["steps"]
        if step.get("env", {}).get("TORCHLENS_CONFORMANCE_ECOSYSTEM") == "1"
    ]
    assert len(ecosystem) == 1
    assert "torchlens.conformance.scorecard --backend" in ecosystem[0]["run"]
    assert any(str(step.get("uses", "")).startswith("actions/cache@") for step in job["steps"])


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("torchlens/backends/registry.py", set(PREVIEW_BACKENDS)),
        ("torchlens/backends/_finalize.py", set(PREVIEW_BACKENDS)),
        ("torchlens/backends/_protocols/tf.py", set(PREVIEW_BACKENDS)),
        ("torchlens/conformance/_cases.py", set(PREVIEW_BACKENDS)),
        ("torchlens/compat/_report.py", set(PREVIEW_BACKENDS)),
        ("tests/backend_conformance/test_preview_tf.py", set(PREVIEW_BACKENDS)),
        ("pyproject.toml", set(PREVIEW_BACKENDS)),
        (".github/workflows/pr.yml", set(PREVIEW_BACKENDS)),
        ("torchlens/backends/mlx/wrappers.py", {"mlx"}),
        ("torchlens/backends/tinygrad/capture.py", {"tinygrad"}),
        ("docs/backends/paddle.md", {"paddle"}),
        ("tests/test_tf_eager_capture.py", {"tf"}),
        ("tests/test_keras_layers.py", {"tf"}),
        ("tests/test_jax_flax_modules.py", {"jax"}),
        ("tests/fixtures/paddle/model.py", {"paddle"}),
        ("tests/test_stuff_after.py", set()),
        ("torchlens/data_classes/op.py", set()),
        ("docs/backends.md", set()),
        ("torchlens/backends/torch/wrappers.py", set()),
    ],
)
def test_change_detection_runs_exactly_the_touched_legs(path: str, expected: set[str]) -> None:
    """A shared path runs every leg, a backend path its own, anything else none."""

    changes = _changes_module()
    ran = {backend for backend in PREVIEW_BACKENDS if changes.leg_should_run(backend, [path])[0]}
    assert ran == expected


def test_change_detection_cli_prints_a_github_output_line(tmp_path: Path) -> None:
    """``--all`` runs the leg; an undiffable base runs it too rather than skipping."""

    def run(*args: str) -> str:
        result = subprocess.run(
            [sys.executable, str(_CHANGES_SCRIPT), "--backend", "jax", *args],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            check=True,
        )
        return result.stdout.strip()

    assert run("--all") == "run=true"
    assert run("--base", "no-such-revision") == "run=true"
    with pytest.raises(KeyError):
        _changes_module().leg_should_run("torch", ["pyproject.toml"])
