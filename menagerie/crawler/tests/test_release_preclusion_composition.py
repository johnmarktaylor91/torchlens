"""Round-21 VS1 real-path and anti-substitution preclusion proofs."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

import pytest

import menagerie.crawler.driver as driver_module
import menagerie.crawler.policy as policy_module
import menagerie.crawler.worker_supervisor as supervisor_module
from menagerie.crawler.tests.conftest import RealEnvironmentFixture
from menagerie.crawler.tests import test_anti_substitution_inventories as structural
from menagerie.crawler.tests.test_environment_authority_composition import (
    _run_host_denial_composition,
)


_CRAWLER_ROOT = Path(structural.__file__).parents[1]
_TEST_ROOT = _CRAWLER_ROOT / "tests"
_EVASION_CASES = (
    pytest.param(
        "direct-patch",
        """
def test_composition(monkeypatch):
    monkeypatch.setattr(driver_module, "_compile_worker_read_manifest", object())
""",
        id="direct-patch",
    ),
    pytest.param(
        "alias-patch",
        """
def test_composition(monkeypatch):
    replace = monkeypatch.setattr
    replace(driver_module, "_compile_worker_read_manifest", object())
""",
        id="alias-patch",
    ),
    pytest.param(
        "helper-indirection",
        """
def replace_boundary(replace, owner, name):
    replace(owner, name, object())

def test_composition(monkeypatch):
    replace_boundary(monkeypatch.setattr, driver_module, "_compile_worker_read_manifest")
""",
        id="helper-indirection",
    ),
    pytest.param(
        "decorator",
        """
boundary = "_compile" + "_worker_read_manifest"

@patch.object(driver_module, boundary, object())
def test_composition():
    pass
""",
        id="decorator",
    ),
    pytest.param(
        "assignment",
        """
def test_composition():
    compiler = driver_module
    compiler._compile_worker_read_manifest = object()
""",
        id="assignment",
    ),
    pytest.param(
        "dynamic-lookup",
        """
def test_composition():
    name = "_compile" + "_worker_read_manifest"
    return getattr(driver_module, name)
""",
        id="dynamic-lookup",
    ),
    pytest.param(
        "fake-environment-result",
        """
def test_composition(tmp_path):
    lane = FakeEnvironments(tmp_path / "fake")
    result = SupervisedResult(observation=object(), worker_receipt={})
    return lane, result
""",
        id="fake-environment-result",
    ),
    pytest.param(
        "legacy-root",
        """
def test_composition():
    grant_kind = "runtime-" + "root"
    return {"runtime_support": [(Path.cwd(), grant_kind)]}
""",
        id="legacy-root",
    ),
    pytest.param(
        "alternate-compiler",
        """
def test_composition(**kwargs):
    return compile_execution_read_manifest_v2(**kwargs)
""",
        id="alternate-compiler",
    ),
    pytest.param(
        "base-interpreter-argv",
        """
def test_composition():
    return (sys.executable, "-B", "-m", "menagerie.crawler.worker")
""",
        id="base-interpreter-argv",
    ),
)


def test_preclusion_real_v3_path_has_no_substitutable_fixture_edge(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """A real award/denial path must have no substitutable fixture edge.

    Parameters
    ----------
    tmp_path:
        Isolated real composition campaign root.
    real_environment_fixture:
        Strictly bound lock-selected hardlink clone used by the shipped v3 compiler.
    """

    scope = set(structural._COMPOSITION_SOURCES)  # noqa: SLF001
    required = {
        _TEST_ROOT / "conftest.py",
        _TEST_ROOT / "test_compiler_os_authority_composition.py",
        *sorted(_TEST_ROOT.glob("test_*composition*.py")),
    }
    assert required <= scope
    assert {
        _CRAWLER_ROOT / "cli.py",
        _TEST_ROOT / "dry_run_support.py",
        _TEST_ROOT / "test_slice_f_driver.py",
    } <= scope
    assert structural._composition_scope_errors() == ()  # noqa: SLF001
    assert set(structural.ROUND21_VS1_PROOF_REGISTRY) == {"P01", "T01", "T01-CI", "T02"}

    legacy_root_token = f"runtime{chr(45)}root"
    for module in (driver_module, policy_module, supervisor_module):
        source = inspect.getsource(module)
        assert "legacy_manifest_audit" not in source
        assert legacy_root_token not in source
    assert "live v3 model worker spawn requires execution-read-manifest.v3" in inspect.getsource(
        supervisor_module.supervise_worker
    )

    expected_sandbox = "bubblewrap" if sys.platform == "linux" else "sandbox-exec"
    _run_host_denial_composition(
        tmp_path,
        real_environment_fixture,
        expected_sandbox=expected_sandbox,
    )


@pytest.mark.parametrize(("evasion_class", "source"), _EVASION_CASES)
def test_tripwire_catches_python_evasion(
    evasion_class: str,
    source: str,
) -> None:
    """Each Python substitution class must produce a fully located diagnostic.

    Parameters
    ----------
    evasion_class:
        Stable §8.3 evasion-class name.
    source:
        In-memory composition mutation that must be rejected.
    """

    errors = structural._substitution_boundary_errors(  # noqa: SLF001
        source,
        source_path=Path("tests/test_mutated_composition.py"),
    )
    assert errors
    assert all("tests/test_mutated_composition.py" in diagnostic for diagnostic in errors)
    assert all("test_composition" in diagnostic for diagnostic in errors)
    assert all(evasion_class in diagnostic for diagnostic in errors)
    if evasion_class == "fake-environment-result":
        assert {diagnostic.split(":")[-2] for diagnostic in errors} == {
            "FakeEnvironments",
            "SupervisedResult",
        }


def test_tripwire_catches_deleted_ci_node() -> None:
    """Deleting one required real node from CI must be a located substitution error."""

    workflow = structural._WORKFLOW_PATH.read_text(encoding="utf-8")  # noqa: SLF001
    required_node = structural._REQUIRED_CI_SELECTIONS[0]  # noqa: SLF001
    errors = structural._required_ci_selection_errors(  # type: ignore[attr-defined]  # noqa: SLF001
        workflow.replace(required_node, "")
    )
    assert errors == (f".github/workflows/tests.yml:<workflow>:{required_node}:deleted-ci-node",)
