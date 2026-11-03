"""Every rare preview-backend test is selected by a nightly leg that installs what it imports.

Preview-backend tests (TensorFlow, JAX, MLX, Paddle, tinygrad) carry ``rare``, which the
default ``addopts`` deselects, so the only place they run is the nightly
``preview-backends`` matrix. Each leg proves its frameworks importable with
``import_check`` and runs ``-m "<marker> and not slow"`` and ``-m "<marker> and slow"``;
a leg that declares a pair also installs ``pair_install`` and runs
``-m "<pair_marker> and not slow"``. A rare test that no selection matches, or that a
matching selection cannot import, runs nowhere: its ``importorskip`` turns it into a
silent skip, or it is never collected at all.

This module reads the matrix from ``.github/workflows/nightly.yml`` and walks every test
module's syntax tree (no framework is imported), so it holds on any CI leg.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TESTS_ROOT = _REPO_ROOT / "tests"
_NIGHTLY = _REPO_ROOT / ".github" / "workflows" / "nightly.yml"
_PREVIEW_JOB = "preview-backends"

#: Top-level module name of each preview framework -> the backend whose leg installs it.
_FRAMEWORK_BACKENDS: dict[str, str] = {
    "mlx": "mlx",
    "mlx_lm": "mlx",
    "tinygrad": "tinygrad",
    "paddle": "paddle",
    "tensorflow": "tf",
    "keras": "tf",
    "keras_hub": "tf",
    "jax": "jax",
    "jaxlib": "jax",
    "flax": "jax",
    "equinox": "jax",
}


@dataclass(frozen=True)
class _Leg:
    """One ``preview-backends`` matrix entry."""

    name: str
    marker: str
    backends: frozenset[str]


@dataclass(frozen=True)
class _PreviewTest:
    """One rare test function that needs at least one preview framework."""

    nodeid: str
    markers: frozenset[str]
    backends: frozenset[str]


def _import_check_backends(import_check: str) -> frozenset[str]:
    """Return the backends a leg's ``import_check`` statement proves importable."""

    tree = ast.parse(import_check)
    modules = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    return frozenset(_FRAMEWORK_BACKENDS[name] for name in modules if name in _FRAMEWORK_BACKENDS)


#: The ``-m`` expressions the preview job's steps run, by the matrix key they read.
_STEP_SELECTIONS: tuple[tuple[str, str], ...] = (
    ("marker", "and not slow"),
    ("marker", "and slow"),
    ("pair_marker", "and not slow"),
)


def _preview_job() -> dict[str, object]:
    """Return the ``preview-backends`` job of the nightly workflow."""

    workflow = yaml.safe_load(_NIGHTLY.read_text(encoding="utf-8"))
    job = workflow["jobs"][_PREVIEW_JOB]
    assert isinstance(job, dict)
    return job


def _nightly_preview_legs() -> list[_Leg]:
    """Return one ``_Leg`` per selection the ``preview-backends`` matrix runs."""

    job = _preview_job()
    entries = job["strategy"]["matrix"]["include"]  # type: ignore[index]
    legs: list[_Leg] = []
    for entry in entries:
        backends = _import_check_backends(str(entry["import_check"]))
        for key, suffix in _STEP_SELECTIONS:
            if key == "marker":
                legs.append(
                    _Leg(f"{entry['backend']} {suffix}", f"({entry[key]}) {suffix}", backends)
                )
            elif key in entry:
                paired = backends | _import_check_backends(str(entry["pair_import_check"]))
                legs.append(_Leg(f"{entry['backend']} pair", f"({entry[key]}) {suffix}", paired))
    return legs


def _evaluate_marker_expression(expression: str, markers: frozenset[str]) -> bool:
    """Evaluate a ``-m`` expression (names, ``and``, ``or``, ``not``, parentheses)."""

    def walk(node: ast.expr) -> bool:
        if isinstance(node, ast.Name):
            return node.id in markers
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not walk(node.operand)
        if isinstance(node, ast.BoolOp):
            combine: Callable[[Iterator[bool]], bool] = all if isinstance(node.op, ast.And) else any
            return combine(walk(value) for value in node.values)
        raise ValueError(f"unsupported marker expression {expression!r}")

    return walk(ast.parse(expression, mode="eval").body)


def _mark_names(node: ast.expr) -> set[str]:
    """Return the ``pytest.mark.<name>`` names in a decorator or ``pytestmark`` value."""

    names: set[str] = set()
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Attribute)
            and isinstance(sub.value, ast.Attribute)
            and sub.value.attr == "mark"
        ):
            names.add(sub.attr)
    return names


def _pytestmark(body: list[ast.stmt]) -> set[str]:
    """Return the markers a module or class body assigns to ``pytestmark``."""

    names: set[str] = set()
    for statement in body:
        if isinstance(statement, ast.Assign | ast.AnnAssign):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if statement.value is not None and any(
                isinstance(t, ast.Name) and t.id == "pytestmark" for t in targets
            ):
                names |= _mark_names(statement.value)
    return names


def _frameworks(node: ast.AST, *, descend_into_defs: bool) -> set[str]:
    """Return preview backends a node imports or ``importorskip``-s.

    With ``descend_into_defs`` False, function and class bodies are not entered (the
    module-level requirements); with True the whole subtree counts (one test body).
    """

    found: set[str] = set()
    stack: list[ast.AST] = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, ast.Import):
            modules = [alias.name for alias in current.names]
        elif isinstance(current, ast.ImportFrom) and current.level == 0 and current.module:
            modules = [current.module]
        elif (
            isinstance(current, ast.Call)
            and isinstance(current.func, ast.Attribute)
            and current.func.attr == "importorskip"
            and current.args
            and isinstance(current.args[0], ast.Constant)
            and isinstance(current.args[0].value, str)
        ):
            modules = [current.args[0].value]
        else:
            modules = []
        found |= {
            _FRAMEWORK_BACKENDS[name.split(".")[0]]
            for name in modules
            if name.split(".")[0] in _FRAMEWORK_BACKENDS
        }
        for child in ast.iter_child_nodes(current):
            if not descend_into_defs and isinstance(
                child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
            ):
                continue
            stack.append(child)
    return found


def _test_functions(
    body: list[ast.stmt], prefix: str, inherited: set[str]
) -> Iterator[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, set[str]]]:
    """Yield ``(nodeid suffix, function, markers)`` for collected test functions."""

    for statement in body:
        if isinstance(statement, ast.ClassDef) and statement.name.startswith("Test"):
            class_marks = inherited | _pytestmark(statement.body)
            for decorator in statement.decorator_list:
                class_marks |= _mark_names(decorator)
            yield from _test_functions(statement.body, f"{prefix}{statement.name}::", class_marks)
        elif isinstance(
            statement, ast.FunctionDef | ast.AsyncFunctionDef
        ) and statement.name.startswith("test"):
            marks = set(inherited)
            for decorator in statement.decorator_list:
                marks |= _mark_names(decorator)
            yield f"{prefix}{statement.name}", statement, marks


def _rare_preview_tests() -> list[_PreviewTest]:
    """Return every rare test that needs a preview framework, across the test tree."""

    tests: list[_PreviewTest] = []
    for path in sorted(_TESTS_ROOT.rglob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        if "rare" not in source:
            continue
        module = ast.parse(source, filename=str(path))
        module_backends = _frameworks(module, descend_into_defs=False)
        module_marks = _pytestmark(module.body)
        relative = path.relative_to(_REPO_ROOT).as_posix()
        for suffix, function, marks in _test_functions(module.body, "", module_marks):
            if "rare" not in marks:
                continue
            backends = module_backends | _frameworks(function, descend_into_defs=True)
            if backends:
                tests.append(
                    _PreviewTest(
                        nodeid=f"{relative}::{suffix}",
                        markers=frozenset(marks),
                        backends=frozenset(backends),
                    )
                )
    return tests


def _selecting_legs(test: _PreviewTest, legs: list[_Leg]) -> list[str]:
    """Return the legs that select ``test`` and install every framework it needs."""

    return [
        leg.name
        for leg in legs
        if _evaluate_marker_expression(leg.marker, test.markers) and test.backends <= leg.backends
    ]


def test_every_rare_preview_test_runs_on_a_nightly_leg() -> None:
    """A rare preview test with no selecting, importing leg would never run anywhere."""

    legs = _nightly_preview_legs()
    tests = _rare_preview_tests()
    orphans = [
        f"{test.nodeid} (needs {sorted(test.backends)}, markers {sorted(test.markers)})"
        for test in tests
        if not _selecting_legs(test, legs)
    ]
    assert not orphans, "rare preview tests no nightly preview leg runs:\n" + "\n".join(orphans)


def test_modelled_selections_are_the_ones_the_steps_run() -> None:
    """Each modelled ``-m`` expression appears verbatim in a pytest step of the job."""

    runs = "\n".join(str(step.get("run", "")) for step in _preview_job()["steps"])  # type: ignore[attr-defined]
    for key, suffix in _STEP_SELECTIONS:
        assert f'pytest tests/ -m "${{{{ matrix.{key} }}}} {suffix}"' in runs, (key, suffix)


def test_scan_finds_the_known_preview_suites() -> None:
    """The scan is not vacuous: every backend has rare tests it recognizes."""

    tests = _rare_preview_tests()
    covered = set().union(*(test.backends for test in tests)) if tests else set()
    assert covered >= {"tf", "jax", "mlx", "paddle", "tinygrad"}
    assert any(test.nodeid.startswith("tests/test_backends_finalize_cert10.py::") for test in tests)


def test_selection_check_flags_an_unselected_or_unimportable_test() -> None:
    """The leg match rejects a test lacking a backend marker or needing an absent framework."""

    legs = [
        _Leg("mlx", "(backend_mlx) and not slow", frozenset({"mlx"})),
        _Leg(
            "both",
            "(backend_mlx and backend_tinygrad) and not slow",
            frozenset({"mlx", "tinygrad"}),
        ),
    ]
    unmarked = _PreviewTest("t::a", frozenset({"rare"}), frozenset({"mlx"}))
    slow = _PreviewTest("t::b", frozenset({"rare", "backend_mlx", "slow"}), frozenset({"mlx"}))
    two = _PreviewTest(
        "t::c",
        frozenset({"rare", "backend_mlx", "backend_tinygrad"}),
        frozenset({"mlx", "tinygrad"}),
    )
    assert _selecting_legs(unmarked, legs) == []
    assert _selecting_legs(slow, legs) == []
    assert _selecting_legs(two, legs) == ["both"]
    assert _import_check_backends("import mlx.core, tinygrad") == frozenset({"mlx", "tinygrad"})
