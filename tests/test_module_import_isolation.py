"""Every torchlens module must be importable STANDALONE (no circular-import trap).

``import torchlens.bundle`` -- a documented PUBLIC module -- raised
``ImportError: cannot import name 'AmbiguousLabelError' from partially initialized
module 'torchlens.bundle'`` whenever it was the first torchlens module imported in a
process. ``torchlens/bundle/__init__.py`` imports
``torchlens.intervention._metrics``, which first executes
``torchlens/intervention/__init__.py``, which eagerly imported ``.bundle`` -- back
into the still-executing ``torchlens.bundle``. Two test modules
(``test_intervention_hardening_cert7/8.py``) were consequently uncollectable when run
alone and passed in the full suite only because an earlier module happened to import
``torchlens.intervention`` first: a green suite hiding a broken public import.

This module is the standing gate for that CLASS. The cheap arm pins the exact members
of the class (fixed vs still-pending); the exhaustive arm walks every module in the
package. Each candidate is imported in a FORKED child so it observes a genuinely cold
``sys.modules`` -- the parent's already-imported ``torch`` is inherited, which is what
makes the sweep affordable at all.

Adding a fix elsewhere is expected to FAIL the ledger test until the module is moved
out of ``PENDING_IMPORT_CYCLES``; that is the point (a residual cannot be quietly
forgotten, and a fixed module cannot silently regress).
"""

from __future__ import annotations

import ast
import os
import sys
import traceback
from pathlib import Path

import pytest
from _import_sim import PackageIndex, simulate_cold_import, simulate_every_module

import torchlens as tl

_PACKAGE_ROOT = Path(tl.__file__).resolve().parent
_TESTS_ROOT = Path(__file__).resolve().parent

# Modules whose standalone import is FIXED and must stay fixed.
FIXED_IMPORT_CYCLES = (
    "torchlens.bundle",
    "torchlens._chunking",
    "torchlens._user_public_impls",
    # _episode_ledger re-exports _episode_fold's names (C07X split); the
    # re-export is now a lazy module __getattr__ instead of an eager bottom-
    # of-file import, so a cold import of _episode_fold first no longer
    # catches _episode_ledger still mid-initialization.
    "torchlens._capture._episode_fold",
    # The selector-evaluation entry points: ``_ir._selector_universe`` imported a
    # ``_data_classes`` leaf at module level, which ran that package's init,
    # whose ``op`` module imports the ``intervention`` package, whose ``spec``
    # imports back into the still-loading ``selector_eval``. Every module whose
    # cold import reached ``selector_eval`` first (the selection resolver, the
    # capture predicates, the torch ops facade, recording) failed with it; the
    # guard import is deferred now and these entries pin it.
    "torchlens._ir.selector_eval",
    "torchlens._ir._selector_universe",
    "torchlens._selection.resolver",
    "torchlens._capture.predicates",
    "torchlens.backends.torch.ops",
    "torchlens.recording._generation",
)

#: Modules whose standalone import legitimately requires a THIRD-PARTY package
#: that is, BY DESIGN, never installed alongside the rest of the test
#: environment -- a genuinely absent optional dependency, not a circular
#: import -- so neither FIXED_IMPORT_CYCLES nor PENDING_IMPORT_CYCLES (both
#: about the import-cycle class) is the right ledger. Keyed to the exact
#: top-level import whose absence explains the failure; checked structurally
#: (``importlib.util.find_spec``) so an environment that DOES install the
#: package still gets the real standalone-import proof.
OPTIONAL_DEPENDENCY_MODULES: dict[str, str] = {
    # pyproject.toml: lit-nlp 1.3 requires numpy<2, so [lit] stays excluded
    # from all/all-stretch (Batch-8 ruling) -- no CI leg ever installs it
    # alongside the rest of the suite's extras.
    "torchlens.bridge.lit._adapters": "lit_nlp",
}

# Modules that still fail a standalone import. Each lives outside this lane's write
# territory (the r3 ops-split and IR packages), so the cycle is DISCLOSED here rather
# than silently tolerated. Fixing one means deleting its row.
PENDING_IMPORT_CYCLES = {
    "torchlens.backends.torch._ops_retention": (
        "_ops_retention still imports back into its sibling ops.py facade "
        "(the other _ops_* splits import standalone since the rebind removal)"
    ),
}


def _optional_dependency_explains_failure(module_name: str) -> bool:
    """Return whether ``module_name``'s declared optional package is absent.

    Parameters
    ----------
    module_name:
        Dotted module path looked up in ``OPTIONAL_DEPENDENCY_MODULES``.

    Returns
    -------
    bool
        True only when the module is registered AND its required package is
        genuinely not installed in this environment -- an environment that
        does install it still gets the real standalone-import proof.
    """

    required_package = OPTIONAL_DEPENDENCY_MODULES.get(module_name)
    if required_package is None:
        return False
    import importlib.util

    return importlib.util.find_spec(required_package) is None


def _package_modules() -> list[str]:
    """Return every importable module name in the installed torchlens package."""

    names: list[str] = []
    for source_path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        parts = list(source_path.relative_to(_PACKAGE_ROOT).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(["torchlens", *parts]))
    return names


def _standalone_import_error(module_name: str) -> str | None:
    """Import ``module_name`` in a forked child and return its failure, if any.

    Parameters
    ----------
    module_name:
        Dotted module path to import cold.

    Returns
    -------
    str | None
        The child's final traceback line, or ``None`` when the import succeeded.
    """

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child process, never measured by coverage
        os.close(read_fd)
        try:
            # Establish the COLD precondition the probe claims. If an earlier
            # test in this session ran a capture, the child inherits WRAPPED
            # torch whose wrappers reference the parent's (about-to-be-
            # orphaned) torchlens modules: an import-time torch call in the
            # fresh import then routes through a stale wrapper whose lazy
            # `from .x import y` resolves against the half-initialized fresh
            # module -- five spurious "circular imports" in backends/torch's
            # central modules (grind p5 §3.9 R76, 4/4 repro). A real fresh
            # process can never have wrapped torch before its first torchlens
            # import, so the child restores torch FIRST (fork-isolated; the
            # parent is untouched). An unwrap failure is reported as the
            # probe's failure, never swallowed into a polluted cold import.
            wrappers = sys.modules.get("torchlens.backends.torch.wrappers")
            state = sys.modules.get("torchlens._state")
            if (
                wrappers is not None
                and state is not None
                and getattr(state, "_is_decorated", False)
            ):
                wrappers.unwrap_torch()
            for key in [k for k in sys.modules if k == "torchlens" or k.startswith("torchlens.")]:
                del sys.modules[key]
            __import__(module_name)
            message = b""
        except BaseException:
            message = traceback.format_exc().strip().splitlines()[-1].encode()[:400]
        try:
            os.write(write_fd, message)
            os.close(write_fd)
        finally:
            os._exit(0)
    os.close(write_fd)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(read_fd, 4096)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(read_fd)
    os.waitpid(pid, 0)
    failure = b"".join(chunks).decode(errors="replace")
    return failure or None


pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork"),
    reason="the cold-import probe needs os.fork to give each module a fresh sys.modules",
)


@pytest.mark.heavy
def test_standalone_import_probe_is_immune_to_a_prior_capture() -> None:
    """The cold-import probe holds after a capture wrapped torch in-session.

    One ``tl.trace()`` earlier in the session used to make this gate fail
    with five spurious circular imports in backends/torch's central modules
    (the forked child inherited wrapped torch plus orphaned wrapper modules
    -- a state no real fresh process can be in). The probe now restores torch
    in the child first; this test pins that immunity by wrapping via a real
    capture and then probing the five previously-failing modules.
    """

    import torch

    import torchlens as tl

    class _OneOp(torch.nn.Module):
        """Minimal module so the capture installs the torch wrappers."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply one logged op.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Activation.
            """

            return torch.relu(x)

    tl.trace(_OneOp(), torch.ones(2))
    failures = {
        name: error
        for name in (
            "torchlens.backends.torch._completeness_cross_thread",
            "torchlens.backends.torch._ops_predicates",
            "torchlens.backends.torch.backend",
            "torchlens.backends.torch.completeness_witness",
            "torchlens.backends.torch.wrappers",
        )
        if (error := _standalone_import_error(name)) is not None
    }
    assert not failures, f"the probe is still capture-polluted: {failures}"


@pytest.mark.heavy
def test_known_cycle_members_import_standalone() -> None:
    """The fixed members of the circular-import class import cold, individually."""

    failures = {
        name: error
        for name in FIXED_IMPORT_CYCLES
        if (error := _standalone_import_error(name)) is not None
    }
    assert not failures, f"standalone import regressed: {failures}"


@pytest.mark.heavy
def test_pending_cycles_are_exactly_the_ledger() -> None:
    """The disclosed residual set is EXACT: no silent fix, no silent new cycle."""

    still_failing = {
        name for name in PENDING_IMPORT_CYCLES if _standalone_import_error(name) is not None
    }
    fixed = set(PENDING_IMPORT_CYCLES) - still_failing
    assert not fixed, (
        "these modules now import standalone -- delete their PENDING_IMPORT_CYCLES rows: "
        f"{sorted(fixed)}"
    )


@pytest.mark.slow
def test_every_package_module_imports_standalone() -> None:
    """Exhaustive class gate: every torchlens module imports cold.

    One forked import per module across the whole package (~470 modules), so this is
    the ``slow`` tier by construction; the ``heavy`` tests above are the per-step arm.
    """

    modules = _package_modules()
    assert len(modules) > 300, f"module discovery found only {len(modules)} modules"
    failures = {
        name: error
        for name in modules
        if name not in PENDING_IMPORT_CYCLES
        and not _optional_dependency_explains_failure(name)
        and (error := _standalone_import_error(name)) is not None
    }
    assert not failures, (
        "modules that cannot be imported standalone (fix the cycle, or disclose it in "
        f"PENDING_IMPORT_CYCLES with a reason): {failures}"
    )


@pytest.mark.heavy
def test_cold_import_simulation_flags_only_the_ledger() -> None:
    """Static arm: replay a fresh interpreter importing each module FIRST.

    The forked exhaustive walk above is the ``slow`` tier, so a cycle that only
    shows when a selector module is the first import reached the integration
    branch unseen (``import torchlens._ir.selector_eval`` failed cold while
    the whole suite, which imports ``torchlens.intervention`` early, stayed
    green). This arm needs no torch and no fork: it parses the package once
    and replays Python's module-level import order from every module as the
    entry, so it runs per step. The forked walk stays the ground truth; this
    is the cheap tripwire in front of it.
    """

    index = PackageIndex.build(_PACKAGE_ROOT)
    failing = simulate_every_module(index)
    unexpected = {
        entry: [str(failure) for failure in failures[:3]]
        for entry, failures in failing.items()
        if entry not in PENDING_IMPORT_CYCLES
    }
    assert not unexpected, (
        "a cold import of these modules would hit a partially initialised module "
        "(fix the cycle, or disclose it in PENDING_IMPORT_CYCLES with a reason): "
        f"{unexpected}"
    )


def test_cold_import_simulation_detects_a_planted_cycle(tmp_path: Path) -> None:
    """The static arm is a live instrument: a planted cycle is flagged, a clean tree is not."""

    package = tmp_path / "torchlens"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "a.py").write_text("from .b import helper\n\ndef shared():\n    return 1\n")
    (package / "b.py").write_text("from .a import shared\n\ndef helper():\n    return shared()\n")
    index = PackageIndex.build(package)

    flagged = simulate_cold_import(index, "torchlens.a")
    assert [(f.module, f.target, f.name) for f in flagged] == [
        ("torchlens.b", "torchlens.a", "shared")
    ]
    assert flagged[0].cycle == ("torchlens.a", "torchlens.b")
    # Entered from ``b`` the same cycle fails at the other edge: ``a`` asks the
    # still-loading ``b`` for ``helper`` before ``b`` has defined it.
    assert [(f.module, f.target, f.name) for f in simulate_cold_import(index, "torchlens.b")] == [
        ("torchlens.a", "torchlens.b", "helper")
    ]

    (package / "b.py").write_text(
        "def helper():\n    from .a import shared\n\n    return shared()\n"
    )
    assert simulate_every_module(PackageIndex.build(package)) == {}


def _is_sys_modules_target(node: ast.expr) -> bool:
    """True when ``node`` spells ``sys.modules`` (as a name or as a string)."""
    if isinstance(node, ast.Constant):
        return node.value == "sys.modules"
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "modules"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _patch_dict_sys_modules_sites(tree: ast.Module) -> list[int]:
    """Line numbers of ``patch.dict(sys.modules, ...)`` calls in ``tree``."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "dict"
        and isinstance(node.func.value, (ast.Name, ast.Attribute))
        and getattr(node.func.value, "id", getattr(node.func.value, "attr", None)) == "patch"
        and node.args
        and _is_sys_modules_target(node.args[0])
    ]


def test_no_test_snapshot_restores_sys_modules() -> None:
    """No test may wrap ``sys.modules`` in ``unittest.mock.patch.dict``.

    ``patch.dict`` restores the WHOLE mapping to its entry snapshot on exit, so
    every module first imported inside the block (a lazily imported torchlens
    submodule reached through the code under test) is purged from
    ``sys.modules`` while the package attribute that was bound during the
    import keeps the old module object. A later test that patches the module
    through the package attribute then patches a ghost: the code under test
    re-imports a fresh copy with default constants and the patch never lands.
    That is an ordering dependence between unrelated test files, visible only
    when the xdist scheduler puts them on one worker. Hide a single dependency
    with ``monkeypatch.setitem(sys.modules, name, None)`` (inside
    ``monkeypatch.context()`` for block scope): it restores only that key.
    """

    offenders: list[str] = []
    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "patch.dict" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        offenders.extend(
            f"{path.relative_to(_TESTS_ROOT.parent)}:{lineno}"
            for lineno in _patch_dict_sys_modules_sites(tree)
        )
    assert not offenders, (
        "patch.dict on sys.modules purges modules first imported inside the block and "
        "desynchronises sys.modules from package attributes; use "
        "monkeypatch.setitem(sys.modules, name, None) instead: " + ", ".join(offenders)
    )
