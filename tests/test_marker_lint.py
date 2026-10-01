"""Tier-marker lint: keep the smoke tier honest.

Two enforcement layers back the documented test tiers (CLAUDE.md "Testing Tiers",
tests/AGENTS.md "Markers"):

1. **Marker disjointness** (collection-level). pytest markers are ADDITIVE: a test
   carrying ``smoke`` together with ``heavy`` or ``slow`` still runs under
   ``-m smoke`` -- the heavier marker does NOT remove it from the fast gate. That
   combination is always a partition mistake; the fix is dropping ``smoke`` from
   the test (per-parametrize-case marks that give DIFFERENT cases different tiers
   are fine and pass this check, because ``get_closest_marker`` sees the resolved
   per-item marks).

2. **Duration budget** (runtime tripwire). The conftest hook accounts for fixture
   setup, call, and teardown time, then checks both each item and each resolved
   parametrized family (smoke AND unmarked) against the documented 5s partition
   boundary. The charged measure is ``min(wall, cpu)`` — load-robust and
   threading-robust; see the budget-constant comment block in conftest.py.

3. **State-isolation census** (static). Every warn-once module global must appear
   in the root autouse reset inventory, and every module-scoped fixture that
   creates a Trace must yield so it can release that Trace at module teardown.

Both checks read their data from the surrounding session, so they enforce on every
smoke/full-tier run and pass vacuously when this file is run alone.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

import pytest
from _source_corpus import package_ast, package_files, package_source

pytestmark = pytest.mark.smoke


#: Markers that may never combine with ``smoke`` on one resolved item.
#: ``heavy``/``slow``: additive markers keep the item in `-m smoke` despite the
#: heavier tier. ``rare`` (R41-2) is a duration-budget EXEMPTION channel, so a
#: smoke+rare item would sit in the commit gate with zero duration enforcement
#: (and `-m smoke` overrides the default `-m 'not rare'`, so smoke+rare items
#: DO run in the commit gate). ``serial`` is no longer budget-exempt (b2 R41
#: round 5) but stays smoke-incompatible: its load-sensitivity claim
#: contradicts running inside the parallel commit gate.
_SMOKE_INCOMPATIBLE_MARKERS = ("heavy", "slow", "serial", "rare")


def _tier_combo_violations(
    marker_names: set[str], callspec_marker_names: set[str], nodeid: str
) -> list[str]:
    """Return tier-combination violations for one resolved item's markers.

    Pure helper so the combination policy is unit-testable (red-capable)
    without planting real mis-marked tests.

    Policy:
    - ``smoke`` may not combine with any of ``_SMOKE_INCOMPATIBLE_MARKERS``.
    - ``heavy`` + ``slow`` is a partition contradiction (a test cannot be both
      5-20s and >20s) UNLESS ``slow`` arrived as a per-parametrize-cell
      refinement of a heavy family (``pytest.param(..., marks=slow)``), the
      sanctioned shape for "this one cell measures beyond heavy's ceiling"
      (R41-3). Budget enforcement already resolves the combo as slow-wins.

    Parameters
    ----------
    marker_names:
        All marker names on the resolved item.
    callspec_marker_names:
        Marker names contributed by the item's parametrize callspec.
    nodeid:
        Item node id used in violation strings.

    Returns
    -------
    list[str]
        Human-readable violation strings (empty when compliant).
    """

    violations = []
    if "smoke" in marker_names:
        for incompatible in _SMOKE_INCOMPATIBLE_MARKERS:
            if incompatible in marker_names:
                violations.append(f"{nodeid} [smoke + {incompatible}]")
    if (
        "heavy" in marker_names
        and "slow" in marker_names
        and "slow" not in callspec_marker_names
        and "heavy" not in callspec_marker_names
    ):
        violations.append(f"{nodeid} [heavy + slow, not a per-cell refinement]")
    return violations


def _item_combo_violations(item: pytest.Item) -> list[str]:
    """Apply the tier-combination policy to one collected pytest item."""

    marker_names = {marker.name for marker in item.iter_markers()}
    callspec = getattr(item, "callspec", None)
    callspec_marker_names = {marker.name for marker in getattr(callspec, "marks", [])}
    return _tier_combo_violations(marker_names, callspec_marker_names, item.nodeid)


def test_no_smoke_test_carries_a_heavier_tier_marker(request: pytest.FixtureRequest) -> None:
    """No collected item may carry a contradictory tier-marker combination."""

    conflicted = []
    for item in request.session.items:
        conflicted.extend(_item_combo_violations(item))
    assert not conflicted, (
        "Tests carry contradictory tier-marker combinations (markers are "
        "additive; exemption markers disarm duration budgets). Fix each "
        "combination:\n  " + "\n  ".join(conflicted)
    )


@pytest.mark.parametrize(
    ("markers", "callspec_markers", "expected_fragments"),
    [
        pytest.param({"smoke", "heavy"}, set(), ["smoke + heavy"], id="smoke-heavy"),
        pytest.param({"smoke", "slow"}, set(), ["smoke + slow"], id="smoke-slow"),
        pytest.param({"smoke", "serial"}, set(), ["smoke + serial"], id="smoke-serial"),
        pytest.param({"smoke", "rare"}, set(), ["smoke + rare"], id="smoke-rare"),
        pytest.param({"heavy", "slow"}, set(), ["heavy + slow"], id="heavy-slow-decorators"),
        pytest.param({"heavy", "slow"}, {"slow"}, [], id="heavy-family-slow-cell-ok"),
        pytest.param({"smoke"}, set(), [], id="smoke-alone-ok"),
        pytest.param({"heavy", "serial"}, set(), [], id="heavy-serial-ok"),
    ],
)
def test_tier_combo_policy_is_red_capable(
    markers: set[str], callspec_markers: set[str], expected_fragments: list[str]
) -> None:
    """The combination policy flags each banned shape and passes each sanctioned one."""

    violations = _tier_combo_violations(markers, callspec_markers, "planted::node")
    assert len(violations) == len(expected_fragments)
    for fragment in expected_fragments:
        assert any(fragment in violation for violation in violations), (fragment, violations)


def test_bounded_tier_tests_stay_within_duration_budget(
    request: pytest.FixtureRequest,
) -> None:
    """Every bounded-tier test must finish within its tier's duration budget.

    The budget is TWO-directional (R41): ``smoke`` AND unmarked tests are held
    to the 5s partition boundary, ``heavy`` to its 20s ceiling (all
    load-scaled); ``slow``/``rare`` are exempt by contract, and ``serial``
    resolves its tier budget normally (b2 R41 round 5: the former blanket
    exemption made one decorator a universal budget dodge). The
    CHARGED time is ``min(wall, cpu)`` so neither orchestrator load (wall
    inflation) nor torch intra-op threading (cpu inflation) can false-fail a
    genuinely in-budget test (round-4 load-flake fix). Budget values live in
    ``tests/conftest.py`` and ride along on each recorded offender -- a bare
    ``conftest`` import here would be ambiguous during full-suite collection
    (nested conftests share the module name).

    This test also asserts its own LAST-position ordering: the offender
    ledger only covers tests that already ran, so a reordering regression
    (e.g. a plugin shuffling after the conftest reorder) must go red here
    rather than silently truncating coverage.
    """

    items = request.session.items
    own_index = next(
        index for index, item in enumerate(items) if item.nodeid == request.node.nodeid
    )
    stragglers = [
        item.nodeid for item in items[own_index + 1 :] if "test_marker_lint" not in item.nodeid
    ]
    assert not stragglers, (
        "the duration-budget lint no longer runs last -- its offender ledger "
        f"would miss these later tests: {stragglers[:5]}"
    )

    guidance = {
        "smoke": "re-tier to `heavy` (5-20s) or `slow` (>20s), or make it faster",
        "unmarked": "unmarked tests run in the mid backstop: add `heavy`/`slow` "
        "consciously, or make it faster",
        "heavy": "re-tier to `slow` (>20s) or make it faster",
    }
    offenders = getattr(request.session, "_tl_duration_budget_offenders", [])
    lines = [
        f"{nodeid} [{tier}]: wall {wall:.1f}s / cpu {cpu:.1f}s "
        f"(budget {budget:.0f}s on min(wall, cpu)) -- {guidance[tier]}"
        for nodeid, tier, wall, cpu, budget in offenders
    ]
    assert not offenders, (
        "Tests exceeded their tier duration budget this session:\n  " + "\n  ".join(lines)
    )


def test_smoke_parametrized_families_stay_within_duration_budget(
    request: pytest.FixtureRequest,
) -> None:
    """Resolved 5s-tier parameter families must stay within the aggregate budget.

    Covers smoke AND unmarked families (R41-4: untiered families previously
    had no aggregate bound). A family of N parameters legitimately costs ~N
    single-test durations (the selector matrix is 278 cells), so each family's
    budget scales with its resolved cell count: load_factor * max(2x the
    per-test budget, the per-cell allowance x n_cells), charged on
    min(wall, cpu). Genuine per-cell ballooning still trips.
    """

    family_stats = getattr(request.session, "_tl_smoke_family_stats", {})
    family_budgets = getattr(request.session, "_tl_smoke_family_budgets", {})
    offenders = [
        (family, total, count, family_budgets.get(family, 0.0))
        for family, (total, count) in family_stats.items()
        if total > family_budgets.get(family, float("inf"))
    ]
    lines = [
        f"{family}: {total:.1f}s over {count} cells (budget {budget:.0f}s)"
        for family, total, count, budget in offenders
    ]
    assert not offenders, (
        "Smoke parametrized families exceeded their aggregate cell-scaled "
        "budget. Split or re-tier the family:\n  " + "\n  ".join(lines)
    )


def test_smoke_module_imports_stay_within_duration_budget(
    request: pytest.FixtureRequest,
) -> None:
    """Smoke-bearing modules must import and collect within the 5s boundary.

    Charged on ``min(wall, cpu)`` like the per-test budgets, so parallel
    orchestrator load cannot false-fail module imports either.
    """

    budget = 5.0
    durations = getattr(request.session, "_tl_module_collection_durations", {})
    smoke_paths = {
        str(item.path)
        for item in request.session.items
        if item.get_closest_marker("smoke") is not None
    }
    offenders = [
        (path, durations[path])
        for path in sorted(smoke_paths)
        if path in durations and min(durations[path]) > budget
    ]
    lines = [
        f"{path}: wall {wall:.1f}s / cpu {cpu:.1f}s (budget {budget:.0f}s on min)"
        for path, (wall, cpu) in offenders
    ]
    assert not offenders, (
        "Smoke-bearing modules exceeded the import/collection budget. Move expensive setup "
        "behind fixtures or re-tier the module:\n  " + "\n  ".join(lines)
    )


def test_every_collected_module_imports_within_the_heavy_boundary(
    request: pytest.FixtureRequest,
) -> None:
    """No collected module may burn more than the 20s heavy boundary importing.

    r7 R41 (b2): the import/collection budget was smoke-only — a 60s import
    in a heavy/slow/unmarked module escaped entirely, even though import cost
    is paid by EVERY selection that collects the file (including smoke runs,
    which collect the whole tree before deselecting). Non-smoke modules get
    the heavy partition boundary rather than smoke's 5s; charged on
    ``min(wall, cpu)`` like everything else.
    """

    budget = 20.0
    durations = getattr(request.session, "_tl_module_collection_durations", {})
    offenders = [
        (path, values) for path, values in sorted(durations.items()) if min(values) > budget
    ]
    lines = [
        f"{path}: wall {wall:.1f}s / cpu {cpu:.1f}s (budget {budget:.0f}s on min)"
        for path, (wall, cpu) in offenders
    ]
    assert not offenders, (
        "Modules exceeded the whole-tree import/collection boundary (import "
        "cost is paid by every selection that collects the file):\n  " + "\n  ".join(lines)
    )


def warm_scan_caches() -> None:
    """Pre-fill the whole-tree parse caches OUTSIDE any test's charged window.

    Called from the root conftest's collection hook when this module's tests
    are collected: the tests/-tree and torchlens/-package parses cost ~5-8s
    of genuine CPU, which would otherwise land in whichever lint test runs
    first and sit exactly on the smoke budget boundary.
    """

    _parsed_test_trees()
    _warn_once_declarations(Path(__file__).resolve().parents[1] / "torchlens")


@functools.lru_cache(maxsize=1)
def _parsed_test_trees() -> tuple[tuple[str, ast.Module], ...]:
    """Parse every test-suite Python file ONCE per session.

    The static lints below each used to re-parse the whole tree (~1300 files),
    costing ~5s PER TEST and sitting exactly on the smoke budget boundary
    under composition noise; one shared parse keeps each lint at ~0.1s.
    """

    tests_root = Path(__file__).resolve().parent
    return tuple(
        (
            str(path.relative_to(tests_root)),
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path)),
        )
        for path in sorted(tests_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    )


@functools.lru_cache(maxsize=1)
def _parsed_root_test_trees() -> tuple[tuple[str, ast.Module], ...]:
    """Root-level ``test*.py`` subset of :func:`_parsed_test_trees`."""

    return tuple(
        (relative, tree)
        for relative, tree in _parsed_test_trees()
        if "/" not in relative and relative.startswith("test")
    )


def _assigned_module_names(statement: ast.stmt) -> set[str]:
    """Return module names assigned by one top-level statement.

    Parameters
    ----------
    statement:
        Top-level syntax-tree statement.

    Returns
    -------
    set[str]
        Directly assigned names.
    """

    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, ast.AnnAssign):
        targets = [statement.target]
    else:
        return set()
    return {target.id for target in targets if isinstance(target, ast.Name)}


@functools.lru_cache(maxsize=1)
def _parsed_package_trees(package_root: Path) -> tuple[tuple[str, str, ast.Module], ...]:
    """Parse every ``torchlens`` package file ONCE per session.

    Same motivation as :func:`_parsed_test_trees` one tier down: the static
    censuses below each used to ``rglob`` and re-parse the whole package
    (~520 modules) independently, so the second one to run paid the full cost
    again. That made ``test_capability_dependent_caches_are_cleared`` measure
    6.57 s on a QUIET box against the 5 s smoke boundary -- 94% of its budget
    with nothing else running, i.e. a session-failing tripwire under any
    parallel load. Sharing one cached parse fixes the cost at its source
    instead of re-tiering an honest test or widening a budget.

    Parameters
    ----------
    package_root:
        Root of the ``torchlens`` package.

    Returns
    -------
    tuple[tuple[str, str, ast.Module], ...]
        ``(module_name, source, tree)`` per package file. ``source`` is carried
        because :func:`_lru_cached_functions` needs it for
        ``ast.get_source_segment``.
    """

    parsed: list[tuple[str, str, ast.Module]] = []
    for path in package_files():
        module_parts = path.relative_to(package_root.parent).with_suffix("").parts
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        module_name = ".".join(module_parts)
        parsed.append((module_name, package_source(path), package_ast(path)))
    return tuple(parsed)


@functools.lru_cache(maxsize=1)
def _warn_once_declarations(package_root: Path) -> set[tuple[str, str]]:
    """Collect warn-once module-global declarations from TorchLens sources.

    Parameters
    ----------
    package_root:
        Root of the ``torchlens`` package.

    Returns
    -------
    set[tuple[str, str]]
        ``(module_name, attribute_name)`` declarations.
    """

    declarations: set[tuple[str, str]] = set()
    for module_name, _source, tree in _parsed_package_trees(package_root):
        for statement in tree.body:
            for name in _assigned_module_names(statement):
                normalized = name.lower()
                if (
                    "warned" in normalized
                    or "warning_emitted" in normalized
                    or normalized.endswith("_warning_types")
                ):
                    declarations.add((module_name, name))
    return declarations


def test_warn_once_sentinel_census_matches_autouse_reset(
    request: pytest.FixtureRequest,
) -> None:
    """Every declared warn-once global must be isolated by the autouse fixture."""

    package_root = Path(__file__).resolve().parents[1] / "torchlens"
    discovered = _warn_once_declarations(package_root)
    configured_specs: tuple[tuple[str, str, object], ...] = (
        request.config._tl_warn_once_sentinel_specs
    )
    configured = {(module_name, name) for module_name, name, _default in configured_specs}
    # The global-mutated copy lives in _render_ordering since the renderer
    # thinning (the binding is import-created, so the source heuristic cannot
    # discover it there either).
    runtime_only = {("torchlens.visualization._render_ordering", "_SIBLING_ORDER_WARNING_EMITTED")}
    # Behavioral fidelity latches the NAME heuristic cannot discover (nothing
    # "warned"-shaped in the identifier), declared here explicitly so the two
    # ledgers (this census and the conftest reset list) can no longer disagree
    # silently (grind p5, B2P3-16 / sol R76-2). A test that trips one of these
    # degrades every later test in the session, so the reset is REQUIRED.
    sticky_latches = {
        ("torchlens.utils.rng", "_cuda_rng_unusable"),
        # Last-degradation record for lazy auto-arm (fix/distributed-r4): not
        # "warned"-shaped, but a test that degrades arming would otherwise
        # leak its reason into every later auto_arm_degradation() read.
        ("torchlens.distributed._lifecycle", "_AUTO_ARM_DEGRADATION"),
    }
    expected = discovered | runtime_only | sticky_latches
    assert configured == expected, (
        "Warn-once sentinel reset inventory drifted. Add/remove entries in "
        "tests/conftest.py::_WARN_ONCE_SENTINELS. "
        f"Missing resets: {sorted(expected - configured)}; "
        f"stale resets: {sorted(configured - expected)}"
    )


def _is_module_scoped_fixture(decorator: ast.expr) -> bool:
    """Return whether a decorator declares a long-lived pytest fixture.

    Any scope wider than the default per-test function scope ("class",
    "module", "package", "session") retains its Trace across tests, so all
    of them need the teardown gate — matching only the literal "module"
    left the wider scopes unguarded (b2p2 opus-B2P2-10 / sol-R77-2).

    Parameters
    ----------
    decorator:
        Function decorator syntax node.

    Returns
    -------
    bool
        Whether the decorator is ``pytest.fixture(scope=<non-function>)``.
    """

    if not isinstance(decorator, ast.Call):
        return False
    function = decorator.func
    is_fixture = (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "pytest"
        and function.attr == "fixture"
    ) or (isinstance(function, ast.Name) and function.id == "fixture")
    return is_fixture and any(
        keyword.arg == "scope"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value in {"class", "module", "package", "session"}
        for keyword in decorator.keywords
    )


def _calls_trace(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a fixture body directly calls a function named ``trace``.

    Parameters
    ----------
    function:
        Fixture function syntax node.

    Returns
    -------
    bool
        Whether a direct ``trace(...)`` or ``*.trace(...)`` call exists.
    """

    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "trace":
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == "trace":
            return True
    return False


def _module_trace_fixtures_without_yield(tests_root: Path) -> list[str]:
    """Find module-scoped Trace fixtures that cannot perform teardown.

    Parameters
    ----------
    tests_root:
        Root of the test suite.

    Returns
    -------
    list[str]
        Stable ``path::fixture`` violations.
    """

    del tests_root
    violations: list[str] = []
    for relative, tree in _parsed_test_trees():
        functions = (
            node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        for function in functions:
            if not any(_is_module_scoped_fixture(item) for item in function.decorator_list):
                continue
            if not _calls_trace(function):
                continue
            if any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(function)):
                continue
            violations.append(f"{relative}::{function.name}")
    return violations


def test_module_scoped_trace_fixtures_have_teardown() -> None:
    """Module-scoped fixtures retaining Traces must yield for explicit cleanup."""

    tests_root = Path(__file__).resolve().parent
    violations = _module_trace_fixtures_without_yield(tests_root)
    assert not violations, (
        "Module-scoped fixtures create live Traces without a teardown path. Yield the Trace "
        "and call cleanup() in finally:\n  " + "\n  ".join(violations)
    )


def test_root_conftest_does_not_inject_repo_into_sys_path() -> None:
    """The suite must not hide an unusable editable install via path mutation."""

    conftest_path = Path(__file__).with_name("conftest.py")
    tree = ast.parse(conftest_path.read_text(encoding="utf-8"), filename=str(conftest_path))
    violations: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        is_sys_path = (
            isinstance(owner, ast.Attribute)
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "sys"
            and owner.attr == "path"
        )
        if is_sys_path and node.func.attr in {"append", "extend", "insert"}:
            violations.append(node.lineno)
    assert not violations, (
        "tests/conftest.py mutates sys.path and can mask a broken installed distribution; "
        f"offending lines: {violations}"
    )


def test_root_tests_do_not_import_ambiguous_conftest_module() -> None:
    """Root tests must consume shared state without bare ``conftest`` imports."""

    violations: list[str] = []
    for relative, tree in _parsed_root_test_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "conftest":
                violations.append(f"{relative}:{node.lineno}")
    assert not violations, (
        "Root tests import the ambiguous bare `conftest` module; use the session output "
        f"environment or a real helper module instead: {violations}"
    )


# r7 R77-3: the phantom `unregister_container` is dropped (no such API
# exists anywhere in the package — the registries deliberately have no
# unregister spelling) and `register_op_rule` joins: it writes the same
# class of process-global registry (_CUSTOM_OP_RULES) the conftest
# restore fixture governs.
_REGISTRY_MUTATOR_NAMES = frozenset({"register_container", "register_op_rule"})
"""Public registry mutators whose import-time call is an order-dependence bug."""


def _import_time_nodes(tree: ast.Module) -> list[ast.AST]:
    """Return every node that executes when the module is IMPORTED.

    Function/lambda bodies run only when called, so they are skipped -- but
    their decorators DO run at import and stay included. Class bodies execute
    at import and are walked.
    """

    nodes: list[ast.AST] = []
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(getattr(node, "decorator_list", []))
            continue
        nodes.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return nodes


def _facet_register_aliases(tree: ast.Module) -> set[str]:
    """Names under which the facet ``register`` decorator is imported bare."""

    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("facets"):
            for imported in node.names:
                if imported.name == "register":
                    aliases.add(imported.asname or imported.name)
    return aliases


def test_no_module_level_registry_mutation_in_tests() -> None:
    """Test modules must not mutate public registries at IMPORT time.

    A module-level ``@tl.facets.register`` or ``tl.register_container(...)``
    fires at pytest COLLECTION -- before any fixture can isolate it -- so a
    full collection left extra recipes/containers in the process-global
    registry for the whole session while a targeted run did not: the same
    trace hashed different recipe/provenance state depending on how pytest
    was invoked (hunt-b2-sol R76/R77). Register inside a restoring fixture.

    Covers all three call spellings (grind p5 §3.9: the original check saw
    only the ``x.facets.register`` decorator form): attribute decorators,
    BARE-NAME decorators (``from ...facets import register``), and
    import-time ``register_container``/``register_op_rule`` calls whether
    attribute-qualified or bare.
    """

    violations: list[str] = []
    # Session-cached parse (one ~5s tree parse per session, not per lint) with
    # the full three-spelling import-time coverage: attribute decorators,
    # bare-name aliases, and bare/qualified register_container calls.
    for relative, tree in _parsed_test_trees():
        if not relative.rsplit("/", 1)[-1].startswith("test"):
            continue
        facet_aliases = _facet_register_aliases(tree)
        for node in _import_time_nodes(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_violation = False
            if isinstance(func, ast.Attribute):
                if func.attr in _REGISTRY_MUTATOR_NAMES:
                    is_violation = True
                elif func.attr == "register":
                    base = func.value
                    if isinstance(base, ast.Attribute) and base.attr == "facets":
                        is_violation = True
            elif isinstance(func, ast.Name) and (
                func.id in _REGISTRY_MUTATOR_NAMES or func.id in facet_aliases
            ):
                is_violation = True
            if is_violation:
                violations.append(f"{relative}:{node.lineno}")
    assert not violations, (
        "import-time registry mutation (facet register / register_container) runs at "
        f"pytest collection; register inside a restoring fixture: {sorted(violations)}"
    )


# ---------------------------------------------------------------------------
# R77 round-3: strengthened teardown lint for long-lived Trace fixtures.
#
# The original scanner (`_module_trace_fixtures_without_yield`) has three
# proven evasion shapes: (a) fixtures nested inside test classes are invisible
# (it only reads `tree.body`); (b) a fixture that builds its Trace through a
# module-local helper (`make_trace()` -> `tl.trace(...)`) is invisible (only
# literal `trace(...)` calls are matched); (c) a fixture with a BARE trailing
# `yield` -- no statement after it and no try/finally -- passes despite having
# no teardown code at all. The scanner below closes all three. Indirection
# through helpers is resolved transitively but only within the SAME module;
# cross-module helper indirection is documented out of scope.
# ---------------------------------------------------------------------------


def _iter_scoped_fixture_functions(
    tree: ast.Module,
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Collect widely-scoped fixtures at module level and inside classes.

    Parameters
    ----------
    tree:
        Parsed test module.

    Returns
    -------
    list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]
        ``(qualified_name, function)`` pairs for every fixture whose scope is
        wider than per-test function scope, including fixtures nested inside
        (arbitrarily nested) test classes.
    """

    found: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(_is_module_scoped_fixture(item) for item in node.decorator_list):
                    found.append((f"{prefix}{node.name}", node))
            elif isinstance(node, ast.ClassDef):
                visit(node.body, f"{prefix}{node.name}.")

    visit(tree.body, "")
    return found


def _module_local_trace_helper_names(tree: ast.Module) -> set[str]:
    """Resolve module-level helper functions that (transitively) call trace.

    Parameters
    ----------
    tree:
        Parsed test module.

    Returns
    -------
    set[str]
        Names of module-level functions whose bodies reach a ``trace(...)`` /
        ``*.trace(...)`` call, directly or through other module-level helpers
        (fixed point within the module; cross-module helpers are out of scope).
    """

    module_functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    trace_callers = {name for name, node in module_functions.items() if _calls_trace(node)}
    changed = True
    while changed:
        changed = False
        for name, node in module_functions.items():
            if name in trace_callers:
                continue
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id in trace_callers
                ):
                    trace_callers.add(name)
                    changed = True
                    break
    return trace_callers


def _calls_trace_or_local_helper(
    function: ast.FunctionDef | ast.AsyncFunctionDef, helper_names: set[str]
) -> bool:
    """Return whether a fixture reaches a trace call directly or via helpers.

    Parameters
    ----------
    function:
        Fixture function syntax node.
    helper_names:
        Module-level helper functions known to (transitively) call trace.

    Returns
    -------
    bool
        Whether the fixture creates a Trace through any in-module path.
    """

    if _calls_trace(function):
        return True
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in helper_names
        ):
            return True
    return False


def _block_has_unguarded_yield(statements: list[ast.stmt], guarded: bool) -> bool:
    """Return whether any yield in a statement block lacks a teardown path.

    A yield is *guarded* when some enclosing block (within the fixture) has at
    least one statement after the statement containing it, or when it sits
    inside the body/handlers/orelse of a ``try`` with a ``finally`` clause.
    Yields inside nested function/class definitions belong to those objects,
    not to the fixture, and are skipped.

    Parameters
    ----------
    statements:
        Statement block to scan.
    guarded:
        Whether an enclosing construct already guarantees teardown.

    Returns
    -------
    bool
        Whether an unguarded (teardown-free) yield exists in the block.
    """

    def _is_noop(stmt: ast.stmt) -> bool:
        # r7 R77 (sol b2): `yield log; pass` (or a trailing docstring/ellipsis)
        # is NOT a teardown path -- only a meaningful statement counts.
        return isinstance(stmt, ast.Pass) or (
            isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
        )

    for index, statement in enumerate(statements):
        followed = guarded or any(not _is_noop(later) for later in statements[index + 1 :])
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(statement, ast.Try):
            inner = followed or bool(statement.finalbody)
            blocks = [statement.body, statement.orelse]
            blocks.extend(handler.body for handler in statement.handlers)
            if any(_block_has_unguarded_yield(block, inner) for block in blocks):
                return True
            if _block_has_unguarded_yield(statement.finalbody, followed):
                return True
            continue
        nested_blocks = [
            value
            for _field, value in ast.iter_fields(statement)
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt)
        ]
        if nested_blocks:
            if any(_block_has_unguarded_yield(block, followed) for block in nested_blocks):
                return True
            continue
        if not followed and any(
            isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(statement)
        ):
            return True
    return False


def _fixture_has_real_teardown(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a fixture owns an actual teardown path, not a bare yield.

    Parameters
    ----------
    function:
        Fixture function syntax node.

    Returns
    -------
    bool
        ``True`` when the fixture yields with at least one statement after the
        yield (or a try/finally around it), or registers a finalizer through
        ``request.addfinalizer(...)``.
    """

    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "addfinalizer"
        ):
            # r7 R77 (sol b2): `addfinalizer(lambda: None)` is a lint dodge,
            # not a teardown; a literal no-op lambda does not count.
            argument = node.args[0] if node.args else None
            if isinstance(argument, ast.Lambda) and isinstance(argument.body, ast.Constant):
                continue
            return True
    has_yield = _block_contains_yield(function.body)
    return has_yield and not _block_has_unguarded_yield(function.body, False)


def _block_contains_yield(statements: list[ast.stmt]) -> bool:
    """Return whether a block yields, ignoring nested function/class bodies.

    Parameters
    ----------
    statements:
        Statement block to scan.

    Returns
    -------
    bool
        Whether the block contains a yield belonging to the enclosing function.
    """

    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        nested_blocks = [
            value
            for _field, value in ast.iter_fields(statement)
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt)
        ]
        if isinstance(statement, ast.Try):
            nested_blocks.extend(handler.body for handler in statement.handlers)
        if nested_blocks:
            if any(_block_contains_yield(block) for block in nested_blocks):
                return True
            continue
        if any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(statement)):
            return True
    return False


def _strict_trace_fixture_violations_in_source(source: str, label: str) -> list[str]:
    """Scan one module's source for teardown-free long-lived Trace fixtures.

    Parameters
    ----------
    source:
        Python source text of a test module.
    label:
        Stable label (relative path) used in violation strings.

    Returns
    -------
    list[str]
        ``label::qualified_fixture_name`` violations.
    """

    tree = ast.parse(source, filename=label)
    return _strict_trace_fixture_violations_in_tree(tree, label)


def _strict_trace_fixture_violations_in_tree(tree: ast.Module, label: str) -> list[str]:
    """Tree-level core of the strict scanner (shared with the cached walk)."""

    helper_names = _module_local_trace_helper_names(tree)
    violations: list[str] = []
    for qualified_name, function in _iter_scoped_fixture_functions(tree):
        if not _calls_trace_or_local_helper(function, helper_names):
            continue
        if not _fixture_has_real_teardown(function):
            violations.append(f"{label}::{qualified_name}")
    return violations


def _strict_trace_fixture_violations(tests_root: Path) -> list[str]:
    """Scan the whole test suite for teardown-free long-lived Trace fixtures.

    Parameters
    ----------
    tests_root:
        Root of the test suite.

    Returns
    -------
    list[str]
        Stable ``path::fixture`` violations.
    """

    del tests_root
    violations: list[str] = []
    for relative, tree in _parsed_test_trees():
        violations.extend(_strict_trace_fixture_violations_in_tree(tree, relative))
    return violations


def test_scoped_trace_fixtures_have_real_teardown() -> None:
    """Widely-scoped Trace fixtures need actual teardown code, however nested."""

    tests_root = Path(__file__).resolve().parent
    violations = _strict_trace_fixture_violations(tests_root)
    assert not violations, (
        "Widely-scoped fixtures create live Traces without a real teardown path "
        "(class-nested fixtures, helper-built traces, and bare trailing yields "
        "all count). Yield the Trace and clean up after the yield (or in a "
        "try/finally):\n  " + "\n  ".join(violations)
    )


_EVASION_CLASS_NESTED = """
import pytest
import torchlens as tl

class TestGroup:
    @pytest.fixture(scope="module")
    def cached_trace(self):
        return tl.trace(model, x)
"""

_EVASION_HELPER_INDIRECTION = """
import pytest
import torchlens as tl

def make_trace():
    return tl.trace(model, x)

def build_log():
    return make_trace()

@pytest.fixture(scope="module")
def cached_trace():
    return build_log()
"""

_EVASION_BARE_YIELD = """
import pytest
import torchlens as tl

@pytest.fixture(scope="session")
def cached_trace():
    yield tl.trace(model, x)
"""

# r7 R77 (sol b2): the two lint-dodge shapes the old predicate accepted --
# syntax after the yield that does nothing, and a literal no-op finalizer.
_EVASION_NOOP_AFTER_YIELD = """
import pytest
import torchlens as tl

@pytest.fixture(scope="session")
def cached_trace():
    log = tl.trace(model, x)
    yield log
    pass
"""

_EVASION_NOOP_FINALIZER = """
import pytest
import torchlens as tl

@pytest.fixture(scope="session")
def cached_trace(request):
    request.addfinalizer(lambda: None)
    yield tl.trace(model, x)
"""

_COMPLIANT_STATEMENT_AFTER_YIELD = """
import pytest
import torchlens as tl

@pytest.fixture(scope="module")
def cached_trace():
    log = tl.trace(model, x)
    yield log
    log.cleanup()
"""

_COMPLIANT_TRY_FINALLY = """
import pytest
import torchlens as tl

def make_trace():
    return tl.trace(model, x)

class TestGroup:
    @pytest.fixture(scope="class")
    def cached_trace(self):
        log = make_trace()
        try:
            yield log
        finally:
            log.cleanup()
"""

_COMPLIANT_FUNCTION_SCOPE_BARE_YIELD = """
import pytest
import torchlens as tl

@pytest.fixture()
def per_test_trace():
    yield tl.trace(model, x)
"""


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        pytest.param(
            _EVASION_CLASS_NESTED, ["planted.py::TestGroup.cached_trace"], id="class-nested"
        ),
        pytest.param(
            _EVASION_HELPER_INDIRECTION, ["planted.py::cached_trace"], id="helper-indirection"
        ),
        pytest.param(_EVASION_BARE_YIELD, ["planted.py::cached_trace"], id="bare-yield"),
        pytest.param(
            _EVASION_NOOP_AFTER_YIELD, ["planted.py::cached_trace"], id="noop-after-yield"
        ),
        pytest.param(_EVASION_NOOP_FINALIZER, ["planted.py::cached_trace"], id="noop-finalizer"),
        pytest.param(_COMPLIANT_STATEMENT_AFTER_YIELD, [], id="ok-statement-after-yield"),
        pytest.param(_COMPLIANT_TRY_FINALLY, [], id="ok-try-finally-class-nested-helper"),
        pytest.param(_COMPLIANT_FUNCTION_SCOPE_BARE_YIELD, [], id="ok-function-scope"),
    ],
)
def test_strict_trace_fixture_scanner_is_red_capable(snippet: str, expected: list[str]) -> None:
    """The strengthened scanner catches each proven evasion shape exactly.

    Red-capability proof for the three R77 evasions: (a) class-nested
    fixtures, (b) one-or-more-hop in-module helper indirection to the trace
    call, (c) a bare trailing yield with no teardown statement. The compliant
    shapes prove the scanner does not overfire.
    """

    assert _strict_trace_fixture_violations_in_source(snippet, "planted.py") == expected


def _lru_cached_functions(package_root: Path) -> dict[tuple[str, str], str]:
    """Collect ``lru_cache``/``cache``-decorated module functions and their source.

    Parameters
    ----------
    package_root:
        Root of the ``torchlens`` package.

    Returns
    -------
    dict[tuple[str, str], str]
        ``(module_name, function_name)`` -> function source segment.
    """

    cached: dict[tuple[str, str], str] = {}
    for module_name, source, tree in _parsed_package_trees(package_root):
        for statement in tree.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorated = ast.unparse(statement.decorator_list) if statement.decorator_list else ""
            if "lru_cache" in decorated or "functools.cache" in decorated:
                cached[(module_name, statement.name)] = (
                    ast.get_source_segment(source, statement) or ""
                )
    return cached


def test_capability_dependent_caches_are_cleared() -> None:
    """Every probe-derived lru_cache must be in the conftest clear list.

    Restoring the lazy ``HAS_*`` capability latches un-poisons the b7fe953e
    incident class at the first layer only: an ``lru_cache`` whose value was
    computed FROM a poisoned probe keeps the poisoned result for the process
    (grind p5 §3.9, the class one layer down). This census flags every
    module-level cached function that references ``_torch_compat`` / a
    ``HAS_*`` flag -- directly or through another flagged cache -- and
    requires it in ``tests/conftest.py::_CAPABILITY_DEPENDENT_CACHES`` so the
    autouse probe-restore clears it. Deliberately process-frozen caches
    (torch-version-fixed inventories) do not reference probes and stay out.
    """

    import re

    from tests.conftest import _CAPABILITY_DEPENDENT_CACHES

    package_root = Path(__file__).resolve().parents[1] / "torchlens"
    cached = _lru_cached_functions(package_root)
    probe_pattern = re.compile(r"_torch_compat|\bHAS_[A-Z_]+\b")
    dependent: set[tuple[str, str]] = {
        key for key, body in cached.items() if probe_pattern.search(body)
    }
    # Fixpoint: a cache calling another dependent cache is dependent too.
    while True:
        names = {name for _, name in dependent}
        grown = dependent | {
            key
            for key, body in cached.items()
            if key not in dependent and any(re.search(rf"\b{name}\s*\(", body) for name in names)
        }
        if grown == dependent:
            break
        dependent = grown
    declared = set(_CAPABILITY_DEPENDENT_CACHES)
    assert dependent <= declared, (
        "lru_cached functions derive from a capability probe but are missing from "
        "tests/conftest.py::_CAPABILITY_DEPENDENT_CACHES (the probe restore cannot "
        f"clear them): {sorted(dependent - declared)}"
    )
    assert declared <= set(cached), (
        "stale _CAPABILITY_DEPENDENT_CACHES rows (no such cached function): "
        f"{sorted(declared - set(cached))}"
    )


def test_capability_dependent_cache_clear_actually_clears() -> None:
    """The conftest clear helper empties every declared probe-derived cache."""

    import importlib

    from tests.conftest import _CAPABILITY_DEPENDENT_CACHES, _clear_capability_dependent_caches

    primed = []
    for module_name, attr in _CAPABILITY_DEPENDENT_CACHES:
        function = getattr(importlib.import_module(module_name), attr)
        function()  # prime
        assert function.cache_info().currsize >= 1
        primed.append(function)
    _clear_capability_dependent_caches()
    for function in primed:
        assert function.cache_info().currsize == 0, f"{function} survived the probe restore"


def test_usage_stats_gate_arms_on_documented_backstop_spellings() -> None:
    """Every documented broad tier spelling arms the ArgSpec usage gate.

    The predicate compared literal markexpr strings, so the DOCUMENTED
    phase-boundary spelling `-m "not rare and not slow"` and the mid
    backstop silently disarmed the usage-coverage audit while the code's own
    comment says any broad subset is sound (b2 R41/B2R5-15, proven twice).
    """

    from types import SimpleNamespace

    from tests.conftest import TESTS_DIR, _is_full_usage_stats_run

    def config(markexpr: str, keyword: str = "", args: list[str] | None = None):
        return SimpleNamespace(
            option=SimpleNamespace(keyword=keyword, markexpr=markexpr),
            args=args if args is not None else [str(TESTS_DIR)],
        )

    for spelling in (
        "",
        "not rare",
        "not slow and not rare",
        "not rare and not slow",
        "not rare and not slow and not heavy",
        "(not rare) and (not slow)",
    ):
        assert _is_full_usage_stats_run(config(spelling)), spelling
    for narrowed in ("smoke", "slow", "not rare and heavy", "rare"):
        assert not _is_full_usage_stats_run(config(narrowed)), narrowed
    assert not _is_full_usage_stats_run(config("", keyword="foo"))
    assert not _is_full_usage_stats_run(config("", args=[str(Path(TESTS_DIR) / "sub")]))


def test_serial_marker_is_not_a_budget_exemption() -> None:
    """A serial item resolves its tier budget; only slow/rare stay exempt.

    ``serial`` formerly returned ``None`` before tier resolution, so any
    unmarked test dodged the 5s partition boundary by adding one decorator,
    and heavy+serial items enforced their 20s intent at nothing (b2 R41
    round 5).
    """

    from tests.conftest import (
        HEAVY_DURATION_BUDGET_SECONDS,
        SMOKE_DURATION_BUDGET_SECONDS,
        _duration_budget_tier,
    )

    class _FakeItem:
        def __init__(self, markers: set[str]) -> None:
            self._markers = markers

        def get_closest_marker(self, name: str):
            return object() if name in self._markers else None

    assert _duration_budget_tier(_FakeItem({"serial"})) == (
        "unmarked",
        SMOKE_DURATION_BUDGET_SECONDS,
    )
    assert _duration_budget_tier(_FakeItem({"serial", "heavy"})) == (
        "heavy",
        HEAVY_DURATION_BUDGET_SECONDS,
    )
    assert _duration_budget_tier(_FakeItem({"serial", "slow"})) is None
    assert _duration_budget_tier(_FakeItem({"rare"})) is None


# ---------------------------------------------------------------------------
# Marker ALGEBRA lockstep (megasprint P03, compo memo row 0.1 / conflict rule
# 16). The algebra -- every marker's one role, incl. the ORTHOGONAL selection
# markers `compo` and `real_model` -- is declared ONCE in
# tests/composition_expectations/marker_algebra.py. This lint holds pyproject
# and the algebra in lockstep BOTH directions and pins the orthogonality
# contract: selection markers never join tier resolution, never exempt a
# budget, and never conflict with any tier.
# ---------------------------------------------------------------------------


def _pyproject_declared_markers() -> dict[str, str]:
    """Parse ``[tool.pytest.ini_options] markers`` into name -> description.

    Line/regex-based on purpose: ``tomllib`` is 3.11+ and the suite's floor
    row runs 3.10 (the test_ci_packaging_gates / coverage-floor idiom).
    """

    import re

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    lines = pyproject.read_text(encoding="utf-8").splitlines()
    start = lines.index("markers = [")
    declared: dict[str, str] = {}
    for line in lines[start + 1 :]:
        if line.strip() == "]":
            break
        match = re.match(r'\s*"([A-Za-z0-9_]+):\s*(.*)",\s*$', line)
        if match is not None:
            declared[match.group(1)] = match.group(2)
    assert declared, "pyproject markers block parsed empty -- the regex idiom drifted"
    return declared


def test_marker_algebra_lockstep_with_pyproject() -> None:
    """Every pyproject marker has exactly one algebra role, and vice versa."""

    from tests.composition_expectations.marker_algebra import (
        MARKER_ALGEBRA,
        algebra_violations,
    )

    violations = algebra_violations(_pyproject_declared_markers(), MARKER_ALGEBRA)
    assert not violations, (
        "pyproject markers and the marker algebra drifted (the algebra is "
        "declared ONCE; land both sides in one change):\n  " + "\n  ".join(violations)
    )


def test_marker_algebra_lockstep_is_red_capable() -> None:
    """A planted undeclared/unroled/unknown-role marker each goes red."""

    from tests.composition_expectations.marker_algebra import algebra_violations

    algebra = {"smoke": "tier", "compo": "selection"}
    declared = {"smoke": "d", "compo": "d"}
    assert algebra_violations(declared, algebra) == []
    assert any(
        "no role in the marker algebra" in violation
        for violation in algebra_violations({**declared, "planted": "d"}, algebra)
    )
    assert any(
        "no pyproject declaration" in violation
        for violation in algebra_violations(declared, {**algebra, "ghost": "tier"})
    )
    assert any(
        "unknown role" in violation
        for violation in algebra_violations(declared, {**algebra, "compo": "vibe"})
    )


def test_selection_markers_are_orthogonal_to_every_tier() -> None:
    """`compo`/`real_model` combine with any tier without a combo violation.

    Orthogonality is the row-0.1 contract: selection markers are selectors,
    never cost tiers, so no tier combination may be flagged and the smoke
    incompatibility set may never absorb them (that would silently turn a
    selector into a tier exemption channel).
    """

    from tests.composition_expectations.marker_algebra import (
        SELECTION_MARKERS,
        TIER_MARKERS,
    )

    assert {"compo", "real_model"} == SELECTION_MARKERS
    assert {"smoke", "heavy", "slow"} == TIER_MARKERS
    assert not SELECTION_MARKERS & set(_SMOKE_INCOMPATIBLE_MARKERS)
    for selection in SELECTION_MARKERS:
        for tier in TIER_MARKERS:
            assert _tier_combo_violations({selection, tier}, set(), "planted::node") == []
        # Selection markers also compose with each other and alone.
        assert _tier_combo_violations(SELECTION_MARKERS | {"smoke"}, set(), "planted::node") == []


def test_selection_markers_do_not_alter_tier_resolution() -> None:
    """Tier/budget resolution is blind to selection markers (never an exemption).

    A `compo`-marked test resolves exactly as its tier markers dictate: alone
    it is `unmarked` (5s budget), with `heavy` it is heavy (20s), and only the
    contract-exempt tiers (`slow`/`rare`) return None -- adding a selection
    marker can never disarm the duration budget the way the pre-fix `serial`
    marker could.
    """

    from tests.composition_expectations.marker_algebra import SELECTION_MARKERS
    from tests.conftest import (
        HEAVY_DURATION_BUDGET_SECONDS,
        SMOKE_DURATION_BUDGET_SECONDS,
        _duration_budget_tier,
    )

    class _FakeItem:
        def __init__(self, markers: set[str]) -> None:
            self._markers = markers

        def get_closest_marker(self, name: str):
            return object() if name in self._markers else None

    for selection in SELECTION_MARKERS:
        assert _duration_budget_tier(_FakeItem({selection})) == (
            "unmarked",
            SMOKE_DURATION_BUDGET_SECONDS,
        )
        assert _duration_budget_tier(_FakeItem({selection, "smoke"})) == (
            "smoke",
            SMOKE_DURATION_BUDGET_SECONDS,
        )
        assert _duration_budget_tier(_FakeItem({selection, "heavy"})) == (
            "heavy",
            HEAVY_DURATION_BUDGET_SECONDS,
        )
        assert _duration_budget_tier(_FakeItem({selection, "slow"})) is None


def test_sessionfinish_budget_tripwire_flips_exit_status(
    request: pytest.FixtureRequest,
) -> None:
    """r7 R41 (sol b2 HIGH): the budget tripwire is always-on, not collectable.

    Enforcement used to live only in this file's assertion tests, so any
    targeted run that did not collect ``test_marker_lint.py`` exited green
    over budget. The conftest ``pytest_sessionfinish`` path must flip a green
    session to failing whenever the offender ledger (or a family aggregate)
    is non-empty -- and must leave already-failing or clean sessions alone.

    The enforcement helper is resolved from the LIVE registered conftest
    plugin (a bare ``conftest`` import is ambiguous during full-suite
    collection; ``tests`` is not a package), which also pins that the hook
    really is loaded in every tests/-scoped session.
    """

    from types import SimpleNamespace

    conftest_plugin = next(
        (
            plugin
            for plugin in request.config.pluginmanager.get_plugins()
            if hasattr(plugin, "_enforce_duration_budget_at_sessionfinish")
        ),
        None,
    )
    assert conftest_plugin is not None, (
        "tests/conftest.py no longer registers the sessionfinish duration-"
        "budget enforcement helper -- the always-on tripwire is gone"
    )
    _enforce_duration_budget_at_sessionfinish = (
        conftest_plugin._enforce_duration_budget_at_sessionfinish
    )

    def _fake_session(**attrs: object) -> SimpleNamespace:
        plugin_manager = SimpleNamespace(get_plugin=lambda name: None)
        return SimpleNamespace(
            config=SimpleNamespace(pluginmanager=plugin_manager),
            exitstatus=0,
            **attrs,
        )

    # Per-item offender on a green session -> forced failure.
    session = _fake_session(
        _tl_duration_budget_offenders=[("tests/x.py::test_slow", "smoke", 9.0, 8.0, 7.0)]
    )
    _enforce_duration_budget_at_sessionfinish(session, 0)  # type: ignore[arg-type]
    assert session.exitstatus == 1

    # Family aggregate offender alone -> forced failure.
    session = _fake_session(
        _tl_smoke_family_stats={"tests/x.py::fam": (30.0, 4)},
        _tl_smoke_family_budgets={"tests/x.py::fam": 12.0},
    )
    _enforce_duration_budget_at_sessionfinish(session, 0)  # type: ignore[arg-type]
    assert session.exitstatus == 1

    # Clean session stays green; failing session is left alone (the
    # marker-lint assertion or an ordinary failure already owns the status).
    session = _fake_session()
    _enforce_duration_budget_at_sessionfinish(session, 0)  # type: ignore[arg-type]
    assert session.exitstatus == 0
    session = _fake_session(
        _tl_duration_budget_offenders=[("tests/x.py::test_slow", "smoke", 9.0, 8.0, 7.0)]
    )
    _enforce_duration_budget_at_sessionfinish(session, 1)  # type: ignore[arg-type]
    assert session.exitstatus == 0
