"""THE marker algebra -- declared ONCE, here (conflict rule 16).

Every pytest marker the suite declares in ``pyproject.toml`` has exactly one
ROLE in this closed algebra, and ``tests/test_marker_lint.py`` enforces the
lockstep both directions: a marker declared in pyproject without a role here
(or a role here without a pyproject declaration) is a red gate. Any lane that
wants a new marker lands it in BOTH places in one change, through this file's
owner.

Roles
-----
- ``tier``: exactly-one cost classification per test (``smoke`` < 5s,
  ``heavy`` 5-20s, ``slow`` > 20s; unmarked resolves to the smoke budget).
  Tiers are mutually exclusive on one resolved item (marker-lint policy).
- ``scheduling``: run-placement channels (``serial``: away from parallel
  load; ``rare``: excluded unless explicitly requested, budget-exempt).
- ``capability``: the test requires an optional dependency/runtime and skips
  honestly without it.
- ``suite``: names a dedicated gate family selected as a unit.
- ``tier_selector``: names WHICH items of a function, class or module get a
  tier. ``smoke_cells("test_x[a]", ...)`` applies ``smoke`` to exactly the
  named items at collection (``tests/conftest.py::pytest_itemcollected``), so
  a parametrized family keeps one or two representative smoke cells. The
  applied ``smoke`` is an ordinary tier mark: every tier rule applies to it.
- ``selection``: ORTHOGONAL selectors, never cost tiers (compo memo,
  build row 0.1). A selection marker composes with ANY tier, with
  ``scheduling``/``capability`` markers, and with other selection markers;
  it never alters tier resolution and never exempts a test from its
  duration budget. Every test carrying one still keeps exactly one
  smoke/heavy/slow classification (or stays unmarked).

The two selection markers this sprint adds:

- ``compo``: the test is a composition-expectation ledger row (consumed by
  ``-m compo`` sweeps and the pair-coverage auditor).
- ``real_model``: the test exercises a real-architecture or real-checkpoint
  fixture (the R0 gates select ``tests/real_model/r0`` by path).
"""

from __future__ import annotations

MARKER_ROLES = ("tier", "tier_selector", "scheduling", "capability", "suite", "selection")

#: Every declared marker -> its one role. Lockstep-gated against pyproject.
MARKER_ALGEBRA: dict[str, str] = {
    "smoke": "tier",
    "heavy": "tier",
    "slow": "tier",
    "smoke_cells": "tier_selector",
    "serial": "scheduling",
    "rare": "scheduling",
    "optional": "capability",
    "requires_assertions": "capability",
    "backend_parity": "suite",
    "backend_jax": "capability",
    "backend_mlx": "capability",
    "backend_tinygrad": "capability",
    "backend_paddle": "capability",
    "tf_backend": "capability",
    "compo": "selection",
    "real_model": "selection",
}

TIER_MARKERS = frozenset(name for name, role in MARKER_ALGEBRA.items() if role == "tier")
SELECTION_MARKERS = frozenset(name for name, role in MARKER_ALGEBRA.items() if role == "selection")


def algebra_violations(declared: dict[str, str], algebra: dict[str, str]) -> list[str]:
    """Return the lockstep violations between pyproject markers and the algebra.

    Pure helper (red-capability-testable without planting real markers).

    Parameters
    ----------
    declared:
        Marker name -> description, parsed from ``pyproject.toml``.
    algebra:
        Marker name -> role (normally :data:`MARKER_ALGEBRA`).

    Returns
    -------
    list[str]
        Human-readable violations; empty when the two declarations agree.
    """

    violations = []
    for name in sorted(set(declared) - set(algebra)):
        violations.append(
            f"marker {name!r} is declared in pyproject but has no role in the "
            "marker algebra (tests/composition_expectations/marker_algebra.py); "
            "the algebra is declared ONCE -- add the role there in the same change"
        )
    for name in sorted(set(algebra) - set(declared)):
        violations.append(
            f"marker {name!r} has an algebra role but no pyproject declaration; "
            "declare it under [tool.pytest.ini_options] markers in the same change"
        )
    for name, role in sorted(algebra.items()):
        if role not in MARKER_ROLES:
            violations.append(
                f"marker {name!r} carries unknown role {role!r} (closed: {MARKER_ROLES})"
            )
    return violations
