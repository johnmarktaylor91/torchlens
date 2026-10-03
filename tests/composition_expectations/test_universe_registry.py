"""Universe registry gates: printed cardinalities, drift lockstep, adequacy plants.

Compo memo row 0.2. The registry is the closure object (D1): every universe
prints its count and fails on unexplained drift; adequacy plants prove each
``planted`` derivation catches a member the weaker derivation misses (the
error-contract gate's blindness survived exactly as long as nobody printed
the second count -- memo 5.1).
"""

from __future__ import annotations

import ast
import textwrap

import pytest

from tests.composition_expectations import _censuses
from tests.composition_expectations.universes import (
    ADEQUACY_STATUSES,
    UNIVERSES,
    baseline_cardinalities,
    derived_cardinalities,
)

pytestmark = pytest.mark.compo


def _parse(module: str, source: str) -> dict[str, ast.Module]:
    return {module: ast.parse(textwrap.dedent(source))}


def test_universe_descriptors_are_well_formed() -> None:
    """Unique ids, closed adequacy vocabulary, owners, and OPEN coherence."""

    ids = [universe.universe_id for universe in UNIVERSES]
    assert len(ids) == len(set(ids)), "duplicate universe ids"
    for universe in UNIVERSES:
        assert universe.universe_id.startswith("U-")
        assert universe.owner, universe.universe_id
        assert universe.derivation, universe.universe_id
        assert universe.public_boundary, universe.universe_id
        assert universe.review_trigger, universe.universe_id
        assert universe.adequacy_status in ADEQUACY_STATUSES, universe.universe_id
        if universe.adequacy_status == "open":
            assert universe.derive is None, (
                f"{universe.universe_id} is OPEN but carries a derivation"
            )
        else:
            assert universe.derive is not None, (
                f"{universe.universe_id} claims adequacy {universe.adequacy_status!r} "
                "without a derivation"
            )
        if universe.independent_census == "single_derivation":
            assert universe.adequacy_status in {"open", "single_derivation"}, (
                f"{universe.universe_id}: a planted status requires an independent census "
                "that can see the plant (memo Dis-3, the review's enforceable form)"
            )


def test_universe_cardinalities_print_and_lockstep() -> None:
    """Every universe's live count matches the committed baseline EXACTLY.

    The counts are printed (the memo's 'printed cardinality' rule: blindness
    survives exactly as long as nobody prints the number). Deliberate changes
    update ``data/universe_cardinalities.tsv`` in the same change; new
    diagnostic sites are classified in S-17/S-18 in that same change.
    """

    live = derived_cardinalities()
    baseline = baseline_cardinalities()
    print("\nuniverse cardinalities (live):")
    for universe_id, count in live.items():
        print(f"  {universe_id}\t{count}")
    assert set(live) == set(baseline), (
        "universe registry rows and the cardinality baseline drifted: "
        f"missing={sorted(set(live) - set(baseline))} "
        f"stale={sorted(set(baseline) - set(live))}"
    )
    drifted = {
        universe_id: (baseline[universe_id], str(count))
        for universe_id, count in live.items()
        if str(count) != baseline[universe_id]
    }
    assert not drifted, (
        "universe cardinalities drifted (baseline, live). If the change is deliberate, "
        "update tests/composition_expectations/data/universe_cardinalities.tsv in the "
        f"same change and classify any new S-17/S-18 sites: {drifted}"
    )


def test_every_declared_universe_has_an_owner_and_no_silent_absence() -> None:
    """OPEN universes are counted as missing, never silently absent (D10)."""

    open_rows = [u.universe_id for u in UNIVERSES if u.adequacy_status == "open"]
    derived_rows = [u.universe_id for u in UNIVERSES if u.derive is not None]
    print(f"\nuniverse registry: {len(derived_rows)} derived, {len(open_rows)} OPEN")
    assert derived_rows, "no derived universes -- the registry lost its substance"
    # The memo's 3.2 roster must stay represented: every roster family below
    # has at least one registry row, derived or OPEN.
    represented = {u.universe_id for u in UNIVERSES}
    for required in (
        "U-BACKENDS",
        "U-PUBLIC-OPERATIONS",
        "U-CAPTURE-OPTIONS",
        "U-RAISE-SITES",
        "U-WARN-SITES",
        "U-ENV-VARS",
        "U-PRODUCT-STATES",
        "U-EDIT-VERBS",
        "U-RENDER-SURFACES",
        "U-MODEL-TRAITS",
        "U-TAUGHT-PATHS",
        "U-DEPRECATED-ALIASES",
        "U-FOREIGN-APIS",
        "U-PROSE-CLAIMS",
        "U-UNTRUSTED-ENTRY",
        "U-GLOSSARY-TERMS",
    ):
        assert required in represented, f"memo 3.2 roster row {required} lost its registry row"


# ---------------------------------------------------------------------------
# Adequacy plants (memo 3.2): each `planted` universe proves its derivation
# catches a member the WEAKER derivation misses. Where no independent census
# exists, the limitation is declared, never theatre-proved.
# ---------------------------------------------------------------------------


def test_raise_site_factory_closure_plant() -> None:
    """S-17 plant: a factory-raised error is invisible to the direct scan.

    The planted module raises through an always-error-returning factory
    (the ``_undispatched_capability_error`` shape). The direct scan MUST
    miss it and the closure derivation MUST find it with the constructed
    class and factory provenance -- proving the closure is load-bearing,
    not decoration (the largest class was undercounted 27% without it).
    """

    trees = _parse(
        "planted.module",
        """
        class PlantedError(Exception):
            pass

        def _make_planted_error(reason):
            return PlantedError(reason, code="planted_code")

        def guard(value):
            if value < 0:
                raise _make_planted_error("negative")
        """,
    )
    direct = _censuses.raise_sites_in_tree(trees, factory_closure=False)
    closed = _censuses.raise_sites_in_tree(trees, factory_closure=True)
    assert [site.error_class for site in direct] == [], "direct scan saw the factory site"
    assert [(s.error_class, s.via_factory, s.has_code) for s in closed] == [
        ("PlantedError", "_make_planted_error", False)
    ]


def test_raise_site_factory_closure_resolves_fixpoint() -> None:
    """A factory returning another factory's call still resolves (fixpoint)."""

    trees = _parse(
        "planted.module",
        """
        class DeepError(Exception):
            pass

        def _inner(reason):
            return DeepError(reason)

        def _outer(reason):
            return _inner(reason)

        def guard():
            raise _outer("boom")
        """,
    )
    closed = _censuses.raise_sites_in_tree(trees, factory_closure=True)
    assert [(s.error_class, s.via_factory) for s in closed] == [("DeepError", "_outer")]


def test_raise_site_direct_code_kwarg_is_seen() -> None:
    """Coded direct raises census as coded; uncoded as uncoded."""

    trees = _parse(
        "planted.module",
        """
        class AError(Exception):
            pass

        def f():
            raise AError("x", code="a_code")

        def g():
            raise AError("y")
        """,
    )
    sites = _censuses.raise_sites_in_tree(trees)
    assert [(s.scope, s.has_code) for s in sites] == [("f", True), ("g", False)]


def test_warn_site_alias_plant() -> None:
    """S-18 plant: aliased warn imports are invisible to the literal scan.

    Both alias shapes (module alias, bare-name import) MUST be missed by the
    literal ``warnings.warn`` scan and found by the alias-resolving census --
    the adequacy plant IS an aliased site (memo 3.5).
    """

    trees = _parse(
        "planted.module",
        """
        import warnings
        import warnings as _w
        from warnings import warn as _bare

        def literal():
            warnings.warn("seen by both")

        def module_alias():
            _w.warn("aliased module")

        def bare_name():
            _bare("aliased bare")
        """,
    )
    literal = _censuses.warn_sites_in_tree(trees, alias_resolving=False)
    resolved = _censuses.warn_sites_in_tree(trees, alias_resolving=True)
    assert [site.scope for site in literal] == ["literal"]
    assert sorted(site.scope for site in resolved) == ["bare_name", "literal", "module_alias"]
    assert sorted(site.scope for site in resolved if site.aliased) == ["bare_name", "module_alias"]


def test_warn_site_plant_is_live_in_the_real_tree() -> None:
    """The live tree CONTAINS aliased warn sites: the plant is real, not staged.

    If a refactor ever removes every aliased site, this assertion (not the
    census) goes red -- the honest act then is re-planting via the snippet
    plant above and recording the tree change, never deleting the census.
    """

    aliased = [site for site in _censuses.warn_sites() if site.aliased]
    print("\nlive aliased warn sites:")
    for site in aliased:
        print(f"  {site.site_key}")
    assert aliased, (
        "the live tree lost its aliased warn sites; the alias-resolving census "
        "can no longer be proven load-bearing against the real tree -- update "
        "this gate deliberately (the snippet plant still proves the mechanism)"
    )


def test_warn_site_code_kwarg_is_seen() -> None:
    """A warn call constructing an instance with ``code=`` censuses as coded."""

    trees = _parse(
        "planted.module",
        """
        import warnings

        class PlantedWarning(UserWarning):
            pass

        def coded():
            warnings.warn(PlantedWarning("x", code="planted_warning"))

        def uncoded():
            warnings.warn("bare string")
        """,
    )
    sites = _censuses.warn_sites_in_tree(trees)
    by_scope = {site.scope: site for site in sites}
    assert by_scope["coded"].has_code and by_scope["coded"].category == "PlantedWarning"
    assert not by_scope["uncoded"].has_code


def test_capture_options_private_filter_both_directions() -> None:
    """The options private filter is explicit and fail-closed BOTH ways.

    Direction 1: a private-named field OUTSIDE the declared filter refuses
    (the census cannot silently narrow). Direction 2: a declared filter row
    the class lacks refuses (a stale filter cannot hide a removed field).
    """

    import dataclasses

    @dataclasses.dataclass
    class PlantedOptions:
        visible: int = 0
        _hidden: int = 0

    with pytest.raises(AssertionError, match="_hidden"):
        _censuses.public_option_fields(PlantedOptions, frozenset())
    with pytest.raises(AssertionError, match="stale"):
        _censuses.public_option_fields(PlantedOptions, frozenset({"_hidden", "_gone"}))
    assert _censuses.public_option_fields(PlantedOptions, frozenset({"_hidden"})) == ("visible",)
    # And the live class resolves under the declared filter.
    # 48 -> 49 (T82d re-reconcile, F44): union of the landed
    # track_device_memory and the F44 log_injections options. As of T85
    # (F01-AMENDED landed) both parents carry log_injections; pins agree.
    assert len(_censuses.capture_option_fields()) == 49


def test_env_var_census_shapes() -> None:
    """All three env-read shapes census; prose mentions do not."""

    trees = _parse(
        "planted.module",
        """
        import os
        from torchlens.utils.env_flags import closed_bool_env

        _AUDIT_ENV = "TORCHLENS_PLANTED_CONSTANT"

        def reads():
            a = os.environ.get("TORCHLENS_PLANTED_GET")
            b = os.environ["TORCHLENS_PLANTED_SUBSCRIPT"]
            c = os.getenv("TORCHLENS_PLANTED_GETENV")
            d = closed_bool_env("TORCHLENS_PLANTED_BOOL")
            return a, b, c, d

        def prose_only():
            return "set TORCHLENS_PLANTED_PROSE to enable"
        """,
    )
    names = _censuses.env_var_reads_in_tree(trees)
    assert names == (
        "TORCHLENS_PLANTED_BOOL",
        "TORCHLENS_PLANTED_CONSTANT",
        "TORCHLENS_PLANTED_GET",
        "TORCHLENS_PLANTED_GETENV",
        "TORCHLENS_PLANTED_SUBSCRIPT",
    ), "an env-read shape drifted or prose leaked into the census"
