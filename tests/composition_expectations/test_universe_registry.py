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

pytestmark = pytest.mark.compo


def _parse(module: str, source: str) -> dict[str, ast.Module]:
    return {module: ast.parse(textwrap.dedent(source))}


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


def test_env_var_census_shapes() -> None:
    """All three env-read shapes census; prose mentions do not."""

    trees = _parse(
        "planted.module",
        """
        import os
        from torchlens._utils.env_flags import closed_bool_env

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
