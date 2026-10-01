"""The package ships ZERO deprecation shims — and stays that way.

History: this module was the R48 deprecation census. It inventoried eleven
families (~183 deprecated spellings: moved top-level names, paper-era API
shims, warning flat option kwargs, renamed callables, alias properties,
no-op kwargs/functions, deprecated values) with removal metadata, because the
shims had accumulated with no expiry. The 2026-08-19 shim-removal lane (JMT
ruling: interim-phase deprecation shims are not justified — remove them)
deleted every family outright, along with ``warn_deprecated_alias``,
``TorchLensDeprecationWarning``, and the ``REMOVED_IN`` window.

This module is now the census INVERTED: the same AST scanners that once
derived family membership prove the package emits no DeprecationWarning from
any site, names no deprecated alias, and documents no surface as deprecated.
Interim-phase policy is remove-and-rename, not shim; a new shim fails here
until the policy is consciously changed.

Deliberately NOT covered (they are not API deprecation shims):

- torch-version compatibility (``torchlens/utils/_torch_compat.py`` HAS_*
  capability flags and guarded fallbacks);
- artifact-format compatibility (the tlspec version floor, legacy-save
  loading, the load-path folding of legacy conditional edge keys, legacy
  2.16 intervention-spec loading) — load-bearing for artifacts in the wild;
- the ``ArtifactSchemaAgeWarning`` advisory (a UserWarning about artifact
  age, not an API deprecation; pinned by tests/test_rehydration_floor.py).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pytest
from _source_corpus import module_ast as _corpus_ast, module_source as _corpus_source

pytestmark = pytest.mark.smoke

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _REPO_ROOT / "torchlens"

#: Warning categories whose emission makes a ``warnings.warn`` call a
#: DEPRECATION emission. ``TorchLensDeprecationWarning`` is deleted, but the
#: scanner still recognizes the name so a resurrected class cannot hide.
_DEPRECATION_CATEGORY_NAMES = frozenset(
    {"DeprecationWarning", "PendingDeprecationWarning", "TorchLensDeprecationWarning"}
)

#: Docstring-summary phrasings by which a surface would declare ITSELF
#: deprecated. Matched against the first line of a function/property
#: docstring only, so prose further down cannot manufacture a finding, and
#: narrow enough to exclude helpers whose summary merely mentions deprecation.
_DEPRECATED_DOC_PREFIXES = ("deprecated",)
_DEPRECATED_DOC_PHRASES = ("deprecated alias for", "legacy alias for", "deprecated: use")

#: Prefilter tokens: cheap literal check deciding which files get parsed.
_DEPRECATED_DOC_TOKENS = ("Deprecated", "deprecated", "Legacy alias", "legacy alias")

#: The historical deprecation-emitting helper names. All deleted; the scanner
#: keeps recognizing them so a re-added helper registers as an emission.
_WARN_HELPERS = frozenset({"warn_deprecated_alias", "_warn_moved_name", "_warn_legacy_api_name"})


@dataclass(frozen=True)
class _PackageScan:
    """One pass of AST facts about a package tree.

    Parameters
    ----------
    sites:
        ``path::function`` keys of raw ``warnings.warn(..., DeprecationWarning)``
        calls.
    alias_sites:
        ``path::function`` keys of calls to a historical warn-helper name.
    documented_deprecated:
        ``path::function`` keys of functions/properties whose DOCSTRING calls
        something deprecated or legacy.
    """

    sites: frozenset[str]
    alias_sites: frozenset[str]
    documented_deprecated: frozenset[str]


@lru_cache(maxsize=8)
def scan_package(package_root: Path, base: Path) -> _PackageScan:
    """Return the deprecation AST facts in ONE cached pass over the tree.

    Parameters
    ----------
    package_root:
        Root of the package tree to scan.
    base:
        Directory reported paths are relative to.

    Returns
    -------
    _PackageScan
        Emission sites, helper-call sites, and documented-deprecated surfaces.
    """

    sites: set[str] = set()
    alias_sites: set[str] = set()
    documented: set[str] = set()
    for path in sorted(package_root.rglob("*.py")):
        text = _corpus_source(path)
        if not any(
            token in text
            for token in ("DeprecationWarning", "warn_deprecated_alias", *_DEPRECATED_DOC_TOKENS)
        ):
            continue
        relative = path.relative_to(base).as_posix()
        tree = _corpus_ast(path)
        _visit(tree, relative, "<module>", sites, alias_sites)
        _visit_docstrings(tree, relative, documented)
    return _PackageScan(frozenset(sites), frozenset(alias_sites), frozenset(documented))


def _visit(
    node: ast.AST,
    relative: str,
    enclosing: str,
    sites: set[str],
    alias_sites: set[str],
) -> None:
    """Collect deprecation-emission facts under ``node``.

    Parameters
    ----------
    node:
        Node to descend from.
    relative:
        Repo-relative path of the module being scanned.
    enclosing:
        Name of the innermost enclosing function.
    sites:
        Accumulator for ``path::function`` raw-emission sites.
    alias_sites:
        Accumulator for ``path::function`` warn-helper call sites.
    """

    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "warn":
            categories = [*node.args, *(keyword.value for keyword in node.keywords)]
            if any(
                isinstance(value, ast.Name) and value.id in _DEPRECATION_CATEGORY_NAMES
                for value in categories
            ):
                sites.add(f"{relative}::{enclosing}")
        elif isinstance(func, ast.Name) and func.id in _WARN_HELPERS:
            alias_sites.add(f"{relative}::{enclosing}")
    for child in ast.iter_child_nodes(node):
        child_enclosing = (
            child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else enclosing
        )
        _visit(child, relative, child_enclosing, sites, alias_sites)


def _visit_docstrings(node: ast.AST, relative: str, documented: set[str]) -> None:
    """Collect ``path::function`` for functions documented as deprecated.

    Parameters
    ----------
    node:
        Node to descend from.
    relative:
        Repo-relative path of the module being scanned.
    documented:
        Accumulator for ``path::function`` keys.
    """

    for child in ast.walk(node):
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        docstring = (ast.get_docstring(child) or "").strip()
        if not docstring:
            continue
        summary = docstring.splitlines()[0].lower()
        declares = summary.startswith(_DEPRECATED_DOC_PREFIXES) or any(
            phrase in summary for phrase in _DEPRECATED_DOC_PHRASES
        )
        if declares:
            documented.add(f"{relative}::{child.name}")


def deprecation_emission_sites(package_root: Path, base: Path | None = None) -> set[str]:
    """Return ``file::function`` for every raw DeprecationWarning emission.

    Parameters
    ----------
    package_root:
        Root of the shipped package.
    base:
        Base directory the reported paths are relative to.

    Returns
    -------
    set[str]
        ``path::function`` keys relative to ``base``.
    """

    return set(scan_package(package_root, base if base is not None else _REPO_ROOT).sites)


def warn_helper_call_sites(package_root: Path, base: Path | None = None) -> set[str]:
    """Return ``file::function`` for every historical warn-helper call.

    Parameters
    ----------
    package_root:
        Root of the shipped package.
    base:
        Base directory the reported paths are relative to.

    Returns
    -------
    set[str]
        ``path::function`` keys of calls to a deleted warn-helper name.
    """

    return set(scan_package(package_root, base if base is not None else _REPO_ROOT).alias_sites)


def documented_deprecated_sites(package_root: Path, base: Path | None = None) -> set[str]:
    """Return ``file::function`` for every function DOCUMENTED as deprecated.

    Parameters
    ----------
    package_root:
        Root of the shipped package.
    base:
        Base directory reported paths are relative to.

    Returns
    -------
    set[str]
        ``path::function`` keys whose docstring summary says deprecated/legacy.
    """

    scan = scan_package(package_root, base if base is not None else _REPO_ROOT)
    return set(scan.documented_deprecated)


# ---------------------------------------------------------------------------
# The no-shim tripwire.
# ---------------------------------------------------------------------------


def test_package_emits_no_deprecation_warnings() -> None:
    """No shipped site emits a DeprecationWarning of any spelling."""

    sites = deprecation_emission_sites(_PACKAGE_ROOT)
    assert sites == set(), (
        f"new deprecation emission sites appeared: {sorted(sites)}; interim-phase "
        "policy is remove-and-rename, not shim (see the module docstring)"
    )


def test_no_warn_helper_calls_remain() -> None:
    """No shipped site calls a historical deprecation warn-helper."""

    sites = warn_helper_call_sites(_PACKAGE_ROOT)
    assert sites == set(), f"warn-helper calls reappeared: {sorted(sites)}"


def test_no_surface_documents_itself_as_deprecated() -> None:
    """No shipped surface declares itself a deprecated/legacy alias."""

    documented = documented_deprecated_sites(_PACKAGE_ROOT)
    assert documented == set(), (
        f"new deprecated-alias surfaces appeared: {sorted(documented)}; "
        "interim-phase policy is remove-and-rename, not shim"
    )


def test_deprecation_warning_class_is_gone() -> None:
    """The dedicated warning category did not quietly come back."""

    import torchlens._deprecations as deprecations_module
    import torchlens.errors as errors_module

    assert not hasattr(deprecations_module, "TorchLensDeprecationWarning")
    assert not hasattr(deprecations_module, "warn_deprecated_alias")
    assert not hasattr(deprecations_module, "REMOVED_IN")
    with pytest.raises(AttributeError):
        errors_module.TorchLensDeprecationWarning  # noqa: B018


# ---------------------------------------------------------------------------
# The tripwire mechanism must be able to go RED.
# ---------------------------------------------------------------------------


class TestTripwireMechanismIsRedCapable:
    """Plant a shim and prove each scanner reports it."""

    def test_site_scanner_finds_a_planted_emission(self, tmp_path: Path) -> None:
        """A new DeprecationWarning site is discovered by the scan."""

        package = tmp_path / "torchlens"
        package.mkdir()
        (package / "mod.py").write_text(
            "import warnings\ndef f():\n    warnings.warn('x', DeprecationWarning, stacklevel=2)\n",
            encoding="utf-8",
        )
        found = deprecation_emission_sites(package, base=tmp_path)
        assert found == {"torchlens/mod.py::f"}, found

    def test_site_scanner_ignores_other_warning_categories(self, tmp_path: Path) -> None:
        """A UserWarning is not a deprecation and is not reported."""

        package = tmp_path / "torchlens"
        package.mkdir()
        (package / "mod.py").write_text(
            "import warnings\ndef f():\n    warnings.warn('x', UserWarning)\n",
            encoding="utf-8",
        )
        assert deprecation_emission_sites(package, base=tmp_path) == set()

    def test_helper_scanner_finds_a_planted_call(self, tmp_path: Path) -> None:
        """A resurrected warn-helper call is discovered by the scan."""

        package = tmp_path / "torchlens"
        package.mkdir()
        (package / "mod.py").write_text(
            "def f():\n    warn_deprecated_alias('old', 'new')\n",
            encoding="utf-8",
        )
        assert warn_helper_call_sites(package, base=tmp_path) == {"torchlens/mod.py::f"}

    def test_docstring_scanner_finds_a_planted_alias_surface(self, tmp_path: Path) -> None:
        """A deprecated-documented surface is reported."""

        package = tmp_path / "torchlens"
        package.mkdir()
        (package / "mod.py").write_text(
            'def f():\n    """Deprecated alias for g."""\n    return 1\n',
            encoding="utf-8",
        )
        assert documented_deprecated_sites(package, base=tmp_path) == {"torchlens/mod.py::f"}

    def test_docstring_scanner_ignores_helpers_that_merely_mention_deprecation(
        self, tmp_path: Path
    ) -> None:
        """A summary ABOUT deprecation is not a deprecated surface."""

        package = tmp_path / "torchlens"
        package.mkdir()
        (package / "mod.py").write_text(
            'def f():\n    """Synchronize one deprecated wrapper on first access."""\n'
            "    return 1\n",
            encoding="utf-8",
        )
        assert documented_deprecated_sites(package, base=tmp_path) == set()
