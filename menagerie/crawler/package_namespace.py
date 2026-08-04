"""The two package-name namespaces an R1 identity spans, and the bridge between them.

A declarative R1 recipe names ONE distribution, but two independent machines read
that one name in two different namespaces:

* the **distribution namespace** -- Python installed-distribution metadata as
  ``importlib.metadata`` reports it. ``recipe.assert_model_provenance`` and
  ``recipe.assert_construct_namespace`` read this namespace to attribute the
  constructed class to the pinned artifact.
* the **inventory namespace** -- the routed environment's own packaging system,
  whose exact ``name``/``version``/``sha256`` rows are the only place the
  machine-derived ``artifact_sha256`` can come from.

The two namespaces agree for most of the catalog (``timm``, ``torchvision``,
``transformers`` are spelled identically in both) and disagree wherever a
packager renamed something: conda-forge installs the Python distribution
``torch`` under the inventory package name ``pytorch``. Before this module that
divergence forced an author to break one of the two machines. Declaring
``pytorch`` refused the constructed model -- correctly, because
``importlib.metadata`` attributes ``torch.nn.Module`` subclasses to ``torch``.
Declaring ``torch`` passed provenance but derived NO digest, and a permanent R1
row was recorded with a null machine-derived identity that nothing complained
about.

The bridge is deliberately one-directional and machine-owned:

* a recipe names a Python **distribution** and never anything else;
* an inventory **row** is what declares which Python distributions it provides;
* resolution asks each row "do you provide the declared distribution?".

The declared name is therefore never rewritten into some other name. An author
can only say which distribution the class must come from; the environment alone
decides whether it has it. A distribution no row provides has no plausible
neighbour to fall back to -- it fails closed at the caller.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Sequence


class PackageNamespaceError(ValueError):
    """Raised when a declared package provision contradicts installed evidence."""


def canonical_distribution_name(value: str) -> str:
    """Return one comparable package name for cross-namespace lookup.

    Parameters
    ----------
    value:
        Declared distribution or inventory package name.

    Returns
    -------
    str
        Case-folded name with ``_``/``.`` normalized to ``-``.
    """

    return value.strip().casefold().replace("_", "-").replace(".", "-")


_PYTHON_METADATA_SUFFIXES = (".dist-info", ".egg-info")


def derive_provided_distributions(files: Iterable[str]) -> frozenset[str]:
    """Derive the Python distributions an installed package's own file list provides.

    An installed package declares its Python distributions in the only place that
    cannot be renamed away: the ``site-packages/<name>-<version>.dist-info/`` (or
    ``.egg-info``) directories it actually installs. This reads exactly that, so
    the inventory-to-distribution link is derived from the package's own
    manifest rather than asserted by anyone.

    Parameters
    ----------
    files:
        Package-relative installed file paths, as an installed-package metadata
        record lists them.

    Returns
    -------
    frozenset[str]
        Canonical Python distribution names the file list installs. Empty for a
        package that installs no Python distribution at all (a shared library,
        a compiler runtime), which is a true answer, not a failure.
    """

    provided: set[str] = set()
    for entry in files:
        parts = str(entry).replace("\\", "/").split("/")
        for parent, part in zip(parts, parts[1:]):
            if parent != "site-packages":
                continue
            for suffix in _PYTHON_METADATA_SUFFIXES:
                if not part.endswith(suffix):
                    continue
                # Distribution names are normalized to ``_`` inside these
                # directory names, so the first ``-`` always begins the version.
                stem = part[: -len(suffix)].split("-", 1)[0]
                if stem:
                    provided.add(canonical_distribution_name(stem))
    return frozenset(provided)


INVENTORY_DISTRIBUTION_PROVISIONS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "pytorch": frozenset({"torch"}),
    }
)
"""Canonical inventory package name -> canonical Python distributions it installs.

ONLY renamed packages appear here. A package whose two spellings agree needs no
entry, and the absence of an entry is what makes an unknown name fail closed
rather than resolve to something plausible.

Every entry is a CACHED DERIVATION, never a guess. It is exactly what
:func:`derive_provided_distributions` returns for that package's own installed
file list; it is pinned in the tests against committed real installed-metadata
evidence; and :func:`assert_inventory_provisions` re-proves it against the live
file list every time an environment is materialized. A stale or wrong entry
therefore stops environment creation instead of quietly mis-selecting a digest.

This table is machine-owned. Nothing an author writes reaches it, and it is only
ever consulted in the row-to-distribution direction, so it cannot be used to
launder a declared name into a different artifact's identity. It also cannot
bless a model: class attribution stays with
``recipe.assert_model_provenance``, which reads the Python metadata namespace
directly and is not consulted here.
"""


def inventory_row_provides_distribution(row_name: str, distribution: str) -> bool:
    """Return whether one inventory row provides the declared Python distribution.

    Parameters
    ----------
    row_name:
        Package name exactly as the environment inventory spells it.
    distribution:
        Python distribution the recipe declares.

    Returns
    -------
    bool
        True when the row is the declared distribution under either spelling.
        An inventory name with no registry entry provides only itself, so an
        unrecognized name simply matches nothing.
    """

    canonical_row = canonical_distribution_name(row_name)
    wanted = canonical_distribution_name(distribution)
    if canonical_row == wanted:
        return True
    return wanted in INVENTORY_DISTRIBUTION_PROVISIONS.get(canonical_row, frozenset())


def assert_inventory_provisions(name: str, files: Optional[Sequence[str]]) -> None:
    """Re-prove one registry entry against the installed package's own file list.

    Called for every package in a materialized prefix. A package with no registry
    entry is not policed here: an undeclared divergence is handled model-locally
    by the caller that fails to resolve it, which refuses one recipe instead of
    refusing to build an environment over an unrelated package's naming quirk.

    Parameters
    ----------
    name:
        Installed package name in the inventory namespace.
    files:
        Installed file paths the package's own metadata record lists, or None
        when the record carries no file list.

    Raises
    ------
    PackageNamespaceError
        If the package has a registry entry and its installed file list either
        cannot be read or provides a different set of Python distributions.
    """

    declared = INVENTORY_DISTRIBUTION_PROVISIONS.get(canonical_distribution_name(name))
    if declared is None:
        return
    if files is None:
        raise PackageNamespaceError(
            f"installed package {name!r} declares a Python distribution provision "
            "that cannot be verified: its metadata record lists no files"
        )
    observed = derive_provided_distributions(files)
    if observed != declared:
        raise PackageNamespaceError(
            f"installed package {name!r} provides {sorted(observed)} but the "
            f"provision registry declares {sorted(declared)}"
        )
