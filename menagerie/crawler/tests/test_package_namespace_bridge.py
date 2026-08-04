"""One R1 identity, two package-name namespaces, and the bridge between them.

``recipe.assert_model_provenance`` attributes the constructed class through
Python installed-distribution metadata, so it reads ``torch``. The environment
inventory that supplies ``artifact_sha256`` speaks the packaging system's
namespace, so it reads ``pytorch``. Before the bridge an author had to break one
of the two machines: declaring ``pytorch`` had the constructor refused, and
declaring ``torch`` derived no digest at all and recorded a permanent R1 row
whose machine-derived identity was silently null.

These tests pin both directions:

- the divergent name now resolves, and the digest that lands is the one belonging
  to the row that actually installs the distribution;
- a name nothing provides fails closed with a typed refusal, at every layer, and
  is never resolved to a plausible neighbour;
- the machine-owned provision registry is a cached derivation and is checked
  against real installed evidence rather than believed.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pytest

import menagerie.crawler as crawler_package
import menagerie.crawler.package_namespace as package_namespace
from menagerie.crawler.env_lifecycle import (
    EnvironmentExactnessError,
    installed_package_inventory_bytes,
    parse_resolved_export,
)
from menagerie.crawler.package_namespace import (
    INVENTORY_DISTRIBUTION_PROVISIONS,
    PackageNamespaceError,
    assert_inventory_provisions,
    canonical_distribution_name,
    derive_provided_distributions,
    inventory_row_provides_distribution,
)
from menagerie.crawler.recipe import (
    RecipeError,
    assert_construct_namespace,
    assert_model_provenance,
    assert_recipe_root_namespace,
    bind_library_artifact_digest,
    resolve_environment_artifact_digest,
)

_EVIDENCE_PATH = Path(__file__).with_name("conda_meta_provision_evidence.json")


def _evidence() -> Mapping[str, Any]:
    """Return the committed real ``conda-meta`` excerpts backing the registry.

    Returns
    -------
    Mapping[str, Any]
        Package name mapped to its recorded version, build, and file excerpt.
    """

    return json.loads(_EVIDENCE_PATH.read_bytes())


def _digest(seed: str) -> str:
    """Return one distinct realistic canonical digest for ``seed``.

    Parameters
    ----------
    seed:
        Distinguishing package name.

    Returns
    -------
    str
        Canonical ``sha256:``-prefixed digest unique to ``seed``.
    """

    import hashlib

    return f"sha256:{hashlib.sha256(seed.encode()).hexdigest()}"


# The real divergence, spelled the way the shipped release lock spells it. Every
# neighbouring row is a same-name package with its own distinct digest, so a
# selection bug cannot hide behind a single-row fixture.
_CONDA_INVENTORY: tuple[Mapping[str, Any], ...] = (
    {
        "name": "pytorch",
        "version": "2.5.1",
        "build": "cpu_generic_py311_hc7d8f6d_17",
        "url": "https://conda.anaconda.org/conda-forge/osx-arm64/pytorch-2.5.1-cpu.conda",
        "sha256": _digest("pytorch"),
    },
    {
        "name": "libtorch",
        "version": "2.5.1",
        "build": "cpu_generic_h9a9d006_17",
        "url": "https://conda.anaconda.org/conda-forge/osx-arm64/libtorch-2.5.1-cpu.conda",
        "sha256": _digest("libtorch"),
    },
    {
        "name": "numpy",
        "version": "2.1.3",
        "build": "py311h649a571_0",
        "url": "https://conda.anaconda.org/conda-forge/osx-arm64/numpy-2.1.3-py311.conda",
        "sha256": _digest("numpy"),
    },
)


def _torch_implementation(**overrides: Any) -> dict[str, Any]:
    """Return one mutable R1 block pinning the Python distribution ``torch``.

    Parameters
    ----------
    **overrides:
        Recipe fields replacing the baseline.

    Returns
    -------
    dict[str, Any]
        Declarative-library implementation block.
    """

    recipe: dict[str, Any] = {
        "distribution": "torch",
        "version": "2.5.1",
        "module": "torch.nn",
        "symbol": "Linear",
        "kwargs": {"in_features": 4, "out_features": 4},
        "pretrained_disable_fields": [],
    }
    recipe.update(overrides)
    return {"recipe_type": "declarative-library", "library_recipe": recipe}


@pytest.mark.smoke
def test_the_divergent_name_now_resolves_to_the_installing_row() -> None:
    """``torch`` is pinned by a row the inventory spells ``pytorch``.

    This is the defect in the failing direction: on ``main`` the same call
    returned ``None`` and the R1 row was recorded with no machine-derived
    artifact identity at all.
    """

    implementation = _torch_implementation()

    assert bind_library_artifact_digest(implementation, list(_CONDA_INVENTORY)) is True
    assert implementation["library_recipe"]["artifact_sha256"] == _digest("pytorch")
    # The C++ runtime ships alongside and must never be mistaken for the artifact
    # that installs the Python distribution.
    assert implementation["library_recipe"]["artifact_sha256"] != _digest("libtorch")


@pytest.mark.smoke
def test_the_bridge_did_not_relax_the_provenance_tripwire() -> None:
    """The packaging spelling still cannot carry an R1 claim.

    Nothing here widens what ``assert_model_provenance`` accepts: it reads the
    Python metadata namespace directly and the provision registry is not
    consulted. So a recipe declaring ``pytorch`` may well bind a digest pre-gate
    -- ``pytorch`` is a real inventory row -- and is then refused at construction
    time, which is the layer that decides whether a class is really the pinned
    library's.
    """

    import torch

    assert (
        bind_library_artifact_digest(
            _torch_implementation(distribution="pytorch"), list(_CONDA_INVENTORY)
        )
        is True
    )

    with pytest.raises(RecipeError) as raised:
        assert_model_provenance(torch.nn.Linear(2, 2), "pytorch")

    assert "is not defined by the pinned distribution 'pytorch'" in str(raised.value)
    # And the spelling the recipe is supposed to use still passes.
    assert_model_provenance(torch.nn.Linear(2, 2), "torch")


@pytest.mark.smoke
def test_an_unknown_name_fails_closed_rather_than_resolving_to_a_neighbour() -> None:
    """No registry entry means the name provides only itself.

    ``torchvision`` is absent from this inventory and is a near neighbour of two
    rows that ARE present, so a mapping that guessed would resolve it here.
    """

    implementation = _torch_implementation(distribution="torchvision", version="0.20.1")
    before = deepcopy(implementation)

    with pytest.raises(RecipeError) as raised:
        bind_library_artifact_digest(implementation, list(_CONDA_INVENTORY))

    assert str(raised.value) == (
        "routed environment does not install distribution 'torchvision': no package "
        "row carries that name and none declares it as a provision"
    )
    assert implementation == before


@pytest.mark.smoke
def test_the_provision_registry_is_never_consulted_in_reverse() -> None:
    """A row provides only itself plus what its own entry declares."""

    assert inventory_row_provides_distribution("pytorch", "torch") is True
    assert inventory_row_provides_distribution("pytorch", "pytorch") is True
    assert inventory_row_provides_distribution("torch", "torch") is True
    # The declared distribution is never rewritten, so ``torch`` does not become
    # a claim on some other row, and an unrelated row never acquires the entry.
    assert inventory_row_provides_distribution("torch", "pytorch") is False
    assert inventory_row_provides_distribution("libtorch", "torch") is False
    assert inventory_row_provides_distribution("numpy", "torch") is False


@pytest.mark.smoke
def test_the_divergent_name_is_still_version_checked() -> None:
    """Bridging spellings must not bridge a version disagreement.

    The digest pins one build, so a version the routed environment does not
    install still refuses -- and the message names both spellings.
    """

    implementation = _torch_implementation(version="2.6.0")

    with pytest.raises(RecipeError) as raised:
        bind_library_artifact_digest(implementation, list(_CONDA_INVENTORY))

    assert str(raised.value) == (
        "recipe declares 'torch' (inventory package 'pytorch') version '2.6.0' "
        "but the routed environment installs '2.5.1'"
    )


@pytest.mark.smoke
def test_two_rows_providing_one_distribution_are_ambiguous() -> None:
    """A pip-spelled row alongside its conda-spelled twin cannot pin a build."""

    both = [
        *_CONDA_INVENTORY,
        {
            "name": "torch",
            "version": "2.5.1",
            "build": "pypi_0",
            "url": "https://files.pythonhosted.org/torch-2.5.1.whl",
            "sha256": _digest("torch-wheel"),
        },
    ]

    with pytest.raises(RecipeError) as raised:
        resolve_environment_artifact_digest(both, distribution="torch", version="2.5.1")

    assert str(raised.value) == "environment inventory names distribution 'torch' ambiguously"


@pytest.mark.smoke
def test_the_registry_entry_is_derived_from_real_installed_evidence() -> None:
    """The table is a cached derivation, not an assertion.

    The evidence file is a committed excerpt of the real ``conda-meta`` records
    from a materialized ``round19-osx-arm64`` prefix, so this recomputes the
    registry's only entry from the bytes conda itself wrote.
    """

    evidence = _evidence()

    assert (
        derive_provided_distributions(evidence["pytorch"]["files_excerpt"])
        == INVENTORY_DISTRIBUTION_PROVISIONS["pytorch"]
        == frozenset({"torch"})
    )
    # A same-name package derives its own name, which is why it needs no entry.
    assert derive_provided_distributions(evidence["numpy"]["files_excerpt"]) == frozenset(
        {"numpy"}
    )
    assert canonical_distribution_name("numpy") not in INVENTORY_DISTRIBUTION_PROVISIONS
    # A package that installs no Python distribution derives nothing, and an
    # empty result is a true answer rather than a failure.
    assert derive_provided_distributions(evidence["libtorch"]["files_excerpt"]) == frozenset()


@pytest.mark.smoke
def test_every_registry_entry_is_a_real_divergence() -> None:
    """An entry that merely restates a package's own name would be dead weight."""

    for package, provided in INVENTORY_DISTRIBUTION_PROVISIONS.items():
        assert package == canonical_distribution_name(package)
        assert provided, package
        assert all(name == canonical_distribution_name(name) for name in provided), package
        assert package not in provided, package


@pytest.mark.smoke
def test_a_stale_registry_entry_stops_environment_creation(tmp_path: Path) -> None:
    """The cached derivation is re-proved against the live file list.

    If conda-forge ever changes what ``pytorch`` installs, materialization
    refuses instead of letting the pre-gate lookup mis-select a digest.
    """

    evidence = _evidence()
    metadata = tmp_path / "conda-meta"
    metadata.mkdir()
    record = {
        "name": "pytorch",
        "version": "2.5.1",
        "build": "cpu_generic_py311_hc7d8f6d_17",
        "url": "https://conda.anaconda.org/conda-forge/osx-arm64/pytorch-2.5.1-cpu.conda",
        "sha256": _digest("pytorch")[len("sha256:") :],
        "files": [
            path.replace("torch-2.5.1.dist-info", "pytorch_renamed-2.5.1.dist-info")
            for path in evidence["pytorch"]["files_excerpt"]
        ],
    }
    (metadata / "pytorch.json").write_bytes(json.dumps(record).encode())

    with pytest.raises(EnvironmentExactnessError) as raised:
        installed_package_inventory_bytes(tmp_path)

    assert "provides ['pytorch-renamed'] but the provision registry declares ['torch']" in str(
        raised.value
    )


@pytest.mark.smoke
def test_a_declared_provision_with_no_file_list_is_unverifiable(tmp_path: Path) -> None:
    """A record that cannot be checked is refused, not waved through."""

    metadata = tmp_path / "conda-meta"
    metadata.mkdir()
    (metadata / "pytorch.json").write_bytes(
        json.dumps(
            {
                "name": "pytorch",
                "version": "2.5.1",
                "build": "cpu_generic_py311_hc7d8f6d_17",
                "url": "https://conda.anaconda.org/conda-forge/osx-arm64/pytorch-2.5.1.conda",
                "sha256": _digest("pytorch")[len("sha256:") :],
            }
        ).encode()
    )

    with pytest.raises(EnvironmentExactnessError) as raised:
        installed_package_inventory_bytes(tmp_path)

    assert "lists no files" in str(raised.value)


@pytest.mark.smoke
def test_a_package_with_no_registry_entry_is_not_policed_at_materialization() -> None:
    """An undeclared divergence is one recipe's problem, not the environment's.

    Refusing to build an environment over an unrelated package's naming quirk
    would take the whole crawler down; the unresolvable recipe refuses
    model-locally instead.
    """

    assert_inventory_provisions("some-unrelated-package", None)
    assert_inventory_provisions("numpy", ["lib/site-packages/other-1.0.dist-info/M"])

    with pytest.raises(PackageNamespaceError):
        assert_inventory_provisions("pytorch", ["lib/site-packages/other-1.0.dist-info/M"])


@pytest.mark.smoke
@pytest.mark.parametrize("target", ["osx-arm64", "linux-64"])
def test_the_shipped_release_locks_resolve_the_torch_distribution(target: str) -> None:
    """The real committed lock family, not a fixture, is what has to resolve.

    Both shipped release targets install the Python distribution ``torch`` under
    the inventory package name ``pytorch``. This is the end-to-end shape of the
    defect: on ``main`` both of these bound ``None``.
    """

    locks = Path(str(crawler_package.__file__)).parent / "envs" / "locks"
    export = locks / f"round19-{target}.resolved.json"
    packages = json.loads(parse_resolved_export(export.read_bytes()))["packages"]
    conda_rows = [row for row in packages if row["name"] == "pytorch"]
    assert len(conda_rows) == 1, target
    assert not [row for row in packages if row["name"] == "torch"], target

    implementation = _torch_implementation(version=conda_rows[0]["version"])

    assert bind_library_artifact_digest(implementation, packages) is True
    assert implementation["library_recipe"]["artifact_sha256"] == conda_rows[0]["sha256"]


@pytest.mark.smoke
def test_the_provision_registry_cannot_reach_the_namespace_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registry selects an inventory row; it never admits a module.

    The recipe-root and construct-node bounds refuse a module that no INSTALLED
    distribution supplies, which is what closes the stdlib laundering path. The
    provision registry maps inventory names to distribution names for one
    purpose -- choosing which package row supplies a digest -- and must never
    become a second way to satisfy those bounds, or a registry entry would turn
    into an import permission.
    """

    # Inject a hostile entry claiming an inventory package provides a stdlib
    # name. If either namespace bound consulted the registry, this would admit
    # ``subprocess`` -- the exact hole the recipe-root bound exists to close.
    hostile = dict(INVENTORY_DISTRIBUTION_PROVISIONS)
    hostile["evil-row"] = frozenset({"subprocess"})
    monkeypatch.setattr(
        package_namespace, "INVENTORY_DISTRIBUTION_PROVISIONS", hostile, raising=True
    )

    with pytest.raises(RecipeError):
        assert_recipe_root_namespace(
            "subprocess", distribution="subprocess", require_installed=True
        )
    with pytest.raises(RecipeError):
        assert_construct_namespace("subprocess", distribution="subprocess", context="kwargs.y")
    # The digest lookup DOES read it, which is what proves the injection was live
    # and the bounds' indifference above is real rather than vacuous.
    assert inventory_row_provides_distribution("evil-row", "subprocess") is True

    # The bounds still refuse the stdlib, and the registry's own spelling is not
    # a way around the root bound either.
    with pytest.raises(RecipeError):
        assert_recipe_root_namespace(
            "subprocess", distribution="subprocess", require_installed=False
        )
    with pytest.raises(RecipeError):
        assert_construct_namespace("subprocess", distribution="subprocess", context="kwargs.x")
    with pytest.raises(RecipeError):
        assert_recipe_root_namespace("torch.nn", distribution="pytorch", require_installed=True)
    # And the spelling the recipe must use is still admitted, so the conda row
    # that provides it resolves without weakening anything.
    assert_recipe_root_namespace("torch.nn", distribution="torch", require_installed=True)
    assert resolve_environment_artifact_digest(
        _CONDA_INVENTORY, distribution="torch", version="2.5.1"
    ) == _digest("pytorch")


@pytest.mark.smoke
def test_the_derivation_reads_only_installed_python_metadata_directories() -> None:
    """Only ``site-packages`` metadata directories name a distribution."""

    assert derive_provided_distributions(
        [
            "lib/python3.11/site-packages/scikit_learn-1.5.2.dist-info/METADATA",
            "lib/python3.11/site-packages/legacy_pkg.egg-info/PKG-INFO",
            "lib/python3.11/site-packages/other_pkg-1.0-py3.11.egg-info/PKG-INFO",
            # A directory that merely looks like metadata but is not installed
            # into site-packages names nothing.
            "share/doc/decoy-9.9.dist-info/METADATA",
            "lib/python3.11/site-packages/torch/nn/__init__.py",
        ]
    ) == frozenset({"scikit-learn", "legacy-pkg", "other-pkg"})
