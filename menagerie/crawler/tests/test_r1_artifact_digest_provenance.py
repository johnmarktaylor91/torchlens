"""The R1 installed-artifact digest is machine-derived, not author-asserted.

``proposed_facts.implementation.library_recipe.artifact_sha256`` identifies the
INSTALLED distribution. The author stage receives no environment identity, no
capability sheet, and no package inventory, runs before the environment stage
exists, and has no interpreter -- so it cannot derive that digest. Requiring it
of the author made the only ``c1-mech`` success path mechanically unreachable
while nothing downstream ever verified the value.

These tests pin the corrected division of labour in the failing direction: an
honest proposal that omits the digest must validate through the REAL
``Draft202012Validator``, the previous contract must be shown to have rejected
that same proposal, and a supplied digest that CONFLICTS with the routed
environment must be refused rather than silently overwritten.
"""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from menagerie.crawler.constants import (
    AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
    MODEL_SCHEMA_VERSION_V3,
)
from menagerie.crawler.driver_admission import (
    _normalize_artifact_modes,
    _routed_environment_packages,
)
from menagerie.crawler.driver_contracts import DriverConfig, DriverIntegrationError
from menagerie.crawler.envs import load_environment_registry
from menagerie.crawler.metadata import authored_fact_leaves
from menagerie.crawler.recipe import (
    RecipeError,
    bind_library_artifact_digest,
    resolve_environment_artifact_digest,
)
from menagerie.crawler.schema import (
    SCHEMA_FILES,
    SCHEMA_RESOURCE_FILES,
    SchemaOwner,
    get_validator,
    load_schema,
    load_schema_resource,
    schema_owner_paths,
)

from menagerie.crawler.tests.conftest import (
    HASH,
    make_author_proposal,
    make_proposed_artifact,
)

_DRIVER_CONFIG = DriverConfig()

_RECIPE_PATH = "$.proposed_facts.implementation.library_recipe"
_DIGEST_LEAF = "$.implementation.library_recipe.artifact_sha256"


def _distinct_digest(seed: str) -> str:
    """Return one realistic, non-placeholder canonical digest for ``seed``.

    A negative test whose two slots hold the SAME sentinel passes vacuously, so
    every fixture digest here is distinct and derived from a distinct name.

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


# A MULTI-source inventory: a single-row fixture hides every selection bug, so
# the routed environment always names several real distributions with distinct
# versions and distinct digests.
_ROUTED_PACKAGES: tuple[Mapping[str, Any], ...] = (
    {
        "name": "pytorch",
        "version": "2.6.0",
        "build": "cpu_generic_py311h0",
        "url": "https://conda.anaconda.org/conda-forge/osx-arm64/pytorch-2.6.0-cpu.conda",
        "sha256": _distinct_digest("pytorch"),
    },
    {
        "name": "timm",
        "version": "1.0.9",
        "build": "pyhd8ed1ab_0",
        "url": "https://conda.anaconda.org/conda-forge/noarch/timm-1.0.9-pyhd8ed1ab_0.conda",
        "sha256": _distinct_digest("timm"),
    },
    {
        "name": "torchvision",
        "version": "0.21.0",
        "build": "cpu_py311h4",
        "url": "https://conda.anaconda.org/conda-forge/osx-arm64/torchvision-0.21.0-cpu.conda",
        "sha256": _distinct_digest("torchvision"),
    },
    {
        "name": "transformers",
        "version": "4.52.1",
        "build": "pyhd8ed1ab_0",
        "url": "https://conda.anaconda.org/conda-forge/noarch/transformers-4.52.1.conda",
        "sha256": _distinct_digest("transformers"),
    },
)


def _library_implementation(**recipe_overrides: Any) -> dict[str, Any]:
    """Return one mutable R1 implementation block naming a real distribution."""

    recipe: dict[str, Any] = {
        "distribution": "timm",
        "version": "1.0.9",
        "module": "timm.models.vision_transformer",
        "symbol": "VisionTransformer",
        "kwargs": {"pretrained": False},
        "pretrained_disable_fields": ["pretrained"],
    }
    recipe.update(recipe_overrides)
    return {"recipe_type": "declarative-library", "library_recipe": recipe}


def _honest_r1_proposal() -> dict[str, Any]:
    """Return an R1 ``PROPOSED`` proposal that omits the underivable digest."""

    proposal = make_author_proposal("m_r1_honest")
    recipe = proposal["proposed_facts"]["implementation"]["library_recipe"]
    assert proposal["proposed_facts"]["source_resolution"]["rung"] == "R1_LIBRARY"
    del recipe["artifact_sha256"]
    return proposal


def _previous_contract_validator() -> Draft202012Validator:
    """Rebuild the validator for the contract as it stood before this change.

    The reconstruction restores exactly the two edits that made the leaf
    mandatory -- the ``required`` entry and the non-nullable ``hash`` ref -- so
    the rejection below is the OLD schema's, not a hand-written stand-in.

    Returns
    -------
    Draft202012Validator
        Author-proposal validator bound to the previous ``model-common``.
    """

    previous = deepcopy(load_schema_resource("model-common.schema.json"))
    library_recipe = previous["$defs"]["library_recipe"]
    library_recipe["required"] = [
        "distribution",
        "version",
        "artifact_sha256",
        "module",
        "symbol",
        "kwargs",
        "pretrained_disable_fields",
    ]
    library_recipe["properties"]["artifact_sha256"] = {
        "$ref": "#/$defs/hash",
        "description": "Mandatory artifact sha256.",
    }
    resources: list[tuple[str, Resource[Any]]] = []
    for version in SCHEMA_FILES:
        schema = load_schema(version)
        resources.append((schema["$id"], Resource.from_contents(schema)))
    for filename in SCHEMA_RESOURCE_FILES:
        schema = (
            previous
            if filename == "model-common.schema.json"
            else load_schema_resource(filename)
        )
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Draft202012Validator(
        load_schema(AUTHOR_PROPOSAL_SCHEMA_VERSION_V3),
        format_checker=FormatChecker(),
        registry=Registry[Any]().with_resources(resources),
    )


@pytest.mark.smoke
def test_an_honest_r1_proposal_now_validates_without_the_installed_digest() -> None:
    """The author may omit a digest it structurally cannot derive."""

    proposal = _honest_r1_proposal()
    validator = get_validator(AUTHOR_PROPOSAL_SCHEMA_VERSION_V3)

    assert list(validator.iter_errors(proposal)) == []


@pytest.mark.smoke
def test_a_null_installed_digest_is_expressible() -> None:
    """An environment that names no matching distribution records an honest null."""

    proposal = _honest_r1_proposal()
    proposal["proposed_facts"]["implementation"]["library_recipe"]["artifact_sha256"] = None
    validator = get_validator(AUTHOR_PROPOSAL_SCHEMA_VERSION_V3)

    assert list(validator.iter_errors(proposal)) == []


@pytest.mark.smoke
def test_the_previous_contract_rejected_that_same_honest_proposal() -> None:
    """The old schema is shown to reject the proposal the new one accepts.

    The assertion is exact-equality on the message: an ``anyOf`` failure dumps
    the whole instance, so a substring check would pass no matter what failed.
    """

    proposal = _honest_r1_proposal()
    errors = list(_previous_contract_validator().iter_errors(proposal))

    assert [error.json_path for error in errors] == [_RECIPE_PATH]
    # The leaf sits behind a ``oneOf`` (recipe | null). The umbrella message dumps
    # the whole instance, so it would swallow ANY substring assertion; the exact
    # cause is asserted on the branch errors instead.
    assert errors[0].validator == "oneOf"
    assert sorted(sub.message for sub in errors[0].context) == [
        "'artifact_sha256' is a required property",
        "{'distribution': 'example', 'version': '1.0', 'module': 'example', "
        "'symbol': 'ExampleNet', 'kwargs': {'weights': None}, "
        "'pretrained_disable_fields': ['weights']} is not of type 'null'",
    ]
    assert sorted(sub.validator for sub in errors[0].context) == ["required", "type"]


@pytest.mark.smoke
def test_the_digest_is_derived_from_the_routed_environment() -> None:
    """The one row naming the declared distribution supplies the digest."""

    implementation = _library_implementation()

    assert bind_library_artifact_digest(implementation, list(_ROUTED_PACKAGES)) is True
    assert implementation["library_recipe"]["artifact_sha256"] == _distinct_digest("timm")
    # A neighbouring row must never be selected.
    assert implementation["library_recipe"]["artifact_sha256"] != _distinct_digest("pytorch")


@pytest.mark.smoke
def test_a_conflicting_supplied_digest_is_refused_not_overwritten() -> None:
    """A supplied digest that contradicts the environment stops the model.

    Unconditional overwriting would leave this check structurally dead, so the
    refusal is typed and the caller's block is left byte-identical.
    """

    supplied = _distinct_digest("some-other-build")
    implementation = _library_implementation(artifact_sha256=supplied)
    before = deepcopy(implementation)

    with pytest.raises(RecipeError) as raised:
        bind_library_artifact_digest(implementation, list(_ROUTED_PACKAGES))

    assert str(raised.value) == (
        "supplied artifact_sha256 conflicts with the routed environment: "
        f"supplied {supplied}, derived {_distinct_digest('timm')}"
    )
    assert implementation == before


@pytest.mark.smoke
def test_a_matching_supplied_digest_needs_no_rebinding() -> None:
    """Agreement is not a conflict and must not churn dependent identities."""

    implementation = _library_implementation(artifact_sha256=_distinct_digest("timm"))
    before = deepcopy(implementation)

    assert bind_library_artifact_digest(implementation, list(_ROUTED_PACKAGES)) is False
    assert implementation == before


@pytest.mark.smoke
def test_binding_is_idempotent_across_repeated_normalization() -> None:
    """A cached, already-normalized artifact re-normalizes without churn.

    ``_preserve_uncommitted_author_result`` writes a NORMALIZED artifact that the
    commit path later re-normalizes, so a second pass must be a no-op: a rebind
    would churn ``recipe_revision`` and every identity derived from it.
    """

    implementation = _library_implementation()
    assert bind_library_artifact_digest(implementation, list(_ROUTED_PACKAGES)) is True
    after_first = deepcopy(implementation)

    assert bind_library_artifact_digest(implementation, list(_ROUTED_PACKAGES)) is False
    assert implementation == after_first

    unresolvable = _library_implementation(distribution="not-in-this-environment")
    assert bind_library_artifact_digest(unresolvable, list(_ROUTED_PACKAGES)) is True
    after_null = deepcopy(unresolvable)

    assert bind_library_artifact_digest(unresolvable, list(_ROUTED_PACKAGES)) is False
    assert unresolvable == after_null


@pytest.mark.smoke
def test_a_malformed_supplied_digest_is_refused() -> None:
    """A non-canonical claim is refused rather than quietly replaced."""

    implementation = _library_implementation(artifact_sha256="deadbeef")

    with pytest.raises(RecipeError) as raised:
        bind_library_artifact_digest(implementation, list(_ROUTED_PACKAGES))

    assert str(raised.value) == "supplied artifact_sha256 is not a canonical sha256 digest"


@pytest.mark.smoke
def test_a_declared_version_that_contradicts_the_environment_is_refused() -> None:
    """The digest pins a build, so a version disagreement is not bindable."""

    implementation = _library_implementation(version="0.9.16")

    with pytest.raises(RecipeError) as raised:
        bind_library_artifact_digest(implementation, list(_ROUTED_PACKAGES))

    assert str(raised.value) == (
        "recipe declares 'timm' version '0.9.16' but the routed environment installs '1.0.9'"
    )


@pytest.mark.smoke
def test_an_unrouted_distribution_degrades_to_an_honest_null() -> None:
    """One odd model must not abort a campaign; it records what is true."""

    implementation = _library_implementation(distribution="not-in-this-environment")

    assert bind_library_artifact_digest(implementation, list(_ROUTED_PACKAGES)) is True
    assert implementation["library_recipe"]["artifact_sha256"] is None


@pytest.mark.smoke
def test_an_empty_inventory_degrades_rather_than_aborting() -> None:
    """An unlocked or unavailable target yields null, never a fabricated digest."""

    implementation = _library_implementation()

    assert bind_library_artifact_digest(implementation, []) is True
    assert implementation["library_recipe"]["artifact_sha256"] is None


@pytest.mark.smoke
def test_distribution_names_are_compared_canonically() -> None:
    """Underscore/dot/case spellings name the same installed distribution."""

    assert (
        resolve_environment_artifact_digest(
            list(_ROUTED_PACKAGES), distribution="TorchVision", version="0.21.0"
        )
        == _distinct_digest("torchvision")
    )


@pytest.mark.smoke
def test_an_ambiguous_inventory_is_refused() -> None:
    """Two rows for one distribution cannot pin a build."""

    duplicated = [
        *_ROUTED_PACKAGES,
        {
            "name": "timm",
            "version": "1.0.9",
            "build": "pyhd8ed1ab_1",
            "url": "https://conda.anaconda.org/conda-forge/noarch/timm-1.0.9-1.conda",
            "sha256": _distinct_digest("timm-rebuild"),
        },
    ]

    with pytest.raises(RecipeError) as raised:
        resolve_environment_artifact_digest(duplicated, distribution="timm", version="1.0.9")

    assert str(raised.value) == "environment inventory names distribution 'timm' ambiguously"


@pytest.mark.smoke
def test_a_non_declarative_implementation_is_untouched() -> None:
    """Only R1 declarative recipes carry an installed-distribution digest."""

    implementation: dict[str, Any] = {"recipe_type": "port", "library_recipe": None}
    before = deepcopy(implementation)

    assert bind_library_artifact_digest(implementation, list(_ROUTED_PACKAGES)) is False
    assert implementation == before


@pytest.mark.smoke
def test_the_digest_leaf_left_the_author_identity_surface() -> None:
    """The model-v3 ownership map -- the one vet identity reads -- says reducer."""

    assert _DIGEST_LEAF in schema_owner_paths(
        MODEL_SCHEMA_VERSION_V3, SchemaOwner.REDUCER_DERIVED
    )
    assert _DIGEST_LEAF not in schema_owner_paths(
        MODEL_SCHEMA_VERSION_V3, SchemaOwner.AUTHOR_GATED
    )

    facts = make_author_proposal("m_r1_ownership")["proposed_facts"]
    authored = authored_fact_leaves(facts, schema_version=MODEL_SCHEMA_VERSION_V3)

    assert "implementation.library_recipe.artifact_sha256" not in authored
    assert "implementation.library_recipe.distribution" in authored


@pytest.mark.smoke
def test_the_pre_gate_rebind_binds_the_digest_and_its_identities(tmp_path: Path) -> None:
    """The driver's own normalization pass -- not just the helper -- does this.

    The digest must land BEFORE the checker gate, together with a full identity
    rebind, or ``recipe_revision`` would no longer describe the recipe bytes the
    worker executes.
    """

    proposal = _honest_r1_proposal()
    recipe = proposal["proposed_facts"]["implementation"]["library_recipe"]
    recipe["distribution"] = "timm"
    recipe["version"] = "1.0.9"
    artifact = make_proposed_artifact(proposal, {"manifest_sha256": HASH}, tmp_path)
    before = deepcopy(artifact.proposal)

    rebound = _normalize_artifact_modes(
        artifact, _DRIVER_CONFIG, environment_packages=list(_ROUTED_PACKAGES)
    )
    bound_recipe = rebound.proposal["proposed_facts"]["implementation"]["library_recipe"]

    assert bound_recipe["artifact_sha256"] == _distinct_digest("timm")
    assert rebound.proposal["recipe_revision"] != before["recipe_revision"]
    assert (
        rebound.proposal["proposed_facts"]["implementation"]["recipe_revision"]
        == rebound.proposal["recipe_revision"]
    )
    assert rebound.proposal["proposal_sha256"] != before["proposal_sha256"]
    # The source artifact is never mutated in place.
    assert artifact.proposal == before


@pytest.mark.smoke
def test_the_pre_gate_rebind_refuses_a_conflicting_digest(tmp_path: Path) -> None:
    """A model whose claim contradicts its routed environment stops -- typed."""

    proposal = _honest_r1_proposal()
    recipe = proposal["proposed_facts"]["implementation"]["library_recipe"]
    recipe["distribution"] = "timm"
    recipe["version"] = "1.0.9"
    recipe["artifact_sha256"] = _distinct_digest("a-different-build")
    artifact = make_proposed_artifact(proposal, {"manifest_sha256": HASH}, tmp_path)

    with pytest.raises(DriverIntegrationError) as raised:
        _normalize_artifact_modes(
            artifact, _DRIVER_CONFIG, environment_packages=list(_ROUTED_PACKAGES)
        )

    assert str(raised.value) == (
        "supplied artifact_sha256 conflicts with the routed environment: "
        f"supplied {_distinct_digest('a-different-build')}, "
        f"derived {_distinct_digest('timm')}"
    )


@pytest.mark.smoke
def test_an_unknown_routed_intent_yields_no_inventory() -> None:
    """Missing routing degrades to an empty inventory instead of raising."""

    assert _routed_environment_packages(None, "core") == ()


@pytest.mark.smoke
def test_the_real_registry_is_read_without_fabricating_rows() -> None:
    """Every routable intent either yields exact rows or yields nothing.

    An UNLOCKED target is the current shipped state for every intent, and it
    must produce an honest null digest rather than a guess. A LOCKED target must
    produce rows carrying the exact identity fields the derivation reads, so the
    digest binds automatically the moment a target is locked.
    """

    registry = load_environment_registry(target="osx-arm64")

    assert registry.intents, "the registry must expose routable intents"
    for name, intent in registry.intents.items():
        packages = _routed_environment_packages(registry, name)
        if intent.lock.export_bytes is None:
            assert packages == (), name
            continue
        assert packages, name
        for row in packages:
            assert {"name", "version", "sha256"} <= set(row), name
            assert re.fullmatch(r"sha256:[0-9a-f]{64}", str(row["sha256"])), name


@pytest.mark.smoke
@pytest.mark.parametrize("recipe_type", ["declarative-library", "typed-adapter", "port"])
def test_an_executable_recipe_may_record_an_honest_unchecked_verdict(
    recipe_type: str,
) -> None:
    """The author has no interpreter, so 'passed' can no longer be compulsory."""

    proposal = _honest_r1_proposal()
    implementation = proposal["proposed_facts"]["implementation"]
    implementation["recipe_type"] = recipe_type
    implementation["torchlens_import_static_check"] = "not-checked"
    validator = get_validator(AUTHOR_PROPOSAL_SCHEMA_VERSION_V3)
    static_check_errors = [
        error
        for error in validator.iter_errors(proposal)
        if error.json_path.endswith("torchlens_import_static_check")
    ]

    assert static_check_errors == []


@pytest.mark.smoke
def test_the_no_code_sentinel_stays_inexpressible_for_executable_recipes() -> None:
    """Widening the verdict vocabulary must not blur the no-code sentinel."""

    proposal = _honest_r1_proposal()
    implementation = proposal["proposed_facts"]["implementation"]
    implementation["torchlens_import_static_check"] = "not-applicable-no-code"
    errors = [
        error
        for error in get_validator(AUTHOR_PROPOSAL_SCHEMA_VERSION_V3).iter_errors(proposal)
        if error.json_path.endswith("torchlens_import_static_check")
    ]

    assert [error.validator for error in errors] == ["enum"]


@pytest.mark.smoke
def test_the_static_check_docstring_names_the_attester() -> None:
    """The schema must say who may assert 'passed', not merely list it."""

    node = load_schema_resource("model-common.schema.json")["$defs"]["implementation"]
    description = node["properties"]["torchlens_import_static_check"]["description"]

    assert re.search(r"\binterpreter\b", description)
