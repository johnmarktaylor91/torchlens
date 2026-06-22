"""Golden tests for menagerie stable identity helpers."""

from __future__ import annotations

from pathlib import Path

from menagerie.catalog import CatalogRow
from menagerie.generate_menagerie import (
    RenderResult,
    append_manifest as append_render_manifest,
    completed_stable_ids as completed_render_stable_ids,
)
from menagerie.identity import canonical_recipe_v1, recipe_revision_sha256
from menagerie.validate_menagerie import (
    ValidationResult,
    append_manifest as append_validation_manifest,
    completed_stable_ids as completed_validation_stable_ids,
)


def _row(**overrides: object) -> CatalogRow:
    """Build a compact catalog row fixture for identity tests.

    Parameters
    ----------
    overrides:
        Field overrides for the default catalog row.

    Returns
    -------
    CatalogRow
        Test catalog row.
    """

    data = {
        "model_id": 1,
        "display_index": 1,
        "stable_id": "m000001",
        "name": "ToyNet",
        "variant": "",
        "family": "toy",
        "family_normalized": "toy",
        "domain": "vision",
        "zoo": "unit-zoo",
        "constructor_call": "torch.nn.Linear(4, 2)",
        "input_shape": "(1, 4)",
        "input_dtype": "float32",
        "era": "2024",
        "verified": True,
        "notes": "",
        "source": "catalog",
        "recipe_revision_sha256": "",
    }
    data.update(overrides)
    return CatalogRow(**data)


def test_catalog_recipe_golden_vector() -> None:
    """Catalog recipe serialization is frozen by a hardcoded digest."""

    row = _row()
    assert (
        canonical_recipe_v1(row)
        == '{"recipe":{"constructor_call":"torch.nn.Linear(4, 2)"},"recipe_scheme_version":1,"source":"catalog"}'
    )
    assert (
        recipe_revision_sha256(row)
        == "db63b4317b50f95020df4ac9325ecb7a9735b4c3563288f746f505b24bb90130"  # pragma: allowlist secret
    )


def test_classics_recipe_golden_vector() -> None:
    """Classics recipe serialization uses module path and build function only."""

    row = _row(
        name="3D Gaussian Splatting (learnable per-Gaussian scene params)",
        zoo="classics-pytorch",
        constructor_call="menagerie.classics.gaussian_splatting.build_gaussian_splatting()",
        input_shape="(1, 16)",
        source="classics",
    )
    assert (
        canonical_recipe_v1(row)
        == '{"recipe":{"build_fn":"build_gaussian_splatting","module_path":"menagerie.classics.gaussian_splatting"},"recipe_scheme_version":1,"source":"classics"}'
    )
    assert (
        recipe_revision_sha256(row)
        == "bb531318c963124274d404c246fd777b8dd07921e018198f236b965a770f3e54"  # pragma: allowlist secret
    )


def test_input_only_change_does_not_change_recipe_hash() -> None:
    """Input spec repairs preserve recipe history."""

    assert recipe_revision_sha256(_row()) == recipe_revision_sha256(_row(input_shape="(2, 4)"))


def test_constructor_change_changes_recipe_hash() -> None:
    """Constructor changes create a new recipe revision."""

    assert recipe_revision_sha256(_row()) != recipe_revision_sha256(
        _row(constructor_call="torch.nn.Linear(4, 3)")
    )


def test_render_manifest_resume_uses_stable_id_for_same_name_siblings(tmp_path: Path) -> None:
    """Render resume keeps same-name siblings distinct by stable ID."""

    manifest_path = tmp_path / "manifest.tsv"
    append_render_manifest(
        manifest_path,
        RenderResult(
            name="SharedName",
            model_id=1,
            status="rendered",
            n_nodes=3,
            render_path="SharedName.svg",
            elapsed=0.1,
            dependency_cluster="unit",
            error="",
            graph_shape_hash="shape-a",
            stable_id="m1",
            recipe_revision_sha256="recipe-a",
        ),
    )

    done = completed_render_stable_ids(manifest_path, retry_failed=False)
    rows = [_row(name="SharedName", stable_id="m1"), _row(name="SharedName", stable_id="m2")]

    assert [row.stable_id for row in rows if row.stable_id not in done] == ["m2"]


def test_validation_manifest_resume_uses_stable_id_for_same_name_siblings(tmp_path: Path) -> None:
    """Validation resume keeps same-name siblings distinct by stable ID."""

    manifest_path = tmp_path / "validation_manifest.tsv"
    append_validation_manifest(
        manifest_path,
        ValidationResult(
            name="SharedName",
            model_id=1,
            status="validated",
            n_ops=3,
            validate_metadata_ok=True,
            scope="forward",
            elapsed=0.1,
            dependency_cluster="unit",
            error="",
            graph_shape_hash="shape-a",
            stable_id="m1",
            recipe_revision_sha256="recipe-a",
        ),
    )

    done = completed_validation_stable_ids(manifest_path, revalidate_failed=False)
    rows = [_row(name="SharedName", stable_id="m1"), _row(name="SharedName", stable_id="m2")]

    assert [row.stable_id for row in rows if row.stable_id not in done] == ["m2"]
