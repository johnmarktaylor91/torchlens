"""Tests for the typed menagerie catalog schema."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from menagerie.schema import (
    CallableInput,
    CatalogRecord,
    ExecStringRecipe,
    ExpressionRecipe,
    ImportCallableRecipe,
    KwargsInput,
    MultiInput,
    NoInput,
    StatementRecipe,
    TensorInput,
    TensorSpec,
    load_jsonl,
)


def _tensor_input() -> TensorInput:
    """Build a simple tensor input builder for tests.

    Returns
    -------
    TensorInput
        Concrete tensor input builder.
    """

    return TensorInput(spec=TensorSpec(shape=(1, 3, 16, 16), dtype="float32"))


def _record() -> CatalogRecord:
    """Build a valid catalog record for tests.

    Returns
    -------
    CatalogRecord
        Valid non-classics record.
    """

    input_builder = _tensor_input()
    return CatalogRecord(
        name="toy",
        zoo="unit",
        family="toy-family",
        domain="vision/classification-backbone",
        era="2026",
        recipe=ImportCallableRecipe(
            expr="torch.nn.Identity()", imports=["torch"], input=input_builder
        ),
        input=input_builder,
        input_shape="(1, 3, 16, 16)",
        input_dtype="float32",
        input_is_real=True,
    )


def test_valid_record_round_trips_jsonl(tmp_path: Path) -> None:
    """A valid record serializes to JSONL and loads back unchanged."""

    record = _record()
    path = tmp_path / "catalog.jsonl"
    path.write_text(record.model_dump_json() + "\n")

    loaded = load_jsonl(path)

    assert loaded == [record]


def test_classics_zoo_is_rejected() -> None:
    """The JSONL schema rejects classics rows."""

    record_data = _record().model_dump()
    record_data["zoo"] = "classics-pytorch"

    with pytest.raises(ValidationError, match="classics-pytorch"):
        CatalogRecord.model_validate(record_data)


def test_unknown_field_is_rejected() -> None:
    """Unknown fields fail because the schema is a migration contract."""

    record_data = _record().model_dump()
    record_data["surprise"] = "not allowed"

    with pytest.raises(ValidationError, match="surprise"):
        CatalogRecord.model_validate(record_data)


def test_each_recipe_variant_constructs() -> None:
    """Every recipe variant accepted by Phase 2c can be constructed."""

    input_builder = _tensor_input()

    recipes = [
        ImportCallableRecipe(expr="torch.nn.Identity()", imports=["torch"], input=input_builder),
        ExecStringRecipe(body="import torch\nmodel=torch.nn.Identity()", input=input_builder),
        ExpressionRecipe(expr="torch.nn.Identity()", imports=["torch"], input=input_builder),
        StatementRecipe(code="import torch\nmodel=torch.nn.Identity()", input=input_builder),
    ]

    assert [recipe.type for recipe in recipes] == [
        "import-callable",
        "exec-string",
        "expression",
        "statement",
    ]


def test_each_input_variant_constructs() -> None:
    """Every input-builder variant accepted by Phase 2c can be constructed."""

    inputs = [
        TensorInput(spec=TensorSpec(shape=(1, 3), dtype="float32")),
        MultiInput(specs=[TensorSpec(shape=(1, 3)), TensorSpec(shape=(1, 4))]),
        KwargsInput(specs={"input_ids": TensorSpec(shape=(1, 16), dtype="long")}),
        NoInput(),
        CallableInput(expr="lambda: torch.randn(1, 3)", imports=["torch"]),
    ]

    assert [input_builder.kind for input_builder in inputs] == [
        "tensor",
        "multi",
        "kwargs",
        "none",
        "callable",
    ]
