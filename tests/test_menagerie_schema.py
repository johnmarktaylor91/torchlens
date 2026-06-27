"""Tests for the typed menagerie catalog schema."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from menagerie.catalog import SOURCE_JSONL
from menagerie.recipe import build_input_from_record
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
    recipe_is_quarantined,
    recipe_uses_code_execution,
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


def test_quarantine_is_narrowed_to_arbitrary_exec_recipes() -> None:
    """Simple expressions/statements are code-exec but not arbitrary-exec quarantined."""

    input_builder = _tensor_input()
    import_callable = ImportCallableRecipe(
        expr="torch.nn.Identity()", imports=["torch"], input=input_builder
    )
    expression = ExpressionRecipe(expr="factory()", input=input_builder, quarantine=True)
    simple_statement = StatementRecipe(
        code="import torch\nmodel = torch.nn.Identity()", input=input_builder, quarantine=True
    )
    arbitrary_statement = StatementRecipe(
        code="import torch\nmodel = type('W', (torch.nn.Module,), {'forward': lambda self, x: x})()",
        input=input_builder,
    )
    exec_string = ExecStringRecipe(body="class W: pass\nmodel = W()", input=input_builder)

    assert recipe_uses_code_execution(import_callable) is False
    assert recipe_uses_code_execution(expression) is True
    assert recipe_uses_code_execution(simple_statement) is True
    assert recipe_is_quarantined(import_callable) is False
    assert recipe_is_quarantined(expression) is False
    assert recipe_is_quarantined(simple_statement) is False
    assert recipe_is_quarantined(arbitrary_statement) is True
    assert recipe_is_quarantined(exec_string) is True
    assert expression.quarantine is False
    assert simple_statement.quarantine is False
    assert arbitrary_statement.quarantine is True


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


def test_widedeep_tabmlp_callable_input_is_in_range() -> None:
    """m3397 WideDeep-TabMlp: callable input produces valid categorical indices.

    The model has cat_embed_input=[('cat0',10,4),('cat1',20,4)], so col 0
    must be in [0,10) and col 1 in [0,20).  The old spec ([1,13] float32)
    caused an IndexError before TorchLens even ran; this gate verifies the
    faithful callable override is correct.
    """
    records = load_jsonl(SOURCE_JSONL)
    (m3397,) = [r for r in records if r.name == "WideDeep-TabMlp" and r.zoo.startswith("web(")]

    # Input kind must be callable (not the old generic tensor spec).
    assert m3397.input.kind == "callable", f"expected callable input, got {m3397.input.kind!r}"

    # Recipe is not quarantined -- the model code contains no arbitrary exec.
    assert m3397.recipe.quarantine is False, (
        "WideDeep-TabMlp recipe should not be quarantined after input fix"
    )

    # Build the input and verify shape and in-range categorical indices.
    x = build_input_from_record(m3397)

    assert isinstance(x, torch.Tensor), f"expected Tensor, got {type(x)}"
    assert x.shape == (1, 4), f"expected shape (1,4), got {x.shape}"

    # cat0 column (index 0): embedding table size 10, indices must be in [0, 10).
    cat0_val = int(x[0, 0].item())
    assert 0 <= cat0_val < 10, f"cat0 index {cat0_val} out of range [0, 10)"

    # cat1 column (index 1): embedding table size 20, indices must be in [0, 20).
    cat1_val = int(x[0, 1].item())
    assert 0 <= cat1_val < 20, f"cat1 index {cat1_val} out of range [0, 20)"
