"""Typed schema for the menagerie JSONL migration candidate."""

from __future__ import annotations

import json
import ast
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VerificationExpectation(str, Enum):
    """Expected validation posture for a catalog record."""

    forward_required = "forward_required"
    deferred = "deferred"
    unsupported = "unsupported"


class TensorSpec(BaseModel):
    """Declarative tensor spec used by typed input builders.

    Parameters
    ----------
    shape:
        Tensor shape. Symbolic strings are allowed for descriptive metadata, but runtime builders
        should use concrete integer shapes.
    dtype:
        Torch dtype name without the ``torch.`` prefix.
    requires_grad:
        Whether the synthetic tensor should require gradients.
    """

    model_config = ConfigDict(extra="forbid")

    shape: tuple[int | str, ...]
    dtype: str = "float32"
    requires_grad: bool = False


class TensorInput(BaseModel):
    """Input builder for one synthetic tensor."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["tensor"] = "tensor"
    spec: TensorSpec


class MultiInput(BaseModel):
    """Input builder for positional multi-tensor inputs."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["multi"] = "multi"
    specs: list[TensorSpec]
    as_tuple: bool = False


class KwargsInput(BaseModel):
    """Input builder for keyword tensor inputs."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["kwargs"] = "kwargs"
    specs: dict[str, TensorSpec]


class NoInput(BaseModel):
    """Input builder for wrappers whose forward method ignores its argument."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["none"] = "none"
    sentinel_shape: tuple[int, ...] = (1,)
    sentinel_dtype: str = "float32"


class CallableInput(BaseModel):
    """Input builder backed by an eval-able zero-argument callable expression."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["callable"] = "callable"
    expr: str
    imports: list[str] = Field(default_factory=list)


InputBuilder = Annotated[
    TensorInput | MultiInput | KwargsInput | NoInput | CallableInput,
    Field(discriminator="kind"),
]


class ImportCallableRecipe(BaseModel):
    """Recipe backed by an eval-able constructor expression and explicit imports."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["import-callable"] = "import-callable"
    expr: str
    imports: list[str] = Field(default_factory=list)
    input: InputBuilder


class ExecStringRecipe(BaseModel):
    """Quarantined legacy recipe backed by a multi-line exec string."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["exec-string"] = "exec-string"
    body: str
    imports: list[str] = Field(default_factory=list)
    input: InputBuilder
    output_name: str = "model"
    quarantine: Literal[True] = True


class ExpressionRecipe(BaseModel):
    """Legacy recipe backed by an eval expression."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["expression"] = "expression"
    expr: str
    imports: list[str] = Field(default_factory=list)
    input: InputBuilder
    quarantine: bool = False

    @model_validator(mode="after")
    def normalize_quarantine(self) -> ExpressionRecipe:
        """Normalize legacy explicit quarantine markers to the narrowed default."""

        self.quarantine = False
        return self


class StatementRecipe(BaseModel):
    """Legacy recipe backed by statements that assign a model variable."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["statement"] = "statement"
    code: str
    imports: list[str] = Field(default_factory=list)
    input: InputBuilder
    output_name: str = "model"
    quarantine: bool = False

    @model_validator(mode="after")
    def normalize_quarantine(self) -> StatementRecipe:
        """Normalize quarantine to statements that contain arbitrary Python code."""

        self.quarantine = statement_contains_arbitrary_code(self.code)
        return self


Recipe = Annotated[
    ImportCallableRecipe | ExecStringRecipe | ExpressionRecipe | StatementRecipe,
    Field(discriminator="type"),
]


ARBITRARY_STATEMENT_NODES = (
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.For,
    ast.FunctionDef,
    ast.If,
    ast.Lambda,
    ast.Try,
    ast.While,
    ast.With,
)
ARBITRARY_CALL_NAMES = {"compile", "eval", "exec", "__import__"}


def statement_contains_arbitrary_code(code: str) -> bool:
    """Return whether a statement recipe contains arbitrary Python code.

    Parameters
    ----------
    code:
        Statement recipe source.

    Returns
    -------
    bool
        ``True`` for dynamic code execution, local class/function definitions, lambdas, or
        control-flow blocks. Syntax failures are treated as arbitrary because they cannot be safely
        classified as simple import-plus-constructor statements.
    """

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ARBITRARY_STATEMENT_NODES):
            return True
        if isinstance(node, ast.Call) and _call_name(node.func) in ARBITRARY_CALL_NAMES:
            return True
    return False


def _call_name(node: ast.AST) -> str:
    """Return the terminal call name for a call expression.

    Parameters
    ----------
    node:
        Function node from an ``ast.Call``.

    Returns
    -------
    str
        Name or attribute suffix used for conservative matching.
    """

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def recipe_uses_code_execution(recipe: Recipe) -> bool:
    """Return whether a recipe is built through Python code execution.

    Parameters
    ----------
    recipe:
        Typed recipe.

    Returns
    -------
    bool
        ``True`` for exec-string, expression, and statement recipe forms.
    """

    return recipe.type in {"exec-string", "expression", "statement"}


def recipe_is_quarantined(recipe: Recipe) -> bool:
    """Return whether a recipe is quarantined under the narrowed arbitrary-exec definition.

    Parameters
    ----------
    recipe:
        Typed recipe.

    Returns
    -------
    bool
        ``True`` for exec strings and statement recipes that contain arbitrary code.
    """

    if recipe.type == "exec-string":
        return True
    if recipe.type == "statement":
        return statement_contains_arbitrary_code(recipe.code)
    return False


class Deferral(BaseModel):
    """Reason a record is not required to pass forward validation yet."""

    model_config = ConfigDict(extra="forbid")

    reason: str


class CatalogRecord(BaseModel):
    """One typed non-classics menagerie catalog record."""

    model_config = ConfigDict(extra="forbid")

    name: str
    zoo: str
    variant: str = ""
    family: str
    domain: str
    era: str
    recipe: Recipe
    input: InputBuilder
    input_shape: str | None = None
    input_dtype: str | None = None
    source_url: str | None = None
    added_wave: str | None = None
    notes: str = ""
    input_is_real: bool = True
    verification_expectation: VerificationExpectation = VerificationExpectation.forward_required
    deferral: Deferral | None = None
    legacy: dict[str, Any] = Field(default_factory=dict)

    @field_validator("zoo")
    @classmethod
    def reject_classics_zoo(cls, value: str) -> str:
        """Reject classics rows because the registry is their sole source."""

        if value == "classics-pytorch":
            raise ValueError("classics-pytorch rows live in the classics registry, not JSONL")
        return value

    @model_validator(mode="after")
    def validate_input_consistency(self) -> CatalogRecord:
        """Require the top-level and recipe-level input builders to match."""

        if self.input != self.recipe.input:
            raise ValueError("CatalogRecord.input must match CatalogRecord.recipe.input")
        if (
            self.verification_expectation == VerificationExpectation.deferred
            and self.deferral is None
        ):
            raise ValueError("deferred records must include a deferral reason")
        return self


def load_jsonl(path: str | Path) -> list[CatalogRecord]:
    """Load and validate catalog records from a JSONL file.

    Parameters
    ----------
    path:
        JSONL file path.

    Returns
    -------
    list[CatalogRecord]
        Validated catalog records.
    """

    records: list[CatalogRecord] = []
    jsonl_path = Path(path)
    with jsonl_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(CatalogRecord.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{jsonl_path}:{line_number}: {exc}") from exc
    validate_records(records)
    return records


def validate_records(records: list[CatalogRecord]) -> None:
    """Validate cross-record catalog invariants.

    Parameters
    ----------
    records:
        Catalog records to validate.

    Raises
    ------
    ValueError
        If natural keys are duplicated or serialization is invalid.
    """

    seen: set[tuple[str, str, str]] = set()
    for record in records:
        key = (record.name, record.zoo, record.variant)
        if key in seen:
            raise ValueError(f"duplicate natural key: {key!r}")
        seen.add(key)
        json.loads(record.model_dump_json())
