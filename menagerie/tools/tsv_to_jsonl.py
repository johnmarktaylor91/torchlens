"""Migrate the legacy menagerie TSV into a typed JSONL candidate."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from menagerie.catalog import SOURCE_COLUMNS, SOURCE_TSV
from menagerie.schema import (
    CallableInput,
    CatalogRecord,
    Deferral,
    ExecStringRecipe,
    ExpressionRecipe,
    ImportCallableRecipe,
    InputBuilder,
    KwargsInput,
    MultiInput,
    NoInput,
    Recipe,
    StatementRecipe,
    TensorInput,
    TensorSpec,
    VerificationExpectation,
    validate_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_OUTPUT = PROJECT_ROOT / "menagerie" / "data" / "master_catalog.jsonl"
CANONICAL_DEFERRED_OUTPUT = PROJECT_ROOT / "menagerie" / "data" / "deferred.jsonl"
DEFAULT_OUTPUT = CANONICAL_OUTPUT.with_suffix(CANONICAL_OUTPUT.suffix + ".candidate")
DEFAULT_DEFERRED_OUTPUT = CANONICAL_DEFERRED_OUTPUT.with_suffix(
    CANONICAL_DEFERRED_OUTPUT.suffix + ".candidate"
)
DEFAULT_STATS = PROJECT_ROOT / ".research" / "menagerie-redesign" / "migration_stats.json"
CLASSICS_ZOO = "classics-pytorch"
SOURCE_URL_RE = re.compile(r"(https?://\S+|arXiv:\s*[0-9.]+)", re.I)
TORCH_DTYPE_TOKENS = {
    "float16",
    "float32",
    "float64",
    "bfloat16",
    "int64",
    "long",
    "int",
    "int32",
    "bool",
    "uint8",
    "int8",
    "int16",
    "complex64",
    "complex128",
}


def _read_source_rows(path: Path) -> list[dict[str, str]]:
    """Read raw TSV rows without filtering classics.

    Parameters
    ----------
    path:
        Source TSV path.

    Returns
    -------
    list[dict[str, str]]
        Source rows with an added ``_line_number`` field.
    """

    with path.open(newline="") as handle:
        raw_rows = list(csv.reader(handle, delimiter="\t"))
    if not raw_rows:
        return []
    first = [value.strip() for value in raw_rows[0]]
    data_rows = raw_rows[1:] if first == list(SOURCE_COLUMNS) else raw_rows
    offset = 2 if first == list(SOURCE_COLUMNS) else 1

    rows = []
    for line_number, row in enumerate(data_rows, start=offset):
        if len(row) != len(SOURCE_COLUMNS):
            raise ValueError(f"{path}:{line_number} has {len(row)} columns, expected 9")
        parsed = dict(zip(SOURCE_COLUMNS, row))
        parsed["_line_number"] = str(line_number)
        rows.append(parsed)
    return rows


def _imports_from_code(code: str) -> list[str]:
    """Extract explicit import roots from a recipe string.

    Parameters
    ----------
    code:
        Constructor expression or statement block.

    Returns
    -------
    list[str]
        Imported module names.
    """

    imports: set[str] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module.split(".", maxsplit=1)[0])
    if imports:
        return sorted(imports)

    for match in re.finditer(r"\b(?:import|from)\s+([A-Za-z_][\w.]*)", code):
        imports.add(match.group(1).split(".", maxsplit=1)[0])
    return sorted(imports)


def _classify_recipe(constructor_call: str, input_builder: InputBuilder) -> Recipe:
    """Classify a legacy constructor string into the typed recipe union.

    Parameters
    ----------
    constructor_call:
        Verbatim legacy constructor string.
    input_builder:
        Runtime input builder for the record.

    Returns
    -------
    Recipe
        Typed recipe preserving the constructor string.
    """

    code = constructor_call.strip()
    imports = _imports_from_code(code)
    if "\n" in code or "exec(" in code:
        return ExecStringRecipe(body=constructor_call, imports=imports, input=input_builder)
    if ";" in code or code.startswith(("import ", "from ")):
        return StatementRecipe(code=constructor_call, imports=imports, input=input_builder)
    try:
        parsed = ast.parse(code, mode="eval")
    except SyntaxError:
        return ExpressionRecipe(expr=constructor_call, imports=imports, input=input_builder)
    if isinstance(parsed.body, ast.Call):
        return ImportCallableRecipe(expr=constructor_call, imports=imports, input=input_builder)
    return ExpressionRecipe(expr=constructor_call, imports=imports, input=input_builder)


def _dtype_tokens(dtype: str) -> list[str]:
    """Split a legacy dtype string into normalized tokens.

    Parameters
    ----------
    dtype:
        Legacy input dtype text.

    Returns
    -------
    list[str]
        Normalized dtype tokens.
    """

    tokens = []
    for raw in re.split(r"\s*[+,;]\s*| and ", dtype.lower()):
        token = raw.strip().split()[0]
        if not token:
            continue
        tokens.append(token if token in TORCH_DTYPE_TOKENS else "float32")
    return tokens


def _dtype_for_index(dtype: str, index: int) -> str:
    """Return the dtype token for one tensor position.

    Parameters
    ----------
    dtype:
        Legacy input dtype text.
    index:
        Tensor index.

    Returns
    -------
    str
        Normalized dtype token.
    """

    tokens = _dtype_tokens(dtype)
    if not tokens:
        return "float32"
    if index < len(tokens):
        return tokens[index]
    return tokens[0]


def _dtypes_for_count(dtype: str, count: int) -> list[str]:
    """Resolve legacy dtype text for a fixed number of tensors.

    Parameters
    ----------
    dtype:
        Legacy dtype text.
    count:
        Number of input tensors.

    Returns
    -------
    list[str]
        One dtype token per tensor, matching the legacy broadcast behavior.
    """

    tokens = _dtype_tokens(dtype)
    if not tokens:
        return ["float32"] * count
    if len(tokens) == count:
        return tokens
    return [tokens[0]] * count


def _tensor_spec(shape: Sequence[int], dtype: str, index: int = 0) -> TensorSpec:
    """Build a tensor spec from a concrete shape.

    Parameters
    ----------
    shape:
        Concrete tensor shape.
    dtype:
        Legacy dtype text.
    index:
        Tensor index used for multi-dtype rows.

    Returns
    -------
    TensorSpec
        Typed tensor spec.
    """

    return TensorSpec(shape=tuple(int(dim) for dim in shape), dtype=_dtype_for_index(dtype, index))


def _tensor_specs(shapes: Sequence[Sequence[int]], dtype: str) -> list[TensorSpec]:
    """Build tensor specs with legacy multi-input dtype broadcasting.

    Parameters
    ----------
    shapes:
        Concrete tensor shapes.
    dtype:
        Legacy dtype text.

    Returns
    -------
    list[TensorSpec]
        One tensor spec per shape.
    """

    dtypes = _dtypes_for_count(dtype, len(shapes))
    return [
        TensorSpec(shape=tuple(int(dim) for dim in shape), dtype=dtypes[index])
        for index, shape in enumerate(shapes)
    ]


def _builder_from_structured(shape: str, dtype: str) -> InputBuilder | None:
    """Build an input builder from a JSON/python-literal structured input spec.

    Parameters
    ----------
    shape:
        Legacy input shape text.
    dtype:
        Legacy input dtype text.

    Returns
    -------
    InputBuilder | None
        Typed builder when the row is a structured spec, otherwise ``None``.
    """

    text = shape.strip()
    if not (text.startswith("[{") or text.startswith('{"')):
        return None
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        try:
            obj = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None

    def spec_from_entry(entry: dict[str, Any], index: int = 0) -> TensorSpec:
        """Convert one structured entry into a tensor spec."""

        entry_dtype = str(entry.get("dtype", _dtype_for_index(dtype, index)))
        return TensorSpec(shape=tuple(int(dim) for dim in entry["shape"]), dtype=entry_dtype)

    if isinstance(obj, dict) and "kwargs" in obj and isinstance(obj["kwargs"], dict):
        return KwargsInput(
            specs={str(key): spec_from_entry(value) for key, value in obj["kwargs"].items()}
        )
    if isinstance(obj, dict) and "inputs" in obj and isinstance(obj["inputs"], list):
        return MultiInput(
            specs=[spec_from_entry(value, index) for index, value in enumerate(obj["inputs"])]
        )
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "shape" in obj[0]:
        return MultiInput(specs=[spec_from_entry(value, index) for index, value in enumerate(obj)])
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        specs = []
        for index, entry in enumerate(obj):
            if len(entry) != 1:
                return None
            shape_value = next(iter(entry.values()))
            specs.append(_tensor_spec(shape_value, dtype, index))
        return MultiInput(specs=specs)
    return None


def _concrete_shapes_from_literal(shape: str) -> tuple[int, ...] | list[tuple[int, ...]]:
    """Parse concrete literal shape metadata without prose heuristics.

    Parameters
    ----------
    shape:
        Legacy input shape text.

    Returns
    -------
    tuple[int, ...] | list[tuple[int, ...]]
        Concrete tensor shape or concrete multi-input shapes.
    """

    parsed = ast.literal_eval(shape.strip())
    if isinstance(parsed, tuple) and all(isinstance(value, int) for value in parsed):
        return parsed
    if (
        isinstance(parsed, tuple)
        and parsed
        and all(
            isinstance(item, tuple) and all(isinstance(value, int) for value in item)
            for item in parsed
        )
    ):
        return list(parsed)
    if (
        isinstance(parsed, list)
        and parsed
        and all(
            isinstance(item, (tuple, list)) and all(isinstance(value, int) for value in item)
            for item in parsed
        )
    ):
        return [tuple(item) for item in parsed]
    raise ValueError(f"expected concrete literal shape, got {shape!r}")


def _deferred_callable(reason: str) -> CallableInput:
    """Build a callable marker for an honestly deferred input.

    Parameters
    ----------
    reason:
        Deferral reason to raise if the callable is invoked.

    Returns
    -------
    CallableInput
        Callable input that raises instead of fabricating an input.
    """

    message = reason.replace("\\", "\\\\").replace("'", "\\'")
    expr = f"lambda: (_ for _ in ()).throw(RuntimeError('{message}'))"
    return CallableInput(expr=expr)


def _wrapper_ignores_input(constructor_call: str, shape: str) -> bool:
    """Return whether a legacy wrapper uses the ``(1,)`` ignored-input sentinel.

    Parameters
    ----------
    constructor_call:
        Verbatim legacy constructor string.
    shape:
        Legacy input shape.

    Returns
    -------
    bool
        Whether the row should use ``NoInput``.
    """

    if shape.strip() != "(1,)":
        return False
    lowered = constructor_call.lower()
    return "forward" in lowered and ("lambda self,x" in lowered or "def forward" in lowered)


def _builder_from_shape(
    constructor_call: str, shape: str, dtype: str
) -> tuple[InputBuilder, bool, str | None, bool]:
    """Convert legacy input metadata into a typed input builder.

    Parameters
    ----------
    constructor_call:
        Verbatim legacy constructor string.
    shape:
        Legacy input shape.
    dtype:
        Legacy input dtype.

    Returns
    -------
    tuple[InputBuilder, bool, str | None, bool]
        Builder, ``input_is_real``, optional deferral reason, and whether the shape was prose.
    """

    if _wrapper_ignores_input(constructor_call, shape):
        return NoInput(), False, None, False

    structured = _builder_from_structured(shape, dtype)
    if structured is not None:
        return structured, True, None, False

    try:
        parsed = _concrete_shapes_from_literal(shape)
    except Exception as exc:  # noqa: BLE001 - migration reports honest deferrals.
        reason = f"could not derive concrete runtime input from input_shape={shape!r}: {exc}"
        return _deferred_callable(reason), False, reason, True

    if isinstance(parsed, list):
        return (
            MultiInput(specs=_tensor_specs(parsed, dtype)),
            True,
            None,
            not shape.strip().startswith(("[", "(")),
        )
    return (
        TensorInput(spec=_tensor_spec(parsed, dtype)),
        True,
        None,
        not shape.strip().startswith("("),
    )


def _source_url_from_notes(notes: str) -> str | None:
    """Extract a simple source URL or arXiv marker from notes when present.

    Parameters
    ----------
    notes:
        Legacy notes text.

    Returns
    -------
    str | None
        First source token, if present.
    """

    match = SOURCE_URL_RE.search(notes)
    return match.group(1).rstrip(";,.)") if match else None


def _record_from_row(row: dict[str, str]) -> tuple[CatalogRecord, bool]:
    """Convert one non-classics TSV row into a typed catalog record.

    Parameters
    ----------
    row:
        Raw TSV row.

    Returns
    -------
    tuple[CatalogRecord, bool]
        Migrated record and whether the input shape was prose/deferred.
    """

    input_builder, input_is_real, deferral_reason, prose_input = _builder_from_shape(
        row["constructor_call"], row["input_shape"], row["input_dtype"]
    )
    recipe = _classify_recipe(row["constructor_call"], input_builder)
    expectation = (
        VerificationExpectation.deferred
        if deferral_reason is not None
        else VerificationExpectation.forward_required
    )
    return (
        CatalogRecord(
            name=row["name"],
            zoo=row["zoo"],
            family=row["family"],
            domain=row["domain"],
            era=row["era"],
            recipe=recipe,
            input=input_builder,
            input_shape=row["input_shape"],
            input_dtype=row["input_dtype"],
            source_url=_source_url_from_notes(row["notes"]),
            added_wave=None,
            notes=row["notes"],
            input_is_real=input_is_real,
            verification_expectation=expectation,
            deferral=Deferral(reason=deferral_reason) if deferral_reason else None,
            legacy={
                "constructor_call": row["constructor_call"],
                "input_shape": row["input_shape"],
                "input_dtype": row["input_dtype"],
                "line_number": row["_line_number"],
            },
        ),
        prose_input,
    )


def migrate_tsv(
    source_tsv: Path = SOURCE_TSV,
    output_path: Path = DEFAULT_OUTPUT,
    stats_path: Path = DEFAULT_STATS,
    *,
    deferred_output_path: Path = DEFAULT_DEFERRED_OUTPUT,
    write_canonical: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    """Migrate non-classics TSV rows to JSONL candidate records.

    Parameters
    ----------
    source_tsv:
        Legacy TSV path.
    output_path:
        Candidate JSONL path for forward-required records.
    stats_path:
        Migration stats JSON path.
    deferred_output_path:
        JSONL path for honestly deferred records.
    write_canonical:
        Whether writing canonical JSONL source paths is explicitly allowed.
    write:
        Whether to write JSONL and stats artifacts.

    Returns
    -------
    dict[str, Any]
        Migration stats.
    """

    if write and not write_canonical:
        canonical_paths = {
            CANONICAL_OUTPUT.resolve(),
            CANONICAL_DEFERRED_OUTPUT.resolve(),
        }
        requested_paths = {output_path.resolve(), deferred_output_path.resolve()}
        if canonical_paths & requested_paths:
            raise ValueError(
                "refusing to write canonical menagerie JSONL sources without --write-canonical"
            )

    raw_rows = _read_source_rows(source_tsv)
    classics_count = sum(row["zoo"] == CLASSICS_ZOO for row in raw_rows)
    non_classics_rows = [row for row in raw_rows if row["zoo"] != CLASSICS_ZOO]
    natural_key_counts = Counter((row["name"], row["zoo"]) for row in non_classics_rows)
    natural_key_seen: Counter[tuple[str, str]] = Counter()
    records: list[CatalogRecord] = []
    prose_inputs = []
    for row in non_classics_rows:
        natural_key = (row["name"], row["zoo"])
        natural_key_seen[natural_key] += 1
        record, prose_input = _record_from_row(row)
        if natural_key_counts[natural_key] > 1 and natural_key_seen[natural_key] > 1:
            record.variant = f"variant-{natural_key_seen[natural_key]}"
        records.append(record)
        if prose_input:
            prose_inputs.append(record.name)

    non_classics_count = len(raw_rows) - classics_count
    if len(records) != non_classics_count:
        raise AssertionError(
            f"mapped {len(records)} records from {non_classics_count} non-classics rows"
        )
    validate_records(records)
    forward_required_records = [
        record
        for record in records
        if record.verification_expectation == VerificationExpectation.forward_required
    ]
    deferred_records = [
        record
        for record in records
        if record.verification_expectation == VerificationExpectation.deferred
    ]

    recipe_counts = Counter(record.recipe.type for record in records)
    input_counts = Counter(record.input.kind for record in records)
    deferred = [
        {"name": record.name, "zoo": record.zoo, "reason": record.deferral.reason}
        for record in deferred_records
        if record.deferral is not None
    ]
    if len(forward_required_records) + len(deferred_records) != non_classics_count:
        raise AssertionError(
            "mapped rows must partition into forward-required records and deferred records"
        )
    stats: dict[str, Any] = {
        "source_tsv": str(source_tsv),
        "output_path": str(output_path),
        "deferred_output_path": str(deferred_output_path),
        "total_tsv_rows": len(raw_rows),
        "classics_rows_skipped": classics_count,
        "non_classics_records_produced": len(records),
        "forward_required_records_written": len(forward_required_records),
        "recipe_type_counts": dict(sorted(recipe_counts.items())),
        "input_builder_kind_counts": dict(sorted(input_counts.items())),
        "input_is_real_counts": {
            "true": sum(record.input_is_real for record in records),
            "false": sum(not record.input_is_real for record in records),
        },
        "deferred_count": len(deferred),
        "deferred": deferred,
        "prose_input_count": len(prose_inputs),
        "prose_input_names": prose_inputs,
    }

    if write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "".join(record.model_dump_json() + "\n" for record in forward_required_records),
            encoding="utf-8",
        )
        deferred_output_path.parent.mkdir(parents=True, exist_ok=True)
        deferred_output_path.write_text(
            "".join(record.model_dump_json() + "\n" for record in deferred_records),
            encoding="utf-8",
        )
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TSV-to-JSONL migration command.

    Parameters
    ----------
    argv:
        Optional CLI arguments.

    Returns
    -------
    int
        Process exit status.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tsv", type=Path, default=SOURCE_TSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--deferred-output", type=Path, default=DEFAULT_DEFERRED_OUTPUT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument(
        "--write-canonical",
        action="store_true",
        help=(
            "write the canonical committed JSONL sources instead of .candidate files; "
            "intended only for one-time migrations"
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="dry-run validation only; do not write candidate JSONL or stats",
    )
    args = parser.parse_args(argv)
    output_path = args.output
    deferred_output_path = args.deferred_output
    if args.write_canonical:
        if output_path == DEFAULT_OUTPUT:
            output_path = CANONICAL_OUTPUT
        if deferred_output_path == DEFAULT_DEFERRED_OUTPUT:
            deferred_output_path = CANONICAL_DEFERRED_OUTPUT
        print(
            "WARNING: writing canonical menagerie JSONL sources; this can overwrite "
            "committed catalog data.",
            file=sys.stderr,
        )
    stats = migrate_tsv(
        args.source_tsv,
        output_path,
        args.stats,
        deferred_output_path=deferred_output_path,
        write_canonical=args.write_canonical,
        write=not args.no_write,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
