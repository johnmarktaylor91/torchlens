"""Pure menagerie recipe builders: row -> (model, input).

This module executes audited catalog recipe code via exec/eval. It must not own dependency
installation, environment mutation, subprocess orchestration, or cache/device cleanup policy.
"""

from __future__ import annotations

import importlib
import ast
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from menagerie.catalog import DEFERRED_JSONL, SOURCE_JSONL, CatalogRow
from menagerie.schema import CatalogRecord, load_jsonl


def _torch_dtype(dtype: str) -> Any:
    """Resolve a schema dtype string to a torch dtype.

    Parameters
    ----------
    dtype:
        Dtype token.

    Returns
    -------
    Any
        Torch dtype object.
    """

    import torch

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "long": torch.int64,
        "int": torch.int32,
        "int32": torch.int32,
        "bool": torch.bool,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }
    try:
        return dtype_map[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported schema dtype={dtype!r}") from exc


def _tensor_from_spec(shape: Sequence[int | str], dtype: str) -> Any:
    """Create a tensor from a typed tensor spec.

    Parameters
    ----------
    shape:
        Tensor shape from the schema.
    dtype:
        Dtype token from the schema.

    Returns
    -------
    Any
        Torch tensor.
    """

    import torch

    concrete_shape = tuple(int(dim) for dim in shape)
    torch_dtype = _torch_dtype(dtype)
    if torch_dtype.is_floating_point or torch_dtype.is_complex:
        return torch.randn(concrete_shape, dtype=torch_dtype)
    return torch.zeros(concrete_shape, dtype=torch_dtype)


def _record_tensor_dtype(record: Any, shape: Sequence[int | str], dtype: str, index: int) -> str:
    """Return the runtime dtype for a typed input tensor.

    Parameters
    ----------
    record:
        Catalog record carrying model and input-shape context.
    shape:
        Tensor shape from the schema.
    dtype:
        Declared dtype token from the schema.
    index:
        Positional tensor index within a multi-input record.

    Returns
    -------
    str
        Dtype token to use for synthetic input construction.
    """

    zoo = str(getattr(record, "zoo", ""))
    input_shape = str(getattr(record, "input_shape", ""))
    if not zoo.startswith("torch_geometric"):
        return dtype
    concrete_shape = tuple(int(dim) for dim in shape)
    if "edge_index" in input_shape and len(concrete_shape) == 2 and concrete_shape[0] == 2:
        return "long"
    if "batch" in input_shape and len(concrete_shape) == 1 and index > 0:
        return "long"
    if "pos" in input_shape and len(concrete_shape) == 2 and concrete_shape[1] == 3:
        return "float32"
    return dtype


def _tensor_from_record_spec(
    record: Any, shape: Sequence[int | str], dtype: str, index: int
) -> Any:
    """Create a tensor from a typed spec plus catalog-record context.

    Parameters
    ----------
    record:
        Catalog record carrying model and input-shape context.
    shape:
        Tensor shape from the schema.
    dtype:
        Dtype token from the schema.
    index:
        Positional tensor index within a multi-input record.

    Returns
    -------
    Any
        Torch tensor.
    """

    return _tensor_from_spec(shape, _record_tensor_dtype(record, shape, dtype, index))


def _special_record_input(record: Any) -> Any | None:
    """Return special-case real inputs for typed records when required.

    Parameters
    ----------
    record:
        Catalog record.

    Returns
    -------
    Any | None
        Input object, or ``None`` when the generic builder should be used.
    """

    if (
        getattr(record, "zoo", "") == "torch_geometric"
        and getattr(record, "name", "") == "JumpingKnowledge"
    ):
        import torch

        return [torch.randn(8, 64), torch.randn(8, 64), torch.randn(8, 64)]
    if getattr(record, "zoo", "") == "torch_geometric" and getattr(record, "name", "") in {
        "SAGPooling",
        "TopKPooling",
    }:
        import torch

        edge_index = torch.tensor(
            [[0, 1, 2, 3, 0, 2, 1, 3], [1, 2, 3, 0, 2, 0, 3, 1]],
            dtype=torch.long,
        )
        return [torch.randn(8, 16), edge_index, None, torch.zeros(8, dtype=torch.long)]
    return None


def _typed_namespace(imports: Sequence[str]) -> dict[str, Any]:
    """Build a permissive namespace for typed recipe/input evaluation.

    Parameters
    ----------
    imports:
        Explicit import roots declared on a typed recipe or callable input.

    Returns
    -------
    dict[str, Any]
        Evaluation namespace.
    """

    namespace: dict[str, Any] = {}
    module_names = {
        "torch",
        "torchvision",
        "timm",
        "monai",
        "snntorch",
        "torchaudio",
        "torch_geometric",
        "torchsr",
        "transformers",
        "diffusers",
        "segmentation_models_pytorch",
        *imports,
    }
    for module_name in sorted(module_names):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        namespace[module_name] = module
        root_name = module_name.partition(".")[0]
        if root_name and root_name not in namespace:
            try:
                namespace[root_name] = importlib.import_module(root_name)
            except ImportError:
                pass
        if module_name == "segmentation_models_pytorch":
            namespace["smp"] = module
        if module_name == "transformers":
            for attr in (
                "AutoConfig",
                "AutoModel",
                "AutoModelForCausalLM",
                "AutoModelForMaskedLM",
                "AutoModelForSeq2SeqLM",
                "AutoModelForAudioClassification",
                "AutoModelForAudioFrameClassification",
                "AutoModelForSemanticSegmentation",
                "AutoModelForTextToSpectrogram",
                "AutoModelForTextToWaveform",
                "AutoModelForZeroShotImageClassification",
                "BertConfig",
                "GPT2Config",
                "T5Config",
            ):
                if hasattr(module, attr):
                    namespace[attr] = getattr(module, attr)
    return namespace


def _exec_statement_with_optional_result(code: str, globals_dict: dict[str, Any]) -> Any | None:
    """Execute statement recipe code and return a trailing expression result.

    Parameters
    ----------
    code:
        Python statement recipe source.
    globals_dict:
        Shared recipe globals namespace.

    Returns
    -------
    Any | None
        Value of a final expression statement, or ``None`` when the recipe has no
        trailing expression.
    """

    tree = ast.parse(code, mode="exec")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        prefix = ast.Module(body=tree.body[:-1], type_ignores=tree.type_ignores)
        ast.fix_missing_locations(prefix)
        exec(compile(prefix, "<menagerie-recipe>", "exec"), globals_dict)  # noqa: S102
        expr = ast.Expression(body=tree.body[-1].value)
        ast.fix_missing_locations(expr)
        return eval(compile(expr, "<menagerie-recipe>", "eval"), globals_dict)  # noqa: S307
    exec(code, globals_dict)  # noqa: S102
    return None


def build_input_from_record(record: Any) -> Any:
    """Build the example input for a typed catalog record.

    Parameters
    ----------
    record:
        ``menagerie.schema.CatalogRecord``.

    Returns
    -------
    Any
        Example input object.
    """

    special_input = _special_record_input(record)
    if special_input is not None:
        return special_input

    input_builder = record.input
    if input_builder.kind == "tensor":
        return _tensor_from_record_spec(
            record,
            input_builder.spec.shape,
            input_builder.spec.dtype,
            0,
        )
    if input_builder.kind == "multi":
        values = [
            _tensor_from_record_spec(record, spec.shape, spec.dtype, index)
            for index, spec in enumerate(input_builder.specs)
        ]
        return tuple(values) if input_builder.as_tuple else values
    if input_builder.kind == "kwargs":
        return {
            key: _tensor_from_spec(spec.shape, spec.dtype)
            for key, spec in input_builder.specs.items()
        }
    if input_builder.kind == "none":
        return _tensor_from_spec(input_builder.sentinel_shape, input_builder.sentinel_dtype)
    if input_builder.kind == "callable":
        namespace = _typed_namespace(input_builder.imports)
        builtins = dict(vars(__import__("builtins")))
        globals_dict = {"__builtins__": builtins, "__name__": "__main__", **namespace}
        builder = eval(input_builder.expr, globals_dict, namespace)  # noqa: S307
        return builder()
    raise ValueError(f"unsupported input builder kind={input_builder.kind!r}")


def instantiate_model_from_record(record: Any) -> Any:
    """Instantiate a model from a typed catalog record.

    Parameters
    ----------
    record:
        ``menagerie.schema.CatalogRecord``.

    Returns
    -------
    Any
        Instantiated model.
    """

    recipe = record.recipe
    imports = getattr(recipe, "imports", [])
    namespace = _typed_namespace(imports)
    builtins = dict(vars(__import__("builtins")))
    globals_dict = {"__builtins__": builtins, "__name__": "__main__", **namespace}
    if recipe.type in {"import-callable", "expression"}:
        return eval(recipe.expr, globals_dict, namespace)  # noqa: S307
    statement_result: Any | None = None
    if recipe.type == "statement":
        statement_result = _exec_statement_with_optional_result(recipe.code, globals_dict)
    elif recipe.type == "exec-string":
        exec(recipe.body, globals_dict)  # noqa: S102
    else:
        raise ValueError(f"unsupported recipe type={recipe.type!r}")
    for output_name in (recipe.output_name, "model", "net", "module", "m"):
        if output_name in globals_dict:
            return globals_dict[output_name]
    if statement_result is not None:
        return statement_result
    raise ValueError("typed statement recipe did not assign a model output")


def build_from_record(record: Any) -> tuple[Any, Any]:
    """Build the model and example input for one typed catalog record.

    Parameters
    ----------
    record:
        ``menagerie.schema.CatalogRecord``.

    Returns
    -------
    tuple[Any, Any]
        Instantiated model and example input.
    """

    input_value = build_input_from_record(record)
    return instantiate_model_from_record(record), input_value


@lru_cache(maxsize=1)
def _record_lookup() -> dict[tuple[str, str, str], CatalogRecord]:
    """Load typed records keyed by natural catalog identity.

    Returns
    -------
    dict[tuple[str, str, str], CatalogRecord]
        Mapping from ``(name, zoo, variant)`` to typed JSONL records.
    """

    records = [*load_jsonl(SOURCE_JSONL), *load_jsonl(DEFERRED_JSONL)]
    return {(record.name, record.zoo, record.variant): record for record in records}


def record_for_row(row: CatalogRow) -> CatalogRecord:
    """Return the typed JSONL record for a non-classics catalog row.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    CatalogRecord
        Typed source record.
    """

    try:
        return _record_lookup()[(row.name, row.zoo, row.variant)]
    except KeyError as exc:
        raise LookupError(
            f"no typed JSONL record for {(row.name, row.zoo, row.variant)!r}"
        ) from exc


def build_input_for_row(row: CatalogRow) -> Any:
    """Build the example input for one catalog row through the typed input path.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    Any
        Example input object.
    """

    if is_classics_row(row):
        return classics_example_input(row)
    return build_input_from_record(record_for_row(row))


def is_classics_row(row: CatalogRow) -> bool:
    """Return whether a catalog row is a local historical classic.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    bool
        Whether the row is provided by ``menagerie.classics``.
    """

    # A row is a local classic when its name is in the CLASSICS registry (the authoritative
    # source of truth), OR it uses the bare ``menagerie.classics.X`` constructor convention.
    # Codex-authored rows use the ``from menagerie.classics.X import ...`` form, so the
    # startswith check alone misclassified them and fell through to input_shape parsing.
    from menagerie.classics import CLASSICS

    if row.name in CLASSICS and "menagerie.classics." in row.constructor_call:
        return True
    return row.zoo == "classics-pytorch" and row.constructor_call.startswith("menagerie.classics.")


def classics_module_name(row: CatalogRow) -> str:
    """Extract the classics module name from a constructor expression.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    str
        Module name under ``menagerie.classics``.
    """

    match = re.fullmatch(
        r"menagerie\.classics\.([A-Za-z_][A-Za-z0-9_]*)\.build\(\)", row.constructor_call
    )
    if match is None:
        raise ValueError(f"unsupported classics constructor={row.constructor_call!r}")
    return match.group(1)


def classics_example_input(row: CatalogRow) -> Any:
    """Return the registered example input for a local historical classic.

    Resolves through the ``menagerie.classics`` registry by canonical name, so
    both singleton modules (``example_input``) and grouped family modules
    (``example_input_<variant>``) are handled uniformly.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    Any
        Example input object from the classics registry.
    """

    from menagerie.classics import CLASSICS

    return CLASSICS[row.name]["example_input"]()


def instantiate_model(row: CatalogRow) -> Any:
    """Instantiate a model from a guarded constructor expression.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    Any
        Instantiated model.
    """

    if is_classics_row(row):
        from menagerie.classics import CLASSICS

        return CLASSICS[row.name]["build"]()

    from menagerie.runtime import import_namespace

    namespace = import_namespace(row)
    # Recipes are our own audited catalog code, and __import__ is already permitted (so a restricted
    # builtins set never actually sandboxed anything -- os/subprocess are reachable via import). The
    # 8-entry whitelist silently broke ~2838 recipes that legitimately need type() class factories,
    # `class` statements (__build_class__), setattr/super, and exec (JSON-encoded multi-line reimpls),
    # all failing with "NameError: name 'type' is not defined". Expose the full builtins.
    builtins = dict(vars(__import__("builtins")))
    # __name__='__main__' so type()-built classes inherit __module__ (else torchlens trace hits
    # AttributeError('__module__') on module-less dynamic classes -- the ignore-input wrapper pattern).
    globals_dict = {"__builtins__": builtins, "__name__": "__main__", **namespace}
    constructor_call = row.constructor_call.strip()
    if ";" in constructor_call or constructor_call.startswith(("import ", "from ")):
        # Single namespace (globals IS locals): recipe-level imports + names are visible inside
        # lambdas/classes the recipe defines (separate locals left them unresolved -> "name 'nn' is
        # not defined" when the wrapper's forward lambda ran during tracing).
        statement_result = _exec_statement_with_optional_result(constructor_call, globals_dict)
        for output_name in ("model", "net", "module", "m"):
            if output_name in globals_dict:
                return globals_dict[output_name]
        if statement_result is not None:
            return statement_result
        raise ValueError("statement recipe did not assign a `model`, `net`, or `module` variable")
    return eval(constructor_call, globals_dict, namespace)  # noqa: S307


def build_model_and_input(row: CatalogRow) -> tuple[Any, Any]:
    """Build the model and example input for one catalog row.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    tuple[Any, Any]
        Instantiated model and example input.
    """

    if is_classics_row(row):
        return instantiate_model(row), classics_example_input(row)
    return build_from_record(record_for_row(row))
