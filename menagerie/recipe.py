"""Pure menagerie recipe builders: row -> (model, input).

This module executes audited catalog recipe code via exec/eval. It must not own dependency
installation, environment mutation, subprocess orchestration, or cache/device cleanup policy.
"""

from __future__ import annotations

import ast
import importlib
import re
from collections.abc import Sequence
from typing import Any

from menagerie.catalog import CatalogRow

_SYMBOLIC_DIMS = {
    "B": 1,
    "N": 8,
    "V": 2,
    "T": 8,
    "L": 16,
    "S": 16,
    "K": 4,
    "C": 3,
    "H": 64,
    "W": 64,
    "D": 64,
    "E": 64,
    "G": 8,
    "M": 8,
    "P": 16,
    "Q": 16,
    "R": 16,
    "F": 64,
    "A": 8,
    "X": 8,
    "Y": 8,
    "Z": 8,
}


def _parse_symbolic_multi(text: str) -> list[tuple[int, ...]]:
    """Parse a multi-input / symbolic shape string into concrete tensor shapes.

    Handles formats like ``imgs=(1,V,3,H,W); proj=(1,V,4,4); pos=(N,3)`` -- each parenthesised
    group becomes one input tensor and symbolic dims (V, N, ...) get small concrete defaults
    (resolved consistently across the recipe). Returns ``[]`` when nothing parseable is found.

    Parameters
    ----------
    text:
        Raw catalog shape string.

    Returns
    -------
    list[tuple[int, ...]]
        One concrete shape per parenthesised group.
    """

    shapes: list[tuple[int, ...]] = []
    for group in re.findall(r"\(([^()]*)\)", text):
        dims: list[int] = []
        ok = True
        for token in group.split(","):
            token = token.strip()
            if not token:
                continue
            if token.isdigit():
                dims.append(int(token))
            elif re.fullmatch(r"[A-Za-z_]\w*", token):
                dims.append(_SYMBOLIC_DIMS.get(token[0].upper(), 8))
            else:
                ok = False  # arithmetic / unparseable token -> skip this group
                break
        if ok and dims:
            shapes.append(tuple(dims))
    return shapes


def _split_dtype_tokens(dtype: str) -> list[str]:
    """Split a (possibly multi-tensor) dtype string into normalized dtype tokens.

    ``"float32+long"`` -> ``["float32", "long"]``; ``"float32"`` -> ``["float32"]``. Each token's
    leading word is taken as the dtype so descriptive suffixes are dropped (``"long ids"`` ->
    ``"long"``). Separators: ``+ / , ;`` and the literal word ``and``.
    """

    out: list[str] = []
    for raw in re.split(r"\s*[+/,;]\s*| and ", dtype.strip()):
        token = raw.strip().lower()
        if token:
            out.append(token.split()[0])
    return out


def _resolve_dtype_tokens(
    tokens: list[str], n_groups: int, dtype_map: dict[str, Any]
) -> list[Any] | None:
    """Map dtype tokens onto ``n_groups`` tensors, broadcasting a single token to all.

    Returns ``None`` when no token resolves to a known torch dtype (caller then raises).
    """

    mapped = [dtype_map.get(t) for t in tokens]
    known = [m for m in mapped if m is not None]
    if not known:
        return None
    if len(mapped) == n_groups and all(m is not None for m in mapped):
        return mapped
    return [known[0]] * n_groups


def _multi_input_from_spec(shape: str, dtype_map: dict[str, Any]) -> Any:
    """Build a multi-input structure from a JSON/python-literal recipe, or ``None``.

    Engages only when ``shape`` is an object/array describing per-input ``{"shape","dtype"}``
    entries. Returns a list (``*args`` for ``tl.trace``), a single dict-positional input, or
    ``None`` when ``shape`` is not a structured recipe.

    Accepted forms::

        [{"shape": [1,3,224,224], "dtype": "float32"}, {"shape": [1,16], "dtype": "int64"}]
        {"inputs": [ ...as above... ]}
        {"kwargs": {"input_ids": {"shape": [1,16], "dtype": "int64"}, ...}}
        [{"image": (3,800,1333)}]                      # detectron2 dict-positional
    """

    import json

    import torch

    text = shape.strip()
    if not (text.startswith("[{") or text.startswith('{"')):
        return None
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        try:
            obj = ast.literal_eval(text)  # python-literal dict-lists (single quotes, tuples)
        except (ValueError, SyntaxError):
            return None

    def build(entry: dict[str, Any]) -> Any:
        td = dtype_map.get(str(entry.get("dtype", "float32")).lower(), torch.float32)
        shp = tuple(int(d) for d in entry["shape"])
        if td.is_floating_point or td.is_complex:
            return torch.randn(shp, dtype=td)
        return torch.zeros(shp, dtype=td)

    if isinstance(obj, dict) and "inputs" in obj:
        return [build(e) for e in obj["inputs"]]
    if isinstance(obj, dict) and "kwargs" in obj:
        return {k: build(v) for k, v in obj["kwargs"].items()}
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "shape" in obj[0]:
        return [build(e) for e in obj]
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return [
            {k: build({"shape": v, "dtype": "float32"}) for k, v in entry.items()} for entry in obj
        ]
    return None


def parse_shape(shape: str) -> tuple[int, ...] | list[tuple[int, ...]]:
    """Parse a concrete tensor shape from the catalog.

    Parameters
    ----------
    shape:
        Catalog shape string.

    Returns
    -------
    tuple[int, ...] | list[tuple[int, ...]]
        Parsed input shape or list of shapes.
    """

    shape_text = shape.strip()
    # Multi-input / symbolic-dim recipes (`name=(1,V,3,H,W); other=(N,3)`): build one tensor per
    # group, resolving symbolic dims to small concrete defaults. tl.trace unpacks the list as *args.
    if ";" in shape_text or re.search(r"\([^)]*[A-Za-z]", shape_text):
        multi = _parse_symbolic_multi(shape_text)
        if len(multi) > 1:
            return multi
        if len(multi) == 1:
            return multi[0]
    parsed_text = shape_text
    # Prose-suffixed concrete shapes (`(1, 3, 800, 1024) + noisy boxes (1, 300, 4)`): pull out
    # every leading concrete ``(ints)`` group. One group -> single tensor; many -> *args list.
    concrete_groups = re.findall(r"\(\s*\d[\d,\s]*\)", shape_text)
    if len(concrete_groups) > 1:
        out: list[tuple[int, ...]] = []
        for group in concrete_groups:
            value = ast.literal_eval(group if "," in group else group.rstrip(")") + ",)")
            out.append(value if isinstance(value, tuple) else (value,))
        return out
    if not shape_text.startswith(("(", "[")):
        match = re.search(r"\(([0-9,\s]+)\)", shape_text)
        if match is None:
            raise ValueError(f"expected concrete tuple shape, got {shape!r}")
        parsed_text = f"({match.group(1)})"
    elif concrete_groups and not shape_text.endswith((")", "]")):
        parsed_text = concrete_groups[0]  # strip a trailing prose tail after a single tuple
    parsed = ast.literal_eval(parsed_text)
    if isinstance(parsed, tuple) and all(isinstance(value, int) for value in parsed):
        return parsed
    # Explicit tuple-of-tuples (`((1, 64), (1, 64, 4))`) is a valid multi-input encoding.
    if (
        isinstance(parsed, tuple)
        and parsed
        and all(
            isinstance(item, tuple) and all(isinstance(value, int) for value in item)
            for item in parsed
        )
    ):
        return list(parsed)
    if isinstance(parsed, list) and all(
        isinstance(item, tuple) and all(isinstance(value, int) for value in item) for item in parsed
    ):
        return parsed
    raise ValueError(f"expected tuple[int, ...] or list[tuple[int, ...]], got {shape!r}")


def tensor_for_recipe(shape: str, dtype: str) -> Any:
    """Create a synthetic input tensor or input list for a catalog recipe.

    Parameters
    ----------
    shape:
        Catalog input shape.
    dtype:
        Catalog dtype string.

    Returns
    -------
    Any
        Torch tensor or list of tensors.
    """

    import torch

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "long": torch.int64,
        "int32": torch.int32,
        "bool": torch.bool,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }

    # Structured multi-input recipe takes priority (set input_dtype to "spec" in the catalog).
    spec_tensors = _multi_input_from_spec(shape, dtype_map)
    if spec_tensors is not None:
        return spec_tensors

    parsed_shape = parse_shape(shape)
    n_groups = len(parsed_shape) if isinstance(parsed_shape, list) else 1
    resolved = _resolve_dtype_tokens(_split_dtype_tokens(dtype), n_groups, dtype_map)
    if resolved is None:
        raise ValueError(f"unsupported input_dtype={dtype!r}")

    def make_tensor(parsed: tuple[int, ...], torch_dtype: Any) -> Any:
        """Create one tensor for an already-parsed shape and resolved dtype."""

        if torch_dtype.is_floating_point or torch_dtype.is_complex:
            return torch.randn(parsed, dtype=torch_dtype)
        return torch.zeros(parsed, dtype=torch_dtype)

    if isinstance(parsed_shape, list):
        return [make_tensor(item, td) for item, td in zip(parsed_shape, resolved)]
    return make_tensor(parsed_shape, resolved[0])


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

    input_builder = record.input
    if input_builder.kind == "tensor":
        return _tensor_from_spec(input_builder.spec.shape, input_builder.spec.dtype)
    if input_builder.kind == "multi":
        values = [_tensor_from_spec(spec.shape, spec.dtype) for spec in input_builder.specs]
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
    if recipe.type == "statement":
        exec(recipe.code, globals_dict)  # noqa: S102
    elif recipe.type == "exec-string":
        exec(recipe.body, globals_dict)  # noqa: S102
    else:
        raise ValueError(f"unsupported recipe type={recipe.type!r}")
    for output_name in (recipe.output_name, "model", "net", "module"):
        if output_name in globals_dict:
            return globals_dict[output_name]
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
        exec(constructor_call, globals_dict)  # noqa: S102
        for output_name in ("model", "net", "module"):
            if output_name in globals_dict:
                return globals_dict[output_name]
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

    input_value = (
        classics_example_input(row)
        if is_classics_row(row)
        else tensor_for_recipe(row.input_shape, row.input_dtype)
    )
    return instantiate_model(row), input_value
