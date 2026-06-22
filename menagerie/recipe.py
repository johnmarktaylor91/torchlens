"""Pure menagerie recipe builders: row -> (model, input).

This module executes audited catalog recipe code via exec/eval. It must not own dependency
installation, environment mutation, subprocess orchestration, or cache/device cleanup policy.
"""

from __future__ import annotations

import ast
import importlib
import re
from typing import Any

from menagerie.catalog import CatalogRow

MODULE_PACKAGE_MAP = {
    "clip": "clip",
    "diffusers": "diffusers",
    "mmcv": "mmcv",
    "mmdet": "mmdet",
    "mmengine": "mmengine",
    "mmseg": "mmsegmentation",
    "monai": "monai",
    "norse": "norse",
    "open_clip": "open-clip-torch",
    "recbole": "recbole",
    "segmentation_models_pytorch": "segmentation-models-pytorch",
    "sentence_transformers": "sentence-transformers",
    "snntorch": "snntorch",
    "speechbrain": "speechbrain",
    "super_image": "super-image",
    "timm": "timm",
    "torch_geometric": "torch-geometric",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "ultralytics": "ultralytics",
}
# Framework module names always treated as real external dependencies even when they have no
# MODULE_PACKAGE_MAP entry (their pip name == module name, or they are vendored framework extras).
KNOWN_FRAMEWORK_MODULES = {
    "timm",
    "torch",
    "torchvision",
    "transformers",
    "diffusers",
    *MODULE_PACKAGE_MAP,
}
# Generic, repo-internal source-layout package names. A recipe that does
# `from models.foo import X` / `from src.bar import Y` is importing the ORIGINAL repo's local
# package tree, NOT a PyPI distribution. These are never pip-installable and must NOT be reported
# as "dependency missing" (that wrongly skipped ~350 base-env-renderable rows and produced bogus
# cluster keys like `models` (226), `src` (48), `lib` (14), ...). Recipes that genuinely need the
# upstream source are surfaced honestly via `unrenderable_reason` as `local_source_unavailable`,
# and self-contained recipes that merely *mention* such a name render normally.
LOCAL_SOURCE_NAMES = {
    "models",
    "model",
    "model_zoo",
    "modeling",
    "modelling",
    "src",
    "lib",
    "libs",
    "networks",
    "network",
    "net",
    "nets",
    "nn_modules",
    "implementations",
    "impl",
    "training",
    "train",
    "core",
    "architectures",
    "architecture",
    "arch",
    "source",
    "sources",
    "types",
    "utils",
    "util",
    "modules",
    "module",
    "backbone",
    "backbones",
    "layers",
    "layer",
    "common",
    "components",
    "component",
    "scripts",
    "script",
    "code",
    "main",
    "config",
    "configs",
    "cfg",
}
# Names that are stdlib or recipe-local and never count as an external dependency.
STDLIB_OR_LOCAL = {
    "model",
    "cfg",
    "os",
    "sys",
    "math",
    "typing",
    "functools",
    "collections",
    "re",
    "itertools",
    "json",
    "copy",
    "abc",
    "dataclasses",
    "warnings",
    "builtins",
    "random",
}
ZOO_PACKAGE_HINTS = (
    (re.compile(r"torchvision", re.I), ("torchvision",)),
    (re.compile(r"\btimm\b", re.I), ("timm",)),
    (re.compile(r"transformers|huggingface", re.I), ("transformers",)),
    (re.compile(r"diffusers", re.I), ("diffusers", "transformers")),
    (re.compile(r"segmentation_models_pytorch|smp", re.I), ("segmentation-models-pytorch",)),
    (re.compile(r"ultralytics", re.I), ("ultralytics",)),
    (re.compile(r"torch_geometric|pyg", re.I), ("torch-geometric",)),
    (re.compile(r"open.?mmlab|mmdet|mmseg|mmpose|mmaction|mmagic", re.I), ("mmengine",)),
    (re.compile(r"recbole", re.I), ("recbole",)),
    (re.compile(r"open_clip", re.I), ("open-clip-torch",)),
)
STDLIB_OR_LOCAL = {
    "model",
    "cfg",
    "os",
    "sys",
    "math",
    "typing",
    "functools",
    "collections",
    "re",
    "itertools",
    "json",
    "copy",
    "abc",
    "dataclasses",
    "warnings",
    "builtins",
    "random",
}

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


def _collect_recipe_names(tree: ast.AST) -> tuple[set[str], set[str], set[str]]:
    """Collect (imported-module-roots, attribute-roots, locally-bound-names) from a parsed recipe.

    ``imported`` are the top-level package of every real absolute ``import``/``from`` statement --
    these are the only reliable EXTERNAL-dependency signal. ``attribute_roots`` are the root names of
    dotted attribute access (``models.networks.X`` -> ``models``); they are used ONLY to detect a
    bare reference to the upstream repo's local source layout (e.g. ``models.networks.X(...)`` with
    no import), never to invent a pip dependency -- a bare attribute root is just as likely a
    namespace alias (``np.array``) or pseudo-code (``nn.GRUCell``) as a module. ``bound`` are import
    aliases, assignment targets, comprehension/lambda/def/class params -- names LOCAL to the recipe.

    Parameters
    ----------
    tree:
        Parsed recipe AST.

    Returns
    -------
    tuple[set[str], set[str], set[str]]
        ``(imported_modules, attribute_roots, bound_names)``.
    """

    imported: set[str] = set()
    bound: set[str] = set()
    attr_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # Only absolute imports name an external module; relative (level>0) imports are local.
            if node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
            for alias in node.names:
                bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        bound.add(name.id)
        elif isinstance(node, ast.comprehension):
            for name in ast.walk(node.target):
                if isinstance(name, ast.Name):
                    bound.add(name.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                bound.add(arg.arg)
            if args.vararg:
                bound.add(args.vararg.arg)
            if args.kwarg:
                bound.add(args.kwarg.arg)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound.add(node.name)
        elif isinstance(node, ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.Attribute):
            value = node.value
            while isinstance(value, ast.Attribute):
                value = value.value
            if isinstance(value, ast.Name):
                attr_roots.add(value.id)
    return imported, attr_roots, bound


def _required_modules_with_error(constructor_call: str, zoo: str) -> tuple[tuple[str, ...], str]:
    """Infer top-level EXTERNAL modules required by a constructor, plus a recipe parse-error string.

    Uses a real AST parse instead of a regex split. The old regex split on ``[;\\n]`` shredded
    inline ``exec``/``code=`` recipe bodies, mistaking fragments of the quoted source for imports
    and producing garbled cluster keys. AST parsing ignores string-literal contents entirely, so
    only genuine outer imports/attribute roots count; an unparseable recipe yields a clean error
    string instead of garbage. Generic repo-internal source names (``models``/``src``/``lib``/...)
    are local layout, NOT pip distributions, so they are excluded from the external-dep set.

    Parameters
    ----------
    constructor_call:
        Catalog constructor expression.
    zoo:
        Source model zoo.

    Returns
    -------
    tuple[tuple[str, ...], str]
        ``(external_module_names, parse_error)`` -- ``parse_error`` is empty on success.
    """

    modules: set[str] = set()
    bound: set[str] = set()
    parse_error = ""
    source = constructor_call.strip()
    try:
        tree: ast.AST | None = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        tree = None
        parse_error = f"recipe parse error: {exc.msg} (line {exc.lineno})"
    if tree is not None:
        # External-dep detection uses ONLY real import statements -- never bare attribute roots,
        # which are just as likely a namespace alias (`np.array`) or pseudo-code (`nn.GRUCell`) as a
        # module and would wrongly gate valid recipes. Local-source attribute refs are handled
        # separately by `unrenderable_reason`.
        modules, _attr_roots, bound = _collect_recipe_names(tree)
    else:
        # Malformed recipe: never regex-split the body (that is what produced garbled cluster keys).
        # Scan ONLY for known framework tokens so a recognizable real dep is still detected.
        for token in KNOWN_FRAMEWORK_MODULES:
            if re.search(rf"\b{re.escape(token)}\b", source):
                modules.add(token)
    zoo_lower = zoo.lower()
    if "torchvision" in zoo_lower:
        modules.add("torchvision")
    if "timm" in zoo_lower:
        modules.add("timm")
    if "transformers" in zoo_lower or "huggingface" in zoo_lower:
        modules.add("transformers")
    if "diffusers" in zoo_lower:
        modules.add("diffusers")
    if "segmentation_models_pytorch" in zoo_lower or "smp" in zoo_lower:
        modules.add("segmentation_models_pytorch")
    if "ultralytics" in zoo_lower:
        modules.add("ultralytics")

    def _is_external_dep(name: str) -> bool:
        if not name or name in STDLIB_OR_LOCAL:
            return False
        if name[0].isupper():
            return False  # a Class/symbol, not a module
        if name in LOCAL_SOURCE_NAMES:
            return False  # repo-internal source layout, not a PyPI distribution
        if name in bound and name not in KNOWN_FRAMEWORK_MODULES:
            return False  # locally bound alias / variable, not an imported module
        return True

    return tuple(sorted(m for m in modules if _is_external_dep(m))), parse_error


def required_modules(constructor_call: str, zoo: str) -> tuple[str, ...]:
    """Infer top-level EXTERNAL modules required by a constructor expression.

    Parameters
    ----------
    constructor_call:
        Catalog constructor expression.
    zoo:
        Source model zoo.

    Returns
    -------
    tuple[str, ...]
        Required top-level import names.
    """

    modules, _error = _required_modules_with_error(constructor_call, zoo)
    return modules


def import_namespace(row: CatalogRow) -> dict[str, Any]:
    """Build the namespace used to instantiate a model recipe.

    Parameters
    ----------
    row:
        Catalog row.

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
        *required_modules(row.constructor_call, row.zoo),
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
