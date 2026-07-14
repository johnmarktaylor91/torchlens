"""Closed declarative R1 recipes and audited typed adapter loading."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Optional, Union

from menagerie.crawler.identity import canonical_json_bytes, compute_recipe_revision, hash_bytes


class RecipeError(ValueError):
    """Raised when a recipe is opaque, executable text, or violates its typed contract."""


BuildModel = Callable[[], object]
MakeDummyCall = Callable[[int, str], tuple[tuple[object, ...], dict[str, object]]]


@dataclass(frozen=True)
class DeclarativeRecipe:
    """Closed R1 library constructor description.

    Parameters
    ----------
    distribution, version:
        Exact package distribution and version.
    module, symbol:
        Importable module and direct constructor attribute.
    kwargs:
        JSON-compatible constructor keyword arguments.
    artifact_sha256:
        Optional exact installed-artifact hash.
    pretrained_disable_fields:
        Keyword names explicitly set to disable pretrained assets.
    """

    distribution: str
    version: str
    module: str
    symbol: str
    kwargs: Mapping[str, Any]
    artifact_sha256: Optional[str] = None
    pretrained_disable_fields: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DeclarativeRecipe":
        """Validate and construct a closed declarative recipe.

        Parameters
        ----------
        value:
            Candidate recipe mapping.

        Returns
        -------
        DeclarativeRecipe
            Strict declarative recipe.
        """

        allowed = {
            "distribution",
            "version",
            "artifact_sha256",
            "module",
            "symbol",
            "kwargs",
            "pretrained_disable_fields",
        }
        unknown = set(value) - allowed
        if unknown:
            raise RecipeError(f"unknown declarative recipe fields: {sorted(unknown)!r}")
        required = {"distribution", "version", "module", "symbol", "kwargs"}
        missing = required - set(value)
        if missing:
            raise RecipeError(f"missing declarative recipe fields: {sorted(missing)!r}")
        strings = {name: value[name] for name in required - {"kwargs"}}
        if any(not isinstance(item, str) or not item for item in strings.values()):
            raise RecipeError("distribution, version, module, and symbol must be non-empty strings")
        module = str(value["module"])
        symbol = str(value["symbol"])
        if not all(part.isidentifier() for part in module.split(".")) or not symbol.isidentifier():
            raise RecipeError("module and symbol must be direct Python identifiers")
        kwargs = value["kwargs"]
        if not isinstance(kwargs, Mapping) or not all(isinstance(key, str) for key in kwargs):
            raise RecipeError("kwargs must be a string-keyed mapping")
        try:
            canonical_json_bytes(kwargs)
        except (TypeError, ValueError) as exc:
            raise RecipeError(
                "kwargs must contain only JSON-compatible declarative values"
            ) from exc
        disable_fields = value.get("pretrained_disable_fields", [])
        if not isinstance(disable_fields, (list, tuple)) or not all(
            isinstance(field, str) and field for field in disable_fields
        ):
            raise RecipeError("pretrained_disable_fields must be a string sequence")
        artifact = value.get("artifact_sha256")
        if artifact is not None and not isinstance(artifact, str):
            raise RecipeError("artifact_sha256 must be a string or null")
        return cls(
            distribution=str(value["distribution"]),
            version=str(value["version"]),
            module=module,
            symbol=symbol,
            kwargs=dict(kwargs),
            artifact_sha256=artifact,
            pretrained_disable_fields=tuple(disable_fields),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible recipe mapping.

        Returns
        -------
        dict[str, Any]
            Declarative recipe payload.
        """

        return {
            "distribution": self.distribution,
            "version": self.version,
            "artifact_sha256": self.artifact_sha256,
            "module": self.module,
            "symbol": self.symbol,
            "kwargs": dict(self.kwargs),
            "pretrained_disable_fields": list(self.pretrained_disable_fields),
        }


@dataclass(frozen=True)
class LoadedRecipe:
    """Executable constructor obtained from a safe recipe form.

    Parameters
    ----------
    kind:
        ``declarative-library`` or ``typed-adapter``.
    build_model:
        Zero-argument random-initialized constructor.
    make_dummy_call:
        Typed adapter input builder, absent for declarative R1 recipes.
    recipe_revision:
        Source-bound recipe hash when a source identity was supplied.
    module:
        Loaded typed module, absent for declarative recipes.
    """

    kind: str
    build_model: BuildModel
    make_dummy_call: Optional[MakeDummyCall]
    recipe_revision: str
    module: Optional[ModuleType]


def reject_opaque_recipe(value: Mapping[str, Any]) -> None:
    """Reject every legacy expression, statement, eval, or exec recipe form.

    Parameters
    ----------
    value:
        Candidate recipe mapping.

    Raises
    ------
    RecipeError
        If the mapping contains an opaque executable recipe marker.
    """

    forbidden_keys = {"code", "expression", "statement", "eval", "exec", "callable_string"}
    offending = forbidden_keys.intersection(value)
    recipe_type = str(value.get("type", ""))
    if offending or recipe_type in {"statement", "expression", "exec-string", "eval-string"}:
        details = sorted(offending) or [recipe_type]
        raise RecipeError(f"opaque executable recipes are forbidden: {details!r}")


def load_declarative_recipe(
    value: Union[DeclarativeRecipe, Mapping[str, Any]], *, source_identity: str = "unbound"
) -> LoadedRecipe:
    """Load a closed R1 recipe without evaluating source text.

    Parameters
    ----------
    value:
        Validated recipe object or mapping.
    source_identity:
        Exact source identity to bind into the recipe revision.

    Returns
    -------
    LoadedRecipe
        Direct import/getattr constructor wrapper.
    """

    if isinstance(value, Mapping):
        reject_opaque_recipe(value)
        recipe = DeclarativeRecipe.from_mapping(value)
    else:
        recipe = value
    module = importlib.import_module(recipe.module)
    constructor = getattr(module, recipe.symbol, None)
    if constructor is None or not callable(constructor):
        raise RecipeError(f"{recipe.module}.{recipe.symbol} is not a callable constructor")

    def build_model() -> object:
        """Invoke the direct library constructor with declarative kwargs.

        Returns
        -------
        object
            Constructed random-initialized model.
        """

        return constructor(**dict(recipe.kwargs))

    revision = compute_recipe_revision(recipe.to_dict(), source_identity)
    return LoadedRecipe("declarative-library", build_model, None, revision, None)


def _check_adapter_source(source: str, path: Path) -> ast.Module:
    """Statically reject opaque execution and TorchLens imports in an adapter.

    Parameters
    ----------
    source:
        Exact Python adapter source.
    path:
        Source path used in diagnostics.

    Returns
    -------
    ast.Module
        Parsed module syntax tree.
    """

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise RecipeError(f"typed adapter has invalid syntax: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "torchlens" or alias.name.startswith("torchlens.")
                for alias in node.names
            ):
                raise RecipeError("typed adapters must not import TorchLens")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "torchlens" or str(node.module).startswith("torchlens."):
                raise RecipeError("typed adapters must not import TorchLens")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "compile"}:
                raise RecipeError(f"typed adapters must not call {node.func.id}()")
    return tree


def _validate_typed_function(
    function: Callable[..., Any], name: str, expected_parameters: tuple[str, ...]
) -> None:
    """Validate one typed adapter entry point.

    Parameters
    ----------
    function:
        Candidate function.
    name:
        Required public symbol name.
    expected_parameters:
        Exact positional parameter names.
    """

    if not inspect.isfunction(function):
        raise RecipeError(f"{name} must be a Python function")
    signature = inspect.signature(function)
    if tuple(signature.parameters) != expected_parameters:
        raise RecipeError(f"{name} must have parameters {expected_parameters!r}")
    annotations = function.__annotations__
    if "return" not in annotations or any(
        parameter not in annotations for parameter in expected_parameters
    ):
        raise RecipeError(f"{name} must carry parameter and return type annotations")


def load_typed_adapter(path: Path, *, source_identity: str = "unbound") -> LoadedRecipe:
    """Statically check and import a typed R2--R4 adapter module.

    Parameters
    ----------
    path:
        Exact adapter Python file.
    source_identity:
        Exact source identity to bind into the recipe revision.

    Returns
    -------
    LoadedRecipe
        Typed constructor and dummy-call functions.
    """

    source_bytes = path.read_bytes()
    try:
        source = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecipeError("typed adapter must be UTF-8 Python source") from exc
    _check_adapter_source(source, path)
    module_name = f"menagerie_crawler_adapter_{hash_bytes(source_bytes)[7:23]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RecipeError(f"cannot load typed adapter at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_model = getattr(module, "build_model", None)
    make_dummy_call = getattr(module, "make_dummy_call", None)
    if not callable(build_model) or not callable(make_dummy_call):
        raise RecipeError("typed adapter must define build_model and make_dummy_call")
    _validate_typed_function(build_model, "build_model", ())
    _validate_typed_function(make_dummy_call, "make_dummy_call", ("seed", "device"))
    revision = compute_recipe_revision(
        {"recipe_type": "typed-adapter", "path": path.name},
        source_identity,
        adapter_bytes=source_bytes,
    )
    return LoadedRecipe("typed-adapter", build_model, make_dummy_call, revision, module)


def load_recipe(value: Mapping[str, Any], *, source_identity: str = "unbound") -> LoadedRecipe:
    """Load one explicitly tagged closed recipe form.

    Parameters
    ----------
    value:
        Mapping with kind ``declarative-library`` or ``typed-adapter``.
    source_identity:
        Source identity bound into the recipe revision.

    Returns
    -------
    LoadedRecipe
        Executable typed recipe contract.
    """

    reject_opaque_recipe(value)
    kind = value.get("kind") or value.get("recipe_type")
    if kind == "declarative-library":
        payload = value.get("recipe", value)
        if not isinstance(payload, Mapping):
            raise RecipeError("declarative recipe payload must be an object")
        if payload is value:
            payload = {
                key: item for key, item in value.items() if key not in {"kind", "recipe_type"}
            }
        return load_declarative_recipe(payload, source_identity=source_identity)
    if kind == "typed-adapter":
        path_value = value.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise RecipeError("typed-adapter recipe requires a path")
        return load_typed_adapter(Path(path_value), source_identity=source_identity)
    raise RecipeError(f"unsupported recipe kind: {kind!r}")
