"""Derived semantic facet views for TorchLens records."""

from __future__ import annotations

import ast
import builtins
import inspect
import textwrap
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Literal, overload


RecordScope = Literal["op", "module", "any"]
RecipeFunc = Callable[[Any], dict[str, Any]]
PredicateFunc = Callable[[Any], bool]


@dataclass(frozen=True)
class FacetRecipe:
    """Registered facet recipe metadata.

    Parameters
    ----------
    recipe_name:
        Function name used as the recipe identifier.
    class_names:
        Short class names matched by this recipe.
    qualnames:
        Fully-qualified class names matched by this recipe.
    has_predicate:
        Whether this recipe has an arbitrary predicate matcher.
    target_scope:
        Record scope the recipe targets.
    """

    recipe_name: str
    class_names: tuple[str, ...]
    qualnames: tuple[str, ...]
    has_predicate: bool
    target_scope: RecordScope


@dataclass
class _RegisteredRecipe:
    """Internal recipe registration entry."""

    public: FacetRecipe
    func: RecipeFunc
    predicate: PredicateFunc | None
    declared_facets: tuple[str, ...]


class MissingFacet:
    """Sentinel for a known facet whose value is intentionally unavailable."""

    def __init__(self, reason: str) -> None:
        """Initialize the missing-facet sentinel.

        Parameters
        ----------
        reason:
            Error message raised when the facet is accessed.
        """

        self.reason = reason

    def raise_error(self) -> None:
        """Raise the missing-facet error."""

        raise RuntimeError(self.reason)

    def __getattr__(self, name: str) -> Any:
        """Raise when any attribute is read from the sentinel."""

        self.raise_error()

    def __getitem__(self, key: Any) -> Any:
        """Raise when any item is read from the sentinel."""

        self.raise_error()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Raise when the sentinel is called."""

        self.raise_error()


class AttentionHeadView:
    """Scoped accessor for one attention head within a parent facet view."""

    def __init__(self, parent: "FacetView", head_index: int) -> None:
        """Initialize a per-head attention view.

        Parameters
        ----------
        parent:
            Parent attention facet view.
        head_index:
            Zero-based query head index.
        """

        self._parent = parent
        self._head_index = head_index

    def __getattr__(self, name: str) -> Any:
        """Return a head-sliced facet by attribute name."""

        return self._slice(name)

    def __getitem__(self, name: str) -> Any:
        """Return a head-sliced facet by key."""

        return self._slice(name)

    def _slice(self, name: str) -> Any:
        """Return the named tensor sliced to this head when applicable."""

        value = self._parent[name]
        if name not in {"q", "k", "v"} or not hasattr(value, "__getitem__"):
            return value
        head_index = self._head_index
        if name in {"k", "v"}:
            n_q_heads = self._parent.get("n_q_heads", self._parent.get("n_heads", None))
            n_kv_heads = self._parent.get("n_kv_heads", n_q_heads)
            if isinstance(n_q_heads, int) and isinstance(n_kv_heads, int) and n_kv_heads:
                group_size = max(1, n_q_heads // n_kv_heads)
                head_index = head_index // group_size
        return value[:, :, head_index, :]


class FacetView(Mapping[str, Any]):
    """Lazy dict-like and attribute-access semantic view on a TorchLens record."""

    def __init__(self, record: Any) -> None:
        """Create a facet view for a record.

        Parameters
        ----------
        record:
            TorchLens Op or Module record.
        """

        self._record = record
        self._recipes = _matching_recipes(record)
        self._cache: dict[str, Any] = {}
        self._computed = False

    @property
    def recipe_source(self) -> str | tuple[str, ...] | None:
        """Return recipe name provenance for this view."""

        names = tuple(recipe.public.recipe_name for recipe in self._recipes)
        if not names:
            return None
        if len(names) == 1:
            return names[0]
        return names

    def keys(self) -> builtins.list[str]:  # type: ignore[override]
        """Return available facet names without invoking recipe functions."""

        names: builtins.list[str] = []
        seen: set[str] = set()
        for recipe in self._recipes:
            for name in recipe.declared_facets:
                if name not in seen:
                    names.append(name)
                    seen.add(name)
        return names

    def has(self, name: str) -> bool:
        """Return whether a facet name is declared without computing values."""

        return name in self.keys()

    def invalidate(self) -> None:
        """Clear cached computed facet values."""

        self._cache.clear()
        self._computed = False

    def get(self, name: str, default: Any = None) -> Any:
        """Return a facet value or a default.

        Parameters
        ----------
        name:
            Facet name.
        default:
            Value returned when the facet is absent.
        """

        if not self.has(name) and name not in self._cache:
            return default
        try:
            return self[name]
        except KeyError:
            return default

    def head(self, head_index: int) -> AttentionHeadView:
        """Return a scoped view for one attention head."""

        return AttentionHeadView(self, head_index)

    def __getitem__(self, name: str) -> Any:
        """Return a facet value by key, computing lazily if needed."""

        if name not in self._cache:
            self._compute()
        if name not in self._cache:
            raise KeyError(name)
        value = self._cache[name]
        if isinstance(value, MissingFacet):
            value.raise_error()
        return value

    def __getattr__(self, name: str) -> Any:
        """Return a facet value by attribute name."""

        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __iter__(self) -> Iterator[str]:
        """Iterate declared facet names."""

        return iter(self.keys())

    def __len__(self) -> int:
        """Return the number of declared facet names."""

        return len(self.keys())

    def _compute(self) -> None:
        """Invoke matching recipes and merge their facet dictionaries."""

        if self._computed:
            return
        values: dict[str, Any] = {}
        for recipe in self._recipes:
            contribution = recipe.func(self._record)
            for name, value in contribution.items():
                if name in values:
                    warnings.warn(
                        f"Facet recipe {recipe.public.recipe_name!r} overrides facet {name!r}.",
                        UserWarning,
                        stacklevel=2,
                    )
                values[name] = value
        self._cache = values
        self._computed = True


_REGISTRY: builtins.list[_RegisteredRecipe] = []


@overload
def register(
    *,
    class_name: str | tuple[str, ...] | None = None,
    class_qualname: str | tuple[str, ...] | None = None,
    predicate: PredicateFunc | None = None,
) -> Callable[[RecipeFunc], RecipeFunc]: ...


@overload
def register(
    *,
    class_name: str | tuple[str, ...] | None = None,
    class_qualname: str | tuple[str, ...] | None = None,
    predicate: PredicateFunc | None = None,
    target_scope: RecordScope = "any",
    facets: tuple[str, ...] = (),
) -> Callable[[RecipeFunc], RecipeFunc]: ...


def register(
    *,
    class_name: str | tuple[str, ...] | None = None,
    class_qualname: str | tuple[str, ...] | None = None,
    predicate: PredicateFunc | None = None,
    target_scope: RecordScope = "any",
    facets: tuple[str, ...] = (),
) -> Callable[[RecipeFunc], RecipeFunc]:
    """Register a facet recipe and return the function unchanged.

    Parameters
    ----------
    class_name:
        Short class name or match-any tuple of names.
    class_qualname:
        Fully-qualified class name or match-any tuple of names.
    predicate:
        Arbitrary predicate on the target record.
    target_scope:
        Internal scope marker used for discoverability filtering.
    facets:
        Internal declared facet names used for cheap ``keys()``.
    """

    def decorator(func: RecipeFunc) -> RecipeFunc:
        """Store the recipe metadata and return ``func`` unchanged."""

        class_names = _as_tuple(class_name)
        qualnames = _as_tuple(class_qualname)
        declared = facets or _literal_return_keys(func)
        _REGISTRY.append(
            _RegisteredRecipe(
                public=FacetRecipe(
                    recipe_name=func.__name__,
                    class_names=class_names,
                    qualnames=qualnames,
                    has_predicate=predicate is not None,
                    target_scope=target_scope,
                ),
                func=func,
                predicate=predicate,
                declared_facets=declared,
            )
        )
        return func

    return decorator


def list(scope: str | None = None, class_name: str | None = None) -> builtins.list[FacetRecipe]:
    """List registered recipes, optionally filtered by scope and class-name glob."""

    return [
        recipe.public
        for recipe in _REGISTRY
        if _scope_matches(recipe.public.target_scope, scope)
        and _class_filter_matches(recipe.public.class_names, class_name)
    ]


def info(class_name: str) -> dict[str, builtins.list[str]]:
    """Return registered recipe names and declared facets for a class name."""

    recipes: builtins.list[str] = []
    facets: builtins.list[str] = []
    seen: set[str] = set()
    for recipe in _REGISTRY:
        if not _class_filter_matches(recipe.public.class_names, class_name):
            continue
        recipes.append(recipe.public.recipe_name)
        for facet_name in recipe.declared_facets:
            if facet_name not in seen:
                facets.append(facet_name)
                seen.add(facet_name)
    return {"recipes": recipes, "facets": facets}


def _as_tuple(value: str | tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalize a string-or-tuple matcher to a tuple."""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return value


def _literal_return_keys(func: RecipeFunc) -> tuple[str, ...]:
    """Best-effort extraction of literal keys from recipe return dictionaries."""

    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError):
        return ()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    keys: builtins.list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
            continue
        for key in node.value.keys:
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and key.value not in keys
            ):
                keys.append(key.value)
    return tuple(keys)


def _matching_recipes(record: Any) -> builtins.list[_RegisteredRecipe]:
    """Return recipes whose matchers all pass for a record."""

    return [recipe for recipe in _REGISTRY if _recipe_matches(recipe, record)]


def _recipe_matches(recipe: _RegisteredRecipe, record: Any) -> bool:
    """Return whether one recipe applies to a record."""

    if not _record_scope_matches(recipe.public.target_scope, record):
        return False
    class_name = getattr(record, "class_name", getattr(record, "type", None))
    class_qualname = getattr(record, "class_qualname", getattr(record, "func_qualname", None))
    if recipe.public.class_names and class_name not in recipe.public.class_names:
        return False
    if recipe.public.qualnames and class_qualname not in recipe.public.qualnames:
        return False
    if recipe.predicate is not None and not recipe.predicate(record):
        return False
    return True


def _record_scope_matches(scope: RecordScope, record: Any) -> bool:
    """Return whether a record is compatible with a recipe scope."""

    if scope == "any":
        return True
    record_scope = "module" if hasattr(record, "calls") else "op"
    return scope == record_scope


def _scope_matches(recipe_scope: RecordScope, requested_scope: str | None) -> bool:
    """Return whether a recipe scope passes a list filter."""

    if requested_scope is None:
        return True
    return recipe_scope in {requested_scope, "any"}


def _class_filter_matches(class_names: tuple[str, ...], class_name: str | None) -> bool:
    """Return whether class names pass an optional exact-or-glob filter."""

    if class_name is None:
        return True
    if not class_names:
        return False
    return builtins.any(fnmatch(name, class_name) for name in class_names)


def _clear_registry_for_tests() -> None:
    """Clear all registered recipes for isolated tests."""

    _REGISTRY.clear()
