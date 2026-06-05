"""Derived semantic facet views for TorchLens records."""

from __future__ import annotations

import ast
import builtins
import contextlib
import contextvars
import hashlib
import inspect
import itertools
import textwrap
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fnmatch import fnmatch
from typing import Any, Literal, TypeAlias, overload

import torch

from ..intervention.types import (
    DataclassField,
    DictKey,
    HFKey,
    NamedField,
    OutputPathComponent,
    TupleIndex,
)


RecordScope = Literal["op", "module", "any"]
RecipeFunc = Callable[[Any], dict[str, Any]]
PredicateFunc = Callable[[Any], bool]
RecipeSource = Literal["built-in", "user"]
FacetKey: TypeAlias = str | tuple[OutputPathComponent, ...]
HomeKind = Literal["op", "module_output", "module_input", "parameter", "computed"]
CapabilityClass = Literal["bijective_view", "selection", "aliasing_selection", "computed"]
ValueVersion = Literal["raw_out", "out_versions_by_child"]


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
    source: RecipeSource = "user"


@dataclass
class _RegisteredRecipe:
    """Internal recipe registration entry."""

    public: FacetRecipe
    func: RecipeFunc
    predicate: PredicateFunc | None
    declared_facets: tuple[str, ...]
    order: int


@dataclass(frozen=True)
class FacetRegistrySnapshot:
    """Immutable facet recipe set captured for one trace.

    Parameters
    ----------
    version:
        Monotonic global registry version at snapshot construction time.
    provenance_id:
        Stable digest of the recipe identities and their source tier.
    recipes:
        Immutable active recipe entries.
    """

    version: int
    provenance_id: str
    recipes: tuple[_RegisteredRecipe, ...]


@dataclass(frozen=True)
class FacetCapabilityFlags:
    """Read/grad/write/portability flags for a facet or primitive.

    Parameters
    ----------
    read:
        Whether the facet can be read.
    grad:
        Whether the same transform may be applied to a saved home gradient.
    write:
        Whether the primitive is scatter-back capable for future intervention.
    portable:
        Whether the primitive can be represented without executable Python code.
    reconstructed:
        Whether the value is reconstructed rather than captured.
    """

    read: bool = True
    grad: bool = False
    write: bool = False
    portable: bool = True
    reconstructed: bool = False

    def intersect(self, other: "FacetCapabilityFlags") -> "FacetCapabilityFlags":
        """Return the weakest-link intersection of two flag sets.

        Parameters
        ----------
        other:
            Flags to intersect with this instance.

        Returns
        -------
        FacetCapabilityFlags
            Combined capability flags.
        """

        return FacetCapabilityFlags(
            read=self.read and other.read,
            grad=self.grad and other.grad,
            write=self.write and other.write,
            portable=self.portable and other.portable,
            reconstructed=self.reconstructed or other.reconstructed,
        )


_CAPABILITY_FLAGS_BY_CLASS: dict[CapabilityClass, FacetCapabilityFlags] = {
    "bijective_view": FacetCapabilityFlags(read=True, grad=True, write=True),
    "selection": FacetCapabilityFlags(read=True, grad=True, write=True),
    "aliasing_selection": FacetCapabilityFlags(read=True, grad=True, write=False),
    "computed": FacetCapabilityFlags(read=True, grad=False, write=False, portable=False),
}


@dataclass(frozen=True)
class TransformPrimitive:
    """One structural transform in a ``FacetSpec`` chain.

    Parameters
    ----------
    kind:
        Primitive name.
    args:
        Positional primitive arguments.
    kwargs:
        Keyword primitive arguments.
    capability_class:
        Safety class for read/grad/write capability.
    pre_shape:
        Optional expected input shape.
    post_shape:
        Optional expected output shape.
    """

    kind: str
    args: tuple[Any, ...] = ()
    kwargs: tuple[tuple[str, Any], ...] = ()
    capability_class: CapabilityClass = "computed"
    pre_shape: tuple[int, ...] | None = None
    post_shape: tuple[int, ...] | None = None

    @property
    def flags(self) -> FacetCapabilityFlags:
        """Return capability flags for this primitive."""

        return _CAPABILITY_FLAGS_BY_CLASS[self.capability_class]

    def apply(self, value: Any) -> Any:
        """Apply this primitive to a tensor-like value.

        Parameters
        ----------
        value:
            Tensor-like value to transform.

        Returns
        -------
        Any
            Transformed value.
        """

        self._assert_shape(value, self.pre_shape, "pre")
        if self.kind == "getitem":
            result = value[self.args[0]]
        elif self.kind == "heads":
            n_heads, d_head = self.args
            result = value.reshape(*value.shape[:-1], n_heads, d_head)
        elif self.kind == "split":
            sections, dim, index = self.args
            size = value.shape[dim]
            if size % sections:
                raise ValueError(
                    f"Cannot split dimension {dim} of shape {tuple(value.shape)} "
                    f"into {sections} equal sections."
                )
            chunk = size // sections
            result = value.narrow(dim, index * chunk, chunk)
        elif self.kind == "reshape":
            result = value.reshape(*self.args)
        elif self.kind == "transpose":
            result = value.transpose(*self.args)
        elif self.kind == "select":
            dim, index = self.args
            result = value.select(dim, index)
        else:
            raise ValueError(f"Unknown facet transform primitive {self.kind!r}.")
        self._assert_shape(result, self.post_shape, "post")
        return result

    def _assert_shape(self, value: Any, expected: tuple[int, ...] | None, label: str) -> None:
        """Assert an optional shape contract for a primitive boundary.

        Parameters
        ----------
        value:
            Tensor-like value to inspect.
        expected:
            Expected shape, or ``None`` to skip the assertion.
        label:
            Boundary label for error messages.
        """

        if expected is None or not hasattr(value, "shape"):
            return
        actual = tuple(value.shape)
        if actual != expected:
            raise ValueError(
                f"Facet transform {self.kind!r} {label}-shape assertion failed: "
                f"expected {expected}, got {actual}."
            )


@dataclass(frozen=True)
class FacetSpec:
    """Portable ABI for one facet's home and transform chain.

    Parameters
    ----------
    home_kind:
        Kind of home value.
    home_label:
        Stable home label when available.
    home_address:
        Stable home address when available.
    pass_index:
        Pass index for pass-qualified homes.
    call_index:
        Module call index for module homes.
    output_path:
        Structural output path inside a container output.
    transforms:
        Primitive transform chain.
    capability_class:
        Weakest capability class of the transform chain.
    capability_flags:
        Weakest-link capability flags.
    value_version:
        Which home value version the spec reads.
    conflict_alias_group:
        Shared-home conflict/alias group for future intervention.
    recipe_id:
        Recipe identifier that produced this facet.
    recipe_version:
        Recipe version that produced this facet.
    home:
        Runtime home object. This is intentionally non-portable.
    """

    home_kind: HomeKind
    home_label: str | None = None
    home_address: str | None = None
    pass_index: int | None = None
    call_index: int | None = None
    output_path: tuple[OutputPathComponent, ...] = ()
    transforms: tuple[TransformPrimitive, ...] = ()
    capability_class: CapabilityClass = "bijective_view"
    capability_flags: FacetCapabilityFlags = field(default_factory=FacetCapabilityFlags)
    value_version: ValueVersion = "raw_out"
    conflict_alias_group: str | None = None
    recipe_id: str | None = None
    recipe_version: str | None = None
    home: Any | None = field(default=None, compare=False, repr=False)

    @classmethod
    def from_home(
        cls,
        home: Any,
        *,
        home_kind: HomeKind = "op",
        recipe_id: str | None = None,
        recipe_version: str | None = None,
    ) -> "FacetSpec":
        """Create a spec anchored to a runtime home object.

        Parameters
        ----------
        home:
            Runtime home object.
        home_kind:
            Kind of home value.
        recipe_id:
            Recipe identifier.
        recipe_version:
            Recipe version.

        Returns
        -------
        FacetSpec
            New home-anchored spec.
        """

        flags = FacetCapabilityFlags(read=True, grad=home_kind == "op", write=home_kind == "op")
        return cls(
            home_kind=home_kind,
            home_label=getattr(home, "label", getattr(home, "call_label", None)),
            home_address=getattr(home, "address", None),
            pass_index=getattr(home, "pass_index", None),
            call_index=getattr(home, "call_index", None),
            output_path=tuple(getattr(home, "container_path", ()) or ()),
            capability_flags=flags,
            recipe_id=recipe_id,
            recipe_version=recipe_version,
            conflict_alias_group=getattr(home, "label", None),
            home=home,
        )

    @classmethod
    def computed(
        cls,
        value_func: Callable[[], Any],
        *,
        recipe_id: str | None = None,
        recipe_version: str | None = None,
    ) -> "FacetSpec":
        """Create a read-only computed facet spec.

        Parameters
        ----------
        value_func:
            Callable that computes the read value.
        recipe_id:
            Recipe identifier.
        recipe_version:
            Recipe version.

        Returns
        -------
        FacetSpec
            Read-only computed spec.
        """

        return cls(
            home_kind="computed",
            capability_class="computed",
            capability_flags=_CAPABILITY_FLAGS_BY_CLASS["computed"],
            recipe_id=recipe_id,
            recipe_version=recipe_version,
            home=value_func,
        )

    def __getitem__(self, key: Any) -> "FacetSpec":
        """Append a ``__getitem__`` selection primitive."""

        return self._append(TransformPrimitive("getitem", (key,), capability_class="selection"))

    def heads(self, n_heads: int, d_head: int) -> "FacetSpec":
        """Append a projection-to-heads reshape primitive."""

        return self._append(
            TransformPrimitive("heads", (n_heads, d_head), capability_class="bijective_view")
        )

    def split(self, sections: int, dim: int = -1) -> tuple["FacetSpec", ...]:
        """Return specs for equal sections along a dimension."""

        return tuple(
            self._append(
                TransformPrimitive("split", (sections, dim, index), capability_class="selection")
            )
            for index in range(sections)
        )

    def reshape(self, *shape: int) -> "FacetSpec":
        """Append a reshape primitive."""

        return self._append(TransformPrimitive("reshape", shape, capability_class="bijective_view"))

    def transpose(self, dim0: int, dim1: int) -> "FacetSpec":
        """Append a transpose primitive."""

        return self._append(
            TransformPrimitive("transpose", (dim0, dim1), capability_class="bijective_view")
        )

    def select(self, dim: int, index: int, *, aliasing: bool = False) -> "FacetSpec":
        """Append a dimension selection primitive."""

        capability: CapabilityClass = "aliasing_selection" if aliasing else "selection"
        return self._append(TransformPrimitive("select", (dim, index), capability_class=capability))

    def read(self) -> Any:
        """Read the facet value from its runtime home."""

        return self.apply(self._home_value("out"))

    def grad(self) -> Any:
        """Read the facet gradient from its runtime home or return ``MissingGradient``."""

        if not self.capability_flags.grad or self.home_kind != "op":
            return MissingGradient(self._missing_gradient_reason("facet is not grad-capable"))
        grad_value = self._home_value("grad")
        if grad_value is None:
            return MissingGradient(self._missing_gradient_reason("home op has no saved gradient"))
        return self.apply(grad_value)

    def apply(self, value: Any) -> Any:
        """Apply the transform chain to a value."""

        result = value
        for primitive in self.transforms:
            result = primitive.apply(result)
        return result

    def _append(self, primitive: TransformPrimitive) -> "FacetSpec":
        """Return a copy with one additional primitive."""

        transforms = (*self.transforms, primitive)
        flags = self.capability_flags.intersect(primitive.flags)
        return replace(
            self,
            transforms=transforms,
            capability_class=_weakest_capability_class(transforms),
            capability_flags=flags,
        )

    def _home_value(self, field_name: Literal["out", "grad"]) -> Any:
        """Return a runtime value from the home object."""

        if self.home is None:
            raise RuntimeError(f"FacetSpec {self.recipe_id!r} has no runtime home object.")
        if self.home_kind == "parameter":
            parameter_value = getattr(self.home, "value", None)
            if parameter_value is None:
                parameter_value = getattr(self.home, "_param_ref", self.home)
            if field_name == "grad":
                return getattr(parameter_value, "grad", None)
            return parameter_value
        if self.home_kind == "module_input":
            if field_name == "grad":
                return None
            return self.home
        if self.home_kind == "computed":
            if field_name == "grad":
                return None
            return self.home() if callable(self.home) else self.home
        return getattr(self.home, field_name, None)

    def _missing_gradient_reason(self, detail: str) -> str:
        """Return an actionable missing-gradient instruction."""

        label = self.home_label or self.home_address or "<unknown>"
        return (
            f"Facet gradient unavailable for home {label!r}: {detail}. "
            "Recapture with backward_ready=True and gradients_to_save including "
            f"{label!r}, then run trace.log_backward(...) or a recorded backward pass."
        )


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

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Raise when a missing facet is used as a torch value."""

        for arg in args:
            if isinstance(arg, MissingFacet):
                arg.raise_error()
        raise RuntimeError("MissingFacet used as a torch value.")


class MissingGradient:
    """Sentinel for a known facet gradient that is not currently saved."""

    def __init__(self, reason: str) -> None:
        """Initialize the missing-gradient sentinel.

        Parameters
        ----------
        reason:
            Actionable explanation for how to recapture the gradient.
        """

        self.reason = reason
        self.recapture_instruction = reason

    def raise_error(self) -> None:
        """Raise the missing-gradient tensor-use error."""

        raise RuntimeError(self.reason)

    def __bool__(self) -> bool:
        """Return ``False`` for availability checks."""

        return False

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Raise when the sentinel is used as a torch value."""

        for arg in args:
            if isinstance(arg, MissingGradient):
                arg.raise_error()
        raise RuntimeError("MissingGradient used as a torch value.")

    def __array__(self, dtype: Any | None = None) -> Any:
        """Raise when NumPy tries to coerce the sentinel."""

        self.raise_error()

    def __getitem__(self, key: Any) -> Any:
        """Raise when item access treats the sentinel as a tensor."""

        self.raise_error()


class Facet:
    """Lazy tensor-like runtime wrapper for a ``FacetSpec``."""

    def __init__(self, spec: FacetSpec) -> None:
        """Initialize a runtime facet wrapper.

        Parameters
        ----------
        spec:
            Facet specification to evaluate lazily.
        """

        self.spec = spec

    @property
    def value(self) -> Any:
        """Return the lazily read facet value."""

        return self.spec.read()

    @property
    def grad(self) -> Any:
        """Return the lazily projected facet gradient or ``MissingGradient``."""

        return self.spec.grad()

    def __getattr__(self, name: str) -> Any:
        """Delegate tensor attributes to the read value."""

        return getattr(self.value, name)

    def __getitem__(self, key: Any) -> Any:
        """Delegate item access to the read value."""

        return self.value[key]

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Delegate torch operations to the read value."""

        return func(*_unwrap_facets(args), **(_unwrap_facets(kwargs or {})))


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
        is_aliasing = False
        if name in {"k", "v"}:
            n_q_heads = self._parent.get("n_q_heads", self._parent.get("n_heads", None))
            n_kv_heads = self._parent.get("n_kv_heads", n_q_heads)
            if isinstance(n_q_heads, int) and isinstance(n_kv_heads, int) and n_kv_heads:
                group_size = max(1, n_q_heads // n_kv_heads)
                head_index = head_index // group_size
                is_aliasing = n_q_heads != n_kv_heads
        if isinstance(value, Facet):
            return Facet(value.spec.select(2, head_index, aliasing=is_aliasing))
        return value[:, :, head_index, :]


class FacetView(Mapping[FacetKey, Any]):
    """Lazy dict-like and attribute-access semantic view on a TorchLens record."""

    def __init__(self, record: Any) -> None:
        """Create a facet view for a record.

        Parameters
        ----------
        record:
            TorchLens Op or Module record.
        """

        self._record = record
        self._snapshot = _snapshot_for_record(record)
        self._recipes = _matching_recipes(record, self._snapshot)
        self._cache: dict[FacetKey, Any] = {}
        self._computed = False
        self._structural = _structural_facets(record)

    @property
    def recipe_source(self) -> str | tuple[str, ...] | None:
        """Return recipe name provenance for this view."""

        names = tuple(
            recipe.public.recipe_name
            for recipe in sorted(
                self._recipes, key=lambda item: _recipe_sort_key(item, self._record)
            )
        )
        if not names:
            return None
        if len(names) == 1:
            return names[0]
        return names

    def keys(self) -> builtins.list[FacetKey]:  # type: ignore[override]
        """Return available facet names without invoking recipe functions."""

        names: builtins.list[FacetKey] = []
        seen: set[FacetKey] = set()
        for name in self._structural:
            if name not in seen:
                names.append(name)
                seen.add(name)
        for recipe in self._recipes:
            for name in recipe.declared_facets:
                if name not in seen:
                    names.append(name)
                    seen.add(name)
        return names

    def has(self, name: FacetKey) -> bool:
        """Return whether a facet name is declared without computing values."""

        return name in self.keys()

    def invalidate(self) -> None:
        """Clear cached computed facet values."""

        self._cache.clear()
        self._computed = False

    def get(self, name: FacetKey, default: Any = None) -> Any:
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

    def __getitem__(self, name: FacetKey) -> Any:
        """Return a facet value by key, computing lazily if needed."""

        if name not in self._cache:
            self._compute()
        if name not in self._cache:
            raise KeyError(name)
        value = self._cache[name]
        if isinstance(value, FacetSpec):
            value = Facet(value)
            self._cache[name] = value
        return value

    def __getattr__(self, name: str) -> Any:
        """Return a facet value by attribute name."""

        if not name.isidentifier() or name in _FACET_VIEW_RESERVED_NAMES:
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __iter__(self) -> Iterator[FacetKey]:
        """Iterate declared facet names."""

        return iter(self.keys())

    def __len__(self) -> int:
        """Return the number of declared facet names."""

        return len(self.keys())

    def _compute(self) -> None:
        """Invoke matching recipes and merge their facet dictionaries."""

        if self._computed:
            return
        values: dict[FacetKey, Any] = dict(self._structural)
        facet_tiers: dict[FacetKey, tuple[int, int]] = {key: (-1, -1) for key in self._structural}
        for recipe in sorted(self._recipes, key=lambda item: _recipe_sort_key(item, self._record)):
            contribution = recipe.func(self._record)
            tier = _recipe_priority(recipe, self._record)
            for name, value in contribution.items():
                if name in values:
                    previous_tier = facet_tiers.get(name)
                    if previous_tier == tier:
                        warnings.warn(
                            f"Facet {name!r} has ambiguous same-tier recipes; "
                            f"{recipe.public.recipe_name!r} wins by deterministic order.",
                            UserWarning,
                            stacklevel=2,
                        )
                values[name] = value
                facet_tiers[name] = tier
        self._cache = values
        self._computed = True


_REGISTRY: builtins.list[_RegisteredRecipe] = []
_BUILTIN_REGISTRY: tuple[_RegisteredRecipe, ...] = ()
_REGISTRY_VERSION = 0
_RECIPE_COUNTER = itertools.count()
_CONTEXT_RECIPES: contextvars.ContextVar[tuple[_RegisteredRecipe, ...]] = contextvars.ContextVar(
    "torchlens_facets_recipes", default=()
)
_FACET_VIEW_RESERVED_NAMES = frozenset(
    {
        "get",
        "has",
        "head",
        "items",
        "keys",
        "recipe_source",
        "values",
    }
)


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

        global _REGISTRY_VERSION
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
                    source="user",
                ),
                func=func,
                predicate=predicate,
                declared_facets=declared,
                order=next(_RECIPE_COUNTER),
            )
        )
        _REGISTRY_VERSION += 1
        return func

    return decorator


def reset() -> None:
    """Reset the process registry to the built-in recipe set."""

    global _REGISTRY_VERSION
    _REGISTRY[:] = _BUILTIN_REGISTRY
    _REGISTRY_VERSION += 1


@contextlib.contextmanager
def using(*recipes: RecipeFunc | Iterable[RecipeFunc]) -> Iterator[None]:
    """Temporarily add recipes to snapshots constructed in this context.

    Parameters
    ----------
    recipes:
        Recipe functions. A single iterable is also accepted for convenience.

    Yields
    ------
    None
        Context body where new trace captures include the supplied recipes.
    """

    flattened = _flatten_recipe_args(recipes)
    entries = tuple(_entry_for_recipe(func) for func in flattened)
    token = _CONTEXT_RECIPES.set((*_CONTEXT_RECIPES.get(), *entries))
    try:
        yield
    finally:
        _CONTEXT_RECIPES.reset(token)


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


def snapshot(extra_recipes: Sequence[RecipeFunc] | None = None) -> FacetRegistrySnapshot:
    """Return an immutable active recipe snapshot for a new trace.

    Parameters
    ----------
    extra_recipes:
        Per-trace additive recipes supplied to ``tl.trace(..., recipes=[...])``.

    Returns
    -------
    FacetRegistrySnapshot
        Frozen recipe snapshot with version and provenance metadata.
    """

    extra_entries = tuple(_entry_for_recipe(func) for func in (extra_recipes or ()))
    recipes = (*_REGISTRY, *_CONTEXT_RECIPES.get(), *extra_entries)
    digest = hashlib.sha256()
    for recipe in recipes:
        digest.update(recipe.public.recipe_name.encode("utf-8"))
        digest.update(repr(recipe.public.class_names).encode("utf-8"))
        digest.update(repr(recipe.public.qualnames).encode("utf-8"))
        digest.update(str(recipe.public.has_predicate).encode("utf-8"))
        digest.update(recipe.public.target_scope.encode("utf-8"))
        digest.update(recipe.public.source.encode("utf-8"))
    return FacetRegistrySnapshot(
        version=_REGISTRY_VERSION,
        provenance_id=digest.hexdigest()[:16],
        recipes=tuple(recipes),
    )


def mark_current_registry_as_builtins() -> None:
    """Record the current registry as the built-in reset target."""

    global _BUILTIN_REGISTRY
    marked: builtins.list[_RegisteredRecipe] = []
    for recipe in _REGISTRY:
        marked.append(replace(recipe, public=replace(recipe.public, source="built-in")))
    _REGISTRY[:] = marked
    _BUILTIN_REGISTRY = tuple(marked)


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


def _matching_recipes(
    record: Any, snapshot_value: FacetRegistrySnapshot
) -> builtins.list[_RegisteredRecipe]:
    """Return recipes whose matchers all pass for a record."""

    return [recipe for recipe in snapshot_value.recipes if _recipe_matches(recipe, record)]


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


def _weakest_capability_class(
    transforms: tuple[TransformPrimitive, ...],
) -> CapabilityClass:
    """Return the weakest capability class in a transform chain."""

    if any(primitive.capability_class == "computed" for primitive in transforms):
        return "computed"
    if any(primitive.capability_class == "aliasing_selection" for primitive in transforms):
        return "aliasing_selection"
    if any(primitive.capability_class == "selection" for primitive in transforms):
        return "selection"
    return "bijective_view"


def _unwrap_facets(value: Any) -> Any:
    """Recursively replace ``Facet`` wrappers with their read values."""

    if isinstance(value, Facet):
        return value.value
    if isinstance(value, tuple):
        return tuple(_unwrap_facets(item) for item in value)
    if isinstance(value, builtins.list):
        return [_unwrap_facets(item) for item in value]
    if isinstance(value, dict):
        return {key: _unwrap_facets(item) for key, item in value.items()}
    return value


def _recipe_priority(recipe: _RegisteredRecipe, record: Any) -> tuple[int, int]:
    """Return ``(specificity, source)`` for conflict resolution."""

    class_name = getattr(record, "class_name", getattr(record, "type", None))
    class_qualname = getattr(record, "class_qualname", getattr(record, "func_qualname", None))
    specificity = 0
    if recipe.public.qualnames and class_qualname in recipe.public.qualnames:
        specificity = 3
    elif recipe.public.class_names and class_name in recipe.public.class_names:
        specificity = 2
    elif recipe.predicate is not None:
        specificity = 1
    source = 1 if recipe.public.source == "user" else 0
    return (specificity, source)


def _recipe_sort_key(recipe: _RegisteredRecipe, record: Any) -> tuple[int, int, int, str]:
    """Return deterministic low-to-high recipe sort key."""

    specificity, source = _recipe_priority(recipe, record)
    return (specificity, source, recipe.order, recipe.public.recipe_name)


def _snapshot_for_record(record: Any) -> FacetRegistrySnapshot:
    """Return the owning trace snapshot for a record, or a current fallback."""

    trace = getattr(record, "trace", None)
    if trace is None:
        trace = getattr(record, "source_trace", None)
    snapshot_value = getattr(trace, "facet_registry_snapshot", None)
    if isinstance(snapshot_value, FacetRegistrySnapshot):
        return snapshot_value
    return snapshot()


def _flatten_recipe_args(
    recipes: tuple[RecipeFunc | Iterable[RecipeFunc], ...],
) -> tuple[RecipeFunc, ...]:
    """Normalize ``using`` positional arguments to a function tuple."""

    if len(recipes) == 1 and not callable(recipes[0]):
        return tuple(recipes[0])
    return tuple(recipe for recipe in recipes if callable(recipe))


def _entry_for_recipe(func: RecipeFunc) -> _RegisteredRecipe:
    """Return registry metadata for a recipe function."""

    for recipe in reversed(_REGISTRY):
        if recipe.func is func:
            return recipe
    return _RegisteredRecipe(
        public=FacetRecipe(
            recipe_name=getattr(func, "__name__", type(func).__name__),
            class_names=(),
            qualnames=(),
            has_predicate=False,
            target_scope="any",
            source="user",
        ),
        func=func,
        predicate=None,
        declared_facets=_literal_return_keys(func),
        order=next(_RECIPE_COUNTER),
    )


def _structural_facets(record: Any) -> dict[FacetKey, Any]:
    """Return structural output facets already captured on a record."""

    if _is_module_record(record):
        try:
            call = record._single_call_or_error()
        except AttributeError:
            call = record
        return _module_call_structural_facets(call)
    if hasattr(record, "out"):
        return _op_structural_facets(record)
    return {}


def _is_module_record(record: Any) -> bool:
    """Return whether a record is module-like for structural facets."""

    return hasattr(record, "output_ops") and hasattr(record, "trace")


def _module_call_structural_facets(record: Any) -> dict[FacetKey, Any]:
    """Build structural output facets for a module call."""

    trace = getattr(record, "trace", None)
    if trace is None:
        return {}
    output_ops = builtins.list(getattr(record, "output_ops", ()) or ())
    if not output_ops:
        return {}
    output_records = [trace.ops[label] for label in output_ops]
    names = _primary_structural_names(output_records)
    facets: dict[FacetKey, Any] = {}
    for op, name_value in zip(output_records, names, strict=True):
        spec = FacetSpec.from_home(op, home_kind="op", recipe_id="structural_outputs")
        if name_value is not None:
            facets[name_value] = spec
        path_key = tuple(getattr(op, "container_path", ()) or ())
        if path_key:
            facets[path_key] = spec
            dotted = _path_to_dotted_name(path_key)
            if dotted is not None and dotted not in facets:
                facets[dotted] = spec
    if len(output_records) == 1:
        facets.setdefault(
            "out",
            FacetSpec.from_home(output_records[0], home_kind="op", recipe_id="structural_outputs"),
        )
    return facets


def _op_structural_facets(record: Any) -> dict[FacetKey, Any]:
    """Build structural output facets for one operation record."""

    spec = FacetSpec.from_home(record, home_kind="op", recipe_id="structural_outputs")
    facets: dict[FacetKey, Any] = {"out": spec}
    name = _canonical_multi_output_name(getattr(record, "multi_output_name", None))
    if name is not None:
        facets[name] = spec
    path_key = tuple(getattr(record, "container_path", ()) or ())
    if path_key:
        facets[path_key] = spec
        dotted = _path_to_dotted_name(path_key)
        if dotted is not None:
            facets[dotted] = spec
    return facets


def _primary_structural_names(records: Sequence[Any]) -> builtins.list[str | None]:
    """Return collision-safe primary string names for output records."""

    candidates: builtins.list[str] = []
    for index, record in enumerate(records):
        name = _canonical_multi_output_name(getattr(record, "multi_output_name", None))
        if name is not None:
            candidates.append(name)
            continue
        path_key = tuple(getattr(record, "container_path", ()) or ())
        dotted = _path_to_dotted_name(path_key)
        candidates.append(dotted or f"out{index}")
    duplicates = {name for name in candidates if candidates.count(name) > 1}
    return [None if name in duplicates else name for name in candidates]


def _path_to_dotted_name(path: tuple[OutputPathComponent, ...]) -> str | None:
    """Return a dotted structural name for ordinary positional paths."""

    if not path:
        return None
    parts: builtins.list[str] = []
    for index, component in enumerate(path):
        part = _path_component_to_name(component)
        if part is None:
            return None
        if index == 0 and isinstance(component, TupleIndex):
            parts.append(f"out{part}")
        else:
            parts.append(part)
    return ".".join(parts)


def _canonical_multi_output_name(name: Any) -> str | None:
    """Return the public structural facet name for a captured output name."""

    if not isinstance(name, str) or not name:
        return None
    first, separator, rest = name.partition(".")
    if first.isdecimal():
        return f"out{first}{separator}{rest}" if separator else f"out{first}"
    return name


def _path_component_to_name(component: OutputPathComponent) -> str | None:
    """Return a string component name when it has unambiguous item syntax."""

    if isinstance(component, TupleIndex):
        return str(component.index)
    if isinstance(component, NamedField | DataclassField):
        return component.name
    if isinstance(component, DictKey | HFKey):
        return str(component.key)
    if isinstance(component, int):
        return str(component)
    if isinstance(component, str):
        return component
    return None


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
