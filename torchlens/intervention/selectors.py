"""Typed selectors for TorchLens intervention site resolution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, overload

from .types import TargetSpec

SelectorKind: TypeAlias = Literal[
    "label",
    "func",
    "module",
    "contains",
    "predicate",
    "in_module",
    "and",
    "or",
]


@dataclass(frozen=True)
class BaseSelector:
    """Base class for typed TorchLens site selectors."""

    selector_kind: SelectorKind
    selector_value: Any

    def __and__(self, other: SelectorLike) -> "CompositeSelector":
        """Return a selector that matches the intersection of two selectors.

        Parameters
        ----------
        other:
            Selector to intersect with this selector.

        Returns
        -------
        CompositeSelector
            Intersection selector.
        """

        return CompositeSelector("and", (self, other))

    def __or__(self, other: SelectorLike) -> "CompositeSelector":
        """Return a selector that matches the union of two selectors.

        Parameters
        ----------
        other:
            Selector to union with this selector.

        Returns
        -------
        CompositeSelector
            Union selector.
        """

        return CompositeSelector("or", (self, other))

    def to_target_spec(self) -> TargetSpec:
        """Convert the selector to a mutable target spec.

        Returns
        -------
        TargetSpec
            Target spec carrying this selector's kind and payload.
        """

        return TargetSpec(
            selector_kind=self.selector_kind,
            selector_value=self.selector_value,
        )

    def __repr__(self) -> str:
        """Return a concise constructor-style representation.

        Returns
        -------
        str
            Repr containing the selector kind and payload.
        """

        return f"tl.{self.selector_kind}({self.selector_value!r})"


@dataclass(frozen=True, repr=False)
class LabelSelector(BaseSelector):
    """Exact TorchLens layer-label selector.

    Parameters
    ----------
    name:
        TorchLens final, raw, short, or pass-qualified label.
    """

    name: str

    def __init__(self, name: str) -> None:
        """Create an exact-label selector.

        Parameters
        ----------
        name:
            TorchLens final, raw, short, or pass-qualified label.
        """

        object.__setattr__(self, "selector_kind", "label")
        object.__setattr__(self, "selector_value", name)
        object.__setattr__(self, "name", name)


@dataclass(frozen=True, repr=False)
class FuncSelector(BaseSelector):
    """Function-name selector.

    Parameters
    ----------
    name:
        Captured function name such as ``"relu"`` or ``"matmul"``.
    """

    name: str

    def __init__(self, name: str) -> None:
        """Create a function-name selector.

        Parameters
        ----------
        name:
            Captured function name such as ``"relu"`` or ``"matmul"``.
        """

        object.__setattr__(self, "selector_kind", "func")
        object.__setattr__(self, "selector_value", name)
        object.__setattr__(self, "name", name)


@dataclass(frozen=True, repr=False)
class ModuleSelector(BaseSelector):
    """Module-output-boundary selector.

    Parameters
    ----------
    address:
        Module address or pass label.
    """

    address: str

    def __init__(self, address: str) -> None:
        """Create a module-address selector.

        Parameters
        ----------
        address:
            Module address or pass label.
        """

        object.__setattr__(self, "selector_kind", "module")
        object.__setattr__(self, "selector_value", address)
        object.__setattr__(self, "address", address)


@dataclass(frozen=True, repr=False)
class ContainsSelector(BaseSelector):
    """Label-substring selector.

    Parameters
    ----------
    substring:
        Substring to match in TorchLens labels.
    """

    substring: str

    def __init__(self, substring: str) -> None:
        """Create a label-substring selector.

        Parameters
        ----------
        substring:
            Substring to match in TorchLens labels.
        """

        object.__setattr__(self, "selector_kind", "contains")
        object.__setattr__(self, "selector_value", substring)
        object.__setattr__(self, "substring", substring)


@dataclass(frozen=True, repr=False)
class WhereSelector(BaseSelector):
    """Predicate selector over ``LayerPassLog`` objects.

    Parameters
    ----------
    predicate:
        Callable that receives a layer pass record.
    name_hint:
        Optional diagnostic label for this non-portable selector.
    """

    predicate: Callable[[Any], bool]
    name_hint: str | None = None

    def __init__(self, predicate: Callable[[Any], bool], *, name_hint: str | None = None) -> None:
        """Create a predicate selector.

        Parameters
        ----------
        predicate:
            Callable that receives a layer pass record.
        name_hint:
            Optional diagnostic label for this non-portable selector.
        """

        payload = (predicate, name_hint)
        object.__setattr__(self, "selector_kind", "predicate")
        object.__setattr__(self, "selector_value", payload)
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "name_hint", name_hint)

    def to_target_spec(self) -> TargetSpec:
        """Convert the predicate selector to a non-portable target spec.

        Returns
        -------
        TargetSpec
            Target spec with predicate metadata.
        """

        return TargetSpec(
            selector_kind=self.selector_kind,
            selector_value=self.predicate,
            metadata={"name_hint": self.name_hint, "portable": False},
        )

    def __repr__(self) -> str:
        """Return a concise predicate selector representation.

        Returns
        -------
        str
            Repr containing the optional name hint.
        """

        if self.name_hint is None:
            return "tl.where(<predicate>)"
        return f"tl.where(<predicate>, name_hint={self.name_hint!r})"


@dataclass(frozen=True, repr=False)
class InModuleSelector(BaseSelector):
    """Module-containment selector.

    Parameters
    ----------
    address:
        Module address whose pass-qualified containment should match.
    """

    address: str

    def __init__(self, address: str) -> None:
        """Create a module-containment selector.

        Parameters
        ----------
        address:
            Module address whose pass-qualified containment should match.
        """

        object.__setattr__(self, "selector_kind", "in_module")
        object.__setattr__(self, "selector_value", address)
        object.__setattr__(self, "address", address)


@dataclass(frozen=True, repr=False)
class CompositeSelector(BaseSelector):
    """Selector composed with ``&`` or ``|``.

    Parameters
    ----------
    operator:
        ``"and"`` for intersection or ``"or"`` for union.
    selectors:
        Pair of selectors to combine.
    """

    operator: Literal["and", "or"]
    selectors: tuple[SelectorLike, SelectorLike]

    def __init__(
        self, operator: Literal["and", "or"], selectors: tuple[SelectorLike, SelectorLike]
    ):
        """Create a composite selector.

        Parameters
        ----------
        operator:
            ``"and"`` for intersection or ``"or"`` for union.
        selectors:
            Pair of selectors to combine.
        """

        object.__setattr__(self, "selector_kind", operator)
        object.__setattr__(self, "selector_value", selectors)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "selectors", selectors)

    def to_target_spec(self) -> TargetSpec:
        """Convert the composite selector to a target spec.

        Returns
        -------
        TargetSpec
            Target spec with nested selector payloads.
        """

        nested = tuple(
            item.to_target_spec() if isinstance(item, BaseSelector) else item
            for item in self.selectors
        )
        return TargetSpec(selector_kind=self.selector_kind, selector_value=nested)

    def __repr__(self) -> str:
        """Return an infix representation of the composite selector.

        Returns
        -------
        str
            Repr with ``&`` or ``|``.
        """

        symbol = "&" if self.operator == "and" else "|"
        left, right = self.selectors
        return f"({left!r} {symbol} {right!r})"


SelectorLike: TypeAlias = BaseSelector | TargetSpec


def label(name: str) -> LabelSelector:
    """Create an exact-label selector.

    Parameters
    ----------
    name:
        TorchLens final, raw, short, or pass-qualified label.

    Returns
    -------
    LabelSelector
        Immutable selector.
    """

    return LabelSelector(name)


def func(name: str) -> FuncSelector:
    """Create a function-name selector.

    Parameters
    ----------
    name:
        Function name to match.

    Returns
    -------
    FuncSelector
        Immutable selector.
    """

    return FuncSelector(name)


def module(address: str) -> ModuleSelector:
    """Create a module-address selector.

    Parameters
    ----------
    address:
        Module address or pass label.

    Returns
    -------
    ModuleSelector
        Immutable selector.
    """

    return ModuleSelector(address)


def contains(substring: str) -> ContainsSelector:
    """Create a label-substring selector.

    Parameters
    ----------
    substring:
        Substring to match against labels.

    Returns
    -------
    ContainsSelector
        Immutable selector.
    """

    return ContainsSelector(substring)


def where(predicate: Callable[[Any], bool], *, name_hint: str | None = None) -> WhereSelector:
    """Create a predicate selector.

    Parameters
    ----------
    predicate:
        Callable that receives a layer pass record.
    name_hint:
        Optional human-readable name for diagnostics and saved specs.

    Returns
    -------
    WhereSelector
        Immutable non-portable selector.
    """

    return WhereSelector(predicate, name_hint=name_hint)


@overload
def in_module(address_or_layer: str) -> InModuleSelector:
    """Create a module-containment selector.

    Parameters
    ----------
    address_or_layer:
        Module address.

    Returns
    -------
    InModuleSelector
        Selector matching sites contained in the module.
    """
    ...


@overload
def in_module(address_or_layer: Any, address: str) -> bool:
    """Test whether a layer pass belongs to a module.

    Parameters
    ----------
    address_or_layer:
        Layer pass record.
    address:
        Module address.

    Returns
    -------
    bool
        Whether the layer pass belongs to the module.
    """
    ...


def in_module(address_or_layer: Any, address: str | None = None) -> InModuleSelector | bool:
    """Create a module-containment selector or test one layer pass.

    Parameters
    ----------
    address_or_layer:
        Module address when called with one argument, or a layer pass record
        when called with two arguments.
    address:
        Module address to test when ``layer_log`` is a record.

    Returns
    -------
    InModuleSelector | bool
        Selector for one-argument calls; containment result for two-argument
        calls retained for architecture-plan compatibility.
    """

    if address is None:
        return InModuleSelector(str(address_or_layer))

    containing_modules = getattr(address_or_layer, "containing_modules", ())
    module_passes = getattr(address_or_layer, "module_passes_exited", ())
    candidates = tuple(containing_modules) + tuple(module_passes)
    return any(_module_pass_matches(candidate, address) for candidate in candidates)


def _module_pass_matches(module_pass: str, address: str) -> bool:
    """Return whether a pass-qualified module label belongs to an address.

    Parameters
    ----------
    module_pass:
        Pass-qualified module label such as ``"encoder:1"``.
    address:
        Module address without pass qualification.

    Returns
    -------
    bool
        Whether the module pass belongs to the requested address.
    """

    module_address = module_pass.rsplit(":", 1)[0]
    return module_pass == address or module_address == address


__all__ = [
    "BaseSelector",
    "CompositeSelector",
    "ContainsSelector",
    "FuncSelector",
    "InModuleSelector",
    "LabelSelector",
    "ModuleSelector",
    "SelectorLike",
    "WhereSelector",
    "contains",
    "func",
    "in_module",
    "label",
    "module",
    "where",
]
