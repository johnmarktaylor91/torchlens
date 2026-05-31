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
    "output",
    "contains",
    "predicate",
    "in_module",
    "and",
    "or",
    "not",
    "grad_fn",
    "intervening",
    "label",
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

        _check_composition(self, other)
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

        _check_composition(self, other)
        return CompositeSelector("or", (self, other))

    def __invert__(self) -> "NotSelector":
        """Return a selector that matches the complement of this selector.

        Returns
        -------
        NotSelector
            Negated selector.
        """

        return NotSelector(self)

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

    def __dir__(self) -> list[str]:
        """Return selector attributes for tab completion.

        Returns
        -------
        list[str]
            Standard selector attributes. Selectors are not bound to a
            ``Trace``, so layer-name completion lives on log accessors.
        """

        return sorted(set(super().__dir__()) | {"selector_kind", "selector_value"})

    def _ipython_key_completions_(self) -> list[str]:
        """Return key completions for IPython.

        Returns
        -------
        list[str]
            Empty list because selector instances have no bound layer universe.
        """

        return []

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
    output: int | str | None = None

    def __init__(self, name: str, *, output: int | str | None = None) -> None:
        """Create a function-name selector.

        Parameters
        ----------
        name:
            Captured function name such as ``"relu"`` or ``"matmul"``.
        """

        object.__setattr__(self, "selector_kind", "func")
        selector_value: Any = name if output is None else {"name": name, "output": output}
        object.__setattr__(self, "selector_value", selector_value)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "output", output)


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
class OutputSelector(BaseSelector):
    """Module or function output selector.

    Parameters
    ----------
    target:
        Output index or semantic output role.
    """

    target: int | str

    def __init__(self, target: int | str) -> None:
        """Create an output selector.

        Parameters
        ----------
        target:
            Output index or semantic output role.
        """

        object.__setattr__(self, "selector_kind", "output")
        object.__setattr__(self, "selector_value", target)
        object.__setattr__(self, "target", target)


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
    """Predicate selector over ``Op`` objects.

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
class GradFnSelector(BaseSelector):
    """Backward-only selector against grad_fn type, label pattern, or custom flag."""

    type: str | None = None
    grad_fn_label_pattern: str | None = None
    is_custom: bool | None = None
    direction: Literal["backward"] = "backward"

    def __init__(
        self,
        type: str | type[Any] | None = None,
        *,
        label: str | None = None,
        is_custom: bool | None = None,
    ) -> None:
        """Create a grad_fn selector.

        Parameters
        ----------
        type:
            Autograd class name or normalized grad_fn type to match.
        label:
            Substring to match against the grad_fn label.
        is_custom:
            Optional custom-autograd predicate.
        """

        if type is not None and not isinstance(type, str):
            type = type.__name__
        payload = {
            "type": type,
            "grad_fn_label_pattern": label,
            "is_custom": is_custom,
        }
        object.__setattr__(self, "selector_kind", "grad_fn")
        object.__setattr__(self, "selector_value", payload)
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "grad_fn_label_pattern", label)
        object.__setattr__(self, "is_custom", is_custom)
        object.__setattr__(self, "direction", "backward")


@dataclass(frozen=True, repr=False)
class InterveningSelector(BaseSelector):
    """Backward-only selector matching grad_fns with no paired forward op."""

    direction: Literal["backward"] = "backward"

    def __init__(self) -> None:
        """Create an intervening-grad_fn selector."""

        object.__setattr__(self, "selector_kind", "intervening")
        object.__setattr__(self, "selector_value", None)
        object.__setattr__(self, "direction", "backward")


@dataclass(frozen=True, repr=False)
class GradFnLabelSelector(BaseSelector):
    """Backward-only selector matching a grad_fn label exactly."""

    label: str
    direction: Literal["backward"] = "backward"

    def __init__(self, name: str) -> None:
        """Create an exact grad_fn label selector.

        Parameters
        ----------
        name:
            GradFn label to match.
        """

        object.__setattr__(self, "selector_kind", "label")
        object.__setattr__(self, "selector_value", name)
        object.__setattr__(self, "label", name)
        object.__setattr__(self, "direction", "backward")


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


@dataclass(frozen=True, repr=False)
class NotSelector(BaseSelector):
    """Selector composed with unary ``~``.

    Parameters
    ----------
    selector:
        Selector to negate.
    """

    selector: SelectorLike

    def __init__(self, selector: SelectorLike) -> None:
        """Create a negated selector.

        Parameters
        ----------
        selector:
            Selector to negate.
        """

        object.__setattr__(self, "selector_kind", "not")
        object.__setattr__(self, "selector_value", selector)
        object.__setattr__(self, "selector", selector)

    def to_target_spec(self) -> TargetSpec:
        """Convert the negated selector to a target spec.

        Returns
        -------
        TargetSpec
            Target spec with a nested selector payload.
        """

        nested = (
            self.selector.to_target_spec()
            if isinstance(self.selector, BaseSelector)
            else self.selector
        )
        return TargetSpec(selector_kind=self.selector_kind, selector_value=nested)

    def __repr__(self) -> str:
        """Return a unary representation of the negated selector.

        Returns
        -------
        str
            Repr with ``~``.
        """

        return f"(~{self.selector!r})"


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


def func(name: str, *, output: int | str | None = None) -> FuncSelector:
    """Create a function-name selector.

    Parameters
    ----------
    name:
        Function name to match.
    output:
        Optional output index or semantic role to match.

    Returns
    -------
    FuncSelector
        Immutable selector.
    """

    return FuncSelector(name, output=output)


def output(target: int | str) -> OutputSelector:
    """Create an output selector.

    Parameters
    ----------
    target:
        Output index or semantic role.

    Returns
    -------
    OutputSelector
        Immutable selector.
    """

    return OutputSelector(target)


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


def grad_fn(
    type: str | type[Any] | None = None,
    *,
    label: str | None = None,
    is_custom: bool | None = None,
) -> GradFnSelector:
    """Create a backward grad_fn selector.

    Parameters
    ----------
    type:
        Autograd class name or normalized grad_fn type to match.
    label:
        Substring to match against the grad_fn label.
    is_custom:
        Optional custom-autograd predicate.

    Returns
    -------
    GradFnSelector
        Immutable selector.
    """

    return GradFnSelector(type, label=label, is_custom=is_custom)


def intervening() -> InterveningSelector:
    """Create a selector for grad_fns without a paired forward op.

    Returns
    -------
    InterveningSelector
        Immutable selector.
    """

    return InterveningSelector()


def grad_fn_label(name: str) -> GradFnLabelSelector:
    """Create an exact grad_fn-label selector.

    Parameters
    ----------
    name:
        GradFn label to match.

    Returns
    -------
    GradFnLabelSelector
        Immutable selector.
    """

    return GradFnLabelSelector(name)


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

    modules = getattr(address_or_layer, "modules", ())
    module_ops = getattr(address_or_layer, "output_of_module_calls", ())
    candidates = tuple(modules) + tuple(module_ops)
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


def _classify_selector_direction(
    sel: SelectorLike,
) -> Literal["forward", "backward"] | None:
    """Return the selector's graph-direction taxonomy bucket.

    Parameters
    ----------
    sel:
        Selector to classify.

    Returns
    -------
    Literal["forward", "backward"] | None
        Explicit graph direction, or None for direction-agnostic selectors.
    """

    from .errors import UnclassifiedSelectorError

    if isinstance(sel, TargetSpec):
        kind = sel.selector_kind
        if kind in {"grad_fn", "intervening", "label"}:
            return "backward"
        if kind == "func":
            return "forward"
        if kind in {
            "label",
            "module",
            "output",
            "contains",
            "predicate",
            "in_module",
            "and",
            "or",
            "not",
        }:
            return None
    if isinstance(sel, (GradFnSelector, InterveningSelector, GradFnLabelSelector)):
        return "backward"
    if isinstance(sel, FuncSelector):
        return "forward"
    if isinstance(
        sel,
        (
            LabelSelector,
            ModuleSelector,
            OutputSelector,
            ContainsSelector,
            WhereSelector,
            InModuleSelector,
            CompositeSelector,
            NotSelector,
        ),
    ):
        return None
    raise UnclassifiedSelectorError(
        f"{type(sel).__name__} has no direction classification; add an explicit bucket."
    )


def _check_composition(a: SelectorLike, b: SelectorLike) -> None:
    """Validate that two selectors can be composed.

    Parameters
    ----------
    a:
        Left selector.
    b:
        Right selector.

    Returns
    -------
    None
        Raises when composition is invalid.
    """

    from .errors import SelectorCompositionError

    a_dir = _classify_selector_direction(a)
    b_dir = _classify_selector_direction(b)
    if a_dir is not None and b_dir is not None and a_dir != b_dir:
        raise SelectorCompositionError(
            "Cross-graph composition not supported: a forward selector and a backward "
            "selector cannot be combined. Use separate forward and backward hook sites."
        )


__all__ = [
    "BaseSelector",
    "CompositeSelector",
    "ContainsSelector",
    "FuncSelector",
    "GradFnLabelSelector",
    "GradFnSelector",
    "InModuleSelector",
    "InterveningSelector",
    "LabelSelector",
    "ModuleSelector",
    "NotSelector",
    "OutputSelector",
    "SelectorLike",
    "WhereSelector",
    "contains",
    "func",
    "grad_fn",
    "label",
    "in_module",
    "intervening",
    "label",
    "module",
    "output",
    "where",
]
