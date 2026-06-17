"""Typed selectors for TorchLens intervention site resolution."""

from __future__ import annotations

import builtins
import re as _re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, overload

from ..ir.container import DataclassField, DictKey, HFKey, NamedField, TupleIndex
from .types import TargetSpec

SelectorKind: TypeAlias = Literal[
    "label",
    "func",
    "func_transform",
    "module",
    "output",
    "output_at",
    "input_at",
    "contains",
    "predicate",
    "in_module",
    "facet",
    "and",
    "or",
    "not",
    "grad_fn",
    "intervening",
    "without_op",
    "regex",
    "followed_by",
    "preceded_by",
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

    def __call__(self, ctx: Any) -> bool:
        """Return whether this selector matches a predicate record context.

        Parameters
        ----------
        ctx:
            Capture-time ``RecordContext`` or a layer-like object.

        Returns
        -------
        bool
            Whether ``ctx`` matches this selector.
        """

        return _selector_matches_record_context(self, ctx)


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
class FuncTransformSelector(BaseSelector):
    """Torch function-transform selector.

    Parameters
    ----------
    kind:
        Optional unsanitized transform kind to match.
    """

    kind: str | None = None

    def __init__(self, kind: str | None = None) -> None:
        """Create a transform selector.

        Parameters
        ----------
        kind:
            Optional transform kind such as ``"vmap"`` or ``"grad"``.
        """

        object.__setattr__(self, "selector_kind", "func_transform")
        object.__setattr__(self, "selector_value", kind)
        object.__setattr__(self, "kind", kind)


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
class OutputPathSelector(BaseSelector):
    """Nested output-path selector.

    Parameters
    ----------
    path:
        Path such as ``("past_key_values", 0, 1)``.
    """

    path: tuple[Any, ...]

    def __init__(self, path: tuple[Any, ...] | list[Any]) -> None:
        """Create a nested output-path selector.

        Parameters
        ----------
        path:
            Nested output path.
        """

        normalized = tuple(path)
        object.__setattr__(self, "selector_kind", "output_at")
        object.__setattr__(self, "selector_value", normalized)
        object.__setattr__(self, "path", normalized)


@dataclass(frozen=True, repr=False)
class InputPathSelector(BaseSelector):
    """Nested model-input path selector.

    Parameters
    ----------
    path:
        Path such as ``("past_key_values", 0, 1)``.
    """

    path: tuple[Any, ...]

    def __init__(self, *path: Any) -> None:
        """Create a nested model-input path selector.

        Parameters
        ----------
        *path:
            Nested input path components.
        """

        normalized = (
            tuple(path[0]) if len(path) == 1 and isinstance(path[0], (tuple, list)) else path
        )
        object.__setattr__(self, "selector_kind", "input_at")
        object.__setattr__(self, "selector_value", normalized)
        object.__setattr__(self, "path", normalized)


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
class RegexSelector(BaseSelector):
    """Label regex-pattern selector.

    Parameters
    ----------
    pattern:
        Regular expression pattern to match against TorchLens labels.
    """

    pattern: str

    def __init__(self, pattern: str) -> None:
        """Create a label regex-pattern selector.

        Parameters
        ----------
        pattern:
            Regular expression pattern to match against TorchLens labels.
        """

        _re.compile(pattern)  # validate at construction time
        object.__setattr__(self, "selector_kind", "regex")
        object.__setattr__(self, "selector_value", pattern)
        object.__setattr__(self, "pattern", pattern)


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
class FollowedBySelector(BaseSelector):
    """Retroactive selector that saves parents when a later op matches.

    Parameters
    ----------
    inner:
        Successor predicate that must match the current operation.
    """

    inner: "SelectorLike"

    def __init__(self, inner: "SelectorLike") -> None:
        """Create a retroactive successor selector."""

        object.__setattr__(self, "selector_kind", "followed_by")
        object.__setattr__(self, "selector_value", inner)
        object.__setattr__(self, "inner", inner)

    def __repr__(self) -> str:
        """Return a concise public representation."""

        return f"tl.followed_by({self.inner!r})"


@dataclass(frozen=True, repr=False)
class PrecededBySelector(BaseSelector):
    """Lookback selector that matches when a recent parent matched.

    Parameters
    ----------
    inner:
        Predecessor predicate evaluated over the retained lookback window.
    """

    inner: "SelectorLike"

    def __init__(self, inner: "SelectorLike") -> None:
        """Create a predecessor selector."""

        object.__setattr__(self, "selector_kind", "preceded_by")
        object.__setattr__(self, "selector_value", inner)
        object.__setattr__(self, "inner", inner)

    def __repr__(self) -> str:
        """Return a concise public representation."""

        return f"tl.preceded_by({self.inner!r})"


@dataclass(frozen=True, repr=False)
class GradFnSelector(BaseSelector):
    """Backward-only selector against grad_fn type, label pattern, or custom flag."""

    type: str | None = None
    grad_fn_label_pattern: str | None = None
    is_custom: bool | None = None
    direction: Literal["backward"] = "backward"

    def __init__(
        self,
        type: str | builtins.type[Any] | None = None,
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
class FacetSelector(BaseSelector):
    """Semantic facet selector for facet-level interventions.

    Parameters
    ----------
    name:
        Facet name to target. ``None`` means the selector targets the default
        attention head facets.
    head_index:
        Optional zero-based head index.
    """

    name: str | None = None
    head_index: int | None = None
    module_address: str | None = None

    def __init__(
        self,
        name: str | None = None,
        *,
        head_index: int | None = None,
        module_address: str | None = None,
    ) -> None:
        """Create a semantic facet selector.

        Parameters
        ----------
        name:
            Facet name to target.
        head_index:
            Optional zero-based head index.
        module_address:
            Optional module address used to scope the selector to one facet owner.
        """

        payload = {"name": name, "head_index": head_index, "module_address": module_address}
        object.__setattr__(self, "selector_kind", "facet")
        object.__setattr__(self, "selector_value", payload)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "head_index", head_index)
        object.__setattr__(self, "module_address", module_address)

    def head(self, head_index: int) -> "FacetSelector":
        """Return a copy scoped to one attention head.

        Parameters
        ----------
        head_index:
            Zero-based head index.

        Returns
        -------
        FacetSelector
            Facet selector with the requested head.
        """

        return FacetSelector(self.name, head_index=head_index, module_address=self.module_address)

    def in_module(self, address: str) -> "FacetSelector":
        """Return a copy scoped to one module address.

        Parameters
        ----------
        address:
            Module address whose facets should be patched.

        Returns
        -------
        FacetSelector
            Facet selector scoped to the requested module address.
        """

        return FacetSelector(self.name, head_index=self.head_index, module_address=address)

    def __repr__(self) -> str:
        """Return a concise public selector representation.

        Returns
        -------
        str
            Constructor-style selector representation.
        """

        if self.name is None:
            base = f"tl.head({self.head_index!r})"
        elif self.head_index is None:
            base = f"tl.facet({self.name!r})"
        else:
            base = f"tl.facet({self.name!r}).head({self.head_index!r})"
        if self.module_address is None:
            return base
        return f"{base}.in_module({self.module_address!r})"


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
class GradKindSelector(BaseSelector):
    """Backward gradient-kind selector for grad inputs or grad outputs."""

    grad_kind: Literal["grad_input", "grad_output"]
    direction: Literal["backward"] = "backward"

    def __init__(self, grad_kind: Literal["grad_input", "grad_output"]) -> None:
        """Create a gradient-kind selector.

        Parameters
        ----------
        grad_kind:
            Gradient event kind to match.
        """

        object.__setattr__(self, "selector_kind", "grad_kind")
        object.__setattr__(self, "selector_value", grad_kind)
        object.__setattr__(self, "grad_kind", grad_kind)
        object.__setattr__(self, "direction", "backward")


@dataclass(frozen=True, repr=False)
class BackwardPassSelector(BaseSelector):
    """Backward selector matching one global backward pass number."""

    pass_index: int
    direction: Literal["backward"] = "backward"

    def __init__(self, pass_index: int) -> None:
        """Create a backward-pass selector.

        Parameters
        ----------
        pass_index:
            One-based backward pass number to match.
        """

        object.__setattr__(self, "selector_kind", "backward_pass")
        object.__setattr__(self, "selector_value", pass_index)
        object.__setattr__(self, "pass_index", pass_index)
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


def func_transform(kind: str | None = None) -> FuncTransformSelector:
    """Create a torch.func transform selector.

    Parameters
    ----------
    kind:
        Optional transform kind to match. Unsanitized and sanitized spellings
        are both accepted.

    Returns
    -------
    FuncTransformSelector
        Immutable selector.
    """

    return FuncTransformSelector(kind)


def followed_by(inner: SelectorLike) -> FollowedBySelector:
    """Create a retroactive successor selector.

    Parameters
    ----------
    inner:
        Predicate that must match a later successor op.

    Returns
    -------
    FollowedBySelector
        Selector interpreted by capture-time save predicates.
    """

    return FollowedBySelector(inner)


def preceded_by(inner: SelectorLike) -> PrecededBySelector:
    """Create a lookback predecessor selector.

    Parameters
    ----------
    inner:
        Predicate that must match a retained predecessor op.

    Returns
    -------
    PrecededBySelector
        Selector matching current ops with a retained predecessor.
    """

    return PrecededBySelector(inner)


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


def output_at(path: tuple[Any, ...] | list[Any]) -> OutputPathSelector:
    """Create a nested output-path selector.

    Parameters
    ----------
    path:
        Nested path into a captured output container.

    Returns
    -------
    OutputPathSelector
        Immutable selector.
    """

    return OutputPathSelector(path)


def input_at(*path: Any) -> InputPathSelector:
    """Create a nested model-input path selector.

    Parameters
    ----------
    *path:
        Nested path into a captured model-input container.

    Returns
    -------
    InputPathSelector
        Immutable selector.
    """

    return InputPathSelector(*path)


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


def regex(pattern: str) -> RegexSelector:
    """Create a label regex-pattern selector.

    Parameters
    ----------
    pattern:
        Regular expression pattern to match against TorchLens labels.
        The pattern is matched with :func:`re.search` (partial match).

    Returns
    -------
    RegexSelector
        Immutable selector.

    Raises
    ------
    re.error
        If ``pattern`` is not a valid regular expression.
    """

    return RegexSelector(pattern)


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


def without_op() -> InterveningSelector:
    """Create a selector for grad_fns without a paired forward op.

    Returns
    -------
    InterveningSelector
        Immutable selector.
    """

    return InterveningSelector()


def intervening() -> InterveningSelector:
    """Deprecated alias for :func:`without_op`.

    Returns
    -------
    InterveningSelector
        Immutable selector.
    """

    from .._deprecations import warn_deprecated_alias

    warn_deprecated_alias("intervening", "without_op")
    return InterveningSelector()


def facet(name: str) -> FacetSelector:
    """Create a semantic facet selector.

    Parameters
    ----------
    name:
        Facet name to target.

    Returns
    -------
    FacetSelector
        Selector resolved to facet home-op hooks by intervention mutators.
    """

    return FacetSelector(name)


def head(index: int, name: str | None = None) -> FacetSelector:
    """Create a selector for one attention head.

    Parameters
    ----------
    index:
        Zero-based attention head index.
    name:
        Optional facet name, such as ``"q"``, ``"k"``, or ``"v"``.

    Returns
    -------
    FacetSelector
        Selector resolved to facet home-op hooks by intervention mutators.
    """

    return FacetSelector(name, head_index=index)


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


def grad_input() -> GradKindSelector:
    """Create a selector matching backward grad-input events.

    Returns
    -------
    GradKindSelector
        Immutable selector.
    """

    return GradKindSelector("grad_input")


def grad_output() -> GradKindSelector:
    """Create a selector matching backward grad-output events.

    Returns
    -------
    GradKindSelector
        Immutable selector.
    """

    return GradKindSelector("grad_output")


def in_backward_pass(pass_index: int) -> BackwardPassSelector:
    """Create a selector matching one backward pass number.

    Parameters
    ----------
    pass_index:
        One-based backward pass number.

    Returns
    -------
    BackwardPassSelector
        Immutable selector.
    """

    return BackwardPassSelector(pass_index)


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


def _context_labels(ctx: Any) -> set[str]:
    """Return all label-like strings visible on a predicate context.

    Parameters
    ----------
    ctx:
        Capture-time context or layer-like object.

    Returns
    -------
    set[str]
        Non-empty label strings available for selector matching.
    """

    labels: set[str] = set()
    for attr in ("label", "raw_label", "label_raw", "layer_label", "layer_label_short"):
        value = getattr(ctx, attr, None)
        if value is not None:
            labels.add(str(value))
    return labels


def _context_module_candidates(ctx: Any) -> tuple[str, ...]:
    """Return module-address candidates from a context.

    Parameters
    ----------
    ctx:
        Capture-time context or layer-like object.

    Returns
    -------
    tuple[str, ...]
        Module addresses and pass-qualified module labels.
    """

    candidates: list[str] = []
    address = getattr(ctx, "address", None)
    if address is not None:
        candidates.append(str(address))
    source_trace = getattr(ctx, "source_trace", None)
    if getattr(source_trace, "module_identity_mode", None) == "function_root":
        candidates.append("self")
        candidates.append("self:1")
    for frame in getattr(ctx, "module_stack", ()):
        frame_address = getattr(frame, "address", None)
        if frame_address is None and isinstance(frame, dict):
            frame_address = frame.get("address")
        if frame_address is None:
            continue
        frame_pass = getattr(frame, "pass_index", None)
        if frame_pass is None and isinstance(frame, dict):
            frame_pass = frame.get("pass_index")
        candidates.append(str(frame_address))
        if frame_pass is not None:
            candidates.append(f"{frame_address}:{frame_pass}")
    modules = getattr(ctx, "modules", ())
    module_ops = getattr(ctx, "output_of_module_calls", ())
    candidates.extend(
        _module_candidate_strings(candidate) for candidate in tuple(modules) + tuple(module_ops)
    )
    return tuple(candidates)


def _module_candidate_strings(candidate: Any) -> str:
    """Return a selector-compatible module candidate string.

    Parameters
    ----------
    candidate
        Module candidate from an op, module call, or capture context.

    Returns
    -------
    str
        Address or pass-qualified address suitable for ``_module_pass_matches``.
    """

    if isinstance(candidate, tuple) and candidate and isinstance(candidate[0], str):
        if len(candidate) > 1:
            return f"{candidate[0]}:{candidate[1]}"
        return candidate[0]
    return str(candidate)


def _sanitize_transform_kind(kind: object) -> str:
    """Return the transform-kind spelling used for labels.

    Parameters
    ----------
    kind:
        Transform kind or selector value.

    Returns
    -------
    str
        Lowercase spelling with underscores and dots removed.
    """

    return str(kind).lower().replace("_", "").replace(".", "")


def _selector_matches_record_context(selector: BaseSelector, ctx: Any) -> bool:
    """Evaluate a selector as a capture-time predicate.

    Parameters
    ----------
    selector:
        Selector to evaluate.
    ctx:
        Capture-time ``RecordContext`` or layer-like object.

    Returns
    -------
    bool
        Whether the selector matches ``ctx``.
    """

    kind = selector.selector_kind
    if kind == "label":
        return str(selector.selector_value) in _context_labels(ctx)
    if kind == "contains":
        needle = str(selector.selector_value)
        return any(needle in label for label in _context_labels(ctx))
    if kind == "regex":
        pattern = str(selector.selector_value)
        return any(_re.search(pattern, label) is not None for label in _context_labels(ctx))
    if kind == "func":
        value = selector.selector_value
        if isinstance(value, dict):
            name = value.get("name")
            output_target = value.get("output")
            if output_target is not None and getattr(ctx, "output_index", None) != output_target:
                return False
        else:
            name = value
        func_name = getattr(ctx, "func_name", None)
        layer_type = getattr(ctx, "layer_type", None)
        return str(name) in {str(func_name), str(layer_type)}
    if kind == "func_transform":
        if not bool(getattr(ctx, "is_transform", False)):
            return False
        value = selector.selector_value
        if value is None:
            return True
        transform_kind = getattr(ctx, "transform_kind", None)
        if transform_kind is None:
            return False
        return _sanitize_transform_kind(transform_kind) == _sanitize_transform_kind(value)
    if kind == "module":
        target = str(selector.selector_value)
        return any(
            _module_pass_matches(candidate, target) for candidate in _context_module_candidates(ctx)
        )
    if kind == "in_module":
        target = str(selector.selector_value)
        return any(
            _module_pass_matches(candidate, target) for candidate in _context_module_candidates(ctx)
        )
    if kind == "output":
        return getattr(ctx, "output_index", None) == selector.selector_value
    if kind == "output_at":
        return _output_path_matches(
            tuple(getattr(ctx, "container_path", ()) or ()),
            tuple(selector.selector_value),
        )
    if kind == "input_at":
        return _input_path_matches(ctx, tuple(selector.selector_value))
    if kind == "predicate":
        predicate = getattr(selector, "predicate")
        return bool(predicate(ctx))
    if kind == "grad_kind":
        return getattr(ctx, "grad_kind", None) == selector.selector_value
    if kind == "backward_pass":
        ctx_pass = getattr(ctx, "backward_pass_index", None)
        if ctx_pass is None:
            ctx_pass = getattr(ctx, "pass_index", None)
        return ctx_pass == selector.selector_value
    if kind == "preceded_by" and isinstance(selector, PrecededBySelector):
        parent_labels = set(
            getattr(ctx, "parent_labels_raw", ()) or getattr(ctx, "parent_labels", ())
        )
        recent_ops = tuple(getattr(ctx, "recent_ops", ()))
        if parent_labels:
            return any(
                (recent.raw_label or recent.label) in parent_labels and bool(selector.inner(recent))  # type: ignore[operator]
                for recent in recent_ops
            )
        return any(bool(selector.inner(recent)) for recent in recent_ops)  # type: ignore[operator]
    if kind == "followed_by":
        return False
    if kind == "and" and isinstance(selector, CompositeSelector):
        left, right = selector.selectors
        return bool(left(ctx)) and bool(right(ctx))  # type: ignore[operator]
    if kind == "or" and isinstance(selector, CompositeSelector):
        left, right = selector.selectors
        return bool(left(ctx)) or bool(right(ctx))  # type: ignore[operator]
    if kind == "not" and isinstance(selector, NotSelector):
        return not bool(selector.selector(ctx))  # type: ignore[operator]
    return False


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


def _output_path_matches(saved_path: tuple[Any, ...], requested_path: tuple[Any, ...]) -> bool:
    """Return whether a captured typed path matches a user path.

    Parameters
    ----------
    saved_path:
        Captured output path.
    requested_path:
        User path.

    Returns
    -------
    bool
        Whether both paths address the same output.
    """

    if len(saved_path) != len(requested_path):
        return False
    return all(
        _output_path_component_matches(saved_component, requested_component)
        for saved_component, requested_component in zip(saved_path, requested_path)
    )


def _output_path_component_matches(saved_component: Any, requested_component: Any) -> bool:
    """Return whether one captured path component matches a user component.

    Parameters
    ----------
    saved_component:
        Captured typed path component.
    requested_component:
        User path component.

    Returns
    -------
    bool
        Whether both components address the same child.
    """

    if isinstance(saved_component, TupleIndex):
        return saved_component.index == requested_component
    if isinstance(saved_component, (DictKey, HFKey)):
        return saved_component.key == requested_component
    if isinstance(saved_component, (NamedField, DataclassField)):
        return saved_component.name == requested_component
    return saved_component == requested_component


def _input_path_matches(ctx: Any, requested_path: tuple[Any, ...]) -> bool:
    """Return whether hook context belongs to an input container path.

    Parameters
    ----------
    ctx:
        Hook context.
    requested_path:
        User-facing path.

    Returns
    -------
    bool
        Whether the context path matches.
    """

    for container in getattr(ctx, "input_containers", ()) or ():
        for occurrence in getattr(container, "leaf_occurrences", ()) or ():
            if _output_path_matches(tuple(occurrence.path), requested_path):
                return True
    return False


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
        if kind in {"grad_fn", "intervening", "without_op", "label"}:
            return "backward"
        if kind in {"func", "func_transform"}:
            return "forward"
        if kind in {
            "label",
            "module",
            "output",
            "output_at",
            "input_at",
            "contains",
            "regex",
            "predicate",
            "in_module",
            "facet",
            "and",
            "or",
            "not",
        }:
            return None
    if isinstance(
        sel,
        (
            GradFnSelector,
            InterveningSelector,
            GradFnLabelSelector,
            GradKindSelector,
            BackwardPassSelector,
        ),
    ):
        return "backward"
    if isinstance(sel, (FuncSelector, FuncTransformSelector)):
        return "forward"
    if isinstance(
        sel,
        (
            LabelSelector,
            ModuleSelector,
            OutputSelector,
            OutputPathSelector,
            InputPathSelector,
            ContainsSelector,
            RegexSelector,
            FacetSelector,
            WhereSelector,
            InModuleSelector,
            FollowedBySelector,
            PrecededBySelector,
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
    "BackwardPassSelector",
    "CompositeSelector",
    "ContainsSelector",
    "FuncSelector",
    "FuncTransformSelector",
    "FollowedBySelector",
    "FacetSelector",
    "GradKindSelector",
    "GradFnLabelSelector",
    "GradFnSelector",
    "InModuleSelector",
    "InputPathSelector",
    "InterveningSelector",
    "LabelSelector",
    "ModuleSelector",
    "NotSelector",
    "OutputSelector",
    "OutputPathSelector",
    "PrecededBySelector",
    "RegexSelector",
    "SelectorLike",
    "WhereSelector",
    "contains",
    "facet",
    "func",
    "func_transform",
    "followed_by",
    "grad_fn",
    "grad_input",
    "grad_output",
    "intervening",
    "label",
    "in_module",
    "in_backward_pass",
    "head",
    "label",
    "module",
    "output",
    "output_at",
    "input_at",
    "preceded_by",
    "regex",
    "where",
    "without_op",
]
