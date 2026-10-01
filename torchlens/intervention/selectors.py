"""Typed selectors for TorchLens intervention site resolution."""

from __future__ import annotations

import builtins
import re as _re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, overload

from .._errors import ArgumentTypeError
from .types import TargetSpec

SelectorKind: TypeAlias = Literal[
    "label",
    "func",
    "func_transform",
    "module",
    "output",
    "output_at",
    "input_at",
    "site",
    "contains",
    "predicate",
    "in_module",
    "facet",
    "and",
    "or",
    "not",
    "grad_fn",
    "grad_fn_label",
    "grad_kind",
    "backward_pass",
    "intervening",
    "without_op",
    "regex",
    "followed_by",
    "preceded_by",
    "site",
    "episode_step",
]


@dataclass(frozen=True)
class BaseSelector:
    """Base class for typed TorchLens site selectors."""

    selector_kind: SelectorKind
    selector_value: Any

    def __and__(self, other: SelectorLike) -> CompositeSelector:
        """Return a selector that matches the intersection of two selectors.

        Parameters
        ----------
        other:
            Selector to intersect with this selector.

        Returns
        -------
        CompositeSelector
            Intersection selector. A non-selector operand that implements
            ``__selection__`` defers to that object's reflected operator, so
            ``selector & rf_box`` composes as a Selection; junk operands keep
            their shipped refusals.
        """

        if not isinstance(other, BaseSelector) and hasattr(other, "__selection__"):
            return NotImplemented
        _check_composition(self, other)
        return CompositeSelector("and", _flatten_same_operator("and", self, other))

    def __or__(self, other: SelectorLike) -> CompositeSelector:
        """Return a selector that matches the union of two selectors.

        Parameters
        ----------
        other:
            Selector to union with this selector.

        Returns
        -------
        CompositeSelector
            Union selector. A non-selector operand that implements
            ``__selection__`` defers to that object's reflected operator.
        """

        if not isinstance(other, BaseSelector) and hasattr(other, "__selection__"):
            return NotImplemented
        from ..ir.selector_eval import contains_followed_by

        if contains_followed_by(self) or contains_followed_by(other):
            from .errors import SelectorCompositionError

            raise SelectorCompositionError(
                "tl.followed_by(...) only supports candidate & tl.followed_by(successor); "
                "OR-composed followed_by selectors cannot be evaluated safely."
            )
        _check_composition(self, other)
        return CompositeSelector("or", _flatten_same_operator("or", self, other))

    def __invert__(self) -> NotSelector:
        """Return a selector that matches the complement of this selector.

        Returns
        -------
        NotSelector
            Negated selector.
        """

        from ..ir.selector_eval import contains_followed_by

        if contains_followed_by(self):
            from .errors import SelectorCompositionError

            raise SelectorCompositionError(
                "tl.followed_by(...) only supports candidate & tl.followed_by(successor); "
                "negated followed_by selectors cannot be evaluated safely."
            )
        return NotSelector(self)

    def __sub__(self, other: SelectorLike) -> CompositeSelector:
        """Return a selector matching this selector minus ``other`` (DESUGAR).

        ``a - b`` desugars to ``CompositeSelector("and", (a, NotSelector(b)))``
        — no new selector kind, no spec round-trip change. Selector evaluation
        is a per-subject characteristic function, so difference of match-sets
        IS ``a(x) and not b(x)``; this coincides with Selection-level ``-`` on
        whole-site lifted terms (law-tested).

        Parameters
        ----------
        other:
            Selector (or predicate callable) to subtract. A non-selector
            operand implementing ``__selection__`` defers to that object's
            reflected operator.

        Returns
        -------
        CompositeSelector
            Difference selector.
        """

        if not isinstance(other, BaseSelector) and hasattr(other, "__selection__"):
            return NotImplemented
        if isinstance(other, BaseSelector):
            negated: SelectorLike = ~other  # shipped guards (followed_by refusal) apply
        elif callable(other):
            negated = NotSelector(other)
        else:
            raise ArgumentTypeError(
                f"cannot subtract {type(other).__name__} from a selector; pass a "
                "selector or predicate callable.",
                code="selector_subtraction_operand_invalid",
                remedy="subtract a selector, predicate callable, or region producer",
            )
        _check_composition(self, negated)
        return CompositeSelector("and", _flatten_same_operator("and", self, negated))

    def __selection__(self) -> Any:
        """Lift this selector as a Selection selector-term.

        The lifted term keeps predicate meaning intact: it resolves to
        whole-site masks over matched sites. Selection-level ``~`` is mask
        complement — users who mean "all sites not matching s" spell ``~s``
        BEFORE lifting (``~lift(s) != lift(~s)``, pinned non-law).
        """

        from ..selection import _selection_from_selector

        return _selection_from_selector(self)

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

        from ..ir.selector_eval import evaluate

        return evaluate(self, ctx, lifecycle="capture")


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
class SiteSelector(BaseSelector):
    """Structural-position selector over ``site_key_v1`` components (C03).

    The site-key-primary WHERE spelling (surgery memo Build 0a/0c): valid in
    EVERY lane under the address law -- post hoc against ``op.site_key``,
    and during a live capture against the streaming live minter
    (``intervention/site_keys.py``). Every spelling DOCUMENTED-UNSTABLE
    pending naming-session ratification.
    """

    key: str | None = None
    module_path: str | None = None
    op_type: str | None = None
    slot: int | None = None
    ordinal: int | None = None

    def __init__(
        self,
        key: str | None = None,
        *,
        module_path: str | None = None,
        op_type: str | None = None,
        slot: int | None = None,
        ordinal: int | None = None,
    ) -> None:
        """Build a site selector from a rendered key or components.

        Raises
        ------
        ArgumentTypeError
            ``site_selector_empty`` without any component;
            ``site_selector_key_conflict`` when a rendered key is combined
            with component filters (the key already fixes every component);
            ``site_selector_key_invalid`` for a malformed rendered key.
        """

        if (
            key is None
            and module_path is None
            and op_type is None
            and ordinal is None
            and slot is None
        ):
            raise ArgumentTypeError(
                "tl.site() needs a rendered site key or at least one "
                "component (module_path=, op_type=, slot=, ordinal=)",
                code="site_selector_empty",
                remedy="pass op.site_key, or components such as "
                "tl.site(module_path='encoder.0', op_type='linear')",
                argument="key",
            )
        if key is not None:
            if (
                module_path is not None
                or op_type is not None
                or slot is not None
                or ordinal is not None
            ):
                raise ArgumentTypeError(
                    "tl.site(key) already fixes every component; combining it "
                    "with component filters has nothing extra to match",
                    code="site_selector_key_conflict",
                    remedy="pass either the rendered key or components, not both",
                    argument="key",
                )
            from ..postprocess._site_key import parse_site_key

            try:
                parse_site_key(key)
            except ValueError as exc:
                raise ArgumentTypeError(
                    f"tl.site(key) received a malformed site key: {exc}",
                    code="site_selector_key_invalid",
                    remedy="pass a rendered op.site_key string (site_key_v1)",
                    argument="key",
                ) from exc
        object.__setattr__(self, "selector_kind", "site")
        object.__setattr__(
            self,
            "selector_value",
            {
                "key": key,
                "module_path": module_path,
                "op_type": op_type,
                "slot": slot,
                "ordinal": ordinal,
            },
        )
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "module_path", module_path)
        object.__setattr__(self, "op_type", op_type)
        object.__setattr__(self, "slot", slot)
        object.__setattr__(self, "ordinal", ordinal)

    def matches_key(self, rendered_key: str | None) -> bool:
        """Whether one rendered ``site_key_v1`` string satisfies this selector.

        A ``None`` key never matches (keyless legacy/orphan rows are not
        silently guessed); component filters compare against the parsed key.
        """

        if not rendered_key:
            return False
        if self.key is not None:
            return rendered_key == self.key
        from ..postprocess._site_key import parse_site_key

        try:
            module_site, layer_type, output_slot, call_ordinal = parse_site_key(rendered_key)
        except ValueError:
            return False
        if self.module_path is not None:
            joined = "/".join(module_site)
            if self.module_path != joined and self.module_path not in module_site:
                return False
        if self.op_type is not None and self.op_type != layer_type:
            return False
        if self.slot is not None and self.slot != output_slot:
            return False
        return not (self.ordinal is not None and self.ordinal != call_ordinal)

    def __repr__(self) -> str:
        """Return the component-form repr."""

        parts = [
            f"{name}={value!r}"
            for name, value in (
                ("key", self.key),
                ("module_path", self.module_path),
                ("op_type", self.op_type),
                ("slot", self.slot),
                ("ordinal", self.ordinal),
            )
            if value is not None
        ]
        return f"tl.site({', '.join(parts)})"


def site(
    key: str | None = None,
    *,
    module_path: str | None = None,
    op_type: str | None = None,
    slot: int | None = None,
    ordinal: int | None = None,
) -> SiteSelector:
    """Select ops by structural position (``site_key_v1`` components).

    Parameters
    ----------
    key:
        A rendered ``op.site_key`` string (exact match).
    module_path:
        Module-site match: the full ``/``-joined site axis or any one
        module address on it.
    op_type:
        TorchLens layer type (``"linear"``, ``"relu"``, ...).
    slot:
        Multi-output slot index (``None`` = single-output ops).
    ordinal:
        1-based occurrence ordinal of the position within its
        pass-qualified innermost call instance.

    Returns
    -------
    SiteSelector
        Structural selector valid in every lane (address law).
    """

    return SiteSelector(
        key,
        module_path=module_path,
        op_type=op_type,
        slot=slot,
        ordinal=ordinal,
    )


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

    inner: SelectorLike

    def __init__(self, inner: SelectorLike) -> None:
        """Create a retroactive successor selector."""

        object.__setattr__(self, "selector_kind", "followed_by")
        object.__setattr__(self, "selector_value", inner)
        object.__setattr__(self, "inner", inner)

    def to_target_spec(self) -> TargetSpec:
        """Convert the successor selector to a target spec.

        Returns
        -------
        TargetSpec
            Target spec with a nested selector payload, so a structural inner
            serializes structurally instead of as an opaque callable.
        """

        nested = self.inner.to_target_spec() if isinstance(self.inner, BaseSelector) else self.inner
        return TargetSpec(selector_kind=self.selector_kind, selector_value=nested)

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

    inner: SelectorLike

    def __init__(self, inner: SelectorLike) -> None:
        """Create a predecessor selector."""

        object.__setattr__(self, "selector_kind", "preceded_by")
        object.__setattr__(self, "selector_value", inner)
        object.__setattr__(self, "inner", inner)

    def to_target_spec(self) -> TargetSpec:
        """Convert the predecessor selector to a target spec.

        Returns
        -------
        TargetSpec
            Target spec with a nested selector payload, so a structural inner
            serializes structurally instead of as an opaque callable.
        """

        nested = self.inner.to_target_spec() if isinstance(self.inner, BaseSelector) else self.inner
        return TargetSpec(selector_kind=self.selector_kind, selector_value=nested)

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

    def __repr__(self) -> str:
        """Return the canonical non-deprecated selector constructor."""

        return "tl.without_op()"


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

    def head(self, head_index: int) -> FacetSelector:
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

    def in_module(self, address: str) -> FacetSelector:
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
    """Backward-only selector matching a grad_fn label exactly.

    Carries its own ``"grad_fn_label"`` selector kind: the earlier ``"label"``
    kind collided with :class:`LabelSelector`, so saving and reloading a spec
    silently converted the selector to a forward label match and post-hoc
    resolution never reached the backward exact-label branch.
    """

    label: str
    direction: Literal["backward"] = "backward"

    def __init__(self, name: str) -> None:
        """Create an exact grad_fn label selector.

        Parameters
        ----------
        name:
            GradFn label to match.
        """

        object.__setattr__(self, "selector_kind", "grad_fn_label")
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
class EpisodeStepSelector(BaseSelector):
    """Episode-step selector: match ops recorded inside declared step(s).

    Lane F42 (attested coupling; spelling DOCUMENTED-UNSTABLE pending the
    naming session): steps are the 0-based episode ledger rows -- stepped-
    module call ``k`` is step ``k``, matching ``trace.episode.rows``. At
    capture time the selector reads the LIVE step position (the armed join
    session's boundary hooks); post hoc it reads the persisted
    ``Op.episode_step`` stamps. Ops between steps (root-loop sampling,
    pre/post work) belong to no step and never match. Compose with the
    existing pass-qualified addressing (``"attn_1_1:2"`` labels) for the
    step-x-pass cross-product.

    Parameters
    ----------
    steps:
        The declared 0-based step indices (at least one; non-negative ints).
    """

    steps: tuple[int, ...] = ()

    def __init__(self, *steps: int) -> None:
        """Create an episode-step selector.

        Parameters
        ----------
        steps:
            The declared 0-based step indices (at least one).
        """

        if not steps:
            raise ArgumentTypeError(
                "at_step() needs at least one 0-based episode step index "
                "(step 0 is the prefill row).",
                code="episode_step_selector_invalid",
                remedy="pass the 0-based step indices to match, e.g. at_step(0, 2)",
                argument="steps",
            )
        normalized: list[int] = []
        for step in steps:
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ArgumentTypeError(
                    f"at_step() steps must be non-negative ints (0-based episode "
                    f"ledger rows); got {step!r}.",
                    code="episode_step_selector_invalid",
                    remedy="pass non-negative 0-based step indices, e.g. at_step(0, 2)",
                    argument="steps",
                    received_type=type(step).__name__,
                )
            normalized.append(int(step))
        ordered = tuple(sorted(set(normalized)))
        object.__setattr__(self, "selector_kind", "episode_step")
        object.__setattr__(self, "selector_value", ordered)
        object.__setattr__(self, "steps", ordered)


@dataclass(frozen=True, repr=False)
class CompositeSelector(BaseSelector):
    """Selector composed with ``&`` or ``|``.

    ``&`` / ``|`` build nested binary composites; deserialized target specs
    may carry a flat n-ary child tuple. Both shapes evaluate identically.
    Degenerate arities follow the standard identity semantics in evaluation
    and spec round-trips alike: an empty ``and`` matches everything, an empty
    ``or`` matches nothing, and a unary composite matches like its child.

    Parameters
    ----------
    operator:
        ``"and"`` for intersection or ``"or"`` for union.
    selectors:
        Selectors to combine.
    """

    operator: Literal["and", "or"]
    selectors: tuple[SelectorLike, ...]

    def __init__(self, operator: Literal["and", "or"], selectors: tuple[SelectorLike, ...]) -> None:
        """Create a composite selector.

        Parameters
        ----------
        operator:
            ``"and"`` for intersection or ``"or"`` for union.
        selectors:
            Selectors to combine.
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
        joined = f" {symbol} ".join(repr(child) for child in self.selectors)
        return f"({joined})"


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

    if not isinstance(name, str):
        raise ArgumentTypeError(
            f"func() pattern has unsupported type {type(name).__name__}",
            code="selector_function_pattern_type_invalid",
            remedy="pass a string function name such as tl.func('relu')",
            argument="name",
            received_type=type(name).__name__,
        )
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


def output_at(path: Any) -> OutputPathSelector:
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

    normalized = tuple(path) if isinstance(path, (tuple, list)) else (path,)
    return OutputPathSelector(normalized)


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


def at_step(*steps: int) -> EpisodeStepSelector:
    """Create an episode-step selector (lane F42's step qualifier).

    Matches ops recorded inside the named 0-based episode step(s) -- the
    stepped-module calls declared by ``tl.trace(..., episode=...)``, aligned
    with ``trace.episode.rows``. Compose with any selector for step-qualified
    capture-time intervention or save predicates
    (``tl.when(tl.func("softmax") & at_step(2), tl.zero_ablate())``), and
    with pass-qualified labels for the step-x-pass cross-product. Refuses
    typed outside an episode capture (``episode_step_selector_without_episode``).

    Parameters
    ----------
    steps:
        The declared 0-based step indices (at least one; step 0 is the
        prefill row).

    Returns
    -------
    EpisodeStepSelector
        Immutable selector.
    """

    return EpisodeStepSelector(*steps)


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

    from ..ir.selector_eval import module_address_matches

    modules = getattr(address_or_layer, "modules", ())
    module_ops = getattr(address_or_layer, "output_of_module_calls", ())
    candidates = tuple(modules) + tuple(module_ops)
    return any(module_address_matches(candidate, address) for candidate in candidates)


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
        if kind in {
            "grad_fn",
            "grad_fn_label",
            "grad_kind",
            "backward_pass",
            "intervening",
            "without_op",
        }:
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
            "site",
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
            SiteSelector,
            FollowedBySelector,
            PrecededBySelector,
            CompositeSelector,
            NotSelector,
            # Step qualification is a position fact, not a graph direction
            # (lane F42): it narrows forward subjects by episode step and
            # composes direction-neutrally like the other position selectors.
            EpisodeStepSelector,
        ),
    ):
        return None
    raise UnclassifiedSelectorError(
        f"{type(sel).__name__} has no direction classification; add an explicit bucket."
    )


def _flatten_same_operator(
    operator: str, left: SelectorLike, right: SelectorLike
) -> tuple[SelectorLike, ...]:
    """Merge same-operator composite operands into one flat child tuple.

    r-b4 R27-6a: ``&``/``|`` used to nest one binary composite per operator, so
    a programmatically composed predicate (``functools.reduce(operator.or_,
    [tl.func(n) for n in names])``) built a 500-deep binary tree and blew the
    interpreter stack at trace entry. Composites are documented n-ary with
    identical evaluation for flat and nested shapes, so chained applications of
    ONE operator now accumulate a flat child tuple: depth stays constant and
    evaluation walks one level. Mixed-operator composition still nests (the
    shape is semantic there), and existing nested trees (deserialized specs)
    keep evaluating unchanged.
    """

    children: list[SelectorLike] = []
    for operand in (left, right):
        if isinstance(operand, CompositeSelector) and operand.operator == operator:
            children.extend(operand.selectors)
        else:
            children.append(operand)
    return tuple(children)


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

    from ..ir.selector_eval import contains_followed_by, flatten_and_conjuncts
    from .errors import SelectorCompositionError

    a_dir = _classify_selector_direction(a)
    b_dir = _classify_selector_direction(b)
    if a_dir is not None and b_dir is not None and a_dir != b_dir:
        raise SelectorCompositionError(
            "Cross-graph composition not supported: a forward selector and a backward "
            "selector cannot be combined. Use separate forward and backward hook sites."
        )
    if contains_followed_by(a) or contains_followed_by(b):
        # `&` nests, so validate the FLAT conjunction: `a & fb & b` must pass
        # exactly like `a & b & fb` and the flat three-child spec.
        conjuncts = flatten_and_conjuncts((a, b))
        direct = [c for c in conjuncts if isinstance(c, FollowedBySelector)]
        buried = [
            c
            for c in conjuncts
            if not isinstance(c, FollowedBySelector) and contains_followed_by(c)
        ]
        if len(direct) != 1 or buried:
            raise SelectorCompositionError(
                "tl.followed_by(...) only supports candidate & tl.followed_by(successor); "
                "nested or multi-followed_by compositions are unsupported."
            )


__all__ = [
    "BaseSelector",
    "BackwardPassSelector",
    "CompositeSelector",
    "ContainsSelector",
    "EpisodeStepSelector",
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
    "SiteSelector",
    "WhereSelector",
    "at_step",
    "contains",
    "facet",
    "func",
    "func_transform",
    "followed_by",
    "grad_fn",
    "grad_input",
    "grad_output",
    "label",
    "in_module",
    "in_backward_pass",
    "grad_fn_label",
    "head",
    "module",
    "output",
    "output_at",
    "input_at",
    "site",
    "preceded_by",
    "regex",
    "where",
    "without_op",
]
