"""Typed selectors for TorchLens intervention site resolution."""

from __future__ import annotations

import builtins
import re as _re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

from .._errors import ArgumentTypeError
from .errors import SelectorCompositionError
from .selector_base import (
    _NODE_KINDS,
    _QUALIFIER_KINDS,
    BaseSelector,
    CompositeSelector as CompositeSelector,
    NotSelector,
    SelectorUniverse,
    classify_selector_universe as classify_selector_universe,
)
from .target import FrozenTargetSpec, TargetSpec

#: A latent facet name, ``"<sae name>[k]"`` or ``"<sae name>[j,k]"`` (see
#: ``FacetSelector.latent``); matched for display only, never for resolution.
_LATENT_FACET_NAME = _re.compile(r"^(?P<base>[^\[\],\s]+)\[(?P<index>\d+(?:,\d+)*,?)\]$")

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
    "grad",
]


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
                "tl.site(module_path='encoder.0', func='linear')",
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
            from .._postprocess._site_key import parse_site_key

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
        from .._postprocess._site_key import parse_site_key

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
    func: str | None = None,
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
    func:
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

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> trace[tl.site(func="relu")].labels
    ('relu_1_2:1',)
    """

    return SiteSelector(
        key,
        module_path=module_path,
        op_type=func,
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

    def latent(self, index: int | Sequence[int]) -> FacetSelector:
        """Return a copy scoped to latents of the SAE registered under this facet name.

        ``tl.facet("res").latent(3442)`` names latent 3442 of the SAE
        registered as ``"res"`` (``tl.bridge.sae.register``) at its one site; a
        list names several latents of that SAE, edited together in one write.
        Reads return the encoder activation; an edit adds the decoder-space
        change of the edited latents to the site and leaves the SAE's
        reconstruction error untouched. DOCUMENTED-UNSTABLE.

        Parameters
        ----------
        index:
            Latent index, or a non-empty sequence of latent indices.

        Returns
        -------
        FacetSelector
            Facet selector for those latents.

        Raises
        ------
        InvalidArgumentError
            ``sae_latent_family_unknown`` when no SAE is registered under
            this name, ``sae_latent_index_invalid`` or
            ``sae_latent_index_unknown`` for an index the SAE does not have.
        """

        from .._semantic._families import family_facet_name

        return FacetSelector(
            family_facet_name(self.name, index), module_address=self.module_address
        )

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

        latent = _LATENT_FACET_NAME.match(self.name) if self.name is not None else None
        if self.name is None:
            base = f"tl.head({self.head_index!r})"
        elif latent is not None and self.head_index is None:
            indices = latent["index"].rstrip(",")
            spelled = f"[{indices}]" if "," in latent["index"] else indices
            base = f"tl.facet({latent['base']!r}).latent({spelled})"
        elif self.head_index is None:
            base = f"tl.facet({self.name!r})"
        else:
            base = f"tl.facet({self.name!r}).head({self.head_index!r})"
        if self.module_address is None:
            return base
        return f"{base}.in_module({self.module_address!r})"

    def __hash__(self) -> int:
        """Hash by the selector's spelling, so it can key a ``hooks=`` mapping."""

        return hash(repr(self))


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
class GradSelector(BaseSelector):
    """Gradient address: the gradient of a forward site's OUTPUT.

    ``tl.grad(site)`` is an address, not a computation: it never calls
    autograd. It names the gradient of the loss with respect to the output of
    every forward op the inner selector matches, observed at the TorchLens
    boundary before the gradient enters the op's backward function (the
    tensor ``trace[site].grad`` stores). A bare selector names the activation
    and a node selector (``tl.grad_fn``) names an autograd node, so the three
    universes never share a site.

    Parameters
    ----------
    inner:
        Normalized forward site: a bare label string (kept verbatim, so it
        means what a bare string means at the door that evaluates it) or an
        activation-universe selector.
    """

    def __init__(self, inner: str | BaseSelector) -> None:
        """Create a gradient address over one forward site."""

        object.__setattr__(self, "selector_kind", "grad")
        object.__setattr__(self, "selector_value", inner)

    @property
    def inner(self) -> str | BaseSelector:
        """Return the forward site whose output gradient this address names.

        Returns
        -------
        str | BaseSelector
            The wrapped forward site.
        """

        return cast("str | BaseSelector", self.selector_value)

    def __invert__(self) -> GradSelector:
        """Return the complement inside the gradient universe.

        ``~tl.grad(s)`` normalises to ``tl.grad(~s)``: every forward op's
        output gradient except those ``s`` names.

        Returns
        -------
        GradSelector
            Gradient address over the complemented forward site.
        """

        inner = self.inner
        negated = ~inner if isinstance(inner, BaseSelector) else NotSelector(cast(Any, inner))
        return GradSelector(negated)

    def __selection__(self) -> Any:
        """Refuse lifting a gradient address into an activation Selection.

        Raises
        ------
        SelectorCompositionError
            Always, coded ``grad_selection_unsupported``.
        """

        raise _grad_selection_refusal(self, code="grad_selection_unsupported")

    def to_target_spec(self) -> TargetSpec:
        """Convert the gradient address to a target spec.

        Returns
        -------
        TargetSpec
            Target spec of kind ``"grad"`` whose value is the inner target
            (a nested spec, or the bare label string).
        """

        inner = self.inner
        nested = inner.to_target_spec() if isinstance(inner, BaseSelector) else inner
        return TargetSpec(selector_kind="grad", selector_value=nested)

    def __repr__(self) -> str:
        """Return the constructor-style representation ``tl.grad(...)``."""

        return f"tl.grad({self.inner!r})"


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

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> trace[tl.label("relu_1_2")].label
    'relu_1_2:1'
    >>> len(trace.find_sites(tl.label("relu_1_2")))
    1
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

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> trace[tl.func("conv2d")].labels
    ('conv2d_1_1:1', 'conv2d_2_3:1')
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
    """Create a torch.func / torch.autograd transform-boundary selector.

    Parameters
    ----------
    kind:
        Optional transform kind to match. Unsanitized and sanitized spellings
        are both accepted.

    Returns
    -------
    FuncTransformSelector
        Immutable selector.

    Examples
    --------
    >>> class Mapped(nn.Module):
    ...     def forward(self, v):
    ...         return torch.func.vmap(torch.sin)(v)
    >>> trace = tl.trace(Mapped(), torch.randn(3, 4))
    >>> trace[tl.func_transform("vmap")].labels
    ('vmap_1_1:1',)
    """

    return FuncTransformSelector(kind)


def followed_by(selector: SelectorLike, /) -> FollowedBySelector:
    """Create a retroactive successor selector.

    Parameters
    ----------
    selector:
        Predicate that must match a later successor op.

    Returns
    -------
    FollowedBySelector
        Selector interpreted by capture-time save predicates.

    Examples
    --------
    >>> keep = tl.func("conv2d") & tl.followed_by(tl.func("relu"))
    >>> trace = tl.trace(model, x, save=keep,
    ...                  capture=tl.options.CaptureOptions(lookback=4, lookback_payload_policy="detached_raw"))
    >>> [op.label for op in trace.saved_ops]
    ['conv2d_1_1:1']
    """

    return FollowedBySelector(selector)


def preceded_by(selector: SelectorLike, /) -> PrecededBySelector:
    """Create a lookback predecessor selector.

    Parameters
    ----------
    selector:
        Predicate that must match a retained predecessor op.

    Returns
    -------
    PrecededBySelector
        Selector matching current ops with a retained predecessor.

    Examples
    --------
    >>> keep = tl.func("conv2d") & tl.preceded_by(tl.func("relu"))
    >>> trace = tl.trace(model, x, save=keep, capture=tl.options.CaptureOptions(lookback=4))
    >>> labels = [op.label for op in trace.saved_ops]
    >>> "conv2d_2_3:1" in labels, "conv2d_1_1:1" in labels
    (True, False)
    """

    return PrecededBySelector(selector)


def model_output(target: int | str) -> OutputSelector:
    """Create an output selector.

    Parameters
    ----------
    target:
        Output index or semantic role.

    Returns
    -------
    OutputSelector
        Immutable selector.

    Examples
    --------
    >>> class Split(nn.Module):
    ...     def forward(self, v):
    ...         first, second = torch.chunk(v, 2, dim=1)
    ...         return first * 2 + second
    >>> trace = tl.trace(Split(), torch.randn(1, 4))
    >>> trace[tl.model_output(1)].labels
    ('chunk_2_2:1',)
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


def model_input(*path: Any) -> InputPathSelector:
    """Create a nested model-input path selector.

    Parameters
    ----------
    *path:
        Nested path into a captured model-input container.

    Returns
    -------
    InputPathSelector
        Immutable selector.

    Examples
    --------
    >>> class DictIn(nn.Module):
    ...     def forward(self, d):
    ...         return d["a"] + d["b"]
    >>> inputs = [{"a": torch.zeros(1, 2), "b": torch.ones(1, 2)}]
    >>> capture = tl.options.CaptureOptions(capture_container_structure=True)
    >>> trace = tl.trace(DictIn(), inputs, capture=capture)
    >>> trace[tl.model_input("b")].labels
    ('input_2:1',)
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

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> trace[tl.module("1")].labels
    ('relu_1_2:1',)
    """

    return ModuleSelector(address)


def label_contains(substring: str) -> ContainsSelector:
    """Create a label-substring selector.

    Parameters
    ----------
    substring:
        Substring to match against labels.

    Returns
    -------
    ContainsSelector
        Immutable selector.

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> trace[tl.label_contains("conv")].labels
    ('conv2d_1_1:1', 'conv2d_2_3:1')
    """

    return ContainsSelector(substring)


def label_matches(pattern: str) -> RegexSelector:
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

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> trace[tl.label_matches("^relu_[0-9]+_2$")].labels
    ('relu_1_2:1',)
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

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> trace[tl.where(lambda op: op.func_name == "relu", name_hint="relus")].labels
    ('relu_1_2:1',)
    """

    return WhereSelector(predicate, name_hint=name_hint)


def grad_fn(
    name: str | type[Any] | None = None,
    *,
    label: str | None = None,
    is_custom: bool | None = None,
) -> GradFnSelector:
    """Create a backward grad_fn selector.

    Parameters
    ----------
    name:
        Autograd class name or normalized grad_fn type to match.
    label:
        Substring to match against the grad_fn label.
    is_custom:
        Optional custom-autograd predicate.

    Returns
    -------
    GradFnSelector
        Immutable selector.

    Examples
    --------
    >>> selector = tl.grad_fn("ReluBackward0")
    >>> selector.selector_kind
    'grad_fn'
    """

    return GradFnSelector(name, label=label, is_custom=is_custom)


def without_op() -> InterveningSelector:
    """Create a selector for grad_fns without a paired forward op.

    Returns
    -------
    InterveningSelector
        Immutable selector.

    Examples
    --------
    >>> selector = tl.without_op()
    >>> selector.selector_kind
    'intervening'
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

    Examples
    --------
    >>> selector = tl.facet("pattern")
    >>> selector.selector_kind, selector.name
    ('facet', 'pattern')
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

    Examples
    --------
    >>> selector = tl.head(2, "q")
    >>> selector.head_index, selector.name
    (2, 'q')
    >>> tl.head(0).head(3).head_index
    3
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

    Examples
    --------
    >>> selector = tl.grad_input()
    >>> selector.selector_value
    'grad_input'
    """

    return GradKindSelector("grad_input")


def grad_output() -> GradKindSelector:
    """Create a selector matching backward grad-output events.

    Returns
    -------
    GradKindSelector
        Immutable selector.

    Examples
    --------
    >>> selector = tl.grad_output()
    >>> selector.selector_value
    'grad_output'
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

    Examples
    --------
    >>> selector = tl.in_backward_pass(1)
    >>> selector.pass_index
    1
    """

    return BackwardPassSelector(pass_index)


def _grad_refusal(problem: str, *, code: str, remedy: str) -> Exception:
    """Build one coded ``tl.grad`` construction refusal."""

    return SelectorCompositionError(problem, code=code, remedy=remedy)


def _grad_selection_refusal(value: Any, *, code: str) -> Exception:
    """Build the refusal for a gradient address over (or lifted into) a Selection."""

    if isinstance(value, BaseSelector):
        problem = (
            f"{value!r} addresses a gradient, and a Selection addresses activations; "
            "gradient selections are not addressable this release."
        )
    else:
        problem = (
            f"tl.grad received a {type(value).__name__} (a Selection-algebra value); "
            "gradient selections are not addressable this release."
        )
    return _grad_refusal(
        problem,
        code=code,
        remedy="use tl.grad(selector) with a site selector such as tl.label, tl.func or tl.module",
    )


def _leaf_kind(node: Any) -> str | None:
    """Return the selector kind of one walked selector-tree node."""

    if isinstance(node, (BaseSelector, TargetSpec)):
        return str(node.selector_kind)
    kind = getattr(node, "selector_kind", None)
    return None if kind is None else str(kind)


def _check_grad_inner_tree(inner: str | BaseSelector) -> None:
    """Refuse a ``tl.grad`` inner that is not a forward-only site address."""

    if isinstance(inner, str):
        return
    from .._ir.selector_eval import walk_selector

    kinds = [_leaf_kind(node) for node in walk_selector(inner)]
    if "grad" in kinds:
        raise _grad_refusal(
            f"tl.grad({inner!r}) wraps a gradient address in another one.",
            code="grad_selector_nested",
            remedy="tl.grad wraps a forward site once; write tl.grad(x)",
        )
    if "facet" in kinds:
        raise _grad_refusal(
            f"tl.grad({inner!r}) addresses a facet or head; gradient facets are not "
            "addressable this release.",
            code="grad_facet_unsupported",
            remedy="edit tl.grad(site) and mask inside a callable",
        )
    if any(kind in _QUALIFIER_KINDS for kind in kinds):
        raise _grad_refusal(
            f"tl.grad({inner!r}) wraps a backward pass qualifier, which filters gradient "
            "fires and is not part of the forward site.",
            code="grad_selector_inner_not_forward",
            remedy="put the pass qualifier outside: tl.grad(s) & tl.in_backward_pass(2)",
        )
    if any(kind in _NODE_KINDS for kind in kinds):
        raise _grad_refusal(
            f"tl.grad({inner!r}) wraps an autograd-node selector; tl.grad wraps a forward site.",
            code="grad_selector_inner_not_forward",
            remedy="the autograd-node selectors already address gradients at node granularity; "
            "use them unwrapped",
        )


def _normalize_grad_inner(site: Any) -> str | BaseSelector:
    """Normalize what ``tl.grad`` wraps to a label string or a selector.

    Raises
    ------
    SelectorCompositionError
        Coded refusal for nested, Selection, list/mapping or other non-site
        inputs (facet and node refusals come from the tree check).
    """

    if isinstance(site, (str, BaseSelector)):
        return site
    if isinstance(site, (TargetSpec, FrozenTargetSpec)):
        from .._ir.selector_eval import normalize_selector_like

        return normalize_selector_like(site, lifecycle="site")
    if type(site).__name__ == "_SiteTable":
        from .._ir.selector_eval import normalize_selector_like

        return normalize_selector_like(site, lifecycle="site")
    from ..intervention.hooks import lower_record_site_target

    lowered = lower_record_site_target(site)
    if isinstance(lowered, BaseSelector):
        return lowered
    if hasattr(site, "__selection__"):
        raise _grad_selection_refusal(site, code="grad_selection_unsupported")
    if isinstance(site, (list, tuple, set, frozenset, dict)) or hasattr(site, "keys"):
        raise _grad_refusal(
            f"tl.grad received a {type(site).__name__}; it addresses one site.",
            code="grad_selector_inner_invalid",
            remedy="pass one site; batch shapes go to the door",
        )
    raise _grad_refusal(
        f"tl.grad received {type(site).__name__} {site!r}, which is not a site.",
        code="grad_selector_inner_invalid",
        remedy="pass a label string, a site selector (tl.label, tl.func, tl.module, ...), "
        "an Op or Layer record, or a find_sites() table",
    )


def grad(site: Any, /) -> GradSelector:
    """Address the gradient of a forward site's output.

    ``tl.grad("relu_3")`` names the gradient of relu_3's output, a bare
    ``"relu_3"`` names its activation, and ``tl.grad("relu_3") | "relu_3"``
    names both. The address never calls autograd; the doors decide when it
    fires (on the next backward pass of the trace that holds it).

    Parameters
    ----------
    site:
        A label string (pass-qualified on multi-pass layers), an
        activation-universe selector or composition of them, an ``Op`` or
        ``Layer`` record, or a ``find_sites()`` table.

    Returns
    -------
    GradSelector
        Immutable, hashable gradient address.

    Raises
    ------
    SelectorCompositionError
        ``grad_selector_nested``, ``grad_selector_inner_not_forward``,
        ``grad_facet_unsupported``, ``grad_selection_unsupported`` or
        ``grad_selector_inner_invalid``, each with a ``remedy``.

    Examples
    --------
    >>> repr(tl.grad("relu_1_2"))
    "tl.grad('relu_1_2')"
    >>> both = tl.grad("relu_1_2") | "relu_1_2"
    >>> try:
    ...     tl.grad(tl.grad("relu_1_2"))
    ... except tl.errors.SelectorCompositionError as e:
    ...     print(e.fields["code"])
    grad_selector_nested
    """

    if isinstance(site, GradSelector):
        raise _grad_refusal(
            f"tl.grad({site!r}) wraps a gradient address in another one.",
            code="grad_selector_nested",
            remedy="tl.grad wraps a forward site once; write tl.grad(x)",
        )
    inner = _normalize_grad_inner(site)
    _check_grad_inner_tree(inner)
    return GradSelector(inner)


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

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> try:
    ...     trace[tl.at_step(0)]
    ... except tl.errors.SiteResolutionError as e:
    ...     print(e.fields["code"])
    episode_step_selector_without_episode
    """

    return EpisodeStepSelector(*steps)


def in_module(address: str) -> InModuleSelector:
    """Create a module-containment selector.

    Parameters
    ----------
    address:
        Module address whose contained sites the selector matches.

    Returns
    -------
    InModuleSelector
        Selector matching sites contained in the module. Whether one record
        belongs to a module is read on the record (``layer.module == address``
        or ``address in layer.modules``).

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> trace[tl.in_module("0")].labels
    ('conv2d_1_1:1',)
    """

    return InModuleSelector(str(address))


def _and_universe(
    left: SelectorUniverse | None,
    right: SelectorUniverse | None,
    left_sel: Any,
    right_sel: Any,
) -> SelectorUniverse | None:
    """Combine two ``&`` operands' universes, refusing pairs that share no site."""

    if left is None or right is None:
        other = right if left is None else left
        if other in (None, "activation"):
            return other
        left = left or "activation"
        right = right or "activation"
    if left == right and left != "mixed":
        return left
    pair = {left, right}
    if "mixed" not in pair and "qualifier" in pair and pair & {"gradient", "node"}:
        return "gradient" if "gradient" in pair else "node"
    if pair == {"activation", "qualifier"}:
        site, qualifier = (left_sel, right_sel) if left == "activation" else (right_sel, left_sel)
        raise SelectorCompositionError(
            f"{left_sel!r} & {right_sel!r} joins a backward pass qualifier to an "
            "activation address; activations have no backward pass.",
            code="selector_signal_mismatch",
            remedy=f"a pass qualifier needs a gradient address: tl.grad({site!r}) & {qualifier!r}",
        )
    if "mixed" in pair:
        remedy = "write & inside each branch of the |, e.g. (tl.grad(a) & q) | (a & b)"
    elif pair == {"activation", "gradient"}:
        remedy = "a gradient and an activation address never share a site; use | for both"
    elif "activation" in pair:
        remedy = (
            "spell the node address alone: tl.grad_fn(label=...), or the tensor address tl.grad(s)"
        )
    else:
        remedy = (
            "the tensor address tl.grad(s) and the node address tl.grad_fn(...) differ in "
            "payload; use one of them"
        )
    raise SelectorCompositionError(
        f"Cross-universe composition not supported: {left_sel!r} & {right_sel!r} joins a "
        f"{left} address and a {right} address, which never share a site.",
        code="selector_cross_universe_and",
        remedy=remedy,
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
    "GradSelector",
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
    "label_contains",
    "facet",
    "func",
    "func_transform",
    "followed_by",
    "grad",
    "grad_fn",
    "grad_input",
    "grad_output",
    "label",
    "in_module",
    "in_backward_pass",
    "grad_fn_label",
    "head",
    "module",
    "model_output",
    "output_at",
    "model_input",
    "site",
    "preceded_by",
    "label_matches",
    "where",
    "without_op",
]
