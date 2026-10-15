"""Site targets: attribute a hidden layer's output, or one unit of it, instead of the model output."""

from __future__ import annotations

import difflib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import KW_ONLY, dataclass
from typing import Any

from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from torchlens.attribution._result import AttributionError

_DOCS_POINTER = "See docs/reference/attribution.md, section Site targets."


@dataclass(frozen=True, repr=False)
class SiteTarget:
    """A hidden site to attribute: a layer's output, optionally one unit of it.

    Pass it as ``target=`` to an attribution method to explain what drives an
    internal activation rather than the model output, for example
    ``tl.attribution.saliency(model, x, target=tl.attribution.SiteTarget("fc1", unit=3))``.
    The scalar attributed is the selected part of the layer's output summed
    over the batch, the same summing rule as an integer ``target=``.

    Attributes
    ----------
    layer
        Name from ``dict(model.named_modules())`` whose output is the site.
    unit
        ``None`` (the default) attributes the whole output, summed. An int
        selects ``output[..., unit]`` (the last axis, as an integer ``target=``
        does on the model output). A tuple indexes the axes after the batch
        axis, so ``(3,)`` is channel 3 of a convolution output and
        ``(3, 1, 2)`` one position of it.
    call
        Which call of the layer, counted from 0, when the layer runs more than
        once in a forward pass. ``None`` requires that the layer runs once.
    """

    layer: str
    _: KW_ONLY
    unit: int | tuple[int, ...] | None = None
    call: int | None = None

    def __post_init__(self) -> None:
        """Validate the selection types; ranges are checked against the activation."""

        if not isinstance(self.layer, str):
            raise AttributionError(
                f"SiteTarget layer must be a module name string; got {type(self.layer).__name__}. "
                "Remedy: pass a name from dict(model.named_modules()), e.g. "
                f"tl.attribution.SiteTarget('fc1'). {_DOCS_POINTER}",
                code="attribution_site_not_found",
            )
        if not _is_unit(self.unit):
            raise _unit_invalid(
                self.layer,
                f"unit must be None, an int or a tuple of ints; got {self.unit!r}",
                SiteTarget(self.layer, call=self.call) if _is_call(self.call) else None,
                code="attribution_site_unit_invalid",
            )
        if not _is_call(self.call):
            raise _firing_invalid(
                self.layer,
                f"call must be None or a call index of at least 0; got {self.call!r}",
                SiteTarget(self.layer, unit=self.unit),
                code="attribution_site_firing_invalid",
            )

    def __repr__(self) -> str:
        """Return the constructor spelling of this site target."""

        parts = [repr(self.layer)]
        if self.unit is not None:
            parts.append(f"unit={self.unit!r}")
        if self.call is not None:
            parts.append(f"call={self.call!r}")
        return f"SiteTarget({', '.join(parts)})"


def _is_index(value: Any) -> bool:
    """Return whether ``value`` is a plain int (bools excluded)."""

    return isinstance(value, int) and not isinstance(value, bool)


def _is_unit(unit: Any) -> bool:
    """Return whether ``unit`` has a valid unit-selection type."""

    if unit is None or _is_index(unit):
        return True
    return isinstance(unit, tuple) and len(unit) > 0 and all(_is_index(item) for item in unit)


def _is_call(call: Any) -> bool:
    """Return whether ``call`` is ``None`` or a nonnegative int."""

    return call is None or (_is_index(call) and call >= 0)


def _spelling(site: SiteTarget | None) -> str:
    """Return the public constructor spelling for a remedy line."""

    return "None" if site is None else f"tl.attribution.{site!r}"


def _target_spelling(target: Any) -> str:
    """Return the ``target=`` spelling of ``target`` for a remedy line."""

    if isinstance(target, SiteTarget):
        return f"target={_spelling(target)}"
    return f"target={target!r}" if _is_index(target) else "target=target"


def _unit_invalid(layer: str, why: str, fixed: SiteTarget | None, *, code: str) -> AttributionError:
    """Build the coded refusal for a unit selection that does not fit the site."""

    fixed = fixed if fixed is not None else SiteTarget(layer)
    return AttributionError(
        f"SiteTarget for layer {layer!r} cannot select its unit: {why}. Remedy: select a unit "
        f"inside the layer's output, e.g. target={_spelling(fixed)}. {_DOCS_POINTER}",
        code=code,
    )


def _firing_invalid(layer: str, why: str, fixed: SiteTarget, *, code: str) -> AttributionError:
    """Build the coded refusal for a site whose firing cannot be chosen."""

    return AttributionError(
        f"SiteTarget for layer {layer!r} has no single output to attribute: {why}. Remedy: "
        f"name the call, e.g. target={_spelling(fixed)}. {_DOCS_POINTER}",
        code=code,
    )


@dataclass(frozen=True)
class _SiteActivation:
    """The site's activation, returned by the model call in place of the model output."""

    tensor: Tensor


@dataclass(frozen=True)
class _SiteRoute:
    """A resolved site target: the module whose output replaces the model output."""

    target: SiteTarget
    module: Module

    @contextmanager
    def capture(self) -> Iterator[list[Any]]:
        """Record every output of the site module during the enclosed forward pass."""

        firings: list[Any] = []

        def _hook(_module: Module, _args: tuple[Any, ...], output: Any) -> None:
            """Keep one site output per call."""

            firings.append(output)

        handle: RemovableHandle = self.module.register_forward_hook(_hook)
        try:
            yield firings
        finally:
            handle.remove()

    def activation(
        self,
        firings: list[Any],
        replacements: tuple[Tensor, ...],
        dependent_site: Callable[[], str | None],
    ) -> _SiteActivation:
        """Return the chosen call's activation after checking it depends on the inputs.

        Parameters
        ----------
        firings
            Outputs the site module produced in one forward pass, in call order.
        replacements
            Differentiable leaves substituted into that forward pass.
        dependent_site
            Callable returning a module name whose output does depend on the
            inputs, used only to spell the remedy of a refusal.

        Returns
        -------
        _SiteActivation
            The selected activation.

        Raises
        ------
        AttributionError
            If the site did not run, ran more than once without ``call=``, its
            output is not a tensor, or it does not depend on the inputs.
        """

        target = self.target
        call = 0 if target.call is None else target.call
        if not firings:
            raise _firing_invalid(
                target.layer,
                "it did not run in the forward pass",
                target,
                code="attribution_site_firing_invalid",
            )
        if target.call is None and len(firings) > 1:
            raise _firing_invalid(
                target.layer,
                f"it ran {len(firings)} times",
                SiteTarget(target.layer, unit=target.unit, call=0),
                code="attribution_site_firing_invalid",
            )
        if call >= len(firings):
            raise _firing_invalid(
                target.layer,
                f"call={call} was asked for but it ran {len(firings)} times",
                SiteTarget(target.layer, unit=target.unit, call=len(firings) - 1),
                code="attribution_site_firing_invalid",
            )
        output = firings[call]
        if not isinstance(output, Tensor):
            raise _firing_invalid(
                target.layer,
                f"its output is a {type(output).__name__}, not a tensor",
                target,
                code="attribution_site_firing_invalid",
            )
        _require_input_dependence(target, output, replacements, dependent_site)
        return _SiteActivation(output)


def _require_input_dependence(
    target: SiteTarget,
    output: Tensor,
    replacements: tuple[Tensor, ...],
    dependent_site: Callable[[], str | None],
) -> None:
    """Refuse a site whose output is not computed from any attributed leaf."""

    from torchlens.attribution._layer import _autograd_leaf_variable_ids

    leaves = [leaf for leaf in replacements if leaf.requires_grad]
    if not leaves:
        return
    reachable = _autograd_leaf_variable_ids((output,))
    if any(id(leaf) in reachable for leaf in leaves):
        return
    other = dependent_site()
    remedy = (
        f"choose a site the input reaches, e.g. target={_spelling(SiteTarget(other))}"
        if other is not None
        else "attribute the model output with an int or callable target"
    )
    raise AttributionError(
        f"SiteTarget layer {target.layer!r} does not depend on the attributed inputs: its "
        "output is computed without them, so every attribution to it is zero. "
        f"Remedy: {remedy}. {_DOCS_POINTER}",
        code="attribution_site_independent_of_input",
    )


def _resolve_site(model: Module, target: SiteTarget) -> _SiteRoute:
    """Resolve ``target.layer`` to a module of ``model``.

    Raises
    ------
    AttributionError
        ``attribution_site_not_found`` when no module has that name.
    """

    named = dict(model.named_modules())
    if target.layer in named:
        return _SiteRoute(target=target, module=named[target.layer])
    names = [name for name in named if name]
    close = difflib.get_close_matches(target.layer, names, n=1)
    suggestion = close[0] if close else (names[0] if names else "")
    fixed = SiteTarget(suggestion, call=target.call)
    shown = ", ".join(names[:8]) or "<no named child modules>"
    raise AttributionError(
        f"SiteTarget layer {target.layer!r} is not a module of this model (modules include: "
        f"{shown}). Remedy: name a module from dict(model.named_modules()), e.g. "
        f"target={_spelling(fixed)}. {_DOCS_POINTER}",
        code="attribution_site_not_found",
    )


def _select_unit(activation: Tensor, target: SiteTarget) -> Tensor:
    """Return the scalar of the selected unit, summed over the batch.

    Raises
    ------
    AttributionError
        ``attribution_site_unit_invalid`` when the unit does not fit the shape.
    """

    unit = target.unit
    shape = tuple(int(size) for size in activation.shape)
    if unit is None:
        return activation.sum()
    if isinstance(unit, int):
        if activation.ndim < 2 or not -shape[-1] <= unit < shape[-1]:
            raise _unit_invalid(
                target.layer,
                f"unit={unit} is outside the last axis of its output of shape {shape}",
                _first_unit(target, shape),
                code="attribution_site_unit_invalid",
            )
        return activation[..., unit].sum()
    if len(unit) > activation.ndim - 1 or any(
        not -size <= index < size for index, size in zip(unit, shape[1:], strict=False)
    ):
        raise _unit_invalid(
            target.layer,
            f"unit={unit} does not index the axes after the batch axis of its output of "
            f"shape {shape}",
            _first_unit(target, shape),
            code="attribution_site_unit_invalid",
        )
    index: tuple[slice | int, ...] = (slice(None), *unit)
    return activation[index].sum()


def _first_unit(target: SiteTarget, shape: tuple[int, ...]) -> SiteTarget:
    """Return the same site selecting unit 0, or the whole output when no unit axis exists."""

    if len(shape) < 2 or shape[-1] == 0:
        return SiteTarget(target.layer, call=target.call)
    return SiteTarget(target.layer, unit=0, call=target.call)


def _site_unsupported(method: str, target: SiteTarget, *, code: str) -> AttributionError:
    """Build the coded refusal for a method that cannot attribute a site."""

    return AttributionError(
        f"{method} cannot attribute a SiteTarget ({target!r}); it scores the model output "
        "only. Remedy: use a gradient method that accepts site targets, e.g. "
        f"tl.attribution.saliency(model, inputs, target={_spelling(target)}). {_DOCS_POINTER}",
        code=code,
    )


def _input_dependent_site(
    model: Module, run: Callable[[], Any], leaves: tuple[Tensor, ...]
) -> str | None:
    """Return the last-run module whose tensor output depends on ``leaves``.

    Used only to spell the remedy of an input-independent site refusal.
    """

    from torchlens.attribution._layer import _autograd_leaf_variable_ids

    outputs: list[tuple[str, Tensor]] = []
    handles: list[RemovableHandle] = []

    def _hook_for(name: str) -> Any:
        """Return a hook keeping ``name``'s tensor outputs."""

        def _hook(_module: Module, _args: tuple[Any, ...], output: Any) -> None:
            """Keep a tensor output in call order."""

            if isinstance(output, Tensor):
                outputs.append((name, output))

        return _hook

    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_hook(_hook_for(name)))
    try:
        run()
    finally:
        for handle in handles:
            handle.remove()
    leaf_ids = {id(leaf) for leaf in leaves}
    for name, output in reversed(outputs):
        if leaf_ids & _autograd_leaf_variable_ids((output,)):
            return name
    return None
