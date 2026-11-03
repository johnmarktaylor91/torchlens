"""The model-parameter randomization test (Adebayo et al., 2018).

"Sanity checks for saliency maps" (Adebayo et al., NeurIPS 2018) asks whether
an explanation depends on what the model learned: re-initialize the model's
layers, top (output side) first, re-run the explanation method, and measure
how similar each new map is to the original. A method whose maps stay the
same when the parameters are destroyed is not explaining the model. Two
orders: *cascading* (layers randomized cumulatively, from the top down) and
*independent* (one layer randomized at a time, the rest left trained).

Similarity is the Spearman rank correlation of the signed maps and of their
absolute values, and the mean SSIM (Wang et al., 2004) of the absolute maps
summed over channels and rescaled to ``[0, 1]`` where the maps are spatial
planes of side 7 or more. Every randomization runs on a deep copy of the
model inside a forked random state, so the caller's model and the global
random state are untouched.
"""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import torch
from torch import Tensor
from torch.nn import Module

from torchlens.attribution._core import InputKwargs, _AttributionTarget, _target_repr
from torchlens.attribution._eval_maps import _DOCS_POINTER, _attribution_tensor, _map_similarity
from torchlens.attribution._result import Attribution, AttributionError, AttributionWarning

RandomizationKind: TypeAlias = Literal["cascading", "independent"]

# Kept randomized maps past this many bytes warn once and name keep_maps=False.
_KEEP_MAPS_WARN_BYTES = 256 * 1024 * 1024
_FALLBACK_STD = 0.02


@dataclass(frozen=True)
class RandomizationStep:
    """One randomization of the test.

    Attributes
    ----------
    layer
        The module re-initialized at this step.
    randomized
        Every module re-initialized when the map was computed (cascading:
        this layer and all above it; independent: this layer alone).
    similarities
        Similarity scores of this map against the original, keyed by measure:
        ``spearman``, ``spearman_abs`` and ``ssim`` (``None`` where the measure
        is undefined or the map is not a spatial plane).
    values
        The attribution computed on the randomized copy, or ``None`` with
        ``keep_maps=False``.
    """

    module_address: str
    randomized_module_addresses: tuple[str, ...]
    similarities: dict[str, float | None]
    values: Tensor | None


@dataclass(frozen=True)
class RandomizationTest:
    """The outcome of a model-parameter randomization test.

    Attributes
    ----------
    randomization
        ``"cascading"`` or ``"independent"``.
    layers
        Randomized modules, top (output side) first.
    original_attribution
        The attribution of the unrandomized model.
    steps
        One :class:`RandomizationStep` per layer, in ``layers`` order.
    settings
        The resolved test policy (randomization, seed, method, initialization).
    extra
        Method-specific extras: the similarity measure names (``measures``) and
        the per-step initialization notes (``initialization_per_layer``).
    """

    randomization: str
    module_addresses: tuple[str, ...]
    original_attribution: Tensor
    steps: tuple[RandomizationStep, ...]
    settings: dict[str, Any]
    extra: dict[str, Any]

    def similarity(self, measure: str = "spearman") -> tuple[float | None, ...]:
        """Return one measure across the steps, in ``layers`` order.

        Parameters
        ----------
        measure
            ``"spearman"`` (default), ``"spearman_abs"`` or ``"ssim"``.

        Returns
        -------
        tuple[float | None, ...]
            The measure per step.
        """

        if measure not in self.extra["measures"]:
            raise AttributionError(
                f"unknown similarity measure {measure!r}; the measures are "
                f"{self.extra['measures']}. Remedy: result.similarity('spearman'). "
                f"{_DOCS_POINTER}",
                code="attribution_eval_setting_invalid",
            )
        return tuple(step.similarities[measure] for step in self.steps)

    def __repr__(self) -> str:
        """Compact representation without dumping tensors."""

        return (
            f"RandomizationResult(randomization={self.randomization!r}, layers={len(self.module_addresses)}, "
            f"spearman={self.similarity('spearman')!r})"
        )


def _parameterized_layers_top_down(model: Module, run: Callable[[Module], Any]) -> list[str]:
    """Return modules with their own parameters, last executed first.

    Parameters
    ----------
    model
        The model copy to inspect.
    run
        Calls the explanation method once on a model.

    Returns
    -------
    list[str]
        Module names in reverse order of first execution; modules with
        parameters that never ran are left out.
    """

    names = {id(module): name for name, module in model.named_modules()}
    executed: list[str] = []
    handles = []

    def _record(module: Module, *_: Any) -> None:
        """Note a parameterized module the first time it runs."""

        name = names.get(id(module))
        if name is not None and name not in executed:
            executed.append(name)

    for _, module in model.named_modules():
        if any(True for _ in module.parameters(recurse=False)):
            handles.append(module.register_forward_hook(_record))
    try:
        run(model)
    finally:
        for handle in handles:
            handle.remove()
    return list(reversed(executed))


def _reinitialize(module: Module) -> str:
    """Re-initialize one module's own parameters in place.

    Parameters
    ----------
    module
        Module of the model copy.

    Returns
    -------
    str
        How it was re-initialized.
    """

    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()
        return "reset_parameters"
    with torch.no_grad():
        for parameter in module.parameters(recurse=False):
            parameter.normal_(0.0, _FALLBACK_STD)
    return f"normal(0, {_FALLBACK_STD})"


def _validate_layers(model: Module, layers: Any) -> list[str]:
    """Validate caller-named layers.

    Parameters
    ----------
    model
        The caller's model.
    layers
        Module names, top first.

    Returns
    -------
    list[str]
        The names.

    Raises
    ------
    AttributionError
        If a name is not a module with its own parameters.
    """

    modules = dict(model.named_modules())
    named = [layers] if isinstance(layers, str) else list(layers)
    for name in named:
        module = modules.get(name) if isinstance(name, str) else None
        if module is None or not any(True for _ in module.parameters(recurse=False)):
            candidates = [
                key
                for key, value in modules.items()
                if any(True for _ in value.parameters(recurse=False))
            ]
            example = candidates[-1] if candidates else "fc"
            raise AttributionError(
                f"randomization_test layers names {name!r}, which is not a module of this "
                f"model with parameters of its own (modules with parameters: {candidates}). "
                "Remedy: name such modules, top first, or leave layers=None for all of them, "
                f"e.g. tl.attribution.randomization_test(model, x, target=0, "
                f"method=tl.attribution.saliency, layers=[{example!r}]). {_DOCS_POINTER}",
                code="attribution_randomization_layer_invalid",
            )
    if not named:
        raise AttributionError(
            "randomization_test layers is empty, so nothing would be randomized. Remedy: "
            "leave layers=None to randomize every module with parameters, e.g. "
            "tl.attribution.randomization_test(model, x, target=0, "
            f"method=tl.attribution.saliency). {_DOCS_POINTER}",
            code="attribution_randomization_layer_invalid",
        )
    return named


def _no_layers_ran(model: Module, *, code: str) -> AttributionError:
    """Build the refusal for a method under which no parameterized module ran.

    Parameters
    ----------
    model
        The caller's model.
    code
        The refusal code (``attribution_randomization_layer_invalid``).

    Returns
    -------
    AttributionError
        The coded refusal, naming ``layers=``.
    """

    candidates = [
        name
        for name, module in model.named_modules()
        if any(True for _ in module.parameters(recurse=False))
    ]
    example = candidates[-1] if candidates else "fc"
    return AttributionError(
        "randomization_test found no module with parameters of its own that ran while the "
        "method computed its map, so nothing would be randomized and the test would pass "
        f"vacuously (modules with parameters: {candidates}). Remedy: name the modules to "
        "randomize, top first, e.g. tl.attribution.randomization_test(model, x, target=0, "
        f"method=method, layers=[{example!r}]). {_DOCS_POINTER}",
        code=code,
    )


def _map_of(result: Any) -> Tensor:
    """Return the map an explanation method produced.

    Parameters
    ----------
    result
        ``AttributionResult`` or tensor.

    Returns
    -------
    Tensor
        Detached float64 values.
    """

    return _attribution_tensor(result).to(torch.float64)


def randomization_test(  # noqa: PLR0913 - public door: keyword-only test settings
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: _AttributionTarget,
    method: Callable[..., Attribution | Tensor],
    method_kwargs: dict[str, Any] | None = None,
    randomization: RandomizationKind = "cascading",
    layers: str | list[str] | tuple[str, ...] | None = None,
    seed: int = 0,
    keep_values: bool = True,
) -> RandomizationTest:
    """Re-initialize layers top down and measure how each map changes.

    The model-parameter randomization test of Adebayo et al. (2018): on a
    deep copy of ``model``, layers are re-initialized from the output side
    (cascading: cumulatively; independent: one at a time on a fresh copy),
    ``method`` is re-run, and each map is compared with the original by
    Spearman rank correlation (signed and absolute) and SSIM (Wang et al.,
    2004). Similarity that stays high as layers are destroyed means the
    method does not depend on the parameters.

    Parameters
    ----------
    model
        PyTorch module the method explains; never modified.
    inputs
        Bare tensor or tuple/list of positional model arguments.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Target passed to ``method``.
    method
        Explanation method called as
        ``method(model_copy, inputs, input_kwargs, target=target, **method_kwargs)``,
        returning an ``AttributionResult`` or a tensor (``grad_cam`` maps are
        compared at input resolution).
    method_kwargs
        Extra keyword arguments for ``method`` (for example ``layer=`` for
        ``grad_cam``).
    randomization
        ``"cascading"`` (default) or ``"independent"``.
    layers
        Module names to randomize, top first; default: every module with
        parameters of its own, in reverse order of first execution.
    seed
        Seed of the re-initialization draws (default 0), drawn inside a forked
        random state that is restored afterwards.
    keep_values
        Keep every randomized map in ``steps`` (default ``True``); past
        256 MiB of kept maps the test warns once and names
        ``keep_maps=False``.

    Returns
    -------
    RandomizationResult
        The original map and one step per randomized layer.

    Raises
    ------
    AttributionError
        On an unknown order, a non-callable method, layers that are not
        parameterized modules, or (with ``layers=None``) no parameterized module
        running while the method computes its map.
    """

    if randomization not in ("cascading", "independent"):
        raise AttributionError(
            "randomization_test randomization must be 'cascading' or 'independent'; got "
            f"{randomization!r}. "
            "Remedy: tl.attribution.randomization_test(model, x, target=0, "
            f"method=tl.attribution.saliency, randomization='cascading'). {_DOCS_POINTER}",
            code="attribution_eval_setting_invalid",
        )
    if not callable(method):
        raise AttributionError(
            "randomization_test method must be an explanation method such as "
            f"tl.attribution.saliency; got {type(method).__name__}. Remedy: "
            "tl.attribution.randomization_test(model, x, target=0, "
            f"method=tl.attribution.saliency). {_DOCS_POINTER}",
            code="attribution_eval_setting_invalid",
        )
    kwargs = dict(method_kwargs or {})

    def _run(copy_model: Module) -> Tensor:
        """Run the explanation method on one model copy."""

        return _map_of(method(copy_model, inputs, input_kwargs, target=target, **kwargs))

    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        trained = copy.deepcopy(model)
        original = _run(trained)
        if layers is None:
            names = _parameterized_layers_top_down(trained, _run)
            if not names:
                raise _no_layers_ran(model, code="attribution_randomization_layer_invalid")
        else:
            names = _validate_layers(model, layers)
        plan = _RandomizationPlan(tuple(names), randomization, keep_values)
        steps, init_notes = _randomize_steps(model, plan, _run, original)
    return RandomizationTest(
        randomization=randomization,
        module_addresses=tuple(names),
        original_attribution=original,
        steps=tuple(steps),
        settings={
            "randomization": randomization,
            "seed": seed,
            "method": getattr(method, "__name__", repr(method)),
            "target": _target_repr(target),
            "initialization": "reset_parameters, else normal(0, 0.02)",
            "keep_maps": keep_values,
        },
        extra={
            "measures": ("spearman", "spearman_abs", "ssim"),
            "initialization_per_layer": init_notes,
        },
    )


@dataclass(frozen=True)
class _RandomizationPlan:
    """What one test randomizes: the layers (top first), the order, and map keeping."""

    names: tuple[str, ...]
    randomization: str
    keep_maps: bool


def _randomize_steps(
    model: Module,
    plan: _RandomizationPlan,
    run: Callable[[Module], Tensor],
    original: Tensor,
) -> tuple[list[RandomizationStep], dict[str, str]]:
    """Run the randomizations in order.

    Parameters
    ----------
    model
        The caller's model (copied, never modified).
    plan
        Layers, order and whether to keep each map.
    run
        Explanation method on a model copy.
    original
        The unrandomized map.

    Returns
    -------
    tuple[list[RandomizationStep], dict[str, str]]
        The steps and how each layer was re-initialized.
    """

    names, randomization, keep_maps = list(plan.names), plan.randomization, plan.keep_maps
    steps: list[RandomizationStep] = []
    notes: dict[str, str] = {}
    kept_bytes = 0
    warned = False
    cascade = copy.deepcopy(model) if randomization == "cascading" else None
    for index, name in enumerate(names):
        working = cascade if cascade is not None else copy.deepcopy(model)
        notes[name] = _reinitialize(dict(working.named_modules())[name])
        randomized = tuple(names[: index + 1]) if cascade is not None else (name,)
        values = run(working)
        if tuple(values.shape) != tuple(original.shape):
            raise AttributionError(
                f"randomization_test: the method's map changed shape from "
                f"{tuple(original.shape)} to {tuple(values.shape)} after randomizing "
                f"{name!r}, so the maps cannot be compared. Remedy: fix the map's layout, "
                "e.g. pass method_kwargs={'layer': ...} so grad_cam does not re-choose its "
                f"layer. {_DOCS_POINTER}",
                code="attribution_randomization_map_changed",
            )
        similarities = _map_similarity(original, values)
        if keep_maps:
            kept_bytes += values.element_size() * values.numel()
            if kept_bytes > _KEEP_MAPS_WARN_BYTES and not warned:
                warned = True
                warnings.warn(
                    AttributionWarning(
                        f"randomization_test is keeping {math.ceil(kept_bytes / 2**20)} MiB "
                        "of randomized maps. Remedy: pass keep_maps=False to keep only the "
                        f"similarity measures. {_DOCS_POINTER}",
                        code="attribution_randomization_maps_large",
                    ),
                    stacklevel=3,
                )
        steps.append(
            RandomizationStep(name, randomized, similarities, values if keep_maps else None)
        )
    return steps, notes
