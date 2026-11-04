"""The one input-routing door every entry that takes model inputs shares.

``tl.trace`` resolves Hugging Face text, modality-dict and PIL inputs through
its auto-route detectors (``torchlens._autoroute._builtin_input``) before the
capture door coerces anything. ``Trace.run``, ``tl.record``, ``tl.recording.dry_run``,
``tl.validation.validate_backward_pass`` and the ``tl.debug`` entries that take model
inputs reach the model through :func:`torchlens._input_coerce._coerce_inputs`
alone, so this module gives that door the same resolution: the same string,
dict or image produces the same model inputs at every entry.
"""

from __future__ import annotations

import collections.abc
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _RoutedInputs:
    """Positional and keyword model inputs already routed through the door.

    A stored ``input_transform`` applied on rerun may return a mapping (a
    tokenizer's ``BatchEncoding``, a processor's output); the capture door
    turns such a mapping into the forward's keyword inputs, and this carrier
    lets the rerun engine do the same without a new keyword on its entries.
    """

    input_args: Any
    input_kwargs: dict[Any, Any] | None


def _route_transformed_input(original: Any, transformed: Any) -> Any:
    """Return a transform's result with a new mapping routed into keyword inputs.

    Parameters
    ----------
    original:
        The input before the transform ran.
    transformed:
        The transform's return value.

    Returns
    -------
    Any
        ``transformed`` unchanged, or a :class:`_RoutedInputs` with an empty
        positional slot when the transform produced a mapping the user did
        not pass themselves (the capture door's own rule).
    """

    if transformed is original or not isinstance(transformed, collections.abc.Mapping):
        return transformed
    return _RoutedInputs([], dict(transformed))


def _fast_run_inputs(inputs: Any) -> Any:
    """Spell routed inputs as the fast engine's mixed ``args``/``kwargs`` mapping.

    Parameters
    ----------
    inputs:
        Forward input, possibly a :class:`_RoutedInputs`.

    Returns
    -------
    Any
        ``inputs`` unchanged, or ``{"args": [...], "kwargs": {...}}``.
    """

    if not isinstance(inputs, _RoutedInputs):
        return inputs
    return {"args": list(inputs.input_args or []), "kwargs": dict(inputs.input_kwargs or {})}


def _processor_parameter_names(processor: Any) -> frozenset[str] | None:
    """Return the named parameters of a processor's call signature.

    Parameters
    ----------
    processor:
        Callable processor (an ``AutoProcessor`` instance or a function).

    Returns
    -------
    frozenset[str] | None
        Named positional-or-keyword and keyword-only parameters, or ``None``
        when no signature is available.
    """

    try:
        signature = inspect.signature(processor)
    except (TypeError, ValueError):
        return None
    return frozenset(
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )


def _spell_for_processor(key: str, names: frozenset[str] | None) -> str:
    """Return the processor's own spelling of one modality key.

    The processor's signature is the oracle: a key it names is kept, a key it
    does not name is re-spelled to the singular or plural it does name
    (``image`` to ``images``), and a key with no named alternative passes
    through to whatever ``**kwargs`` the processor declares.

    Parameters
    ----------
    key:
        Modality key the user wrote.
    names:
        Named parameters of the processor, or ``None`` when unknown.

    Returns
    -------
    str
        The key to call the processor with.
    """

    if names is None or key in names:
        return key
    alternatives = [key + "s"]
    if key.endswith("s"):
        alternatives.append(key[:-1])
    for alternative in alternatives:
        if alternative in names:
            return alternative
    return key


def _apply_hf_processor(
    processor: Any,
    modalities: collections.abc.Mapping[str, Any],
    *,
    sampling_rate: Any | None = None,
) -> Any:
    """Run a modality dict through a processor, spelling its keys as it does.

    Parameters
    ----------
    processor:
        Callable processor (``AutoProcessor``-like).
    modalities:
        Modality dict (``text``, ``image``/``images``, ``audio``, ``videos``, ...).
    sampling_rate:
        Sampling rate to pass alongside an ``audio`` payload when the dict
        carries none; ``None`` passes nothing.

    Returns
    -------
    Any
        The processor's mapping of model-ready tensors.
    """

    names = _processor_parameter_names(processor)
    kwargs = {_spell_for_processor(str(key), names): value for key, value in modalities.items()}
    if "audio" in kwargs and "sampling_rate" not in kwargs and sampling_rate is not None:
        kwargs["sampling_rate"] = sampling_rate
    return processor(**kwargs, return_tensors="pt")


#: Input routes higher layers install at import: ``(claims, transform)`` pairs
#: tried in order. ``claims(model, x)`` says whether the route takes the whole
#: input; ``transform(model, x)`` returns the model-ready inputs exactly as the
#: ``tl.trace`` auto-route builds them. Append-only for the process.
_ROUTES: list[tuple[Callable[[Any, Any], bool], Callable[[Any, Any], Any]]] = []


def _install_route(
    claims: Callable[[Any, Any], bool], transform: Callable[[Any, Any], Any]
) -> None:
    """Install one input route at the door (idempotent per pair).

    Parameters
    ----------
    claims:
        ``claims(model, x)``: whether the route takes the whole input ``x``.
    transform:
        ``transform(model, x)``: the model-ready inputs for a claimed ``x``.
    """

    if (claims, transform) not in _ROUTES:
        _ROUTES.append((claims, transform))


def _route_claims(model: Any, x: Any) -> bool:
    """Return whether an installed route takes the whole input ``x`` for ``model``.

    Parameters
    ----------
    model:
        Model or model-like object.
    x:
        Whole user input.

    Returns
    -------
    bool
        ``True`` when :func:`_route_input` would transform ``x``.
    """

    return any(claims(model, x) for claims, _transform in _ROUTES)


def _route_input(model: Any, x: Any) -> Any:
    """Transform a claimed whole input through the first route that claims it.

    Parameters
    ----------
    model:
        Model the route claims (:func:`_route_claims` returned ``True``).
    x:
        Whole user input.

    Returns
    -------
    Any
        The route's model-ready inputs: a mapping (a tokenizer's or processor's
        output) becomes the forward's keyword inputs at the door; anything
        else stays positional.
    """

    for claims, transform in _ROUTES:
        if claims(model, x):
            return transform(model, x)
    return x


def _routed_payload(x: Any) -> Any:
    """Return the tensors a routed input carries, for batch-size inspection.

    Parameters
    ----------
    x:
        Forward input, possibly a :class:`_RoutedInputs`.

    Returns
    -------
    Any
        ``x`` itself, or the routed keyword inputs when the positional slot is
        empty.
    """

    if not isinstance(x, _RoutedInputs):
        return x
    return x.input_args or x.input_kwargs


def _owner_of(model: Any) -> Any:
    """Return the object whose tokenizer and processors resolve ``model``'s inputs.

    A bound method of a module (``tl.trace(model.generate, "text")``) and the
    synthetic root TorchLens wraps it in both resolve through their owner
    (``method.__self__``), so text, images and modality dicts reach
    ``model.generate`` exactly as they reach ``model``: an attached
    ``tokenizer``, a ``to_tokens`` method or the owner's ``name_or_path``.

    Parameters
    ----------
    model:
        Model, bound method of a module, or TorchLens's bound-method root.

    Returns
    -------
    Any
        The owning module for a bound method or its root; ``model`` otherwise.
    """

    # Duck-typed so this module adds no import edge to the backend layer: the
    # root (``TLBoundMethodRoot``) exposes its owner as the ``tl_owner``
    # property, and a bound method carries it as ``__self__``.
    if isinstance(getattr(type(model), "tl_owner", None), property):
        return model.tl_owner
    if inspect.ismethod(model) and hasattr(model.__self__, "forward"):
        return model.__self__
    return model
